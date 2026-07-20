# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Callable

import torch

from torchcts.core.comparer import mark_usable_fallback, restore_metrics, snapshot_metrics
from torchcts.core.fft_contract import (
    compare_fft_nonfinite_groups,
    public_fft_contract_spec,
)
from torchcts.core.reference_oracles import (
    matmul_family_determinate_reference,
)


REPRESENTATION_AMBIGUOUS = "representation_ambiguous"
TIE_INDEX_AMBIGUOUS = "tie_index_ambiguous"
RANDOM_VALUE = "random_value"
UNINITIALIZED_VALUE = "uninitialized_value"
STRICT_BY_API = "strict_by_api"
INTENTIONALLY_EXACT = "intentionally_exact"
STRUCTURAL_ONLY = "structural_only"
COVERED_BY_CONTRACT = "covered_by_contract"


@dataclass(frozen=True)
class NonUniqueOutputContract:
    family: str
    ambiguity_type: str
    status: str
    matcher: Callable[[str], bool]
    checker: Callable | None
    reason: str
    strict_cases: tuple[str, ...] = ()
    unsupported_cases: tuple[str, ...] = ()
    try_direct_first: bool = True
    supports_special_tiers: bool = False


class ContractNotApplicable(Exception):
    pass


def _normalize_name(op_name: str) -> str:
    name = str(op_name)
    if name.startswith("torch."):
        return name[len("torch."):]
    return name


def _name_has_family(op_name: str, family: str) -> bool:
    name = _normalize_name(op_name)
    aten_family = family.replace(".", "_")
    return (
        name == family
        or name == aten_family
        or name.startswith(f"{family}.")
        or name.startswith(f"{aten_family}.")
        or name.endswith(f".{family}")
        or name.endswith(f"_{aten_family}")
        or name == f"aten::{aten_family}"
        or name.startswith(f"aten::{aten_family}.")
        or name == f"aten::{family}"
        or name.startswith(f"aten::{family}.")
    )


def _matches_any(families: tuple[str, ...]) -> Callable[[str], bool]:
    return lambda op_name: any(_name_has_family(op_name, family) for family in families)


def _contains_any(tokens: tuple[str, ...]) -> Callable[[str], bool]:
    return lambda op_name: any(token in _normalize_name(op_name) for token in tokens)


def _matches_random_value(op_name: str) -> bool:
    name = _normalize_name(op_name)
    # These operators consume the random decision made by a forward operator.
    # Their mask/noise argument makes them deterministic, so a random-output
    # fallback would hide real backward bugs.
    if "backward" in name:
        return False
    if _matches_any((
        "bernoulli",
        "geometric",
        "multinomial",
        "rand_like",
        "randint",
        "randint_like",
        "randn",
        "randn_like",
        "normal",
        "uniform",
        "log_normal",
        "cauchy",
        "exponential",
        "rand",
        "nn.functional.dropout",
        "nn.functional.dropout2d",
        "nn.functional.dropout3d",
        "nn.functional.alpha_dropout",
        "nn.functional.feature_alpha_dropout",
        "native_dropout",
    ))(name):
        return True
    if any(token in name for token in (
        "_philox_normal",
        "_philox_uniform",
        "_fused_dropout",
        "_fill_mem_eff_dropout_mask",
        "_cudnn_init_dropout_state",
        "bernoulli_",
        "cauchy_",
        "exponential_",
        "geometric_",
        "log_normal_",
        "normal_",
        "random_",
        "uniform_",
        "randperm",
    )):
        return True
    if "rand" in frozenset(part for part in re.split(r"[^A-Za-z0-9]+|_", name) if part):
        return True
    return False


def _context_value(context: dict, key: str, default=None):
    if key in context and context[key] is not None:
        return context[key]
    sample = context.get("sample")
    if sample is not None and hasattr(sample, key):
        return getattr(sample, key)
    return default


def _context_input(context: dict) -> torch.Tensor:
    value = _context_value(context, "input")
    if not isinstance(value, torch.Tensor):
        raise AssertionError("non-unique output contract requires tensor input context")
    return value


def _context_args(context: dict) -> tuple:
    args = _context_value(context, "args", ())
    return tuple(args or ())


def _context_kwargs(context: dict) -> dict:
    kwargs = _context_value(context, "kwargs", {})
    return dict(kwargs or {})


def sample_uplo(sample) -> str:
    kwargs = getattr(sample, "kwargs", {}) or {}
    if "UPLO" in kwargs:
        return str(kwargs["UPLO"])
    for arg in getattr(sample, "args", ()) or ():
        if isinstance(arg, str) and arg.upper() in {"L", "U"}:
            return arg.upper()
    return "L"


def _compare_direct(actual, expected, category: str, dtype: torch.dtype, compare) -> None:
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        compare(actual, expected, category=category, dtype=actual.dtype)
        return
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError(f"Output sequence lengths differ: got {len(actual)}, expected {len(expected)}")
        for actual_item, expected_item in zip(actual, expected):
            _compare_direct(actual_item, expected_item, category, dtype, compare)
        return
    if isinstance(actual, dict) and isinstance(expected, dict):
        if len(actual) != len(expected):
            raise AssertionError(f"Output dict sizes differ: got {len(actual)}, expected {len(expected)}")
        for key in actual:
            if key not in expected:
                raise AssertionError(f"Key {key} not in CPU reference output keys")
            _compare_direct(actual[key], expected[key], category, dtype, compare)
        return
    raise AssertionError(f"Output type mismatch: got {type(actual).__name__}, expected {type(expected).__name__}")


def _try_compare_direct(actual, expected, category: str, dtype: torch.dtype, compare):
    metrics = snapshot_metrics()
    try:
        _compare_direct(actual, expected, category, dtype, compare)
    except AssertionError as exc:
        restore_metrics(metrics)
        return exc
    return None


def _tensor_tuple(value, op_name: str, min_len: int = 1) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        items = (value,)
    elif isinstance(value, (list, tuple)):
        items = tuple(item for item in value if isinstance(item, torch.Tensor))
    else:
        items = ()
    if len(items) < min_len:
        raise AssertionError(f"{op_name} expected at least {min_len} tensor outputs, got {type(value).__name__}")
    return items


def _assert_tensor_metadata(op_name: str, actual: torch.Tensor, expected: torch.Tensor, index: int = 0) -> None:
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(f"{op_name} output {index} shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}")
    if actual.dtype != expected.dtype:
        raise AssertionError(f"{op_name} output {index} dtype mismatch: {actual.dtype} vs {expected.dtype}")


def _real_dtype_for_complex(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex128:
        return torch.float64
    if dtype == torch.complex64:
        return torch.float32
    if getattr(torch, "complex32", None) is not None and dtype == torch.complex32:
        return torch.float16
    return dtype


def _identity(shape, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
    n = shape[-1]
    eye = torch.eye(n, dtype=dtype, device=device)
    if len(shape) > 2:
        return eye.expand(*shape[:-2], n, n)
    return eye


def _identity_for_square(matrix: torch.Tensor) -> torch.Tensor:
    return _identity(matrix.shape, matrix.dtype, matrix.device)


def _finite_values(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.resolve_conj()
    return torch.view_as_real(tensor) if tensor.is_complex() else tensor


def _effective_hermitian_matrix(matrix: torch.Tensor, uplo: str, dtype: torch.dtype, device) -> torch.Tensor:
    matrix = matrix.to(device=device, dtype=dtype)
    triangular = torch.triu(matrix) if str(uplo).upper() == "U" else torch.tril(matrix)
    diagonal = torch.diagonal(triangular, dim1=-2, dim2=-1)
    off_diagonal = triangular - torch.diag_embed(diagonal)
    real_diagonal = diagonal.real.to(dtype)
    return off_diagonal + off_diagonal.mH + torch.diag_embed(real_diagonal)


def _compare_unordered_values(
    actual_values: torch.Tensor,
    expected_values: torch.Tensor,
    *,
    category: str,
    compare,
) -> tuple[torch.Tensor, torch.Tensor]:
    _assert_tensor_metadata("unordered values", actual_values, expected_values)
    actual_cpu = actual_values.detach().cpu().reshape(-1, actual_values.shape[-1])
    expected_cpu = expected_values.detach().cpu().reshape(-1, expected_values.shape[-1])
    index_rows = []
    for actual_row, expected_row in zip(actual_cpu, expected_cpu):
        cost = torch.abs(expected_row.unsqueeze(1) - actual_row.unsqueeze(0)).to(torch.float64)
        if not bool(torch.isfinite(cost).all()):
            raise AssertionError("unordered value matching received non-finite assignment costs")
        index_rows.append(_minimum_cost_assignment(cost))
    indices = torch.tensor(index_rows, dtype=torch.long, device=actual_values.device).reshape(actual_values.shape)
    ordered_values = torch.gather(actual_values, -1, indices)
    compare(ordered_values, expected_values, category=category, dtype=actual_values.dtype)
    return ordered_values, indices


def _minimum_cost_assignment(cost: torch.Tensor) -> list[int]:
    """Return the minimum-cost column for each row using Hungarian assignment."""

    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError(f"assignment cost must be square, got {tuple(cost.shape)}")
    n = cost.shape[0]
    values = cost.detach().cpu().tolist()
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for row in range(1, n + 1):
        p[0] = row
        min_value = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        column = 0
        while True:
            used[column] = True
            current_row = p[column]
            delta = float("inf")
            next_column = 0
            for candidate in range(1, n + 1):
                if used[candidate]:
                    continue
                reduced = values[current_row - 1][candidate - 1] - u[current_row] - v[candidate]
                if reduced < min_value[candidate]:
                    min_value[candidate] = reduced
                    way[candidate] = column
                if min_value[candidate] < delta:
                    delta = min_value[candidate]
                    next_column = candidate
            for candidate in range(n + 1):
                if used[candidate]:
                    u[p[candidate]] += delta
                    v[candidate] -= delta
                else:
                    min_value[candidate] -= delta
            column = next_column
            if p[column] == 0:
                break
        while True:
            previous = way[column]
            p[column] = p[previous]
            column = previous
            if column == 0:
                break
    assignment = [0] * n
    for column in range(1, n + 1):
        assignment[p[column] - 1] = column - 1
    return assignment


def _gather_columns(matrix: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gather_index = indices.unsqueeze(-2).expand(*matrix.shape[:-2], matrix.shape[-2], indices.shape[-1])
    return torch.gather(matrix, -1, gather_index)


def _check_eigh(op_name, actual, expected, context, category, dtype, compare) -> None:
    matrix = _context_input(context)
    uplo = sample_uplo(context.get("sample")) if context.get("sample") is not None else _context_kwargs(context).get("UPLO", "L")
    actual_values, actual_vectors = _tensor_tuple(actual, op_name, min_len=2)[:2]
    expected_values, expected_vectors = _tensor_tuple(expected, op_name, min_len=2)[:2]
    _assert_tensor_metadata(op_name, actual_values, expected_values, 0)
    _assert_tensor_metadata(op_name, actual_vectors, expected_vectors, 1)
    compare(actual_values, expected_values, category=category, dtype=actual_values.dtype)
    matrix_effective = _effective_hermitian_matrix(matrix, uplo, actual_vectors.dtype, actual_vectors.device)
    if actual_vectors.shape[-2:] != matrix_effective.shape[-2:]:
        raise AssertionError(f"{op_name} eigenvector shape is incompatible with input matrix")
    compare(actual_vectors.mH @ actual_vectors, _identity(actual_vectors.shape, actual_vectors.dtype, actual_vectors.device), category=category, dtype=actual_vectors.dtype)
    right = actual_vectors * actual_values.to(actual_vectors.dtype).unsqueeze(-2)
    compare(matrix_effective @ actual_vectors, right, category=category, dtype=actual_vectors.dtype)
    reconstructed = actual_vectors @ torch.diag_embed(actual_values.to(actual_vectors.dtype)) @ actual_vectors.mH
    compare(reconstructed, matrix_effective, category=category, dtype=actual_vectors.dtype)


def _check_eig(op_name, actual, expected, context, category, dtype, compare) -> None:
    matrix = _context_input(context)
    actual_values, actual_vectors = _tensor_tuple(actual, op_name, min_len=2)[:2]
    expected_values, expected_vectors = _tensor_tuple(expected, op_name, min_len=2)[:2]
    _assert_tensor_metadata(op_name, actual_values, expected_values, 0)
    _assert_tensor_metadata(op_name, actual_vectors, expected_vectors, 1)
    ordered_values, order = _compare_unordered_values(actual_values, expected_values, category=category, compare=compare)
    ordered_vectors = _gather_columns(actual_vectors, order)
    if not bool(torch.isfinite(_finite_values(ordered_vectors).detach().cpu()).all()):
        raise AssertionError(f"{op_name} eigenvectors contain non-finite values")
    vector_norms = torch.linalg.vector_norm(ordered_vectors, dim=-2)
    if not bool((vector_norms.detach().cpu() > 0).all()):
        raise AssertionError(f"{op_name} eigenvectors contain zero vectors")
    matrix_complex = matrix.to(device=ordered_vectors.device, dtype=ordered_vectors.dtype)
    compare(matrix_complex @ ordered_vectors, ordered_vectors * ordered_values.unsqueeze(-2), category=category, dtype=ordered_vectors.dtype)


def _check_eigvals(op_name, actual, expected, context, category, dtype, compare) -> None:
    actual_values = _tensor_tuple(actual, op_name, min_len=1)[0]
    expected_values = _tensor_tuple(expected, op_name, min_len=1)[0]
    _compare_unordered_values(
        actual_values,
        expected_values,
        category=category,
        compare=compare,
    )


def _check_svd(op_name, actual, expected, context, category, dtype, compare) -> None:
    matrix = _context_input(context)
    actual_items = _tensor_tuple(actual, op_name, min_len=3)
    expected_items = _tensor_tuple(expected, op_name, min_len=3)
    actual_u, actual_s, actual_tail = actual_items[:3]
    expected_u, expected_s, expected_tail = expected_items[:3]
    for index, (actual_item, expected_item) in enumerate(((actual_u, expected_u), (actual_s, expected_s), (actual_tail, expected_tail))):
        _assert_tensor_metadata(op_name, actual_item, expected_item, index)
    compare(actual_s, expected_s, category=category, dtype=actual_s.dtype)
    k = actual_s.shape[-1]
    if "linalg_svd" in op_name or "linalg.svd" in op_name:
        actual_vh = actual_tail[..., :k, :]
    else:
        actual_vh = actual_tail[..., :, :k].mH
    actual_u_reduced = actual_u[..., :, :k]
    u_orthogonality = actual_u.mH @ actual_u
    vh_orthogonality = actual_vh @ actual_vh.mH
    compare(u_orthogonality, _identity_for_square(u_orthogonality), category=category, dtype=actual_u.dtype)
    compare(vh_orthogonality, _identity_for_square(vh_orthogonality), category=category, dtype=actual_vh.dtype)
    reconstructed = (actual_u_reduced * actual_s.to(actual_u_reduced.dtype).unsqueeze(-2)) @ actual_vh.to(actual_u_reduced.dtype)
    compare(reconstructed, matrix.to(device=reconstructed.device, dtype=reconstructed.dtype), category=category, dtype=reconstructed.dtype)


def _check_qr(op_name, actual, expected, context, category, dtype, compare) -> None:
    matrix = _context_input(context)
    actual_q, actual_r = _tensor_tuple(actual, op_name, min_len=2)[:2]
    expected_q, expected_r = _tensor_tuple(expected, op_name, min_len=2)[:2]
    _assert_tensor_metadata(op_name, actual_q, expected_q, 0)
    _assert_tensor_metadata(op_name, actual_r, expected_r, 1)
    compare(actual_q.mH @ actual_q, _identity(actual_q.shape, actual_q.dtype, actual_q.device), category=category, dtype=actual_q.dtype)
    lower = torch.tril(actual_r, diagonal=-1)
    compare(lower, torch.zeros_like(lower), category=category, dtype=actual_r.dtype)
    reconstructed = actual_q @ actual_r
    compare(reconstructed, matrix.to(device=reconstructed.device, dtype=reconstructed.dtype), category=category, dtype=reconstructed.dtype)


def _check_lu(op_name, actual, expected, context, category, dtype, compare) -> None:
    matrix = _context_input(context)
    actual_items = _tensor_tuple(actual, op_name, min_len=2)
    expected_items = _tensor_tuple(expected, op_name, min_len=2)
    for index, (actual_item, expected_item) in enumerate(zip(actual_items, expected_items)):
        _assert_tensor_metadata(op_name, actual_item, expected_item, index)
    matrix_dev = matrix.to(device=actual_items[0].device, dtype=actual_items[0].dtype)
    if "lu_factor" in op_name or "_lu_with_info" in op_name:
        actual_lu, pivots = actual_items[:2]
        if len(actual_items) > 2:
            compare(actual_items[2], expected_items[2], category="exact", dtype=actual_items[2].dtype)
        if not bool(((pivots.detach().cpu() >= 1) & (pivots.detach().cpu() <= matrix_dev.shape[-1])).all()):
            raise AssertionError(f"{op_name} pivots are outside the valid 1-based row range")
        p, l, u = torch.lu_unpack(actual_lu, pivots)
    else:
        p, l, u = actual_items[:3]
        row_sums = p.detach().cpu().sum(dim=-1)
        col_sums = p.detach().cpu().sum(dim=-2)
        if not bool(torch.equal(row_sums, torch.ones_like(row_sums)) and torch.equal(col_sums, torch.ones_like(col_sums))):
            raise AssertionError(f"{op_name} P is not a permutation matrix")
    compare(p @ l @ u, matrix_dev, category=category, dtype=matrix_dev.dtype)


def _check_pinv(op_name, actual, expected, context, category, dtype, compare) -> None:
    matrix = _context_input(context)
    actual_pinv = _tensor_tuple(actual, op_name, min_len=1)[0]
    expected_pinv = _tensor_tuple(expected, op_name, min_len=1)[0]
    _assert_tensor_metadata(op_name, actual_pinv, expected_pinv, 0)
    matrix_dev = matrix.to(device=actual_pinv.device, dtype=actual_pinv.dtype)
    compare(matrix_dev @ actual_pinv @ matrix_dev, matrix_dev, category=category, dtype=matrix_dev.dtype)
    compare(actual_pinv @ matrix_dev @ actual_pinv, actual_pinv, category=category, dtype=actual_pinv.dtype)
    ax = matrix_dev @ actual_pinv
    xa = actual_pinv @ matrix_dev
    compare(ax.mH, ax, category=category, dtype=ax.dtype)
    compare(xa.mH, xa, category=category, dtype=xa.dtype)


def _check_lstsq(op_name, actual, expected, context, category, dtype, compare) -> None:
    matrix = _context_input(context)
    args = _context_args(context)
    if not args or not isinstance(args[0], torch.Tensor):
        raise AssertionError(f"{op_name} requires RHS tensor context")
    rhs = args[0]
    actual_items = _tensor_tuple(actual, op_name, min_len=1)
    expected_items = _tensor_tuple(expected, op_name, min_len=1)
    actual_solution = actual_items[0]
    expected_solution = expected_items[0]
    _assert_tensor_metadata(op_name, actual_solution, expected_solution, 0)
    matrix_dev = matrix.to(device=actual_solution.device, dtype=actual_solution.dtype)
    rhs_dev = rhs.to(device=actual_solution.device, dtype=actual_solution.dtype)
    compare(matrix_dev @ actual_solution, matrix_dev @ expected_solution.to(device=actual_solution.device), category=category, dtype=actual_solution.dtype)
    actual_residual = matrix_dev @ actual_solution - rhs_dev
    expected_residual = matrix_dev @ expected_solution.to(device=actual_solution.device) - rhs_dev
    compare(torch.linalg.vector_norm(actual_residual), torch.linalg.vector_norm(expected_residual), category=category, dtype=_real_dtype_for_complex(actual_solution.dtype))
    for index in range(1, min(len(actual_items), len(expected_items))):
        compare(actual_items[index], expected_items[index], category=category, dtype=actual_items[index].dtype)


def _normalize_dim(dim, ndim: int) -> int:
    dim = int(dim)
    return dim + ndim if dim < 0 else dim


def _selection_dim(op_name: str, context: dict, input_tensor: torch.Tensor) -> int:
    kwargs = _context_kwargs(context)
    args = _context_args(context)
    if "dim" in kwargs and kwargs["dim"] is not None:
        return _normalize_dim(kwargs["dim"], input_tensor.ndim)
    name = _normalize_name(op_name)
    if any(token in name for token in ("topk", "kthvalue")):
        return _normalize_dim(args[1] if len(args) > 1 else -1, input_tensor.ndim)
    if args:
        first = args[0]
        if isinstance(first, str) and input_tensor.names:
            return input_tensor.names.index(first)
        if isinstance(first, int):
            return _normalize_dim(first, input_tensor.ndim)
    return input_tensor.ndim - 1


def _has_reduction_dim(context: dict) -> bool:
    kwargs = _context_kwargs(context)
    args = _context_args(context)
    return kwargs.get("dim") is not None or (bool(args) and isinstance(args[0], int))


def _gather_input_values(input_tensor: torch.Tensor, indices: torch.Tensor, dim: int) -> torch.Tensor:
    source = input_tensor.to(indices.device)
    gather_indices = indices
    squeeze = False
    if gather_indices.ndim < source.ndim:
        gather_indices = gather_indices.unsqueeze(dim)
        squeeze = True
    gathered = torch.gather(source, dim, gather_indices.to(torch.long))
    return gathered.squeeze(dim) if squeeze else gathered


def _compare_indexed_values(op_name: str, input_tensor: torch.Tensor, values: torch.Tensor, indices: torch.Tensor, dim: int, compare) -> None:
    gathered = _gather_input_values(input_tensor, indices, dim)
    compare(gathered, values, category="exact", dtype=values.dtype)


def _check_sortlike(op_name, actual, expected, context, category, dtype, compare) -> None:
    input_tensor = _context_input(context)
    dim = _selection_dim(op_name, context, input_tensor)
    kwargs = _context_kwargs(context)
    stable = bool(kwargs.get("stable", False)) or "stable" in _normalize_name(op_name)
    sorted_output = bool(kwargs.get("sorted", True))
    if "argsort" in _normalize_name(op_name):
        actual_indices = _tensor_tuple(actual, op_name, min_len=1)[0]
        expected_indices = _tensor_tuple(expected, op_name, min_len=1)[0]
        _assert_tensor_metadata(op_name, actual_indices, expected_indices, 0)
        if stable:
            compare(actual_indices, expected_indices, category="exact", dtype=actual_indices.dtype)
            return
        actual_values = _gather_input_values(input_tensor, actual_indices, dim)
        expected_values = _gather_input_values(input_tensor, expected_indices.to(actual_indices.device), dim)
        compare(actual_values, expected_values, category=category, dtype=input_tensor.dtype)
        return
    actual_values, actual_indices = _tensor_tuple(actual, op_name, min_len=2)[:2]
    expected_values, expected_indices = _tensor_tuple(expected, op_name, min_len=2)[:2]
    _assert_tensor_metadata(op_name, actual_values, expected_values, 0)
    _assert_tensor_metadata(op_name, actual_indices, expected_indices, 1)
    if "topk" in _normalize_name(op_name) and not sorted_output:
        compare(
            torch.sort(actual_values, dim=dim).values,
            torch.sort(expected_values.to(actual_values.device), dim=dim).values,
            category=category,
            dtype=actual_values.dtype,
        )
    else:
        compare(actual_values, expected_values, category=category, dtype=actual_values.dtype)
    if stable:
        compare(actual_indices, expected_indices, category="exact", dtype=actual_indices.dtype)
    else:
        _compare_indexed_values(op_name, input_tensor, actual_values, actual_indices, dim, compare)


def _check_arg_reduce(op_name, actual, expected, context, category, dtype, compare) -> None:
    input_tensor = _context_input(context)
    indices = _tensor_tuple(actual, op_name, min_len=1)[0]
    expected_indices = _tensor_tuple(expected, op_name, min_len=1)[0]
    _assert_tensor_metadata(op_name, indices, expected_indices, 0)
    source = input_tensor.to(indices.device)
    if not _has_reduction_dim(context):
        flat_source = source.reshape(-1)
        flat_indices = indices.to(torch.long)
        if not bool(((flat_indices.detach().cpu() >= 0) & (flat_indices.detach().cpu() < flat_source.numel())).all()):
            raise AssertionError(f"{op_name} flattened indices are outside input bounds")
        values = flat_source[flat_indices]
        target = torch.amin(flat_source) if "argmin" in _normalize_name(op_name) else torch.amax(flat_source)
        compare(values, target.expand_as(values), category=category, dtype=input_tensor.dtype)
        return
    dim = _selection_dim(op_name, context, input_tensor)
    values = _gather_input_values(input_tensor, indices, dim)
    if "argmin" in _normalize_name(op_name):
        target = torch.amin(input_tensor.to(values.device), dim=dim)
    else:
        target = torch.amax(input_tensor.to(values.device), dim=dim)
    compare(values, target, category=category, dtype=input_tensor.dtype)


def _check_value_index_pair(op_name, actual, expected, context, category, dtype, compare) -> None:
    input_tensor = _context_input(context)
    actual_tensors = _tensor_tuple(actual, op_name, min_len=1)
    if len(actual_tensors) < 2:
        raise ContractNotApplicable
    actual_values, actual_indices = actual_tensors[:2]
    expected_values, expected_indices = _tensor_tuple(expected, op_name, min_len=2)[:2]
    _assert_tensor_metadata(op_name, actual_values, expected_values, 0)
    _assert_tensor_metadata(op_name, actual_indices, expected_indices, 1)
    compare(actual_values, expected_values, category=category, dtype=actual_values.dtype)
    dim = _selection_dim(op_name, context, input_tensor)
    _compare_indexed_values(op_name, input_tensor, actual_values, actual_indices, dim, compare)


def _check_cumminmax(op_name, actual, expected, context, category, dtype, compare) -> None:
    _check_value_index_pair(op_name, actual, expected, context, category, dtype, compare)


def _check_mode(op_name, actual, expected, context, category, dtype, compare) -> None:
    input_tensor = _context_input(context)
    actual_values, actual_indices = _tensor_tuple(actual, op_name, min_len=2)[:2]
    expected_values, expected_indices = _tensor_tuple(expected, op_name, min_len=2)[:2]
    _assert_tensor_metadata(op_name, actual_values, expected_values, 0)
    _assert_tensor_metadata(op_name, actual_indices, expected_indices, 1)
    dim = _selection_dim(op_name, context, input_tensor)
    rows = input_tensor.detach().cpu().movedim(dim, -1).reshape(-1, input_tensor.shape[dim])
    values = actual_values.detach().cpu().reshape(-1)
    indices = actual_indices.detach().cpu().reshape(-1).to(torch.long)
    if rows.shape[0] != values.numel() or values.numel() != indices.numel():
        raise AssertionError(f"{op_name} value/index shapes are incompatible with reduction slices")
    for row_index, (row, value, index) in enumerate(zip(rows, values, indices)):
        index_value = int(index.item())
        if not 0 <= index_value < row.numel():
            raise AssertionError(f"{op_name} row {row_index} index {index_value} is out of bounds")
        if not _scalar_semantically_equal(row[index_value], value):
            raise AssertionError(f"{op_name} row {row_index} index does not point to its returned value")
        candidate_count = _semantic_scalar_count(row, value)
        maximum_count = max(
            _semantic_scalar_count(row, candidate)
            for candidate in row
        )
        if candidate_count != maximum_count:
            raise AssertionError(
                f"{op_name} row {row_index} returned frequency {candidate_count}, "
                f"but the modal frequency is {maximum_count}"
            )


def _scalar_semantically_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if bool(torch.isnan(left)) and bool(torch.isnan(right)):
        return True
    return bool(left == right)


def _semantic_scalar_count(row: torch.Tensor, value: torch.Tensor) -> int:
    if bool(torch.isnan(value)):
        return int(torch.isnan(row).sum().item())
    return int((row == value).sum().item())


def _matches_value_index(op_name: str) -> bool:
    return _matches_any(("median", "nanmedian", "cummax", "cummin"))(op_name) or _contains_any(("cummax", "cummin"))(op_name)


def _max_pool_spatial_ndim(op_name: str, input_tensor: torch.Tensor) -> int:
    name = _normalize_name(op_name)
    if "3d" in name:
        return 3
    if "2d" in name:
        return 2
    if "1d" in name:
        return 1
    return max(1, input_tensor.ndim - 2)


def _check_max_pool_indices(op_name, actual, expected, context, category, dtype, compare) -> None:
    input_tensor = _context_input(context)
    actual_tensors = _tensor_tuple(actual, op_name, min_len=1)
    if len(actual_tensors) < 2:
        raise ContractNotApplicable
    actual_values, actual_indices = actual_tensors[:2]
    expected_values, expected_indices = _tensor_tuple(expected, op_name, min_len=2)[:2]
    _assert_tensor_metadata(op_name, actual_values, expected_values, 0)
    _assert_tensor_metadata(op_name, actual_indices, expected_indices, 1)
    compare(actual_values, expected_values, category=category, dtype=actual_values.dtype)
    spatial_ndim = _max_pool_spatial_ndim(op_name, input_tensor)
    leading_shape = tuple(input_tensor.shape[:-spatial_ndim])
    if tuple(actual_values.shape[: len(leading_shape)]) != leading_shape:
        raise AssertionError(f"{op_name} output leading dimensions do not match input")
    source = input_tensor.to(actual_values.device)
    flat_source = source.reshape(*leading_shape, -1)
    flat_indices = actual_indices.to(torch.long).reshape(*leading_shape, -1)
    if not bool(((flat_indices.detach().cpu() >= 0) & (flat_indices.detach().cpu() < flat_source.shape[-1])).all()):
        raise AssertionError(f"{op_name} indices are outside input pooling plane bounds")
    gathered = torch.gather(flat_source, -1, flat_indices).reshape_as(actual_values)
    compare(gathered, actual_values, category="exact", dtype=actual_values.dtype)


def _check_max_unpool(op_name, actual, expected, context, category, dtype, compare) -> None:
    input_tensor = _context_input(context)
    args = _context_args(context)
    if not args or not isinstance(args[0], torch.Tensor):
        raise AssertionError(f"{op_name} requires max-unpool index context")
    indices = args[0]
    actual_output = _tensor_tuple(actual, op_name, min_len=1)[0]
    expected_output = _tensor_tuple(expected, op_name, min_len=1)[0]
    _assert_tensor_metadata(op_name, actual_output, expected_output, 0)
    spatial_ndim = _max_pool_spatial_ndim(op_name, input_tensor)
    spatial_size = 1
    for size in input_tensor.shape[-spatial_ndim:]:
        spatial_size *= size
    source_rows = input_tensor.detach().cpu().reshape(-1, spatial_size)
    index_rows = indices.detach().cpu().to(torch.long).reshape(source_rows.shape)
    output_rows = actual_output.detach().cpu().reshape(source_rows.shape[0], -1)
    for row_index, (source_row, index_row, output_row) in enumerate(zip(source_rows, index_rows, output_rows)):
        legal_writers: dict[int, list[torch.Tensor]] = {}
        for source_value, destination in zip(source_row, index_row):
            destination_index = int(destination.item())
            if not 0 <= destination_index < output_row.numel():
                raise AssertionError(f"{op_name} row {row_index} index {destination_index} is out of bounds")
            legal_writers.setdefault(destination_index, []).append(source_value)
        for destination_index, actual_value in enumerate(output_row):
            candidates = legal_writers.get(destination_index)
            if candidates is None:
                if not _scalar_semantically_equal(actual_value, torch.zeros_like(actual_value)):
                    raise AssertionError(f"{op_name} row {row_index} unwritten destination is not zero")
            elif not any(_scalar_semantically_equal(actual_value, candidate) for candidate in candidates):
                raise AssertionError(
                    f"{op_name} row {row_index} destination {destination_index} "
                    "does not contain any legal writer value"
                )


def _check_structural(op_name, actual, expected, context, category, dtype, compare) -> None:
    def check_item(actual_item, expected_item):
        if isinstance(actual_item, torch.Tensor) and isinstance(expected_item, torch.Tensor):
            _assert_tensor_metadata(op_name, actual_item, expected_item)
            if not _matches_uninitialized(op_name) and (actual_item.is_floating_point() or actual_item.is_complex()):
                values = _finite_values(actual_item)
                if not bool(torch.isfinite(values.detach().cpu()).all()):
                    raise AssertionError(f"{op_name} produced non-finite random/structural output")
            return
        if isinstance(actual_item, (list, tuple)) and isinstance(expected_item, (list, tuple)):
            if len(actual_item) != len(expected_item):
                raise AssertionError(f"{op_name} output length mismatch: {len(actual_item)} vs {len(expected_item)}")
            for nested_actual, nested_expected in zip(actual_item, expected_item):
                check_item(nested_actual, nested_expected)
            return
        raise AssertionError(f"{op_name} structural output type mismatch")

    check_item(actual, expected)


def _matches_rrelu_forward(op_name: str) -> bool:
    name = _normalize_name(op_name)
    return "backward" not in name and (
        _matches_any(("nn.functional.rrelu", "rrelu_with_noise", "rrelu_with_noise_functional"))(name)
        or "rrelu_with_noise" in name
    )


def _rrelu_parameters(op_name: str, context: dict) -> tuple[float, float, bool]:
    name = _normalize_name(op_name)
    args = _context_args(context)
    kwargs = _context_kwargs(context)
    offset = 1 if "rrelu_with_noise" in name else 0

    def argument(key, index, default):
        if key in kwargs:
            return kwargs[key]
        position = offset + index
        return args[position] if position < len(args) else default

    return (
        float(argument("lower", 0, 1.0 / 8.0)),
        float(argument("upper", 1, 1.0 / 3.0)),
        bool(argument("training", 2, False)),
    )


def _check_rrelu(op_name, actual, expected, context, category, dtype, compare) -> None:
    input_tensor = _context_input(context)
    actual_output = _tensor_tuple(actual, op_name, min_len=1)[0]
    expected_output = _tensor_tuple(expected, op_name, min_len=1)[0]
    _assert_tensor_metadata(op_name, actual_output, expected_output, 0)
    if tuple(actual_output.shape) != tuple(input_tensor.shape):
        raise AssertionError(f"{op_name} output shape does not match its input")

    lower, upper, training = _rrelu_parameters(op_name, context)
    if lower > upper:
        raise AssertionError(f"{op_name} received lower={lower} greater than upper={upper}")

    source = input_tensor.to(actual_output.device)
    nonnegative = source >= 0
    if bool(nonnegative.any()):
        compare(
            actual_output[nonnegative],
            source[nonnegative],
            category="exact",
            dtype=actual_output.dtype,
        )
    nan_source = torch.isnan(source)
    if bool(nan_source.any()) and not bool(torch.isnan(actual_output[nan_source]).all()):
        raise AssertionError(f"{op_name} must propagate NaN inputs")

    negative = source < 0
    finite_negative = negative & torch.isfinite(source)
    if bool(finite_negative.any()):
        slopes = (actual_output[finite_negative] / source[finite_negative]).detach().cpu()
        if training:
            epsilon = 32 * torch.finfo(slopes.dtype).eps
            if not bool(((slopes >= lower - epsilon) & (slopes <= upper + epsilon)).all()):
                raise AssertionError(
                    f"{op_name} negative-input slopes must be in [{lower}, {upper}] during training"
                )
        else:
            expected_slope = torch.full_like(slopes, (lower + upper) / 2)
            torch.testing.assert_close(slopes, expected_slope)

    negative_infinite = negative & torch.isneginf(source)
    if not bool(negative_infinite.any()):
        return
    outputs = actual_output[negative_infinite].detach().cpu()
    if training:
        reachable = (
            (torch.isneginf(outputs) if upper > 0 else torch.zeros_like(outputs, dtype=torch.bool))
            | (torch.isposinf(outputs) if lower < 0 else torch.zeros_like(outputs, dtype=torch.bool))
            | (torch.isnan(outputs) if lower <= 0 <= upper else torch.zeros_like(outputs, dtype=torch.bool))
        )
        if not bool(reachable.all()):
            raise AssertionError(
                f"{op_name} -inf outputs are not reachable from slopes in [{lower}, {upper}]"
            )
    else:
        midpoint = (lower + upper) / 2
        if midpoint > 0:
            valid = torch.isneginf(outputs)
        elif midpoint < 0:
            valid = torch.isposinf(outputs)
        else:
            valid = torch.isnan(outputs)
        if not bool(valid.all()):
            raise AssertionError(f"{op_name} -inf outputs do not match evaluation slope {midpoint}")


def _randomized_linalg_matrix(op_name: str, context: dict) -> torch.Tensor:
    matrix = _context_input(context)
    args = _context_args(context)
    # PyTorch's OpInfo wrapper factors a @ b.mT to exercise non-square and
    # rank-deficient inputs. Dispatcher/public calls simply use the input.
    if args and isinstance(args[0], torch.Tensor):
        right = args[0].to(matrix.device)
        if matrix.ndim >= 2 and right.ndim >= 2 and matrix.shape[-1] == right.shape[-1]:
            matrix = matrix @ right.mT
    if "pca_lowrank" in _normalize_name(op_name) and _context_kwargs(context).get("center", True):
        matrix = matrix - matrix.mean(dim=-2, keepdim=True)
    return matrix


def _check_randomized_linalg(op_name, actual, expected, context, category, dtype, compare) -> None:
    actual_u, actual_s, actual_v = _tensor_tuple(actual, op_name, min_len=3)[:3]
    expected_u, expected_s, expected_v = _tensor_tuple(expected, op_name, min_len=3)[:3]
    for index, (actual_item, expected_item) in enumerate(
        zip((actual_u, actual_s, actual_v), (expected_u, expected_s, expected_v))
    ):
        _assert_tensor_metadata(op_name, actual_item, expected_item, index)
        if not bool(torch.isfinite(_finite_values(actual_item).detach().cpu()).all()):
            raise AssertionError(f"{op_name} output {index} contains non-finite values")

    rank = actual_s.shape[-1]
    if actual_u.shape[-1] != rank or actual_v.shape[-1] != rank:
        raise AssertionError(f"{op_name} factor dimensions do not agree with singular values")
    if bool((actual_s.detach().cpu() < 0).any()):
        raise AssertionError(f"{op_name} singular values must be nonnegative")
    if rank > 1 and bool((actual_s[..., 1:].detach().cpu() > actual_s[..., :-1].detach().cpu()).any()):
        raise AssertionError(f"{op_name} singular values must be nonincreasing")

    gram_u = actual_u.mH @ actual_u
    gram_v = actual_v.mH @ actual_v
    compare(gram_u, _identity_for_square(gram_u), category="linalg", dtype=actual_u.dtype)
    compare(gram_v, _identity_for_square(gram_v), category="linalg", dtype=actual_v.dtype)

    matrix = _randomized_linalg_matrix(op_name, context).to(actual_u.device)
    reconstruction = (actual_u * actual_s.unsqueeze(-2)) @ actual_v.mH
    if tuple(reconstruction.shape) != tuple(matrix.shape):
        raise AssertionError(f"{op_name} factors do not reconstruct the input shape")
    matrix_norm = torch.linalg.vector_norm(matrix).detach().cpu()
    residual_norm = torch.linalg.vector_norm(matrix - reconstruction).detach().cpu()
    if bool(matrix_norm > 0) and not bool(residual_norm < matrix_norm):
        raise AssertionError(
            f"{op_name} randomized approximation residual {residual_norm.item()} "
            f"is not better than the zero approximation {matrix_norm.item()}"
        )


def _complex_lane_matches(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype) -> bool:
    actual_lanes = torch.view_as_real(actual.reshape(()))
    expected_lanes = torch.view_as_real(expected.reshape(()))
    for actual_lane, expected_lane in zip(actual_lanes, expected_lanes):
        if bool(torch.isnan(expected_lane)):
            if not bool(torch.isnan(actual_lane)):
                return False
        elif bool(torch.isinf(expected_lane)):
            if not bool(actual_lane == expected_lane):
                return False
        else:
            tolerance = 2e-5 if dtype == torch.complex128 else 2e-4
            if not bool(torch.isclose(actual_lane, expected_lane, rtol=tolerance, atol=tolerance)):
                return False
    return True


def _complex_product_outcomes(values: tuple[complex, ...], dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    cache: dict[tuple[int, int], set[complex]] = {}

    def outcomes(start: int, stop: int) -> set[complex]:
        key = (start, stop)
        if key in cache:
            return cache[key]
        if stop - start == 1:
            result = {values[start]}
        else:
            result = set()
            for split in range(start + 1, stop):
                for left in outcomes(start, split):
                    for right in outcomes(split, stop):
                        result.add(left * right)
                if len(result) > 256:
                    break
        cache[key] = result
        return result

    return tuple(torch.tensor(value, dtype=dtype) for value in outcomes(0, len(values)))


def _check_complex_product(op_name, actual, expected, context, category, dtype, compare) -> None:
    if not dtype.is_complex:
        raise ContractNotApplicable
    input_tensor = _context_input(context).detach().cpu()
    args = _context_args(context)
    kwargs = _context_kwargs(context)
    raw_dim = args[0] if args else kwargs.get("dim")
    if raw_dim is None or raw_dim == ():
        raw_dims = tuple(range(input_tensor.ndim))
    elif isinstance(raw_dim, int):
        raw_dims = (raw_dim,)
    elif isinstance(raw_dim, (tuple, list)):
        raw_dims = tuple(int(dim) for dim in raw_dim)
    else:
        raise AssertionError(f"{op_name} received unsupported dim value {raw_dim!r}")

    mask = kwargs.get("mask")
    if input_tensor.ndim == 0:
        if raw_dims not in {(), (0,), (-1,)}:
            raise AssertionError(f"{op_name} scalar reduction has invalid dims {raw_dims}")
        input_tensor = input_tensor.reshape(1)
        dims = (0,)
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().reshape(1)
    else:
        dims = tuple(dim % input_tensor.ndim for dim in raw_dims)
        if len(set(dims)) != len(dims):
            raise AssertionError(f"{op_name} reduction dims contain duplicates")

    if mask is None:
        mask_tensor = torch.ones(input_tensor.shape, dtype=torch.bool)
    elif isinstance(mask, torch.Tensor):
        try:
            mask_tensor = torch.broadcast_to(mask.detach().cpu().to(torch.bool), input_tensor.shape)
        except RuntimeError as error:
            raise AssertionError(f"{op_name} mask is not broadcastable to its input") from error
    else:
        mask_tensor = torch.full(input_tensor.shape, bool(mask), dtype=torch.bool)

    actual_tensor = _tensor_tuple(actual, op_name, min_len=1)[0].detach().cpu()
    expected_tensor = _tensor_tuple(expected, op_name, min_len=1)[0]
    _assert_tensor_metadata(op_name, actual_tensor, expected_tensor, 0)
    unreduced_dims = tuple(dim for dim in range(input_tensor.ndim) if dim not in dims)
    permutation = (*unreduced_dims, *dims)
    permuted_input = input_tensor.permute(permutation)
    permuted_mask = mask_tensor.permute(permutation)
    reduction_size = math.prod(input_tensor.shape[dim] for dim in dims)
    group_count = math.prod(input_tensor.shape[dim] for dim in unreduced_dims)
    rows = permuted_input.reshape(group_count, reduction_size)
    mask_rows = permuted_mask.reshape(group_count, reduction_size)
    outputs = actual_tensor.reshape(-1)
    if len(rows) != outputs.numel():
        raise AssertionError(f"{op_name} output shape is incompatible with its reduction slices")
    for row_index, (row, mask_row, output) in enumerate(zip(rows, mask_rows, outputs)):
        values = tuple(
            complex(value) for value, included in zip(row.tolist(), mask_row.tolist()) if included
        )
        if not values:
            candidates = (torch.tensor(1 + 0j, dtype=dtype),)
        elif len(values) == 1:
            candidates = (torch.tensor(values[0], dtype=dtype),)
        else:
            candidates = _complex_product_outcomes(values, dtype)
        if not any(_complex_lane_matches(output, candidate, dtype) for candidate in candidates):
            raise AssertionError(f"{op_name} row {row_index} is not produced by any legal multiplication order")


def _check_welford_mean(op_name, actual, expected, context, category, dtype, compare) -> None:
    actual_dispersion, actual_mean = _tensor_tuple(actual, op_name, min_len=2)[:2]
    expected_dispersion, expected_mean = _tensor_tuple(expected, op_name, min_len=2)[:2]
    _assert_tensor_metadata(op_name, actual_dispersion, expected_dispersion, 0)
    _assert_tensor_metadata(op_name, actual_mean, expected_mean, 1)
    input_tensor = _context_input(context)
    args = _context_args(context)
    kwargs = _context_kwargs(context)
    dim = kwargs.get("dim", args[0] if args else None)
    keepdim = bool(kwargs.get("keepdim", args[-1] if args and isinstance(args[-1], bool) else False))
    if input_tensor.is_complex():
        semantic_mean = torch.complex(
            input_tensor.real.mean(dim=dim, keepdim=keepdim),
            input_tensor.imag.mean(dim=dim, keepdim=keepdim),
        )
    else:
        semantic_mean = input_tensor.mean(dim=dim, keepdim=keepdim)
    compare(actual_mean, semantic_mean.to(actual_mean.device), category="reduction", dtype=actual_mean.dtype)
    compare(actual_dispersion, expected_dispersion, category="reduction", dtype=actual_dispersion.dtype)


_FFT_TRANSFORM_NAMES = frozenset({
    "fft.fft",
    "fft.ifft",
    "fft.fft2",
    "fft.ifft2",
    "fft.fftn",
    "fft.ifftn",
    "fft.rfft",
    "fft.irfft",
    "fft.rfft2",
    "fft.irfft2",
    "fft.rfftn",
    "fft.irfftn",
    "fft.hfft",
    "fft.ihfft",
    "fft.hfft2",
    "fft.ihfft2",
    "fft.hfftn",
    "fft.ihfftn",
    "stft",
    "istft",
})

_MATMUL_OPINFO_DISPATCHERS = {
    "matmul": "aten::matmul",
    "mm": "aten::mm",
    "bmm": "aten::bmm",
    "addmm": "aten::addmm",
    "addbmm": "aten::addbmm",
    "baddbmm": "aten::baddbmm",
    "nn.functional.linear": "aten::linear",
}


def _matches_complex_matmul(op_name: str) -> bool:
    name = _normalize_name(op_name)
    if name in _MATMUL_OPINFO_DISPATCHERS:
        return True
    return name == "aten::linalg_matmul" or name.startswith("aten::linalg_matmul.")


def _complex_matmul_dispatcher(op_name: str) -> str:
    name = _normalize_name(op_name)
    return _MATMUL_OPINFO_DISPATCHERS.get(name, name)


def _check_complex_matmul_determinate(op_name, actual, expected, context, category, dtype, compare) -> None:
    if context.get("input_condition") in (None, "clean") or not dtype.is_complex:
        raise ContractNotApplicable
    actual_tensor = _tensor_tuple(actual, op_name, min_len=1)[0]
    expected_tensor = _tensor_tuple(expected, op_name, min_len=1)[0]
    _assert_tensor_metadata(op_name, actual_tensor, expected_tensor, 0)
    semantic_expected, determinate = matmul_family_determinate_reference(
        _complex_matmul_dispatcher(op_name),
        (_context_input(context), *_context_args(context)),
        _context_kwargs(context),
    )
    if tuple(semantic_expected.shape) != tuple(expected_tensor.shape):
        raise AssertionError(
            f"{op_name} semantic matmul shape mismatch: "
            f"{tuple(semantic_expected.shape)} vs {tuple(expected_tensor.shape)}"
        )
    actual_lanes = torch.view_as_real(actual_tensor)
    expected_lanes = torch.view_as_real(
        semantic_expected.to(expected_tensor.dtype)
    ).to(actual_tensor.device)
    device_mask = determinate.to(actual_tensor.device)
    compare(
        actual_lanes[device_mask],
        expected_lanes[device_mask],
        category="matmul",
        dtype=_real_dtype_for_complex(actual_tensor.dtype),
    )


def _matches_fft_transform(op_name: str) -> bool:
    return _normalize_name(op_name) in _FFT_TRANSFORM_NAMES


def _all_values_finite(tensor: torch.Tensor) -> bool:
    values = torch.view_as_real(tensor) if tensor.is_complex() else tensor
    return bool(torch.isfinite(values.detach().cpu()).all())


def _check_fft_special_contract(op_name, actual, expected, context, category, dtype, compare) -> None:
    if context.get("input_condition") in (None, "clean"):
        raise ContractNotApplicable
    actual_tensor = _tensor_tuple(actual, op_name, min_len=1)[0]
    expected_tensor = _tensor_tuple(expected, op_name, min_len=1)[0]
    _assert_tensor_metadata(op_name, actual_tensor, expected_tensor, 0)
    source = _context_input(context)
    if not isinstance(source, torch.Tensor):
        raise ContractNotApplicable
    name = _normalize_name(op_name)
    if name in {"stft", "istft"}:
        if not _all_values_finite(source) and _all_values_finite(actual_tensor):
            raise AssertionError(f"{op_name} dropped every nonfinite input contribution")
        return

    spec = public_fft_contract_spec(
        op_name,
        source,
        _context_args(context),
        _context_kwargs(context),
    )
    compare_fft_nonfinite_groups(
        actual_tensor,
        expected_tensor.to(actual_tensor.device),
        source,
        spec,
        dtype=actual_tensor.dtype,
        compare=compare,
        label=op_name,
    )


_UNINITIALIZED_NAMES = frozenset({
    "empty",
    "_empty_affine_quantized",
    "empty_like",
    "empty_permuted",
    "empty_strided",
    "new_empty",
    "new_empty_strided",
})


def _matches_uninitialized(op_name: str) -> bool:
    name = _normalize_name(op_name)
    return _matches_any(tuple(sorted(_UNINITIALIZED_NAMES)))(name) or "_empty_affine_quantized" in name


def _matches_value_only_linalg(op_name: str) -> bool:
    return _matches_any((
        "eigvalsh",
        "svdvals",
        "linalg.eigvalsh",
        "linalg.svdvals",
        "linalg_eigvalsh",
        "linalg_svdvals",
        "solve",
        "cholesky",
        "det",
        "lu_solve",
        "lu_unpack",
        "linalg.lu_solve",
        "linalg_lu_solve",
    ))(op_name)


def _matches_eigh_eigenvalues_only(op_name: str) -> bool:
    name = _normalize_name(op_name)
    return "eigh.eigenvalues" in name or "eigh_eigenvalues" in name


NON_UNIQUE_OUTPUT_CONTRACTS: tuple[NonUniqueOutputContract, ...] = (
    NonUniqueOutputContract("eigh_eigenvalues", REPRESENTATION_AMBIGUOUS, STRICT_BY_API, _matches_eigh_eigenvalues_only, None, "Eigenvalue-only eigh overloads have a unique value contract."),
    NonUniqueOutputContract("eigh", REPRESENTATION_AMBIGUOUS, COVERED_BY_CONTRACT, _matches_any(("eigh", "linalg.eigh", "_linalg_eigh", "linalg_eigh")), _check_eigh, "Eigenvectors are sign/phase/eigenspace ambiguous."),
    NonUniqueOutputContract("eig", REPRESENTATION_AMBIGUOUS, COVERED_BY_CONTRACT, _matches_any(("eig", "linalg.eig", "linalg_eig")), _check_eig, "Eigenvalue order and eigenvector phase are ambiguous."),
    NonUniqueOutputContract("eigvals", REPRESENTATION_AMBIGUOUS, COVERED_BY_CONTRACT, _matches_any(("eigvals", "linalg.eigvals", "linalg_eigvals")), _check_eigvals, "General eigenvalue order is unspecified."),
    NonUniqueOutputContract("svd", REPRESENTATION_AMBIGUOUS, COVERED_BY_CONTRACT, _matches_any(("svd", "linalg.svd", "_linalg_svd", "linalg_svd")), _check_svd, "Singular vectors are sign/phase/subspace ambiguous."),
    NonUniqueOutputContract("qr", REPRESENTATION_AMBIGUOUS, COVERED_BY_CONTRACT, _matches_any(("qr", "linalg.qr", "linalg_qr")), _check_qr, "QR factors have sign/phase ambiguity."),
    NonUniqueOutputContract("lu", REPRESENTATION_AMBIGUOUS, COVERED_BY_CONTRACT, _matches_any(("linalg.lu", "linalg_lu", "lu")), _check_lu, "LU pivot choices can be representation ambiguous."),
    NonUniqueOutputContract("lu_factor", REPRESENTATION_AMBIGUOUS, COVERED_BY_CONTRACT, _matches_any(("linalg.lu_factor", "linalg_lu_factor", "linalg.lu_factor_ex", "linalg_lu_factor_ex", "_lu_with_info")), _check_lu, "Packed LU pivots can differ while reconstructing the same input."),
    NonUniqueOutputContract("pinv", REPRESENTATION_AMBIGUOUS, COVERED_BY_CONTRACT, _matches_any(("pinv", "pinverse", "linalg.pinv", "linalg_pinv")), _check_pinv, "Pseudo-inverse entries can be unstable around rank thresholds."),
    NonUniqueOutputContract("lstsq", REPRESENTATION_AMBIGUOUS, COVERED_BY_CONTRACT, _matches_any(("linalg.lstsq", "linalg_lstsq")), _check_lstsq, "Least-squares solutions can be non-unique for rank-deficient inputs."),
    NonUniqueOutputContract("sort", TIE_INDEX_AMBIGUOUS, COVERED_BY_CONTRACT, _matches_any(("sort", "argsort", "topk", "kthvalue", "msort")), _check_sortlike, "Tie indices are ambiguous unless stable=True.", strict_cases=("stable sort indices are exact",)),
    NonUniqueOutputContract("arg_reduce", TIE_INDEX_AMBIGUOUS, COVERED_BY_CONTRACT, _matches_any(("argmax", "argmin", "masked.argmax", "masked.argmin")), _check_arg_reduce, "Tie indices are ambiguous for arg reductions."),
    NonUniqueOutputContract("value_index", TIE_INDEX_AMBIGUOUS, COVERED_BY_CONTRACT, _matches_value_index, _check_cumminmax, "Value/index pair indices are ambiguous under ties."),
    NonUniqueOutputContract("mode", TIE_INDEX_AMBIGUOUS, COVERED_BY_CONTRACT, _matches_any(("mode",)), _check_mode, "Mode tie indices are ambiguous; values remain checked."),
    NonUniqueOutputContract("complex_matmul_determinate", REPRESENTATION_AMBIGUOUS, COVERED_BY_CONTRACT, _matches_complex_matmul, _check_complex_matmul_determinate, "Exceptional complex matmul algorithms may differ on indeterminate lanes; expanded real-arithmetic lanes with unique values remain strict.", try_direct_first=False, supports_special_tiers=True),
    NonUniqueOutputContract("fft_special", REPRESENTATION_AMBIGUOUS, COVERED_BY_CONTRACT, _matches_fft_transform, _check_fft_special_contract, "Nonfinite FFT masks vary by legal factorization; metadata, finite transform groups, and propagation into affected groups remain checked.", try_direct_first=False, supports_special_tiers=True),
    NonUniqueOutputContract("complex_product", REPRESENTATION_AMBIGUOUS, COVERED_BY_CONTRACT, _matches_any(("prod", "masked.prod")), _check_complex_product, "Complex nonfinite product masks may vary by legal reassociation, but every result must match a valid multiplication tree.", supports_special_tiers=True),
    NonUniqueOutputContract("welford_mean", REPRESENTATION_AMBIGUOUS, COVERED_BY_CONTRACT, _matches_any(("std_mean", "var_mean")), _check_welford_mean, "The public mean follows arithmetic-mean semantics even when Welford dispersion state is nonfinite.", supports_special_tiers=True),
    NonUniqueOutputContract("fractional_max_pool", TIE_INDEX_AMBIGUOUS, COVERED_BY_CONTRACT, _matches_any(("nn.functional.fractional_max_pool2d", "nn.functional.fractional_max_pool3d", "fractional_max_pool2d", "fractional_max_pool3d")), _check_max_pool_indices, "Fractional pooling uses shared explicit random samples; only tied argmax indices may differ."),
    NonUniqueOutputContract("max_pool_indices", TIE_INDEX_AMBIGUOUS, COVERED_BY_CONTRACT, _contains_any(("max_pool1d", "max_pool2d", "max_pool3d", "adaptive_max_pool1d", "adaptive_max_pool2d", "adaptive_max_pool3d")), _check_max_pool_indices, "Max-pool tie indices are ambiguous; values and index-to-value legality remain checked."),
    NonUniqueOutputContract("max_unpool_writers", TIE_INDEX_AMBIGUOUS, COVERED_BY_CONTRACT, _contains_any(("max_unpool1d", "max_unpool2d", "max_unpool3d")), _check_max_unpool, "Duplicate max-unpool destinations may contain any legal writer value."),
    NonUniqueOutputContract("rrelu", RANDOM_VALUE, COVERED_BY_CONTRACT, _matches_rrelu_forward, _check_rrelu, "Training RReLU samples slopes randomly; positive values are exact and negative slopes remain within the requested interval."),
    NonUniqueOutputContract("random", RANDOM_VALUE, STRUCTURAL_ONLY, _matches_random_value, _check_structural, "Random values have no CPU equality contract without a seeded API guarantee.", try_direct_first=False),
    NonUniqueOutputContract("randomized_linalg", RANDOM_VALUE, COVERED_BY_CONTRACT, _matches_any(("svd_lowrank", "pca_lowrank")), _check_randomized_linalg, "Randomized low-rank bases may differ, but must be orthonormal and form a useful approximation.", try_direct_first=False),
    NonUniqueOutputContract("uninitialized", UNINITIALIZED_VALUE, STRUCTURAL_ONLY, _matches_uninitialized, _check_structural, "Uninitialized allocation values have no value contract.", try_direct_first=False),
    NonUniqueOutputContract("geqrf", REPRESENTATION_AMBIGUOUS, INTENTIONALLY_EXACT, _matches_any(("geqrf",)), None, "Packed reflector format has no standalone legality checker here."),
    NonUniqueOutputContract("orgqr_ormqr", REPRESENTATION_AMBIGUOUS, INTENTIONALLY_EXACT, _matches_any(("orgqr", "ormqr")), None, "These consume packed reflector state; exact output remains the practical contract."),
    NonUniqueOutputContract("value_only_linalg", REPRESENTATION_AMBIGUOUS, STRICT_BY_API, _matches_value_only_linalg, None, "Value-only or unique linalg outputs keep numeric comparison."),
)


def contract_for_op_name(op_name: str) -> NonUniqueOutputContract | None:
    name = _normalize_name(op_name)
    for contract in NON_UNIQUE_OUTPUT_CONTRACTS:
        if contract.matcher(name):
            return contract
    return None


_AMBIGUOUS_AUDIT_PARTS = frozenset({
    "eigh",
    "eig",
    "eigvals",
    "eigvalsh",
    "svd",
    "svdvals",
    "qr",
    "lu",
    "lstsq",
    "pinv",
    "pinverse",
    "sort",
    "argsort",
    "topk",
    "kthvalue",
    "median",
    "nanmedian",
    "cummax",
    "cummin",
    "argmax",
    "argmin",
    "empty",
    "randperm",
    "rand",
    "randn",
    "randint",
    "normal",
    "uniform",
    "bernoulli",
    "geqrf",
    "orgqr",
    "ormqr",
    "svd_lowrank",
    "pca_lowrank",
})


def _looks_ambiguous_for_audit(op_name: str) -> bool:
    name = _normalize_name(op_name)
    parts = frozenset(part for part in re.split(r"[^A-Za-z0-9]+|_", name) if part)
    if _matches_random_value(name):
        return True
    if "max_pool" in name or "adaptive_max_pool" in name:
        return True
    if "lu_factor" in name or "_lu_with_info" in name:
        return True
    return bool(parts & _AMBIGUOUS_AUDIT_PARTS)


def _audit_surface_record(source: str, name: str, *, location: str = "") -> dict | None:
    contract = contract_for_op_name(name)
    if contract is not None:
        return {
            "source": source,
            "name": name,
            "location": location,
            "family": contract.family,
            "ambiguity_type": contract.ambiguity_type,
            "status": contract.status,
        }
    if _looks_ambiguous_for_audit(name):
        return {
            "source": source,
            "name": name,
            "location": location,
            "family": "",
            "ambiguity_type": "",
            "status": "unclassified",
        }
    return None


def _scan_opinfo_surfaces() -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    errors: list[str] = []
    try:
        from torch.testing._internal.common_methods_invocations import op_db
    except Exception as exc:
        return records, [f"opinfo import failed: {type(exc).__name__}: {exc}"]
    for op in op_db:
        names = [str(getattr(op, "name", ""))]
        variant = str(getattr(op, "variant_test_name", "") or "")
        if variant and names[0]:
            names.append(f"{names[0]}.{variant}")
        for alias in getattr(op, "aliases", ()) or ():
            alias_name = str(getattr(alias, "name", "") or "")
            if alias_name:
                names.append(alias_name)
        for name in sorted(set(filter(None, names))):
            record = _audit_surface_record("opinfo", name, location=f"OpInfo:{names[0]}")
            if record is not None:
                records.append(record)
    return records, errors


def _scan_generated_surfaces() -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    errors: list[str] = []
    try:
        from torchcts.core.coverage import SURFACE_KINDS, generated_entries_for
    except Exception as exc:
        return records, [f"generated coverage import failed: {type(exc).__name__}: {exc}"]
    for surface_kind in sorted(SURFACE_KINDS):
        try:
            entries = generated_entries_for(surface_kind)
        except Exception as exc:
            errors.append(f"generated {surface_kind} scan failed: {type(exc).__name__}: {exc}")
            continue
        for entry in entries:
            names = {
                str(entry.get("name") or ""),
                str(entry.get("base_name") or ""),
                str(entry.get("operator") or ""),
            }
            schema = str(entry.get("schema") or "")
            if schema:
                names.add(schema.split("(", 1)[0])
            for name in sorted(name for name in names if name):
                record = _audit_surface_record("generated", name, location=surface_kind)
                if record is not None:
                    records.append(record)
    return records, errors


def _scan_path_shape_surfaces() -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    errors: list[str] = []
    try:
        from torchcts.path_shapes import load_path_shape_corpus
    except Exception as exc:
        return records, [f"path-shape import failed: {type(exc).__name__}: {exc}"]
    try:
        corpus = load_path_shape_corpus()
    except Exception as exc:
        return records, [f"path-shape corpus scan failed: {type(exc).__name__}: {exc}"]
    for case in corpus.get("cases", []):
        names = set(case.get("covers") or ())
        runner = str(case.get("runner") or "")
        if runner:
            names.add(runner)
        for name in sorted(name for name in names if name):
            record = _audit_surface_record("path_shape", name, location=str(case.get("case_id") or ""))
            if record is not None:
                records.append(record)
    return records, errors


def _scan_handwritten_surfaces() -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "operators").glob("test_*.py"))
    patterns = (
        re.compile(r"_compare_non_unique_linalg\(\s*[\"']([^\"']+)[\"']"),
        re.compile(r"torch\.linalg\.([A-Za-z_][A-Za-z0-9_]*)"),
        re.compile(r"torch\.([A-Za-z_][A-Za-z0-9_]*)"),
        re.compile(r"aten::([A-Za-z_][A-Za-z0-9_:.]*)"),
    )
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            return records, [f"handwritten scan failed for {path}: {type(exc).__name__}: {exc}"]
        for pattern in patterns:
            for match in pattern.finditer(text):
                raw_name = match.group(1)
                name = f"linalg.{raw_name}" if pattern.pattern.startswith("torch\\.linalg") else raw_name
                if pattern.pattern.startswith("aten::"):
                    name = f"aten::{raw_name}"
                record = _audit_surface_record("handwritten", name, location=path.name)
                if record is not None:
                    records.append(record)
    return records, []


def _audit_route_files() -> list[dict]:
    root = Path(__file__).resolve().parents[1]
    expected_routes = {
        "opinfo": root / "opinfo" / "test_opinfo_forward.py",
        "generated": root / "generated" / "coverage_helpers.py",
        "path_shape": root / "path_shapes" / "runners" / "common.py",
        "handwritten": root / "operators" / "test_linalg.py",
    }
    routes = []
    for source, path in expected_routes.items():
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            routes.append({
                "source": source,
                "path": str(path),
                "uses_shared_comparator": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        routes.append({
            "source": source,
            "path": str(path),
            "uses_shared_comparator": "compare_non_unique_output" in text,
            "error": "",
        })
    return routes


def compare_non_unique_output_if_applicable(
    op_name: str,
    actual,
    expected,
    *,
    sample=None,
    input=None,
    args=None,
    kwargs=None,
    input_condition: str | None = None,
    category: str,
    dtype: torch.dtype,
    compare,
) -> bool:
    contract = contract_for_op_name(op_name)
    if contract is None or contract.checker is None:
        return False
    if (
        input_condition not in (None, "clean")
        and contract.ambiguity_type == REPRESENTATION_AMBIGUOUS
        and not contract.supports_special_tiers
    ):
        return False

    context = {
        "sample": sample,
        "input": input,
        "args": args,
        "kwargs": kwargs,
        "input_condition": input_condition,
    }
    direct_error = None
    if contract.try_direct_first:
        direct_error = _try_compare_direct(actual, expected, category, dtype, compare)
        if direct_error is None:
            return True

    try:
        contract.checker(op_name, actual, expected, context, category, dtype, compare)
    except ContractNotApplicable:
        return False
    if contract.status in {COVERED_BY_CONTRACT, STRUCTURAL_ONLY} and (contract.try_direct_first or direct_error is None):
        if direct_error is None:
            detail = "CPU value equality is not part of this contract"
        else:
            detail = f"did not match the CPU representation: {direct_error}"
        mark_usable_fallback(
            f"Quality warning: {op_name} matched the {contract.family} legal-output contract "
            f"({contract.ambiguity_type}); {detail}"
        )
    return True


def compare_non_unique_output(
    op_name: str,
    actual,
    expected,
    *,
    sample=None,
    input=None,
    args=None,
    kwargs=None,
    input_condition: str | None = None,
    category: str,
    dtype: torch.dtype,
    compare,
) -> None:
    if not compare_non_unique_output_if_applicable(
        op_name,
        actual,
        expected,
        sample=sample,
        input=input,
        args=args,
        kwargs=kwargs,
        input_condition=input_condition,
        category=category,
        dtype=dtype,
        compare=compare,
    ):
        _compare_direct(actual, expected, category, dtype, compare)


def ambiguous_output_audit() -> dict:
    rows = []
    for contract in NON_UNIQUE_OUTPUT_CONTRACTS:
        rows.append({
            "family": contract.family,
            "ambiguity_type": contract.ambiguity_type,
            "status": contract.status,
            "reason": contract.reason,
            "strict_cases": list(contract.strict_cases),
            "unsupported_cases": list(contract.unsupported_cases),
        })
    status_counts = Counter(row["status"] for row in rows)
    records: list[dict] = []
    source_errors: list[str] = []
    for scanner in (
        _scan_opinfo_surfaces,
        _scan_generated_surfaces,
        _scan_path_shape_surfaces,
        _scan_handwritten_surfaces,
    ):
        scanned, errors = scanner()
        records.extend(scanned)
        source_errors.extend(errors)
    unique_records = {
        (record["source"], record["name"], record["location"], record["status"], record["family"]): record
        for record in records
    }
    records = [unique_records[key] for key in sorted(unique_records)]
    surface_status_counts = Counter(record["status"] for record in records)
    source_counts = Counter(record["source"] for record in records)
    unclassified = [record for record in records if record["status"] == "unclassified"]
    routes = _audit_route_files()
    route_errors = [
        f"{route['source']} route does not use the shared non-unique comparator"
        for route in routes
        if not route["uses_shared_comparator"]
    ]
    return {
        "version": 1,
        "families": rows,
        "status_counts": dict(sorted(status_counts.items())),
        "scanned_surface_count": len(records),
        "classified_surface_count": len(records) - len(unclassified),
        "surface_status_counts": dict(sorted(surface_status_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "routes": routes,
        "source_errors": source_errors,
        "route_errors": route_errors,
        "unclassified": unclassified,
        "ok": not unclassified and not source_errors and not route_errors,
    }


def run_ambiguous_output_audit_command(*, json_output: bool = False) -> int:
    import json

    audit = ambiguous_output_audit()
    if json_output:
        print(json.dumps(audit, indent=2, sort_keys=True))
    else:
        print("TorchCTS non-unique output contract audit")
        print(f"Families: {len(audit['families'])}")
        print("Status: " + ", ".join(f"{key}={value}" for key, value in sorted(audit["status_counts"].items())))
        print(f"Scanned surfaces: {audit['scanned_surface_count']}")
        print("Sources: " + ", ".join(f"{key}={value}" for key, value in sorted(audit["source_counts"].items())))
        print(f"Unclassified: {len(audit['unclassified'])}")
        for row in audit["families"]:
            print(f"- {row['family']}: {row['status']} ({row['ambiguity_type']})")
        for error in audit["source_errors"] + audit["route_errors"]:
            print(f"ERROR: {error}")
        for row in audit["unclassified"][:20]:
            print(f"UNCLASSIFIED: {row['source']} {row['name']} ({row['location']})")
    return 0 if audit["ok"] else 1
