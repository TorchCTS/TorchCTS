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
from pathlib import Path
import re
from typing import Callable

import torch

from torchcts.core.comparer import mark_usable_fallback, restore_metrics, snapshot_metrics


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
        "nn.functional.fractional_max_pool2d",
        "nn.functional.fractional_max_pool3d",
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
    if "dropout" in name and "no_dropout" not in name:
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
        unused = set(range(actual_row.numel()))
        ordered = []
        for expected_item in expected_row:
            best = min(unused, key=lambda idx: float(torch.abs(actual_row[idx] - expected_item).item()))
            unused.remove(best)
            ordered.append(best)
        index_rows.append(ordered)
    indices = torch.tensor(index_rows, dtype=torch.long, device=actual_values.device).reshape(actual_values.shape)
    ordered_values = torch.gather(actual_values, -1, indices)
    compare(ordered_values, expected_values, category=category, dtype=actual_values.dtype)
    return ordered_values, indices


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
    _check_value_index_pair(op_name, actual, expected, context, category, dtype, compare)


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
        "eigvals",
        "eigvalsh",
        "svdvals",
        "linalg.eigvals",
        "linalg.eigvalsh",
        "linalg.svdvals",
        "linalg_eigvals",
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
    NonUniqueOutputContract("max_pool_indices", TIE_INDEX_AMBIGUOUS, COVERED_BY_CONTRACT, _contains_any(("max_pool1d", "max_pool2d", "max_pool3d", "adaptive_max_pool1d", "adaptive_max_pool2d", "adaptive_max_pool3d")), _check_max_pool_indices, "Max-pool tie indices are ambiguous; values and index-to-value legality remain checked."),
    NonUniqueOutputContract("random", RANDOM_VALUE, STRUCTURAL_ONLY, _matches_random_value, _check_structural, "Random values have no CPU equality contract without a seeded API guarantee.", try_direct_first=False),
    NonUniqueOutputContract("randomized_linalg", RANDOM_VALUE, STRUCTURAL_ONLY, _matches_any(("svd_lowrank", "pca_lowrank")), _check_structural, "Randomized low-rank linalg returns approximation bases without a CPU representation contract.", try_direct_first=False),
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
    if input_condition not in (None, "clean") and contract.ambiguity_type == REPRESENTATION_AMBIGUOUS:
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
