# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
import json
import re
import torch

from torchcts.core.version_rules import (
    add_remove_items,
    cumulative_versioned_set,
    iter_version_rule_entries,
    parse_version_rule_key,
)

# ---------------------------------------------------------------------------
# Input Condition Tiers
# ---------------------------------------------------------------------------

class InputCondition:
    """Classification of opinfo sample inputs by NaN/Inf content."""
    CLEAN = "clean"      # All finite values, no NaN/Inf
    HAS_NAN = "has_nan"  # NaN + ±Inf present — tests NaN dominance
    HAS_INF = "has_inf"  # ±Inf present, no NaN — tests Inf arithmetic


_ORTHOGONAL_POLYNOMIAL_OPS = frozenset({
    "special_chebyshev_polynomial_t",
    "special_chebyshev_polynomial_u",
    "special_chebyshev_polynomial_v",
    "special_chebyshev_polynomial_w",
    "special_hermite_polynomial_h",
    "special_hermite_polynomial_he",
    "special_laguerre_polynomial_l",
    "special_legendre_polynomial_p",
    "special_shifted_chebyshev_polynomial_t",
    "special_shifted_chebyshev_polynomial_u",
    "special_shifted_chebyshev_polynomial_v",
    "special_shifted_chebyshev_polynomial_w",
})


_NO_IEEE754_PROPAGATION_OPS = frozenset({
    # Uninitialized memory has no value contract, so NaN/Inf propagation is not
    # meaningful.
    "empty",
    "empty_like",
    "empty_permuted",
    "empty_strided",
    "new_empty",
    "new_empty_strided",
    # Random and dropout-family outputs are nondeterministic. Some inputs are
    # distribution parameters with stricter domains than arbitrary IEEE values.
    "bernoulli",
    "geometric",
    "multinomial",
    "rand_like",
    "randint",
    "randint_like",
    "randn",
    "randn_like",
    "nn.functional.dropout",
    "nn.functional.dropout2d",
    "nn.functional.dropout3d",
    "nn.functional.alpha_dropout",
    "nn.functional.feature_alpha_dropout",
    "nn.functional.fractional_max_pool2d",
    "nn.functional.fractional_max_pool3d",
    "normal",
    "uniform",
    "log_normal",
    "cauchy",
    "exponential",
})


_NO_GENERIC_BACKWARD_ORACLE_OPS = _NO_IEEE754_PROPAGATION_OPS


def _canonical_op_base(op_name):
    if not op_name:
        return None
    name = str(op_name)
    if name.startswith("aten::"):
        name = name[len("aten::"):]
    if name.startswith("torch."):
        name = name[len("torch."):]
    if name.startswith("special."):
        return "special_" + name[len("special."):]
    return name.split(".", 1)[0]


def _preserve_nonfinite_control_args(op_name):
    """Return True when IEEE injection must not touch control-parameter tensors."""

    return _canonical_op_base(op_name) in _ORTHOGONAL_POLYNOMIAL_OPS


def _resolve_tensor_value_bits(t):
    if t.is_conj():
        t = t.resolve_conj()
    if t.is_neg():
        t = t.resolve_neg()
    return t


_SPARSE_VALUE_LAYOUTS = frozenset({
    torch.sparse_coo,
    torch.sparse_csr,
    torch.sparse_csc,
    torch.sparse_bsr,
    torch.sparse_bsc,
})


def _dense_value_tensor_for_scan(t):
    t = _resolve_tensor_value_bits(t)
    if t.layout == torch.sparse_coo:
        return t.values() if t.is_coalesced() else t._values()
    if t.layout in _SPARSE_VALUE_LAYOUTS:
        return t.values()
    return t


def _rebuild_sparse_tensor_with_values(t, values):
    if t.layout == torch.sparse_coo:
        rebuilt = torch.sparse_coo_tensor(
            t._indices(),
            values,
            size=t.shape,
            requires_grad=t.requires_grad,
        )
        return rebuilt.coalesce() if t.is_coalesced() else rebuilt
    if t.layout == torch.sparse_csr:
        return torch.sparse_csr_tensor(
            t.crow_indices(),
            t.col_indices(),
            values,
            size=t.shape,
            requires_grad=t.requires_grad,
        )
    if t.layout == torch.sparse_csc:
        return torch.sparse_csc_tensor(
            t.ccol_indices(),
            t.row_indices(),
            values,
            size=t.shape,
            requires_grad=t.requires_grad,
        )
    if t.layout == torch.sparse_bsr:
        return torch.sparse_bsr_tensor(
            t.crow_indices(),
            t.col_indices(),
            values,
            size=t.shape,
            requires_grad=t.requires_grad,
        )
    if t.layout == torch.sparse_bsc:
        return torch.sparse_bsc_tensor(
            t.ccol_indices(),
            t.row_indices(),
            values,
            size=t.shape,
            requires_grad=t.requires_grad,
        )
    return t


def classify_sample(sample):
    """Classify a SampleInput by whether its tensors contain NaN/Inf.

    Returns InputCondition.CLEAN, HAS_NAN, or HAS_INF.
    When both NaN and Inf are present, returns HAS_NAN (NaN dominates).
    """
    has_nan = False
    has_inf = False

    def _check(obj):
        nonlocal has_nan, has_inf
        if isinstance(obj, torch.Tensor) and (obj.is_floating_point() or obj.is_complex()):
            obj = _dense_value_tensor_for_scan(obj)
            flat = obj.detach().reshape(-1)
            if torch.isnan(flat).any():
                has_nan = True
            if torch.isinf(flat).any():
                has_inf = True
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _check(item)
        elif isinstance(obj, dict):
            for item in obj.values():
                _check(item)

    _check(sample.input)
    _check(sample.args)
    _check(sample.kwargs)

    if has_nan:
        return InputCondition.HAS_NAN  # NaN dominates — has_both maps here
    if has_inf:
        return InputCondition.HAS_INF
    return InputCondition.CLEAN


def _transform_tensor(t, target_condition, seed):
    """Transform a single float/complex tensor toward target_condition.

    Returns transformed tensor (cloned). Original untouched.
    """
    if not isinstance(t, torch.Tensor):
        return t
    if not (t.is_floating_point() or t.is_complex()):
        return t

    if t.layout in _SPARSE_VALUE_LAYOUTS:
        values = _dense_value_tensor_for_scan(t)
        new_values = _transform_tensor(values, target_condition, seed)
        return _rebuild_sparse_tensor_with_values(t, new_values)

    t = _resolve_tensor_value_bits(t).clone()

    # For complex, work on the real view
    is_complex = t.is_complex()
    if is_complex:
        work = torch.view_as_real(t)
    else:
        work = t

    flat = work.reshape(-1)
    n = flat.numel()
    if n == 0:
        return t

    cur_nan = torch.isnan(flat)
    cur_inf = torch.isinf(flat)

    if target_condition == InputCondition.CLEAN:
        # Scrub all NaN/Inf -> finite values
        if cur_nan.any() or cur_inf.any():
            finfo = torch.finfo(flat.dtype)
            safe_max = finfo.max / 2
            flat = torch.nan_to_num(flat, nan=0.0, posinf=safe_max, neginf=-safe_max)
            work.copy_(flat.view_as(work))
        return t

    # Deterministic RNG for injection positions
    g = torch.Generator()
    g.manual_seed(seed ^ n)
    n_inject = max(1, n // 20)  # ~5% of elements

    if target_condition == InputCondition.HAS_NAN:
        # Must have NaN + ±Inf. Scrub nothing, add what's missing.
        need_nan = not cur_nan.any()
        need_pinf = not (flat == float('inf')).any()
        need_ninf = not (flat == float('-inf')).any()

        if need_nan or need_pinf or need_ninf:
            # Get injection positions from finite elements only
            finite_mask = torch.isfinite(flat)
            finite_indices = finite_mask.nonzero(as_tuple=False).squeeze(-1)
            if finite_indices.numel() > 0:
                perm = torch.randperm(finite_indices.numel(), generator=g)
                pos = 0
                if need_nan:
                    count = max(1, n_inject // 2)
                    for i in range(min(count, perm.numel() - pos)):
                        flat[finite_indices[perm[pos + i]]] = float('nan')
                    pos += count
                if need_pinf:
                    count = max(1, n_inject // 4)
                    for i in range(min(count, perm.numel() - pos)):
                        flat[finite_indices[perm[pos + i]]] = float('inf')
                    pos += count
                if need_ninf:
                    count = max(1, n_inject // 4)
                    for i in range(min(count, perm.numel() - pos)):
                        flat[finite_indices[perm[pos + i]]] = float('-inf')
            work.copy_(flat.view_as(work))

    elif target_condition == InputCondition.HAS_INF:
        # Must have ±Inf, must NOT have NaN.
        # First scrub NaN
        if cur_nan.any():
            flat[cur_nan] = 0.0

        need_pinf = not (flat == float('inf')).any()
        need_ninf = not (flat == float('-inf')).any()

        if need_pinf or need_ninf:
            finite_mask = torch.isfinite(flat)
            finite_indices = finite_mask.nonzero(as_tuple=False).squeeze(-1)
            if finite_indices.numel() > 0:
                perm = torch.randperm(finite_indices.numel(), generator=g)
                pos = 0
                if need_pinf:
                    count = max(1, n_inject // 2)
                    for i in range(min(count, perm.numel() - pos)):
                        flat[finite_indices[perm[pos + i]]] = float('inf')
                    pos += count
                if need_ninf:
                    count = max(1, n_inject // 2)
                    for i in range(min(count, perm.numel() - pos)):
                        flat[finite_indices[perm[pos + i]]] = float('-inf')
        work.copy_(flat.view_as(work))

    return t


def _transform_obj(obj, target_condition, seed):
    """Recursively transform all float/complex tensors in a nested structure."""
    if isinstance(obj, torch.Tensor):
        return _transform_tensor(obj, target_condition, seed)
    elif isinstance(obj, (list, tuple)):
        transformed = [_transform_obj(item, target_condition, seed + i) for i, item in enumerate(obj)]
        return type(obj)(transformed)
    elif isinstance(obj, dict):
        return {k: _transform_obj(v, target_condition, seed + hash(k) % (2**31)) for k, v in obj.items()}
    return obj


def prepare_sample(sample, target_condition, ieee754_seed=67, sample_index=0, op_name=None):
    """Clone sample and transform ALL float tensors to match the target condition.

    Transforms all float/complex tensors in sample.input, sample.args,
    and sample.kwargs. Complex tensors handled via view_as_real internally.
    For op families with tensor-valued control parameters, such as special
    orthogonal polynomials, non-finite injection is limited to the data input so
    the sample remains valid for the dispatcher contract.

    Args:
        sample: opinfo SampleInput
        target_condition: InputCondition.CLEAN, HAS_NAN, or HAS_INF
        ieee754_seed: base seed from manifest (default 67)
        sample_index: index of this sample in the op's sample list
        op_name: optional OpInfo/dispatcher name used for op-specific sample rules

    Returns new SampleInput. Original untouched.
    """
    from torch.testing._internal.opinfo.core import SampleInput

    if target_condition == InputCondition.CLEAN:
        natural = classify_sample(sample)
        if natural == InputCondition.CLEAN:
            return sample  # Already clean, no clone needed

    seed = ieee754_seed ^ (sample_index * 2654435761)  # Knuth multiplicative hash

    new_input = _transform_obj(sample.input, target_condition, seed)
    if _preserve_nonfinite_control_args(op_name):
        new_args = sample.args
        new_kwargs = sample.kwargs
    else:
        new_args = _transform_obj(sample.args, target_condition, seed + 1000000)
        new_kwargs = _transform_obj(sample.kwargs, target_condition, seed + 2000000)

    return SampleInput(new_input, args=new_args, kwargs=new_kwargs)



DTYPE_TO_STR = {
    torch.float32: "torch.float32",
    torch.float64: "torch.float64",
    torch.float16: "torch.float16",
    torch.bfloat16: "torch.bfloat16",
    torch.int64: "torch.int64",
    torch.int32: "torch.int32",
    torch.int16: "torch.int16",
    torch.int8: "torch.int8",
    torch.uint8: "torch.uint8",
    torch.bool: "torch.bool",
    torch.complex32: "torch.complex32",
    torch.complex64: "torch.complex64",
    torch.complex128: "torch.complex128",
    torch.float8_e4m3fn: "torch.float8_e4m3fn",
    torch.float8_e5m2: "torch.float8_e5m2",
    torch.float8_e4m3fnuz: "torch.float8_e4m3fnuz",
    torch.float8_e5m2fnuz: "torch.float8_e5m2fnuz",
}
STR_TO_DTYPE = {v: k for k, v in DTYPE_TO_STR.items()}

def dtype_to_str(dt):
    return DTYPE_TO_STR.get(dt, str(dt))

def str_to_dtype(s):
    if s.startswith("torch."):
        name = s.replace("torch.", "")
        if hasattr(torch, name):
            return getattr(torch, name)
    return STR_TO_DTYPE.get(s, None)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Dtypes that can carry gradients
DIFFERENTIABLE = frozenset({
    torch.float16, torch.float32, torch.float64, torch.bfloat16,
    torch.complex32, torch.complex64, torch.complex128,
})

# Ops that only work on CUDA — skip everywhere else
SKIP_OPS = frozenset({
    "jiterator_unary", "jiterator_binary",
    "jiterator_4inputs_with_extra_args",
    "jiterator_binary_return_by_ref",
    "jiterator_2inputs_2outputs",
})

def is_cpu_reference_failure(exc):
    """Return True if the exception indicates CPU doesn't implement this op/dtype."""
    if isinstance(exc, NotImplementedError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "not implemented" in msg:
            return True
        if "svd_backward" in msg:
            return True
    return False

# ---------------------------------------------------------------------------
# Manifest dtype filter helper
# ---------------------------------------------------------------------------

def _dtype_matches_manifest(dt, dt_str, supported_dtypes, op_name):
    """Check if a dtype is allowed by the manifest's supported_dtypes config."""
    dtype_filter = supported_dtypes.get(dt, supported_dtypes.get(dt_str))
    if dtype_filter is None:
        return False
    if dtype_filter is True:
        return True
    if isinstance(dtype_filter, str):
        return bool(re.search(dtype_filter, op_name))
    return False

# ---------------------------------------------------------------------------
# IEEE 754 undefined ops + capability check
# ---------------------------------------------------------------------------

# Ops where CPU NaN/Inf behavior is undefined or non-deterministic.
# Excluded from NaN/Inf testing even when ieee754 capability is True.
# Discovery: sync-opinfo --discover-ieee754-undefined
_IEEE754_UNDEFINED_OPS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "opinfo_cache", "ieee754_undefined.json"
)


def _parse_ieee754_version_key(version):
    parsed = parse_version_rule_key(version)
    if parsed is None:
        return None
    return parsed.parts, parsed.specificity, parsed.exact_only, parsed.text


def _ieee754_rule_items(value):
    return add_remove_items(value, default_add_key="ops")


def _ieee754_undefined_rule_entries(data, runtime_version):
    """Return cumulative cache entries that apply to a PyTorch version."""

    return iter_version_rule_entries(data, runtime_version)


def _load_ieee754_undefined():
    """Load the set of ops with undefined CPU NaN/Inf behavior."""
    if os.path.exists(_IEEE754_UNDEFINED_OPS_PATH):
        try:
            with open(_IEEE754_UNDEFINED_OPS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cumulative_versioned_set(data, torch.__version__, default_add_key="ops")
        except Exception:
            pass
    return frozenset()

_IEEE754_UNDEFINED_OPS = _load_ieee754_undefined()


def _ieee754_enabled_for_op(op_name, ieee754_cap):
    """Check if IEEE 754 testing is enabled for this op.

    ieee754_cap can be:
        True      - all float ops (minus _IEEE754_UNDEFINED_OPS)
        False     - disabled
        str       - single regex matched against op_name
        list[str] - array of regexes, any match enables the op
    """
    if ieee754_cap is False or ieee754_cap is None:
        return False
    if _has_invalid_ieee754_cpu_oracle(op_name):
        return False
    if op_name in _NO_IEEE754_PROPAGATION_OPS:
        return False
    if ieee754_cap is True:
        return True
    if isinstance(ieee754_cap, str):
        return bool(re.search(ieee754_cap, op_name))
    if isinstance(ieee754_cap, (list, tuple)):
        return any(re.search(pat, op_name) for pat in ieee754_cap)
    return False

# ---------------------------------------------------------------------------
# Test list builders (replace get_filtered_op_tests + probing)
# ---------------------------------------------------------------------------


def get_forward_op_tests(manifest):
    """Build forward test list from op_db metadata.
    
    Returns list of (op_name, dtype_str, input_condition) triples.
    For float/complex dtypes with ieee754 enabled: emits clean + has_nan + has_inf.
    For int/bool dtypes or ieee754 disabled: emits clean only.
    """
    import torch.testing._internal.common_methods_invocations as cmi

    supported_dtypes = manifest.get("supported_dtypes", {})
    skip_ops = set(manifest.get("skip_ops", [])) | SKIP_OPS
    capabilities = manifest.get("capabilities", {})
    ieee754_cap = capabilities.get("ieee754", True)
    tests = []

    seen = set()
    for op in cmi.op_db:
        if op.name in seen or op.name in skip_ops:
            continue
        seen.add(op.name)

        # Skip sparse ops if backend doesn't support sparse
        if (getattr(op, "supports_sparse", False) or
                getattr(op, "supports_sparse_csr", False)):
            if not capabilities.get("sparse", False):
                continue

        for d in sorted(op.dtypes, key=lambda x: str(x)):
            dt_str = dtype_to_str(d)

            if _dtype_matches_manifest(d, dt_str, supported_dtypes, op.name):
                tests.append((op.name, dt_str, InputCondition.CLEAN))
                # Add IEEE 754 tiers for float/complex dtypes
                if (d.is_floating_point or d.is_complex) and _ieee754_enabled_for_op(op.name, ieee754_cap):
                    tests.append((op.name, dt_str, InputCondition.HAS_NAN))
                    tests.append((op.name, dt_str, InputCondition.HAS_INF))

    return tests


def get_backward_op_tests(manifest):
    """Build backward test list from op_db metadata.
    
    Static rules applied:
      1. Filter to differentiable dtypes (float/complex)
      2. Skip if supports_autograd == False
      3. Skip jiterator_* (in SKIP_OPS)
    """
    import torch.testing._internal.common_methods_invocations as cmi

    supported_dtypes = manifest.get("supported_dtypes", {})
    skip_ops = set(manifest.get("skip_ops", [])) | SKIP_OPS
    capabilities = manifest.get("capabilities", {})
    tests = []

    seen = set()
    for op in cmi.op_db:
        if op.name in seen or op.name in skip_ops:
            continue
        seen.add(op.name)
        if op.name in _NO_GENERIC_BACKWARD_ORACLE_OPS:
            continue

        # Rule 2: skip ops that don't support autograd
        if not getattr(op, "supports_autograd", True):
            continue

        if (getattr(op, "supports_sparse", False) or
                getattr(op, "supports_sparse_csr", False)):
            if not capabilities.get("sparse", False):
                continue

        for d in sorted(op.backward_dtypes, key=lambda x: str(x)):
            # Rule 1: only differentiable dtypes
            if d not in DIFFERENTIABLE:
                continue

            dt_str = dtype_to_str(d)

            if _dtype_matches_manifest(d, dt_str, supported_dtypes, op.name):
                tests.append((op.name, dt_str))

    return tests


# Ops that use iterative algorithms converging to a solution.
# NaN/Inf inputs prevent convergence, causing infinite loops or extreme slowness.
# These are pre-classified as having undefined NaN/Inf behavior.
_CONVERGENCE_OPS = frozenset({
    # Eigendecomposition (iterative QR/divide-and-conquer)
    "linalg.eig", "linalg.eigh", "linalg.eigvals", "linalg.eigvalsh",
    "eig", "eigvals",
    # SVD (iterative bidiagonal reduction)
    "linalg.svd", "linalg.svdvals", "svd",
    # Matrix decompositions that iterate
    "linalg.cholesky", "linalg.cholesky_ex", "cholesky",
    "linalg.qr", "qr", "geqrf", "ormqr", "linalg.householder_product",
    "linalg.lu", "linalg.lu_factor", "linalg.lu_factor_ex", "lu",
    # Solvers (forward/back substitution with pivoting)
    "linalg.solve", "linalg.solve_triangular", "linalg.solve_ex",
    "linalg.lstsq", "triangular_solve", "lu_solve",
    # Inverse (uses LU internally)
    "linalg.inv", "linalg.inv_ex", "linalg.pinv",
    # Matrix functions (series expansion / eigendecomposition)
    "linalg.matrix_exp", "matrix_exp", "linalg.matrix_power",
    # Determinant / norm (can loop on degenerate inputs)
    "linalg.det", "linalg.slogdet", "det", "slogdet", "logdet",
    # Condition number
    "linalg.cond",
})


def _has_invalid_ieee754_cpu_oracle(op_name):
    """Return True when NaN/Inf CPU oracle construction is unsafe for an op."""

    return op_name in _CONVERGENCE_OPS or op_name in _IEEE754_UNDEFINED_OPS


def discover_ieee754_undefined_ops():
    """Discover ops where CPU NaN/Inf behavior is non-deterministic.

    For each float op, injects NaN/Inf and runs twice on CPU.
    If outputs don't match, the op has undefined NaN/Inf behavior.

    Returns set of op names with undefined behavior.
    """
    import warnings
    import torch.testing._internal.common_methods_invocations as cmi

    undefined = set(_CONVERGENCE_OPS)
    seen = set(_CONVERGENCE_OPS)

    # Suppress PyTorch deprecation/beta warnings during discovery
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Per-op timeout to catch linalg ops that hang with NaN/Inf
        import signal

        class _OpTimeout(Exception):
            pass

        def _timeout_handler(signum, frame):
            raise _OpTimeout()

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)

        try:
            for op in cmi.op_db:
                if op.name in seen or op.name in SKIP_OPS:
                    continue
                seen.add(op.name)

                # Only test with float32
                if torch.float32 not in op.dtypes:
                    continue

                try:
                    signal.alarm(5)  # 5 second timeout per op
                    samples = list(op.sample_inputs("cpu", torch.float32, requires_grad=False))
                    if not samples:
                        signal.alarm(0)
                        continue

                    sample = samples[0]
                    for condition in (InputCondition.HAS_NAN, InputCondition.HAS_INF):
                        prepared = prepare_sample(
                            sample,
                            condition,
                            ieee754_seed=67,
                            sample_index=0,
                            op_name=op.name,
                        )

                        try:
                            result1 = op.op(prepared.input, *prepared.args, **prepared.kwargs)
                            result2 = op.op(prepared.input, *prepared.args, **prepared.kwargs)
                        except (_OpTimeout, KeyboardInterrupt):
                            raise
                        except Exception:
                            # Both raise = consistent rejection, not undefined
                            continue

                    # Compare outputs (NaN-aware: matching NaN positions = equal)
                    def _bitwise_equal(a, b):
                        """True if tensors are identical including NaN positions."""
                        if a.shape != b.shape or a.dtype != b.dtype:
                            return False
                        if a.is_floating_point() or a.is_complex():
                            # NaN == NaN should be True for this check
                            nan_match = torch.isnan(a) == torch.isnan(b)
                            if not nan_match.all():
                                return False
                            finite = ~torch.isnan(a)
                            if finite.any():
                                return torch.equal(a[finite], b[finite])
                            return True
                        return torch.equal(a, b)

                    try:
                        if isinstance(result1, torch.Tensor) and isinstance(result2, torch.Tensor):
                            if not _bitwise_equal(result1, result2):
                                undefined.add(op.name)
                                break
                        elif isinstance(result1, (tuple, list)):
                            for r1, r2 in zip(result1, result2):
                                if isinstance(r1, torch.Tensor) and isinstance(r2, torch.Tensor):
                                    if not _bitwise_equal(r1, r2):
                                        undefined.add(op.name)
                                        break
                    except (_OpTimeout, KeyboardInterrupt):
                        raise
                    except Exception:
                        undefined.add(op.name)
                        break

                    signal.alarm(0)  # cancel alarm after successful op
                except _OpTimeout:
                    # Op hung with NaN/Inf input — mark as undefined
                    undefined.add(op.name)
                    continue
                except Exception:
                    # Skip ops that crash during sample generation or have
                    # unsupported tensor types (e.g. sparse CSR)
                    signal.alarm(0)
                    continue
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    return undefined


def save_ieee754_undefined(undefined_ops):
    """Save discovered undefined ops to cache file."""
    version = torch.__version__
    path = _IEEE754_UNDEFINED_OPS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass

    data[version] = sorted(undefined_ops)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_error_op_tests(manifest):
    """Build error test list. Calls op.error_inputs at collection time (fast, no tensor ops)."""
    import torch.testing._internal.common_methods_invocations as cmi

    skip_ops = set(manifest.get("skip_ops", [])) | SKIP_OPS
    ops = []
    failures = []

    def _move_to_cpu(obj):
        if isinstance(obj, torch.Tensor):
            return obj.to("cpu")
        if isinstance(obj, list):
            return [_move_to_cpu(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(_move_to_cpu(item) for item in obj)
        if isinstance(obj, dict):
            return {key: _move_to_cpu(value) for key, value in obj.items()}
        return obj

    def _still_errors_on_cpu(op, err):
        sample = err.sample_input
        try:
            cpu_input = _move_to_cpu(sample.input)
            cpu_args = _move_to_cpu(sample.args)
            cpu_kwargs = _move_to_cpu(sample.kwargs)
        except Exception:
            return True
        try:
            op.op(cpu_input, *cpu_args, **cpu_kwargs)
        except Exception:
            return True
        return False

    seen = set()
    for op in cmi.op_db:
        if op.name in seen or op.name in skip_ops:
            continue
        seen.add(op.name)

        if not hasattr(op, "error_inputs_func") or op.error_inputs_func is None:
            continue
        try:
            errs = list(op.error_inputs("cpu"))
            if any(_still_errors_on_cpu(op, err) for err in errs):
                ops.append(op.name)
        except Exception as exc:
            failures.append(f"{op.name}: {type(exc).__name__}: {exc}")

    if failures:
        sample = "; ".join(failures[:5])
        extra = "" if len(failures) <= 5 else f"; ... {len(failures) - 5} more"
        raise RuntimeError(f"OpInfo error_inputs failed during collection: {sample}{extra}")

    return ops

# ---------------------------------------------------------------------------
# Backward dtype helper
# ---------------------------------------------------------------------------

def get_backward_dtypes(op_name):
    """Get backward-compatible dtypes from OpInfo metadata. No probing."""
    op = get_live_opinfo(op_name)
    if op is None:
        return []
    if not getattr(op, "supports_autograd", True):
        return []
    return [d for d in sorted(op.backward_dtypes, key=lambda x: str(x)) if d in DIFFERENTIABLE]

# ---------------------------------------------------------------------------
# Live OpInfo access
# ---------------------------------------------------------------------------

_live_opinfo_cache = {}

def get_live_opinfo(op_name):
    if op_name in _live_opinfo_cache:
        return _live_opinfo_cache[op_name]
    try:
        import torch.testing._internal.common_methods_invocations as cmi
        for op in cmi.op_db:
            if op.name == op_name:
                _live_opinfo_cache[op_name] = op
                return op
    except Exception as e:
        print(f"Error loading live generator for {op_name}: {e}", file=sys.stderr)
    _live_opinfo_cache[op_name] = None
    return None


def get_op_sample_inputs(op_name, device, dtype, requires_grad=False):
    op = get_live_opinfo(op_name)
    if op is None:
        return

    generation_device = "cpu" if device != "cpu" else device
    try:
        samples = op.sample_inputs(generation_device, dtype, requires_grad=requires_grad)
        for sample in samples:
            yield sample
    except Exception as exc:
        raise RuntimeError(
            f"sample_inputs failed for {op_name} on {generation_device} with {dtype}"
        ) from exc


def get_op_error_inputs(op_name, device):
    op = get_live_opinfo(op_name)
    if op is None:
        return

    try:
        # error_inputs accepts device as positional argument
        errors = op.error_inputs(device)
        for err in errors:
            yield err
    except Exception as exc:
        raise RuntimeError(f"error_inputs failed for {op_name} on {device}") from exc
