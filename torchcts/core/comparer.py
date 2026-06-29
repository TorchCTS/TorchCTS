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

import warnings

import torch
from torchcts.core.tolerances import get_tolerance

# Global dictionary to track metrics of the currently running test comparison
_ACTIVE_TEST_METRICS = {
    "max_abs_err": 0.0,
    "max_rel_err": 0.0,
    "cosim": 1.0,
    "passed": True,
    "error_msg": None,
    "golden_pass": True,
    "usable_pass": True,
    "quality_warning": None,
}

def clear_metrics():
    global _ACTIVE_TEST_METRICS
    _ACTIVE_TEST_METRICS = {
        "max_abs_err": 0.0,
        "max_rel_err": 0.0,
        "cosim": 1.0,
        "passed": True,
        "error_msg": None,
        "golden_pass": True,
        "usable_pass": True,
        "quality_warning": None,
    }

def get_metrics():
    return _ACTIVE_TEST_METRICS


def _record_quality_warning(message):
    current = _ACTIVE_TEST_METRICS.get("quality_warning")
    if current:
        _ACTIVE_TEST_METRICS["quality_warning"] = f"{current}\n{message}"
    else:
        _ACTIVE_TEST_METRICS["quality_warning"] = message


def _fail_compare(message):
    _ACTIVE_TEST_METRICS["passed"] = False
    _ACTIVE_TEST_METRICS["error_msg"] = message
    raise AssertionError(message)


def compute_cosim(a, b):
    # Flatten both
    a_flat = a.detach().cpu().to(torch.float32).flatten()
    b_flat = b.detach().cpu().to(torch.float32).flatten()
    if a_flat.numel() != b_flat.numel():
        return 0.0
    
    # Exclude NaNs and Infs for cosim to avoid nan cosim
    mask = torch.isfinite(a_flat) & torch.isfinite(b_flat)
    if not torch.any(mask):
        return 1.0
    
    a_flat = a_flat[mask]
    b_flat = b_flat[mask]
    
    norm_a = torch.linalg.vector_norm(a_flat)
    norm_b = torch.linalg.vector_norm(b_flat)
    if norm_a == 0.0 and norm_b == 0.0:
        return 1.0
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float((torch.dot(a_flat, b_flat) / (norm_a * norm_b)).item())

def prepare_for_compare(x):
    if x.is_conj():
        x = x.resolve_conj()
    if x.is_neg():
        x = x.resolve_neg()
    if x.dtype.is_complex:
        x = torch.view_as_real(x)
    return x


_HALF_FLOAT_DTYPES = (
    torch.float16, torch.bfloat16,
    torch.float8_e4m3fn, torch.float8_e5m2,
    torch.float8_e4m3fnuz, torch.float8_e5m2fnuz,
)

_UNSIGNED_INT_DTYPES = (torch.uint8, torch.uint16, torch.uint32, torch.uint64)


def _normalize_dense_compare_tensor(t, *, clone=False):
    t_cpu = t.detach().cpu()
    if clone:
        t_cpu = t_cpu.clone()
    t_cpu = prepare_for_compare(t_cpu)

    # Cast half-precision to float32 on CPU to avoid comparison artifacts
    # also float8 dtypes
    if t_cpu.dtype in _HALF_FLOAT_DTYPES:
        t_cpu = t_cpu.to(torch.float32)

    return t_cpu


def prepare_compare_tensor(t):
    return _normalize_dense_compare_tensor(t)


def _layout_name(layout):
    return str(layout)


def _coo_indices(t):
    return (("indices", t._indices()),)


def _csr_indices(t):
    return (
        ("crow_indices", t.crow_indices()),
        ("col_indices", t.col_indices()),
    )


def _csc_indices(t):
    return (
        ("ccol_indices", t.ccol_indices()),
        ("row_indices", t.row_indices()),
    )


def _bsr_indices(t):
    return _csr_indices(t)


def _bsc_indices(t):
    return _csc_indices(t)


def _coo_values(t):
    return t._values()


def _compressed_values(t):
    return t.values()


_SPARSE_LAYOUT_HANDLERS = {
    torch.sparse_coo: {
        "index_accessors": _coo_indices,
        "value_accessor": _coo_values,
        "metadata": ("sparse_dim", "dense_dim", "is_coalesced"),
        "block_values": False,
    },
    torch.sparse_csr: {
        "index_accessors": _csr_indices,
        "value_accessor": _compressed_values,
        "metadata": (),
        "block_values": False,
    },
    torch.sparse_csc: {
        "index_accessors": _csc_indices,
        "value_accessor": _compressed_values,
        "metadata": (),
        "block_values": False,
    },
    torch.sparse_bsr: {
        "index_accessors": _bsr_indices,
        "value_accessor": _compressed_values,
        "metadata": (),
        "block_values": True,
    },
    torch.sparse_bsc: {
        "index_accessors": _bsc_indices,
        "value_accessor": _compressed_values,
        "metadata": (),
        "block_values": True,
    },
}


def _is_known_sparse_layout(layout):
    return layout in _SPARSE_LAYOUT_HANDLERS


def _is_sparse_like_tensor(t):
    if not isinstance(t, torch.Tensor):
        return False
    layout = getattr(t, "layout", None)
    if _is_known_sparse_layout(layout):
        return True
    if bool(getattr(t, "is_sparse", False)):
        return True
    if bool(getattr(t, "is_sparse_csr", False)):
        return True
    return "sparse" in _layout_name(layout).lower()


def _sparse_container_for_compare(t):
    return t.detach().cpu()


def _assert_index_tensor_equal(actual, expected, layout, name):
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    if actual_cpu.shape != expected_cpu.shape:
        _fail_compare(
            f"Sparse structure mismatch for {_layout_name(layout)} {name}: "
            f"actual shape {tuple(actual_cpu.shape)} vs expected shape {tuple(expected_cpu.shape)}"
        )
    if actual_cpu.dtype != expected_cpu.dtype:
        _fail_compare(
            f"Sparse structure mismatch for {_layout_name(layout)} {name}: "
            f"actual dtype {actual_cpu.dtype} vs expected dtype {expected_cpu.dtype}"
        )
    if not torch.equal(actual_cpu, expected_cpu):
        _fail_compare(
            f"Sparse structure mismatch for {_layout_name(layout)} {name}: indices differ"
        )


def _assert_sparse_structure_matches(actual, expected):
    if actual.shape != expected.shape:
        _fail_compare(f"Shape mismatch: actual {actual.shape} vs expected {expected.shape}")
    if actual.layout != expected.layout:
        _fail_compare(
            f"Sparse layout mismatch: actual {_layout_name(actual.layout)} "
            f"vs expected {_layout_name(expected.layout)}"
        )
    if not _is_known_sparse_layout(actual.layout):
        _fail_compare(f"Unsupported sparse layout for structural comparison: {_layout_name(actual.layout)}")

    handler = _SPARSE_LAYOUT_HANDLERS[actual.layout]
    for metadata in handler["metadata"]:
        actual_value = getattr(actual, metadata)()
        expected_value = getattr(expected, metadata)()
        if actual_value != expected_value:
            _fail_compare(
                f"Sparse structure mismatch for {_layout_name(actual.layout)} {metadata}: "
                f"actual {actual_value} vs expected {expected_value}"
            )

    actual_values = handler["value_accessor"](actual)
    expected_values = handler["value_accessor"](expected)
    if handler["block_values"] and actual_values.shape[1:] != expected_values.shape[1:]:
        _fail_compare(
            f"Sparse block value shape mismatch for {_layout_name(actual.layout)}: "
            f"actual {tuple(actual_values.shape[1:])} vs expected {tuple(expected_values.shape[1:])}"
        )

    for (actual_name, actual_index), (expected_name, expected_index) in zip(
        handler["index_accessors"](actual),
        handler["index_accessors"](expected),
    ):
        if actual_name != expected_name:
            _fail_compare(
                f"Sparse comparer internal error for {_layout_name(actual.layout)}: "
                f"index accessor mismatch {actual_name} vs {expected_name}"
            )
        _assert_index_tensor_equal(actual_index, expected_index, actual.layout, actual_name)


def _sparse_values_for_compare(t):
    handler = _SPARSE_LAYOUT_HANDLERS[t.layout]
    values = handler["value_accessor"](t)
    return _normalize_dense_compare_tensor(values, clone=True)


def _compute_effective_dtype(actual, expected, dtype):
    effective_dtype = dtype
    if actual.is_floating_point() or actual.is_complex():
        act_size = actual.element_size()
        exp_size = expected.element_size()
        effective_dtype = actual.dtype if act_size <= exp_size else expected.dtype
    return effective_dtype


def _get_compare_tolerances(category, effective_dtype, manifest_overrides, scale_factor):
    golden_tol = get_tolerance(category, effective_dtype, tier="golden", manifest_overrides=manifest_overrides)
    usable_tol = get_tolerance(category, effective_dtype, tier="usable", manifest_overrides=manifest_overrides)
    if scale_factor != 1.0:
        golden_tol = golden_tol.scaled(scale_factor)
        usable_tol = usable_tol.scaled(scale_factor)
    return golden_tol, usable_tol


def _dense_diff(actual, expected):
    if actual.dtype == torch.bool:
        return (actual ^ expected).to(torch.float32)
    if actual.dtype in (torch.uint8, torch.int8, torch.uint16, torch.uint32, torch.uint64):
        # Cast to float64 to avoid unsigned overflow or NotImplementedError on CPU/MPS
        return torch.abs(actual.to(torch.float64) - expected.to(torch.float64))
    return torch.abs(actual - expected)


def _compute_dense_metrics(actual, expected):
    diff = _dense_diff(actual, expected)

    finite_mask = torch.isfinite(diff)
    if torch.any(finite_mask):
        max_abs = float(torch.max(diff[finite_mask]).item())
    else:
        max_abs = 0.0

    if expected.dtype in (torch.bool, *_UNSIGNED_INT_DTYPES):
        max_rel = 0.0
    else:
        denom = torch.abs(expected) + 1e-8
        rel_diff = diff / denom
        finite_rel_mask = torch.isfinite(rel_diff)
        if torch.any(finite_rel_mask):
            max_rel = float(torch.max(rel_diff[finite_rel_mask]).item())
        else:
            max_rel = 0.0

    return max_abs, max_rel, compute_cosim(actual, expected)


def _compare_dense_tensors(actual, expected, golden_tol, usable_tol, equal_nan=True):
    global _ACTIVE_TEST_METRICS

    if actual.shape != expected.shape:
        _fail_compare(
            f"Shape mismatch after comparison normalization: actual {actual.shape} vs expected {expected.shape}"
        )

    if actual.dtype != expected.dtype:
        expected = expected.to(actual.dtype)

    max_abs, max_rel, cosim = _compute_dense_metrics(actual, expected)
    _ACTIVE_TEST_METRICS["max_abs_err"] = max(_ACTIVE_TEST_METRICS["max_abs_err"], max_abs)
    _ACTIVE_TEST_METRICS["max_rel_err"] = max(_ACTIVE_TEST_METRICS["max_rel_err"], max_rel)
    _ACTIVE_TEST_METRICS["cosim"] = min(_ACTIVE_TEST_METRICS["cosim"], cosim)

    golden_passed = True
    golden_error = None
    try:
        torch.testing.assert_close(
            actual,
            expected,
            rtol=golden_tol.rtol,
            atol=golden_tol.atol,
            equal_nan=equal_nan,
        )
    except AssertionError as e:
        golden_passed = False
        golden_error = e

    if golden_passed:
        _ACTIVE_TEST_METRICS["golden_pass"] = _ACTIVE_TEST_METRICS["golden_pass"] and True
        _ACTIVE_TEST_METRICS["usable_pass"] = _ACTIVE_TEST_METRICS["usable_pass"] and True
        return

    usable_passed = True
    try:
        torch.testing.assert_close(
            actual,
            expected,
            rtol=usable_tol.rtol,
            atol=usable_tol.atol,
            equal_nan=equal_nan,
        )
    except AssertionError:
        usable_passed = False

    _ACTIVE_TEST_METRICS["golden_pass"] = _ACTIVE_TEST_METRICS["golden_pass"] and golden_passed
    _ACTIVE_TEST_METRICS["usable_pass"] = _ACTIVE_TEST_METRICS["usable_pass"] and usable_passed

    if usable_passed:
        warning_msg = (
            f"Quality warning: passed usable tolerance "
            f"(rtol={usable_tol.rtol}, atol={usable_tol.atol}) "
            f"but failed golden tolerance "
            f"(rtol={golden_tol.rtol}, atol={golden_tol.atol}): {golden_error}"
        )
        _record_quality_warning(warning_msg)
        return

    _ACTIVE_TEST_METRICS["passed"] = False
    _ACTIVE_TEST_METRICS["error_msg"] = str(golden_error)
    raise golden_error


def _unknown_sparse_warning(actual, expected):
    return (
        "TorchCTS does not know how to compare one or more sparse layouts structurally "
        f"(actual {_layout_name(getattr(actual, 'layout', None))}, "
        f"expected {_layout_name(getattr(expected, 'layout', None))}); "
        "falling back to dense comparison."
    )


def _densify_sparse_for_unknown_compare(t):
    if not _is_sparse_like_tensor(t):
        return _normalize_dense_compare_tensor(t)
    try:
        return _normalize_dense_compare_tensor(t.detach().cpu().to_dense())
    except Exception as exc:
        msg = (
            f"Could not densify sparse layout {_layout_name(getattr(t, 'layout', None))} "
            f"for fallback comparison: {exc}"
        )
        _ACTIVE_TEST_METRICS["passed"] = False
        _ACTIVE_TEST_METRICS["error_msg"] = msg
        raise AssertionError(msg) from exc


def _compare_unknown_sparse_as_dense(actual, expected, golden_tol, usable_tol, equal_nan=True):
    warning_msg = _unknown_sparse_warning(actual, expected)
    warnings.warn(warning_msg, stacklevel=2)
    _record_quality_warning(warning_msg)
    actual_dense = _densify_sparse_for_unknown_compare(actual)
    expected_dense = _densify_sparse_for_unknown_compare(expected)
    return _compare_dense_tensors(actual_dense, expected_dense, golden_tol, usable_tol, equal_nan=equal_nan)


def _describe_compare_value(value):
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"
    return type(value).__name__

def compare_tensors(actual, expected, category, dtype, manifest_overrides=None, equal_nan=True, scale_factor=1.0):
    __tracebackhide__ = True
    global _ACTIVE_TEST_METRICS

    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        _ACTIVE_TEST_METRICS["passed"] = False
        msg = (
            "Tensor comparison requires tensor values: "
            f"actual {_describe_compare_value(actual)} vs expected {_describe_compare_value(expected)}"
        )
        _ACTIVE_TEST_METRICS["error_msg"] = msg
        raise AssertionError(msg)

    effective_dtype = _compute_effective_dtype(actual, expected, dtype)
    golden_tol, usable_tol = _get_compare_tolerances(
        category,
        effective_dtype,
        manifest_overrides,
        scale_factor,
    )

    actual_sparse = _is_sparse_like_tensor(actual)
    expected_sparse = _is_sparse_like_tensor(expected)
    if actual_sparse or expected_sparse:
        actual_known = actual_sparse and _is_known_sparse_layout(actual.layout)
        expected_known = expected_sparse and _is_known_sparse_layout(expected.layout)
        if (actual_sparse and not actual_known) or (expected_sparse and not expected_known):
            return _compare_unknown_sparse_as_dense(
                actual,
                expected,
                golden_tol,
                usable_tol,
                equal_nan=equal_nan,
            )
        if actual_sparse != expected_sparse:
            _fail_compare(
                f"Sparse layout mismatch: actual {_layout_name(actual.layout)} "
                f"vs expected {_layout_name(expected.layout)}"
            )

        act_sparse = _sparse_container_for_compare(actual)
        exp_sparse = _sparse_container_for_compare(expected)
        _assert_sparse_structure_matches(act_sparse, exp_sparse)
        return _compare_dense_tensors(
            _sparse_values_for_compare(act_sparse),
            _sparse_values_for_compare(exp_sparse),
            golden_tol,
            usable_tol,
            equal_nan=equal_nan,
        )

    if actual.shape != expected.shape:
        _fail_compare(f"Shape mismatch: actual {actual.shape} vs expected {expected.shape}")

    return _compare_dense_tensors(
        prepare_compare_tensor(actual),
        prepare_compare_tensor(expected),
        golden_tol,
        usable_tol,
        equal_nan=equal_nan,
    )


def _propagation_values(actual, expected):
    actual_sparse = _is_sparse_like_tensor(actual)
    expected_sparse = _is_sparse_like_tensor(expected)
    if actual_sparse or expected_sparse:
        actual_known = actual_sparse and _is_known_sparse_layout(actual.layout)
        expected_known = expected_sparse and _is_known_sparse_layout(expected.layout)
        if (actual_sparse and not actual_known) or (expected_sparse and not expected_known):
            warning_msg = _unknown_sparse_warning(actual, expected)
            warnings.warn(warning_msg, stacklevel=3)
            _record_quality_warning(warning_msg)
            return (
                _densify_sparse_for_unknown_compare(actual),
                _densify_sparse_for_unknown_compare(expected),
            )
        if actual_sparse != expected_sparse:
            _fail_compare(
                f"Sparse layout mismatch: actual {_layout_name(actual.layout)} "
                f"vs expected {_layout_name(expected.layout)}"
            )
        act_sparse = _sparse_container_for_compare(actual)
        exp_sparse = _sparse_container_for_compare(expected)
        _assert_sparse_structure_matches(act_sparse, exp_sparse)
        return _sparse_values_for_compare(act_sparse), _sparse_values_for_compare(exp_sparse)

    act_cpu = prepare_compare_tensor(actual)
    exp_cpu = prepare_compare_tensor(expected)
    if act_cpu.shape != exp_cpu.shape:
        raise AssertionError(f"Shape mismatch: {act_cpu.shape} vs {exp_cpu.shape}")
    return act_cpu, exp_cpu

def compare_nan_propagation(actual, expected):
    """Check that NaN positions match between actual and expected."""
    __tracebackhide__ = True
    act_values, exp_values = _propagation_values(actual, expected)

    act_nan = torch.isnan(act_values)
    exp_nan = torch.isnan(exp_values)

    if not torch.equal(act_nan, exp_nan):
        mismatch = (act_nan != exp_nan)
        n_mismatch = mismatch.sum().item()
        device_extra_nan = (act_nan & ~exp_nan).sum().item()
        device_missing_nan = (~act_nan & exp_nan).sum().item()
        raise AssertionError(
            f"NaN propagation mismatch: {n_mismatch} positions differ. "
            f"Device has {device_extra_nan} extra NaNs, missing {device_missing_nan} NaNs."
        )

def compare_inf_propagation(actual, expected):
    """Check that Inf positions and signs match between actual and expected."""
    __tracebackhide__ = True
    act_values, exp_values = _propagation_values(actual, expected)

    act_inf = torch.isinf(act_values)
    exp_inf = torch.isinf(exp_values)

    if not torch.equal(act_inf, exp_inf):
        mismatch = (act_inf != exp_inf)
        raise AssertionError(
            f"Inf propagation mismatch: {mismatch.sum().item()} positions differ."
        )

    # Where both are inf, check sign matches
    both_inf = act_inf & exp_inf
    if both_inf.any():
        act_sign = torch.signbit(act_values[both_inf])
        exp_sign = torch.signbit(exp_values[both_inf])
        if not torch.equal(act_sign, exp_sign):
            raise AssertionError("Inf sign mismatch at some positions.")
