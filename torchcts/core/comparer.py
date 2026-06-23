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

import torch
import math
from torchcts.core.tolerances import get_tolerance, Tol, TieredTol

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

def compute_cosim(a, b):
    # Flatten both
    a_flat = a.detach().cpu().to(torch.float32).flatten()
    b_flat = b.detach().cpu().to(torch.float32).flatten()
    
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

def prepare_compare_tensor(t):
    t_cpu = t.detach().cpu()
    t_cpu = prepare_for_compare(t_cpu)
    
    # Cast half-precision to float32 on CPU to avoid comparison artifacts
    # also float8 dtypes
    half_float_dtypes = (
        torch.float16, torch.bfloat16,
        torch.float8_e4m3fn, torch.float8_e5m2,
        torch.float8_e4m3fnuz, torch.float8_e5m2fnuz
    )
    if t_cpu.dtype in half_float_dtypes:
        t_cpu = t_cpu.to(torch.float32)
        
    return t_cpu

def compare_tensors(actual, expected, category, dtype, manifest_overrides=None, equal_nan=True, scale_factor=1.0):
    __tracebackhide__ = True
    global _ACTIVE_TEST_METRICS
    
    # Bypass comparison on CPU targets (validation mode)
    if actual.device.type == "cpu" and expected.device.type == "cpu":
        return
    
    # 1. Resolve effective dtype for tolerance lookup.
    # If the output tensor is floating-point or complex, use the lower-precision of the actual/expected dtypes.
    # This ensures that type promotion (e.g. acosh(int64) -> float64/float32) resolves to a floating-point tolerance.
    effective_dtype = dtype
    if actual.is_floating_point() or actual.is_complex():
        act_size = actual.element_size()
        exp_size = expected.element_size()
        effective_dtype = actual.dtype if act_size <= exp_size else expected.dtype

    golden_tol = get_tolerance(category, effective_dtype, tier="golden", manifest_overrides=manifest_overrides)
    usable_tol = get_tolerance(category, effective_dtype, tier="usable", manifest_overrides=manifest_overrides)
    if scale_factor != 1.0:
        golden_tol = golden_tol.scaled(scale_factor)
        usable_tol = usable_tol.scaled(scale_factor)
        
    # 2. Check shapes
    if actual.shape != expected.shape:
        _ACTIVE_TEST_METRICS["passed"] = False
        msg = f"Shape mismatch: actual {actual.shape} vs expected {expected.shape}"
        _ACTIVE_TEST_METRICS["error_msg"] = msg
        raise AssertionError(msg)

    # 3. Cast to CPU and prepare
    act_cpu = prepare_compare_tensor(actual)
    exp_cpu = prepare_compare_tensor(expected)
    
    # Ensure they are the same type for comparison (e.g. bool, int, float)
    if act_cpu.dtype != exp_cpu.dtype:
        exp_cpu = exp_cpu.to(act_cpu.dtype)

    # 4. Compute metrics
    # Max Absolute Error: max(|A - B|)
    # Handle NaNs / Infs by replacing them or ignoring them for max error calculations
    if act_cpu.dtype == torch.bool:
        diff = (act_cpu ^ exp_cpu).to(torch.float32)
    elif act_cpu.dtype in (torch.uint8, torch.int8, torch.uint16, torch.uint32, torch.uint64):
        # Cast to float64 to avoid unsigned overflow or NotImplementedError on CPU/MPS
        diff = torch.abs(act_cpu.to(torch.float64) - exp_cpu.to(torch.float64))
    else:
        diff = torch.abs(act_cpu - exp_cpu)
    
    # Filter finite values for max_abs_err
    finite_mask = torch.isfinite(diff)
    if torch.any(finite_mask):
        max_abs = float(torch.max(diff[finite_mask]).item())
    else:
        # If no finite values, check if they are identical (e.g., both NaN or both Inf)
        # We can set max_abs to 0.0 if all values are equal_nan/equal_inf
        max_abs = 0.0
        
    # Max Relative Error: max(|A - B| / (|B| + epsilon))
    if exp_cpu.dtype in (torch.bool, torch.uint8, torch.uint16, torch.uint32, torch.uint64):
        max_rel = 0.0
    else:
        denom = torch.abs(exp_cpu) + 1e-8
        rel_diff = diff / denom
        finite_rel_mask = torch.isfinite(rel_diff)
        if torch.any(finite_rel_mask):
            max_rel = float(torch.max(rel_diff[finite_rel_mask]).item())
        else:
            max_rel = 0.0

    cosim = compute_cosim(act_cpu, exp_cpu)

    # Update active test metrics
    _ACTIVE_TEST_METRICS["max_abs_err"] = max(_ACTIVE_TEST_METRICS["max_abs_err"], max_abs)
    _ACTIVE_TEST_METRICS["max_rel_err"] = max(_ACTIVE_TEST_METRICS["max_rel_err"], max_rel)
    _ACTIVE_TEST_METRICS["cosim"] = min(_ACTIVE_TEST_METRICS["cosim"], cosim)

    # 5. Two-tier assertion check (golden then usable)
    golden_passed = True
    golden_error = None
    try:
        torch.testing.assert_close(
            act_cpu,
            exp_cpu,
            rtol=golden_tol.rtol,
            atol=golden_tol.atol,
            equal_nan=equal_nan
        )
    except AssertionError as e:
        golden_passed = False
        golden_error = e

    if golden_passed:
        # Both tiers pass
        _ACTIVE_TEST_METRICS["golden_pass"] = _ACTIVE_TEST_METRICS["golden_pass"] and True
        _ACTIVE_TEST_METRICS["usable_pass"] = _ACTIVE_TEST_METRICS["usable_pass"] and True
        return

    # Golden failed — try usable tier
    usable_passed = True
    try:
        torch.testing.assert_close(
            act_cpu,
            exp_cpu,
            rtol=usable_tol.rtol,
            atol=usable_tol.atol,
            equal_nan=equal_nan
        )
    except AssertionError:
        usable_passed = False

    _ACTIVE_TEST_METRICS["golden_pass"] = _ACTIVE_TEST_METRICS["golden_pass"] and golden_passed
    _ACTIVE_TEST_METRICS["usable_pass"] = _ACTIVE_TEST_METRICS["usable_pass"] and usable_passed

    if usable_passed:
        # Usable passed but golden failed — record warning, don't raise
        warning_msg = (
            f"Quality warning: passed usable tolerance "
            f"(rtol={usable_tol.rtol}, atol={usable_tol.atol}) "
            f"but failed golden tolerance "
            f"(rtol={golden_tol.rtol}, atol={golden_tol.atol}): {golden_error}"
        )
        _ACTIVE_TEST_METRICS["quality_warning"] = warning_msg
        return

    # Both tiers failed — raise the golden error
    _ACTIVE_TEST_METRICS["passed"] = False
    _ACTIVE_TEST_METRICS["error_msg"] = str(golden_error)
    raise golden_error

def compare_nan_propagation(actual, expected):
    """Check that NaN positions match between actual and expected."""
    __tracebackhide__ = True
    act_cpu = prepare_compare_tensor(actual)
    exp_cpu = prepare_compare_tensor(expected)
    
    if act_cpu.shape != exp_cpu.shape:
        raise AssertionError(f"Shape mismatch: {act_cpu.shape} vs {exp_cpu.shape}")
    
    act_nan = torch.isnan(act_cpu)
    exp_nan = torch.isnan(exp_cpu)
    
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
    act_cpu = prepare_compare_tensor(actual)
    exp_cpu = prepare_compare_tensor(expected)
    
    if act_cpu.shape != exp_cpu.shape:
        raise AssertionError(f"Shape mismatch: {act_cpu.shape} vs {exp_cpu.shape}")
    
    act_inf = torch.isinf(act_cpu)
    exp_inf = torch.isinf(exp_cpu)
    
    if not torch.equal(act_inf, exp_inf):
        mismatch = (act_inf != exp_inf)
        raise AssertionError(
            f"Inf propagation mismatch: {mismatch.sum().item()} positions differ."
        )
    
    # Where both are inf, check sign matches
    both_inf = act_inf & exp_inf
    if both_inf.any():
        act_sign = torch.sign(act_cpu[both_inf])
        exp_sign = torch.sign(exp_cpu[both_inf])
        if not torch.equal(act_sign, exp_sign):
            raise AssertionError("Inf sign mismatch at some positions.")
