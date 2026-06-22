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

import pytest
import torch
import torchcts.conftest as conftest
from torchcts.core.opinfo_adapter import (
    get_forward_op_tests,
    get_live_opinfo,
    get_op_sample_inputs,
    str_to_dtype,
    record_known_failure,
    is_cpu_reference_failure,
)
from torchcts.core.device import synchronize

# Determine tolerance category based on op name
def get_op_category(op_name):
    op_name = op_name.lower()
    if any(k in op_name for k in ("_mm", "matmul", "dot", "mv", "bmm", "addmm")):
        return "matmul"
    elif any(k in op_name for k in ("sum", "mean", "std", "var", "norm", "amax", "amin", "prod")):
        return "reduction"
    elif "conv" in op_name:
        return "conv"
    elif any(k in op_name for k in ("norm", "group", "batch")):
        return "norm"
    elif "linalg" in op_name:
        return "linalg"
    elif "fft" in op_name:
        return "fft"
    elif any(k in op_name for k in ("copy", "clone", "to_copy")):
        return "copy"
    return "elementwise"

# Build test list from op_db metadata + known failures (no probing)
try:
    op_tests = get_forward_op_tests(conftest._MANIFEST)
except Exception:
    op_tests = []

if not op_tests:
    # Dummy parameter to avoid pytest collection errors
    op_tests = [("dummy", "dummy")]

# Ops whose outputs are inherently nondeterministic — value comparison is invalid
_NONDETERMINISTIC_OPS = frozenset({
    # Uninitialized memory
    "empty", "empty_like", "empty_permuted", "empty_strided",
    "new_empty", "new_empty_strided",
    # Random sampling
    "bernoulli", "geometric", "multinomial",
    "rand_like", "randint", "randint_like", "randn", "randn_like",
    "nn.functional.dropout", "nn.functional.dropout2d", "nn.functional.dropout3d",
    "nn.functional.alpha_dropout", "nn.functional.feature_alpha_dropout",
    "nn.functional.fractional_max_pool2d", "nn.functional.fractional_max_pool3d",
    # Decompositions with inherent sign/order ambiguity
    "svd_lowrank", "pca_lowrank",
})

_UNINITIALIZED_OPS = frozenset({
    "empty", "empty_like", "empty_permuted", "empty_strided",
    "new_empty", "new_empty_strided",
})

@pytest.mark.opinfo
@pytest.mark.parametrize("op_name, dtype_str", op_tests)
def test_op_forward(op_name, dtype_str, device, compare):
    if op_name == "dummy":
        pytest.skip("No OpInfo tests matched the manifest filters.")
        
    dtype = str_to_dtype(dtype_str)

    # Resolve samples
    samples = list(get_op_sample_inputs(op_name, device, dtype))
    assert samples, f"No sample inputs generated for {op_name} with {dtype_str}"

    # Determine category
    category = get_op_category(op_name)
    
    tested_any = False
    for sample in samples:
        # Clone sample input to CPU
        cpu_input = sample.input.cpu() if isinstance(sample.input, torch.Tensor) else sample.input
        cpu_args = [a.cpu() if isinstance(a, torch.Tensor) else a for a in sample.args]
        cpu_kwargs = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in sample.kwargs.items()}
        
        # Get live generator function from OpInfo (filtered at collection time, must exist)
        op_info = get_live_opinfo(op_name)
        assert op_info is not None, f"Live OpInfo for {op_name} disappeared after collection"
        op_fn = op_info.op

        if device == "cpu":
            # CPU validation mode: execute target op once, skip reference run and comparison
            try:
                dev_input = sample.input.to(device) if isinstance(sample.input, torch.Tensor) else sample.input
                dev_args = [a.to(device) if isinstance(a, torch.Tensor) else a for a in sample.args]
                dev_kwargs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sample.kwargs.items()}
                
                actual = op_fn(dev_input, *dev_args, **dev_kwargs)
                synchronize(device)
            except Exception:
                continue
            tested_any = True
            continue

        # Run reference CPU op
        try:
            expected = op_fn(cpu_input, *cpu_args, **cpu_kwargs)
        except Exception as e:
            if is_cpu_reference_failure(e):
                record_known_failure("forward", op_name, dtype_str, f"{type(e).__name__}: {e}")
                continue
            # Other CPU failures (e.g. shape mismatch in sample params) — skip sample
            continue

        # Run target device op
        try:
            # Ensure inputs are on target device
            dev_input = sample.input.to(device) if isinstance(sample.input, torch.Tensor) else sample.input
            dev_args = [a.to(device) if isinstance(a, torch.Tensor) else a for a in sample.args]
            dev_kwargs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sample.kwargs.items()}
            
            actual = op_fn(dev_input, *dev_args, **dev_kwargs)
            synchronize(device)
        except Exception as e:
            # Target device execution failed -> raise ERROR/FAIL
            raise RuntimeError(f"Execution failed on device {device}: {e}") from e

        # Compare outputs recursively
        def compare_recursive(act, exp):
            __tracebackhide__ = True
            if isinstance(act, torch.Tensor) and isinstance(exp, torch.Tensor):
                # Ensure actual result is copied back for validation
                compare(act, exp, category=category, dtype=dtype)
            elif isinstance(act, (list, tuple)) and isinstance(exp, (list, tuple)):
                assert len(act) == len(exp), f"Output sequence lengths differ: got {len(act)}, expected {len(exp)}"
                for a, e in zip(act, exp):
                    compare_recursive(a, e)
            elif isinstance(act, dict) and isinstance(exp, dict):
                assert len(act) == len(exp), f"Output dict sizes differ: got {len(act)}, expected {len(exp)}"
                for k in act:
                    assert k in exp, f"Key {k} not in CPU reference output keys"
                    compare_recursive(act[k], exp[k])

        # Structural comparison for nondeterministic/random/uninitialized ops
        def compare_nondeterministic(act, exp):
            __tracebackhide__ = True
            if isinstance(act, torch.Tensor) and isinstance(exp, torch.Tensor):
                assert act.shape == exp.shape, f"Shape mismatch: got {act.shape}, expected {exp.shape}"
                assert act.dtype == exp.dtype, f"Dtype mismatch: got {act.dtype}, expected {exp.dtype}"
                if op_name not in _UNINITIALIZED_OPS:
                    if act.is_floating_point() or act.is_complex():
                        if torch.isfinite(exp).all():
                            assert torch.isfinite(act).all(), f"Output tensor contains non-finite values (NaN/Inf) but CPU reference was finite."
            elif isinstance(act, (list, tuple)) and isinstance(exp, (list, tuple)):
                assert len(act) == len(exp), f"Output sequence lengths differ: got {len(act)}, expected {len(exp)}"
                for a, e in zip(act, exp):
                    compare_nondeterministic(a, e)
            elif isinstance(act, dict) and isinstance(exp, dict):
                assert len(act) == len(exp), f"Output dict sizes differ: got {len(act)}, expected {len(exp)}"
                for k in act:
                    assert k in exp, f"Key {k} not in CPU reference output keys"
                    compare_nondeterministic(act[k], exp[k])

        if op_name in _NONDETERMINISTIC_OPS:
            compare_nondeterministic(actual, expected)
        else:
            compare_recursive(actual, expected)
        tested_any = True

    if not tested_any:
        pytest.skip(f"All sample inputs for {op_name} were skipped or failed on CPU reference")
