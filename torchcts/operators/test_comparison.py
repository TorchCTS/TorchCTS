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
from torchcts.core.device import synchronize

COMP_DTYPES = [torch.float32, torch.int64, torch.bool]

# Construct valid comparison test cases to avoid invalid bool parameterizations
# Also filter out ops that don't exist in torch at collection time
COMP_TESTS = []
for op in ["eq", "ne", "lt", "le", "gt", "ge"]:
    if not hasattr(torch, op):
        continue
    for dt in COMP_DTYPES:
        if dt == torch.bool and op in ("lt", "le", "gt", "ge"):
            continue
        COMP_TESTS.append((op, dt))

# Filter nan/inf/finite ops to those that exist
_NAN_INF_OPS = [op for op in ["isnan", "isinf", "isfinite"] if hasattr(torch, op)]

@pytest.mark.smoke
@pytest.mark.covers("aten::eq.Tensor")
@pytest.mark.covers("aten::ge.Tensor")
@pytest.mark.covers("aten::gt.Tensor")
@pytest.mark.covers("aten::le.Tensor")
@pytest.mark.covers("aten::lt.Tensor")
@pytest.mark.covers("aten::ne.Tensor")
@pytest.mark.parametrize("op_name, dtype", COMP_TESTS)
def test_comparison_op(op_name, dtype, device, manifest, compare, input_gen):
    op_fn = getattr(torch, op_name)

    shape = (32, 32)
    a_dev = input_gen(shape, dtype, device)
    b_dev = input_gen(shape, dtype, device)
    
    a_cpu, b_cpu = a_dev.cpu(), b_dev.cpu()
    
    try:
        expected = op_fn(a_cpu, b_cpu)
        actual = op_fn(a_dev, b_dev)
        synchronize(device)
    except Exception as e:
        raise RuntimeError(f"Comparison op '{op_name}' failed on {device}: {e}") from e
        
    compare(actual, expected, category="exact", dtype=torch.bool)

@pytest.mark.smoke
@pytest.mark.covers("aten::isfinite")
@pytest.mark.covers("aten::isinf")
@pytest.mark.covers("aten::isnan")
@pytest.mark.parametrize("op_name", _NAN_INF_OPS)
def test_nan_inf_finite(op_name, device, manifest, compare):
    dtype = torch.float32
    shape = (10, 10)
    
    x_cpu = torch.randn(shape, dtype=dtype)
    x_cpu[0, 0] = float("nan")
    x_cpu[1, 1] = float("inf")
    x_cpu[2, 2] = float("-inf")
    
    x_dev = x_cpu.to(device)
    
    op_fn = getattr(torch, op_name)
        
    expected = op_fn(x_cpu)
    actual = op_fn(x_dev)
    synchronize(device)
    
    compare(actual, expected, category="exact", dtype=torch.bool)

@pytest.mark.smoke
@pytest.mark.covers("aten::where.self")
@pytest.mark.parametrize("dtype", COMP_DTYPES)
def test_where(dtype, device, manifest, compare, input_gen):
    shape = (32, 32)
    cond_dev = input_gen(shape, torch.bool, device)
    x_dev = input_gen(shape, dtype, device)
    y_dev = input_gen(shape, dtype, device)
    
    cond_cpu, x_cpu, y_cpu = cond_dev.cpu(), x_dev.cpu(), y_dev.cpu()
    
    expected = torch.where(cond_cpu, x_cpu, y_cpu)
    actual = torch.where(cond_dev, x_dev, y_dev)
    synchronize(device)
    
    compare(actual, expected, category="exact", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.covers("aten::clamp")
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_clamp(dtype, device, manifest, compare, input_gen):
    shape = (32, 32)
    x_dev = input_gen(shape, dtype, device)
    x_cpu = x_dev.cpu()
    
    min_val = -0.5 if dtype.is_floating_point else -5
    max_val = 0.5 if dtype.is_floating_point else 5
    
    expected = torch.clamp(x_cpu, min_val, max_val)
    actual = torch.clamp(x_dev, min_val, max_val)
    synchronize(device)
    
    compare(actual, expected, category="exact" if dtype == torch.int64 else "elementwise", dtype=dtype)
