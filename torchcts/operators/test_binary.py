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

BINARY_FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
BINARY_INT_DTYPES = [torch.int64, torch.int32]

# Filter ops to only those that exist in torch at collection time
_BINARY_FLOAT_OPS = [op for op in ["add", "sub", "mul", "div", "pow", "fmod", "remainder", "atan2"] if hasattr(torch, op)]
_BINARY_INT_OPS = [op for op in ["add", "sub", "mul", "div", "fmod", "remainder"] if hasattr(torch, op)]

def make_binary_inputs(op_name, shape, dtype, device, input_gen):
    if op_name in ("div", "fmod", "remainder"):
        # Avoid zeros in the divisor
        a = input_gen(shape, dtype, device)
        b = input_gen(shape, dtype, device, positive_only=True) + 0.5
        return a, b
    elif op_name == "pow":
        # Keep base positive and exponent small
        a = input_gen(shape, dtype, device, positive_only=True)
        # Exponent can be integers or float
        b = input_gen(shape, dtype, device, positive_only=True) * 0.5
        return a, b
    elif op_name == "atan2":
        a = input_gen(shape, dtype, device)
        b = input_gen(shape, dtype, device)
        return a, b
    
    a = input_gen(shape, dtype, device)
    b = input_gen(shape, dtype, device)
    return a, b

@pytest.mark.smoke
@pytest.mark.benchmarkable
@pytest.mark.covers("aten::add.Tensor")
@pytest.mark.covers("aten::atan2")
@pytest.mark.covers("aten::div.Tensor")
@pytest.mark.covers("aten::fmod.Tensor")
@pytest.mark.covers("aten::mul.Tensor")
@pytest.mark.covers("aten::pow.Tensor_Tensor")
@pytest.mark.covers("aten::remainder.Tensor")
@pytest.mark.covers("aten::sub.Tensor")
@pytest.mark.parametrize("op_name", _BINARY_FLOAT_OPS)
@pytest.mark.parametrize("dtype", BINARY_FLOAT_DTYPES)
def test_binary_float_op(op_name, dtype, device, manifest, compare, input_gen):
    op_fn = getattr(torch, op_name)

    shape = (32, 32)
    a_dev, b_dev = make_binary_inputs(op_name, shape, dtype, device, input_gen)
    a_cpu, b_cpu = a_dev.cpu(), b_dev.cpu()

    try:
        expected = op_fn(a_cpu, b_cpu)
        actual = op_fn(a_dev, b_dev)
        synchronize(device)
    except Exception as e:
        raise RuntimeError(f"Binary op '{op_name}' failed on {device}: {e}") from e

    compare(actual, expected, category="elementwise", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.benchmarkable
@pytest.mark.covers("aten::add.Tensor")
@pytest.mark.covers("aten::div.Tensor")
@pytest.mark.covers("aten::fmod.Tensor")
@pytest.mark.covers("aten::mul.Tensor")
@pytest.mark.covers("aten::remainder.Tensor")
@pytest.mark.covers("aten::sub.Tensor")
@pytest.mark.parametrize("op_name", _BINARY_INT_OPS)
@pytest.mark.parametrize("dtype", BINARY_INT_DTYPES)
def test_binary_int_op(op_name, dtype, device, manifest, compare, input_gen):
    op_fn = getattr(torch, op_name)

    shape = (32, 32)
    a_dev, b_dev = make_binary_inputs(op_name, shape, dtype, device, input_gen)
    a_cpu, b_cpu = a_dev.cpu(), b_dev.cpu()

    try:
        expected = op_fn(a_cpu, b_cpu)
        actual = op_fn(a_dev, b_dev)
        synchronize(device)
    except Exception as e:
        raise RuntimeError(f"Binary integer op '{op_name}' failed on {device}: {e}") from e

    compare(actual, expected, category="exact", dtype=actual.dtype)
