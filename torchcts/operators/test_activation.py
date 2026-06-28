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

ACTIVATION_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.smoke
@pytest.mark.benchmarkable
@pytest.mark.covers("aten::gelu")
@pytest.mark.covers("aten::hardswish")
@pytest.mark.covers("aten::mish")
@pytest.mark.covers("aten::relu")
@pytest.mark.covers("aten::silu")
@pytest.mark.parametrize("op_name", ["relu", "gelu", "silu", "mish", "hardswish"])
@pytest.mark.parametrize("dtype", ACTIVATION_DTYPES)
def test_activations_basic(op_name, dtype, device, compare, input_gen):
    op_fn = getattr(torch, op_name, None)
    if op_fn is None:
        op_fn = getattr(torch.nn.functional, op_name, None)
        
    if op_fn is None:
        pytest.fail(f"Activation op {op_name} not found")

    shape = (32, 32)
    x_dev = input_gen(shape, dtype, device)
    x_cpu = x_dev.cpu()
    
    expected = op_fn(x_cpu)
    actual = op_fn(x_dev)
    synchronize(device)
    
    compare(actual, expected, category="elementwise", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.benchmarkable
@pytest.mark.covers("aten::_log_softmax")
@pytest.mark.covers("aten::_softmax")
@pytest.mark.parametrize("op_name", ["softmax", "log_softmax"])
@pytest.mark.parametrize("dtype", ACTIVATION_DTYPES)
def test_softmax_log_softmax(op_name, dtype, device, compare, input_gen):
    op_fn = getattr(torch.nn.functional, op_name)
    
    shape = (16, 32)
    x_dev = input_gen(shape, dtype, device)
    x_cpu = x_dev.cpu()
    
    expected = op_fn(x_cpu, dim=-1)
    actual = op_fn(x_dev, dim=-1)
    synchronize(device)
    
    compare(actual, expected, category="elementwise", dtype=dtype)
