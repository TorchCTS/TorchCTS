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

BACKWARD_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
ACTIVATIONS = ["relu", "gelu", "sigmoid", "tanh", "silu"]

@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
@pytest.mark.parametrize("op_name", ACTIVATIONS)
def test_first_order_backward(dtype, op_name, device, compare, input_gen):
    # Test simple model backpropagation (Linear + Activation)
    x_dev = input_gen((4, 8), dtype, device)
    x_dev.requires_grad = True
    
    w_cpu = torch.randn(8, 4, dtype=dtype)
    w_dev = w_cpu.to(device)
    w_dev.requires_grad = True
    
    x_cpu = x_dev.cpu().detach()
    x_cpu.requires_grad = True
    
    w_ref = w_cpu.clone().detach()
    w_ref.requires_grad = True
    
    op_fn = getattr(torch.nn.functional, op_name)
    
    out_dev = op_fn(torch.mm(x_dev, w_dev)).sum()
    out_cpu = op_fn(torch.mm(x_cpu, w_ref)).sum()
    
    out_dev.backward()
    out_cpu.backward()
    synchronize(device)
    
    compare(x_dev.grad, x_cpu.grad, category="matmul_backward", dtype=dtype)
    compare(w_dev.grad, w_ref.grad, category="matmul_backward", dtype=dtype)
