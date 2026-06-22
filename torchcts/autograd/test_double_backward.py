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

DOUBLE_BACKWARD_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
ACTIVATIONS = ["sigmoid", "tanh", "gelu", "silu"]

@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.requires("double_backward")
@pytest.mark.parametrize("dtype", DOUBLE_BACKWARD_DTYPES)
@pytest.mark.parametrize("op_name", ACTIVATIONS)
def test_double_backward_ops(dtype, op_name, device, compare, input_gen):
    x_dev = input_gen((4, 4), dtype, device)
    x_dev.requires_grad = True
    
    x_cpu = x_dev.cpu().detach()
    x_cpu.requires_grad = True
    
    op_fn = getattr(torch, op_name, None) or getattr(torch.nn.functional, op_name)
    
    y_dev = op_fn(x_dev)
    grad_y_dev = torch.autograd.grad(y_dev.sum(), x_dev, create_graph=True)[0]
    loss_dev = grad_y_dev.pow(2).sum()
    loss_dev.backward()
    
    y_cpu = op_fn(x_cpu)
    grad_y_cpu = torch.autograd.grad(y_cpu.sum(), x_cpu, create_graph=True)[0]
    loss_cpu = grad_y_cpu.pow(2).sum()
    loss_cpu.backward()
    
    synchronize(device)
    compare(x_dev.grad, x_cpu.grad, category="backward", dtype=dtype)
