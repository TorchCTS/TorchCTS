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

@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_gradient_accumulation(dtype, device, manifest, compare):
    # Model and inputs
    weight_cpu = torch.randn(8, 8, dtype=dtype)
    x_cpu = torch.randn(4, 8, dtype=dtype) # 4 batches
    
    # 1. Accumulate over 4 micro-batches
    weight_acc = weight_cpu.clone().to(device)
    weight_acc.requires_grad = True
    
    for i in range(4):
        x_micro = x_cpu[i:i+1].to(device)
        out = torch.mm(x_micro, weight_acc).sum()
        out.backward()
        
    # 2. Run big-batch once
    weight_single = weight_cpu.clone().to(device)
    weight_single.requires_grad = True
    
    out_single = torch.mm(x_cpu.to(device), weight_single).sum()
    out_single.backward()
    
    synchronize(device)
    
    # G grads must match exactly
    compare(weight_acc.grad, weight_single.grad, category="exact", dtype=dtype)
