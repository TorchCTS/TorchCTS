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

pytestmark = pytest.mark.covers_category("workload")

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

class LoRALinear(torch.nn.Module):
    def __init__(self, in_features, out_features, r=4, lora_alpha=8):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)
        self.lora_A = torch.nn.Parameter(torch.randn(in_features, r))
        self.lora_B = torch.nn.Parameter(torch.randn(r, out_features))
        self.scaling = lora_alpha / r
        
    def forward(self, x):
        base = self.linear(x)
        adapter = (x @ self.lora_A) @ self.lora_B * self.scaling
        return base + adapter

@pytest.mark.workload
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("r", [4, 8])
def test_lora_forward_backward(r, dtype, device, manifest, compare):

    model_cpu = LoRALinear(16, 8, r=r)
    model_dev = LoRALinear(16, 8, r=r).to(device)
    
    with torch.no_grad():
        model_dev.linear.weight.copy_(model_cpu.linear.weight)
        model_dev.lora_A.copy_(model_cpu.lora_A)
        model_dev.lora_B.copy_(model_cpu.lora_B)

    if dtype != torch.float32:
        model_cpu = model_cpu.to(dtype)
        model_dev = model_dev.to(dtype)
        
    x_cpu = torch.randn(4, 16, dtype=dtype, requires_grad=True)
    x_dev = x_cpu.clone().detach().to(device)
    x_dev.requires_grad = True
    
    # Forward
    out_cpu = model_cpu(x_cpu).sum()
    out_dev = model_dev(x_dev).sum()
    
    # Backward
    out_cpu.backward()
    out_dev.backward()
    
    synchronize(device)
    
    compare(model_dev.lora_A.grad, model_cpu.lora_A.grad, category="workload_e2e", dtype=dtype)
    compare(model_dev.lora_B.grad, model_cpu.lora_B.grad, category="workload_e2e", dtype=dtype)
    compare(x_dev.grad, x_cpu.grad, category="workload_e2e", dtype=dtype)
