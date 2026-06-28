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

pytestmark = pytest.mark.covers_category("module_hooks")

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

class HookModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)
    def forward(self, x):
        return self.fc(x)

@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.requires("module_hooks")
@pytest.mark.parametrize("dtype", DTYPES)
def test_module_hooks(dtype, device, manifest):
    model = HookModel().to(device)
    if dtype != torch.float32:
        model = model.to(dtype)
    
    # 1. Forward and backward hook
    forward_called = False
    backward_called = False
    
    def fwd_hook(module, inputs, outputs):
        nonlocal forward_called
        forward_called = True
        
    def bwd_hook(module, grad_input, grad_output):
        nonlocal backward_called
        backward_called = True
        
    model.register_forward_hook(fwd_hook)
    model.register_full_backward_hook(bwd_hook)
    
    x = torch.randn(2, 4, dtype=dtype, device=device, requires_grad=True)
    out = model(x).sum()
    out.backward()
    synchronize(device)
    
    assert forward_called, "Forward hook not triggered."
    assert backward_called, "Backward hook not triggered."


@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.requires("gradient_checkpointing")
@pytest.mark.parametrize("dtype", DTYPES)
def test_gradient_checkpointing(dtype, device, manifest):
    model = HookModel().to(device)
    if dtype != torch.float32:
        model = model.to(dtype)
    x = torch.randn(2, 4, dtype=dtype, device=device, requires_grad=True)
    
    # Checkpoint forward
    out_cp = torch.utils.checkpoint.checkpoint(model, x, use_reentrant=False).sum()
    out_cp.backward()
    synchronize(device)
    
    assert x.grad is not None
