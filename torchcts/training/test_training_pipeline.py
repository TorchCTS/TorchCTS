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

pytestmark = pytest.mark.covers_category("training_workflow")

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 4)
    def forward(self, x):
        return self.fc(x)

@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("optimizer_name", ["SGD", "AdamW", "Adagrad"])
def test_optimizer_pipelines(optimizer_name, dtype, device, manifest, compare):
    opt_cls = getattr(torch.optim, optimizer_name)
    
    # Check that model parameters update identically on device vs CPU
    # for SGD, AdamW, Adagrad
    model_cpu = LinearModel()
    model_dev = LinearModel().to(device)
    if dtype != torch.float32:
        model_cpu = model_cpu.to(dtype)
        model_dev = model_dev.to(dtype)
    
    # Clone weights to be identical
    with torch.no_grad():
        model_dev.fc.weight.copy_(model_cpu.fc.weight)
        model_dev.fc.bias.copy_(model_cpu.fc.bias)
        
    opt_cpu = opt_cls(model_cpu.parameters(), lr=0.1)
    opt_dev = opt_cls(model_dev.parameters(), lr=0.1)
    
    # Train step
    x_cpu = torch.randn(4, 8, dtype=dtype)
    x_dev = x_cpu.to(device)
    
    # CPU step
    opt_cpu.zero_grad()
    loss_cpu = model_cpu(x_cpu).sum()
    loss_cpu.backward()
    opt_cpu.step()
    
    # Device step
    opt_dev.zero_grad()
    loss_dev = model_dev(x_dev).sum()
    loss_dev.backward()
    opt_dev.step()
    
    synchronize(device)
    
    # Verify model parameters updated identically
    compare(model_dev.fc.weight, model_cpu.fc.weight, category="optimizer", dtype=dtype)
    compare(model_dev.fc.bias, model_cpu.fc.bias, category="optimizer", dtype=dtype)


@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.requires("fused_optimizer")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("optimizer_name", ["SGD", "AdamW", "Adagrad"])
def test_fused_optimizer_pipelines(optimizer_name, dtype, device, manifest, compare):
    opt_cls = getattr(torch.optim, optimizer_name)

    model_cpu = LinearModel()
    model_dev = LinearModel().to(device)
    if dtype != torch.float32:
        model_cpu = model_cpu.to(dtype)
        model_dev = model_dev.to(dtype)

    with torch.no_grad():
        model_dev.fc.weight.copy_(model_cpu.fc.weight)
        model_dev.fc.bias.copy_(model_cpu.fc.bias)

    opt_cpu = opt_cls(model_cpu.parameters(), lr=0.1)
    opt_dev = opt_cls(model_dev.parameters(), lr=0.1, fused=True)

    x_cpu = torch.randn(4, 8, dtype=dtype)
    x_dev = x_cpu.to(device)

    opt_cpu.zero_grad()
    loss_cpu = model_cpu(x_cpu).sum()
    loss_cpu.backward()
    opt_cpu.step()

    opt_dev.zero_grad()
    loss_dev = model_dev(x_dev).sum()
    loss_dev.backward()
    opt_dev.step()

    synchronize(device)

    compare(model_dev.fc.weight, model_cpu.fc.weight, category="optimizer", dtype=dtype)
    compare(model_dev.fc.bias, model_cpu.fc.bias, category="optimizer", dtype=dtype)
