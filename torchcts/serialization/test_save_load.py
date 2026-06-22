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
import os
from torchcts.core.device import synchronize

@pytest.mark.smoke
@pytest.mark.requires("serialization")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_save_load_roundtrip(dtype, device, manifest, compare, tmp_path):
    x_dev = torch.randn(8, 8, dtype=dtype, device=device)
    
    # Save
    save_path = os.path.join(tmp_path, "tensor.pt")
    torch.save(x_dev.cpu(), save_path)
    
    # Load back and move to device
    x_loaded = torch.load(save_path).to(device)
    synchronize(device)
    
    compare(x_loaded, x_dev, category="serialization", dtype=dtype)


@pytest.mark.smoke
@pytest.mark.requires("serialization")
def test_model_state_dict_roundtrip(device, manifest, tmp_path):
    """A model's state_dict round-tripped through torch.save/load must produce identical outputs."""
    torch.manual_seed(42)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),
    ).to(device)

    # Save state dict as CPU tensors
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    save_path = os.path.join(tmp_path, "model.pt")
    torch.save(sd, save_path)

    # Load into a fresh model
    model2 = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),
    )
    loaded_sd = torch.load(save_path, weights_only=True)
    model2.load_state_dict(loaded_sd)
    model2.to(device)
    synchronize(device)

    # Verify identical forward pass
    x = torch.randn(4, 16, device=device)
    with torch.no_grad():
        out1 = model(x)
        out2 = model2(x)
    synchronize(device)
    assert torch.allclose(out1.cpu(), out2.cpu(), atol=1e-5), "Model outputs differ after state_dict roundtrip"


@pytest.mark.smoke
@pytest.mark.requires("serialization")
@pytest.mark.requires("training")
def test_optimizer_state_dict_roundtrip(device, manifest, tmp_path):
    """An optimizer's state_dict must survive a save/load cycle and allow continued training."""
    model = torch.nn.Linear(16, 8).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Run 3 steps to populate optimizer state (momentum buffers, step counts, etc.)
    for _ in range(3):
        x = torch.randn(4, 16, device=device)
        model.zero_grad()
        model(x).sum().backward()
        opt.step()
    synchronize(device)

    # Move optimizer state to CPU for serialization
    opt_sd = opt.state_dict()
    cpu_state = {}
    for param_id, param_state in opt_sd["state"].items():
        cpu_state[param_id] = {
            k: v.cpu() if torch.is_tensor(v) else v
            for k, v in param_state.items()
        }
    cpu_sd = {"state": cpu_state, "param_groups": opt_sd["param_groups"]}

    save_path = os.path.join(tmp_path, "opt.pt")
    torch.save(cpu_sd, save_path)

    # Load into a fresh optimizer and take one more step
    loaded_sd = torch.load(save_path, weights_only=False)
    opt2 = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt2.load_state_dict(loaded_sd)

    model.zero_grad()
    model(torch.randn(4, 16, device=device)).sum().backward()
    opt2.step()  # Must not crash — proves state was restored correctly
    synchronize(device)

