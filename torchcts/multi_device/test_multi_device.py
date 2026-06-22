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

@pytest.mark.smoke
@pytest.mark.requires("multi_device")
@pytest.mark.parametrize("size", [10, 20])
def test_multi_device_placements_and_copies(size, device, manifest):
    # This test is gated by requiring device_count >= 2
    dev0 = f"{device}:0"
    dev1 = f"{device}:1"
    
    # 1. Allocation on specific device indices
    x = torch.randn(size, size, device=dev0)
    y = torch.randn(size, size, device=dev1)
    
    assert x.device.index == 0
    assert y.device.index == 1
    
    # 2. D2D Copy
    x_copied = x.to(dev1)
    synchronize(device)
    assert x_copied.device.index == 1
    assert torch.equal(x.cpu(), x_copied.cpu())
    
    # 3. Cross-device operation must raise RuntimeError
    with pytest.raises(RuntimeError):
        # mm on two different device indices
        torch.mm(x, y)

@pytest.mark.smoke
@pytest.mark.requires("multi_device")
@pytest.mark.parametrize("target_idx", [1])
def test_set_device_context(target_idx, device):
    # set_device support is verified at collection time via conftest.
    mod = torch.cuda if device == "cuda" else getattr(torch, device, None)
        
    # Get current
    orig_idx = mod.current_device() if hasattr(mod, "current_device") else 0
    
    try:
        mod.set_device(target_idx)
        # Allocation should default to index 1
        x = torch.randn(5, device=device)
        assert x.device.index == 1
    finally:
        # Restore original
        mod.set_device(orig_idx)
