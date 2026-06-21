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
