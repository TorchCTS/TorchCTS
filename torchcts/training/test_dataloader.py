import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.medium
@pytest.mark.requires("dataloader")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch_size", [10, 20])
def test_dataloader_pin_memory(batch_size, dtype, device, manifest):
    # Dataloader with pin_memory=True
    x_cpu = torch.randn(100, 10, dtype=dtype)
    y_cpu = torch.randn(100, 1, dtype=dtype)
    
    dataset = TensorDataset(x_cpu, y_cpu)
    
    # We use pin_memory only if device supports it (e.g. cuda, mps)
    # Gated by pinned_memory capability
    caps = manifest.get("capabilities", {})
    pin = caps.get("pinned_memory", False)
    if not pin:
        pytest.skip("Pinned memory support is not declared for this backend.")
    
    loader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin, num_workers=0)
    
    # Iterate and copy to device
    for batch_x, batch_y in loader:
        assert batch_x.is_pinned(), "pin_memory=True did not produce a pinned input batch."
        assert batch_y.is_pinned(), "pin_memory=True did not produce a pinned target batch."
        bx_dev = batch_x.to(device)
        by_dev = batch_y.to(device)
        assert bx_dev.device.type == device
        assert by_dev.device.type == device
        break
