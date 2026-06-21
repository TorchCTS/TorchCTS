import pytest
import torch

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.stress
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scalar_val", [3.14, 2.71])
def test_zero_element_and_scalar_tensors(scalar_val, dtype, device):
    # 0-element tensor
    x = torch.empty(0, 10, dtype=dtype, device=device)
    assert x.numel() == 0
    y = x + 1.0
    assert y.numel() == 0
    
    # Scalar tensor (0D)
    s = torch.tensor(scalar_val, dtype=dtype, device=device)
    assert s.ndim == 0
    assert s.item() == pytest.approx(scalar_val, abs=1e-2)

@pytest.mark.stress
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scale", [1.0])
def test_large_allocations(scale, dtype, device, manifest):
    # Check limits
    limits = manifest.get("resource_limits", {})
    max_tensor = limits.get("max_tensor_size_mb")
    
    hw = manifest.get("hardware", {})
    dev_mem = hw.get("device_memory_gb", [2])[0]
    
    if max_tensor is not None and max_tensor < 2048:
        pytest.skip(f"Max tensor size is limited to {max_tensor}MB by resource limits.")
    if dev_mem < 6:
        pytest.skip(f"Device memory is too small ({dev_mem}GB) for >2GB tensor stress test.")
        
    # Allocate a 2.1 GB tensor (adjust element count based on dtype element size)
    target_bytes = 2.1 * (1024 ** 3)
    elem_size = torch.tensor([], dtype=dtype).element_size()
    num_elements = int(target_bytes / elem_size)
    try:
        x = torch.empty(num_elements, dtype=dtype, device=device)
        # Verify 64-bit indexing works in kernels
        x.fill_(1.5)
        assert float(x[0].item()) == pytest.approx(1.5, abs=1e-2)
        assert float(x[-1].item()) == pytest.approx(1.5, abs=1e-2)
        del x
    except (RuntimeError, MemoryError):
        pytest.skip("OOM during large tensor allocation.")
