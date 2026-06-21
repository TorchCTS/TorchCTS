import pytest
import torch
from torchcts.core.device import get_device_module

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.stress
@pytest.mark.requires("guard_alloc")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [
    (100, 100),
    (50, 50)
])
def test_guard_alloc_canary(shape, dtype, device):
    mod = get_device_module(device)
    if mod is None or not hasattr(mod, "guard_allocator_enabled"):
        pytest.skip("Guard allocator validation requires a backend-specific verifier hook.")

    assert mod.guard_allocator_enabled() is True
    x = torch.randn(*shape, dtype=dtype, device=device)
    y = x + 1.0
    assert y.shape == shape
