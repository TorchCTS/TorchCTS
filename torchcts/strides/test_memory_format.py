import pytest
import torch
from torchcts.core.device import synchronize

@pytest.mark.medium
@pytest.mark.parametrize("dtype", [torch.float32])
def test_memory_format_propagation(dtype, device, manifest):
    x = torch.randn(2, 3, 16, 16, dtype=dtype, device=device).to(memory_format=torch.channels_last)
    
    # Operations like activation functions should preserve memory format
    y = torch.nn.functional.relu(x)
    synchronize(device)
    
    assert y.is_contiguous(memory_format=torch.channels_last), "Memory format was not propagated (got contiguous instead)."
