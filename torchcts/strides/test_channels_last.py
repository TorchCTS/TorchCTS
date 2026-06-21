import pytest
import torch
from torchcts.core.device import synchronize

@pytest.mark.medium
@pytest.mark.requires("channels_last")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_channels_last_conv2d(dtype, device, manifest, compare, input_gen):
    # Create input in channels_last layout
    x_dev = input_gen((2, 3, 16, 16), dtype, device, layout="channels_last")
    w_dev = input_gen((4, 3, 3, 3), dtype, device)
    
    assert x_dev.is_contiguous(memory_format=torch.channels_last)
    
    # Conv2d output should also propagate/preserve channels_last layout if supported
    expected = torch.nn.functional.conv2d(x_dev.cpu(), w_dev.cpu(), padding=1)
    actual = torch.nn.functional.conv2d(x_dev, w_dev, padding=1)
    synchronize(device)
    
    compare(actual, expected, category="conv", dtype=dtype)
