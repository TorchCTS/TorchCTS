import pytest
import torch
from torchcts.core.device import synchronize

PADDING_DTYPES = [torch.float32, torch.int64]

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", PADDING_DTYPES)
def test_constant_pad(dtype, device, compare, input_gen):
    x_dev = input_gen((4, 4), dtype, device)
    pad = (1, 2, 1, 2) # left, right, top, bottom
    
    expected = torch.nn.functional.pad(x_dev.cpu(), pad, mode="constant", value=42)
    actual = torch.nn.functional.pad(x_dev, pad, mode="constant", value=42)
    synchronize(device)
    
    compare(actual, expected, category="exact", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("mode", ["reflect", "replicate", "circular"])
def test_other_padding_modes(dtype, mode, device, compare, input_gen):
    x_dev = input_gen((1, 1, 8, 8), dtype, device)
    pad = (1, 1, 1, 1)
    
    expected = torch.nn.functional.pad(x_dev.cpu(), pad, mode=mode)
    actual = torch.nn.functional.pad(x_dev, pad, mode=mode)
    synchronize(device)
    compare(actual, expected, category="exact", dtype=dtype)
