import pytest
import torch
from torchcts.core.device import synchronize

UPSAMPLE_DTYPES = [torch.float32]

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", UPSAMPLE_DTYPES)
@pytest.mark.parametrize("mode", ["nearest", "bilinear", "bicubic"])
def test_upsample_2d(dtype, mode, device, compare, input_gen):
    x_dev = input_gen((2, 3, 8, 8), dtype, device)
    
    expected = torch.nn.functional.interpolate(x_dev.cpu(), size=(16, 16), mode=mode)
    actual = torch.nn.functional.interpolate(x_dev, size=(16, 16), mode=mode)
    synchronize(device)
    
    compare(actual, expected, category="elementwise", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", UPSAMPLE_DTYPES)
@pytest.mark.parametrize("mode", ["trilinear"])
def test_upsample_3d(dtype, mode, device, compare, input_gen):
    x_dev = input_gen((1, 2, 4, 4, 4), dtype, device)
    
    expected = torch.nn.functional.interpolate(x_dev.cpu(), size=(8, 8, 8), mode=mode, align_corners=False)
    actual = torch.nn.functional.interpolate(x_dev, size=(8, 8, 8), mode=mode, align_corners=False)
    synchronize(device)
    
    compare(actual, expected, category="elementwise", dtype=dtype)
