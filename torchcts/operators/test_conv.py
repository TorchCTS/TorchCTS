import pytest
import torch
from torchcts.core.device import synchronize

CONV_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", CONV_DTYPES)
def test_conv1d(dtype, device, compare, input_gen):
    # Input: (batch, in_channels, width)
    x_dev = input_gen((2, 3, 32), dtype, device)
    weight_dev = input_gen((4, 3, 3), dtype, device)
    bias_dev = input_gen((4,), dtype, device)
    
    expected = torch.nn.functional.conv1d(x_dev.cpu(), weight_dev.cpu(), bias_dev.cpu(), stride=1, padding=1)
    actual = torch.nn.functional.conv1d(x_dev, weight_dev, bias_dev, stride=1, padding=1)
    synchronize(device)
    
    compare(actual, expected, category="conv", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", CONV_DTYPES)
def test_conv2d(dtype, device, compare, input_gen):
    # Input: (batch, in_channels, height, width)
    x_dev = input_gen((2, 3, 16, 16), dtype, device)
    weight_dev = input_gen((8, 3, 3, 3), dtype, device)
    bias_dev = input_gen((8,), dtype, device)
    
    expected = torch.nn.functional.conv2d(x_dev.cpu(), weight_dev.cpu(), bias_dev.cpu(), stride=1, padding=1)
    actual = torch.nn.functional.conv2d(x_dev, weight_dev, bias_dev, stride=1, padding=1)
    synchronize(device)
    
    compare(actual, expected, category="conv", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", CONV_DTYPES)
def test_conv3d(dtype, device, compare, input_gen):
    # Input: (batch, in_channels, depth, height, width)
    x_dev = input_gen((2, 2, 8, 8, 8), dtype, device)
    weight_dev = input_gen((4, 2, 3, 3, 3), dtype, device)
    bias_dev = input_gen((4,), dtype, device)
    
    expected = torch.nn.functional.conv3d(x_dev.cpu(), weight_dev.cpu(), bias_dev.cpu(), stride=1, padding=1)
    actual = torch.nn.functional.conv3d(x_dev, weight_dev, bias_dev, stride=1, padding=1)
    synchronize(device)
    
    compare(actual, expected, category="conv", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", CONV_DTYPES)
def test_conv_transpose2d(dtype, device, compare, input_gen):
    # Input: (batch, in_channels, height, width)
    x_dev = input_gen((2, 4, 8, 8), dtype, device)
    weight_dev = input_gen((4, 8, 3, 3), dtype, device)
    bias_dev = input_gen((8,), dtype, device)
    
    expected = torch.nn.functional.conv_transpose2d(x_dev.cpu(), weight_dev.cpu(), bias_dev.cpu(), stride=1, padding=1)
    actual = torch.nn.functional.conv_transpose2d(x_dev, weight_dev, bias_dev, stride=1, padding=1)
    synchronize(device)
    
    compare(actual, expected, category="conv", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", CONV_DTYPES)
def test_conv_depthwise_groups(dtype, device, compare, input_gen):
    # Depthwise: groups = in_channels
    x_dev = input_gen((2, 4, 16, 16), dtype, device)
    weight_dev = input_gen((4, 1, 3, 3), dtype, device) # out_channels = 4, groups = 4
    
    expected = torch.nn.functional.conv2d(x_dev.cpu(), weight_dev.cpu(), stride=1, padding=1, groups=4)
    actual = torch.nn.functional.conv2d(x_dev, weight_dev, stride=1, padding=1, groups=4)
    synchronize(device)
    
    compare(actual, expected, category="conv", dtype=dtype)
