import pytest
import torch
from torchcts.core.device import synchronize

POOL_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", POOL_DTYPES)
def test_max_pool2d(dtype, device, compare, input_gen):
    shape = (2, 3, 16, 16)
    x_dev = input_gen(shape, dtype, device)
    x_cpu = x_dev.cpu()
    
    expected = torch.nn.functional.max_pool2d(x_cpu, kernel_size=2, stride=2)
    actual = torch.nn.functional.max_pool2d(x_dev, kernel_size=2, stride=2)
    synchronize(device)
    
    compare(actual, expected, category="elementwise", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", POOL_DTYPES)
def test_avg_pool2d(dtype, device, compare, input_gen):
    shape = (2, 3, 16, 16)
    x_dev = input_gen(shape, dtype, device)
    x_cpu = x_dev.cpu()
    
    expected = torch.nn.functional.avg_pool2d(x_cpu, kernel_size=2, stride=2)
    actual = torch.nn.functional.avg_pool2d(x_dev, kernel_size=2, stride=2)
    synchronize(device)
    
    compare(actual, expected, category="elementwise", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", POOL_DTYPES)
def test_adaptive_avg_pool2d(dtype, device, compare, input_gen):
    shape = (2, 3, 16, 16)
    x_dev = input_gen(shape, dtype, device)
    x_cpu = x_dev.cpu()
    
    expected = torch.nn.functional.adaptive_avg_pool2d(x_cpu, output_size=(8, 8))
    actual = torch.nn.functional.adaptive_avg_pool2d(x_dev, output_size=(8, 8))
    synchronize(device)
    
    compare(actual, expected, category="elementwise", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", POOL_DTYPES)
def test_adaptive_max_pool2d(dtype, device, compare, input_gen):
    shape = (2, 3, 16, 16)
    x_dev = input_gen(shape, dtype, device)
    x_cpu = x_dev.cpu()
    
    expected_out, expected_ind = torch.nn.functional.adaptive_max_pool2d(x_cpu, output_size=(8, 8))
    actual_out, actual_ind = torch.nn.functional.adaptive_max_pool2d(x_dev, output_size=(8, 8))
    synchronize(device)
    
    compare(actual_out, expected_out, category="elementwise", dtype=dtype)
    compare(actual_ind, expected_ind, category="exact", dtype=torch.int64)
