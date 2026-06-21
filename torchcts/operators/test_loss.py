import pytest
import torch
from torchcts.core.device import synchronize

LOSS_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", LOSS_DTYPES)
def test_cross_entropy(dtype, device, compare, input_gen):
    shape = (4, 10)
    x_dev = input_gen(shape, dtype, device)
    y_dev = torch.randint(0, 10, (4,), dtype=torch.int64, device=device)
    
    expected = torch.nn.functional.cross_entropy(x_dev.cpu(), y_dev.cpu())
    actual = torch.nn.functional.cross_entropy(x_dev, y_dev)
    synchronize(device)
    
    compare(actual, expected, category="loss", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", LOSS_DTYPES)
def test_mse_loss(dtype, device, compare, input_gen):
    shape = (16, 16)
    x_dev = input_gen(shape, dtype, device)
    y_dev = input_gen(shape, dtype, device)
    
    expected = torch.nn.functional.mse_loss(x_dev.cpu(), y_dev.cpu())
    actual = torch.nn.functional.mse_loss(x_dev, y_dev)
    synchronize(device)
    
    compare(actual, expected, category="loss", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", LOSS_DTYPES)
def test_nll_loss(dtype, device, compare, input_gen):
    shape = (4, 10)
    x_dev = torch.nn.functional.log_softmax(input_gen(shape, dtype, device), dim=-1)
    y_dev = torch.randint(0, 10, (4,), dtype=torch.int64, device=device)
    
    expected = torch.nn.functional.nll_loss(x_dev.cpu(), y_dev.cpu())
    actual = torch.nn.functional.nll_loss(x_dev, y_dev)
    synchronize(device)
    
    compare(actual, expected, category="loss", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", LOSS_DTYPES)
def test_bce_with_logits(dtype, device, compare, input_gen):
    shape = (8, 8)
    x_dev = input_gen(shape, dtype, device)
    y_dev = torch.randint(0, 2, shape, dtype=dtype, device=device)
    
    expected = torch.nn.functional.binary_cross_entropy_with_logits(x_dev.cpu(), y_dev.cpu())
    actual = torch.nn.functional.binary_cross_entropy_with_logits(x_dev, y_dev)
    synchronize(device)
    
    compare(actual, expected, category="loss", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", LOSS_DTYPES)
@pytest.mark.parametrize("op_name", ["smooth_l1", "huber"])
def test_smooth_l1_huber_loss(dtype, op_name, device, compare, input_gen):
    shape = (16, 16)
    x_dev = input_gen(shape, dtype, device)
    y_dev = input_gen(shape, dtype, device)
    
    if op_name == "smooth_l1":
        expected = torch.nn.functional.smooth_l1_loss(x_dev.cpu(), y_dev.cpu())
        actual = torch.nn.functional.smooth_l1_loss(x_dev, y_dev)
    else:
        expected = torch.nn.functional.huber_loss(x_dev.cpu(), y_dev.cpu(), delta=1.0)
        actual = torch.nn.functional.huber_loss(x_dev, y_dev, delta=1.0)
        
    synchronize(device)
    compare(actual, expected, category="loss", dtype=dtype)
