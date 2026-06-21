import pytest
import torch
from torchcts.core.device import synchronize

NORM_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", NORM_DTYPES)
def test_layer_norm(dtype, device, compare, input_gen):
    shape = (4, 16, 32)
    normalized_shape = (32,)
    
    x_dev = input_gen(shape, dtype, device)
    weight_dev = input_gen(normalized_shape, dtype, device)
    bias_dev = input_gen(normalized_shape, dtype, device)
    
    expected = torch.nn.functional.layer_norm(x_dev.cpu(), normalized_shape, weight_dev.cpu(), bias_dev.cpu())
    actual = torch.nn.functional.layer_norm(x_dev, normalized_shape, weight_dev, bias_dev)
    synchronize(device)
    
    compare(actual, expected, category="norm", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", NORM_DTYPES)
def test_group_norm(dtype, device, compare, input_gen):
    shape = (2, 8, 16, 16)
    num_groups = 4
    num_channels = 8
    
    x_dev = input_gen(shape, dtype, device)
    weight_dev = input_gen((num_channels,), dtype, device)
    bias_dev = input_gen((num_channels,), dtype, device)
    
    x_cpu = x_dev.cpu()
    w_cpu = weight_dev.cpu()
    b_cpu = bias_dev.cpu()
    if dtype in (torch.float16, torch.bfloat16):
        x_cpu = x_cpu.float()
        w_cpu = w_cpu.float()
        b_cpu = b_cpu.float()
        
    expected = torch.nn.functional.group_norm(x_cpu, num_groups, w_cpu, b_cpu)
    if dtype == torch.float16:
        expected = expected.half()
        
    actual = torch.nn.functional.group_norm(x_dev, num_groups, weight_dev, bias_dev)
    synchronize(device)
    
    compare(actual, expected, category="norm", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", NORM_DTYPES)
def test_batch_norm(dtype, device, compare, input_gen):
    if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
        # CPU batch_norm doesn't support half/bfloat16 in eager PyTorch
        return
        
    shape = (4, 8, 16, 16)
    num_features = 8
    
    x_dev = input_gen(shape, dtype, device)
    running_mean_dev = torch.zeros(num_features, dtype=torch.float32, device=device)
    running_var_dev = torch.ones(num_features, dtype=torch.float32, device=device)
    weight_dev = input_gen((num_features,), dtype, device)
    bias_dev = input_gen((num_features,), dtype, device)
    
    running_mean_cpu = running_mean_dev.cpu()
    running_var_cpu = running_var_dev.cpu()
    
    x_cpu = x_dev.cpu()
    w_cpu = weight_dev.cpu()
    b_cpu = bias_dev.cpu()
    
    if dtype in (torch.float16, torch.bfloat16):
        x_cpu = x_cpu.float()
        w_cpu = w_cpu.float()
        b_cpu = b_cpu.float()
        
    expected = torch.nn.functional.batch_norm(
        x_cpu, running_mean_cpu, running_var_cpu, w_cpu, b_cpu, training=True
    )
    if dtype == torch.float16:
        expected = expected.half()
    elif dtype == torch.bfloat16:
        expected = expected.to(torch.bfloat16)
        
    actual = torch.nn.functional.batch_norm(
        x_dev, running_mean_dev, running_var_dev, weight_dev, bias_dev, training=True
    )
    synchronize(device)
    
    compare(actual, expected, category="norm", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", NORM_DTYPES)
def test_instance_norm(dtype, device, compare, input_gen):
    shape = (2, 4, 16, 16)
    x_dev = input_gen(shape, dtype, device)
    
    x_cpu = x_dev.cpu()
    if dtype == torch.float16:
        x_cpu = x_cpu.float()
        
    expected = torch.nn.functional.instance_norm(x_cpu, use_input_stats=True)
    if dtype == torch.float16:
        expected = expected.half()
        
    actual = torch.nn.functional.instance_norm(x_dev, use_input_stats=True)
    synchronize(device)
    
    compare(actual, expected, category="norm", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", NORM_DTYPES)
def test_rms_norm_custom(dtype, device, compare, input_gen):
    shape = (4, 16, 32)
    eps = 1e-6
    
    x_dev = input_gen(shape, dtype, device)
    weight_dev = input_gen((32,), dtype, device)
    
    x_cpu = x_dev.cpu()
    w_cpu = weight_dev.cpu()
    
    variance = x_cpu.pow(2).mean(-1, keepdim=True)
    expected = x_cpu * torch.rsqrt(variance + eps) * w_cpu
    
    var_dev = x_dev.pow(2).mean(-1, keepdim=True)
    actual = x_dev * torch.rsqrt(var_dev + eps) * weight_dev
    synchronize(device)
    
    compare(actual, expected, category="norm", dtype=dtype)
