import pytest
import torch
from torchcts.core.device import synchronize

ACTIVATION_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.smoke
@pytest.mark.benchmarkable
@pytest.mark.parametrize("op_name", ["relu", "gelu", "silu", "mish", "hardswish"])
@pytest.mark.parametrize("dtype", ACTIVATION_DTYPES)
def test_activations_basic(op_name, dtype, device, compare, input_gen):
    op_fn = getattr(torch, op_name, None)
    if op_fn is None:
        op_fn = getattr(torch.nn.functional, op_name, None)
        
    if op_fn is None:
        pytest.fail(f"Activation op {op_name} not found")

    shape = (32, 32)
    x_dev = input_gen(shape, dtype, device)
    x_cpu = x_dev.cpu()
    
    expected = op_fn(x_cpu)
    actual = op_fn(x_dev)
    synchronize(device)
    
    compare(actual, expected, category="elementwise", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.benchmarkable
@pytest.mark.parametrize("op_name", ["softmax", "log_softmax"])
@pytest.mark.parametrize("dtype", ACTIVATION_DTYPES)
def test_softmax_log_softmax(op_name, dtype, device, compare, input_gen):
    op_fn = getattr(torch.nn.functional, op_name)
    
    shape = (16, 32)
    x_dev = input_gen(shape, dtype, device)
    x_cpu = x_dev.cpu()
    
    expected = op_fn(x_cpu, dim=-1)
    actual = op_fn(x_dev, dim=-1)
    synchronize(device)
    
    compare(actual, expected, category="elementwise", dtype=dtype)
