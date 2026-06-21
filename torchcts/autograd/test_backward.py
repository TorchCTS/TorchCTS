import pytest
import torch
from torchcts.core.device import synchronize

BACKWARD_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
ACTIVATIONS = ["relu", "gelu", "sigmoid", "tanh", "silu"]

@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
@pytest.mark.parametrize("op_name", ACTIVATIONS)
def test_first_order_backward(dtype, op_name, device, compare, input_gen):
    # Test simple model backpropagation (Linear + Activation)
    x_dev = input_gen((4, 8), dtype, device)
    x_dev.requires_grad = True
    
    w_cpu = torch.randn(8, 4, dtype=dtype)
    w_dev = w_cpu.to(device)
    w_dev.requires_grad = True
    
    x_cpu = x_dev.cpu().detach()
    x_cpu.requires_grad = True
    
    w_ref = w_cpu.clone().detach()
    w_ref.requires_grad = True
    
    op_fn = getattr(torch.nn.functional, op_name)
    
    out_dev = op_fn(torch.mm(x_dev, w_dev)).sum()
    out_cpu = op_fn(torch.mm(x_cpu, w_ref)).sum()
    
    out_dev.backward()
    out_cpu.backward()
    synchronize(device)
    
    compare(x_dev.grad, x_cpu.grad, category="matmul_backward", dtype=dtype)
    compare(w_dev.grad, w_ref.grad, category="matmul_backward", dtype=dtype)
