import pytest
import torch
from torchcts.core.device import synchronize

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("clipping_method", ["norm", "value"])
def test_gradient_clipping(clipping_method, dtype, device, manifest, compare):
    w_cpu = torch.randn(8, 8, dtype=dtype)
    w_cpu.grad = torch.randn(8, 8, dtype=dtype) * 10.0 # large grads
    
    w_dev = w_cpu.clone().to(device)
    w_dev.grad = w_cpu.grad.clone().to(device)
    
    if clipping_method == "norm":
        # 1. Clip norm
        torch.nn.utils.clip_grad_norm_([w_cpu], max_norm=1.0)
        torch.nn.utils.clip_grad_norm_([w_dev], max_norm=1.0)
        synchronize(device)
        compare(w_dev.grad, w_cpu.grad, category="elementwise", dtype=dtype)
        
    elif clipping_method == "value":
        # 2. Clip value
        torch.nn.utils.clip_grad_value_([w_cpu], clip_value=0.5)
        torch.nn.utils.clip_grad_value_([w_dev], clip_value=0.5)
        synchronize(device)
        compare(w_dev.grad, w_cpu.grad, category="exact", dtype=dtype)
