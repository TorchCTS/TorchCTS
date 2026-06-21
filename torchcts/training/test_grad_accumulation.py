import pytest
import torch
from torchcts.core.device import synchronize

@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_gradient_accumulation(dtype, device, manifest, compare):
    # Model and inputs
    weight_cpu = torch.randn(8, 8, dtype=dtype)
    x_cpu = torch.randn(4, 8, dtype=dtype) # 4 batches
    
    # 1. Accumulate over 4 micro-batches
    weight_acc = weight_cpu.clone().to(device)
    weight_acc.requires_grad = True
    
    for i in range(4):
        x_micro = x_cpu[i:i+1].to(device)
        out = torch.mm(x_micro, weight_acc).sum()
        out.backward()
        
    # 2. Run big-batch once
    weight_single = weight_cpu.clone().to(device)
    weight_single.requires_grad = True
    
    out_single = torch.mm(x_cpu.to(device), weight_single).sum()
    out_single.backward()
    
    synchronize(device)
    
    # G grads must match exactly
    compare(weight_acc.grad, weight_single.grad, category="exact", dtype=dtype)
