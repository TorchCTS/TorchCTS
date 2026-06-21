import pytest
import torch
from torchcts.core.device import synchronize

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.medium
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", ["transpose", "slice"])
def test_strided_inplace_updates(case, dtype, device, manifest, compare, input_gen):
    if case == "transpose":
        # 1. In-place fill on transpose view
        x_dev = input_gen((16, 16), dtype, device)
        x_trans_dev = x_dev.T
        
        x_cpu = x_dev.cpu()
        x_trans_cpu = x_cpu.T
        
        x_trans_dev.fill_(42.0)
        x_trans_cpu.fill_(42.0)
        synchronize(device)
        compare(x_dev, x_cpu, category="exact", dtype=dtype)
        
    elif case == "slice":
        # 2. In-place clamp on sliced view
        y_base_dev = input_gen((32,), dtype, device)
        y_base_cpu = y_base_dev.cpu()
        
        y_slice_dev = y_base_dev[::2]
        y_slice_cpu = y_base_cpu[::2]
        
        y_slice_dev.clamp_(min=-0.5, max=0.5)
        y_slice_cpu.clamp_(min=-0.5, max=0.5)
        synchronize(device)
        compare(y_base_dev, y_base_cpu, category="exact", dtype=dtype)
