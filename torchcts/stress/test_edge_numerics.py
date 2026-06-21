import pytest
import torch
import math

@pytest.mark.stress
@pytest.mark.parametrize("offset", [1.0, 2.0])
def test_nan_inf_propagation(offset, device):
    # NaN
    nan_tensor = torch.tensor([float('nan'), 1.0, 2.0], device=device)
    out_nan = nan_tensor + offset
    assert torch.isnan(out_nan[0])
    assert out_nan[1].item() == 1.0 + offset
    
    # Inf
    inf_tensor = torch.tensor([float('inf'), -float('inf'), 3.0], device=device)
    out_inf = inf_tensor + offset
    assert torch.isinf(out_inf[0])
    assert torch.isinf(out_inf[1])
    assert out_inf[2].item() == 3.0 + offset

@pytest.mark.stress
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_dtype_min_max(dtype, device, manifest):
    if dtype.is_floating_point:
        info = torch.finfo(dtype)
        min_val, max_val = info.min, info.max
    else:
        info = torch.iinfo(dtype)
        min_val, max_val = info.min, info.max
        
    t_min = torch.tensor([min_val], dtype=dtype, device=device)
    t_max = torch.tensor([max_val], dtype=dtype, device=device)
    
    assert t_min.item() == min_val
    assert t_max.item() == max_val
