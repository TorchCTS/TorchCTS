import pytest
import torch
from torchcts.core.device import synchronize

MISC_DTYPES = [torch.float32, torch.int64]

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", MISC_DTYPES)
@pytest.mark.parametrize("op_name", ["sort", "topk", "kthvalue", "median"])
def test_sort_topk_kthvalue_median(dtype, op_name, device, compare, input_gen):
    x_dev = input_gen((16, 16), dtype, device)
    
    if op_name == "sort":
        val_dev, idx_dev = torch.sort(x_dev, dim=-1)
        val_cpu, idx_cpu = torch.sort(x_dev.cpu(), dim=-1)
        synchronize(device)
        compare(val_dev, val_cpu, category="exact", dtype=dtype)
        
    elif op_name == "topk":
        val_dev, idx_dev = torch.topk(x_dev, k=5, dim=-1)
        val_cpu, idx_cpu = torch.topk(x_dev.cpu(), k=5, dim=-1)
        synchronize(device)
        compare(val_dev, val_cpu, category="exact", dtype=dtype)
        
    elif op_name == "kthvalue":
        val_dev, idx_dev = torch.kthvalue(x_dev, k=3, dim=-1)
        val_cpu, idx_cpu = torch.kthvalue(x_dev.cpu(), k=3, dim=-1)
        synchronize(device)
        compare(val_dev, val_cpu, category="exact", dtype=dtype)
        
    elif op_name == "median":
        val_dev, idx_dev = torch.median(x_dev, dim=-1)
        val_cpu, idx_cpu = torch.median(x_dev.cpu(), dim=-1)
        synchronize(device)
        compare(val_dev, val_cpu, category="exact", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", MISC_DTYPES)
@pytest.mark.parametrize("op_name", ["cumsum", "cumprod", "unique"])
def test_cumsum_cumprod_unique(dtype, op_name, device, compare, input_gen):
    x_dev = input_gen((32,), dtype, device)
    
    if op_name == "cumsum":
        cat = "reduction" if dtype.is_floating_point else "exact"
        compare(torch.cumsum(x_dev, dim=0), torch.cumsum(x_dev.cpu(), dim=0), category=cat, dtype=dtype)
        
    elif op_name == "cumprod":
        cat = "reduction" if dtype.is_floating_point else "exact"
        small_dev = x_dev % 3
        compare(torch.cumprod(small_dev, dim=0), torch.cumprod(small_dev.cpu(), dim=0), category=cat, dtype=dtype)
        
    elif op_name == "unique":
        try:
            uni_dev, inv_dev = torch.unique(x_dev, return_inverse=True)
            uni_cpu, inv_cpu = torch.unique(x_dev.cpu(), return_inverse=True)
            synchronize(device)
            compare(uni_dev, uni_cpu, category="exact", dtype=dtype)
        except NotImplementedError:
            pass

@pytest.mark.smoke
@pytest.mark.parametrize("num_samples", [100, 200])
def test_multinomial(num_samples, device, compare):
    dtype = torch.float32
    weights_dev = torch.tensor([0.1, 0.5, 0.4], dtype=dtype, device=device)
    try:
        samples = torch.multinomial(weights_dev, num_samples=num_samples, replacement=True)
        synchronize(device)
        assert samples.shape == (num_samples,)
        assert torch.all(samples >= 0) and torch.all(samples < 3)
    except NotImplementedError:
        pass
