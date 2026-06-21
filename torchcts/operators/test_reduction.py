import pytest
import torch
from torchcts.core.device import synchronize

REDUCTION_FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
REDUCTION_INT_DTYPES = [torch.int64, torch.int32]

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", REDUCTION_FLOAT_DTYPES)
def test_reductions_basic(dtype, device, manifest, compare, input_gen):
    shape = (16, 16)
    x_dev = input_gen(shape, dtype, device)
    x_cpu = x_dev.cpu()
    
    # sum
    compare(torch.sum(x_dev), torch.sum(x_cpu), category="reduction", dtype=dtype)
    compare(torch.sum(x_dev, dim=0), torch.sum(x_cpu, dim=0), category="reduction", dtype=dtype)
    compare(torch.sum(x_dev, dim=1, keepdim=True), torch.sum(x_cpu, dim=1, keepdim=True), category="reduction", dtype=dtype)
    
    # mean
    compare(torch.mean(x_dev), torch.mean(x_cpu), category="reduction", dtype=dtype)
    compare(torch.mean(x_dev, dim=0), torch.mean(x_cpu, dim=0), category="reduction", dtype=dtype)
    
    # std and var
    if dtype != torch.bfloat16: # bfloat16 std can be unstable on small tensors
        compare(torch.std(x_dev), torch.std(x_cpu), category="reduction", dtype=dtype)
        compare(torch.var(x_dev, dim=1), torch.var(x_cpu, dim=1), category="reduction", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", REDUCTION_FLOAT_DTYPES + REDUCTION_INT_DTYPES)
def test_amax_amin(dtype, device, manifest, compare, input_gen):
    shape = (16, 16)
    x_dev = input_gen(shape, dtype, device)
    x_cpu = x_dev.cpu()
    
    compare(torch.amax(x_dev, dim=0), torch.amax(x_cpu, dim=0), category="exact" if dtype in REDUCTION_INT_DTYPES else "elementwise", dtype=dtype)
    compare(torch.amin(x_dev, dim=1, keepdim=True), torch.amin(x_cpu, dim=1, keepdim=True), category="exact" if dtype in REDUCTION_INT_DTYPES else "elementwise", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", REDUCTION_FLOAT_DTYPES + REDUCTION_INT_DTYPES)
def test_argmax_argmin(dtype, device, manifest, compare, input_gen):
    shape = (16, 16)
    x_dev = input_gen(shape, dtype, device)
    x_cpu = x_dev.cpu()
    
    compare(torch.argmax(x_dev, dim=0), torch.argmax(x_cpu, dim=0), category="exact", dtype=torch.int64)
    compare(torch.argmin(x_dev, dim=1), torch.argmin(x_cpu, dim=1), category="exact", dtype=torch.int64)

@pytest.mark.smoke
@pytest.mark.parametrize("dim", [0, 1])
def test_any_all(dim, device, manifest, compare, input_gen):
    shape = (16, 16)
    x_dev = input_gen(shape, torch.bool, device)
    x_cpu = x_dev.cpu()
    
    compare(torch.any(x_dev), torch.any(x_cpu), category="exact", dtype=torch.bool)
    compare(torch.any(x_dev, dim=dim), torch.any(x_cpu, dim=dim), category="exact", dtype=torch.bool)
    compare(torch.all(x_dev, dim=dim, keepdim=True), torch.all(x_cpu, dim=dim, keepdim=True), category="exact", dtype=torch.bool)

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", REDUCTION_FLOAT_DTYPES)
def test_norm(dtype, device, manifest, compare, input_gen):
    shape = (16, 16)
    x_dev = input_gen(shape, dtype, device)
    x_cpu = x_dev.cpu()
    
    # Frobenius norm
    compare(torch.norm(x_dev), torch.norm(x_cpu), category="reduction", dtype=dtype)
    # L1 norm
    compare(torch.norm(x_dev, p=1, dim=0), torch.norm(x_cpu, p=1, dim=0), category="reduction", dtype=dtype)
