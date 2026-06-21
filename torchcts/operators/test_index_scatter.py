import pytest
import torch
from torchcts.core.device import synchronize

INDEX_DTYPES = [torch.float32, torch.int64]

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_index_select(dtype, device, compare, input_gen):
    shape = (8, 8)
    x_dev = input_gen(shape, dtype, device)
    indices_dev = torch.tensor([0, 2, 4], dtype=torch.int64, device=device)
    
    expected = torch.index_select(x_dev.cpu(), 0, indices_dev.cpu())
    actual = torch.index_select(x_dev, 0, indices_dev)
    synchronize(device)
    
    compare(actual, expected, category="exact", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_index_fill(dtype, device, compare, input_gen):
    shape = (8, 8)
    x_dev = input_gen(shape, dtype, device)
    indices_dev = torch.tensor([1, 3, 5], dtype=torch.int64, device=device)
    val = 9.99 if dtype.is_floating_point else 99
    
    expected = x_dev.cpu().clone().index_fill_(0, indices_dev.cpu(), val)
    actual = x_dev.clone().index_fill_(0, indices_dev, val)
    synchronize(device)
    
    compare(actual, expected, category="exact", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_index_copy(dtype, device, compare, input_gen):
    shape = (8, 8)
    x_dev = input_gen(shape, dtype, device)
    indices_dev = torch.tensor([1, 3, 5], dtype=torch.int64, device=device)
    source_dev = input_gen((3, 8), dtype, device)
    
    expected = x_dev.cpu().clone().index_copy_(0, indices_dev.cpu(), source_dev.cpu())
    actual = x_dev.clone().index_copy_(0, indices_dev, source_dev)
    synchronize(device)
    
    compare(actual, expected, category="exact", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_index_add(dtype, device, compare, input_gen):
    shape = (8, 8)
    x_dev = input_gen(shape, dtype, device)
    indices_dev = torch.tensor([1, 3, 5], dtype=torch.int64, device=device)
    source_dev = input_gen((3, 8), dtype, device)
    
    expected = x_dev.cpu().clone().index_add_(0, indices_dev.cpu(), source_dev.cpu())
    actual = x_dev.clone().index_add_(0, indices_dev, source_dev)
    synchronize(device)
    
    compare(actual, expected, category="elementwise", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_gather(dtype, device, compare, input_gen):
    shape = (8, 8)
    x_dev = input_gen(shape, dtype, device)
    indices_dev = torch.randint(0, 8, (4, 8), dtype=torch.int64, device=device)
    
    expected = torch.gather(x_dev.cpu(), 0, indices_dev.cpu())
    actual = torch.gather(x_dev, 0, indices_dev)
    synchronize(device)
    
    compare(actual, expected, category="exact", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", INDEX_DTYPES)
@pytest.mark.parametrize("op_name", ["scatter", "scatter_add"])
def test_scatter_ops(dtype, op_name, device, compare, input_gen):
    shape = (8, 8)
    x_dev = input_gen(shape, dtype, device)
    indices_cpu = torch.zeros((4, 8), dtype=torch.int64)
    for col in range(8):
        indices_cpu[:, col] = torch.randperm(8)[:4]
    indices_dev = indices_cpu.to(device)
    src_dev = input_gen((4, 8), dtype, device)
    
    if op_name == "scatter":
        expected = x_dev.cpu().clone().scatter_(0, indices_dev.cpu(), src_dev.cpu())
        actual = x_dev.clone().scatter_(0, indices_dev, src_dev)
        synchronize(device)
        compare(actual, expected, category="exact", dtype=dtype)
    else:
        expected = x_dev.cpu().clone().scatter_add_(0, indices_dev.cpu(), src_dev.cpu())
        actual = x_dev.clone().scatter_add_(0, indices_dev, src_dev)
        synchronize(device)
        compare(actual, expected, category="elementwise", dtype=dtype)
