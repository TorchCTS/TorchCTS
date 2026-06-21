import pytest
import torch
from torchcts.core.device import synchronize

MATMUL_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.smoke
@pytest.mark.benchmarkable
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
@pytest.mark.parametrize("op_name", ["dot", "mv"])
def test_dot_mv(dtype, op_name, device, compare, input_gen):
    if op_name == "dot":
        x_dev = input_gen((128,), dtype, device)
        y_dev = input_gen((128,), dtype, device)
        compare(torch.dot(x_dev, y_dev), torch.dot(x_dev.cpu(), y_dev.cpu()), category="matmul", dtype=dtype)
    else:
        M_dev = input_gen((32, 64), dtype, device)
        v_dev = input_gen((64,), dtype, device)
        compare(torch.mv(M_dev, v_dev), torch.mv(M_dev.cpu(), v_dev.cpu()), category="matmul", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.benchmarkable
@pytest.mark.parametrize("layout_a", ["contiguous", "transpose"])
@pytest.mark.parametrize("layout_b", ["contiguous", "transpose"])
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_mm_layouts(dtype, layout_a, layout_b, device, compare, input_gen):
    M, K, N = 32, 64, 48
    shape_a = (M, K)
    shape_b = (K, N)
    
    a_dev = input_gen(shape_a, dtype, device, layout=layout_a)
    b_dev = input_gen(shape_b, dtype, device, layout=layout_b)
    
    a_cpu, b_cpu = a_dev.cpu(), b_dev.cpu()
    
    expected = torch.mm(a_cpu, b_cpu)
    actual = torch.mm(a_dev, b_dev)
    synchronize(device)
    
    category = "noncontiguous_mm" if (layout_a == "transpose" or layout_b == "transpose") else "matmul"
    compare(actual, expected, category=category, dtype=dtype)

@pytest.mark.smoke
@pytest.mark.benchmarkable
@pytest.mark.parametrize("layout_a", ["contiguous", "transpose"])
@pytest.mark.parametrize("layout_b", ["contiguous", "transpose"])
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_bmm_layouts(dtype, layout_a, layout_b, device, compare, input_gen):
    B, M, K, N = 4, 16, 32, 24
    shape_a = (B, M, K)
    shape_b = (B, K, N)
    
    a_dev = input_gen(shape_a, dtype, device, layout=layout_a)
    b_dev = input_gen(shape_b, dtype, device, layout=layout_b)
    
    a_cpu, b_cpu = a_dev.cpu(), b_dev.cpu()
    
    expected = torch.bmm(a_cpu, b_cpu)
    actual = torch.bmm(a_dev, b_dev)
    synchronize(device)
    
    compare(actual, expected, category="matmul", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.benchmarkable
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_addmm(dtype, device, compare, input_gen):
    M, K, N = 32, 64, 48
    beta, alpha = 0.5, 1.5
    mat_dev = input_gen((M, N), dtype, device)
    a_dev = input_gen((M, K), dtype, device)
    b_dev = input_gen((K, N), dtype, device)
    
    expected = torch.addmm(mat_dev.cpu(), a_dev.cpu(), b_dev.cpu(), beta=beta, alpha=alpha)
    actual = torch.addmm(mat_dev, a_dev, b_dev, beta=beta, alpha=alpha)
    synchronize(device)
    
    compare(actual, expected, category="matmul", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.benchmarkable
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_matmul_general(dtype, device, compare, input_gen):
    # Broadcasting matmul: (2, 3, 4) x (4, 5) -> (2, 3, 5)
    a_dev = input_gen((2, 3, 4), dtype, device)
    b_dev = input_gen((4, 5), dtype, device)
    
    expected = torch.matmul(a_dev.cpu(), b_dev.cpu())
    actual = torch.matmul(a_dev, b_dev)
    synchronize(device)
    
    compare(actual, expected, category="matmul", dtype=dtype)
