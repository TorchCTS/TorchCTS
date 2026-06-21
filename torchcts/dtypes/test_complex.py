import pytest
import torch
from torchcts.core.device import synchronize

COMPLEX_DTYPES = [torch.complex64, torch.complex128]

@pytest.mark.medium
@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
@pytest.mark.parametrize("op_name", ["view_as_real", "resolve_conj", "abs", "angle"])
def test_complex_operations(dtype, op_name, device, compare):
    try:
        torch.empty(1, dtype=dtype, device=device)
    except Exception:
        pytest.skip(f"Dtype {dtype} not supported on device {device}")

    # Create complex tensor on CPU
    x_cpu = torch.randn(8, 8, dtype=dtype)
    x_dev = x_cpu.to(device)
    
    if op_name == "view_as_real":
        expected = torch.view_as_real(x_cpu)
        actual = torch.view_as_real(x_dev)
        synchronize(device)
        ref_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        compare(actual, expected, category="exact", dtype=ref_dtype)
        
    elif op_name == "resolve_conj":
        y_cpu = x_cpu.conj()
        y_dev = x_dev.conj()
        synchronize(device)
        compare(y_dev.resolve_conj(), y_cpu.resolve_conj(), category="exact", dtype=dtype)
        
    elif op_name == "abs":
        expected = torch.abs(x_cpu)
        actual = torch.abs(x_dev)
        synchronize(device)
        ref_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        compare(actual, expected, category="elementwise", dtype=ref_dtype)
        
    elif op_name == "angle":
        expected = torch.angle(x_cpu)
        actual = torch.angle(x_dev)
        synchronize(device)
        ref_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        compare(actual, expected, category="elementwise", dtype=ref_dtype)


@pytest.mark.medium
@pytest.mark.parametrize("dtype", [torch.complex64])
def test_complex_gather_scatter(dtype, device, compare):
    """Gather, take_along_dim, index_select, and scatter must preserve imaginary parts."""
    try:
        torch.empty(1, dtype=dtype, device=device)
    except Exception:
        pytest.skip(f"Dtype {dtype} not supported on device {device}")

    torch.manual_seed(42)
    x = torch.randn(8, 16, dtype=dtype, device=device)

    # 1. Gather (using unique indices per row via argsort to ensure deterministic scatter behavior)
    idx = torch.rand(8, 16, device=device).argsort(dim=1)[:, :6]
    result_gather = torch.gather(x, 1, idx)
    expected_gather = torch.gather(x.cpu(), 1, idx.cpu())
    synchronize(device)
    compare(result_gather, expected_gather, category="elementwise", dtype=dtype)
    assert result_gather.cpu().imag.abs().sum() > 0, "Imaginary parts are all zero after gather"

    # 2. Take Along Dim
    result_take = torch.take_along_dim(x, idx, dim=1)
    expected_take = torch.take_along_dim(x.cpu(), idx.cpu(), dim=1)
    synchronize(device)
    compare(result_take, expected_take, category="elementwise", dtype=dtype)

    # 3. Index Select
    indices = torch.tensor([0, 3, 7], dtype=torch.long, device=device)
    result_select = torch.index_select(x, 1, indices)
    expected_select = torch.index_select(x.cpu(), 1, indices.cpu())
    synchronize(device)
    compare(result_select, expected_select, category="elementwise", dtype=dtype)

    # 4. Scatter
    src = torch.randn(8, 6, dtype=dtype, device=device)
    result_scatter = x.clone().scatter(1, idx, src)
    expected_scatter = x.cpu().clone().scatter(1, idx.cpu(), src.cpu())
    synchronize(device)
    compare(result_scatter, expected_scatter, category="elementwise", dtype=dtype)

