import pytest
import torch
from torchcts.core.device import synchronize

# Helper to compare dense tensors
def compare_dense(cpu_t, dev_t_cpu):
    __tracebackhide__ = True
    assert cpu_t.shape == dev_t_cpu.shape, f"Shape mismatch: {cpu_t.shape} vs {dev_t_cpu.shape}"
    assert cpu_t.dtype == dev_t_cpu.dtype, f"Dtype mismatch: {cpu_t.dtype} vs {dev_t_cpu.dtype}"
    assert torch.allclose(cpu_t, dev_t_cpu, rtol=1e-4, atol=1e-4), "Value mismatch"

# Helper to run a sparse operation on CPU vs Device, catching NotImplementedError and RuntimeError skips
def check_sparse_op(op_fn, device, *args, **kwargs):
    # Prepare arguments for CPU
    cpu_args = [x.to("cpu") if isinstance(x, torch.Tensor) else x for x in args]
    cpu_kwargs = {k: (v.to("cpu") if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
    
    cpu_exception = None
    try:
        cpu_out = op_fn(*cpu_args, **cpu_kwargs)
    except Exception as e:
        cpu_exception = e
        
    # Prepare arguments for Device
    dev_args = [x.to(device) if isinstance(x, torch.Tensor) else x for x in args]
    dev_kwargs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
    
    if cpu_exception is not None:
        try:
            op_fn(*dev_args, **dev_kwargs)
            synchronize(device)
        except (NotImplementedError, RuntimeError) as dev_e:
            err_msg = str(dev_e).lower()
            if any(term in err_msg for term in ["not implemented", "not supported", "support", "requires compiling", "without mkl"]):
                pytest.skip(f"Operation not implemented/supported on CPU reference: {dev_e}")
            return
        except Exception:
            return
        pytest.fail(f"CPU raised {type(cpu_exception).__name__}: {cpu_exception}, but backend did not raise")
    else:
        try:
            dev_out = op_fn(*dev_args, **dev_kwargs)
            synchronize(device)
        except (NotImplementedError, RuntimeError) as dev_e:
            err_msg = str(dev_e).lower()
            if any(term in err_msg for term in ["not implemented", "not supported", "support", "requires compiling", "without mkl"]):
                pytest.skip(f"Operation/format not implemented on backend: {dev_e}")
            raise
            
        # Compare outputs
        if isinstance(cpu_out, torch.Tensor):
            if cpu_out.is_sparse or cpu_out.layout in (torch.sparse_coo, torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc):
                compare_dense(cpu_out.to_dense(), dev_out.to("cpu").to_dense())
            else:
                compare_dense(cpu_out, dev_out.to("cpu"))
        elif isinstance(cpu_out, tuple):
            assert len(cpu_out) == len(dev_out)
            for c_o, d_o in zip(cpu_out, dev_out):
                if isinstance(c_o, torch.Tensor):
                    if c_o.is_sparse or c_o.layout in (torch.sparse_coo, torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc):
                        compare_dense(c_o.to_dense(), d_o.to("cpu").to_dense())
                    else:
                        compare_dense(c_o, d_o.to("cpu"))
                else:
                    assert c_o == d_o
        else:
            assert cpu_out == dev_out

# ──────────────────────────────────────────────────────────────────────
# COO Layout Tests
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def coo_data():
    i = torch.tensor([[0, 1, 1], [2, 0, 2]], dtype=torch.int64)
    v = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32)
    size = (3, 3)
    return i, v, size

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sparse_coo_creation(device, dtype, coo_data):
    i, v, size = coo_data
    check_sparse_op(lambda i_t, v_t: torch.sparse_coo_tensor(i_t, v_t, size).to_dense(), device, i, v)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sparse_coo_coalesce(device, dtype, coo_data):
    i, v, size = coo_data
    check_sparse_op(lambda i_t, v_t: torch.sparse_coo_tensor(i_t, v_t, size).coalesce(), device, i, v)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sparse_coo_transpose(device, dtype, coo_data):
    i, v, size = coo_data
    check_sparse_op(lambda i_t, v_t: torch.sparse_coo_tensor(i_t, v_t, size).t(), device, i, v)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sparse_coo_add(device, dtype, coo_data):
    i, v, size = coo_data
    check_sparse_op(lambda i_t, v_t: torch.sparse_coo_tensor(i_t, v_t, size) + torch.sparse_coo_tensor(i_t, v_t, size), device, i, v)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sparse_coo_mm(device, dtype, coo_data):
    i, v, size = coo_data
    dense = torch.randn(3, 2, dtype=torch.float32)
    check_sparse_op(lambda i_t, v_t, d: torch.sparse.mm(torch.sparse_coo_tensor(i_t, v_t, size), d), device, i, v, dense)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sparse_coo_conversions(device, dtype, coo_data):
    i, v, size = coo_data
    check_sparse_op(lambda i_t, v_t: torch.sparse_coo_tensor(i_t, v_t, size).to_sparse_csr(), device, i, v)
    check_sparse_op(lambda i_t, v_t: torch.sparse_coo_tensor(i_t, v_t, size).to_sparse_csc(), device, i, v)
    check_sparse_op(lambda i_t, v_t: torch.sparse_coo_tensor(i_t, v_t, size).to_sparse_bsr((1, 1)), device, i, v)

# ──────────────────────────────────────────────────────────────────────
# CSR Layout Tests
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def csr_data():
    crow = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col = torch.tensor([0, 2, 1], dtype=torch.int64)
    val = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    size = (3, 3)
    return crow, col, val, size

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sparse_csr_creation(device, dtype, csr_data):
    crow, col, val, size = csr_data
    check_sparse_op(lambda r, c, v: torch.sparse_csr_tensor(r, c, v, size).to_dense(), device, crow, col, val)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sparse_csr_transpose(device, dtype, csr_data):
    crow, col, val, size = csr_data
    check_sparse_op(lambda r, c, v: torch.sparse_csr_tensor(r, c, v, size).transpose(0, 1), device, crow, col, val)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sparse_csr_mm(device, dtype, csr_data):
    crow, col, val, size = csr_data
    dense = torch.randn(3, 2, dtype=torch.float32)
    check_sparse_op(lambda r, c, v, d: torch.sparse.mm(torch.sparse_csr_tensor(r, c, v, size), d), device, crow, col, val, dense)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sparse_csr_conversions(device, dtype, csr_data):
    crow, col, val, size = csr_data
    check_sparse_op(lambda r, c, v: torch.sparse_csr_tensor(r, c, v, size).to_sparse_coo(), device, crow, col, val)

# ──────────────────────────────────────────────────────────────────────
# CSC Layout Tests
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def csc_data():
    ccol = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    row = torch.tensor([0, 2, 1], dtype=torch.int64)
    val = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    size = (3, 3)
    return ccol, row, val, size

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sparse_csc_creation(device, dtype, csc_data):
    ccol, row, val, size = csc_data
    check_sparse_op(lambda c, r, v: torch.sparse_csc_tensor(c, r, v, size).to_dense(), device, ccol, row, val)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sparse_csc_transpose(device, dtype, csc_data):
    ccol, row, val, size = csc_data
    check_sparse_op(lambda c, r, v: torch.sparse_csc_tensor(c, r, v, size).transpose(0, 1), device, ccol, row, val)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sparse_csc_mm(device, dtype, csc_data):
    ccol, row, val, size = csc_data
    dense = torch.randn(3, 2, dtype=torch.float32)
    check_sparse_op(lambda c, r, v, d: torch.sparse.mm(torch.sparse_csc_tensor(c, r, v, size), d), device, ccol, row, val, dense)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sparse_csc_conversions(device, dtype, csc_data):
    ccol, row, val, size = csc_data
    check_sparse_op(lambda c, r, v: torch.sparse_csc_tensor(c, r, v, size).to_sparse_csr(), device, ccol, row, val)

# ──────────────────────────────────────────────────────────────────────
# BSR Layout Tests
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def bsr_data():
    crow = torch.tensor([0, 1, 2], dtype=torch.int64)
    col = torch.tensor([0, 1], dtype=torch.int64)
    val = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float32)
    size = (4, 4)
    return crow, col, val, size

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sparse_bsr_creation(device, dtype, bsr_data):
    crow, col, val, size = bsr_data
    check_sparse_op(lambda r, c, v: torch.sparse_bsr_tensor(r, c, v, size).to_dense(), device, crow, col, val)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sparse_bsr_conversions(device, dtype, bsr_data):
    crow, col, val, size = bsr_data
    check_sparse_op(lambda r, c, v: torch.sparse_bsr_tensor(r, c, v, size).to_sparse_bsc((2, 2)), device, crow, col, val)

# ──────────────────────────────────────────────────────────────────────
# BSC Layout Tests
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def bsc_data():
    ccol = torch.tensor([0, 1, 2], dtype=torch.int64)
    row = torch.tensor([0, 1], dtype=torch.int64)
    val = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float32)
    size = (4, 4)
    return ccol, row, val, size

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sparse_bsc_creation(device, dtype, bsc_data):
    ccol, row, val, size = bsc_data
    check_sparse_op(lambda c, r, v: torch.sparse_bsc_tensor(c, r, v, size).to_dense(), device, ccol, row, val)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sparse_bsc_conversions(device, dtype, bsc_data):
    ccol, row, val, size = bsc_data
    check_sparse_op(lambda c, r, v: torch.sparse_bsc_tensor(c, r, v, size).to_sparse_bsr((2, 2)), device, ccol, row, val)
