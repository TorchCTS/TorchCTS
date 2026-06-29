# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest
import torch
from torchcts.core.device import synchronize

# Helper to compare dense tensors
def compare_dense(cpu_t, dev_t_cpu):
    __tracebackhide__ = True
    assert cpu_t.shape == dev_t_cpu.shape, f"Shape mismatch: {cpu_t.shape} vs {dev_t_cpu.shape}"
    assert cpu_t.dtype == dev_t_cpu.dtype, f"Dtype mismatch: {cpu_t.dtype} vs {dev_t_cpu.dtype}"
    assert torch.allclose(cpu_t, dev_t_cpu, rtol=1e-4, atol=1e-4), "Value mismatch"

# Helper to run a sparse operation on CPU vs Device.
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
            if any(term in err_msg for term in ["not implemented", "not supported", "requires compiling", "without mkl"]):
                pytest.fail(
                    f"CPU raised {type(cpu_exception).__name__}, but backend reported unsupported: {dev_e}"
                )
            return
        except Exception:
            return
        pytest.fail(f"CPU raised {type(cpu_exception).__name__}: {cpu_exception}, but backend did not raise")
    else:
        try:
            dev_out = op_fn(*dev_args, **dev_kwargs)
            synchronize(device)
        except Exception as dev_e:
            pytest.fail(f"CPU succeeded, but backend raised {type(dev_e).__name__}: {dev_e}")
            
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
@pytest.mark.covers("aten::sparse_coo_tensor.indices_size")
def test_sparse_coo_creation(device, coo_data):
    i, v, size = coo_data
    check_sparse_op(lambda i_t, v_t: torch.sparse_coo_tensor(i_t, v_t, size).to_dense(), device, i, v)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::coalesce")
def test_sparse_coo_coalesce(device, coo_data):
    i, v, size = coo_data
    check_sparse_op(lambda i_t, v_t: torch.sparse_coo_tensor(i_t, v_t, size).coalesce(), device, i, v)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::t")
def test_sparse_coo_transpose(device, coo_data):
    i, v, size = coo_data
    check_sparse_op(lambda i_t, v_t: torch.sparse_coo_tensor(i_t, v_t, size).t(), device, i, v)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::add.Tensor")
def test_sparse_coo_add(device, coo_data):
    i, v, size = coo_data
    check_sparse_op(lambda i_t, v_t: torch.sparse_coo_tensor(i_t, v_t, size) + torch.sparse_coo_tensor(i_t, v_t, size), device, i, v)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::_sparse_mm")
def test_sparse_coo_mm(device, coo_data):
    i, v, size = coo_data
    dense = torch.randn(3, 2, dtype=torch.float32)
    check_sparse_op(lambda i_t, v_t, d: torch.sparse.mm(torch.sparse_coo_tensor(i_t, v_t, size), d), device, i, v, dense)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::to_sparse_csr")
@pytest.mark.covers("aten::to_sparse_csc")
@pytest.mark.covers("aten::to_sparse_bsr")
def test_sparse_coo_conversions(device, coo_data):
    i, v, size = coo_data
    check_sparse_op(lambda i_t, v_t: torch.sparse_coo_tensor(i_t, v_t, size).to_sparse_csr(), device, i, v)
    check_sparse_op(lambda i_t, v_t: torch.sparse_coo_tensor(i_t, v_t, size).to_sparse_csc(), device, i, v)
    check_sparse_op(lambda i_t, v_t: torch.sparse_coo_tensor(i_t, v_t, size).to_sparse_bsr((1, 1)), device, i, v)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::indices")
@pytest.mark.covers("aten::values")
@pytest.mark.covers("aten::is_coalesced")
@pytest.mark.covers("aten::sparse_dim")
def test_sparse_coo_public_accessors(device, coo_data):
    i, v, size = coo_data

    def op(i_t, v_t):
        sparse = torch.sparse_coo_tensor(i_t, v_t, size).coalesce()
        return sparse.indices(), sparse.values(), sparse.is_coalesced(), sparse.sparse_dim()

    check_sparse_op(op, device, i, v)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::_indices")
@pytest.mark.covers("aten::_values")
def test_sparse_coo_raw_accessors(device, coo_data):
    i, v, size = coo_data

    def op(i_t, v_t):
        sparse = torch.sparse_coo_tensor(i_t, v_t, size)
        return sparse._indices(), sparse._values()

    check_sparse_op(op, device, i, v)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::_sparse_sum")
@pytest.mark.covers("aten::_sparse_sum.dim")
@pytest.mark.covers("aten::_sparse_softmax.int")
@pytest.mark.covers("aten::_sparse_log_softmax.int")
def test_sparse_coo_reductions_and_softmax(device, coo_data):
    i, v, size = coo_data

    def op_sum(i_t, v_t):
        sparse = torch.sparse_coo_tensor(i_t, v_t, size).coalesce()
        return torch.sparse.sum(sparse), torch.sparse.sum(sparse, dim=1)

    def op_softmax(i_t, v_t):
        sparse = torch.sparse_coo_tensor(i_t, v_t, size).coalesce()
        return torch.sparse.softmax(sparse, dim=1), torch.sparse.log_softmax(sparse, dim=1)

    check_sparse_op(op_sum, device, i, v)
    check_sparse_op(op_softmax, device, i, v)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::_sparse_addmm")
def test_sparse_addmm(device, coo_data):
    i, v, size = coo_data
    dense_a = torch.randn(3, 2, dtype=torch.float32)
    dense_b = torch.randn(2, 3, dtype=torch.float32)
    check_sparse_op(
        lambda i_t, v_t, a, b: torch.sparse.addmm(
            torch.sparse_coo_tensor(i_t, v_t, size),
            a,
            b,
        ),
        device,
        i,
        v,
        dense_a,
        dense_b,
    )

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::sparse_mask")
def test_sparse_mask(device, coo_data):
    i, v, size = coo_data
    dense = torch.randn(size, dtype=torch.float32)
    check_sparse_op(
        lambda i_t, v_t, d: d.sparse_mask(torch.sparse_coo_tensor(i_t, v_t, size).coalesce()),
        device,
        i,
        v,
        dense,
    )

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
@pytest.mark.covers("aten::sparse_csr_tensor.crow_col_value_size")
def test_sparse_csr_creation(device, csr_data):
    crow, col, val, size = csr_data
    check_sparse_op(lambda r, c, v: torch.sparse_csr_tensor(r, c, v, size).to_dense(), device, crow, col, val)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::transpose.int")
def test_sparse_csr_transpose(device, csr_data):
    crow, col, val, size = csr_data
    check_sparse_op(lambda r, c, v: torch.sparse_csr_tensor(r, c, v, size).transpose(0, 1), device, crow, col, val)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::_sparse_mm")
def test_sparse_csr_mm(device, csr_data):
    crow, col, val, size = csr_data
    dense = torch.randn(3, 2, dtype=torch.float32)
    check_sparse_op(lambda r, c, v, d: torch.sparse.mm(torch.sparse_csr_tensor(r, c, v, size), d), device, crow, col, val, dense)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::_to_sparse")
def test_sparse_csr_conversions(device, csr_data):
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
@pytest.mark.covers("aten::crow_indices")
@pytest.mark.covers("aten::col_indices")
@pytest.mark.covers("aten::values")
def test_sparse_csr_accessors(device, csr_data):
    crow, col, val, size = csr_data

    def op(r, c, v):
        sparse = torch.sparse_csr_tensor(r, c, v, size)
        return sparse.crow_indices(), sparse.col_indices(), sparse.values()

    check_sparse_op(op, device, crow, col, val)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::sparse_csc_tensor.ccol_row_value_size")
def test_sparse_csc_creation(device, csc_data):
    ccol, row, val, size = csc_data
    check_sparse_op(lambda c, r, v: torch.sparse_csc_tensor(c, r, v, size).to_dense(), device, ccol, row, val)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::transpose.int")
def test_sparse_csc_transpose(device, csc_data):
    ccol, row, val, size = csc_data
    check_sparse_op(lambda c, r, v: torch.sparse_csc_tensor(c, r, v, size).transpose(0, 1), device, ccol, row, val)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::_sparse_addmm")
def test_sparse_csc_mm(device, csc_data):
    ccol, row, val, size = csc_data
    dense = torch.randn(3, 2, dtype=torch.float32)
    check_sparse_op(lambda c, r, v, d: torch.sparse.mm(torch.sparse_csc_tensor(c, r, v, size), d), device, ccol, row, val, dense)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::_to_sparse_csr")
def test_sparse_csc_conversions(device, csc_data):
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
@pytest.mark.covers("aten::ccol_indices")
@pytest.mark.covers("aten::row_indices")
@pytest.mark.covers("aten::values")
def test_sparse_csc_accessors(device, csc_data):
    ccol, row, val, size = csc_data

    def op(c, r, v):
        sparse = torch.sparse_csc_tensor(c, r, v, size)
        return sparse.ccol_indices(), sparse.row_indices(), sparse.values()

    check_sparse_op(op, device, ccol, row, val)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::sparse_bsr_tensor.crow_col_value_size")
def test_sparse_bsr_creation(device, bsr_data):
    crow, col, val, size = bsr_data
    check_sparse_op(lambda r, c, v: torch.sparse_bsr_tensor(r, c, v, size).to_dense(), device, crow, col, val)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::to_sparse_bsc")
def test_sparse_bsr_conversions(device, bsr_data):
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
@pytest.mark.covers("aten::sparse_bsc_tensor.ccol_row_value_size")
def test_sparse_bsc_creation(device, bsc_data):
    ccol, row, val, size = bsc_data
    check_sparse_op(lambda c, r, v: torch.sparse_bsc_tensor(c, r, v, size).to_dense(), device, ccol, row, val)

@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::to_sparse_bsr")
def test_sparse_bsc_conversions(device, bsc_data):
    ccol, row, val, size = bsc_data
    check_sparse_op(lambda c, r, v: torch.sparse_bsc_tensor(c, r, v, size).to_sparse_bsr((2, 2)), device, ccol, row, val)


def _backend_sparse_call(fn, opname):
    try:
        return fn()
    except Exception as exc:
        pytest.fail(f"{opname} failed on backend: {type(exc).__name__}: {exc}")


def _assert_copy_accessor(op, sparse_cpu, sparse_dev, device, compare):
    opname = str(op)
    expected = op(sparse_cpu)
    actual = _backend_sparse_call(lambda: op(sparse_dev), opname)
    synchronize(device)
    compare(actual, expected, category="exact", dtype=expected.dtype)

    expected_out = torch.empty_like(expected)
    actual_out = torch.empty_like(actual)
    expected_return = op.out(sparse_cpu, out=expected_out)
    actual_return = _backend_sparse_call(lambda: op.out(sparse_dev, out=actual_out), f"{opname}.out")
    synchronize(device)
    assert expected_return is expected_out
    assert actual_return is actual_out
    compare(actual_out, expected_out, category="exact", dtype=expected_out.dtype)


def _empty_sparse_coo(size, device):
    sparse_dim = len(size)
    indices = torch.empty((sparse_dim, 0), dtype=torch.int64, device=device)
    values = torch.empty((0,), dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, size)


def _assert_sparse_tensor_equal(actual, expected, compare):
    assert actual.layout == expected.layout
    assert tuple(actual.shape) == tuple(expected.shape)
    compare(actual.to_dense(), expected.to_dense(), category="exact", dtype=expected.dtype)
    if actual.layout == torch.sparse_coo:
        assert actual.is_coalesced() == expected.is_coalesced()
        actual_coalesced = actual.coalesce()
        expected_coalesced = expected.coalesce()
        compare(actual_coalesced.indices(), expected_coalesced.indices(), category="exact", dtype=torch.int64)
        compare(actual_coalesced.values(), expected_coalesced.values(), category="exact", dtype=expected_coalesced.values().dtype)


def _assert_sparse_out(op_call, out_cpu, out_dev, device, compare):
    expected_return = op_call(out_cpu, "cpu")
    actual_return = _backend_sparse_call(lambda: op_call(out_dev, device), "sparse out variant")
    synchronize(device)
    assert expected_return is out_cpu
    assert actual_return is out_dev
    _assert_sparse_tensor_equal(out_dev, out_cpu, compare)


@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::values_copy")
@pytest.mark.covers("aten::values_copy.out")
@pytest.mark.covers("aten::crow_indices_copy")
@pytest.mark.covers("aten::crow_indices_copy.out")
@pytest.mark.covers("aten::col_indices_copy")
@pytest.mark.covers("aten::col_indices_copy.out")
@pytest.mark.covers("aten::ccol_indices_copy")
@pytest.mark.covers("aten::ccol_indices_copy.out")
@pytest.mark.covers("aten::row_indices_copy")
@pytest.mark.covers("aten::row_indices_copy.out")
def test_sparse_compressed_copy_accessors(device, csr_data, csc_data, bsr_data, bsc_data, compare):
    crow, col, val, csr_size = csr_data
    csr_cpu = torch.sparse_csr_tensor(crow, col, val, csr_size)
    csr_dev = torch.sparse_csr_tensor(crow.to(device), col.to(device), val.to(device), csr_size)

    ccol, row, val, csc_size = csc_data
    csc_cpu = torch.sparse_csc_tensor(ccol, row, val, csc_size)
    csc_dev = torch.sparse_csc_tensor(ccol.to(device), row.to(device), val.to(device), csc_size)

    crow, col, val, bsr_size = bsr_data
    bsr_cpu = torch.sparse_bsr_tensor(crow, col, val, bsr_size)
    bsr_dev = torch.sparse_bsr_tensor(crow.to(device), col.to(device), val.to(device), bsr_size)

    ccol, row, val, bsc_size = bsc_data
    bsc_cpu = torch.sparse_bsc_tensor(ccol, row, val, bsc_size)
    bsc_dev = torch.sparse_bsc_tensor(ccol.to(device), row.to(device), val.to(device), bsc_size)

    for sparse_cpu, sparse_dev in (
        (csr_cpu, csr_dev),
        (csc_cpu, csc_dev),
        (bsr_cpu, bsr_dev),
        (bsc_cpu, bsc_dev),
    ):
        _assert_copy_accessor(torch.ops.aten.values_copy, sparse_cpu, sparse_dev, device, compare)

    for op, sparse_cpu, sparse_dev in (
        (torch.ops.aten.crow_indices_copy, csr_cpu, csr_dev),
        (torch.ops.aten.col_indices_copy, csr_cpu, csr_dev),
        (torch.ops.aten.ccol_indices_copy, csc_cpu, csc_dev),
        (torch.ops.aten.row_indices_copy, csc_cpu, csc_dev),
        (torch.ops.aten.crow_indices_copy, bsr_cpu, bsr_dev),
        (torch.ops.aten.col_indices_copy, bsr_cpu, bsr_dev),
        (torch.ops.aten.ccol_indices_copy, bsc_cpu, bsc_dev),
        (torch.ops.aten.row_indices_copy, bsc_cpu, bsc_dev),
    ):
        _assert_copy_accessor(op, sparse_cpu, sparse_dev, device, compare)


@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::_coalesce")
@pytest.mark.covers("aten::_coalesce.out")
@pytest.mark.covers("aten::_coalesced")
@pytest.mark.covers("aten::_coalesced.out")
@pytest.mark.covers("aten::_coalesced_")
@pytest.mark.covers("aten::_convert_indices_from_coo_to_csr")
@pytest.mark.covers("aten::_convert_indices_from_coo_to_csr.out")
@pytest.mark.covers("aten::_convert_indices_from_csr_to_coo")
@pytest.mark.covers("aten::_convert_indices_from_csr_to_coo.out")
def test_sparse_low_level_coalesce_and_index_conversion(device, coo_data, compare):
    indices, values, size = coo_data
    cpu_sparse = torch.sparse_coo_tensor(indices, values, size)
    dev_sparse = torch.sparse_coo_tensor(indices.to(device), values.to(device), size)

    expected = torch.ops.aten._coalesce(cpu_sparse)
    actual = _backend_sparse_call(lambda: torch.ops.aten._coalesce(dev_sparse), "aten::_coalesce")
    synchronize(device)
    _assert_sparse_tensor_equal(actual, expected, compare)

    out_cpu = _empty_sparse_coo(size, "cpu")
    out_dev = _empty_sparse_coo(size, device)
    expected_return = torch.ops.aten._coalesce.out(cpu_sparse, out=out_cpu)
    actual_return = _backend_sparse_call(
        lambda: torch.ops.aten._coalesce.out(dev_sparse, out=out_dev),
        "aten::_coalesce.out",
    )
    synchronize(device)
    assert expected_return is out_cpu
    assert actual_return is out_dev
    _assert_sparse_tensor_equal(out_dev, out_cpu, compare)

    expected = torch.ops.aten._coalesced(cpu_sparse, True)
    actual = _backend_sparse_call(lambda: torch.ops.aten._coalesced(dev_sparse, True), "aten::_coalesced")
    synchronize(device)
    _assert_sparse_tensor_equal(actual, expected, compare)

    out_cpu = _empty_sparse_coo(size, "cpu")
    out_dev = _empty_sparse_coo(size, device)
    expected_return = torch.ops.aten._coalesced.out(cpu_sparse, True, out=out_cpu)
    actual_return = _backend_sparse_call(
        lambda: torch.ops.aten._coalesced.out(dev_sparse, True, out=out_dev),
        "aten::_coalesced.out",
    )
    synchronize(device)
    assert expected_return is out_cpu
    assert actual_return is out_dev
    _assert_sparse_tensor_equal(out_dev, out_cpu, compare)

    mutable_cpu = torch.sparse_coo_tensor(indices, values, size)
    mutable_dev = torch.sparse_coo_tensor(indices.to(device), values.to(device), size)
    returned_cpu = torch.ops.aten._coalesced_(mutable_cpu, True)
    returned_dev = _backend_sparse_call(
        lambda: torch.ops.aten._coalesced_(mutable_dev, True),
        "aten::_coalesced_",
    )
    synchronize(device)
    assert returned_cpu is mutable_cpu
    assert returned_dev is mutable_dev
    _assert_sparse_tensor_equal(mutable_dev, mutable_cpu, compare)

    coo_rows_cpu = indices[0]
    coo_rows_dev = indices[0].to(device)
    expected_crow = torch.ops.aten._convert_indices_from_coo_to_csr(coo_rows_cpu, size[0])
    actual_crow = _backend_sparse_call(
        lambda: torch.ops.aten._convert_indices_from_coo_to_csr(coo_rows_dev, size[0]),
        "aten::_convert_indices_from_coo_to_csr",
    )
    synchronize(device)
    compare(actual_crow, expected_crow, category="exact", dtype=torch.int64)

    expected_crow_out = torch.empty_like(expected_crow)
    actual_crow_out = torch.empty_like(actual_crow)
    expected_return = torch.ops.aten._convert_indices_from_coo_to_csr.out(coo_rows_cpu, size[0], out=expected_crow_out)
    actual_return = _backend_sparse_call(
        lambda: torch.ops.aten._convert_indices_from_coo_to_csr.out(coo_rows_dev, size[0], out=actual_crow_out),
        "aten::_convert_indices_from_coo_to_csr.out",
    )
    synchronize(device)
    assert expected_return is expected_crow_out
    assert actual_return is actual_crow_out
    compare(actual_crow_out, expected_crow_out, category="exact", dtype=torch.int64)

    col_cpu = indices[1]
    col_dev = indices[1].to(device)
    expected_coo = torch.ops.aten._convert_indices_from_csr_to_coo(expected_crow, col_cpu)
    actual_coo = _backend_sparse_call(
        lambda: torch.ops.aten._convert_indices_from_csr_to_coo(actual_crow, col_dev),
        "aten::_convert_indices_from_csr_to_coo",
    )
    synchronize(device)
    compare(actual_coo, expected_coo, category="exact", dtype=torch.int64)

    expected_coo_out = torch.empty_like(expected_coo)
    actual_coo_out = torch.empty_like(actual_coo)
    expected_return = torch.ops.aten._convert_indices_from_csr_to_coo.out(expected_crow, col_cpu, out=expected_coo_out)
    actual_return = _backend_sparse_call(
        lambda: torch.ops.aten._convert_indices_from_csr_to_coo.out(actual_crow, col_dev, out=actual_coo_out),
        "aten::_convert_indices_from_csr_to_coo.out",
    )
    synchronize(device)
    assert expected_return is expected_coo_out
    assert actual_return is actual_coo_out
    compare(actual_coo_out, expected_coo_out, category="exact", dtype=torch.int64)


@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::sparse_coo_tensor.indices")
@pytest.mark.covers("aten::sparse_coo_tensor.size")
@pytest.mark.covers("aten::sparse_coo_tensor.size_out")
@pytest.mark.covers("aten::_sparse_coo_tensor_with_dims")
@pytest.mark.covers("aten::_sparse_coo_tensor_with_dims.out")
@pytest.mark.covers("aten::_sparse_coo_tensor_with_dims_and_tensors")
@pytest.mark.covers("aten::_sparse_coo_tensor_with_dims_and_tensors.out")
@pytest.mark.covers("aten::sparse_csr_tensor.crow_col_value")
@pytest.mark.covers("aten::sparse_csc_tensor.ccol_row_value")
@pytest.mark.covers("aten::sparse_bsr_tensor.crow_col_value")
@pytest.mark.covers("aten::sparse_bsc_tensor.ccol_row_value")
def test_sparse_low_level_constructor_overloads(device, coo_data, csr_data, csc_data, bsr_data, bsc_data, compare):
    indices, values, size = coo_data
    expected = torch.ops.aten.sparse_coo_tensor.indices(indices, values)
    actual = _backend_sparse_call(
        lambda: torch.ops.aten.sparse_coo_tensor.indices(indices.to(device), values.to(device)),
        "aten::sparse_coo_tensor.indices",
    )
    synchronize(device)
    _assert_sparse_tensor_equal(actual, expected, compare)

    device_obj = torch.device(device)
    expected = torch.ops.aten.sparse_coo_tensor.size(
        list(size),
        dtype=torch.float32,
        layout=torch.sparse_coo,
        device=torch.device("cpu"),
        pin_memory=False,
    )
    actual = _backend_sparse_call(
        lambda: torch.ops.aten.sparse_coo_tensor.size(
            list(size),
            dtype=torch.float32,
            layout=torch.sparse_coo,
            device=device_obj,
            pin_memory=False,
        ),
        "aten::sparse_coo_tensor.size",
    )
    synchronize(device)
    _assert_sparse_tensor_equal(actual, expected, compare)

    _assert_sparse_out(
        lambda out, _target: torch.ops.aten.sparse_coo_tensor.size_out(list(size), out=out),
        _empty_sparse_coo(size, "cpu"),
        _empty_sparse_coo(size, device),
        device,
        compare,
    )

    expected = torch.ops.aten._sparse_coo_tensor_with_dims(
        2,
        0,
        list(size),
        dtype=torch.float32,
        layout=torch.sparse_coo,
        device=torch.device("cpu"),
        pin_memory=False,
    )
    actual = _backend_sparse_call(
        lambda: torch.ops.aten._sparse_coo_tensor_with_dims(
            2,
            0,
            list(size),
            dtype=torch.float32,
            layout=torch.sparse_coo,
            device=device_obj,
            pin_memory=False,
        ),
        "aten::_sparse_coo_tensor_with_dims",
    )
    synchronize(device)
    _assert_sparse_tensor_equal(actual, expected, compare)

    _assert_sparse_out(
        lambda out, _target: torch.ops.aten._sparse_coo_tensor_with_dims.out(2, 0, list(size), out=out),
        _empty_sparse_coo(size, "cpu"),
        _empty_sparse_coo(size, device),
        device,
        compare,
    )

    expected = torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors(
        2,
        0,
        list(size),
        indices,
        values,
        dtype=torch.float32,
        layout=torch.sparse_coo,
        device=torch.device("cpu"),
        pin_memory=False,
        is_coalesced=False,
    )
    actual = _backend_sparse_call(
        lambda: torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors(
            2,
            0,
            list(size),
            indices.to(device),
            values.to(device),
            dtype=torch.float32,
            layout=torch.sparse_coo,
            device=device_obj,
            pin_memory=False,
            is_coalesced=False,
        ),
        "aten::_sparse_coo_tensor_with_dims_and_tensors",
    )
    synchronize(device)
    _assert_sparse_tensor_equal(actual, expected, compare)

    _assert_sparse_out(
        lambda out, target: torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors.out(
            2,
            0,
            list(size),
            indices.to(target),
            values.to(target),
            is_coalesced=False,
            out=out,
        ),
        _empty_sparse_coo(size, "cpu"),
        _empty_sparse_coo(size, device),
        device,
        compare,
    )

    crow, col, val, _ = csr_data
    expected = torch.ops.aten.sparse_csr_tensor.crow_col_value(crow, col, val)
    actual = _backend_sparse_call(
        lambda: torch.ops.aten.sparse_csr_tensor.crow_col_value(crow.to(device), col.to(device), val.to(device)),
        "aten::sparse_csr_tensor.crow_col_value",
    )
    synchronize(device)
    _assert_sparse_tensor_equal(actual, expected, compare)

    ccol, row, val, _ = csc_data
    expected = torch.ops.aten.sparse_csc_tensor.ccol_row_value(ccol, row, val)
    actual = _backend_sparse_call(
        lambda: torch.ops.aten.sparse_csc_tensor.ccol_row_value(ccol.to(device), row.to(device), val.to(device)),
        "aten::sparse_csc_tensor.ccol_row_value",
    )
    synchronize(device)
    _assert_sparse_tensor_equal(actual, expected, compare)

    crow, col, val, _ = bsr_data
    expected = torch.ops.aten.sparse_bsr_tensor.crow_col_value(crow, col, val)
    actual = _backend_sparse_call(
        lambda: torch.ops.aten.sparse_bsr_tensor.crow_col_value(crow.to(device), col.to(device), val.to(device)),
        "aten::sparse_bsr_tensor.crow_col_value",
    )
    synchronize(device)
    _assert_sparse_tensor_equal(actual, expected, compare)

    ccol, row, val, _ = bsc_data
    expected = torch.ops.aten.sparse_bsc_tensor.ccol_row_value(ccol, row, val)
    actual = _backend_sparse_call(
        lambda: torch.ops.aten.sparse_bsc_tensor.ccol_row_value(ccol.to(device), row.to(device), val.to(device)),
        "aten::sparse_bsc_tensor.ccol_row_value",
    )
    synchronize(device)
    _assert_sparse_tensor_equal(actual, expected, compare)


@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::_validate_sparse_coo_tensor_args")
@pytest.mark.covers("aten::_validate_sparse_csr_tensor_args")
@pytest.mark.covers("aten::_validate_sparse_csc_tensor_args")
@pytest.mark.covers("aten::_validate_sparse_bsr_tensor_args")
@pytest.mark.covers("aten::_validate_sparse_bsc_tensor_args")
def test_sparse_low_level_validation_helpers(device, coo_data, csr_data, csc_data, bsr_data, bsc_data):
    indices, values, size = coo_data
    assert torch.ops.aten._validate_sparse_coo_tensor_args(indices, values, list(size), False, None) is None
    assert _backend_sparse_call(
        lambda: torch.ops.aten._validate_sparse_coo_tensor_args(
            indices.to(device),
            values.to(device),
            list(size),
            False,
            None,
        ),
        "aten::_validate_sparse_coo_tensor_args",
    ) is None

    crow, col, values, size = csr_data
    assert torch.ops.aten._validate_sparse_csr_tensor_args(crow, col, values, list(size), None) is None
    assert _backend_sparse_call(
        lambda: torch.ops.aten._validate_sparse_csr_tensor_args(
            crow.to(device),
            col.to(device),
            values.to(device),
            list(size),
            None,
        ),
        "aten::_validate_sparse_csr_tensor_args",
    ) is None

    ccol, row, values, size = csc_data
    assert torch.ops.aten._validate_sparse_csc_tensor_args(ccol, row, values, list(size), None) is None
    assert _backend_sparse_call(
        lambda: torch.ops.aten._validate_sparse_csc_tensor_args(
            ccol.to(device),
            row.to(device),
            values.to(device),
            list(size),
            None,
        ),
        "aten::_validate_sparse_csc_tensor_args",
    ) is None

    crow, col, values, size = bsr_data
    assert torch.ops.aten._validate_sparse_bsr_tensor_args(crow, col, values, list(size), None) is None
    assert _backend_sparse_call(
        lambda: torch.ops.aten._validate_sparse_bsr_tensor_args(
            crow.to(device),
            col.to(device),
            values.to(device),
            list(size),
            None,
        ),
        "aten::_validate_sparse_bsr_tensor_args",
    ) is None

    ccol, row, values, size = bsc_data
    assert torch.ops.aten._validate_sparse_bsc_tensor_args(ccol, row, values, list(size), None) is None
    assert _backend_sparse_call(
        lambda: torch.ops.aten._validate_sparse_bsc_tensor_args(
            ccol.to(device),
            row.to(device),
            values.to(device),
            list(size),
            None,
        ),
        "aten::_validate_sparse_bsc_tensor_args",
    ) is None


@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::_to_dense")
@pytest.mark.covers("aten::_to_dense.out")
@pytest.mark.covers("aten::_to_sparse.out")
@pytest.mark.covers("aten::_to_sparse.sparse_dim")
@pytest.mark.covers("aten::_to_sparse.sparse_dim_out")
@pytest.mark.covers("aten::_to_sparse_csr.out")
@pytest.mark.covers("aten::_to_sparse_csc")
@pytest.mark.covers("aten::_to_sparse_csc.out")
@pytest.mark.covers("aten::_to_sparse_bsr")
@pytest.mark.covers("aten::_to_sparse_bsr.out")
@pytest.mark.covers("aten::_to_sparse_bsc")
@pytest.mark.covers("aten::_to_sparse_bsc.out")
def test_sparse_low_level_dense_conversion_variants(device, compare):
    dense_cpu = torch.tensor([[0.0, 3.0, 0.0], [4.0, 0.0, 5.0]], dtype=torch.float32)
    dense_dev = dense_cpu.to(device)
    coo_cpu = dense_cpu.to_sparse()
    coo_dev = dense_dev.to_sparse()

    expected_dense = torch.ops.aten._to_dense(coo_cpu, None, None)
    actual_dense = _backend_sparse_call(lambda: torch.ops.aten._to_dense(coo_dev, None, None), "aten::_to_dense")
    synchronize(device)
    compare(actual_dense, expected_dense, category="exact", dtype=torch.float32)

    expected_dense_out = torch.empty_like(dense_cpu)
    actual_dense_out = torch.empty_like(dense_dev)
    expected_return = torch.ops.aten._to_dense.out(coo_cpu, None, None, out=expected_dense_out)
    actual_return = _backend_sparse_call(
        lambda: torch.ops.aten._to_dense.out(coo_dev, None, None, out=actual_dense_out),
        "aten::_to_dense.out",
    )
    synchronize(device)
    assert expected_return is expected_dense_out
    assert actual_return is actual_dense_out
    compare(actual_dense_out, expected_dense_out, category="exact", dtype=torch.float32)

    expected_sparse = torch.ops.aten._to_sparse.sparse_dim(dense_cpu, 2)
    actual_sparse = _backend_sparse_call(
        lambda: torch.ops.aten._to_sparse.sparse_dim(dense_dev, 2),
        "aten::_to_sparse.sparse_dim",
    )
    synchronize(device)
    _assert_sparse_tensor_equal(actual_sparse, expected_sparse, compare)

    _assert_sparse_out(
        lambda out, target: torch.ops.aten._to_sparse.sparse_dim_out(dense_cpu.to(target), 2, out=out),
        _empty_sparse_coo(tuple(dense_cpu.shape), "cpu"),
        _empty_sparse_coo(tuple(dense_cpu.shape), device),
        device,
        compare,
    )

    _assert_sparse_out(
        lambda out, target: torch.ops.aten._to_sparse.out(
            dense_cpu.to(target),
            layout=torch.sparse_coo,
            out=out,
        ),
        _empty_sparse_coo(tuple(dense_cpu.shape), "cpu"),
        _empty_sparse_coo(tuple(dense_cpu.shape), device),
        device,
        compare,
    )

    for opname, functional, out_call in (
        (
            "aten::_to_sparse_csc",
            lambda value: torch.ops.aten._to_sparse_csc(value, None),
            lambda value, out: torch.ops.aten._to_sparse_csc.out(value, out=out),
        ),
        (
            "aten::_to_sparse_bsr",
            lambda value: torch.ops.aten._to_sparse_bsr(value, [1, 1], None),
            lambda value, out: torch.ops.aten._to_sparse_bsr.out(value, [1, 1], out=out),
        ),
        (
            "aten::_to_sparse_bsc",
            lambda value: torch.ops.aten._to_sparse_bsc(value, [1, 1], None),
            lambda value, out: torch.ops.aten._to_sparse_bsc.out(value, [1, 1], out=out),
        ),
    ):
        expected = functional(dense_cpu)
        actual = _backend_sparse_call(lambda: functional(dense_dev), opname)
        synchronize(device)
        _assert_sparse_tensor_equal(actual, expected, compare)

        expected_out = expected.clone()
        actual_out = actual.clone()
        expected_return = out_call(dense_cpu, expected_out)
        actual_return = _backend_sparse_call(lambda: out_call(dense_dev, actual_out), f"{opname}.out")
        synchronize(device)
        assert expected_return is expected_out
        assert actual_return is actual_out
        _assert_sparse_tensor_equal(actual_out, expected_out, compare)

    expected_csr = dense_cpu.to_sparse_csr()
    actual_csr = dense_dev.to_sparse_csr()
    expected_csr_out = expected_csr.clone()
    actual_csr_out = actual_csr.clone()
    expected_return = torch.ops.aten._to_sparse_csr.out(dense_cpu, out=expected_csr_out)
    actual_return = _backend_sparse_call(
        lambda: torch.ops.aten._to_sparse_csr.out(dense_dev, out=actual_csr_out),
        "aten::_to_sparse_csr.out",
    )
    synchronize(device)
    assert expected_return is expected_csr_out
    assert actual_return is actual_csr_out
    _assert_sparse_tensor_equal(actual_csr_out, expected_csr_out, compare)


@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::_sparse_compressed_tensor_with_dims")
@pytest.mark.covers("aten::sparse_compressed_tensor.comp_plain_value")
@pytest.mark.covers("aten::sparse_compressed_tensor.comp_plain_value_size")
@pytest.mark.covers("aten::_validate_compressed_sparse_indices")
@pytest.mark.covers("aten::_validate_sparse_compressed_tensor_args")
def test_sparse_low_level_compressed_generic_helpers(device, csr_data, compare):
    crow, col, values, size = csr_data
    device_obj = torch.device(device)

    expected = torch.ops.aten.sparse_compressed_tensor.comp_plain_value(
        crow,
        col,
        values,
        layout=torch.sparse_csr,
        device=torch.device("cpu"),
    )
    actual = _backend_sparse_call(
        lambda: torch.ops.aten.sparse_compressed_tensor.comp_plain_value(
            crow.to(device),
            col.to(device),
            values.to(device),
            layout=torch.sparse_csr,
            device=device_obj,
        ),
        "aten::sparse_compressed_tensor.comp_plain_value",
    )
    synchronize(device)
    _assert_sparse_tensor_equal(actual, expected, compare)

    expected = torch.ops.aten.sparse_compressed_tensor.comp_plain_value_size(
        crow,
        col,
        values,
        list(size),
        layout=torch.sparse_csr,
        device=torch.device("cpu"),
    )
    actual = _backend_sparse_call(
        lambda: torch.ops.aten.sparse_compressed_tensor.comp_plain_value_size(
            crow.to(device),
            col.to(device),
            values.to(device),
            list(size),
            layout=torch.sparse_csr,
            device=device_obj,
        ),
        "aten::sparse_compressed_tensor.comp_plain_value_size",
    )
    synchronize(device)
    _assert_sparse_tensor_equal(actual, expected, compare)

    expected = torch.ops.aten._sparse_compressed_tensor_with_dims(
        int(values.numel()),
        0,
        list(size),
        [],
        torch.int64,
        dtype=torch.float32,
        layout=torch.sparse_csr,
        device=torch.device("cpu"),
        pin_memory=False,
    )
    actual = _backend_sparse_call(
        lambda: torch.ops.aten._sparse_compressed_tensor_with_dims(
            int(values.numel()),
            0,
            list(size),
            [],
            torch.int64,
            dtype=torch.float32,
            layout=torch.sparse_csr,
            device=device_obj,
            pin_memory=False,
        ),
        "aten::_sparse_compressed_tensor_with_dims",
    )
    synchronize(device)
    _assert_sparse_tensor_equal(actual, expected, compare)

    assert torch.ops.aten._validate_compressed_sparse_indices(True, crow, col, size[0], size[1], int(values.numel())) is None
    assert _backend_sparse_call(
        lambda: torch.ops.aten._validate_compressed_sparse_indices(
            True,
            crow.to(device),
            col.to(device),
            size[0],
            size[1],
            int(values.numel()),
        ),
        "aten::_validate_compressed_sparse_indices",
    ) is None

    assert torch.ops.aten._validate_sparse_compressed_tensor_args(crow, col, values, list(size), torch.sparse_csr, None) is None
    assert _backend_sparse_call(
        lambda: torch.ops.aten._validate_sparse_compressed_tensor_args(
            crow.to(device),
            col.to(device),
            values.to(device),
            list(size),
            torch.sparse_csr,
            None,
        ),
        "aten::_validate_sparse_compressed_tensor_args",
    ) is None


@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::_sparse_sum.dtype")
@pytest.mark.covers("aten::_sparse_sum.dim_dtype")
@pytest.mark.covers("aten::_sparse_sum.dim_out")
@pytest.mark.covers("aten::_sparse_csr_sum.dim_dtype")
@pytest.mark.covers("aten::_sparse_csr_sum.dim_dtype_out")
@pytest.mark.covers("aten::_sparse_csr_prod.dim_dtype")
@pytest.mark.covers("aten::_sparse_csr_prod.dim_dtype_out")
@pytest.mark.covers("aten::_sparse_softmax")
@pytest.mark.covers("aten::_sparse_softmax.out")
@pytest.mark.covers("aten::_sparse_log_softmax")
@pytest.mark.covers("aten::_sparse_log_softmax.out")
def test_sparse_low_level_reduction_and_softmax_variants(device, compare):
    dense_cpu = torch.tensor([[0.0, 3.0, 0.0], [4.0, 0.0, 5.0]], dtype=torch.float32)
    dense_dev = dense_cpu.to(device)
    coo_cpu = dense_cpu.to_sparse().coalesce()
    coo_dev = dense_dev.to_sparse().coalesce()
    csr_cpu = dense_cpu.to_sparse_csr()
    csr_dev = dense_dev.to_sparse_csr()

    expected_scalar = torch.ops.aten._sparse_sum.dtype(coo_cpu, dtype=torch.float32)
    actual_scalar = _backend_sparse_call(
        lambda: torch.ops.aten._sparse_sum.dtype(coo_dev, dtype=torch.float32),
        "aten::_sparse_sum.dtype",
    )
    synchronize(device)
    compare(actual_scalar, expected_scalar, category="exact", dtype=torch.float32)

    expected_sum = torch.ops.aten._sparse_sum.dim_dtype(coo_cpu, [1], dtype=torch.float32)
    actual_sum = _backend_sparse_call(
        lambda: torch.ops.aten._sparse_sum.dim_dtype(coo_dev, [1], dtype=torch.float32),
        "aten::_sparse_sum.dim_dtype",
    )
    synchronize(device)
    _assert_sparse_tensor_equal(actual_sum, expected_sum, compare)

    expected_sum_out = expected_sum.clone()
    actual_sum_out = actual_sum.clone()
    expected_return = torch.ops.aten._sparse_sum.dim_out(coo_cpu, [1], out=expected_sum_out)
    actual_return = _backend_sparse_call(
        lambda: torch.ops.aten._sparse_sum.dim_out(coo_dev, [1], out=actual_sum_out),
        "aten::_sparse_sum.dim_out",
    )
    synchronize(device)
    assert expected_return is expected_sum_out
    assert actual_return is actual_sum_out
    _assert_sparse_tensor_equal(actual_sum_out, expected_sum_out, compare)

    for opname, functional, out_op in (
        (
            "aten::_sparse_csr_sum.dim_dtype",
            torch.ops.aten._sparse_csr_sum.dim_dtype,
            torch.ops.aten._sparse_csr_sum.dim_dtype_out,
        ),
        (
            "aten::_sparse_csr_prod.dim_dtype",
            torch.ops.aten._sparse_csr_prod.dim_dtype,
            torch.ops.aten._sparse_csr_prod.dim_dtype_out,
        ),
    ):
        expected = functional(csr_cpu, [1], True, dtype=torch.float32)
        actual = _backend_sparse_call(
            lambda: functional(csr_dev, [1], True, dtype=torch.float32),
            opname,
        )
        synchronize(device)
        _assert_sparse_tensor_equal(actual, expected, compare)

        expected_out = expected.clone()
        actual_out = actual.clone()
        expected_return = out_op(csr_cpu, [1], True, dtype=torch.float32, out=expected_out)
        actual_return = _backend_sparse_call(
            lambda: out_op(csr_dev, [1], True, dtype=torch.float32, out=actual_out),
            f"{opname}_out",
        )
        synchronize(device)
        assert expected_return is expected_out
        assert actual_return is actual_out
        _assert_sparse_tensor_equal(actual_out, expected_out, compare)

    for opname, functional, out_op in (
        ("aten::_sparse_softmax", torch.ops.aten._sparse_softmax.default, torch.ops.aten._sparse_softmax.out),
        ("aten::_sparse_log_softmax", torch.ops.aten._sparse_log_softmax.default, torch.ops.aten._sparse_log_softmax.out),
    ):
        expected = functional(coo_cpu, 1, False)
        actual = _backend_sparse_call(lambda: functional(coo_dev, 1, False), opname)
        synchronize(device)
        _assert_sparse_tensor_equal(actual, expected, compare)

        expected_out = expected.clone()
        actual_out = actual.clone()
        expected_return = out_op(coo_cpu, 1, False, out=expected_out)
        actual_return = _backend_sparse_call(
            lambda: out_op(coo_dev, 1, False, out=actual_out),
            f"{opname}.out",
        )
        synchronize(device)
        assert expected_return is expected_out
        assert actual_return is actual_out
        _assert_sparse_tensor_equal(actual_out, expected_out, compare)


@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::_sparse_addmm.out")
@pytest.mark.covers("aten::_sparse_sparse_matmul")
@pytest.mark.covers("aten::_sparse_sparse_matmul.out")
@pytest.mark.covers("aten::sparse_sampled_addmm")
@pytest.mark.covers("aten::sparse_sampled_addmm.out")
@pytest.mark.covers("aten::sspaddmm")
@pytest.mark.covers("aten::sspaddmm.out")
@pytest.mark.covers("aten::hspmm")
@pytest.mark.covers("aten::hspmm.out")
@pytest.mark.covers("aten::_spdiags")
@pytest.mark.covers("aten::_spdiags.out")
@pytest.mark.covers("aten::sparse_mask.out")
@pytest.mark.covers("aten::_sparse_mask_projection")
@pytest.mark.covers("aten::_sparse_mask_projection.out")
def test_sparse_low_level_math_and_mask_variants(device, compare):
    dense_cpu = torch.tensor([[0.0, 3.0, 0.0], [4.0, 0.0, 5.0]], dtype=torch.float32)
    dense_dev = dense_cpu.to(device)
    coo_cpu = dense_cpu.to_sparse().coalesce()
    coo_dev = dense_dev.to_sparse().coalesce()
    csr_cpu = dense_cpu.to_sparse_csr()
    csr_dev = dense_dev.to_sparse_csr()
    rhs_cpu = torch.ones(3, 2, dtype=torch.float32)
    rhs_dev = rhs_cpu.to(device)

    expected_addmm_out = torch.empty(2, 2, dtype=torch.float32)
    actual_addmm_out = torch.empty(2, 2, dtype=torch.float32, device=device)
    expected_return = torch.ops.aten._sparse_addmm.out(torch.zeros(2, 2), coo_cpu, rhs_cpu, out=expected_addmm_out)
    actual_return = _backend_sparse_call(
        lambda: torch.ops.aten._sparse_addmm.out(
            torch.zeros(2, 2, device=device),
            coo_dev,
            rhs_dev,
            out=actual_addmm_out,
        ),
        "aten::_sparse_addmm.out",
    )
    synchronize(device)
    assert expected_return is expected_addmm_out
    assert actual_return is actual_addmm_out
    compare(actual_addmm_out, expected_addmm_out, category="exact", dtype=torch.float32)

    expected_sparse_mm = torch.ops.aten._sparse_sparse_matmul(coo_cpu, coo_cpu.t())
    actual_sparse_mm = _backend_sparse_call(
        lambda: torch.ops.aten._sparse_sparse_matmul(coo_dev, coo_dev.t()),
        "aten::_sparse_sparse_matmul",
    )
    synchronize(device)
    _assert_sparse_tensor_equal(actual_sparse_mm, expected_sparse_mm, compare)

    expected_sparse_mm_out = expected_sparse_mm.clone()
    actual_sparse_mm_out = actual_sparse_mm.clone()
    expected_return = torch.ops.aten._sparse_sparse_matmul.out(coo_cpu, coo_cpu.t(), out=expected_sparse_mm_out)
    actual_return = _backend_sparse_call(
        lambda: torch.ops.aten._sparse_sparse_matmul.out(coo_dev, coo_dev.t(), out=actual_sparse_mm_out),
        "aten::_sparse_sparse_matmul.out",
    )
    synchronize(device)
    assert expected_return is expected_sparse_mm_out
    assert actual_return is actual_sparse_mm_out
    _assert_sparse_tensor_equal(actual_sparse_mm_out, expected_sparse_mm_out, compare)

    sampled_lhs_cpu = torch.ones(2, 4, dtype=torch.float32)
    sampled_rhs_cpu = torch.ones(4, 3, dtype=torch.float32)
    sampled_lhs_dev = sampled_lhs_cpu.to(device)
    sampled_rhs_dev = sampled_rhs_cpu.to(device)
    expected_sampled = torch.ops.aten.sparse_sampled_addmm(csr_cpu, sampled_lhs_cpu, sampled_rhs_cpu)
    actual_sampled = _backend_sparse_call(
        lambda: torch.ops.aten.sparse_sampled_addmm(csr_dev, sampled_lhs_dev, sampled_rhs_dev),
        "aten::sparse_sampled_addmm",
    )
    synchronize(device)
    _assert_sparse_tensor_equal(actual_sampled, expected_sampled, compare)

    expected_sampled_out = expected_sampled.clone()
    actual_sampled_out = actual_sampled.clone()
    expected_return = torch.ops.aten.sparse_sampled_addmm.out(csr_cpu, sampled_lhs_cpu, sampled_rhs_cpu, out=expected_sampled_out)
    actual_return = _backend_sparse_call(
        lambda: torch.ops.aten.sparse_sampled_addmm.out(csr_dev, sampled_lhs_dev, sampled_rhs_dev, out=actual_sampled_out),
        "aten::sparse_sampled_addmm.out",
    )
    synchronize(device)
    assert expected_return is expected_sampled_out
    assert actual_return is actual_sampled_out
    _assert_sparse_tensor_equal(actual_sampled_out, expected_sampled_out, compare)

    self_22_cpu = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32).to_sparse().coalesce()
    self_22_dev = self_22_cpu.to(device)
    expected_sspaddmm = torch.ops.aten.sspaddmm(self_22_cpu, coo_cpu, rhs_cpu)
    actual_sspaddmm = _backend_sparse_call(
        lambda: torch.ops.aten.sspaddmm(self_22_dev, coo_dev, rhs_dev),
        "aten::sspaddmm",
    )
    synchronize(device)
    _assert_sparse_tensor_equal(actual_sspaddmm, expected_sspaddmm, compare)

    expected_sspaddmm_out = expected_sspaddmm.clone()
    actual_sspaddmm_out = actual_sspaddmm.clone()
    expected_return = torch.ops.aten.sspaddmm.out(self_22_cpu, coo_cpu, rhs_cpu, out=expected_sspaddmm_out)
    actual_return = _backend_sparse_call(
        lambda: torch.ops.aten.sspaddmm.out(self_22_dev, coo_dev, rhs_dev, out=actual_sspaddmm_out),
        "aten::sspaddmm.out",
    )
    synchronize(device)
    assert expected_return is expected_sspaddmm_out
    assert actual_return is actual_sspaddmm_out
    _assert_sparse_tensor_equal(actual_sspaddmm_out, expected_sspaddmm_out, compare)

    expected_hspmm = torch.ops.aten.hspmm(coo_cpu, rhs_cpu)
    actual_hspmm = _backend_sparse_call(lambda: torch.ops.aten.hspmm(coo_dev, rhs_dev), "aten::hspmm")
    synchronize(device)
    _assert_sparse_tensor_equal(actual_hspmm, expected_hspmm, compare)

    expected_hspmm_out = expected_hspmm.clone()
    actual_hspmm_out = actual_hspmm.clone()
    expected_return = torch.ops.aten.hspmm.out(coo_cpu, rhs_cpu, out=expected_hspmm_out)
    actual_return = _backend_sparse_call(
        lambda: torch.ops.aten.hspmm.out(coo_dev, rhs_dev, out=actual_hspmm_out),
        "aten::hspmm.out",
    )
    synchronize(device)
    assert expected_return is expected_hspmm_out
    assert actual_return is actual_hspmm_out
    _assert_sparse_tensor_equal(actual_hspmm_out, expected_hspmm_out, compare)

    diagonals_cpu = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    offsets_cpu = torch.tensor([0, 1], dtype=torch.int64)
    diagonals_dev = diagonals_cpu.to(device)
    offsets_dev = offsets_cpu.to(device)
    expected_spdiags = torch.ops.aten._spdiags(diagonals_cpu, offsets_cpu, [3, 3], torch.sparse_coo)
    actual_spdiags = _backend_sparse_call(
        lambda: torch.ops.aten._spdiags(diagonals_dev, offsets_dev, [3, 3], torch.sparse_coo),
        "aten::_spdiags",
    )
    synchronize(device)
    _assert_sparse_tensor_equal(actual_spdiags, expected_spdiags, compare)

    expected_spdiags_out = expected_spdiags.clone()
    actual_spdiags_out = actual_spdiags.clone()
    expected_return = torch.ops.aten._spdiags.out(
        diagonals_cpu,
        offsets_cpu,
        [3, 3],
        torch.sparse_coo,
        out=expected_spdiags_out,
    )
    actual_return = _backend_sparse_call(
        lambda: torch.ops.aten._spdiags.out(
            diagonals_dev,
            offsets_dev,
            [3, 3],
            torch.sparse_coo,
            out=actual_spdiags_out,
        ),
        "aten::_spdiags.out",
    )
    synchronize(device)
    assert expected_return is expected_spdiags_out
    assert actual_return is actual_spdiags_out
    _assert_sparse_tensor_equal(actual_spdiags_out, expected_spdiags_out, compare)

    mask_cpu = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32).to_sparse().coalesce()
    mask_dev = mask_cpu.to(device)
    expected_mask_out = mask_cpu.clone()
    actual_mask_out = mask_dev.clone()
    expected_return = torch.ops.aten.sparse_mask.out(dense_cpu, mask_cpu, out=expected_mask_out)
    actual_return = _backend_sparse_call(
        lambda: torch.ops.aten.sparse_mask.out(dense_dev, mask_dev, out=actual_mask_out),
        "aten::sparse_mask.out",
    )
    synchronize(device)
    assert expected_return is expected_mask_out
    assert actual_return is actual_mask_out
    _assert_sparse_tensor_equal(actual_mask_out, expected_mask_out, compare)

    expected_projection = torch.ops.aten._sparse_mask_projection(coo_cpu, mask_cpu, False)
    actual_projection = _backend_sparse_call(
        lambda: torch.ops.aten._sparse_mask_projection(coo_dev, mask_dev, False),
        "aten::_sparse_mask_projection",
    )
    synchronize(device)
    _assert_sparse_tensor_equal(actual_projection, expected_projection, compare)

    expected_projection_out = expected_projection.clone()
    actual_projection_out = actual_projection.clone()
    expected_return = torch.ops.aten._sparse_mask_projection.out(coo_cpu, mask_cpu, False, out=expected_projection_out)
    actual_return = _backend_sparse_call(
        lambda: torch.ops.aten._sparse_mask_projection.out(coo_dev, mask_dev, False, out=actual_projection_out),
        "aten::_sparse_mask_projection.out",
    )
    synchronize(device)
    assert expected_return is expected_projection_out
    assert actual_return is actual_projection_out
    _assert_sparse_tensor_equal(actual_projection_out, expected_projection_out, compare)


@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::_sparse_broadcast_to")
@pytest.mark.covers("aten::_sparse_mm.reduce")
@pytest.mark.covers("aten::_sparse_mm_reduce_impl")
@pytest.mark.covers("aten::copy_sparse_to_sparse")
@pytest.mark.covers("aten::copy_sparse_to_sparse.out")
@pytest.mark.covers("aten::copy_sparse_to_sparse_")
@pytest.mark.covers("aten::resize_as_sparse")
@pytest.mark.covers("aten::resize_as_sparse.out")
@pytest.mark.covers("aten::resize_as_sparse_")
@pytest.mark.covers("aten::sparse_resize")
@pytest.mark.covers("aten::sparse_resize.out")
@pytest.mark.covers("aten::sparse_resize_")
@pytest.mark.covers("aten::sparse_resize_and_clear")
@pytest.mark.covers("aten::sparse_resize_and_clear.out")
@pytest.mark.covers("aten::sparse_resize_and_clear_")
def test_sparse_low_level_copy_resize_and_reduce_helpers(device, compare):
    dense_cpu = torch.tensor([[0.0, 3.0, 0.0], [4.0, 0.0, 5.0]], dtype=torch.float32)
    dense_dev = dense_cpu.to(device)
    coo_cpu = dense_cpu.to_sparse().coalesce()
    coo_dev = dense_dev.to_sparse().coalesce()
    csr_cpu = dense_cpu.to_sparse_csr()
    csr_dev = dense_dev.to_sparse_csr()
    rhs_cpu = torch.ones(3, 2, dtype=torch.float32)
    rhs_dev = rhs_cpu.to(device)

    expected_broadcast = torch.ops.aten._sparse_broadcast_to(coo_cpu, [2, 3])
    actual_broadcast = _backend_sparse_call(
        lambda: torch.ops.aten._sparse_broadcast_to(coo_dev, [2, 3]),
        "aten::_sparse_broadcast_to",
    )
    synchronize(device)
    _assert_sparse_tensor_equal(actual_broadcast, expected_broadcast, compare)

    expected_reduce = torch.ops.aten._sparse_mm.reduce(csr_cpu, rhs_cpu, "sum")
    actual_reduce = _backend_sparse_call(
        lambda: torch.ops.aten._sparse_mm.reduce(csr_dev, rhs_dev, "sum"),
        "aten::_sparse_mm.reduce",
    )
    synchronize(device)
    compare(actual_reduce, expected_reduce, category="exact", dtype=torch.float32)

    expected_values, expected_arg = torch.ops.aten._sparse_mm_reduce_impl(csr_cpu, rhs_cpu, "sum")
    actual_values, actual_arg = _backend_sparse_call(
        lambda: torch.ops.aten._sparse_mm_reduce_impl(csr_dev, rhs_dev, "sum"),
        "aten::_sparse_mm_reduce_impl",
    )
    synchronize(device)
    compare(actual_values, expected_values, category="exact", dtype=torch.float32)
    compare(actual_arg, expected_arg, category="exact", dtype=torch.int64)

    dst_cpu = torch.zeros_like(dense_cpu).to_sparse().coalesce()
    dst_dev = torch.zeros_like(dense_dev).to_sparse().coalesce()
    expected_copy = torch.ops.aten.copy_sparse_to_sparse(dst_cpu, coo_cpu, False)
    actual_copy = _backend_sparse_call(
        lambda: torch.ops.aten.copy_sparse_to_sparse(dst_dev, coo_dev, False),
        "aten::copy_sparse_to_sparse",
    )
    synchronize(device)
    _assert_sparse_tensor_equal(actual_copy, expected_copy, compare)

    expected_copy_out = expected_copy.clone()
    actual_copy_out = actual_copy.clone()
    expected_return = torch.ops.aten.copy_sparse_to_sparse.out(dst_cpu, coo_cpu, False, out=expected_copy_out)
    actual_return = _backend_sparse_call(
        lambda: torch.ops.aten.copy_sparse_to_sparse.out(dst_dev, coo_dev, False, out=actual_copy_out),
        "aten::copy_sparse_to_sparse.out",
    )
    synchronize(device)
    assert expected_return is expected_copy_out
    assert actual_return is actual_copy_out
    _assert_sparse_tensor_equal(actual_copy_out, expected_copy_out, compare)

    inplace_cpu = dst_cpu.clone()
    inplace_dev = dst_dev.clone()
    expected_return = torch.ops.aten.copy_sparse_to_sparse_(inplace_cpu, coo_cpu, False)
    actual_return = _backend_sparse_call(
        lambda: torch.ops.aten.copy_sparse_to_sparse_(inplace_dev, coo_dev, False),
        "aten::copy_sparse_to_sparse_",
    )
    synchronize(device)
    assert expected_return is inplace_cpu
    assert actual_return is inplace_dev
    _assert_sparse_tensor_equal(inplace_dev, inplace_cpu, compare)

    empty_cpu = _empty_sparse_coo((2, 3), "cpu")
    empty_dev = _empty_sparse_coo((2, 3), device)
    expected_resize_as = torch.ops.aten.resize_as_sparse(empty_cpu, coo_cpu)
    actual_resize_as = _backend_sparse_call(
        lambda: torch.ops.aten.resize_as_sparse(empty_dev, coo_dev),
        "aten::resize_as_sparse",
    )
    synchronize(device)
    _assert_sparse_tensor_equal(actual_resize_as, expected_resize_as, compare)

    expected_resize_as_out = _empty_sparse_coo((2, 3), "cpu")
    actual_resize_as_out = _empty_sparse_coo((2, 3), device)
    expected_return = torch.ops.aten.resize_as_sparse.out(empty_cpu, coo_cpu, out=expected_resize_as_out)
    actual_return = _backend_sparse_call(
        lambda: torch.ops.aten.resize_as_sparse.out(empty_dev, coo_dev, out=actual_resize_as_out),
        "aten::resize_as_sparse.out",
    )
    synchronize(device)
    assert expected_return is expected_resize_as_out
    assert actual_return is actual_resize_as_out
    _assert_sparse_tensor_equal(actual_resize_as_out, expected_resize_as_out, compare)

    inplace_cpu = _empty_sparse_coo((2, 3), "cpu")
    inplace_dev = _empty_sparse_coo((2, 3), device)
    expected_return = torch.ops.aten.resize_as_sparse_(inplace_cpu, coo_cpu)
    actual_return = _backend_sparse_call(
        lambda: torch.ops.aten.resize_as_sparse_(inplace_dev, coo_dev),
        "aten::resize_as_sparse_",
    )
    synchronize(device)
    assert expected_return is inplace_cpu
    assert actual_return is inplace_dev
    _assert_sparse_tensor_equal(inplace_dev, inplace_cpu, compare)

    expected_resize = torch.ops.aten.sparse_resize(empty_cpu, [4, 5], 2, 0)
    actual_resize = _backend_sparse_call(
        lambda: torch.ops.aten.sparse_resize(empty_dev, [4, 5], 2, 0),
        "aten::sparse_resize",
    )
    synchronize(device)
    _assert_sparse_tensor_equal(actual_resize, expected_resize, compare)

    expected_resize_out = _empty_sparse_coo((4, 5), "cpu")
    actual_resize_out = _empty_sparse_coo((4, 5), device)
    expected_return = torch.ops.aten.sparse_resize.out(empty_cpu, [4, 5], 2, 0, out=expected_resize_out)
    actual_return = _backend_sparse_call(
        lambda: torch.ops.aten.sparse_resize.out(empty_dev, [4, 5], 2, 0, out=actual_resize_out),
        "aten::sparse_resize.out",
    )
    synchronize(device)
    assert expected_return is expected_resize_out
    assert actual_return is actual_resize_out
    _assert_sparse_tensor_equal(actual_resize_out, expected_resize_out, compare)

    inplace_cpu = _empty_sparse_coo((2, 3), "cpu")
    inplace_dev = _empty_sparse_coo((2, 3), device)
    expected_return = torch.ops.aten.sparse_resize_(inplace_cpu, [4, 5], 2, 0)
    actual_return = _backend_sparse_call(
        lambda: torch.ops.aten.sparse_resize_(inplace_dev, [4, 5], 2, 0),
        "aten::sparse_resize_",
    )
    synchronize(device)
    assert expected_return is inplace_cpu
    assert actual_return is inplace_dev
    _assert_sparse_tensor_equal(inplace_dev, inplace_cpu, compare)

    expected_clear = torch.ops.aten.sparse_resize_and_clear(coo_cpu, [4, 5], 2, 0)
    actual_clear = _backend_sparse_call(
        lambda: torch.ops.aten.sparse_resize_and_clear(coo_dev, [4, 5], 2, 0),
        "aten::sparse_resize_and_clear",
    )
    synchronize(device)
    _assert_sparse_tensor_equal(actual_clear, expected_clear, compare)

    expected_clear_out = _empty_sparse_coo((4, 5), "cpu")
    actual_clear_out = _empty_sparse_coo((4, 5), device)
    expected_return = torch.ops.aten.sparse_resize_and_clear.out(coo_cpu, [4, 5], 2, 0, out=expected_clear_out)
    actual_return = _backend_sparse_call(
        lambda: torch.ops.aten.sparse_resize_and_clear.out(coo_dev, [4, 5], 2, 0, out=actual_clear_out),
        "aten::sparse_resize_and_clear.out",
    )
    synchronize(device)
    assert expected_return is expected_clear_out
    assert actual_return is actual_clear_out
    _assert_sparse_tensor_equal(actual_clear_out, expected_clear_out, compare)

    inplace_cpu = coo_cpu.clone()
    inplace_dev = coo_dev.clone()
    expected_return = torch.ops.aten.sparse_resize_and_clear_(inplace_cpu, [4, 5], 2, 0)
    actual_return = _backend_sparse_call(
        lambda: torch.ops.aten.sparse_resize_and_clear_(inplace_dev, [4, 5], 2, 0),
        "aten::sparse_resize_and_clear_",
    )
    synchronize(device)
    assert expected_return is inplace_cpu
    assert actual_return is inplace_dev
    _assert_sparse_tensor_equal(inplace_dev, inplace_cpu, compare)
