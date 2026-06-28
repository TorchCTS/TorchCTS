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
from torchcts.core.layout_tracker import LayoutDispatchTracker

SHAPE_DTYPES = [torch.float32, torch.int64, torch.bool]

@pytest.mark.smoke
@pytest.mark.covers("aten::permute")
@pytest.mark.covers("aten::transpose.int")
@pytest.mark.covers("aten::view")
@pytest.mark.parametrize("dtype", SHAPE_DTYPES)
def test_view_reshape_permute_transpose(dtype, device, manifest, compare, input_gen):
    shape = (2, 3, 4)
    x_dev = input_gen(shape, dtype, device)
    x_cpu = x_dev.cpu()
    
    # view
    compare(x_dev.view(2, 12), x_cpu.view(2, 12), category="exact", dtype=dtype)
    # reshape
    compare(x_dev.reshape(6, 4), x_cpu.reshape(6, 4), category="exact", dtype=dtype)
    # permute
    compare(x_dev.permute(2, 0, 1), x_cpu.permute(2, 0, 1), category="exact", dtype=dtype)
    # transpose
    compare(x_dev.transpose(0, 2), x_cpu.transpose(0, 2), category="exact", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.covers("aten::expand")
@pytest.mark.covers("aten::select.int")
@pytest.mark.covers("aten::slice.Tensor")
@pytest.mark.covers("aten::split.Tensor")
@pytest.mark.parametrize("dtype", SHAPE_DTYPES)
def test_expand_narrow_select_chunk_split(dtype, device, manifest, compare, input_gen):
    # expand: (1, 4) -> (3, 4)
    x_dev = input_gen((1, 4), dtype, device)
    compare(x_dev.expand(3, 4), x_dev.cpu().expand(3, 4), category="exact", dtype=dtype)
    
    # narrow
    y_dev = input_gen((10, 10), dtype, device)
    compare(y_dev.narrow(0, 2, 5), y_dev.cpu().narrow(0, 2, 5), category="exact", dtype=dtype)
    
    # select
    compare(y_dev.select(1, 3), y_dev.cpu().select(1, 3), category="exact", dtype=dtype)
    
    # chunk
    z_dev = input_gen((12, 12), dtype, device)
    chunks_dev = z_dev.chunk(3, dim=0)
    chunks_cpu = z_dev.cpu().chunk(3, dim=0)
    for cd, cc in zip(chunks_dev, chunks_cpu):
        compare(cd, cc, category="exact", dtype=dtype)
        
    # split
    splits_dev = z_dev.split(4, dim=1)
    splits_cpu = z_dev.cpu().split(4, dim=1)
    for sd, sc in zip(splits_dev, splits_cpu):
        compare(sd, sc, category="exact", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.covers_category("layout_copy_detection")
@pytest.mark.parametrize("shape", [(4, 4)])
def test_layout_tracker_warning(shape, device, manifest):
    x_dev = torch.randn(*shape, device=device)
    x_non_cont = x_dev.T
    
    with pytest.warns(UserWarning, match="Silent copy to contiguous detected"):
        with LayoutDispatchTracker() as tracker:
            cloned = x_non_cont.contiguous()
            assert tracker.copy_count == 1


@pytest.mark.smoke
@pytest.mark.covers("aten::cat")
@pytest.mark.parametrize("dtype", SHAPE_DTYPES)
def test_cat_basic(dtype, device, compare, input_gen):
    """torch.cat along dim 0 and dim 1."""
    a = input_gen((4, 8), dtype, device)
    b = input_gen((6, 8), dtype, device)
    res = torch.cat([a, b], dim=0)
    expected = torch.cat([a.cpu(), b.cpu()], dim=0)
    synchronize(device)
    compare(res, expected, category="exact", dtype=dtype)

    # dim=1
    c = input_gen((4, 3), dtype, device)
    d = input_gen((4, 5), dtype, device)
    res1 = torch.cat([c, d], dim=1)
    expected1 = torch.cat([c.cpu(), d.cpu()], dim=1)
    synchronize(device)
    compare(res1, expected1, category="exact", dtype=dtype)


@pytest.mark.smoke
@pytest.mark.covers("aten::cat")
@pytest.mark.parametrize("dtype", SHAPE_DTYPES)
def test_cat_empty_tensor(dtype, device, compare, input_gen):
    """torch.cat with an empty tensor in the list."""
    a = input_gen((4, 8), dtype, device)
    empty = torch.empty(0, 8, dtype=dtype, device=device)
    res = torch.cat([empty, a], dim=0)
    expected = torch.cat([empty.cpu(), a.cpu()], dim=0)
    synchronize(device)
    compare(res, expected, category="exact", dtype=dtype)
    assert res.shape == (4, 8)


@pytest.mark.smoke
@pytest.mark.covers("aten::cat")
@pytest.mark.parametrize("dtype", SHAPE_DTYPES)
def test_cat_negative_dim(dtype, device, compare, input_gen):
    """torch.cat with negative dim index."""
    a = input_gen((4, 3), dtype, device)
    b = input_gen((4, 5), dtype, device)
    res = torch.cat([a, b], dim=-1)
    expected = torch.cat([a.cpu(), b.cpu()], dim=-1)
    synchronize(device)
    compare(res, expected, category="exact", dtype=dtype)
    assert res.shape == (4, 8)


@pytest.mark.smoke
@pytest.mark.covers("aten::stack")
@pytest.mark.parametrize("dtype", SHAPE_DTYPES)
def test_stack_basic(dtype, device, compare, input_gen):
    """torch.stack along dim 0 and dim 1."""
    tensors = [input_gen((4, 4), dtype, device) for _ in range(3)]
    res = torch.stack(tensors, dim=0)
    expected = torch.stack([t.cpu() for t in tensors], dim=0)
    synchronize(device)
    compare(res, expected, category="exact", dtype=dtype)
    assert res.shape == (3, 4, 4)

    # dim=1
    res1 = torch.stack(tensors, dim=1)
    expected1 = torch.stack([t.cpu() for t in tensors], dim=1)
    synchronize(device)
    compare(res1, expected1, category="exact", dtype=dtype)
    assert res1.shape == (4, 3, 4)


@pytest.mark.smoke
@pytest.mark.covers("aten::stack")
@pytest.mark.parametrize("dtype", SHAPE_DTYPES)
def test_stack_negative_dim(dtype, device, compare, input_gen):
    """torch.stack with negative dim."""
    tensors = [input_gen((4, 4), dtype, device) for _ in range(3)]
    res = torch.stack(tensors, dim=-1)
    expected = torch.stack([t.cpu() for t in tensors], dim=-1)
    synchronize(device)
    compare(res, expected, category="exact", dtype=dtype)
    assert res.shape == (4, 4, 3)


@pytest.mark.smoke
@pytest.mark.covers("aten::unbind.int")
@pytest.mark.parametrize("dtype", SHAPE_DTYPES)
def test_unbind(dtype, device, compare, input_gen):
    """torch.unbind along dim 0 and dim 1."""
    x = input_gen((4, 8), dtype, device)
    parts_dev = torch.unbind(x, dim=0)
    parts_cpu = torch.unbind(x.cpu(), dim=0)
    assert len(parts_dev) == 4
    for pd, pc in zip(parts_dev, parts_cpu):
        compare(pd, pc, category="exact", dtype=dtype)

    # dim=1
    parts_dev1 = torch.unbind(x, dim=1)
    parts_cpu1 = torch.unbind(x.cpu(), dim=1)
    assert len(parts_dev1) == 8
    for pd, pc in zip(parts_dev1, parts_cpu1):
        compare(pd, pc, category="exact", dtype=dtype)


@pytest.mark.smoke
@pytest.mark.covers("aten::split.Tensor")
@pytest.mark.parametrize("dtype", SHAPE_DTYPES)
def test_split_uneven(dtype, device, compare, input_gen):
    """torch.split with a size that doesn't evenly divide the dim."""
    x = input_gen((10, 8), dtype, device)
    # 10 / 3 = [3, 3, 3, 1]
    splits_dev = x.split(3, dim=0)
    splits_cpu = x.cpu().split(3, dim=0)
    assert len(splits_dev) == 4
    assert splits_dev[-1].shape[0] == 1
    for sd, sc in zip(splits_dev, splits_cpu):
        compare(sd, sc, category="exact", dtype=dtype)


@pytest.mark.smoke
@pytest.mark.covers("aten::split.Tensor")
@pytest.mark.parametrize("dtype", SHAPE_DTYPES)
def test_chunk_negative_dim(dtype, device, compare, input_gen):
    """torch.chunk with negative dim."""
    x = input_gen((12, 12), dtype, device)
    chunks_dev = x.chunk(3, dim=-1)
    chunks_cpu = x.cpu().chunk(3, dim=-1)
    assert len(chunks_dev) == 3
    for cd, cc in zip(chunks_dev, chunks_cpu):
        compare(cd, cc, category="exact", dtype=dtype)


@pytest.mark.smoke
@pytest.mark.covers("aten::_has_same_storage_numel")
@pytest.mark.covers("aten::resize")
@pytest.mark.covers("aten::resize.out", surface="out_variant")
@pytest.mark.covers("aten::resize_")
@pytest.mark.covers("aten::resize_as")
@pytest.mark.covers("aten::resize_as.out", surface="out_variant")
@pytest.mark.covers("aten::resize_as_")
@pytest.mark.covers("aten::_efficientzerotensor.out", surface="out_variant")
@pytest.mark.covers("aten::_to_cpu")
def test_storage_resize_and_cpu_transfer_dispatcher_surfaces(device, compare):
    base_cpu = torch.arange(6, dtype=torch.float32)
    base_dev = base_cpu.to(device)
    assert torch.ops.aten._has_same_storage_numel.default(
        base_dev, base_dev.view(2, 3),
    ) == torch.ops.aten._has_same_storage_numel.default(base_cpu, base_cpu.view(2, 3))
    assert torch.ops.aten._has_same_storage_numel.default(
        base_dev, torch.arange(7, dtype=torch.float32, device=device),
    ) == torch.ops.aten._has_same_storage_numel.default(
        base_cpu, torch.arange(7, dtype=torch.float32),
    )

    resized_cpu = torch.ops.aten.resize.default(torch.ones(2, 2), [2, 3])
    resized_dev = torch.ops.aten.resize.default(torch.ones(2, 2, device=device), [2, 3])
    synchronize(device)
    assert resized_dev.shape == resized_cpu.shape
    assert resized_dev.device.type == device

    resize_out_cpu = torch.empty(0)
    resize_out_dev = torch.empty(0, device=device)
    returned_cpu = torch.ops.aten.resize.out(
        torch.ones(2, 2), [2, 3], out=resize_out_cpu,
    )
    returned_dev = torch.ops.aten.resize.out(
        torch.ones(2, 2, device=device), [2, 3], out=resize_out_dev,
    )
    synchronize(device)
    assert returned_cpu is resize_out_cpu
    assert returned_dev is resize_out_dev
    assert resize_out_dev.shape == resize_out_cpu.shape

    inplace_cpu = torch.ones(2, 2)
    inplace_dev = torch.ones(2, 2, device=device)
    returned_cpu = torch.ops.aten.resize_.default(inplace_cpu, [2, 3])
    returned_dev = torch.ops.aten.resize_.default(inplace_dev, [2, 3])
    synchronize(device)
    assert returned_cpu is inplace_cpu
    assert returned_dev is inplace_dev
    assert inplace_dev.shape == inplace_cpu.shape

    template_cpu = torch.empty(3, 4)
    template_dev = torch.empty(3, 4, device=device)
    resized_as_cpu = torch.ops.aten.resize_as.default(torch.ones(2, 2), template_cpu)
    resized_as_dev = torch.ops.aten.resize_as.default(
        torch.ones(2, 2, device=device), template_dev,
    )
    synchronize(device)
    assert resized_as_dev.shape == resized_as_cpu.shape

    resize_as_out_cpu = torch.empty(0)
    resize_as_out_dev = torch.empty(0, device=device)
    returned_cpu = torch.ops.aten.resize_as.out(
        torch.ones(2, 2), template_cpu, out=resize_as_out_cpu,
    )
    returned_dev = torch.ops.aten.resize_as.out(
        torch.ones(2, 2, device=device), template_dev, out=resize_as_out_dev,
    )
    synchronize(device)
    assert returned_cpu is resize_as_out_cpu
    assert returned_dev is resize_as_out_dev
    assert resize_as_out_dev.shape == resize_as_out_cpu.shape

    inplace_as_cpu = torch.ones(2, 2)
    inplace_as_dev = torch.ones(2, 2, device=device)
    returned_cpu = torch.ops.aten.resize_as_.default(inplace_as_cpu, template_cpu)
    returned_dev = torch.ops.aten.resize_as_.default(inplace_as_dev, template_dev)
    synchronize(device)
    assert returned_cpu is inplace_as_cpu
    assert returned_dev is inplace_as_dev
    assert inplace_as_dev.shape == inplace_as_cpu.shape

    zero_out_cpu = torch.empty(0)
    zero_out_dev = torch.empty(0, device=device)
    returned_cpu = torch.ops.aten._efficientzerotensor.out([2, 3], out=zero_out_cpu)
    returned_dev = torch.ops.aten._efficientzerotensor.out([2, 3], out=zero_out_dev)
    synchronize(device)
    assert returned_cpu is zero_out_cpu
    assert returned_dev is zero_out_dev
    compare(zero_out_dev, zero_out_cpu, category="exact", dtype=torch.float32)

    cpu_results = torch.ops.aten._to_cpu.default(
        [torch.arange(3, dtype=torch.float32, device=device), torch.ones(2, device=device)],
    )
    assert all(result.device.type == "cpu" for result in cpu_results)
    torch.testing.assert_close(cpu_results[0], torch.arange(3, dtype=torch.float32))
    torch.testing.assert_close(cpu_results[1], torch.ones(2))


@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::_nnz")
def test_sparse_nnz_dispatcher_surface(device):
    indices_cpu = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    values_cpu = torch.tensor([2.0, 3.0], dtype=torch.float32)
    sparse_cpu = torch.sparse_coo_tensor(indices_cpu, values_cpu, (2, 2))
    sparse_dev = torch.sparse_coo_tensor(
        indices_cpu.to(device), values_cpu.to(device), (2, 2), device=device,
    )
    synchronize(device)
    assert torch.ops.aten._nnz.default(sparse_dev) == torch.ops.aten._nnz.default(sparse_cpu)
