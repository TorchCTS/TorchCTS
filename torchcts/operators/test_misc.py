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

MISC_DTYPES = [torch.float32, torch.int64]


def _compare_tuple(actual, expected, compare, category="exact", dtype=torch.float32):
    assert len(actual) == len(expected)
    for actual_tensor, expected_tensor in zip(actual, expected):
        synchronize(actual_tensor.device.type)
        compare(actual_tensor, expected_tensor, category=category, dtype=dtype)

@pytest.mark.smoke
@pytest.mark.covers("aten::kthvalue")
@pytest.mark.covers("aten::median.dim")
@pytest.mark.covers("aten::sort")
@pytest.mark.covers("aten::topk")
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
@pytest.mark.covers("aten::_unique2")
@pytest.mark.covers("aten::cumprod")
@pytest.mark.covers("aten::cumsum")
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
@pytest.mark.covers("aten::multinomial")
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


@pytest.mark.smoke
@pytest.mark.covers("aten::_compute_linear_combination")
@pytest.mark.covers("aten::_compute_linear_combination.out", surface="out_variant")
@pytest.mark.covers("aten::_jagged_to_padded_dense_forward")
@pytest.mark.covers("aten::_lazy_clone")
@pytest.mark.covers("aten::_padded_dense_to_jagged_forward")
@pytest.mark.covers("aten::_prelu_kernel")
@pytest.mark.covers("aten::_rowwise_prune")
@pytest.mark.covers("aten::_saturate_weight_to_fp16")
@pytest.mark.covers("aten::_trilinear")
@pytest.mark.covers("aten::_trilinear.out", surface="out_variant")
@pytest.mark.covers("aten::choose_qparams_optimized")
def test_low_level_misc_dispatcher_helpers(device, compare):
    input_cpu = torch.linspace(-1.0, 1.0, 12, dtype=torch.float32).reshape(2, 2, 3)
    input_dev = input_cpu.to(device)
    coefficients_cpu = torch.tensor([[1.0, 0.5], [0.25, 2.0]], dtype=torch.float32)
    coefficients_dev = coefficients_cpu.to(device)
    expected = torch.ops.aten._compute_linear_combination.default(
        input_cpu, coefficients_cpu,
    )
    actual = torch.ops.aten._compute_linear_combination.default(input_dev, coefficients_dev)
    synchronize(device)
    compare(actual, expected, category="elementwise", dtype=torch.float32)
    out = torch.empty_like(input_dev)
    returned = torch.ops.aten._compute_linear_combination.out(
        input_dev, coefficients_dev, out=out,
    )
    synchronize(device)
    assert returned.data_ptr() == out.data_ptr()
    compare(out, expected, category="elementwise", dtype=torch.float32)

    expected = torch.ops.aten._prelu_kernel.default(input_cpu, torch.tensor([0.25]))
    actual = torch.ops.aten._prelu_kernel.default(
        input_dev, torch.tensor([0.25], device=device),
    )
    synchronize(device)
    compare(actual, expected, category="elementwise", dtype=torch.float32)

    saturated_cpu = torch.ops.aten._saturate_weight_to_fp16.default(input_cpu * 100000.0)
    saturated_dev = torch.ops.aten._saturate_weight_to_fp16.default(input_dev * 100000.0)
    synchronize(device)
    compare(saturated_dev, saturated_cpu, category="elementwise", dtype=torch.float32)

    cloned_cpu = torch.ops.aten._lazy_clone.default(input_cpu)
    cloned_dev = torch.ops.aten._lazy_clone.default(input_dev)
    synchronize(device)
    compare(cloned_dev, cloned_cpu, category="exact", dtype=torch.float32)
    assert cloned_dev.data_ptr() != input_dev.data_ptr()

    tri_args_cpu = (
        torch.ones(2, 3, dtype=torch.float32),
        torch.ones(2, 3, dtype=torch.float32) * 2,
        torch.ones(2, 3, dtype=torch.float32) * 3,
        [2, 3],
        [2, 3],
        [2, 3],
        [1],
        1,
    )
    tri_args_dev = (
        tri_args_cpu[0].to(device),
        tri_args_cpu[1].to(device),
        tri_args_cpu[2].to(device),
        *tri_args_cpu[3:],
    )
    expected = torch.ops.aten._trilinear.default(*tri_args_cpu)
    actual = torch.ops.aten._trilinear.default(*tri_args_dev)
    synchronize(device)
    compare(actual, expected, category="elementwise", dtype=torch.float32)
    out = torch.empty(0, dtype=torch.float32, device=device)
    returned = torch.ops.aten._trilinear.out(*tri_args_dev, out=out)
    synchronize(device)
    assert returned.data_ptr() == out.data_ptr()
    compare(out, expected, category="elementwise", dtype=torch.float32)

    expected_scale, expected_zero = torch.ops.aten.choose_qparams_optimized.default(
        input_cpu.flatten(), input_cpu.numel(), 5, 0.5, 8,
    )
    actual_scale, actual_zero = torch.ops.aten.choose_qparams_optimized.default(
        input_dev.flatten(), input_dev.numel(), 5, 0.5, 8,
    )
    synchronize(device)
    compare(actual_scale, expected_scale, category="elementwise", dtype=torch.float32)
    compare(actual_zero, expected_zero, category="exact", dtype=actual_zero.dtype)

    weight_cpu = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    mask_cpu = torch.tensor([True, False, True], dtype=torch.bool)
    pruned_cpu, indices_cpu = torch.ops.aten._rowwise_prune.default(
        weight_cpu, mask_cpu, torch.int32,
    )
    pruned_dev, indices_dev = torch.ops.aten._rowwise_prune.default(
        weight_cpu.to(device), mask_cpu.to(device), torch.int32,
    )
    synchronize(device)
    compare(pruned_dev, pruned_cpu, category="exact", dtype=torch.float32)
    compare(indices_dev, indices_cpu, category="exact", dtype=torch.int32)

    values_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    offsets_cpu = torch.tensor([0, 2, 3], dtype=torch.int64)
    padded_cpu = torch.ops.aten._jagged_to_padded_dense_forward.default(
        values_cpu, [offsets_cpu], [2], 0.0,
    )
    padded_dev = torch.ops.aten._jagged_to_padded_dense_forward.default(
        values_cpu.to(device), [offsets_cpu.to(device)], [2], 0.0,
    )
    synchronize(device)
    compare(padded_dev, padded_cpu, category="exact", dtype=torch.float32)
    jagged_cpu = torch.ops.aten._padded_dense_to_jagged_forward.default(
        padded_cpu, [offsets_cpu], None,
    )
    jagged_dev = torch.ops.aten._padded_dense_to_jagged_forward.default(
        padded_dev, [offsets_cpu.to(device)], None,
    )
    synchronize(device)
    compare(jagged_dev, jagged_cpu, category="exact", dtype=torch.float32)


@pytest.mark.smoke
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::smm")
def test_sparse_smm_dispatcher_surface(device, compare):
    indices_cpu = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    values_cpu = torch.tensor([1.0, 2.0], dtype=torch.float32)
    sparse_cpu = torch.sparse_coo_tensor(indices_cpu, values_cpu, (2, 2)).coalesce()
    sparse_dev = torch.sparse_coo_tensor(
        indices_cpu.to(device), values_cpu.to(device), (2, 2), device=device,
    ).coalesce()
    rhs_cpu = torch.eye(2, dtype=torch.float32)
    rhs_dev = rhs_cpu.to(device)
    expected = torch.ops.aten.smm.default(sparse_cpu, rhs_cpu)
    actual = torch.ops.aten.smm.default(sparse_dev, rhs_dev)
    synchronize(device)
    compare(actual, expected, category="matmul", dtype=torch.float32)


@pytest.mark.smoke
@pytest.mark.covers("aten::_cummax_helper")
@pytest.mark.covers("aten::_cummin_helper")
@pytest.mark.covers("aten::_grouped_mm")
@pytest.mark.covers("aten::_histogramdd_bin_edges.out", surface="out_variant")
@pytest.mark.covers("aten::_transform_bias_rescale_qkv")
@pytest.mark.covers("aten::_transform_bias_rescale_qkv.out", surface="out_variant")
@pytest.mark.covers("aten::_weight_int8pack_mm")
@pytest.mark.covers("aten::hash_tensor.out", surface="out_variant")
def test_low_level_numeric_dispatcher_helpers(device, compare):
    input_cpu = torch.tensor(
        [[1.0, 3.0, 2.0], [4.0, 0.5, -1.0]], dtype=torch.float32,
    )
    input_dev = input_cpu.to(device)
    expected_values = torch.empty_like(input_cpu)
    expected_indices = torch.empty(input_cpu.shape, dtype=torch.long)
    actual_values = torch.empty_like(input_dev)
    actual_indices = torch.empty(input_dev.shape, dtype=torch.long, device=device)
    torch.ops.aten._cummax_helper.default(input_cpu, expected_values, expected_indices, 1)
    torch.ops.aten._cummax_helper.default(input_dev, actual_values, actual_indices, 1)
    synchronize(device)
    compare(actual_values, expected_values, category="exact", dtype=torch.float32)
    compare(actual_indices, expected_indices, category="exact", dtype=torch.long)

    expected_values = torch.empty_like(input_cpu)
    expected_indices = torch.empty(input_cpu.shape, dtype=torch.long)
    actual_values = torch.empty_like(input_dev)
    actual_indices = torch.empty(input_dev.shape, dtype=torch.long, device=device)
    torch.ops.aten._cummin_helper.default(input_cpu, expected_values, expected_indices, 1)
    torch.ops.aten._cummin_helper.default(input_dev, actual_values, actual_indices, 1)
    synchronize(device)
    compare(actual_values, expected_values, category="exact", dtype=torch.float32)
    compare(actual_indices, expected_indices, category="exact", dtype=torch.long)

    grouped_lhs_cpu = torch.linspace(-1.0, 1.0, 2 * 16 * 16, dtype=torch.float32).reshape(
        2, 16, 16,
    )
    grouped_rhs_cpu = torch.linspace(0.5, -0.5, 2 * 16 * 16, dtype=torch.float32).reshape(
        2, 16, 16,
    )
    expected = torch.ops.aten._grouped_mm.default(
        grouped_lhs_cpu, grouped_rhs_cpu, None, None, None,
    )
    actual = torch.ops.aten._grouped_mm.default(
        grouped_lhs_cpu.to(device), grouped_rhs_cpu.to(device), None, None, None,
    )
    synchronize(device)
    compare(actual, expected, category="matmul", dtype=torch.float32)

    qkv_cpu = torch.linspace(-1.0, 1.0, 2 * 4 * 18, dtype=torch.float32).reshape(2, 4, 18)
    qkv_bias_cpu = torch.linspace(-0.25, 0.25, 18, dtype=torch.float32)
    expected = torch.ops.aten._transform_bias_rescale_qkv.default(qkv_cpu, qkv_bias_cpu, 2)
    actual = torch.ops.aten._transform_bias_rescale_qkv.default(
        qkv_cpu.to(device), qkv_bias_cpu.to(device), 2,
    )
    _compare_tuple(actual, expected, compare, category="elementwise", dtype=torch.float32)
    outs = [
        torch.empty_like(expected_tensor, device=device)
        for expected_tensor in expected
    ]
    actual = torch.ops.aten._transform_bias_rescale_qkv.out(
        qkv_cpu.to(device),
        qkv_bias_cpu.to(device),
        2,
        out0=outs[0],
        out1=outs[1],
        out2=outs[2],
    )
    assert all(actual_tensor.data_ptr() == out_tensor.data_ptr() for actual_tensor, out_tensor in zip(actual, outs))
    _compare_tuple(actual, expected, compare, category="elementwise", dtype=torch.float32)

    int8_weight_cpu = torch.arange(-24, 24, dtype=torch.int8).reshape(12, 4)
    int8_scales_cpu = torch.linspace(0.1, 1.2, 12, dtype=torch.float32)
    int8_input_cpu = torch.linspace(-1.0, 1.0, 8, dtype=torch.float32).reshape(2, 4)
    expected = torch.ops.aten._weight_int8pack_mm.default(
        int8_input_cpu, int8_weight_cpu, int8_scales_cpu,
    )
    actual = torch.ops.aten._weight_int8pack_mm.default(
        int8_input_cpu.to(device), int8_weight_cpu.to(device), int8_scales_cpu.to(device),
    )
    synchronize(device)
    compare(actual, expected, category="matmul", dtype=torch.float32)

    hash_input_cpu = torch.arange(6, dtype=torch.int64).reshape(2, 3)
    expected_hash = torch.empty(2, dtype=torch.uint64)
    actual_hash = torch.empty(2, dtype=torch.uint64, device=device)
    torch.ops.aten.hash_tensor.out(
        hash_input_cpu, [1], keepdim=False, mode=0, out=expected_hash,
    )
    torch.ops.aten.hash_tensor.out(
        hash_input_cpu.to(device), [1], keepdim=False, mode=0, out=actual_hash,
    )
    synchronize(device)
    compare(actual_hash, expected_hash, category="exact", dtype=torch.uint64)

    histogram_input_cpu = torch.tensor(
        [[0.0, 0.0], [0.2, 0.4], [0.8, 0.6], [1.0, 1.0]], dtype=torch.float32,
    )
    expected_bins = [torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.float32)]
    actual_bins = [
        torch.empty(0, dtype=torch.float32, device=device),
        torch.empty(0, dtype=torch.float32, device=device),
    ]
    torch.ops.aten._histogramdd_bin_edges.out(
        histogram_input_cpu,
        [2, 3],
        range=None,
        weight=None,
        density=False,
        out=expected_bins,
    )
    torch.ops.aten._histogramdd_bin_edges.out(
        histogram_input_cpu.to(device),
        [2, 3],
        range=None,
        weight=None,
        density=False,
        out=actual_bins,
    )
    for actual, expected in zip(actual_bins, expected_bins):
        synchronize(device)
        compare(actual, expected, category="exact", dtype=torch.float32)
