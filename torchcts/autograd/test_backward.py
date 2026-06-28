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

BACKWARD_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
ACTIVATIONS = ["relu", "gelu", "sigmoid", "tanh", "silu"]

@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.covers("aten::_backward")
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
@pytest.mark.parametrize("op_name", ACTIVATIONS)
def test_first_order_backward(dtype, op_name, device, compare, input_gen):
    # Test simple model backpropagation (Linear + Activation)
    x_dev = input_gen((4, 8), dtype, device)
    x_dev.requires_grad = True
    
    w_cpu = torch.randn(8, 4, dtype=dtype)
    w_dev = w_cpu.to(device)
    w_dev.requires_grad = True
    
    x_cpu = x_dev.cpu().detach()
    x_cpu.requires_grad = True
    
    w_ref = w_cpu.clone().detach()
    w_ref.requires_grad = True
    
    op_fn = getattr(torch.nn.functional, op_name)
    
    out_dev = op_fn(torch.mm(x_dev, w_dev)).sum()
    out_cpu = op_fn(torch.mm(x_cpu, w_ref)).sum()
    
    out_dev.backward()
    out_cpu.backward()
    synchronize(device)
    
    compare(x_dev.grad, x_cpu.grad, category="matmul_backward", dtype=dtype)
    compare(w_dev.grad, w_ref.grad, category="matmul_backward", dtype=dtype)


def _compare_backward_tensor(actual, expected, compare, dtype):
    synchronize(actual.device.type)
    compare(actual, expected, category="backward", dtype=dtype)


def _compare_out_tensor(actual, out, expected, compare, dtype):
    assert actual.data_ptr() == out.data_ptr()
    _compare_backward_tensor(out, expected, compare, dtype)


def _compare_tensor_tuple(actual, expected, compare, dtype):
    assert len(actual) == len(expected)
    for actual_tensor, expected_tensor in zip(actual, expected):
        _compare_sparse_or_dense(actual_tensor, expected_tensor, compare, dtype)


def _compare_sparse_or_dense(actual, expected, compare, dtype=torch.float32):
    sparse_layouts = {
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    }
    if expected.layout in sparse_layouts:
        if expected.layout == torch.sparse_coo:
            actual = actual.coalesce()
            expected = expected.coalesce()
        actual = actual.to_dense()
        expected = expected.to_dense()
    _compare_backward_tensor(actual, expected, compare, dtype)


@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.covers("aten::threshold_backward")
def test_direct_threshold_backward(device, compare):
    grad = torch.randn(3, 4, dtype=torch.float32)
    x = torch.randn(3, 4, dtype=torch.float32)
    expected = torch.ops.aten.threshold_backward(grad, x, 0.0)
    actual = torch.ops.aten.threshold_backward(grad.to(device), x.to(device), 0.0)
    _compare_backward_tensor(actual, expected, compare, torch.float32)


@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.covers("aten::slice_backward")
@pytest.mark.covers("aten::slice_backward.out", surface="out_variant")
def test_direct_slice_backward(device, compare):
    grad = torch.randn(2, 3, dtype=torch.float32)
    args = ([4, 3], 0, 1, 3, 1)
    expected = torch.ops.aten.slice_backward(grad, *args)
    actual = torch.ops.aten.slice_backward(grad.to(device), *args)
    _compare_backward_tensor(actual, expected, compare, torch.float32)
    out = torch.empty(4, 3, dtype=torch.float32, device=device)
    actual = torch.ops.aten.slice_backward.out(grad.to(device), *args, out=out)
    _compare_out_tensor(actual, out, expected, compare, torch.float32)


@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.covers("aten::unfold_backward")
@pytest.mark.covers("aten::unfold_backward.out", surface="out_variant")
def test_direct_unfold_backward(device, compare):
    grad = torch.randn(3, 3, dtype=torch.float32)
    args = ([5], 0, 3, 1)
    expected = torch.ops.aten.unfold_backward(grad, *args)
    actual = torch.ops.aten.unfold_backward(grad.to(device), *args)
    _compare_backward_tensor(actual, expected, compare, torch.float32)
    out = torch.empty(5, dtype=torch.float32, device=device)
    actual = torch.ops.aten.unfold_backward.out(grad.to(device), *args, out=out)
    _compare_out_tensor(actual, out, expected, compare, torch.float32)


@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.covers("aten::avg_pool2d_backward")
@pytest.mark.covers("aten::avg_pool2d_backward.grad_input")
def test_direct_avg_pool2d_backward(device, compare):
    grad = torch.randn(1, 1, 2, 2, dtype=torch.float32)
    x = torch.randn(1, 1, 4, 4, dtype=torch.float32)
    args = ([2, 2], [2, 2], [0, 0], False, True, None)
    expected = torch.ops.aten.avg_pool2d_backward(grad, x, *args)
    actual = torch.ops.aten.avg_pool2d_backward(grad.to(device), x.to(device), *args)
    _compare_backward_tensor(actual, expected, compare, torch.float32)
    out = torch.empty_like(x, device=device)
    actual = torch.ops.aten.avg_pool2d_backward.grad_input(
        grad.to(device), x.to(device), *args, grad_input=out,
    )
    _compare_out_tensor(actual, out, expected, compare, torch.float32)


@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.covers("aten::upsample_nearest2d_backward")
@pytest.mark.covers("aten::upsample_nearest2d_backward.grad_input")
def test_direct_upsample_nearest2d_backward(device, compare):
    grad = torch.randn(1, 1, 4, 4, dtype=torch.float32)
    args = ([4, 4], [1, 1, 2, 2], None, None)
    expected = torch.ops.aten.upsample_nearest2d_backward(grad, *args)
    actual = torch.ops.aten.upsample_nearest2d_backward(grad.to(device), *args)
    _compare_backward_tensor(actual, expected, compare, torch.float32)
    out = torch.empty(1, 1, 2, 2, dtype=torch.float32, device=device)
    actual = torch.ops.aten.upsample_nearest2d_backward.grad_input(
        grad.to(device), *args, grad_input=out,
    )
    _compare_out_tensor(actual, out, expected, compare, torch.float32)


@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.covers("aten::embedding_backward")
def test_direct_embedding_backward(device, compare):
    grad = torch.randn(3, 2, dtype=torch.float32)
    indices = torch.tensor([0, 2, 1], dtype=torch.int64)
    args = (5, -1, False, False)
    expected = torch.ops.aten.embedding_backward(grad, indices, *args)
    actual = torch.ops.aten.embedding_backward(grad.to(device), indices.to(device), *args)
    _compare_backward_tensor(actual, expected, compare, torch.float32)


@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.covers("aten::binary_cross_entropy_backward.grad_input")
@pytest.mark.covers("aten::elu_backward.grad_input")
@pytest.mark.covers("aten::gelu_backward.grad_input")
@pytest.mark.covers("aten::hardshrink_backward.grad_input")
@pytest.mark.covers("aten::hardsigmoid_backward.grad_input")
@pytest.mark.covers("aten::hardswish_backward.out", surface="out_variant")
@pytest.mark.covers("aten::hardtanh_backward.grad_input")
@pytest.mark.covers("aten::huber_loss_backward.out", surface="out_variant")
@pytest.mark.covers("aten::leaky_relu_backward.grad_input")
@pytest.mark.covers("aten::logit_backward.grad_input")
@pytest.mark.covers("aten::mse_loss_backward.grad_input")
@pytest.mark.covers("aten::sigmoid_backward.grad_input")
@pytest.mark.covers("aten::silu_backward.grad_input")
@pytest.mark.covers("aten::smooth_l1_loss_backward")
@pytest.mark.covers("aten::smooth_l1_loss_backward.grad_input")
@pytest.mark.covers("aten::soft_margin_loss_backward")
@pytest.mark.covers("aten::soft_margin_loss_backward.grad_input")
@pytest.mark.covers("aten::softplus_backward.grad_input")
@pytest.mark.covers("aten::softshrink_backward.grad_input")
@pytest.mark.covers("aten::tanh_backward.grad_input")
@pytest.mark.covers("aten::threshold_backward.grad_input")
def test_direct_elementwise_and_loss_backward_out_overloads(device, compare):
    x_cpu = torch.linspace(-1.0, 1.0, 12, dtype=torch.float32).reshape(3, 4)
    grad_cpu = torch.linspace(0.1, 1.2, 12, dtype=torch.float32).reshape(3, 4)
    target_cpu = torch.zeros_like(x_cpu)
    label_cpu = torch.ones_like(x_cpu)
    x_dev = x_cpu.to(device)
    grad_dev = grad_cpu.to(device)
    target_dev = target_cpu.to(device)
    label_dev = label_cpu.to(device)

    def check_out(expected, actual, out):
        _compare_out_tensor(actual, out, expected, compare, torch.float32)

    expected = torch.ops.aten.sigmoid_backward.default(grad_cpu, torch.sigmoid(x_cpu))
    out = torch.empty_like(x_dev)
    actual = torch.ops.aten.sigmoid_backward.grad_input(
        grad_dev, torch.sigmoid(x_dev), grad_input=out,
    )
    check_out(expected, actual, out)

    expected = torch.ops.aten.tanh_backward.default(grad_cpu, torch.tanh(x_cpu))
    out = torch.empty_like(x_dev)
    actual = torch.ops.aten.tanh_backward.grad_input(
        grad_dev, torch.tanh(x_dev), grad_input=out,
    )
    check_out(expected, actual, out)

    expected = torch.ops.aten.silu_backward.default(grad_cpu, x_cpu)
    out = torch.empty_like(x_dev)
    actual = torch.ops.aten.silu_backward.grad_input(grad_dev, x_dev, grad_input=out)
    check_out(expected, actual, out)

    expected = torch.ops.aten.gelu_backward.default(
        grad_cpu, x_cpu, approximate="none",
    )
    out = torch.empty_like(x_dev)
    actual = torch.ops.aten.gelu_backward.grad_input(
        grad_dev, x_dev, approximate="none", grad_input=out,
    )
    check_out(expected, actual, out)

    expected = torch.ops.aten.hardsigmoid_backward.default(grad_cpu, x_cpu)
    out = torch.empty_like(x_dev)
    actual = torch.ops.aten.hardsigmoid_backward.grad_input(
        grad_dev, x_dev, grad_input=out,
    )
    check_out(expected, actual, out)

    expected = torch.ops.aten.hardtanh_backward.default(grad_cpu, x_cpu, -0.5, 0.5)
    out = torch.empty_like(x_dev)
    actual = torch.ops.aten.hardtanh_backward.grad_input(
        grad_dev, x_dev, -0.5, 0.5, grad_input=out,
    )
    check_out(expected, actual, out)

    expected = torch.ops.aten.leaky_relu_backward.default(grad_cpu, x_cpu, 0.01, False)
    out = torch.empty_like(x_dev)
    actual = torch.ops.aten.leaky_relu_backward.grad_input(
        grad_dev, x_dev, 0.01, False, grad_input=out,
    )
    check_out(expected, actual, out)

    expected = torch.ops.aten.softplus_backward.default(grad_cpu, x_cpu, 1, 20)
    out = torch.empty_like(x_dev)
    actual = torch.ops.aten.softplus_backward.grad_input(
        grad_dev, x_dev, 1, 20, grad_input=out,
    )
    check_out(expected, actual, out)

    expected = torch.ops.aten.softshrink_backward.default(grad_cpu, x_cpu, 0.5)
    out = torch.empty_like(x_dev)
    actual = torch.ops.aten.softshrink_backward.grad_input(
        grad_dev, x_dev, 0.5, grad_input=out,
    )
    check_out(expected, actual, out)

    expected = torch.ops.aten.hardshrink_backward.default(grad_cpu, x_cpu, 0.5)
    out = torch.empty_like(x_dev)
    actual = torch.ops.aten.hardshrink_backward.grad_input(
        grad_dev, x_dev, 0.5, grad_input=out,
    )
    check_out(expected, actual, out)

    expected = torch.ops.aten.elu_backward.default(grad_cpu, 1, 1, 1, False, x_cpu)
    out = torch.empty_like(x_dev)
    actual = torch.ops.aten.elu_backward.grad_input(
        grad_dev, 1, 1, 1, False, x_dev, grad_input=out,
    )
    check_out(expected, actual, out)

    logits_cpu = torch.sigmoid(x_cpu).clamp(0.1, 0.9)
    logits_dev = logits_cpu.to(device)
    expected = torch.ops.aten.logit_backward.default(grad_cpu, logits_cpu, None)
    out = torch.empty_like(x_dev)
    actual = torch.ops.aten.logit_backward.grad_input(
        grad_dev, logits_dev, None, grad_input=out,
    )
    check_out(expected, actual, out)

    expected = torch.ops.aten.threshold_backward.default(grad_cpu, x_cpu, 0.0)
    out = torch.empty_like(x_dev)
    actual = torch.ops.aten.threshold_backward.grad_input(
        grad_dev, x_dev, 0.0, grad_input=out,
    )
    check_out(expected, actual, out)

    expected = torch.ops.aten.mse_loss_backward.default(grad_cpu, x_cpu, target_cpu, 1)
    out = torch.empty_like(x_dev)
    actual = torch.ops.aten.mse_loss_backward.grad_input(
        grad_dev, x_dev, target_dev, 1, grad_input=out,
    )
    check_out(expected, actual, out)

    probabilities_cpu = torch.sigmoid(x_cpu).clamp(0.1, 0.9)
    probabilities_dev = probabilities_cpu.to(device)
    expected = torch.ops.aten.binary_cross_entropy_backward.default(
        grad_cpu, probabilities_cpu, target_cpu, None, 1,
    )
    out = torch.empty_like(x_dev)
    actual = torch.ops.aten.binary_cross_entropy_backward.grad_input(
        grad_dev, probabilities_dev, target_dev, None, 1, grad_input=out,
    )
    check_out(expected, actual, out)

    expected = torch.ops.aten.smooth_l1_loss_backward.default(
        grad_cpu, x_cpu, target_cpu, 1, 1.0,
    )
    actual = torch.ops.aten.smooth_l1_loss_backward.default(
        grad_dev, x_dev, target_dev, 1, 1.0,
    )
    _compare_backward_tensor(actual, expected, compare, torch.float32)
    out = torch.empty_like(x_dev)
    actual = torch.ops.aten.smooth_l1_loss_backward.grad_input(
        grad_dev, x_dev, target_dev, 1, 1.0, grad_input=out,
    )
    check_out(expected, actual, out)

    expected = torch.ops.aten.soft_margin_loss_backward.default(
        grad_cpu, x_cpu, label_cpu, 1,
    )
    actual = torch.ops.aten.soft_margin_loss_backward.default(
        grad_dev, x_dev, label_dev, 1,
    )
    _compare_backward_tensor(actual, expected, compare, torch.float32)
    out = torch.empty_like(x_dev)
    actual = torch.ops.aten.soft_margin_loss_backward.grad_input(
        grad_dev, x_dev, label_dev, 1, grad_input=out,
    )
    check_out(expected, actual, out)

    expected = torch.ops.aten.hardswish_backward.default(grad_cpu, x_cpu)
    out = torch.empty_like(x_dev)
    actual = torch.ops.aten.hardswish_backward.out(grad_dev, x_dev, out=out)
    check_out(expected, actual, out)

    expected = torch.ops.aten.huber_loss_backward.default(
        grad_cpu, x_cpu, target_cpu, 1, 1.0,
    )
    out = torch.empty_like(x_dev)
    actual = torch.ops.aten.huber_loss_backward.out(
        grad_dev, x_dev, target_dev, 1, 1.0, grad_input=out,
    )
    check_out(expected, actual, out)


@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.covers("aten::cummaxmin_backward")
@pytest.mark.covers("aten::cumprod_backward")
@pytest.mark.covers("aten::diagonal_backward.out", surface="out_variant")
@pytest.mark.covers("aten::gather_backward")
@pytest.mark.covers("aten::index_select_backward")
@pytest.mark.covers("aten::masked_scatter_backward")
@pytest.mark.covers("aten::masked_select_backward")
@pytest.mark.covers("aten::select_backward.out", surface="out_variant")
@pytest.mark.covers("aten::trace_backward")
@pytest.mark.covers("aten::value_selecting_reduction_backward")
def test_direct_indexing_and_reduction_backward_surfaces(device, compare):
    grad_cpu = torch.linspace(0.1, 0.6, 6, dtype=torch.float32).reshape(2, 3)
    input_cpu = torch.linspace(-1.0, 1.0, 6, dtype=torch.float32).reshape(2, 3)
    grad_dev = grad_cpu.to(device)
    input_dev = input_cpu.to(device)

    indices_cpu = torch.tensor([[0, 2], [1, 0]], dtype=torch.int64)
    indices_dev = indices_cpu.to(device)
    expected = torch.ops.aten.gather_backward.default(
        torch.ones(2, 2), input_cpu, 1, indices_cpu, False,
    )
    actual = torch.ops.aten.gather_backward.default(
        torch.ones(2, 2, device=device), input_dev, 1, indices_dev, False,
    )
    _compare_backward_tensor(actual, expected, compare, torch.float32)

    select_index_cpu = torch.tensor([0, 2], dtype=torch.int64)
    select_index_dev = select_index_cpu.to(device)
    expected = torch.ops.aten.index_select_backward.default(
        torch.ones(2, 3), [4, 3], 0, select_index_cpu,
    )
    actual = torch.ops.aten.index_select_backward.default(
        torch.ones(2, 3, device=device), [4, 3], 0, select_index_dev,
    )
    _compare_backward_tensor(actual, expected, compare, torch.float32)

    expected = torch.ops.aten.select_backward.default(torch.ones(3), [2, 3], 0, 1)
    out = torch.empty(2, 3, dtype=torch.float32, device=device)
    actual = torch.ops.aten.select_backward.out(
        torch.ones(3, device=device), [2, 3], 0, 1, out=out,
    )
    _compare_out_tensor(actual, out, expected, compare, torch.float32)

    expected = torch.ops.aten.diagonal_backward.default(
        torch.ones(3), [3, 3], 0, 0, 1,
    )
    out = torch.empty(3, 3, dtype=torch.float32, device=device)
    actual = torch.ops.aten.diagonal_backward.out(
        torch.ones(3, device=device), [3, 3], 0, 0, 1, out=out,
    )
    _compare_out_tensor(actual, out, expected, compare, torch.float32)

    expected = torch.ops.aten.trace_backward.default(torch.tensor(2.0), [3, 3])
    actual = torch.ops.aten.trace_backward.default(
        torch.tensor(2.0, device=device), [3, 3],
    )
    _compare_backward_tensor(actual, expected, compare, torch.float32)

    cum_indices_cpu = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.int64)
    cum_indices_dev = cum_indices_cpu.to(device)
    expected = torch.ops.aten.cummaxmin_backward.default(
        grad_cpu, input_cpu, cum_indices_cpu, 1,
    )
    actual = torch.ops.aten.cummaxmin_backward.default(
        grad_dev, input_dev, cum_indices_dev, 1,
    )
    _compare_backward_tensor(actual, expected, compare, torch.float32)

    output_cpu = torch.cumprod(input_cpu + 2.0, dim=1)
    output_dev = output_cpu.to(device)
    expected = torch.ops.aten.cumprod_backward.default(
        grad_cpu, input_cpu + 2.0, 1, output_cpu,
    )
    actual = torch.ops.aten.cumprod_backward.default(
        grad_dev, input_dev + 2.0, 1, output_dev,
    )
    _compare_backward_tensor(actual, expected, compare, torch.float32)

    expected = torch.ops.aten.value_selecting_reduction_backward.default(
        torch.ones(2), 1, torch.tensor([1, 0]), [2, 3], False,
    )
    actual = torch.ops.aten.value_selecting_reduction_backward.default(
        torch.ones(2, device=device), 1, torch.tensor([1, 0], device=device), [2, 3], False,
    )
    _compare_backward_tensor(actual, expected, compare, torch.float32)

    mask_cpu = torch.tensor(
        [[True, False, True], [False, True, False]], dtype=torch.bool,
    )
    mask_dev = mask_cpu.to(device)
    expected = torch.ops.aten.masked_select_backward.default(
        torch.ones(3), input_cpu, mask_cpu,
    )
    actual = torch.ops.aten.masked_select_backward.default(
        torch.ones(3, device=device), input_dev, mask_dev,
    )
    _compare_backward_tensor(actual, expected, compare, torch.float32)

    expected = torch.ops.aten.masked_scatter_backward.default(
        grad_cpu, mask_cpu, [2, 3],
    )
    actual = torch.ops.aten.masked_scatter_backward.default(
        grad_dev, mask_dev, [2, 3],
    )
    _compare_backward_tensor(actual, expected, compare, torch.float32)


@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.covers("aten::_adaptive_avg_pool2d_backward")
@pytest.mark.covers("aten::_adaptive_avg_pool2d_backward.out", surface="out_variant")
@pytest.mark.covers("aten::_adaptive_avg_pool3d_backward")
@pytest.mark.covers("aten::_adaptive_avg_pool3d_backward.out", surface="out_variant")
@pytest.mark.covers("aten::adaptive_avg_pool3d_backward.grad_input")
@pytest.mark.covers("aten::adaptive_max_pool2d_backward")
@pytest.mark.covers("aten::adaptive_max_pool2d_backward.grad_input")
@pytest.mark.covers("aten::adaptive_max_pool3d_backward")
@pytest.mark.covers("aten::adaptive_max_pool3d_backward.grad_input")
@pytest.mark.covers("aten::avg_pool3d_backward")
@pytest.mark.covers("aten::avg_pool3d_backward.grad_input")
@pytest.mark.covers("aten::max_pool2d_with_indices_backward.grad_input")
@pytest.mark.covers("aten::max_pool3d_with_indices_backward")
@pytest.mark.covers("aten::max_pool3d_with_indices_backward.grad_input")
def test_direct_pooling_backward_dispatcher_surfaces(device, compare):
    x2_cpu = torch.linspace(-1.0, 1.0, 16, dtype=torch.float32).reshape(1, 1, 4, 4)
    x2_dev = x2_cpu.to(device)
    grad2_cpu = torch.ones(1, 1, 2, 2, dtype=torch.float32)
    grad2_dev = grad2_cpu.to(device)

    expected = torch.ops.aten._adaptive_avg_pool2d_backward.default(grad2_cpu, x2_cpu)
    actual = torch.ops.aten._adaptive_avg_pool2d_backward.default(grad2_dev, x2_dev)
    _compare_backward_tensor(actual, expected, compare, torch.float32)
    out = torch.empty_like(x2_dev)
    actual = torch.ops.aten._adaptive_avg_pool2d_backward.out(grad2_dev, x2_dev, out=out)
    _compare_out_tensor(actual, out, expected, compare, torch.float32)

    pooled2_cpu, indices2_cpu = torch.nn.functional.adaptive_max_pool2d(
        x2_cpu, (2, 2), return_indices=True,
    )
    indices2_dev = indices2_cpu.to(device)
    expected = torch.ops.aten.adaptive_max_pool2d_backward.default(
        torch.ones_like(pooled2_cpu), x2_cpu, indices2_cpu,
    )
    actual = torch.ops.aten.adaptive_max_pool2d_backward.default(
        torch.ones_like(pooled2_cpu, device=device), x2_dev, indices2_dev,
    )
    _compare_backward_tensor(actual, expected, compare, torch.float32)
    out = torch.empty_like(x2_dev)
    actual = torch.ops.aten.adaptive_max_pool2d_backward.grad_input(
        torch.ones_like(pooled2_cpu, device=device), x2_dev, indices2_dev, grad_input=out,
    )
    _compare_out_tensor(actual, out, expected, compare, torch.float32)

    max2_cpu, max2_indices_cpu = torch.nn.functional.max_pool2d(
        x2_cpu, 2, 2, return_indices=True,
    )
    max2_indices_dev = max2_indices_cpu.to(device)
    expected = torch.ops.aten.max_pool2d_with_indices_backward.default(
        torch.ones_like(max2_cpu), x2_cpu, [2, 2], [2, 2], [0, 0], [1, 1], False, max2_indices_cpu,
    )
    out = torch.empty_like(x2_dev)
    actual = torch.ops.aten.max_pool2d_with_indices_backward.grad_input(
        torch.ones_like(max2_cpu, device=device),
        x2_dev,
        [2, 2],
        [2, 2],
        [0, 0],
        [1, 1],
        False,
        max2_indices_dev,
        grad_input=out,
    )
    _compare_out_tensor(actual, out, expected, compare, torch.float32)

    x3_cpu = torch.linspace(-1.0, 1.0, 64, dtype=torch.float32).reshape(1, 1, 4, 4, 4)
    x3_dev = x3_cpu.to(device)
    grad3_cpu = torch.ones(1, 1, 2, 2, 2, dtype=torch.float32)
    grad3_dev = grad3_cpu.to(device)

    expected = torch.ops.aten._adaptive_avg_pool3d_backward.default(grad3_cpu, x3_cpu)
    actual = torch.ops.aten._adaptive_avg_pool3d_backward.default(grad3_dev, x3_dev)
    _compare_backward_tensor(actual, expected, compare, torch.float32)
    out = torch.empty_like(x3_dev)
    actual = torch.ops.aten._adaptive_avg_pool3d_backward.out(grad3_dev, x3_dev, out=out)
    _compare_out_tensor(actual, out, expected, compare, torch.float32)
    out = torch.empty_like(x3_dev)
    actual = torch.ops.aten.adaptive_avg_pool3d_backward.grad_input(
        grad3_dev, x3_dev, grad_input=out,
    )
    _compare_out_tensor(actual, out, expected, compare, torch.float32)

    pooled3_cpu, indices3_cpu = torch.nn.functional.adaptive_max_pool3d(
        x3_cpu, (2, 2, 2), return_indices=True,
    )
    indices3_dev = indices3_cpu.to(device)
    expected = torch.ops.aten.adaptive_max_pool3d_backward.default(
        torch.ones_like(pooled3_cpu), x3_cpu, indices3_cpu,
    )
    actual = torch.ops.aten.adaptive_max_pool3d_backward.default(
        torch.ones_like(pooled3_cpu, device=device), x3_dev, indices3_dev,
    )
    _compare_backward_tensor(actual, expected, compare, torch.float32)
    out = torch.empty_like(x3_dev)
    actual = torch.ops.aten.adaptive_max_pool3d_backward.grad_input(
        torch.ones_like(pooled3_cpu, device=device), x3_dev, indices3_dev, grad_input=out,
    )
    _compare_out_tensor(actual, out, expected, compare, torch.float32)

    expected = torch.ops.aten.avg_pool3d_backward.default(
        grad3_cpu, x3_cpu, [2, 2, 2], [2, 2, 2], [0, 0, 0], False, True, None,
    )
    actual = torch.ops.aten.avg_pool3d_backward.default(
        grad3_dev, x3_dev, [2, 2, 2], [2, 2, 2], [0, 0, 0], False, True, None,
    )
    _compare_backward_tensor(actual, expected, compare, torch.float32)
    out = torch.empty_like(x3_dev)
    actual = torch.ops.aten.avg_pool3d_backward.grad_input(
        grad3_dev,
        x3_dev,
        [2, 2, 2],
        [2, 2, 2],
        [0, 0, 0],
        False,
        True,
        None,
        grad_input=out,
    )
    _compare_out_tensor(actual, out, expected, compare, torch.float32)

    max3_cpu, max3_indices_cpu = torch.nn.functional.max_pool3d(
        x3_cpu, 2, 2, return_indices=True,
    )
    max3_indices_dev = max3_indices_cpu.to(device)
    expected = torch.ops.aten.max_pool3d_with_indices_backward.default(
        torch.ones_like(max3_cpu),
        x3_cpu,
        [2, 2, 2],
        [2, 2, 2],
        [0, 0, 0],
        [1, 1, 1],
        False,
        max3_indices_cpu,
    )
    actual = torch.ops.aten.max_pool3d_with_indices_backward.default(
        torch.ones_like(max3_cpu, device=device),
        x3_dev,
        [2, 2, 2],
        [2, 2, 2],
        [0, 0, 0],
        [1, 1, 1],
        False,
        max3_indices_dev,
    )
    _compare_backward_tensor(actual, expected, compare, torch.float32)
    out = torch.empty_like(x3_dev)
    actual = torch.ops.aten.max_pool3d_with_indices_backward.grad_input(
        torch.ones_like(max3_cpu, device=device),
        x3_dev,
        [2, 2, 2],
        [2, 2, 2],
        [0, 0, 0],
        [1, 1, 1],
        False,
        max3_indices_dev,
        grad_input=out,
    )
    _compare_out_tensor(actual, out, expected, compare, torch.float32)


@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.covers("aten::_upsample_bicubic2d_aa_backward")
@pytest.mark.covers("aten::_upsample_bicubic2d_aa_backward.grad_input")
@pytest.mark.covers("aten::_upsample_bilinear2d_aa_backward")
@pytest.mark.covers("aten::_upsample_bilinear2d_aa_backward.grad_input")
@pytest.mark.covers("aten::_upsample_lanczos2d_aa_backward")
@pytest.mark.covers("aten::_upsample_lanczos2d_aa_backward.grad_input")
@pytest.mark.covers("aten::_upsample_nearest_exact1d_backward")
@pytest.mark.covers("aten::_upsample_nearest_exact1d_backward.grad_input")
@pytest.mark.covers("aten::_upsample_nearest_exact2d_backward")
@pytest.mark.covers("aten::_upsample_nearest_exact2d_backward.grad_input")
@pytest.mark.covers("aten::_upsample_nearest_exact3d_backward")
@pytest.mark.covers("aten::_upsample_nearest_exact3d_backward.grad_input")
@pytest.mark.covers("aten::upsample_bicubic2d_backward")
@pytest.mark.covers("aten::upsample_bicubic2d_backward.grad_input")
@pytest.mark.covers("aten::upsample_bilinear2d_backward")
@pytest.mark.covers("aten::upsample_bilinear2d_backward.grad_input")
@pytest.mark.covers("aten::upsample_linear1d_backward")
@pytest.mark.covers("aten::upsample_linear1d_backward.grad_input")
@pytest.mark.covers("aten::upsample_nearest1d_backward")
@pytest.mark.covers("aten::upsample_nearest1d_backward.grad_input")
@pytest.mark.covers("aten::upsample_nearest3d_backward")
@pytest.mark.covers("aten::upsample_nearest3d_backward.grad_input")
@pytest.mark.covers("aten::upsample_trilinear3d_backward")
@pytest.mark.covers("aten::upsample_trilinear3d_backward.grad_input")
def test_direct_upsample_backward_dispatcher_surfaces(device, compare):
    def check(default_op, grad_op, grad_cpu, args_cpu):
        grad_dev = grad_cpu.to(device)
        expected = default_op(grad_cpu, *args_cpu)
        actual = default_op(grad_dev, *args_cpu)
        _compare_backward_tensor(actual, expected, compare, torch.float32)
        out = torch.empty_like(expected, device=device)
        actual = grad_op(grad_dev, *args_cpu, grad_input=out)
        _compare_out_tensor(actual, out, expected, compare, torch.float32)

    check(
        torch.ops.aten.upsample_linear1d_backward.default,
        torch.ops.aten.upsample_linear1d_backward.grad_input,
        torch.ones(1, 1, 4, dtype=torch.float32),
        ([4], [1, 1, 2], False, None),
    )
    check(
        torch.ops.aten.upsample_nearest1d_backward.default,
        torch.ops.aten.upsample_nearest1d_backward.grad_input,
        torch.ones(1, 1, 4, dtype=torch.float32),
        ([4], [1, 1, 2], None),
    )
    check(
        torch.ops.aten._upsample_nearest_exact1d_backward.default,
        torch.ops.aten._upsample_nearest_exact1d_backward.grad_input,
        torch.ones(1, 1, 4, dtype=torch.float32),
        ([4], [1, 1, 2], None),
    )

    for default_op, grad_op in (
        (
            torch.ops.aten.upsample_bilinear2d_backward.default,
            torch.ops.aten.upsample_bilinear2d_backward.grad_input,
        ),
        (
            torch.ops.aten.upsample_bicubic2d_backward.default,
            torch.ops.aten.upsample_bicubic2d_backward.grad_input,
        ),
        (
            torch.ops.aten._upsample_bilinear2d_aa_backward.default,
            torch.ops.aten._upsample_bilinear2d_aa_backward.grad_input,
        ),
        (
            torch.ops.aten._upsample_bicubic2d_aa_backward.default,
            torch.ops.aten._upsample_bicubic2d_aa_backward.grad_input,
        ),
        (
            torch.ops.aten._upsample_lanczos2d_aa_backward.default,
            torch.ops.aten._upsample_lanczos2d_aa_backward.grad_input,
        ),
    ):
        check(
            default_op,
            grad_op,
            torch.ones(1, 1, 4, 4, dtype=torch.float32),
            ([4, 4], [1, 1, 2, 2], False, None, None),
        )
    check(
        torch.ops.aten._upsample_nearest_exact2d_backward.default,
        torch.ops.aten._upsample_nearest_exact2d_backward.grad_input,
        torch.ones(1, 1, 4, 4, dtype=torch.float32),
        ([4, 4], [1, 1, 2, 2], None, None),
    )

    check(
        torch.ops.aten.upsample_nearest3d_backward.default,
        torch.ops.aten.upsample_nearest3d_backward.grad_input,
        torch.ones(1, 1, 4, 4, 4, dtype=torch.float32),
        ([4, 4, 4], [1, 1, 2, 2, 2], None, None, None),
    )
    check(
        torch.ops.aten._upsample_nearest_exact3d_backward.default,
        torch.ops.aten._upsample_nearest_exact3d_backward.grad_input,
        torch.ones(1, 1, 4, 4, 4, dtype=torch.float32),
        ([4, 4, 4], [1, 1, 2, 2, 2], None, None, None),
    )
    check(
        torch.ops.aten.upsample_trilinear3d_backward.default,
        torch.ops.aten.upsample_trilinear3d_backward.grad_input,
        torch.ones(1, 1, 4, 4, 4, dtype=torch.float32),
        ([4, 4, 4], [1, 1, 2, 2, 2], False, None, None, None),
    )


@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.covers("aten::reflection_pad1d_backward")
@pytest.mark.covers("aten::reflection_pad1d_backward.grad_input")
@pytest.mark.covers("aten::reflection_pad2d_backward")
@pytest.mark.covers("aten::reflection_pad2d_backward.grad_input")
@pytest.mark.covers("aten::reflection_pad3d_backward")
@pytest.mark.covers("aten::reflection_pad3d_backward.grad_input")
@pytest.mark.covers("aten::replication_pad1d_backward")
@pytest.mark.covers("aten::replication_pad1d_backward.grad_input")
@pytest.mark.covers("aten::replication_pad2d_backward")
@pytest.mark.covers("aten::replication_pad2d_backward.grad_input")
@pytest.mark.covers("aten::replication_pad3d_backward")
@pytest.mark.covers("aten::replication_pad3d_backward.grad_input")
def test_direct_padding_backward_dispatcher_surfaces(device, compare):
    def check(op, input_cpu, grad_cpu, padding):
        input_dev = input_cpu.to(device)
        grad_dev = grad_cpu.to(device)
        expected = op.default(grad_cpu, input_cpu, padding)
        actual = op.default(grad_dev, input_dev, padding)
        _compare_backward_tensor(actual, expected, compare, torch.float32)
        out = torch.empty_like(input_dev)
        actual = op.grad_input(grad_dev, input_dev, padding, grad_input=out)
        _compare_out_tensor(actual, out, expected, compare, torch.float32)

    check(
        torch.ops.aten.reflection_pad1d_backward,
        torch.linspace(-1.0, 1.0, 4, dtype=torch.float32).reshape(1, 1, 4),
        torch.ones(1, 1, 6, dtype=torch.float32),
        [1, 1],
    )
    check(
        torch.ops.aten.replication_pad1d_backward,
        torch.linspace(-1.0, 1.0, 4, dtype=torch.float32).reshape(1, 1, 4),
        torch.ones(1, 1, 6, dtype=torch.float32),
        [1, 1],
    )
    check(
        torch.ops.aten.reflection_pad2d_backward,
        torch.linspace(-1.0, 1.0, 16, dtype=torch.float32).reshape(1, 1, 4, 4),
        torch.ones(1, 1, 6, 6, dtype=torch.float32),
        [1, 1, 1, 1],
    )
    check(
        torch.ops.aten.replication_pad2d_backward,
        torch.linspace(-1.0, 1.0, 16, dtype=torch.float32).reshape(1, 1, 4, 4),
        torch.ones(1, 1, 6, 6, dtype=torch.float32),
        [1, 1, 1, 1],
    )
    check(
        torch.ops.aten.reflection_pad3d_backward,
        torch.linspace(-1.0, 1.0, 64, dtype=torch.float32).reshape(1, 1, 4, 4, 4),
        torch.ones(1, 1, 6, 6, 6, dtype=torch.float32),
        [1, 1, 1, 1, 1, 1],
    )
    check(
        torch.ops.aten.replication_pad3d_backward,
        torch.linspace(-1.0, 1.0, 64, dtype=torch.float32).reshape(1, 1, 4, 4, 4),
        torch.ones(1, 1, 6, 6, 6, dtype=torch.float32),
        [1, 1, 1, 1, 1, 1],
    )


@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.covers("aten::_cdist_backward")
@pytest.mark.covers("aten::_cdist_backward.out", surface="out_variant")
@pytest.mark.covers("aten::_fake_quantize_learnable_per_channel_affine_backward")
@pytest.mark.covers("aten::_fake_quantize_learnable_per_tensor_affine_backward")
@pytest.mark.covers("aten::_masked_softmax_backward")
@pytest.mark.covers("aten::_masked_softmax_backward.out", surface="out_variant")
@pytest.mark.covers("aten::_pdist_backward")
@pytest.mark.covers("aten::_pdist_backward.out", surface="out_variant")
@pytest.mark.covers("aten::glu_backward")
@pytest.mark.covers("aten::glu_backward.grad_input")
@pytest.mark.covers("aten::glu_backward_jvp")
@pytest.mark.covers("aten::glu_backward_jvp.out", surface="out_variant")
@pytest.mark.covers("aten::glu_jvp")
@pytest.mark.covers("aten::glu_jvp.out", surface="out_variant")
def test_direct_dense_math_backward_dispatcher_surfaces(device, compare):
    x1_cpu = torch.linspace(-1.0, 1.0, 6, dtype=torch.float32).reshape(2, 3)
    x2_cpu = torch.linspace(-0.5, 0.5, 12, dtype=torch.float32).reshape(4, 3)
    x1_dev = x1_cpu.to(device)
    x2_dev = x2_cpu.to(device)
    cdist_cpu = torch.cdist(x1_cpu, x2_cpu, p=2)
    cdist_dev = cdist_cpu.to(device)
    expected = torch.ops.aten._cdist_backward.default(
        torch.ones_like(cdist_cpu), x1_cpu, x2_cpu, 2.0, cdist_cpu,
    )
    actual = torch.ops.aten._cdist_backward.default(
        torch.ones_like(cdist_cpu, device=device), x1_dev, x2_dev, 2.0, cdist_dev,
    )
    _compare_backward_tensor(actual, expected, compare, torch.float32)
    out = torch.empty_like(x1_dev)
    actual = torch.ops.aten._cdist_backward.out(
        torch.ones_like(cdist_cpu, device=device), x1_dev, x2_dev, 2.0, cdist_dev, out=out,
    )
    _compare_out_tensor(actual, out, expected, compare, torch.float32)

    pdist_cpu = torch.pdist(x1_cpu, p=2)
    pdist_dev = pdist_cpu.to(device)
    expected = torch.ops.aten._pdist_backward.default(
        torch.ones_like(pdist_cpu), x1_cpu, 2.0, pdist_cpu,
    )
    actual = torch.ops.aten._pdist_backward.default(
        torch.ones_like(pdist_cpu, device=device), x1_dev, 2.0, pdist_dev,
    )
    _compare_backward_tensor(actual, expected, compare, torch.float32)
    out = torch.empty_like(x1_dev)
    actual = torch.ops.aten._pdist_backward.out(
        torch.ones_like(pdist_cpu, device=device), x1_dev, 2.0, pdist_dev, out=out,
    )
    _compare_out_tensor(actual, out, expected, compare, torch.float32)

    x_cpu = torch.linspace(-1.5, 1.5, 12, dtype=torch.float32).reshape(2, 6)
    x_dev = x_cpu.to(device)
    glu_cpu = torch.nn.functional.glu(x_cpu, dim=1)
    glu_dev = glu_cpu.to(device)
    grad_glu_cpu = torch.linspace(0.1, 0.6, 6, dtype=torch.float32).reshape(2, 3)
    grad_glu_dev = grad_glu_cpu.to(device)
    expected = torch.ops.aten.glu_backward.default(grad_glu_cpu, x_cpu, 1)
    actual = torch.ops.aten.glu_backward.default(grad_glu_dev, x_dev, 1)
    _compare_backward_tensor(actual, expected, compare, torch.float32)
    out = torch.empty_like(x_dev)
    actual = torch.ops.aten.glu_backward.grad_input(
        grad_glu_dev, x_dev, 1, grad_input=out,
    )
    _compare_out_tensor(actual, out, expected, compare, torch.float32)

    dx_cpu = torch.linspace(0.2, 1.3, 12, dtype=torch.float32).reshape(2, 6)
    dx_dev = dx_cpu.to(device)
    expected = torch.ops.aten.glu_jvp.default(glu_cpu, x_cpu, dx_cpu, 1)
    actual = torch.ops.aten.glu_jvp.default(glu_dev, x_dev, dx_dev, 1)
    _compare_backward_tensor(actual, expected, compare, torch.float32)
    out = torch.empty_like(glu_dev)
    actual = torch.ops.aten.glu_jvp.out(glu_dev, x_dev, dx_dev, 1, out=out)
    _compare_out_tensor(actual, out, expected, compare, torch.float32)

    dgrad_glu_cpu = torch.full_like(grad_glu_cpu, 0.25)
    dgrad_glu_dev = dgrad_glu_cpu.to(device)
    grad_x_cpu = torch.full_like(x_cpu, 0.5)
    grad_x_dev = grad_x_cpu.to(device)
    expected = torch.ops.aten.glu_backward_jvp.default(
        grad_x_cpu, grad_glu_cpu, x_cpu, dgrad_glu_cpu, dx_cpu, 1,
    )
    actual = torch.ops.aten.glu_backward_jvp.default(
        grad_x_dev, grad_glu_dev, x_dev, dgrad_glu_dev, dx_dev, 1,
    )
    _compare_backward_tensor(actual, expected, compare, torch.float32)
    out = torch.empty_like(x_dev)
    actual = torch.ops.aten.glu_backward_jvp.out(
        grad_x_dev, grad_glu_dev, x_dev, dgrad_glu_dev, dx_dev, 1, out=out,
    )
    _compare_out_tensor(actual, out, expected, compare, torch.float32)

    softmax_cpu = torch.softmax(x1_cpu, dim=-1)
    softmax_dev = softmax_cpu.to(device)
    mask_cpu = torch.tensor([[False, True, False], [False, False, True]], dtype=torch.bool)
    mask_dev = mask_cpu.to(device)
    expected = torch.ops.aten._masked_softmax_backward.default(
        torch.ones_like(softmax_cpu), softmax_cpu, mask_cpu, -1,
    )
    actual = torch.ops.aten._masked_softmax_backward.default(
        torch.ones_like(softmax_cpu, device=device), softmax_dev, mask_dev, -1,
    )
    _compare_backward_tensor(actual, expected, compare, torch.float32)
    out = torch.empty_like(softmax_dev)
    actual = torch.ops.aten._masked_softmax_backward.out(
        torch.ones_like(softmax_cpu, device=device), softmax_dev, mask_dev, -1, out=out,
    )
    _compare_out_tensor(actual, out, expected, compare, torch.float32)

    fq_input_cpu = torch.linspace(-1.0, 1.0, 6, dtype=torch.float32)
    fq_input_dev = fq_input_cpu.to(device)
    expected = torch.ops.aten._fake_quantize_learnable_per_tensor_affine_backward.default(
        torch.ones_like(fq_input_cpu),
        fq_input_cpu,
        torch.tensor([0.1]),
        torch.tensor([0.0]),
        -128,
        127,
        1.0,
    )
    actual = torch.ops.aten._fake_quantize_learnable_per_tensor_affine_backward.default(
        torch.ones_like(fq_input_dev),
        fq_input_dev,
        torch.tensor([0.1], device=device),
        torch.tensor([0.0], device=device),
        -128,
        127,
        1.0,
    )
    _compare_tensor_tuple(actual, expected, compare, torch.float32)

    fq_channel_cpu = torch.linspace(-1.0, 1.0, 6, dtype=torch.float32).reshape(2, 3)
    fq_channel_dev = fq_channel_cpu.to(device)
    expected = torch.ops.aten._fake_quantize_learnable_per_channel_affine_backward.default(
        torch.ones_like(fq_channel_cpu),
        fq_channel_cpu,
        torch.full((3,), 0.1),
        torch.zeros(3),
        1,
        -128,
        127,
        1.0,
    )
    actual = torch.ops.aten._fake_quantize_learnable_per_channel_affine_backward.default(
        torch.ones_like(fq_channel_dev),
        fq_channel_dev,
        torch.full((3,), 0.1, device=device),
        torch.zeros(3, device=device),
        1,
        -128,
        127,
        1.0,
    )
    _compare_tensor_tuple(actual, expected, compare, torch.float32)


@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.requires("sparse")
@pytest.mark.covers("aten::_embedding_bag_backward")
@pytest.mark.covers("aten::_embedding_bag_dense_backward")
@pytest.mark.covers("aten::_embedding_bag_dense_backward.out", surface="out_variant")
@pytest.mark.covers("aten::_embedding_bag_per_sample_weights_backward")
@pytest.mark.covers("aten::_embedding_bag_per_sample_weights_backward.out", surface="out_variant")
@pytest.mark.covers("aten::_embedding_bag_sparse_backward")
@pytest.mark.covers("aten::_gather_sparse_backward")
@pytest.mark.covers("aten::_sparse_log_softmax_backward_data")
@pytest.mark.covers("aten::_sparse_mm_reduce_impl_backward")
@pytest.mark.covers("aten::_sparse_softmax_backward_data")
@pytest.mark.covers("aten::_sparse_sum_backward")
@pytest.mark.covers("aten::embedding_dense_backward.out", surface="out_variant")
@pytest.mark.covers("aten::embedding_sparse_backward")
def test_direct_sparse_and_embedding_backward_dispatcher_surfaces(device, compare):
    weight_cpu = torch.linspace(-1.0, 1.0, 10, dtype=torch.float32).reshape(5, 2)
    weight_dev = weight_cpu.to(device)
    indices_cpu = torch.tensor([0, 1, 2, 1], dtype=torch.int64)
    offsets_cpu = torch.tensor([0, 2, 4], dtype=torch.int64)
    indices_dev = indices_cpu.to(device)
    offsets_dev = offsets_cpu.to(device)
    offset2bag_cpu = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    offset2bag_dev = offset2bag_cpu.to(device)
    bag_size_cpu = torch.tensor([2, 2], dtype=torch.int64)
    bag_size_dev = bag_size_cpu.to(device)
    maximum_indices_cpu = torch.tensor([0, 0], dtype=torch.int64)
    maximum_indices_dev = maximum_indices_cpu.to(device)
    grad_bag_cpu = torch.ones(2, 2, dtype=torch.float32)
    grad_bag_dev = grad_bag_cpu.to(device)

    expected = torch.ops.aten._embedding_bag_backward.default(
        grad_bag_cpu,
        indices_cpu,
        offsets_cpu,
        offset2bag_cpu,
        bag_size_cpu,
        maximum_indices_cpu,
        weight_cpu.size(0),
        False,
        0,
        False,
        None,
        -1,
    )
    actual = torch.ops.aten._embedding_bag_backward.default(
        grad_bag_dev,
        indices_dev,
        offsets_dev,
        offset2bag_dev,
        bag_size_dev,
        maximum_indices_dev,
        weight_cpu.size(0),
        False,
        0,
        False,
        None,
        -1,
    )
    _compare_sparse_or_dense(actual, expected, compare)

    expected = torch.ops.aten._embedding_bag_backward.default(
        grad_bag_cpu,
        indices_cpu,
        offsets_cpu,
        offset2bag_cpu,
        bag_size_cpu,
        maximum_indices_cpu,
        weight_cpu.size(0),
        False,
        0,
        True,
        None,
        -1,
    )
    actual = torch.ops.aten._embedding_bag_backward.default(
        grad_bag_dev,
        indices_dev,
        offsets_dev,
        offset2bag_dev,
        bag_size_dev,
        maximum_indices_dev,
        weight_cpu.size(0),
        False,
        0,
        True,
        None,
        -1,
    )
    _compare_sparse_or_dense(actual, expected, compare)

    expected = torch.ops.aten._embedding_bag_dense_backward.default(
        grad_bag_cpu,
        indices_cpu,
        offset2bag_cpu,
        bag_size_cpu,
        maximum_indices_cpu,
        weight_cpu.size(0),
        False,
        0,
        None,
        -1,
    )
    actual = torch.ops.aten._embedding_bag_dense_backward.default(
        grad_bag_dev,
        indices_dev,
        offset2bag_dev,
        bag_size_dev,
        maximum_indices_dev,
        weight_cpu.size(0),
        False,
        0,
        None,
        -1,
    )
    _compare_sparse_or_dense(actual, expected, compare)
    out = torch.empty_like(weight_dev)
    actual = torch.ops.aten._embedding_bag_dense_backward.out(
        grad_bag_dev,
        indices_dev,
        offset2bag_dev,
        bag_size_dev,
        maximum_indices_dev,
        weight_cpu.size(0),
        False,
        0,
        None,
        -1,
        out=out,
    )
    _compare_out_tensor(actual, out, expected, compare, torch.float32)

    expected = torch.ops.aten._embedding_bag_sparse_backward.default(
        grad_bag_cpu,
        indices_cpu,
        offsets_cpu,
        offset2bag_cpu,
        bag_size_cpu,
        weight_cpu.size(0),
        False,
        0,
        None,
        -1,
    )
    actual = torch.ops.aten._embedding_bag_sparse_backward.default(
        grad_bag_dev,
        indices_dev,
        offsets_dev,
        offset2bag_dev,
        bag_size_dev,
        weight_cpu.size(0),
        False,
        0,
        None,
        -1,
    )
    _compare_sparse_or_dense(actual, expected, compare)

    expected = torch.ops.aten._embedding_bag_per_sample_weights_backward.default(
        grad_bag_cpu, weight_cpu, indices_cpu, offsets_cpu, offset2bag_cpu, 0, -1,
    )
    actual = torch.ops.aten._embedding_bag_per_sample_weights_backward.default(
        grad_bag_dev, weight_dev, indices_dev, offsets_dev, offset2bag_dev, 0, -1,
    )
    _compare_sparse_or_dense(actual, expected, compare)
    out = torch.empty(indices_cpu.numel(), dtype=torch.float32, device=device)
    actual = torch.ops.aten._embedding_bag_per_sample_weights_backward.out(
        grad_bag_dev,
        weight_dev,
        indices_dev,
        offsets_dev,
        offset2bag_dev,
        0,
        -1,
        out=out,
    )
    _compare_out_tensor(actual, out, expected, compare, torch.float32)

    grad_embedding_cpu = torch.ones(4, 2, dtype=torch.float32)
    grad_embedding_dev = grad_embedding_cpu.to(device)
    expected = torch.ops.aten.embedding_sparse_backward.default(
        grad_embedding_cpu, indices_cpu, weight_cpu.size(0), -1, False,
    )
    actual = torch.ops.aten.embedding_sparse_backward.default(
        grad_embedding_dev, indices_dev, weight_cpu.size(0), -1, False,
    )
    _compare_sparse_or_dense(actual, expected, compare)
    expected = torch.ops.aten.embedding_dense_backward.default(
        grad_embedding_cpu, indices_cpu, weight_cpu.size(0), -1, False,
    )
    out = torch.empty_like(weight_dev)
    actual = torch.ops.aten.embedding_dense_backward.out(
        grad_embedding_dev, indices_dev, weight_cpu.size(0), -1, False, out=out,
    )
    _compare_out_tensor(actual, out, expected, compare, torch.float32)

    gather_input_cpu = torch.linspace(-1.0, 1.0, 6, dtype=torch.float32).reshape(2, 3)
    gather_input_dev = gather_input_cpu.to(device)
    gather_index_cpu = torch.tensor([[0, 2], [1, 0]], dtype=torch.int64)
    gather_index_dev = gather_index_cpu.to(device)
    expected = torch.ops.aten._gather_sparse_backward.default(
        gather_input_cpu, 1, gather_index_cpu, torch.ones(2, 2),
    )
    actual = torch.ops.aten._gather_sparse_backward.default(
        gather_input_dev, 1, gather_index_dev, torch.ones(2, 2, device=device),
    )
    _compare_sparse_or_dense(actual, expected, compare)

    sparse_indices_cpu = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    sparse_values_cpu = torch.tensor([1.0, 2.0], dtype=torch.float32)
    sparse_cpu = torch.sparse_coo_tensor(
        sparse_indices_cpu, sparse_values_cpu, (2, 2),
    ).coalesce()
    sparse_dev = torch.sparse_coo_tensor(
        sparse_indices_cpu.to(device), sparse_values_cpu.to(device), (2, 2), device=device,
    ).coalesce()
    sparse_grad_cpu = torch.sparse_coo_tensor(
        sparse_indices_cpu, torch.ones(2), (2, 2),
    ).coalesce()
    sparse_grad_dev = torch.sparse_coo_tensor(
        sparse_indices_cpu.to(device), torch.ones(2, device=device), (2, 2), device=device,
    ).coalesce()

    softmax_cpu = torch.sparse.softmax(sparse_cpu, dim=1)
    softmax_dev = torch.sparse.softmax(sparse_dev, dim=1)
    expected = torch.ops.aten._sparse_softmax_backward_data.default(
        sparse_grad_cpu, softmax_cpu, 1, sparse_cpu,
    )
    actual = torch.ops.aten._sparse_softmax_backward_data.default(
        sparse_grad_dev, softmax_dev, 1, sparse_dev,
    )
    _compare_sparse_or_dense(actual, expected, compare)

    log_softmax_cpu = torch.sparse.log_softmax(sparse_cpu, dim=1)
    log_softmax_dev = torch.sparse.log_softmax(sparse_dev, dim=1)
    expected = torch.ops.aten._sparse_log_softmax_backward_data.default(
        sparse_grad_cpu, log_softmax_cpu, 1, sparse_cpu,
    )
    actual = torch.ops.aten._sparse_log_softmax_backward_data.default(
        sparse_grad_dev, log_softmax_dev, 1, sparse_dev,
    )
    _compare_sparse_or_dense(actual, expected, compare)

    sum_grad_cpu = torch.sparse_coo_tensor(
        torch.tensor([[0, 1]], dtype=torch.int64), torch.ones(2), (2,),
    ).coalesce()
    sum_grad_dev = torch.sparse_coo_tensor(
        torch.tensor([[0, 1]], dtype=torch.int64, device=device),
        torch.ones(2, device=device),
        (2,),
        device=device,
    ).coalesce()
    expected = torch.ops.aten._sparse_sum_backward.default(sum_grad_cpu, sparse_cpu, [1])
    actual = torch.ops.aten._sparse_sum_backward.default(sum_grad_dev, sparse_dev, [1])
    _compare_sparse_or_dense(actual, expected, compare)

    csr_cpu = torch.tensor([[1.0, 0.0], [0.0, 2.0]]).to_sparse_csr()
    csr_dev = torch.tensor([[1.0, 0.0], [0.0, 2.0]], device=device).to_sparse_csr()
    rhs_cpu = torch.eye(2)
    rhs_dev = torch.eye(2, device=device)
    reduced_cpu, arg_cpu = torch.ops.aten._sparse_mm_reduce_impl.default(
        csr_cpu, rhs_cpu, "sum",
    )
    reduced_dev, arg_dev = torch.ops.aten._sparse_mm_reduce_impl.default(
        csr_dev, rhs_dev, "sum",
    )
    expected = torch.ops.aten._sparse_mm_reduce_impl_backward.default(
        csr_cpu, torch.ones_like(reduced_cpu), rhs_cpu, "sum", arg_cpu, [True, True],
    )
    actual = torch.ops.aten._sparse_mm_reduce_impl_backward.default(
        csr_dev, torch.ones_like(reduced_dev), rhs_dev, "sum", arg_dev, [True, True],
    )
    _compare_tensor_tuple(actual, expected, compare, torch.float32)


@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.covers("aten::_add_batch_dim")
@pytest.mark.covers("aten::_fw_primal")
@pytest.mark.covers("aten::_make_dual")
@pytest.mark.covers("aten::_remove_batch_dim")
@pytest.mark.covers("aten::_unpack_dual")
def test_direct_forward_ad_and_vmap_plumbing_surfaces(device, compare):
    primal_cpu = torch.linspace(-1.0, 1.0, 6, dtype=torch.float32).reshape(2, 3)
    tangent_cpu = torch.ones_like(primal_cpu)
    primal_dev = primal_cpu.to(device)
    tangent_dev = tangent_cpu.to(device)

    with torch.autograd.forward_ad.dual_level() as level:
        dual_cpu = torch.ops.aten._make_dual.default(primal_cpu, tangent_cpu, level)
        dual_dev = torch.ops.aten._make_dual.default(primal_dev, tangent_dev, level)
        primal_part_cpu, tangent_part_cpu = torch.ops.aten._unpack_dual.default(dual_cpu, level)
        primal_part_dev, tangent_part_dev = torch.ops.aten._unpack_dual.default(dual_dev, level)
        _compare_backward_tensor(primal_part_dev, primal_part_cpu, compare, torch.float32)
        _compare_backward_tensor(tangent_part_dev, tangent_part_cpu, compare, torch.float32)

        expected = torch.ops.aten._fw_primal.default(dual_cpu, level)
        actual = torch.ops.aten._fw_primal.default(dual_dev, level)
        _compare_backward_tensor(actual, expected, compare, torch.float32)

    batched_cpu = torch.ops.aten._add_batch_dim.default(primal_cpu, 0, 1)
    batched_dev = torch.ops.aten._add_batch_dim.default(primal_dev, 0, 1)
    restored_cpu = torch.ops.aten._remove_batch_dim.default(
        batched_cpu, 1, primal_cpu.size(0), 0,
    )
    restored_dev = torch.ops.aten._remove_batch_dim.default(
        batched_dev, 1, primal_cpu.size(0), 0,
    )
    _compare_backward_tensor(restored_dev, restored_cpu, compare, torch.float32)


@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.covers("aten::log_sigmoid_backward.grad_input")
@pytest.mark.covers("aten::multi_margin_loss_backward")
@pytest.mark.covers("aten::multi_margin_loss_backward.grad_input")
@pytest.mark.covers("aten::multilabel_margin_loss_backward")
@pytest.mark.covers("aten::multilabel_margin_loss_backward.grad_input")
@pytest.mark.covers("aten::nll_loss_backward")
@pytest.mark.covers("aten::nll_loss_backward.grad_input")
@pytest.mark.covers("aten::nll_loss2d_backward")
@pytest.mark.covers("aten::nll_loss2d_backward.grad_input")
def test_direct_remaining_loss_backward_dispatcher_surfaces(device, compare):
    x_cpu = torch.linspace(-1.2, 1.2, 12, dtype=torch.float32).reshape(3, 4)
    grad_cpu = torch.linspace(0.1, 1.2, 12, dtype=torch.float32).reshape(3, 4)
    x_dev = x_cpu.to(device)
    grad_dev = grad_cpu.to(device)

    buffer_cpu = torch.ops.aten.log_sigmoid_forward.default(x_cpu)[1]
    buffer_dev = torch.ops.aten.log_sigmoid_forward.default(x_dev)[1]
    expected = torch.ops.aten.log_sigmoid_backward.default(grad_cpu, x_cpu, buffer_cpu)
    out = torch.empty_like(x_dev)
    actual = torch.ops.aten.log_sigmoid_backward.grad_input(
        grad_dev, x_dev, buffer_dev, grad_input=out,
    )
    _compare_out_tensor(actual, out, expected, compare, torch.float32)

    target_cpu = torch.tensor([1, 2, 0], dtype=torch.long)
    target_dev = target_cpu.to(device)
    expected = torch.ops.aten.multi_margin_loss_backward.default(
        torch.tensor(1.0), x_cpu, target_cpu, 1, 1.0, None, 1,
    )
    actual = torch.ops.aten.multi_margin_loss_backward.default(
        torch.tensor(1.0, device=device), x_dev, target_dev, 1, 1.0, None, 1,
    )
    _compare_backward_tensor(actual, expected, compare, torch.float32)
    out = torch.empty_like(x_dev)
    actual = torch.ops.aten.multi_margin_loss_backward.grad_input(
        torch.tensor(1.0, device=device),
        x_dev,
        target_dev,
        1,
        1.0,
        None,
        1,
        grad_input=out,
    )
    _compare_out_tensor(actual, out, expected, compare, torch.float32)

    multilabel_target_cpu = torch.tensor(
        [[1, 2, -1, -1], [0, 3, -1, -1], [2, 3, -1, -1]], dtype=torch.long,
    )
    multilabel_target_dev = multilabel_target_cpu.to(device)
    _, is_target_cpu = torch.ops.aten.multilabel_margin_loss_forward.default(
        x_cpu, multilabel_target_cpu, 1,
    )
    _, is_target_dev = torch.ops.aten.multilabel_margin_loss_forward.default(
        x_dev, multilabel_target_dev, 1,
    )
    expected = torch.ops.aten.multilabel_margin_loss_backward.default(
        torch.tensor(1.0), x_cpu, multilabel_target_cpu, 1, is_target_cpu,
    )
    actual = torch.ops.aten.multilabel_margin_loss_backward.default(
        torch.tensor(1.0, device=device),
        x_dev,
        multilabel_target_dev,
        1,
        is_target_dev,
    )
    _compare_backward_tensor(actual, expected, compare, torch.float32)
    out = torch.empty_like(x_dev)
    actual = torch.ops.aten.multilabel_margin_loss_backward.grad_input(
        torch.tensor(1.0, device=device),
        x_dev,
        multilabel_target_dev,
        1,
        is_target_dev,
        grad_input=out,
    )
    _compare_out_tensor(actual, out, expected, compare, torch.float32)

    logits_cpu = torch.log_softmax(x_cpu, dim=1)
    logits_dev = logits_cpu.to(device)
    total_weight_cpu = torch.tensor(float(target_cpu.numel()))
    total_weight_dev = total_weight_cpu.to(device)
    expected = torch.ops.aten.nll_loss_backward.default(
        torch.tensor(1.0), logits_cpu, target_cpu, None, 1, -100, total_weight_cpu,
    )
    actual = torch.ops.aten.nll_loss_backward.default(
        torch.tensor(1.0, device=device),
        logits_dev,
        target_dev,
        None,
        1,
        -100,
        total_weight_dev,
    )
    _compare_backward_tensor(actual, expected, compare, torch.float32)
    out = torch.empty_like(logits_dev)
    actual = torch.ops.aten.nll_loss_backward.grad_input(
        torch.tensor(1.0, device=device),
        logits_dev,
        target_dev,
        None,
        1,
        -100,
        total_weight_dev,
        grad_input=out,
    )
    _compare_out_tensor(actual, out, expected, compare, torch.float32)

    logits2d_cpu = torch.log_softmax(
        torch.linspace(-1.0, 1.0, 24, dtype=torch.float32).reshape(2, 3, 2, 2),
        dim=1,
    )
    target2d_cpu = torch.tensor(
        [[[0, 1], [2, 1]], [[1, 0], [2, 0]]], dtype=torch.long,
    )
    logits2d_dev = logits2d_cpu.to(device)
    target2d_dev = target2d_cpu.to(device)
    total_weight2d_cpu = torch.tensor(float(target2d_cpu.numel()))
    total_weight2d_dev = total_weight2d_cpu.to(device)
    expected = torch.ops.aten.nll_loss2d_backward.default(
        torch.tensor(1.0),
        logits2d_cpu,
        target2d_cpu,
        None,
        1,
        -100,
        total_weight2d_cpu,
    )
    actual = torch.ops.aten.nll_loss2d_backward.default(
        torch.tensor(1.0, device=device),
        logits2d_dev,
        target2d_dev,
        None,
        1,
        -100,
        total_weight2d_dev,
    )
    _compare_backward_tensor(actual, expected, compare, torch.float32)
    out = torch.empty_like(logits2d_dev)
    actual = torch.ops.aten.nll_loss2d_backward.grad_input(
        torch.tensor(1.0, device=device),
        logits2d_dev,
        target2d_dev,
        None,
        1,
        -100,
        total_weight2d_dev,
        grad_input=out,
    )
    _compare_out_tensor(actual, out, expected, compare, torch.float32)


def _segment_backward_inputs(device):
    data_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
    lengths_cpu = torch.tensor([2, 3], dtype=torch.long)
    data_dev = data_cpu.to(device)
    lengths_dev = lengths_cpu.to(device)
    out_cpu = torch.ops.aten.segment_reduce.default(data_cpu, "sum", lengths=lengths_cpu)
    out_dev = torch.ops.aten.segment_reduce.default(data_dev, "sum", lengths=lengths_dev)
    return data_cpu, lengths_cpu, data_dev, lengths_dev, out_cpu, out_dev


def _grid2_inputs(device):
    input_cpu = torch.linspace(-1.0, 1.0, 9, dtype=torch.float32).reshape(1, 1, 3, 3)
    grid_cpu = torch.tensor(
        [[[[-0.5, -0.5], [0.5, -0.5]], [[-0.5, 0.5], [0.5, 0.5]]]],
        dtype=torch.float32,
    )
    grad_cpu = torch.linspace(0.1, 0.4, 4, dtype=torch.float32).reshape(1, 1, 2, 2)
    return input_cpu, grid_cpu, grad_cpu, input_cpu.to(device), grid_cpu.to(device), grad_cpu.to(device)


def _grid3_inputs(device):
    input_cpu = torch.linspace(-1.0, 1.0, 27, dtype=torch.float32).reshape(1, 1, 3, 3, 3)
    grid_cpu = torch.zeros(1, 2, 2, 2, 3, dtype=torch.float32)
    grad_cpu = torch.linspace(0.1, 0.8, 8, dtype=torch.float32).reshape(1, 1, 2, 2, 2)
    return input_cpu, grid_cpu, grad_cpu, input_cpu.to(device), grid_cpu.to(device), grad_cpu.to(device)


def _run_segment_backward_default(device, compare):
    data_cpu, lengths_cpu, data_dev, lengths_dev, out_cpu, out_dev = _segment_backward_inputs(device)
    expected = torch.ops.aten._segment_reduce_backward.default(
        torch.ones_like(out_cpu),
        out_cpu,
        data_cpu,
        "sum",
        lengths=lengths_cpu,
    )
    actual = torch.ops.aten._segment_reduce_backward.default(
        torch.ones_like(out_dev),
        out_dev,
        data_dev,
        "sum",
        lengths=lengths_dev,
    )
    _compare_backward_tensor(actual, expected, compare, torch.float32)


def _run_segment_backward_out(device, compare):
    data_cpu, lengths_cpu, data_dev, lengths_dev, out_cpu, out_dev = _segment_backward_inputs(device)
    expected = torch.ops.aten._segment_reduce_backward.default(
        torch.ones_like(out_cpu),
        out_cpu,
        data_cpu,
        "sum",
        lengths=lengths_cpu,
    )
    out = torch.empty_like(data_dev)
    actual = torch.ops.aten._segment_reduce_backward.out(
        torch.ones_like(out_dev),
        out_dev,
        data_dev,
        "sum",
        lengths=lengths_dev,
        out=out,
    )
    _compare_out_tensor(actual, out, expected, compare, torch.float32)


def _run_grid2_cpu_fallback_default(device, compare):
    input_cpu, grid_cpu, _, input_dev, grid_dev, _ = _grid2_inputs(device)
    expected = torch.ops.aten._grid_sampler_2d_cpu_fallback.default(
        input_cpu, grid_cpu, 0, 0, False,
    )
    actual = torch.ops.aten._grid_sampler_2d_cpu_fallback.default(
        input_dev, grid_dev, 0, 0, False,
    )
    _compare_backward_tensor(actual, expected, compare, torch.float32)


def _run_grid2_cpu_fallback_out(device, compare):
    input_cpu, grid_cpu, _, input_dev, grid_dev, _ = _grid2_inputs(device)
    expected = torch.ops.aten._grid_sampler_2d_cpu_fallback.default(
        input_cpu, grid_cpu, 0, 0, False,
    )
    out = torch.empty_like(expected, device=device)
    actual = torch.ops.aten._grid_sampler_2d_cpu_fallback.out(
        input_dev, grid_dev, 0, 0, False, out=out,
    )
    _compare_out_tensor(actual, out, expected, compare, torch.float32)


def _run_grid2_cpu_fallback_backward(device, compare):
    input_cpu, grid_cpu, grad_cpu, input_dev, grid_dev, grad_dev = _grid2_inputs(device)
    expected = torch.ops.aten._grid_sampler_2d_cpu_fallback_backward.default(
        grad_cpu, input_cpu, grid_cpu, 0, 0, False,
    )
    actual = torch.ops.aten._grid_sampler_2d_cpu_fallback_backward.default(
        grad_dev, input_dev, grid_dev, 0, 0, False,
    )
    _compare_tensor_tuple(actual, expected, compare, torch.float32)


def _run_grid2_backward_default(device, compare):
    input_cpu, grid_cpu, grad_cpu, input_dev, grid_dev, grad_dev = _grid2_inputs(device)
    expected = torch.ops.aten.grid_sampler_2d_backward.default(
        grad_cpu, input_cpu, grid_cpu, 0, 0, False, [True, True],
    )
    actual = torch.ops.aten.grid_sampler_2d_backward.default(
        grad_dev, input_dev, grid_dev, 0, 0, False, [True, True],
    )
    _compare_tensor_tuple(actual, expected, compare, torch.float32)


def _run_grid2_backward_out(device, compare):
    input_cpu, grid_cpu, grad_cpu, input_dev, grid_dev, grad_dev = _grid2_inputs(device)
    expected = torch.ops.aten.grid_sampler_2d_backward.default(
        grad_cpu, input_cpu, grid_cpu, 0, 0, False, [True, True],
    )
    out0 = torch.empty_like(input_dev)
    out1 = torch.empty_like(grid_dev)
    actual = torch.ops.aten.grid_sampler_2d_backward.out(
        grad_dev,
        input_dev,
        grid_dev,
        0,
        0,
        False,
        [True, True],
        out0=out0,
        out1=out1,
    )
    assert actual[0].data_ptr() == out0.data_ptr()
    assert actual[1].data_ptr() == out1.data_ptr()
    _compare_tensor_tuple(actual, expected, compare, torch.float32)


def _run_grid3_backward_default(device, compare):
    input_cpu, grid_cpu, grad_cpu, input_dev, grid_dev, grad_dev = _grid3_inputs(device)
    expected = torch.ops.aten.grid_sampler_3d_backward.default(
        grad_cpu, input_cpu, grid_cpu, 0, 0, False, [True, True],
    )
    actual = torch.ops.aten.grid_sampler_3d_backward.default(
        grad_dev, input_dev, grid_dev, 0, 0, False, [True, True],
    )
    _compare_tensor_tuple(actual, expected, compare, torch.float32)


def _run_grid3_backward_out(device, compare):
    input_cpu, grid_cpu, grad_cpu, input_dev, grid_dev, grad_dev = _grid3_inputs(device)
    expected = torch.ops.aten.grid_sampler_3d_backward.default(
        grad_cpu, input_cpu, grid_cpu, 0, 0, False, [True, True],
    )
    out0 = torch.empty_like(input_dev)
    out1 = torch.empty_like(grid_dev)
    actual = torch.ops.aten.grid_sampler_3d_backward.out(
        grad_dev,
        input_dev,
        grid_dev,
        0,
        0,
        False,
        [True, True],
        out0=out0,
        out1=out1,
    )
    assert actual[0].data_ptr() == out0.data_ptr()
    assert actual[1].data_ptr() == out1.data_ptr()
    _compare_tensor_tuple(actual, expected, compare, torch.float32)


INTERNAL_DISPATCHER_SURFACE_CASES = {
    "segment_backward_default": _run_segment_backward_default,
    "segment_backward_out": _run_segment_backward_out,
    "grid2_cpu_fallback_default": _run_grid2_cpu_fallback_default,
    "grid2_cpu_fallback_out": _run_grid2_cpu_fallback_out,
    "grid2_cpu_fallback_backward": _run_grid2_cpu_fallback_backward,
    "grid2_backward_default": _run_grid2_backward_default,
    "grid2_backward_out": _run_grid2_backward_out,
    "grid3_backward_default": _run_grid3_backward_default,
    "grid3_backward_out": _run_grid3_backward_out,
}


@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.parametrize(
    "case_id",
    [
        pytest.param(
            "segment_backward_default",
            marks=pytest.mark.covers("aten::_segment_reduce_backward.default"),
        ),
        pytest.param(
            "segment_backward_out",
            marks=pytest.mark.covers("aten::_segment_reduce_backward.out", surface="out_variant"),
        ),
        pytest.param(
            "grid2_cpu_fallback_default",
            marks=pytest.mark.covers("aten::_grid_sampler_2d_cpu_fallback.default"),
        ),
        pytest.param(
            "grid2_cpu_fallback_out",
            marks=pytest.mark.covers("aten::_grid_sampler_2d_cpu_fallback.out", surface="out_variant"),
        ),
        pytest.param(
            "grid2_cpu_fallback_backward",
            marks=pytest.mark.covers("aten::_grid_sampler_2d_cpu_fallback_backward.default"),
        ),
        pytest.param(
            "grid2_backward_default",
            marks=pytest.mark.covers("aten::grid_sampler_2d_backward.default"),
        ),
        pytest.param(
            "grid2_backward_out",
            marks=pytest.mark.covers("aten::grid_sampler_2d_backward.out", surface="out_variant"),
        ),
        pytest.param(
            "grid3_backward_default",
            marks=pytest.mark.covers("aten::grid_sampler_3d_backward.default"),
        ),
        pytest.param(
            "grid3_backward_out",
            marks=pytest.mark.covers("aten::grid_sampler_3d_backward.out", surface="out_variant"),
        ),
    ],
)
def test_internal_dispatcher_surface(case_id, device, compare):
    INTERNAL_DISPATCHER_SURFACE_CASES[case_id](device, compare)


@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.covers("aten::affine_grid_generator_backward")
@pytest.mark.covers("aten::infinitely_differentiable_gelu_backward")
@pytest.mark.covers("aten::matrix_exp_backward")
@pytest.mark.covers("aten::to_dense_backward")
def test_direct_misc_backward_surfaces(device, compare):
    x_cpu = torch.linspace(-1.0, 1.0, 6, dtype=torch.float32).reshape(2, 3)
    grad_cpu = torch.linspace(0.2, 1.2, 6, dtype=torch.float32).reshape(2, 3)
    x_dev = x_cpu.to(device)
    grad_dev = grad_cpu.to(device)

    expected = torch.ops.aten.infinitely_differentiable_gelu_backward.default(
        grad_cpu, x_cpu,
    )
    actual = torch.ops.aten.infinitely_differentiable_gelu_backward.default(
        grad_dev, x_dev,
    )
    _compare_backward_tensor(actual, expected, compare, torch.float32)

    matrix_cpu = torch.tensor([[0.1, 0.2], [-0.3, 0.4]], dtype=torch.float32)
    matrix_grad_cpu = torch.tensor([[0.5, -0.25], [0.75, 0.125]], dtype=torch.float32)
    expected = torch.ops.aten.matrix_exp_backward.default(matrix_cpu, matrix_grad_cpu)
    actual = torch.ops.aten.matrix_exp_backward.default(
        matrix_cpu.to(device), matrix_grad_cpu.to(device),
    )
    _compare_backward_tensor(actual, expected, compare, torch.float32)

    affine_grad_cpu = torch.linspace(0.1, 0.8, 8, dtype=torch.float32).reshape(1, 2, 2, 2)
    expected = torch.ops.aten.affine_grid_generator_backward.default(
        affine_grad_cpu, [1, 1, 2, 2], False,
    )
    actual = torch.ops.aten.affine_grid_generator_backward.default(
        affine_grad_cpu.to(device), [1, 1, 2, 2], False,
    )
    _compare_backward_tensor(actual, expected, compare, torch.float32)

    indices_cpu = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    values_cpu = torch.tensor([1.0, 2.0], dtype=torch.float32)
    sparse_cpu = torch.sparse_coo_tensor(indices_cpu, values_cpu, (2, 2)).coalesce()
    sparse_dev = torch.sparse_coo_tensor(
        indices_cpu.to(device), values_cpu.to(device), (2, 2), device=device,
    ).coalesce()
    dense_grad_cpu = torch.ones(2, 2, dtype=torch.float32)
    dense_grad_dev = dense_grad_cpu.to(device)
    expected = torch.ops.aten.to_dense_backward.default(
        dense_grad_cpu, sparse_cpu, True,
    )
    actual = torch.ops.aten.to_dense_backward.default(
        dense_grad_dev, sparse_dev, True,
    )
    _compare_sparse_or_dense(actual, expected, compare)


@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.covers("aten::_convolution_double_backward")
@pytest.mark.covers("aten::_slow_conv2d_backward.grad_input")
@pytest.mark.covers("aten::_slow_conv2d_backward.output_mask")
@pytest.mark.covers("aten::_slow_conv2d_backward.output_mask_out")
@pytest.mark.covers("aten::_thnn_differentiable_gru_cell_backward")
@pytest.mark.covers("aten::_thnn_differentiable_lstm_cell_backward")
@pytest.mark.covers("aten::conv_tbc_backward")
@pytest.mark.covers("aten::convolution_backward")
@pytest.mark.covers("aten::convolution_backward.out", surface="out_variant")
def test_direct_convolution_and_rnn_backward_surfaces(device, compare):
    input_cpu = torch.linspace(-1.0, 1.0, 50, dtype=torch.float32).reshape(1, 2, 5, 5)
    weight_cpu = torch.linspace(-0.5, 0.5, 54, dtype=torch.float32).reshape(3, 2, 3, 3)
    grad_cpu = torch.linspace(0.1, 1.0, 75, dtype=torch.float32).reshape(1, 3, 5, 5)
    input_dev = input_cpu.to(device)
    weight_dev = weight_cpu.to(device)
    grad_dev = grad_cpu.to(device)
    bias_sizes = [3]
    stride = [1, 1]
    padding = [1, 1]
    dilation = [1, 1]
    output_padding = [0, 0]
    mask = [True, True, True]

    expected = torch.ops.aten.convolution_backward.default(
        grad_cpu,
        input_cpu,
        weight_cpu,
        bias_sizes,
        stride,
        padding,
        dilation,
        False,
        output_padding,
        1,
        mask,
    )
    actual = torch.ops.aten.convolution_backward.default(
        grad_dev,
        input_dev,
        weight_dev,
        bias_sizes,
        stride,
        padding,
        dilation,
        False,
        output_padding,
        1,
        mask,
    )
    _compare_tensor_tuple(actual, expected, compare, torch.float32)
    out0 = torch.empty_like(input_dev)
    out1 = torch.empty_like(weight_dev)
    out2 = torch.empty(3, dtype=torch.float32, device=device)
    actual = torch.ops.aten.convolution_backward.out(
        grad_dev,
        input_dev,
        weight_dev,
        bias_sizes,
        stride,
        padding,
        dilation,
        False,
        output_padding,
        1,
        mask,
        out0=out0,
        out1=out1,
        out2=out2,
    )
    assert actual[0].data_ptr() == out0.data_ptr()
    assert actual[1].data_ptr() == out1.data_ptr()
    assert actual[2].data_ptr() == out2.data_ptr()
    _compare_tensor_tuple(actual, expected, compare, torch.float32)

    expected = torch.ops.aten._convolution_double_backward.default(
        input_cpu,
        weight_cpu,
        torch.ones(3, dtype=torch.float32),
        grad_cpu,
        weight_cpu,
        input_cpu,
        stride,
        padding,
        dilation,
        False,
        output_padding,
        1,
        mask,
    )
    actual = torch.ops.aten._convolution_double_backward.default(
        input_dev,
        weight_dev,
        torch.ones(3, dtype=torch.float32, device=device),
        grad_dev,
        weight_dev,
        input_dev,
        stride,
        padding,
        dilation,
        False,
        output_padding,
        1,
        mask,
    )
    _compare_tensor_tuple(actual, expected, compare, torch.float32)

    expected = torch.ops.aten._slow_conv2d_backward.output_mask(
        grad_cpu, input_cpu, weight_cpu, [3, 3], stride, padding, mask,
    )
    actual = torch.ops.aten._slow_conv2d_backward.output_mask(
        grad_dev, input_dev, weight_dev, [3, 3], stride, padding, mask,
    )
    _compare_tensor_tuple(actual, expected, compare, torch.float32)
    out0 = torch.empty_like(input_dev)
    out1 = torch.empty_like(weight_dev)
    out2 = torch.empty(3, dtype=torch.float32, device=device)
    actual = torch.ops.aten._slow_conv2d_backward.grad_input(
        grad_dev,
        input_dev,
        weight_dev,
        [3, 3],
        stride,
        padding,
        grad_input=out0,
        grad_weight=out1,
        grad_bias=out2,
    )
    _compare_tensor_tuple(actual, expected, compare, torch.float32)
    out0 = torch.empty_like(input_dev)
    out1 = torch.empty_like(weight_dev)
    out2 = torch.empty(3, dtype=torch.float32, device=device)
    actual = torch.ops.aten._slow_conv2d_backward.output_mask_out(
        grad_dev,
        input_dev,
        weight_dev,
        [3, 3],
        stride,
        padding,
        mask,
        out0=out0,
        out1=out1,
        out2=out2,
    )
    _compare_tensor_tuple(actual, expected, compare, torch.float32)

    tbc_input_cpu = torch.linspace(-1.0, 1.0, 30, dtype=torch.float32).reshape(5, 2, 3)
    tbc_weight_cpu = torch.linspace(-0.5, 0.5, 36, dtype=torch.float32).reshape(3, 3, 4)
    tbc_bias_cpu = torch.linspace(-0.2, 0.2, 4, dtype=torch.float32)
    tbc_grad_cpu = torch.linspace(0.1, 1.0, 40, dtype=torch.float32).reshape(5, 2, 4)
    expected = torch.ops.aten.conv_tbc_backward.default(
        tbc_grad_cpu, tbc_input_cpu, tbc_weight_cpu, tbc_bias_cpu, 1,
    )
    actual = torch.ops.aten.conv_tbc_backward.default(
        tbc_grad_cpu.to(device),
        tbc_input_cpu.to(device),
        tbc_weight_cpu.to(device),
        tbc_bias_cpu.to(device),
        1,
    )
    _compare_tensor_tuple(actual, expected, compare, torch.float32)

    batch = 2
    hidden = 3
    grad_hy_cpu = torch.linspace(0.1, 0.6, batch * hidden, dtype=torch.float32).reshape(
        batch, hidden,
    )
    input_gates_gru_cpu = torch.linspace(
        -1.0, 1.0, batch * 3 * hidden, dtype=torch.float32,
    ).reshape(batch, 3 * hidden)
    hidden_gates_gru_cpu = torch.linspace(
        0.5, -0.5, batch * 3 * hidden, dtype=torch.float32,
    ).reshape(batch, 3 * hidden)
    hx_cpu = torch.linspace(-0.25, 0.25, batch * hidden, dtype=torch.float32).reshape(
        batch, hidden,
    )
    input_bias_gru_cpu = torch.linspace(-0.1, 0.1, 3 * hidden, dtype=torch.float32)
    hidden_bias_gru_cpu = torch.linspace(0.2, -0.2, 3 * hidden, dtype=torch.float32)
    expected = torch.ops.aten._thnn_differentiable_gru_cell_backward.default(
        grad_hy_cpu,
        input_gates_gru_cpu,
        hidden_gates_gru_cpu,
        hx_cpu,
        input_bias_gru_cpu,
        hidden_bias_gru_cpu,
    )
    actual = torch.ops.aten._thnn_differentiable_gru_cell_backward.default(
        grad_hy_cpu.to(device),
        input_gates_gru_cpu.to(device),
        hidden_gates_gru_cpu.to(device),
        hx_cpu.to(device),
        input_bias_gru_cpu.to(device),
        hidden_bias_gru_cpu.to(device),
    )
    _compare_tensor_tuple(actual, expected, compare, torch.float32)

    grad_cy_cpu = torch.linspace(0.2, 0.7, batch * hidden, dtype=torch.float32).reshape(
        batch, hidden,
    )
    input_gates_lstm_cpu = torch.linspace(
        -1.0, 1.0, batch * 4 * hidden, dtype=torch.float32,
    ).reshape(batch, 4 * hidden)
    hidden_gates_lstm_cpu = torch.linspace(
        0.75, -0.75, batch * 4 * hidden, dtype=torch.float32,
    ).reshape(batch, 4 * hidden)
    input_bias_lstm_cpu = torch.linspace(-0.1, 0.1, 4 * hidden, dtype=torch.float32)
    hidden_bias_lstm_cpu = torch.linspace(0.2, -0.2, 4 * hidden, dtype=torch.float32)
    cy_cpu = torch.linspace(-0.3, 0.3, batch * hidden, dtype=torch.float32).reshape(
        batch, hidden,
    )
    expected = torch.ops.aten._thnn_differentiable_lstm_cell_backward.default(
        grad_hy_cpu,
        grad_cy_cpu,
        input_gates_lstm_cpu,
        hidden_gates_lstm_cpu,
        input_bias_lstm_cpu,
        hidden_bias_lstm_cpu,
        hx_cpu,
        cy_cpu,
    )
    actual = torch.ops.aten._thnn_differentiable_lstm_cell_backward.default(
        grad_hy_cpu.to(device),
        grad_cy_cpu.to(device),
        input_gates_lstm_cpu.to(device),
        hidden_gates_lstm_cpu.to(device),
        input_bias_lstm_cpu.to(device),
        hidden_bias_lstm_cpu.to(device),
        hx_cpu.to(device),
        cy_cpu.to(device),
    )
    _compare_tensor_tuple(actual, expected, compare, torch.float32)
