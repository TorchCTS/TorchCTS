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

NORM_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _compare_tensor_tuple(actual, expected, compare, *, category, dtype=torch.float32):
    assert len(actual) == len(expected)
    for actual_item, expected_item in zip(actual, expected):
        compare(actual_item, expected_item, category=category, dtype=dtype)


def _assert_returned_outputs(returned, outputs):
    assert len(returned) == len(outputs)
    for returned_item, output_item in zip(returned, outputs):
        assert returned_item is output_item


@pytest.mark.smoke
@pytest.mark.covers("aten::layer_norm")
@pytest.mark.parametrize("dtype", NORM_DTYPES)
def test_layer_norm(dtype, device, compare, input_gen):
    shape = (4, 16, 32)
    normalized_shape = (32,)
    
    x_dev = input_gen(shape, dtype, device)
    weight_dev = input_gen(normalized_shape, dtype, device)
    bias_dev = input_gen(normalized_shape, dtype, device)
    
    expected = torch.nn.functional.layer_norm(x_dev.cpu(), normalized_shape, weight_dev.cpu(), bias_dev.cpu())
    actual = torch.nn.functional.layer_norm(x_dev, normalized_shape, weight_dev, bias_dev)
    synchronize(device)
    
    compare(actual, expected, category="norm", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.covers("aten::group_norm")
@pytest.mark.parametrize("dtype", NORM_DTYPES)
def test_group_norm(dtype, device, compare, input_gen):
    shape = (2, 8, 16, 16)
    num_groups = 4
    num_channels = 8
    
    x_dev = input_gen(shape, dtype, device)
    weight_dev = input_gen((num_channels,), dtype, device)
    bias_dev = input_gen((num_channels,), dtype, device)
    
    x_cpu = x_dev.cpu()
    w_cpu = weight_dev.cpu()
    b_cpu = bias_dev.cpu()
    if dtype in (torch.float16, torch.bfloat16):
        x_cpu = x_cpu.float()
        w_cpu = w_cpu.float()
        b_cpu = b_cpu.float()
        
    expected = torch.nn.functional.group_norm(x_cpu, num_groups, w_cpu, b_cpu)
    if dtype == torch.float16:
        expected = expected.half()
    elif dtype == torch.bfloat16:
        expected = expected.to(torch.bfloat16)
        
    actual = torch.nn.functional.group_norm(x_dev, num_groups, weight_dev, bias_dev)
    synchronize(device)
    
    compare(actual, expected, category="norm", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.covers("aten::batch_norm")
@pytest.mark.parametrize("dtype", NORM_DTYPES)
def test_batch_norm(dtype, device, compare, input_gen):
    if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
        pytest.skip("CPU batch_norm doesn't support half/bfloat16 in eager PyTorch")
        
    shape = (4, 8, 16, 16)
    num_features = 8
    
    x_dev = input_gen(shape, dtype, device)
    running_mean_dev = torch.zeros(num_features, dtype=torch.float32, device=device)
    running_var_dev = torch.ones(num_features, dtype=torch.float32, device=device)
    weight_dev = input_gen((num_features,), dtype, device)
    bias_dev = input_gen((num_features,), dtype, device)
    
    running_mean_cpu = running_mean_dev.cpu()
    running_var_cpu = running_var_dev.cpu()
    
    x_cpu = x_dev.cpu()
    w_cpu = weight_dev.cpu()
    b_cpu = bias_dev.cpu()
    
    if dtype in (torch.float16, torch.bfloat16):
        x_cpu = x_cpu.float()
        w_cpu = w_cpu.float()
        b_cpu = b_cpu.float()
        
    expected = torch.nn.functional.batch_norm(
        x_cpu, running_mean_cpu, running_var_cpu, w_cpu, b_cpu, training=True
    )
    if dtype == torch.float16:
        expected = expected.half()
    elif dtype == torch.bfloat16:
        expected = expected.to(torch.bfloat16)
        
    actual = torch.nn.functional.batch_norm(
        x_dev, running_mean_dev, running_var_dev, weight_dev, bias_dev, training=True
    )
    synchronize(device)
    
    compare(actual, expected, category="norm", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.covers("aten::instance_norm")
@pytest.mark.parametrize("dtype", NORM_DTYPES)
def test_instance_norm(dtype, device, compare, input_gen):
    shape = (2, 4, 16, 16)
    x_dev = input_gen(shape, dtype, device)
    
    x_cpu = x_dev.cpu()
    if dtype in (torch.float16, torch.bfloat16):
        x_cpu = x_cpu.float()
        
    expected = torch.nn.functional.instance_norm(x_cpu, use_input_stats=True)
    if dtype == torch.float16:
        expected = expected.half()
    elif dtype == torch.bfloat16:
        expected = expected.to(torch.bfloat16)
        
    actual = torch.nn.functional.instance_norm(x_dev, use_input_stats=True)
    synchronize(device)
    
    compare(actual, expected, category="norm", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.covers("aten::add.Tensor")
@pytest.mark.covers("aten::mean.dim")
@pytest.mark.covers("aten::mul.Tensor")
@pytest.mark.covers("aten::pow.Tensor_Scalar")
@pytest.mark.covers("aten::rsqrt")
@pytest.mark.parametrize("dtype", NORM_DTYPES)
def test_rms_norm_custom(dtype, device, compare, input_gen):
    shape = (4, 16, 32)
    eps = 1e-6
    
    x_dev = input_gen(shape, dtype, device)
    weight_dev = input_gen((32,), dtype, device)
    
    x_cpu = x_dev.cpu()
    w_cpu = weight_dev.cpu()
    
    variance = x_cpu.pow(2).mean(-1, keepdim=True)
    expected = x_cpu * torch.rsqrt(variance + eps) * w_cpu
    
    var_dev = x_dev.pow(2).mean(-1, keepdim=True)
    actual = x_dev * torch.rsqrt(var_dev + eps) * weight_dev
    synchronize(device)
    
    compare(actual, expected, category="norm", dtype=dtype)


@pytest.mark.smoke
@pytest.mark.covers("aten::native_layer_norm.out")
@pytest.mark.covers("aten::native_layer_norm_backward")
@pytest.mark.covers("aten::native_layer_norm_backward.out")
def test_native_layer_norm_dispatcher_variants(device, compare):
    input_cpu = torch.linspace(-1.5, 1.5, steps=24, dtype=torch.float32).reshape(2, 3, 4)
    weight_cpu = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float32)
    bias_cpu = torch.tensor([-0.25, 0.0, 0.25, 0.5], dtype=torch.float32)
    input_dev = input_cpu.to(device)
    weight_dev = weight_cpu.to(device)
    bias_dev = bias_cpu.to(device)

    expected = torch.ops.aten.native_layer_norm(input_cpu, [4], weight_cpu, bias_cpu, 1e-5)
    actual = torch.ops.aten.native_layer_norm(input_dev, [4], weight_dev, bias_dev, 1e-5)
    synchronize(device)

    out_cpu = [torch.empty_like(expected[0]), torch.empty_like(expected[1]), torch.empty_like(expected[2])]
    out_dev = [torch.empty_like(actual[0]), torch.empty_like(actual[1]), torch.empty_like(actual[2])]
    expected_return = torch.ops.aten.native_layer_norm.out(
        input_cpu,
        [4],
        weight_cpu,
        bias_cpu,
        1e-5,
        out0=out_cpu[0],
        out1=out_cpu[1],
        out2=out_cpu[2],
    )
    actual_return = torch.ops.aten.native_layer_norm.out(
        input_dev,
        [4],
        weight_dev,
        bias_dev,
        1e-5,
        out0=out_dev[0],
        out1=out_dev[1],
        out2=out_dev[2],
    )
    synchronize(device)
    _assert_returned_outputs(expected_return, out_cpu)
    _assert_returned_outputs(actual_return, out_dev)
    _compare_tensor_tuple(out_dev, out_cpu, compare, category="norm")

    grad_cpu = torch.linspace(0.25, 2.5, steps=24, dtype=torch.float32).reshape_as(input_cpu)
    grad_dev = grad_cpu.to(device)
    expected_backward = torch.ops.aten.native_layer_norm_backward(
        grad_cpu,
        input_cpu,
        [4],
        expected[1],
        expected[2],
        weight_cpu,
        bias_cpu,
        [True, True, True],
    )
    actual_backward = torch.ops.aten.native_layer_norm_backward(
        grad_dev,
        input_dev,
        [4],
        actual[1],
        actual[2],
        weight_dev,
        bias_dev,
        [True, True, True],
    )
    synchronize(device)
    _compare_tensor_tuple(actual_backward, expected_backward, compare, category="backward")

    out_cpu = [torch.empty_like(item) for item in expected_backward]
    out_dev = [torch.empty_like(item) for item in actual_backward]
    expected_return = torch.ops.aten.native_layer_norm_backward.out(
        grad_cpu,
        input_cpu,
        [4],
        expected[1],
        expected[2],
        weight_cpu,
        bias_cpu,
        [True, True, True],
        out0=out_cpu[0],
        out1=out_cpu[1],
        out2=out_cpu[2],
    )
    actual_return = torch.ops.aten.native_layer_norm_backward.out(
        grad_dev,
        input_dev,
        [4],
        actual[1],
        actual[2],
        weight_dev,
        bias_dev,
        [True, True, True],
        out0=out_dev[0],
        out1=out_dev[1],
        out2=out_dev[2],
    )
    synchronize(device)
    _assert_returned_outputs(expected_return, out_cpu)
    _assert_returned_outputs(actual_return, out_dev)
    _compare_tensor_tuple(out_dev, out_cpu, compare, category="backward")


@pytest.mark.smoke
@pytest.mark.covers("aten::native_group_norm")
@pytest.mark.covers("aten::native_group_norm.out")
@pytest.mark.covers("aten::native_group_norm_backward")
@pytest.mark.covers("aten::native_group_norm_backward.out")
def test_native_group_norm_dispatcher_variants(device, compare):
    input_cpu = torch.linspace(-2.0, 2.0, steps=24, dtype=torch.float32).reshape(2, 4, 3)
    weight_cpu = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float32)
    bias_cpu = torch.tensor([-0.5, -0.25, 0.25, 0.5], dtype=torch.float32)
    input_dev = input_cpu.to(device)
    weight_dev = weight_cpu.to(device)
    bias_dev = bias_cpu.to(device)

    expected = torch.ops.aten.native_group_norm(input_cpu, weight_cpu, bias_cpu, 2, 4, 3, 2, 1e-5)
    actual = torch.ops.aten.native_group_norm(input_dev, weight_dev, bias_dev, 2, 4, 3, 2, 1e-5)
    synchronize(device)
    _compare_tensor_tuple(actual, expected, compare, category="norm")

    out_cpu = [torch.empty_like(item) for item in expected]
    out_dev = [torch.empty_like(item) for item in actual]
    expected_return = torch.ops.aten.native_group_norm.out(
        input_cpu,
        weight_cpu,
        bias_cpu,
        2,
        4,
        3,
        2,
        1e-5,
        out0=out_cpu[0],
        out1=out_cpu[1],
        out2=out_cpu[2],
    )
    actual_return = torch.ops.aten.native_group_norm.out(
        input_dev,
        weight_dev,
        bias_dev,
        2,
        4,
        3,
        2,
        1e-5,
        out0=out_dev[0],
        out1=out_dev[1],
        out2=out_dev[2],
    )
    synchronize(device)
    _assert_returned_outputs(expected_return, out_cpu)
    _assert_returned_outputs(actual_return, out_dev)
    _compare_tensor_tuple(out_dev, out_cpu, compare, category="norm")

    grad_cpu = torch.linspace(0.25, 2.5, steps=24, dtype=torch.float32).reshape_as(input_cpu)
    grad_dev = grad_cpu.to(device)
    expected_backward = torch.ops.aten.native_group_norm_backward(
        grad_cpu,
        input_cpu,
        expected[1],
        expected[2],
        weight_cpu,
        2,
        4,
        3,
        2,
        [True, True, True],
    )
    actual_backward = torch.ops.aten.native_group_norm_backward(
        grad_dev,
        input_dev,
        actual[1],
        actual[2],
        weight_dev,
        2,
        4,
        3,
        2,
        [True, True, True],
    )
    synchronize(device)
    _compare_tensor_tuple(actual_backward, expected_backward, compare, category="backward")

    out_cpu = [torch.empty_like(item) for item in expected_backward]
    out_dev = [torch.empty_like(item) for item in actual_backward]
    expected_return = torch.ops.aten.native_group_norm_backward.out(
        grad_cpu,
        input_cpu,
        expected[1],
        expected[2],
        weight_cpu,
        2,
        4,
        3,
        2,
        [True, True, True],
        out0=out_cpu[0],
        out1=out_cpu[1],
        out2=out_cpu[2],
    )
    actual_return = torch.ops.aten.native_group_norm_backward.out(
        grad_dev,
        input_dev,
        actual[1],
        actual[2],
        weight_dev,
        2,
        4,
        3,
        2,
        [True, True, True],
        out0=out_dev[0],
        out1=out_dev[1],
        out2=out_dev[2],
    )
    synchronize(device)
    _assert_returned_outputs(expected_return, out_cpu)
    _assert_returned_outputs(actual_return, out_dev)
    _compare_tensor_tuple(out_dev, out_cpu, compare, category="backward")


@pytest.mark.smoke
@pytest.mark.covers("aten::_weight_norm")
@pytest.mark.covers("aten::_weight_norm_interface")
@pytest.mark.covers("aten::_weight_norm_interface.out")
@pytest.mark.covers("aten::_weight_norm_interface_backward")
@pytest.mark.covers("aten::_weight_norm_interface_backward.out")
@pytest.mark.covers("aten::_weight_norm_differentiable_backward")
def test_weight_norm_dispatcher_variants(device, compare):
    v_cpu = torch.linspace(-1.5, 1.5, steps=12, dtype=torch.float32).reshape(3, 4)
    g_cpu = torch.tensor([[1.0, 1.5, 2.0, 2.5]], dtype=torch.float32)
    v_dev = v_cpu.to(device)
    g_dev = g_cpu.to(device)

    expected_weight = torch.ops.aten._weight_norm(v_cpu, g_cpu, 1)
    actual_weight = torch.ops.aten._weight_norm(v_dev, g_dev, 1)
    synchronize(device)
    compare(actual_weight, expected_weight, category="norm", dtype=torch.float32)

    expected_interface = torch.ops.aten._weight_norm_interface(v_cpu, g_cpu, 1)
    actual_interface = torch.ops.aten._weight_norm_interface(v_dev, g_dev, 1)
    synchronize(device)
    _compare_tensor_tuple(actual_interface, expected_interface, compare, category="norm")

    out_cpu = [torch.empty_like(item) for item in expected_interface]
    out_dev = [torch.empty_like(item) for item in actual_interface]
    expected_return = torch.ops.aten._weight_norm_interface.out(v_cpu, g_cpu, 1, out0=out_cpu[0], out1=out_cpu[1])
    actual_return = torch.ops.aten._weight_norm_interface.out(v_dev, g_dev, 1, out0=out_dev[0], out1=out_dev[1])
    synchronize(device)
    _assert_returned_outputs(expected_return, out_cpu)
    _assert_returned_outputs(actual_return, out_dev)
    _compare_tensor_tuple(out_dev, out_cpu, compare, category="norm")

    grad_cpu = torch.linspace(0.25, 1.75, steps=12, dtype=torch.float32).reshape_as(v_cpu)
    grad_dev = grad_cpu.to(device)
    expected_backward = torch.ops.aten._weight_norm_interface_backward(
        grad_cpu,
        v_cpu,
        g_cpu,
        expected_interface[1],
        1,
    )
    actual_backward = torch.ops.aten._weight_norm_interface_backward(
        grad_dev,
        v_dev,
        g_dev,
        actual_interface[1],
        1,
    )
    synchronize(device)
    _compare_tensor_tuple(actual_backward, expected_backward, compare, category="backward")

    out_cpu = [torch.empty_like(item) for item in expected_backward]
    out_dev = [torch.empty_like(item) for item in actual_backward]
    expected_return = torch.ops.aten._weight_norm_interface_backward.out(
        grad_cpu,
        v_cpu,
        g_cpu,
        expected_interface[1],
        1,
        out0=out_cpu[0],
        out1=out_cpu[1],
    )
    actual_return = torch.ops.aten._weight_norm_interface_backward.out(
        grad_dev,
        v_dev,
        g_dev,
        actual_interface[1],
        1,
        out0=out_dev[0],
        out1=out_dev[1],
    )
    synchronize(device)
    _assert_returned_outputs(expected_return, out_cpu)
    _assert_returned_outputs(actual_return, out_dev)
    _compare_tensor_tuple(out_dev, out_cpu, compare, category="backward")

    expected_differentiable = torch.ops.aten._weight_norm_differentiable_backward(
        grad_cpu,
        v_cpu,
        g_cpu,
        expected_interface[1],
        1,
    )
    actual_differentiable = torch.ops.aten._weight_norm_differentiable_backward(
        grad_dev,
        v_dev,
        g_dev,
        actual_interface[1],
        1,
    )
    synchronize(device)
    _compare_tensor_tuple(actual_differentiable, expected_differentiable, compare, category="backward")


@pytest.mark.smoke
@pytest.mark.covers("aten::_fused_rms_norm")
def test_fused_rms_norm_dispatcher_variant(device, compare):
    input_cpu = torch.linspace(-1.0, 1.0, steps=8, dtype=torch.float32).reshape(2, 4)
    weight_cpu = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float32)
    input_dev = input_cpu.to(device)
    weight_dev = weight_cpu.to(device)

    expected = torch.ops.aten._fused_rms_norm(input_cpu, [4], weight_cpu, 1e-5)
    actual = torch.ops.aten._fused_rms_norm(input_dev, [4], weight_dev, 1e-5)
    synchronize(device)
    _compare_tensor_tuple(actual, expected, compare, category="norm")
