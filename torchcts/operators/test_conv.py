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

CONV_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _compare_tensor_tuple(actual, expected, compare, dtype):
    assert len(actual) == len(expected)
    for actual_tensor, expected_tensor in zip(actual, expected):
        synchronize(actual_tensor.device.type)
        compare(actual_tensor, expected_tensor, category="conv", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.covers("aten::conv1d")
@pytest.mark.parametrize("dtype", CONV_DTYPES)
def test_conv1d(dtype, device, compare, input_gen):
    # Input: (batch, in_channels, width)
    x_dev = input_gen((2, 3, 32), dtype, device)
    weight_dev = input_gen((4, 3, 3), dtype, device)
    bias_dev = input_gen((4,), dtype, device)
    
    expected = torch.nn.functional.conv1d(x_dev.cpu(), weight_dev.cpu(), bias_dev.cpu(), stride=1, padding=1)
    actual = torch.nn.functional.conv1d(x_dev, weight_dev, bias_dev, stride=1, padding=1)
    synchronize(device)
    
    compare(actual, expected, category="conv", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.covers("aten::convolution")
@pytest.mark.parametrize("dtype", CONV_DTYPES)
def test_conv2d(dtype, device, compare, input_gen):
    # Input: (batch, in_channels, height, width)
    x_dev = input_gen((2, 3, 16, 16), dtype, device)
    weight_dev = input_gen((8, 3, 3, 3), dtype, device)
    bias_dev = input_gen((8,), dtype, device)
    
    expected = torch.nn.functional.conv2d(x_dev.cpu(), weight_dev.cpu(), bias_dev.cpu(), stride=1, padding=1)
    actual = torch.nn.functional.conv2d(x_dev, weight_dev, bias_dev, stride=1, padding=1)
    synchronize(device)
    
    compare(actual, expected, category="conv", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.covers("aten::conv3d")
@pytest.mark.parametrize("dtype", CONV_DTYPES)
def test_conv3d(dtype, device, compare, input_gen):
    # Input: (batch, in_channels, depth, height, width)
    x_dev = input_gen((2, 2, 8, 8, 8), dtype, device)
    weight_dev = input_gen((4, 2, 3, 3, 3), dtype, device)
    bias_dev = input_gen((4,), dtype, device)
    
    expected = torch.nn.functional.conv3d(x_dev.cpu(), weight_dev.cpu(), bias_dev.cpu(), stride=1, padding=1)
    actual = torch.nn.functional.conv3d(x_dev, weight_dev, bias_dev, stride=1, padding=1)
    synchronize(device)
    
    compare(actual, expected, category="conv", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.covers("aten::conv_transpose2d.input")
@pytest.mark.parametrize("dtype", CONV_DTYPES)
def test_conv_transpose2d(dtype, device, compare, input_gen):
    # Input: (batch, in_channels, height, width)
    x_dev = input_gen((2, 4, 8, 8), dtype, device)
    weight_dev = input_gen((4, 8, 3, 3), dtype, device)
    bias_dev = input_gen((8,), dtype, device)
    
    expected = torch.nn.functional.conv_transpose2d(x_dev.cpu(), weight_dev.cpu(), bias_dev.cpu(), stride=1, padding=1)
    actual = torch.nn.functional.conv_transpose2d(x_dev, weight_dev, bias_dev, stride=1, padding=1)
    synchronize(device)
    
    compare(actual, expected, category="conv", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.covers("aten::convolution")
@pytest.mark.parametrize("dtype", CONV_DTYPES)
def test_conv_depthwise_groups(dtype, device, compare, input_gen):
    # Depthwise: groups = in_channels
    x_dev = input_gen((2, 4, 16, 16), dtype, device)
    weight_dev = input_gen((4, 1, 3, 3), dtype, device) # out_channels = 4, groups = 4
    
    expected = torch.nn.functional.conv2d(x_dev.cpu(), weight_dev.cpu(), stride=1, padding=1, groups=4)
    actual = torch.nn.functional.conv2d(x_dev, weight_dev, stride=1, padding=1, groups=4)
    synchronize(device)
    
    compare(actual, expected, category="conv", dtype=dtype)


@pytest.mark.smoke
@pytest.mark.covers("aten::_convolution_mode")
@pytest.mark.covers("aten::conv_tbc")
@pytest.mark.covers("aten::conv_tbc.out", surface="out_variant")
@pytest.mark.covers("aten::conv_transpose1d")
@pytest.mark.covers("aten::conv_transpose3d.input")
@pytest.mark.covers("aten::slow_conv_transpose2d")
@pytest.mark.covers("aten::slow_conv_transpose2d.out", surface="out_variant")
@pytest.mark.covers("aten::slow_conv_transpose3d")
@pytest.mark.covers("aten::slow_conv_transpose3d.out", surface="out_variant")
def test_direct_remaining_convolution_dispatcher_surfaces(device, compare):
    input2_cpu = torch.linspace(-1.0, 1.0, 50, dtype=torch.float32).reshape(1, 2, 5, 5)
    weight2_cpu = torch.linspace(-0.5, 0.5, 54, dtype=torch.float32).reshape(3, 2, 3, 3)
    bias2_cpu = torch.linspace(-0.1, 0.1, 3, dtype=torch.float32)
    expected = torch.ops.aten._convolution_mode.default(
        input2_cpu, weight2_cpu, bias2_cpu, [1, 1], "same", [1, 1], 1,
    )
    actual = torch.ops.aten._convolution_mode.default(
        input2_cpu.to(device),
        weight2_cpu.to(device),
        bias2_cpu.to(device),
        [1, 1],
        "same",
        [1, 1],
        1,
    )
    synchronize(device)
    compare(actual, expected, category="conv", dtype=torch.float32)

    input1_cpu = torch.linspace(-1.0, 1.0, 10, dtype=torch.float32).reshape(1, 2, 5)
    weight1_cpu = torch.linspace(-0.5, 0.5, 18, dtype=torch.float32).reshape(2, 3, 3)
    bias1_cpu = torch.linspace(-0.1, 0.1, 3, dtype=torch.float32)
    expected = torch.ops.aten.conv_transpose1d.default(
        input1_cpu, weight1_cpu, bias1_cpu, [1], [1], [0], 1, [1],
    )
    actual = torch.ops.aten.conv_transpose1d.default(
        input1_cpu.to(device),
        weight1_cpu.to(device),
        bias1_cpu.to(device),
        [1],
        [1],
        [0],
        1,
        [1],
    )
    synchronize(device)
    compare(actual, expected, category="conv", dtype=torch.float32)

    trans_weight2_cpu = torch.linspace(-0.5, 0.5, 54, dtype=torch.float32).reshape(
        2, 3, 3, 3,
    )
    expected = torch.ops.aten.slow_conv_transpose2d.default(
        input2_cpu,
        trans_weight2_cpu,
        [3, 3],
        bias2_cpu,
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 1],
    )
    actual = torch.ops.aten.slow_conv_transpose2d.default(
        input2_cpu.to(device),
        trans_weight2_cpu.to(device),
        [3, 3],
        bias2_cpu.to(device),
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 1],
    )
    synchronize(device)
    compare(actual, expected, category="conv", dtype=torch.float32)
    out = torch.empty_like(expected, device=device)
    actual = torch.ops.aten.slow_conv_transpose2d.out(
        input2_cpu.to(device),
        trans_weight2_cpu.to(device),
        [3, 3],
        bias2_cpu.to(device),
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 1],
        out=out,
    )
    assert actual.data_ptr() == out.data_ptr()
    synchronize(device)
    compare(out, expected, category="conv", dtype=torch.float32)

    input3_cpu = torch.linspace(-1.0, 1.0, 128, dtype=torch.float32).reshape(
        1, 2, 4, 4, 4,
    )
    weight3_cpu = torch.linspace(-0.5, 0.5, 162, dtype=torch.float32).reshape(
        2, 3, 3, 3, 3,
    )
    bias3_cpu = torch.linspace(-0.1, 0.1, 3, dtype=torch.float32)
    expected = torch.ops.aten.conv_transpose3d.input(
        input3_cpu,
        weight3_cpu,
        bias3_cpu,
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 0],
        1,
        [1, 1, 1],
    )
    actual = torch.ops.aten.conv_transpose3d.input(
        input3_cpu.to(device),
        weight3_cpu.to(device),
        bias3_cpu.to(device),
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 0],
        1,
        [1, 1, 1],
    )
    synchronize(device)
    compare(actual, expected, category="conv", dtype=torch.float32)

    expected = torch.ops.aten.slow_conv_transpose3d.default(
        input3_cpu,
        weight3_cpu,
        [3, 3, 3],
        bias3_cpu,
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
    )
    actual = torch.ops.aten.slow_conv_transpose3d.default(
        input3_cpu.to(device),
        weight3_cpu.to(device),
        [3, 3, 3],
        bias3_cpu.to(device),
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
    )
    synchronize(device)
    compare(actual, expected, category="conv", dtype=torch.float32)
    out = torch.empty_like(expected, device=device)
    actual = torch.ops.aten.slow_conv_transpose3d.out(
        input3_cpu.to(device),
        weight3_cpu.to(device),
        [3, 3, 3],
        bias3_cpu.to(device),
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        out=out,
    )
    assert actual.data_ptr() == out.data_ptr()
    synchronize(device)
    compare(out, expected, category="conv", dtype=torch.float32)

    tbc_input_cpu = torch.linspace(-1.0, 1.0, 30, dtype=torch.float32).reshape(5, 2, 3)
    tbc_weight_cpu = torch.linspace(-0.5, 0.5, 36, dtype=torch.float32).reshape(3, 3, 4)
    tbc_bias_cpu = torch.linspace(-0.2, 0.2, 4, dtype=torch.float32)
    expected = torch.ops.aten.conv_tbc.default(
        tbc_input_cpu, tbc_weight_cpu, tbc_bias_cpu, 1,
    )
    actual = torch.ops.aten.conv_tbc.default(
        tbc_input_cpu.to(device), tbc_weight_cpu.to(device), tbc_bias_cpu.to(device), 1,
    )
    synchronize(device)
    compare(actual, expected, category="conv", dtype=torch.float32)
    out = torch.empty_like(expected, device=device)
    actual = torch.ops.aten.conv_tbc.out(
        tbc_input_cpu.to(device),
        tbc_weight_cpu.to(device),
        tbc_bias_cpu.to(device),
        1,
        out=out,
    )
    assert actual.data_ptr() == out.data_ptr()
    synchronize(device)
    compare(out, expected, category="conv", dtype=torch.float32)


@pytest.mark.smoke
@pytest.mark.covers("aten::gru.data")
@pytest.mark.covers("aten::gru.input")
@pytest.mark.covers("aten::lstm.data")
@pytest.mark.covers("aten::lstm.input")
@pytest.mark.covers("aten::rnn_relu.data")
@pytest.mark.covers("aten::rnn_relu.input")
@pytest.mark.covers("aten::rnn_tanh.data")
@pytest.mark.covers("aten::rnn_tanh.input")
def test_direct_recurrent_dispatcher_surfaces(device, compare):
    seq_len = 4
    batch = 2
    input_size = 3
    hidden_size = 5
    input_cpu = torch.linspace(
        -1.0, 1.0, seq_len * batch * input_size, dtype=torch.float32,
    ).reshape(seq_len, batch, input_size)
    packed_data_cpu = torch.linspace(-1.0, 1.0, 6 * input_size, dtype=torch.float32).reshape(
        6, input_size,
    )
    batch_sizes_cpu = torch.tensor([2, 2, 1, 1], dtype=torch.long)
    hx_cpu = torch.linspace(-0.5, 0.5, batch * hidden_size, dtype=torch.float32).reshape(
        1, batch, hidden_size,
    )
    hx_packed_cpu = hx_cpu.clone()
    params_rnn_cpu = [
        torch.linspace(-0.5, 0.5, hidden_size * input_size, dtype=torch.float32).reshape(
            hidden_size, input_size,
        ),
        torch.linspace(-0.25, 0.25, hidden_size * hidden_size, dtype=torch.float32).reshape(
            hidden_size, hidden_size,
        ),
        torch.linspace(-0.1, 0.1, hidden_size, dtype=torch.float32),
        torch.linspace(0.1, -0.1, hidden_size, dtype=torch.float32),
    ]

    def to_dev_list(tensors):
        return [tensor.to(device) for tensor in tensors]

    for op in (torch.ops.aten.rnn_tanh, torch.ops.aten.rnn_relu):
        expected = op.input(
            input_cpu, hx_cpu, params_rnn_cpu, True, 1, 0.0, False, False, False,
        )
        actual = op.input(
            input_cpu.to(device),
            hx_cpu.to(device),
            to_dev_list(params_rnn_cpu),
            True,
            1,
            0.0,
            False,
            False,
            False,
        )
        _compare_tensor_tuple(actual, expected, compare, torch.float32)

        expected = op.data(
            packed_data_cpu,
            batch_sizes_cpu,
            hx_packed_cpu,
            params_rnn_cpu,
            True,
            1,
            0.0,
            False,
            False,
        )
        actual = op.data(
            packed_data_cpu.to(device),
            # Packed-sequence batch_sizes is host metadata in PyTorch's RNN contract.
            batch_sizes_cpu,
            hx_packed_cpu.to(device),
            to_dev_list(params_rnn_cpu),
            True,
            1,
            0.0,
            False,
            False,
        )
        _compare_tensor_tuple(actual, expected, compare, torch.float32)

    params_gru_cpu = [
        torch.linspace(-0.5, 0.5, 3 * hidden_size * input_size, dtype=torch.float32).reshape(
            3 * hidden_size, input_size,
        ),
        torch.linspace(-0.25, 0.25, 3 * hidden_size * hidden_size, dtype=torch.float32).reshape(
            3 * hidden_size, hidden_size,
        ),
        torch.linspace(-0.1, 0.1, 3 * hidden_size, dtype=torch.float32),
        torch.linspace(0.1, -0.1, 3 * hidden_size, dtype=torch.float32),
    ]
    expected = torch.ops.aten.gru.input(
        input_cpu, hx_cpu, params_gru_cpu, True, 1, 0.0, False, False, False,
    )
    actual = torch.ops.aten.gru.input(
        input_cpu.to(device),
        hx_cpu.to(device),
        to_dev_list(params_gru_cpu),
        True,
        1,
        0.0,
        False,
        False,
        False,
    )
    _compare_tensor_tuple(actual, expected, compare, torch.float32)
    expected = torch.ops.aten.gru.data(
        packed_data_cpu,
        batch_sizes_cpu,
        hx_packed_cpu,
        params_gru_cpu,
        True,
        1,
        0.0,
        False,
        False,
    )
    actual = torch.ops.aten.gru.data(
        packed_data_cpu.to(device),
        # Packed-sequence batch_sizes is host metadata in PyTorch's RNN contract.
        batch_sizes_cpu,
        hx_packed_cpu.to(device),
        to_dev_list(params_gru_cpu),
        True,
        1,
        0.0,
        False,
        False,
    )
    _compare_tensor_tuple(actual, expected, compare, torch.float32)

    c0_cpu = torch.linspace(0.25, -0.25, batch * hidden_size, dtype=torch.float32).reshape(
        1, batch, hidden_size,
    )
    params_lstm_cpu = [
        torch.linspace(-0.5, 0.5, 4 * hidden_size * input_size, dtype=torch.float32).reshape(
            4 * hidden_size, input_size,
        ),
        torch.linspace(-0.25, 0.25, 4 * hidden_size * hidden_size, dtype=torch.float32).reshape(
            4 * hidden_size, hidden_size,
        ),
        torch.linspace(-0.1, 0.1, 4 * hidden_size, dtype=torch.float32),
        torch.linspace(0.1, -0.1, 4 * hidden_size, dtype=torch.float32),
    ]
    expected = torch.ops.aten.lstm.input(
        input_cpu,
        [hx_cpu, c0_cpu],
        params_lstm_cpu,
        True,
        1,
        0.0,
        False,
        False,
        False,
    )
    actual = torch.ops.aten.lstm.input(
        input_cpu.to(device),
        [hx_cpu.to(device), c0_cpu.to(device)],
        to_dev_list(params_lstm_cpu),
        True,
        1,
        0.0,
        False,
        False,
        False,
    )
    _compare_tensor_tuple(actual, expected, compare, torch.float32)
    expected = torch.ops.aten.lstm.data(
        packed_data_cpu,
        batch_sizes_cpu,
        [hx_packed_cpu, c0_cpu],
        params_lstm_cpu,
        True,
        1,
        0.0,
        False,
        False,
    )
    actual = torch.ops.aten.lstm.data(
        packed_data_cpu.to(device),
        # Packed-sequence batch_sizes is host metadata in PyTorch's RNN contract.
        batch_sizes_cpu,
        [hx_packed_cpu.to(device), c0_cpu.to(device)],
        to_dev_list(params_lstm_cpu),
        True,
        1,
        0.0,
        False,
        False,
    )
    _compare_tensor_tuple(actual, expected, compare, torch.float32)
