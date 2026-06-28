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

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _packed_sequence_inputs(device):
    padded_cpu = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
            [[5.0, 6.0], [0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    lengths_cpu = torch.tensor([2, 1], dtype=torch.int64)
    return padded_cpu.to(device), lengths_cpu


def _assert_close_to_cpu(actual, expected):
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0.0, atol=0.0)


def _assert_nested_padded_close(actual, expected, padding=-9.0):
    assert actual.is_nested
    assert expected.is_nested
    actual_padded = torch.ops.aten.to_padded_tensor.default(actual, padding, None)
    expected_padded = torch.ops.aten.to_padded_tensor.default(expected, padding, None)
    _assert_close_to_cpu(actual_padded, expected_padded)


@pytest.mark.smoke
@pytest.mark.requires("nested")
@pytest.mark.covers("aten::_nested_tensor_from_tensor_list")
@pytest.mark.parametrize("dtype", DTYPES)
def test_nested_tensor_construction(dtype, device):
    components = [
        torch.randn(2, 3, dtype=dtype),
        torch.randn(4, 3, dtype=dtype),
        torch.randn(1, 3, dtype=dtype),
    ]
    nt = torch.nested.nested_tensor(components, dtype=dtype, device=device)
    synchronize(device)
    assert nt.is_nested
    assert nt.device.type == device


@pytest.mark.smoke
@pytest.mark.requires("nested")
@pytest.mark.covers("aten::nested_to_padded_tensor")
@pytest.mark.parametrize("dtype", DTYPES)
def test_nested_tensor_to_padded(dtype, device):
    components = [
        torch.randn(2, 4, dtype=dtype),
        torch.randn(5, 4, dtype=dtype),
    ]
    nt = torch.nested.nested_tensor(components, dtype=dtype, device=device)
    padded = nt.to_padded_tensor(padding=0.0)
    synchronize(device)
    assert padded.shape == (2, 5, 4)
    assert padded.device.type == device
    padded_cpu = padded.cpu()
    assert torch.all(padded_cpu[0, 2:, :] == 0.0)


@pytest.mark.smoke
@pytest.mark.requires("nested")
@pytest.mark.covers("aten::_nested_tensor_from_tensor_list")
@pytest.mark.parametrize("dtype", DTYPES)
def test_jagged_tensor_construction(dtype, device):
    components = [
        torch.randn(3, 5, dtype=dtype),
        torch.randn(7, 5, dtype=dtype),
        torch.randn(2, 5, dtype=dtype),
    ]
    nt = torch.nested.nested_tensor(
        components, dtype=dtype, device=device, layout=torch.jagged,
    )
    synchronize(device)
    assert nt.is_nested
    assert nt.device.type == device


@pytest.mark.smoke
@pytest.mark.requires("nested")
@pytest.mark.covers("aten::abs")
@pytest.mark.covers("aten::nested_to_padded_tensor")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_nested_tensor_unary_op(dtype, device):
    components_cpu = [torch.randn(3, 4, dtype=dtype), torch.randn(5, 4, dtype=dtype)]
    nt_cpu = torch.nested.nested_tensor(components_cpu, dtype=dtype, device="cpu")
    nt_dev = torch.nested.nested_tensor(components_cpu, dtype=dtype, device=device)

    result_cpu = torch.abs(nt_cpu).to_padded_tensor(0.0)
    result_dev = torch.abs(nt_dev).to_padded_tensor(0.0).cpu()
    synchronize(device)
    assert torch.allclose(result_dev, result_cpu), "Nested tensor abs() mismatch"


@pytest.mark.smoke
@pytest.mark.requires("nested")
@pytest.mark.covers("aten::add.Tensor")
@pytest.mark.covers("aten::nested_to_padded_tensor")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_nested_tensor_add(dtype, device):
    c1 = [torch.randn(2, 3, dtype=dtype), torch.randn(4, 3, dtype=dtype)]
    c2 = [torch.randn(2, 3, dtype=dtype), torch.randn(4, 3, dtype=dtype)]

    nt1_cpu = torch.nested.nested_tensor(c1, dtype=dtype, device="cpu")
    nt2_cpu = torch.nested.nested_tensor(c2, dtype=dtype, device="cpu")
    nt1_dev = torch.nested.nested_tensor(c1, dtype=dtype, device=device)
    nt2_dev = torch.nested.nested_tensor(c2, dtype=dtype, device=device)

    result_cpu = torch.add(nt1_cpu, nt2_cpu).to_padded_tensor(0.0)
    result_dev = torch.add(nt1_dev, nt2_dev).to_padded_tensor(0.0).cpu()
    synchronize(device)
    assert torch.allclose(result_dev, result_cpu), "Nested tensor add mismatch"


@pytest.mark.smoke
@pytest.mark.covers("aten::_pack_padded_sequence")
@pytest.mark.covers("aten::_pack_padded_sequence.out", surface="out_variant")
@pytest.mark.covers("aten::_pad_packed_sequence")
@pytest.mark.covers("aten::_pack_padded_sequence_backward", surface="autograd_backward")
def test_packed_sequence_dispatcher_variants(device):
    input_cpu, lengths_cpu = _packed_sequence_inputs("cpu")
    input_dev, _ = _packed_sequence_inputs(device)

    data_cpu, batch_sizes_cpu = torch.ops.aten._pack_padded_sequence.default(
        input_cpu, lengths_cpu, True,
    )
    data_dev, batch_sizes_dev = torch.ops.aten._pack_padded_sequence.default(
        input_dev, lengths_cpu, True,
    )
    synchronize(device)
    _assert_close_to_cpu(data_dev, data_cpu)
    _assert_close_to_cpu(batch_sizes_dev, batch_sizes_cpu)

    out_data = torch.empty_like(data_dev)
    out_batch_sizes = torch.empty_like(batch_sizes_dev)
    ret_data, ret_batch_sizes = torch.ops.aten._pack_padded_sequence.out(
        input_dev, lengths_cpu, True, out0=out_data, out1=out_batch_sizes,
    )
    synchronize(device)
    assert ret_data.data_ptr() == out_data.data_ptr()
    assert ret_batch_sizes.data_ptr() == out_batch_sizes.data_ptr()
    _assert_close_to_cpu(out_data, data_cpu)
    _assert_close_to_cpu(out_batch_sizes, batch_sizes_cpu)

    padded_cpu, restored_lengths_cpu = torch.ops.aten._pad_packed_sequence.default(
        data_cpu, batch_sizes_cpu, True, 0.0, input_cpu.size(1),
    )
    padded_dev, restored_lengths_dev = torch.ops.aten._pad_packed_sequence.default(
        data_dev, batch_sizes_dev.cpu(), True, 0.0, input_cpu.size(1),
    )
    synchronize(device)
    _assert_close_to_cpu(padded_dev, padded_cpu)
    _assert_close_to_cpu(restored_lengths_dev, restored_lengths_cpu)

    grad_cpu = torch.tensor(
        [[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]], dtype=torch.float32,
    )
    grad_dev = grad_cpu.to(device)
    back_cpu = torch.ops.aten._pack_padded_sequence_backward.default(
        grad_cpu, list(input_cpu.shape), batch_sizes_cpu, True,
    )
    back_dev = torch.ops.aten._pack_padded_sequence_backward.default(
        grad_dev, list(input_cpu.shape), batch_sizes_dev.cpu(), True,
    )
    synchronize(device)
    _assert_close_to_cpu(back_dev, back_cpu)


@pytest.mark.smoke
@pytest.mark.requires("nested")
@pytest.mark.covers("aten::_nested_tensor_size")
@pytest.mark.covers("aten::_nested_tensor_strides")
@pytest.mark.covers("aten::_nested_tensor_storage_offsets")
@pytest.mark.covers("aten::to_padded_tensor")
@pytest.mark.covers("aten::_nested_tensor_from_mask_left_aligned")
def test_nested_tensor_metadata_dispatcher_surfaces(device):
    components = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        torch.tensor([[5.0, 6.0]], dtype=torch.float32),
    ]
    nt_cpu = torch.nested.nested_tensor(components, dtype=torch.float32, device="cpu")
    nt_dev = torch.nested.nested_tensor(components, dtype=torch.float32, device=device)

    for op in (
        torch.ops.aten._nested_tensor_size.default,
        torch.ops.aten._nested_tensor_strides.default,
        torch.ops.aten._nested_tensor_storage_offsets.default,
    ):
        expected = op(nt_cpu)
        actual = op(nt_dev)
        synchronize(device)
        _assert_close_to_cpu(actual, expected)

    padded_cpu = torch.ops.aten.to_padded_tensor.default(nt_cpu, -9.0, None)
    padded_dev = torch.ops.aten.to_padded_tensor.default(nt_dev, -9.0, None)
    synchronize(device)
    _assert_close_to_cpu(padded_dev, padded_cpu)

    dense_cpu = torch.zeros(2, 3, 2, dtype=torch.float32)
    dense_dev = dense_cpu.to(device)
    mask_cpu = torch.tensor(
        [[True, True, False], [True, False, False]], dtype=torch.bool,
    )
    mask_dev = mask_cpu.to(device)
    expected_aligned = torch.ops.aten._nested_tensor_from_mask_left_aligned.default(
        dense_cpu, mask_cpu,
    )
    actual_aligned = torch.ops.aten._nested_tensor_from_mask_left_aligned.default(
        dense_dev, mask_dev,
    )
    synchronize(device)
    assert actual_aligned == expected_aligned


@pytest.mark.smoke
@pytest.mark.requires("nested")
@pytest.mark.covers("aten::_nested_compute_contiguous_strides_offsets")
@pytest.mark.covers("aten::_nested_from_padded")
@pytest.mark.covers("aten::_nested_from_padded_and_nested_example")
@pytest.mark.covers("aten::_nested_from_padded_tensor")
@pytest.mark.covers("aten::_nested_get_jagged_dummy")
@pytest.mark.covers("aten::_nested_get_lengths")
@pytest.mark.covers("aten::_nested_get_max_seqlen")
@pytest.mark.covers("aten::_nested_get_min_seqlen")
@pytest.mark.covers("aten::_nested_get_offsets")
@pytest.mark.covers("aten::_nested_get_ragged_idx")
@pytest.mark.covers("aten::_nested_get_values")
@pytest.mark.covers("aten::_nested_tensor_from_mask")
@pytest.mark.covers("aten::_nested_view_from_buffer")
@pytest.mark.covers("aten::_nested_view_from_buffer_copy")
@pytest.mark.covers("aten::_nested_view_from_jagged")
def test_nested_low_level_dispatcher_surfaces(device):
    components = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        torch.tensor([[5.0, 6.0]], dtype=torch.float32),
    ]
    nt_cpu = torch.nested.nested_tensor(components, dtype=torch.float32, device="cpu")
    nt_dev = torch.nested.nested_tensor(components, dtype=torch.float32, device=device)

    size_cpu = torch.ops.aten._nested_tensor_size.default(nt_cpu)
    strides_cpu = torch.ops.aten._nested_tensor_strides.default(nt_cpu)
    offsets_cpu = torch.ops.aten._nested_tensor_storage_offsets.default(nt_cpu)
    size_dev = torch.ops.aten._nested_tensor_size.default(nt_dev)
    strides_dev = torch.ops.aten._nested_tensor_strides.default(nt_dev)
    offsets_dev = torch.ops.aten._nested_tensor_storage_offsets.default(nt_dev)

    contiguous_cpu, storage_cpu = (
        torch.ops.aten._nested_compute_contiguous_strides_offsets.default(size_cpu)
    )
    contiguous_dev, storage_dev = (
        torch.ops.aten._nested_compute_contiguous_strides_offsets.default(size_dev.cpu())
    )
    _assert_close_to_cpu(contiguous_dev, contiguous_cpu)
    _assert_close_to_cpu(storage_dev, storage_cpu)

    padded_cpu = torch.ops.aten.to_padded_tensor.default(nt_cpu, 0.0, None)
    padded_dev = torch.ops.aten.to_padded_tensor.default(nt_dev, 0.0, None)
    from_padded_cpu = torch.ops.aten._nested_from_padded.default(
        padded_cpu, size_cpu, False,
    )
    from_padded_dev = torch.ops.aten._nested_from_padded.default(
        padded_dev, size_cpu, False,
    )
    from_example_cpu = torch.ops.aten._nested_from_padded_and_nested_example.default(
        padded_cpu, nt_cpu,
    )
    from_example_dev = torch.ops.aten._nested_from_padded_and_nested_example.default(
        padded_dev, nt_dev,
    )
    synchronize(device)
    _assert_nested_padded_close(from_padded_dev, from_padded_cpu)
    _assert_nested_padded_close(from_example_dev, from_example_cpu)

    values_cpu = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32,
    )
    offsets_jagged_cpu = torch.tensor([0, 2, 3], dtype=torch.int64)
    lengths_cpu = torch.tensor([2, 1], dtype=torch.int64)
    values_dev = values_cpu.to(device)
    offsets_jagged_dev = offsets_jagged_cpu.to(device)
    lengths_dev = lengths_cpu.to(device)
    jagged_cpu = torch.nested.nested_tensor_from_jagged(
        values_cpu,
        offsets=offsets_jagged_cpu,
        lengths=lengths_cpu,
        min_seqlen=1,
        max_seqlen=2,
    )
    jagged_dev = torch.nested.nested_tensor_from_jagged(
        values_dev,
        offsets=offsets_jagged_dev,
        lengths=lengths_dev,
        min_seqlen=1,
        max_seqlen=2,
    )
    dummy_cpu = torch.ops.aten._nested_get_jagged_dummy.default(jagged_cpu)
    dummy_dev = torch.ops.aten._nested_get_jagged_dummy.default(jagged_dev)
    assert dummy_cpu.is_nested
    assert dummy_dev.is_nested

    from_padded_tensor_cpu = torch.ops.aten._nested_from_padded_tensor.default(
        padded_cpu, offsets_jagged_cpu, dummy_cpu, 1, None, None, None,
    )
    from_padded_tensor_dev = torch.ops.aten._nested_from_padded_tensor.default(
        padded_dev, offsets_jagged_dev, dummy_dev, 1, None, None, None,
    )
    synchronize(device)
    _assert_nested_padded_close(from_padded_tensor_dev, from_padded_tensor_cpu)

    for op in (
        torch.ops.aten._nested_get_lengths.default,
        torch.ops.aten._nested_get_max_seqlen.default,
        torch.ops.aten._nested_get_min_seqlen.default,
        torch.ops.aten._nested_get_offsets.default,
        torch.ops.aten._nested_get_values.default,
    ):
        expected = op(jagged_cpu)
        actual = op(jagged_dev)
        if expected is not None:
            synchronize(device)
            _assert_close_to_cpu(actual, expected)
        else:
            assert actual is None
    assert torch.ops.aten._nested_get_ragged_idx.default(jagged_dev) == (
        torch.ops.aten._nested_get_ragged_idx.default(jagged_cpu)
    )

    mask_cpu = torch.tensor(
        [[True, True, False], [True, False, False]], dtype=torch.bool,
    )
    mask_dev = mask_cpu.to(device)
    dense_masked_cpu = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    dense_masked_dev = dense_masked_cpu.to(device)
    from_mask_cpu = torch.ops.aten._nested_tensor_from_mask.default(
        dense_masked_cpu, mask_cpu, True,
    )
    from_mask_dev = torch.ops.aten._nested_tensor_from_mask.default(
        dense_masked_dev, mask_dev, True,
    )
    synchronize(device)
    _assert_nested_padded_close(from_mask_dev, from_mask_cpu)

    flat_cpu = torch.cat([component.reshape(-1) for component in components])
    flat_dev = flat_cpu.to(device)
    view_buffer_cpu = torch.ops.aten._nested_view_from_buffer.default(
        flat_cpu, size_cpu, strides_cpu, offsets_cpu,
    )
    view_buffer_dev = torch.ops.aten._nested_view_from_buffer.default(
        flat_dev, size_dev.cpu(), strides_dev.cpu(), offsets_dev.cpu(),
    )
    view_buffer_copy_cpu = torch.ops.aten._nested_view_from_buffer_copy.default(
        flat_cpu, size_cpu, strides_cpu, offsets_cpu,
    )
    view_buffer_copy_dev = torch.ops.aten._nested_view_from_buffer_copy.default(
        flat_dev, size_dev.cpu(), strides_dev.cpu(), offsets_dev.cpu(),
    )
    synchronize(device)
    _assert_nested_padded_close(view_buffer_dev, view_buffer_cpu)
    _assert_nested_padded_close(view_buffer_copy_dev, view_buffer_copy_cpu)

    view_jagged_cpu = torch.ops.aten._nested_view_from_jagged.default(
        values_cpu, offsets_jagged_cpu, dummy_cpu, None, 1, None, None,
    )
    view_jagged_dev = torch.ops.aten._nested_view_from_jagged.default(
        values_dev, offsets_jagged_dev, dummy_dev, None, 1, None, None,
    )
    synchronize(device)
    _assert_nested_padded_close(view_jagged_dev, view_jagged_cpu)


@pytest.mark.smoke
@pytest.mark.requires("nested")
@pytest.mark.covers("aten::_nested_sum_backward", surface="autograd_backward")
def test_nested_sum_backward_dispatcher_surface(device):
    components = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        torch.tensor([[5.0, 6.0]], dtype=torch.float32),
    ]
    nt_cpu = torch.nested.nested_tensor(components, dtype=torch.float32, device="cpu")
    nt_dev = torch.nested.nested_tensor(components, dtype=torch.float32, device=device)
    grad_components = [
        torch.ones(2, 1, dtype=torch.float32),
        torch.ones(1, 1, dtype=torch.float32),
    ]
    grad_cpu = torch.nested.nested_tensor(
        grad_components, dtype=torch.float32, device="cpu",
    )
    grad_dev = torch.nested.nested_tensor(
        grad_components, dtype=torch.float32, device=device,
    )

    back_cpu = torch.ops.aten._nested_sum_backward.default(
        grad_cpu, nt_cpu, [1], True,
    )
    back_dev = torch.ops.aten._nested_sum_backward.default(
        grad_dev, nt_dev, [1], True,
    )
    synchronize(device)
    _assert_nested_padded_close(back_dev, back_cpu)
