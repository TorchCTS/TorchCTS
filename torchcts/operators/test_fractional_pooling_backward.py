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

from __future__ import annotations

import pytest
import torch

from torchcts.core.device import synchronize


pytestmark = [pytest.mark.smoke, pytest.mark.covers_category("pooling")]


def _assert_same(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype
    assert tuple(actual.shape) == tuple(expected.shape)
    torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu())


@pytest.mark.covers("aten::fractional_max_pool2d_backward")
@pytest.mark.covers("aten::fractional_max_pool2d_backward.grad_input")
@pytest.mark.covers("aten::fractional_max_pool3d_backward")
@pytest.mark.covers("aten::fractional_max_pool3d_backward.grad_input")
@pytest.mark.parametrize("rank", [2, 3])
def test_fractional_max_pool_backward_variants(rank, device):
    if rank == 2:
        cpu_input = torch.arange(2 * 3 * 6 * 7, dtype=torch.float32).reshape(2, 3, 6, 7) / 13.0
        dev_input = cpu_input.to(device)
        kernel_size = [2, 2]
        output_size = [2, 2]
        random_samples_cpu = torch.full((2, 3, 2), 0.5, dtype=torch.float32)
        random_samples_dev = random_samples_cpu.to(device)
        forward = torch.ops.aten.fractional_max_pool2d.default
        backward = torch.ops.aten.fractional_max_pool2d_backward.default
        backward_out = torch.ops.aten.fractional_max_pool2d_backward.grad_input
    else:
        cpu_input = torch.arange(1 * 2 * 5 * 6 * 7, dtype=torch.float32).reshape(1, 2, 5, 6, 7) / 17.0
        dev_input = cpu_input.to(device)
        kernel_size = [2, 2, 2]
        output_size = [2, 2, 2]
        random_samples_cpu = torch.full((1, 2, 3), 0.5, dtype=torch.float32)
        random_samples_dev = random_samples_cpu.to(device)
        forward = torch.ops.aten.fractional_max_pool3d.default
        backward = torch.ops.aten.fractional_max_pool3d_backward.default
        backward_out = torch.ops.aten.fractional_max_pool3d_backward.grad_input

    cpu_output, cpu_indices = forward(cpu_input, kernel_size, output_size, random_samples_cpu)
    dev_output, dev_indices = forward(dev_input, kernel_size, output_size, random_samples_dev)
    grad_cpu = torch.ones_like(cpu_output)
    grad_dev = torch.ones_like(dev_output)

    expected = backward(grad_cpu, cpu_input, kernel_size, output_size, cpu_indices)
    actual = backward(grad_dev, dev_input, kernel_size, output_size, dev_indices)
    synchronize(device)
    _assert_same(actual, expected)

    expected_out = torch.empty_like(cpu_input)
    actual_out = torch.empty_like(dev_input)
    expected_returned = backward_out(grad_cpu, cpu_input, kernel_size, output_size, cpu_indices, grad_input=expected_out)
    actual_returned = backward_out(grad_dev, dev_input, kernel_size, output_size, dev_indices, grad_input=actual_out)
    synchronize(device)
    assert expected_returned is expected_out
    assert actual_returned is actual_out
    _assert_same(actual_returned, expected_returned)
