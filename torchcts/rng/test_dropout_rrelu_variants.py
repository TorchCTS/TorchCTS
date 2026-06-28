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


pytestmark = [pytest.mark.smoke, pytest.mark.covers_category("rng")]


def _input(device: str) -> torch.Tensor:
    return torch.linspace(-2.0, 2.0, 12, dtype=torch.float32, device=device).reshape(3, 4)


def _assert_same(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype
    assert tuple(actual.shape) == tuple(expected.shape)
    torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu())


@pytest.mark.covers("aten::alpha_dropout_")
@pytest.mark.covers("aten::dropout_")
@pytest.mark.covers("aten::feature_alpha_dropout_")
@pytest.mark.covers("aten::feature_dropout_")
def test_dropout_inplace_train_false_variants(device):
    for op in (
        torch.ops.aten.dropout_.default,
        torch.ops.aten.alpha_dropout_.default,
        torch.ops.aten.feature_dropout_.default,
        torch.ops.aten.feature_alpha_dropout_.default,
    ):
        cpu = _input("cpu")
        actual = _input(device)
        returned = op(actual, 0.5, False)
        expected = op(cpu, 0.5, False)
        synchronize(device)
        assert returned is actual
        _assert_same(actual, expected)


@pytest.mark.covers("aten::native_dropout.out")
@pytest.mark.covers("aten::native_dropout_backward.out")
def test_native_dropout_out_variants_train_false(device):
    cpu = _input("cpu")
    dev = _input(device)

    expected_out = torch.empty_like(cpu)
    expected_mask = torch.empty_like(cpu, dtype=torch.bool)
    actual_out = torch.empty_like(dev)
    actual_mask = torch.empty_like(dev, dtype=torch.bool)
    expected = torch.ops.aten.native_dropout.out(cpu, 0.5, False, out0=expected_out, out1=expected_mask)
    actual = torch.ops.aten.native_dropout.out(dev, 0.5, False, out0=actual_out, out1=actual_mask)
    synchronize(device)
    assert actual[0] is actual_out
    assert actual[1] is actual_mask
    _assert_same(actual[0], expected[0])
    _assert_same(actual[1].to(torch.float32), expected[1].to(torch.float32))

    expected_grad = torch.empty_like(cpu)
    actual_grad = torch.empty_like(dev)
    expected_backward = torch.ops.aten.native_dropout_backward.out(cpu, expected[1], 1.25, out=expected_grad)
    actual_backward = torch.ops.aten.native_dropout_backward.out(dev, actual[1], 1.25, out=actual_grad)
    synchronize(device)
    assert actual_backward is actual_grad
    _assert_same(actual_backward, expected_backward)


@pytest.mark.covers("aten::rrelu")
@pytest.mark.covers("aten::rrelu_")
@pytest.mark.covers("aten::rrelu_with_noise")
@pytest.mark.covers("aten::rrelu_with_noise.out")
@pytest.mark.covers("aten::rrelu_with_noise_")
@pytest.mark.covers("aten::rrelu_with_noise_backward.out")
@pytest.mark.covers("aten::rrelu_with_noise_functional")
def test_rrelu_train_false_variants(device):
    cpu = _input("cpu")
    dev = _input(device)
    cpu_noise = torch.empty_like(cpu)
    dev_noise = torch.empty_like(dev)

    _assert_same(
        torch.ops.aten.rrelu.default(dev, 0.125, 1.0 / 3.0, False, None),
        torch.ops.aten.rrelu.default(cpu, 0.125, 1.0 / 3.0, False, None),
    )

    actual_inplace = dev.clone()
    expected_inplace = cpu.clone()
    actual_returned = torch.ops.aten.rrelu_.default(actual_inplace, 0.125, 1.0 / 3.0, False, None)
    expected_returned = torch.ops.aten.rrelu_.default(expected_inplace, 0.125, 1.0 / 3.0, False, None)
    synchronize(device)
    assert actual_returned is actual_inplace
    _assert_same(actual_inplace, expected_returned)

    _assert_same(
        torch.ops.aten.rrelu_with_noise.default(dev, dev_noise.clone(), 0.125, 1.0 / 3.0, False, None),
        torch.ops.aten.rrelu_with_noise.default(cpu, cpu_noise.clone(), 0.125, 1.0 / 3.0, False, None),
    )

    actual_out = torch.empty_like(dev)
    expected_out = torch.empty_like(cpu)
    actual_returned = torch.ops.aten.rrelu_with_noise.out(dev, dev_noise.clone(), 0.125, 1.0 / 3.0, False, None, out=actual_out)
    expected_returned = torch.ops.aten.rrelu_with_noise.out(cpu, cpu_noise.clone(), 0.125, 1.0 / 3.0, False, None, out=expected_out)
    synchronize(device)
    assert actual_returned is actual_out
    _assert_same(actual_returned, expected_returned)

    actual_inplace = dev.clone()
    expected_inplace = cpu.clone()
    actual_returned = torch.ops.aten.rrelu_with_noise_.default(actual_inplace, dev_noise.clone(), 0.125, 1.0 / 3.0, False, None)
    expected_returned = torch.ops.aten.rrelu_with_noise_.default(expected_inplace, cpu_noise.clone(), 0.125, 1.0 / 3.0, False, None)
    synchronize(device)
    assert actual_returned is actual_inplace
    _assert_same(actual_returned, expected_returned)

    actual_functional = torch.ops.aten.rrelu_with_noise_functional.default(dev, dev_noise.clone(), 0.125, 1.0 / 3.0, False, None)
    expected_functional = torch.ops.aten.rrelu_with_noise_functional.default(cpu, cpu_noise.clone(), 0.125, 1.0 / 3.0, False, None)
    synchronize(device)
    assert len(actual_functional) == len(expected_functional) == 2
    _assert_same(actual_functional[0], expected_functional[0])
    _assert_same(actual_functional[1], expected_functional[1])

    actual_backward = torch.empty_like(dev)
    expected_backward = torch.empty_like(cpu)
    actual_returned = torch.ops.aten.rrelu_with_noise_backward.out(
        dev,
        dev,
        torch.ones_like(dev),
        0.125,
        1.0 / 3.0,
        False,
        False,
        out=actual_backward,
    )
    expected_returned = torch.ops.aten.rrelu_with_noise_backward.out(
        cpu,
        cpu,
        torch.ones_like(cpu),
        0.125,
        1.0 / 3.0,
        False,
        False,
        out=expected_backward,
    )
    synchronize(device)
    assert actual_returned is actual_backward
    _assert_same(actual_returned, expected_returned)
