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

FP8_DTYPES = []
if hasattr(torch, "float8_e4m3fn"):
    FP8_DTYPES.append(torch.float8_e4m3fn)
if hasattr(torch, "float8_e5m2"):
    FP8_DTYPES.append(torch.float8_e5m2)


@pytest.mark.smoke
@pytest.mark.requires("fp8")
@pytest.mark.covers("aten::_to_copy")
@pytest.mark.covers("aten::lift_fresh")
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
def test_fp8_cast_roundtrip(fp8_dtype, device):
    x = torch.tensor([0.0, 0.5, 1.0, -1.0, 2.0, -0.25], dtype=torch.float32, device=device)
    x_fp8 = x.to(fp8_dtype)
    synchronize(device)
    assert x_fp8.dtype == fp8_dtype
    x_back = x_fp8.to(torch.float32)
    synchronize(device)
    assert torch.allclose(x_back, x, atol=0.5)


@pytest.mark.smoke
@pytest.mark.requires("fp8")
@pytest.mark.covers("aten::empty.memory_format")
@pytest.mark.covers("aten::zeros")
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
def test_fp8_tensor_creation(fp8_dtype, device):
    z = torch.zeros(16, dtype=fp8_dtype, device=device)
    synchronize(device)
    assert z.dtype == fp8_dtype
    assert torch.all(z.to(torch.float32) == 0.0)

    e = torch.empty(16, dtype=fp8_dtype, device=device)
    synchronize(device)
    assert e.dtype == fp8_dtype


@pytest.mark.smoke
@pytest.mark.requires("fp8")
@pytest.mark.covers("aten::_to_copy")
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
def test_fp8_copy_to_device(fp8_dtype, device):
    x_cpu = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float32).to(fp8_dtype)
    x_dev = x_cpu.to(device)
    synchronize(device)
    assert x_dev.device.type == device
    x_back = x_dev.cpu()
    assert torch.equal(x_back.to(torch.float32), x_cpu.to(torch.float32))


@pytest.mark.smoke
@pytest.mark.requires("fp8")
@pytest.mark.covers("aten::_scaled_mm")
@pytest.mark.covers("aten::_scaled_mm.out")
@pytest.mark.covers("aten::_scaled_mm_v2")
@pytest.mark.covers("aten::_scaled_mm_v2.out")
def test_fp8_scaled_mm_dispatcher_variants(device, compare):
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("float8_e4m3fn is not available in this PyTorch build")

    a_cpu = torch.linspace(-1.0, 1.0, steps=6, dtype=torch.float32).reshape(2, 3).to(torch.float8_e4m3fn)
    b_cpu = torch.linspace(0.75, -0.75, steps=12, dtype=torch.float32).reshape(3, 4).to(torch.float8_e4m3fn)
    a_dev = a_cpu.to(device)
    b_dev = b_cpu.to(device)
    scale_a_cpu = torch.tensor(1.0, dtype=torch.float32)
    scale_b_cpu = torch.tensor(1.0, dtype=torch.float32)
    scale_a_dev = scale_a_cpu.to(device)
    scale_b_dev = scale_b_cpu.to(device)

    expected = torch.ops.aten._scaled_mm(a_cpu, b_cpu, scale_a_cpu, scale_b_cpu, None, None, torch.float32, False)
    actual = torch.ops.aten._scaled_mm(a_dev, b_dev, scale_a_dev, scale_b_dev, None, None, torch.float32, False)
    synchronize(device)
    compare(actual, expected, category="matmul", dtype=torch.float32)

    out_cpu = torch.empty_like(expected)
    out_dev = torch.empty_like(actual)
    expected_return = torch.ops.aten._scaled_mm.out(
        a_cpu,
        b_cpu,
        scale_a_cpu,
        scale_b_cpu,
        None,
        None,
        torch.float32,
        False,
        out=out_cpu,
    )
    actual_return = torch.ops.aten._scaled_mm.out(
        a_dev,
        b_dev,
        scale_a_dev,
        scale_b_dev,
        None,
        None,
        torch.float32,
        False,
        out=out_dev,
    )
    synchronize(device)
    assert expected_return is out_cpu
    assert actual_return is out_dev
    compare(out_dev, out_cpu, category="matmul", dtype=torch.float32)

    expected_v2 = torch.ops.aten._scaled_mm_v2(
        a_cpu,
        b_cpu,
        [scale_a_cpu],
        [0],
        [],
        [scale_b_cpu],
        [0],
        [],
        None,
        torch.float32,
        [],
        False,
    )
    actual_v2 = torch.ops.aten._scaled_mm_v2(
        a_dev,
        b_dev,
        [scale_a_dev],
        [0],
        [],
        [scale_b_dev],
        [0],
        [],
        None,
        torch.float32,
        [],
        False,
    )
    synchronize(device)
    compare(actual_v2, expected_v2, category="matmul", dtype=torch.float32)

    out_cpu = torch.empty_like(expected_v2)
    out_dev = torch.empty_like(actual_v2)
    expected_return = torch.ops.aten._scaled_mm_v2.out(
        a_cpu,
        b_cpu,
        [scale_a_cpu],
        [0],
        [],
        [scale_b_cpu],
        [0],
        [],
        None,
        torch.float32,
        [],
        False,
        out=out_cpu,
    )
    actual_return = torch.ops.aten._scaled_mm_v2.out(
        a_dev,
        b_dev,
        [scale_a_dev],
        [0],
        [],
        [scale_b_dev],
        [0],
        [],
        None,
        torch.float32,
        [],
        False,
        out=out_dev,
    )
    synchronize(device)
    assert expected_return is out_cpu
    assert actual_return is out_dev
    compare(out_dev, out_cpu, category="matmul", dtype=torch.float32)
