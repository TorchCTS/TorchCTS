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
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
def test_fp8_copy_to_device(fp8_dtype, device):
    x_cpu = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float32).to(fp8_dtype)
    x_dev = x_cpu.to(device)
    synchronize(device)
    assert x_dev.device.type == device
    x_back = x_dev.cpu()
    assert torch.equal(x_back.to(torch.float32), x_cpu.to(torch.float32))
