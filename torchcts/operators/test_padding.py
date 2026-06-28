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

PADDING_DTYPES = [torch.float32, torch.int64]

@pytest.mark.smoke
@pytest.mark.covers("aten::constant_pad_nd")
@pytest.mark.parametrize("dtype", PADDING_DTYPES)
def test_constant_pad(dtype, device, compare, input_gen):
    x_dev = input_gen((4, 4), dtype, device)
    pad = (1, 2, 1, 2) # left, right, top, bottom
    
    expected = torch.nn.functional.pad(x_dev.cpu(), pad, mode="constant", value=42)
    actual = torch.nn.functional.pad(x_dev, pad, mode="constant", value=42)
    synchronize(device)
    
    compare(actual, expected, category="exact", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.covers("aten::copy_")
@pytest.mark.covers("aten::reflection_pad2d")
@pytest.mark.covers("aten::replication_pad2d")
@pytest.mark.covers("aten::slice.Tensor")
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("mode", ["reflect", "replicate", "circular"])
def test_other_padding_modes(dtype, mode, device, compare, input_gen):
    x_dev = input_gen((1, 1, 8, 8), dtype, device)
    pad = (1, 1, 1, 1)
    
    expected = torch.nn.functional.pad(x_dev.cpu(), pad, mode=mode)
    actual = torch.nn.functional.pad(x_dev, pad, mode=mode)
    synchronize(device)
    compare(actual, expected, category="exact", dtype=dtype)
