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

UPSAMPLE_DTYPES = [torch.float32]

@pytest.mark.smoke
@pytest.mark.covers("aten::upsample_bicubic2d")
@pytest.mark.covers("aten::upsample_bilinear2d")
@pytest.mark.covers("aten::upsample_nearest2d")
@pytest.mark.parametrize("dtype", UPSAMPLE_DTYPES)
@pytest.mark.parametrize("mode", ["nearest", "bilinear", "bicubic"])
def test_upsample_2d(dtype, mode, device, compare, input_gen):
    x_dev = input_gen((2, 3, 8, 8), dtype, device)
    
    expected = torch.nn.functional.interpolate(x_dev.cpu(), size=(16, 16), mode=mode)
    actual = torch.nn.functional.interpolate(x_dev, size=(16, 16), mode=mode)
    synchronize(device)
    
    compare(actual, expected, category="elementwise", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.covers("aten::upsample_trilinear3d")
@pytest.mark.parametrize("dtype", UPSAMPLE_DTYPES)
@pytest.mark.parametrize("mode", ["trilinear"])
def test_upsample_3d(dtype, mode, device, compare, input_gen):
    x_dev = input_gen((1, 2, 4, 4, 4), dtype, device)
    
    expected = torch.nn.functional.interpolate(x_dev.cpu(), size=(8, 8, 8), mode=mode, align_corners=False)
    actual = torch.nn.functional.interpolate(x_dev, size=(8, 8, 8), mode=mode, align_corners=False)
    synchronize(device)
    
    compare(actual, expected, category="elementwise", dtype=dtype)
