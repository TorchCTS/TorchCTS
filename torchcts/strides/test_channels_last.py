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

@pytest.mark.medium
@pytest.mark.requires("channels_last")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_channels_last_conv2d(dtype, device, manifest, compare, input_gen):
    # Create input in channels_last layout
    x_dev = input_gen((2, 3, 16, 16), dtype, device, layout="channels_last")
    w_dev = input_gen((4, 3, 3, 3), dtype, device)
    
    assert x_dev.is_contiguous(memory_format=torch.channels_last)
    
    # Conv2d output should also propagate/preserve channels_last layout if supported
    expected = torch.nn.functional.conv2d(x_dev.cpu(), w_dev.cpu(), padding=1)
    actual = torch.nn.functional.conv2d(x_dev, w_dev, padding=1)
    synchronize(device)
    
    compare(actual, expected, category="conv", dtype=dtype)
