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
@pytest.mark.parametrize("dtype", [torch.float32])
def test_noncontiguous_tensors(dtype, device, manifest, compare, input_gen):
    # Sliced non-contiguous layout input
    # create shape (32, 32) from (64, 64) base using slicing [::2, ::2]
    x_dev = input_gen((32, 32), dtype, device, layout="sliced")
    y_dev = input_gen((32, 32), dtype, device, layout="sliced")
    
    assert not x_dev.is_contiguous()
    assert not y_dev.is_contiguous()
    
    expected = x_dev.cpu() + y_dev.cpu()
    actual = x_dev + y_dev
    synchronize(device)
    
    compare(actual, expected, category="strided_reduction", dtype=dtype)
