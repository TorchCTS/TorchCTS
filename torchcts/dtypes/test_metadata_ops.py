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


@pytest.mark.smoke
@pytest.mark.covers("aten::can_cast")
@pytest.mark.covers("aten::promote_types")
@pytest.mark.covers("aten::result_type.Scalar_Scalar")
def test_scalar_type_promotion_metadata():
    assert torch.can_cast(torch.int32, torch.float32) is True
    assert torch.can_cast(torch.float32, torch.int32) is False

    assert torch.promote_types(torch.int32, torch.float32) is torch.float32
    assert torch.promote_types(torch.bool, torch.int64) is torch.int64

    assert torch.result_type(1, 1.0) is torch.float32
    assert torch.result_type(True, 1) is torch.int64
