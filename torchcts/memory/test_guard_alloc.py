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
from torchcts.core.device import get_device_module

pytestmark = pytest.mark.covers_category("guard_alloc")

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.stress
@pytest.mark.requires("guard_alloc")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [
    (100, 100),
    (50, 50)
])
def test_guard_alloc_canary(shape, dtype, device):
    mod = get_device_module(device)
    if mod is None or not hasattr(mod, "guard_allocator_enabled"):
        pytest.skip("Guard allocator validation requires a backend-specific verifier hook.")

    assert mod.guard_allocator_enabled() is True
    x = torch.randn(*shape, dtype=dtype, device=device)
    y = x + 1.0
    assert y.shape == shape
