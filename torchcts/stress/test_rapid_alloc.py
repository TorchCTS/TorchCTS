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
import random

pytestmark = pytest.mark.covers_category("stress")

@pytest.mark.stress
@pytest.mark.parametrize("num_iterations", [100, 500])
def test_rapid_alloc_free(num_iterations, device):
    # Stress the allocator with dynamic sizes in a loop
    tensors = []
    sizes = [1024, 2048, 4096, 8192, 16384, 32768]
    
    for _ in range(num_iterations):
        sz = random.choice(sizes)
        # Allocate
        t = torch.randn(sz, device=device)
        tensors.append(t)
        
        # Randomly release some
        if len(tensors) > 20:
            idx = random.randint(0, len(tensors) - 1)
            del tensors[idx]
            
    # Clean up remaining
    tensors.clear()
