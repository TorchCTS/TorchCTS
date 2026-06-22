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

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.stress
@pytest.mark.requires("deterministic")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shapes", [
    (64, 128, 64),
    (32, 64, 32)
])
def test_determinism_stale_buffers(shapes, dtype, device):
    # Run a multi-kernel chain twice to verify bitwise equality
    # matmul -> silu -> matmul -> layernorm -> softmax
    M, K, N = shapes
    
    # Run twice
    outputs = []
    
    for run in range(2):
        # We manually seed torch for inputs generation but we want to ensure
        # that the memory allocator reuse of buffers doesn't pollute results
        # if there are recycler bugs.
        g = torch.Generator(device=device)
        g.manual_seed(12345)
        
        x = torch.randn(M, K, device=device, generator=g).to(dtype)
        w1 = torch.randn(K, N, device=device, generator=g).to(dtype)
        w2 = torch.randn(N, N, device=device, generator=g).to(dtype)
        
        # 1. Matmul
        h1 = torch.mm(x, w1)
        # 2. SiLU
        h2 = torch.nn.functional.silu(h1)
        # 3. Matmul
        h3 = torch.mm(h2, w2)
        # 4. LayerNorm
        ln = torch.nn.LayerNorm(N, device=device, dtype=dtype)
        # Copy weights to keep them identical
        # (LayerNorm creates parameters internally)
        with torch.no_grad():
            ln.weight.fill_(1.0)
            ln.bias.fill_(0.0)
        h4 = ln(h3)
        # 5. Softmax
        out = torch.nn.functional.softmax(h4, dim=-1)
        
        synchronize(device)
        outputs.append(out.cpu())
        
        # Delete inputs and activations to release buffers back to allocator cache
        del x, w1, w2, h1, h2, h3, ln, h4, out
        
    assert torch.equal(outputs[0], outputs[1]), "Allocator reuse resulted in non-deterministic outputs (stale buffer leak)."
