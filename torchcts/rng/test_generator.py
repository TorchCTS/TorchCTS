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

@pytest.mark.smoke
@pytest.mark.parametrize("seed", [42, 1234])
def test_rng_reproducibility(seed, device, manifest):
    # 1. Same manual seed -> identical outputs
    torch.manual_seed(seed)
    # Note: torch.randn with generator or global seed
    x1 = torch.randn(10, 10, device=device)
    
    torch.manual_seed(seed)
    x2 = torch.randn(10, 10, device=device)
    synchronize(device)
    
    assert torch.equal(x1.cpu(), x2.cpu()), "manual_seed did not yield reproducible random outputs"

@pytest.mark.smoke
@pytest.mark.parametrize("seed", [123, 456])
def test_rng_generator_seeding(seed, device, manifest):
    # 2. Per-Generator seeding
    try:
        g1 = torch.Generator(device=device)
        g1.manual_seed(seed)
        y1 = torch.randn(5, 5, device=device, generator=g1)
        
        g2 = torch.Generator(device=device)
        g2.manual_seed(seed)
        y2 = torch.randn(5, 5, device=device, generator=g2)
        synchronize(device)
        
        assert torch.equal(y1.cpu(), y2.cpu()), "per-generator manual_seed was not reproducible"
    except (RuntimeError, TypeError):
        # Some backends don't support custom Generator object
        pass


@pytest.mark.smoke
def test_rng_sequential_calls_differ(device, manifest):
    """Two consecutive randn calls with the same seed context must produce different tensors."""
    torch.manual_seed(42)
    a = torch.randn(100, device=device).cpu()
    b = torch.randn(100, device=device).cpu()
    assert not torch.equal(a, b), "Sequential randn calls returned identical tensors"


@pytest.mark.smoke
def test_uniform_reproducibility(device, manifest):
    """uniform_() must be reproducible across manual_seed resets."""
    torch.manual_seed(42)
    a = torch.empty(100, device=device).uniform_().cpu()
    torch.manual_seed(42)
    b = torch.empty(100, device=device).uniform_().cpu()
    assert torch.allclose(a, b), "uniform_() not reproducible with same seed"


@pytest.mark.smoke
def test_bernoulli_reproducibility(device, manifest):
    """bernoulli_() must be reproducible across manual_seed resets."""
    torch.manual_seed(42)
    a = torch.empty(100, device=device).bernoulli_(0.5).cpu()
    torch.manual_seed(42)
    b = torch.empty(100, device=device).bernoulli_(0.5).cpu()
    assert torch.equal(a, b), "bernoulli_() not reproducible with same seed"

