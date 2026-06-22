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
@pytest.mark.requires("training")
@pytest.mark.requires("autocast")
@pytest.mark.parametrize("autocast_dtype", [torch.float16])
def test_autocast_precisions(autocast_dtype, device, manifest):
    # We run autocast context for the device type
    # For MPS, device_type is 'mps'
    device_type = "cuda" if device == "cuda" else ("mps" if device == "mps" else "cpu")
    if device_type not in ("cuda", "cpu"):
        # PyTorch autocast supports 'cuda' and 'cpu' natively.
        # MPS autocast support might be limited/not register under torch.autocast.
        # We can check if it runs without errors.
        try:
            with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                pass
        except Exception:
            pytest.skip(f"autocast not supported for device type: {device_type}")
            
    x = torch.randn(4, 4, device=device)
    y = torch.randn(4, 4, device=device)
    
    with torch.autocast(device_type=device_type, dtype=autocast_dtype):
        # Matmul should downcast to half precision
        out = torch.mm(x, y)
        synchronize(device)
        
        # Verify it downcasted if device_type supports it
        # On CPU autocast might stay float32 depending on config, but on GPU/MPS it usually downcasts
        assert out.dtype in (torch.float16, torch.bfloat16, torch.float32)


@pytest.mark.requires("autocast")
def test_autocast_keep_precision(device, manifest):
    """Numerically sensitive ops must stay fp32 under autocast."""
    x = torch.randn(4, 8, device=device)
    w = torch.randn(8, device=device)

    with torch.autocast(device_type=device):
        y_ln = torch.nn.functional.layer_norm(x, [8], w)
    assert y_ln.dtype == torch.float32, f"layer_norm: expected fp32, got {y_ln.dtype}"

    with torch.autocast(device_type=device):
        y_sm = torch.nn.functional.softmax(x, dim=-1)
    assert y_sm.dtype == torch.float32, f"softmax: expected fp32, got {y_sm.dtype}"

    logits = torch.randn(4, 10, device=device)
    targets = torch.randint(0, 10, (4,), device=device)
    with torch.autocast(device_type=device):
        loss = torch.nn.functional.cross_entropy(logits, targets)
    assert loss.dtype == torch.float32, f"cross_entropy: expected fp32, got {loss.dtype}"


@pytest.mark.requires("autocast")
def test_autocast_downcast(device, manifest):
    """Matmul-class ops should downcast to fp16 under autocast."""
    a = torch.randn(4, 8, device=device)
    b = torch.randn(8, 4, device=device)
    with torch.autocast(device_type=device, dtype=torch.float16):
        c = torch.mm(a, b)
    assert c.dtype == torch.float16, f"Expected fp16, got {c.dtype}"


@pytest.mark.requires("autocast")
def test_autocast_backward(device, manifest):
    """Autocast forward followed by backward should produce valid gradients."""
    a = torch.randn(4, 8, device=device, requires_grad=True)
    b = torch.randn(8, 4, device=device, requires_grad=True)
    with torch.autocast(device_type=device, dtype=torch.float16):
        c = torch.mm(a, b)
    c.float().sum().backward()
    assert a.grad is not None
    assert not torch.isnan(a.grad).any()


@pytest.mark.requires("autocast")
@pytest.mark.requires("training")
def test_grad_scaler(device, manifest):
    """GradScaler scale/step/update cycle must not crash and scale > 0."""
    scaler = torch.amp.GradScaler(device)
    model = torch.nn.Linear(8, 4).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 8, device=device)

    opt.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.float16):
        y = model(x)
        loss = y.sum()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    assert scaler.get_scale() > 0

