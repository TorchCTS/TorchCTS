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
@pytest.mark.requires("streams")
def test_stream_construction(device):
    s = torch.Stream(device=device)
    assert s is not None


@pytest.mark.smoke
@pytest.mark.requires("streams")
def test_stream_synchronize(device):
    s = torch.Stream(device=device)
    with torch.stream(s):
        x = torch.randn(64, 64, device=device)
        y = torch.mm(x, x)
    s.synchronize()
    assert y.shape == (64, 64)
    assert torch.isfinite(y).all()


@pytest.mark.smoke
@pytest.mark.requires("streams")
def test_stream_wait_stream(device):
    s1 = torch.Stream(device=device)
    s2 = torch.Stream(device=device)
    with torch.stream(s1):
        x = torch.randn(32, 32, device=device)
        y = x + 1.0
    s2.wait_stream(s1)
    with torch.stream(s2):
        z = y * 2.0
    s2.synchronize()
    assert torch.isfinite(z).all()


@pytest.mark.smoke
@pytest.mark.requires("events")
def test_event_construction(device):
    e = torch.Event(device=device)
    assert e is not None


@pytest.mark.smoke
@pytest.mark.requires("events")
def test_event_record_and_query(device):
    s = torch.Stream(device=device)
    e = torch.Event(device=device)
    with torch.stream(s):
        x = torch.randn(64, 64, device=device)
        _ = torch.mm(x, x)
    e.record(s)
    s.synchronize()
    assert e.query()


@pytest.mark.smoke
@pytest.mark.requires("events")
def test_event_elapsed_time(device):
    s = torch.Stream(device=device)
    e_start = torch.Event(device=device, enable_timing=True)
    e_end = torch.Event(device=device, enable_timing=True)
    e_start.record(s)
    with torch.stream(s):
        x = torch.randn(256, 256, device=device)
        for _ in range(10):
            x = torch.mm(x, x)
    e_end.record(s)
    s.synchronize()
    elapsed = e_start.elapsed_time(e_end)
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0
