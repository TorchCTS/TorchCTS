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

FFT_DTYPES = [torch.float32]

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", FFT_DTYPES)
@pytest.mark.parametrize("op_name", ["fft", "ifft", "rfft", "irfft"])
def test_fft_ops_1d(dtype, op_name, device, compare, input_gen):
    length = 128
    
    if op_name in ("fft", "ifft"):
        x_dev = input_gen((length,), dtype, device)
        x_comp_dev = torch.complex(x_dev, torch.zeros_like(x_dev))
        x_comp_cpu = torch.complex(x_dev.cpu(), torch.zeros_like(x_dev.cpu()))
        
        if op_name == "fft":
            expected = torch.fft.fft(x_comp_cpu)
            actual = torch.fft.fft(x_comp_dev)
        else:
            fft_cpu = torch.fft.fft(x_comp_cpu)
            fft_dev = torch.fft.fft(x_comp_dev)
            expected = torch.fft.ifft(fft_cpu)
            actual = torch.fft.ifft(fft_dev)
            
        synchronize(device)
        compare(actual, expected, category="fft", dtype=dtype)
        
    elif op_name == "rfft":
        x_dev = input_gen((length,), dtype, device)
        expected = torch.fft.rfft(x_dev.cpu())
        actual = torch.fft.rfft(x_dev)
        synchronize(device)
        compare(actual, expected, category="fft", dtype=dtype)
        
    elif op_name == "irfft":
        x_dev = input_gen((length // 2 + 1,), dtype, device)
        x_comp_dev = torch.complex(x_dev, torch.zeros_like(x_dev))
        x_comp_cpu = torch.complex(x_dev.cpu(), torch.zeros_like(x_dev.cpu()))
        
        expected = torch.fft.irfft(x_comp_cpu, n=length)
        actual = torch.fft.irfft(x_comp_dev, n=length)
        synchronize(device)
        compare(actual, expected, category="fft", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", FFT_DTYPES)
def test_fft_ops_prime_length(dtype, device, compare, input_gen):
    # Prime length (Rader/Bluestein algorithm trigger)
    prime_len = 97
    x_dev = input_gen((prime_len,), dtype, device)
    
    expected = torch.fft.rfft(x_dev.cpu())
    actual = torch.fft.rfft(x_dev)
    synchronize(device)
    compare(actual, expected, category="fft", dtype=dtype)
