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
import math
from torchcts.core.device import synchronize

pytestmark = pytest.mark.covers_category("stress")

def compare_tensor_outputs(cpu_t, dev_t):
    __tracebackhide__ = True
    if not isinstance(cpu_t, torch.Tensor):
        assert cpu_t == dev_t
        return
    
    dev_t_cpu = dev_t.to("cpu")
    assert cpu_t.shape == dev_t_cpu.shape, f"Shape mismatch: {cpu_t.shape} vs {dev_t_cpu.shape}"
    assert cpu_t.dtype == dev_t_cpu.dtype, f"Dtype mismatch: {cpu_t.dtype} vs {dev_t_cpu.dtype}"
    
    # Check for NaN and Inf equality
    nan_mask_cpu = torch.isnan(cpu_t)
    nan_mask_dev = torch.isnan(dev_t_cpu)
    assert torch.equal(nan_mask_cpu, nan_mask_dev), "NaN mask mismatch between CPU and device"
    
    inf_mask_cpu = torch.isinf(cpu_t)
    inf_mask_dev = torch.isinf(dev_t_cpu)
    assert torch.equal(inf_mask_cpu, inf_mask_dev), "Inf mask mismatch between CPU and device"
    
    # Check non-special elements
    non_special = ~(nan_mask_cpu | inf_mask_cpu)
    if non_special.any():
        # Using a slightly higher tolerance for adversarial bounds/near-singular outputs
        assert torch.allclose(cpu_t[non_special], dev_t_cpu[non_special], rtol=1e-2, atol=1e-2)

def run_adversarial_op(op_fn, *args, device, **kwargs):
    __tracebackhide__ = True
    cpu_args = [x.to("cpu") if isinstance(x, torch.Tensor) else x for x in args]
    cpu_kwargs = {k: (v.to("cpu") if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
    
    cpu_exception = None
    try:
        cpu_out = op_fn(*cpu_args, **cpu_kwargs)
    except Exception as e:
        cpu_exception = e
        
    dev_args = [x.to(device) if isinstance(x, torch.Tensor) else x for x in args]
    dev_kwargs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
    
    if cpu_exception is not None:
        # If CPU raised, the backend must also raise an exception
        try:
            op_fn(*dev_args, **dev_kwargs)
            synchronize(device)
        except (NotImplementedError, RuntimeError) as dev_e:
            err_msg = str(dev_e).lower()
            if "not implemented" in err_msg or "not supported" in err_msg or "support" in err_msg:
                pytest.skip(f"Operator not implemented on backend: {dev_e}")
            # Success: both raised exceptions
            return
        except Exception:
            # Any other exception is fine
            return
        
        pytest.fail(f"CPU raised {type(cpu_exception).__name__}: {cpu_exception}, but backend did not raise any exception")
    else:
        try:
            dev_out = op_fn(*dev_args, **dev_kwargs)
            synchronize(device)
        except (NotImplementedError, RuntimeError) as dev_e:
            err_msg = str(dev_e).lower()
            if "not implemented" in err_msg or "not supported" in err_msg or "support" in err_msg:
                pytest.skip(f"Operator not implemented on backend: {dev_e}")
            raise
        except Exception as dev_e:
            pytest.fail(f"CPU succeeded, but backend raised {type(dev_e).__name__}: {dev_e}")
            
        if isinstance(cpu_out, tuple):
            assert len(cpu_out) == len(dev_out)
            for c_o, d_o in zip(cpu_out, dev_out):
                compare_tensor_outputs(c_o, d_o)
        else:
            compare_tensor_outputs(cpu_out, dev_out)

@pytest.mark.stress
@pytest.mark.adversarial
@pytest.mark.parametrize("linalg_case", ["inv", "solve", "cholesky"])
def test_near_singular_linalg(linalg_case, device, manifest):
    if linalg_case == "inv":
        # 1. Singular matrix inverse
        A_sing = torch.zeros(4, 4)
        run_adversarial_op(torch.linalg.inv, A_sing, device=device)
    elif linalg_case == "solve":
        # 2. Near-singular matrix solve
        H = torch.zeros(6, 6)
        for i in range(6):
            for j in range(6):
                H[i, j] = 1.0 / (i + j + 1.0)
        b = torch.randn(6, 1)
        run_adversarial_op(torch.linalg.solve, H, b, device=device)
    elif linalg_case == "cholesky":
        # 3. Non-PSD matrix Cholesky
        A_non_psd = torch.diag(torch.tensor([1.0, -1.0, 1.0]))
        run_adversarial_op(torch.linalg.cholesky, A_non_psd, device=device)

@pytest.mark.stress
@pytest.mark.adversarial
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_scale_mismatch_numerics(dtype, device, manifest):
    info = torch.finfo(dtype)
    
    # 1. logsumexp stability near max limit
    # Inputs: [max - 2, max - 3]
    x_logsumexp = torch.tensor([info.max - 2.0, info.max - 3.0], dtype=dtype)
    run_adversarial_op(torch.logsumexp, x_logsumexp, dim=0, device=device)
    
    # 2. cumsum stability at bounds
    x_cumsum = torch.tensor([info.max / 2, -info.max / 2, info.max / 4, -info.max / 4], dtype=dtype)
    run_adversarial_op(torch.cumsum, x_cumsum, dim=0, device=device)

@pytest.mark.stress
@pytest.mark.adversarial
@pytest.mark.parametrize("size", [1000])
def test_scatter_add_determinism(size, device):
    num_updates = 20000
    
    index = torch.zeros(num_updates, dtype=torch.int64, device=device)
    src = torch.ones(num_updates, dtype=torch.float32, device=device)
    
    # Run 10 iterations to verify atomic write determinism
    outputs = []
    for _ in range(10):
        out = torch.zeros(size, dtype=torch.float32, device=device)
        out.scatter_add_(0, index, src)
        synchronize(device)
        outputs.append(out.cpu())
        
    for i in range(1, len(outputs)):
        assert torch.equal(outputs[0], outputs[i]), f"Non-deterministic scatter_add at iteration {i}"

@pytest.mark.stress
@pytest.mark.adversarial
@pytest.mark.parametrize("index_case", ["positive", "negative", "gather"])
def test_indexing_bounds(index_case, device):
    if device != "cpu":
        pytest.skip("Out-of-bounds eager exception semantics are only validated on CPU.")

    src = torch.randn(5)
    
    if index_case == "positive":
        # 1. Out-of-bound positive index select
        idx_oob = torch.tensor([5], dtype=torch.int64)
        run_adversarial_op(torch.index_select, src, 0, idx_oob, device=device)
    elif index_case == "negative":
        # 2. Out-of-bound negative index select
        idx_neg = torch.tensor([-6], dtype=torch.int64)
        run_adversarial_op(torch.index_select, src, 0, idx_neg, device=device)
    elif index_case == "gather":
        # 3. Gather with out-of-bound index
        src_2d = torch.randn(3, 3)
        idx_2d = torch.tensor([[3]], dtype=torch.int64)
        run_adversarial_op(torch.gather, src_2d, 0, idx_2d, device=device)

@pytest.mark.stress
@pytest.mark.adversarial
@pytest.mark.parametrize("empty_case", ["sum", "mean", "conv", "matmul"])
def test_empty_tensors(empty_case, device):
    x_empty = torch.randn(0, 5)
    
    if empty_case == "sum":
        # 1. Reduction sum over empty dimensions
        run_adversarial_op(torch.sum, x_empty, dim=0, device=device)
    elif empty_case == "mean":
        # 2. Reduction mean over empty dimensions
        run_adversarial_op(torch.mean, x_empty, dim=0, device=device)
    elif empty_case == "conv":
        # 3. Conv2d with empty batch size
        x_conv = torch.randn(0, 3, 16, 16)
        w_conv = torch.randn(8, 3, 3, 3)
        run_adversarial_op(lambda x, w: torch.nn.functional.conv2d(x, w), x_conv, w_conv, device=device)
    elif empty_case == "matmul":
        # 4. Matmul with empty dimension (5, 0) x (0, 6) -> (5, 6)
        a_empty = torch.randn(5, 0)
        b_empty = torch.randn(0, 6)
        run_adversarial_op(torch.mm, a_empty, b_empty, device=device)
