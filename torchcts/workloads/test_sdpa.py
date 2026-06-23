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

@pytest.mark.workload
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("is_causal", [True, False])
def test_sdpa_causal(dtype, is_causal, device, manifest, compare, input_gen):
    B, H, S, D = 2, 4, 32, 16
    
    q_dev = input_gen((B, H, S, D), dtype, device)
    k_dev = input_gen((B, H, S, D), dtype, device)
    v_dev = input_gen((B, H, S, D), dtype, device)
    
    if is_causal:
        expected = torch.nn.functional.scaled_dot_product_attention(
            q_dev.cpu(), k_dev.cpu(), v_dev.cpu(), is_causal=True
        )
        actual = torch.nn.functional.scaled_dot_product_attention(
            q_dev, k_dev, v_dev, is_causal=True
        )
        synchronize(device)
        compare(actual, expected, category="sdpa", dtype=dtype)
    else:
        # Non-causal with attention mask
        mask_cpu = torch.ones(B, 1, S, S, dtype=torch.bool)
        # create triangular causal mask manually
        mask_cpu = torch.tril(mask_cpu)
        mask_dev = mask_cpu.to(device)
        
        expected_masked = torch.nn.functional.scaled_dot_product_attention(
            q_dev.cpu(), k_dev.cpu(), v_dev.cpu(), attn_mask=mask_cpu
        )
        actual_masked = torch.nn.functional.scaled_dot_product_attention(
            q_dev, k_dev, v_dev, attn_mask=mask_dev
        )
        synchronize(device)
        compare(actual_masked, expected_masked, category="sdpa", dtype=dtype)


@pytest.mark.workload
@pytest.mark.parametrize("Sq,Sk", [
    (1, 1), (1, 7), (1, 32),   # single-query cross-attention
    (7, 7), (7, 32),            # short-query
    (32, 32),                   # matched
])
def test_sdpa_asymmetric_lengths(Sq, Sk, device, compare, input_gen):
    """SDPA with Q and K/V having different sequence lengths (non-causal)."""
    B, H, D = 1, 2, 16
    q = input_gen((B, H, Sq, D), torch.float32, device)
    k = input_gen((B, H, Sk, D), torch.float32, device)
    v = input_gen((B, H, Sk, D), torch.float32, device)

    # Non-causal — always valid regardless of Sq vs Sk
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    ref = torch.nn.functional.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu(), is_causal=False)
    synchronize(device)
    compare(out, ref, category="sdpa", dtype=torch.float32)


@pytest.mark.workload
@pytest.mark.parametrize("S", [3, 7, 15, 33, 65])
def test_sdpa_non_power_of_2(S, device, compare, input_gen):
    """Causal SDPA with non-power-of-2 sequence lengths."""
    B, H, D = 1, 2, 16
    q = input_gen((B, H, S, D), torch.float32, device)
    k = input_gen((B, H, S, D), torch.float32, device)
    v = input_gen((B, H, S, D), torch.float32, device)

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = torch.nn.functional.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu(), is_causal=True)
    synchronize(device)
    compare(out, ref, category="sdpa", dtype=torch.float32)


import itertools
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers for nested-tensor SDPA tests
# ---------------------------------------------------------------------------

def _make_nested_qkv(seq_lens, H, D, dtype, device, requires_grad=False):
    """Create nested QKV tensors in jagged layout (B, S*, H, D).

    Values are stored as (total_S, H, D) and the jagged dim is S (dim 1).
    On CUDA, the nested SDPA backend handles the internal layout transposition.
    On CPU, the fallback path dispatches values() through standard SDPA.

    NOTE: These tests are expected to run on accelerator backends where the
    nested SDPA kernel correctly handles the (B, S*, H, D) jagged layout.
    CPU validation mode exercises the fallback path which has different semantics.
    """
    total_S = sum(seq_lens)
    offsets = torch.tensor([0] + list(itertools.accumulate(seq_lens)), device=device)
    q_vals = torch.randn(total_S, H, D, device=device, dtype=dtype, requires_grad=requires_grad)
    k_vals = torch.randn(total_S, H, D, device=device, dtype=dtype, requires_grad=requires_grad)
    v_vals = torch.randn(total_S, H, D, device=device, dtype=dtype, requires_grad=requires_grad)
    nt_q = torch.nested.nested_tensor_from_jagged(q_vals, offsets=offsets)
    nt_k = torch.nested.nested_tensor_from_jagged(k_vals, offsets=offsets)
    nt_v = torch.nested.nested_tensor_from_jagged(v_vals, offsets=offsets)
    return nt_q, nt_k, nt_v, q_vals, k_vals, v_vals, offsets


def _per_seq_cpu_ref(q_vals, k_vals, v_vals, offsets, seq_lens, is_causal=False):
    """Compute per-sequence CPU SDPA reference in (1, H, S, D) layout."""
    results = []
    for i, s in enumerate(seq_lens):
        start, end = offsets[i].item(), offsets[i + 1].item()
        # vals slice: (S_i, H, D) → transpose to (H, S_i, D) → unsqueeze to (1, H, S_i, D)
        q = q_vals[start:end].float().cpu().transpose(0, 1).unsqueeze(0)
        k = k_vals[start:end].float().cpu().transpose(0, 1).unsqueeze(0)
        v = v_vals[start:end].float().cpu().transpose(0, 1).unsqueeze(0)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        # (1, H, S_i, D) → (S_i, H, D)
        results.append(ref.squeeze(0).transpose(0, 1))
    return results


# ---------------------------------------------------------------------------
# 1. Nested SDPA forward
# ---------------------------------------------------------------------------

@pytest.mark.workload
@pytest.mark.requires("nested")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [True, False])
def test_sdpa_nested_forward(dtype, causal, device, manifest, compare, input_gen):
    torch.manual_seed(42)
    seq_lens = [3, 7, 5]
    H, D = 4, 64

    nt_q, nt_k, nt_v, q_vals, k_vals, v_vals, offsets = _make_nested_qkv(
        seq_lens, H, D, dtype, device
    )
    out = F.scaled_dot_product_attention(nt_q, nt_k, nt_v, is_causal=causal)
    synchronize(device)

    assert out.is_nested, "Output should be a nested tensor"

    refs = _per_seq_cpu_ref(q_vals, k_vals, v_vals, offsets, seq_lens, is_causal=causal)
    tol = {torch.float32: 1e-4, torch.float16: 1e-2, torch.bfloat16: 1e-1}

    out_vals = out.values()
    for i, s in enumerate(seq_lens):
        start, end = offsets[i].item(), offsets[i + 1].item()
        actual_slice = out_vals[start:end].float().cpu()
        ref_slice = refs[i]
        max_diff = (actual_slice - ref_slice).abs().max().item()
        assert max_diff < tol[dtype], (
            f"Seq {i} (len={s}): max diff {max_diff} >= tol {tol[dtype]} for {dtype}"
        )


# ---------------------------------------------------------------------------
# 2. Nested SDPA sequence configurations
# ---------------------------------------------------------------------------

@pytest.mark.workload
@pytest.mark.requires("nested")
@pytest.mark.parametrize("seq_lens", [
    [1, 10, 2, 8],
    [16],
    [1, 1, 1],
    [3, 7, 5, 2, 9],
])
def test_sdpa_nested_seq_configs(seq_lens, device, manifest, compare, input_gen):
    torch.manual_seed(42)
    H, D = 4, 64
    dtype = torch.float32

    nt_q, nt_k, nt_v, q_vals, k_vals, v_vals, offsets = _make_nested_qkv(
        seq_lens, H, D, dtype, device
    )
    out = F.scaled_dot_product_attention(nt_q, nt_k, nt_v, is_causal=False)
    synchronize(device)

    refs = _per_seq_cpu_ref(q_vals, k_vals, v_vals, offsets, seq_lens, is_causal=False)
    out_vals = out.values()
    for i, s in enumerate(seq_lens):
        start, end = offsets[i].item(), offsets[i + 1].item()
        actual_slice = out_vals[start:end].float().cpu()
        ref_slice = refs[i]
        max_diff = (actual_slice - ref_slice).abs().max().item()
        assert max_diff < 1e-4, (
            f"Seq {i} (len={s}): max diff {max_diff} >= 1e-4"
        )


# ---------------------------------------------------------------------------
# 3. Nested SDPA head shapes
# ---------------------------------------------------------------------------

@pytest.mark.workload
@pytest.mark.requires("nested")
@pytest.mark.parametrize("H,D", [(8, 128), (2, 64), (1, 32), (16, 64)])
def test_sdpa_nested_head_shapes(H, D, device, manifest, compare, input_gen):
    torch.manual_seed(42)
    seq_lens = [4, 6, 3]
    dtype = torch.float32

    nt_q, nt_k, nt_v, q_vals, k_vals, v_vals, offsets = _make_nested_qkv(
        seq_lens, H, D, dtype, device
    )
    out = F.scaled_dot_product_attention(nt_q, nt_k, nt_v, is_causal=False)
    synchronize(device)

    refs = _per_seq_cpu_ref(q_vals, k_vals, v_vals, offsets, seq_lens, is_causal=False)
    out_vals = out.values()
    for i, s in enumerate(seq_lens):
        start, end = offsets[i].item(), offsets[i + 1].item()
        actual_slice = out_vals[start:end].float().cpu()
        ref_slice = refs[i]
        max_diff = (actual_slice - ref_slice).abs().max().item()
        assert max_diff < 1e-4, (
            f"Seq {i} (len={s}), H={H}, D={D}: max diff {max_diff} >= 1e-4"
        )


# ---------------------------------------------------------------------------
# 4. Nested SDPA backward smoke
# ---------------------------------------------------------------------------

@pytest.mark.workload
@pytest.mark.requires("nested")
@pytest.mark.parametrize("causal", [True, False])
def test_sdpa_nested_backward_smoke(causal, device, manifest, compare, input_gen):
    torch.manual_seed(42)
    seq_lens = [4, 6]
    H, D = 4, 64
    dtype = torch.float32

    nt_q, nt_k, nt_v, q_vals, k_vals, v_vals, offsets = _make_nested_qkv(
        seq_lens, H, D, dtype, device, requires_grad=True
    )
    out = F.scaled_dot_product_attention(nt_q, nt_k, nt_v, is_causal=causal)
    out.values().sum().backward()
    synchronize(device)

    assert q_vals.grad is not None, "q_vals.grad is None"
    assert k_vals.grad is not None, "k_vals.grad is None"
    assert v_vals.grad is not None, "v_vals.grad is None"
    assert torch.isfinite(q_vals.grad).all(), "q_vals.grad has non-finite values"
    assert torch.isfinite(k_vals.grad).all(), "k_vals.grad has non-finite values"
    assert torch.isfinite(v_vals.grad).all(), "v_vals.grad has non-finite values"
    assert q_vals.grad.abs().sum() > 0, "q_vals.grad is all zeros"
    assert k_vals.grad.abs().sum() > 0, "k_vals.grad is all zeros"
    assert v_vals.grad.abs().sum() > 0, "v_vals.grad is all zeros"


# ---------------------------------------------------------------------------
# 5. Nested SDPA backward vs CPU reference
# ---------------------------------------------------------------------------

@pytest.mark.workload
@pytest.mark.requires("nested")
def test_sdpa_nested_backward_vs_cpu(device, manifest, compare, input_gen):
    torch.manual_seed(42)
    seq_lens = [3, 5]
    H, D = 4, 64
    dtype = torch.float32

    nt_q, nt_k, nt_v, q_vals, k_vals, v_vals, offsets = _make_nested_qkv(
        seq_lens, H, D, dtype, device, requires_grad=True
    )
    out = F.scaled_dot_product_attention(nt_q, nt_k, nt_v, is_causal=False)
    out.values().sum().backward()
    synchronize(device)

    # CPU reference: per-sequence forward + backward
    cpu_dq_parts, cpu_dk_parts, cpu_dv_parts = [], [], []
    for i, s in enumerate(seq_lens):
        start, end = offsets[i].item(), offsets[i + 1].item()
        q_cpu = q_vals[start:end].detach().clone().float().cpu().transpose(0, 1).unsqueeze(0).requires_grad_(True)
        k_cpu = k_vals[start:end].detach().clone().float().cpu().transpose(0, 1).unsqueeze(0).requires_grad_(True)
        v_cpu = v_vals[start:end].detach().clone().float().cpu().transpose(0, 1).unsqueeze(0).requires_grad_(True)
        ref_out = F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu, is_causal=False)
        ref_out.sum().backward()
        # Undo the transpose: (1, H, S, D) -> (S, H, D)
        cpu_dq_parts.append(q_cpu.grad.squeeze(0).transpose(0, 1))
        cpu_dk_parts.append(k_cpu.grad.squeeze(0).transpose(0, 1))
        cpu_dv_parts.append(v_cpu.grad.squeeze(0).transpose(0, 1))

    cpu_dq = torch.cat(cpu_dq_parts, dim=0)
    cpu_dk = torch.cat(cpu_dk_parts, dim=0)
    cpu_dv = torch.cat(cpu_dv_parts, dim=0)

    dev_dq = q_vals.grad.float().cpu()
    dev_dk = k_vals.grad.float().cpu()
    dev_dv = v_vals.grad.float().cpu()

    assert (dev_dq - cpu_dq).abs().max().item() < 1e-4, "dQ mismatch"
    assert (dev_dk - cpu_dk).abs().max().item() < 1e-4, "dK mismatch"
    assert (dev_dv - cpu_dv).abs().max().item() < 1e-4, "dV mismatch"


# ---------------------------------------------------------------------------
# 6. Nested SDPA backward dtypes
# ---------------------------------------------------------------------------

@pytest.mark.workload
@pytest.mark.requires("nested")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sdpa_nested_backward_dtypes(dtype, device, manifest, compare, input_gen):
    torch.manual_seed(42)
    seq_lens = [4, 6]
    H, D = 2, 32

    nt_q, nt_k, nt_v, q_vals, k_vals, v_vals, offsets = _make_nested_qkv(
        seq_lens, H, D, dtype, device, requires_grad=True
    )
    out = F.scaled_dot_product_attention(nt_q, nt_k, nt_v, is_causal=False)
    out.values().sum().backward()
    synchronize(device)

    assert q_vals.grad is not None, "q_vals.grad is None"
    assert k_vals.grad is not None, "k_vals.grad is None"
    assert v_vals.grad is not None, "v_vals.grad is None"
    assert torch.isfinite(q_vals.grad).all(), f"q_vals.grad has non-finite values for {dtype}"
    assert torch.isfinite(k_vals.grad).all(), f"k_vals.grad has non-finite values for {dtype}"
    assert torch.isfinite(v_vals.grad).all(), f"v_vals.grad has non-finite values for {dtype}"
    assert q_vals.grad.abs().sum() > 0, f"q_vals.grad is all zeros for {dtype}"
    assert k_vals.grad.abs().sum() > 0, f"k_vals.grad is all zeros for {dtype}"
    assert v_vals.grad.abs().sum() > 0, f"v_vals.grad is all zeros for {dtype}"


# ---------------------------------------------------------------------------
# 7. Nested SDPA backward multi-sequence
# ---------------------------------------------------------------------------

@pytest.mark.workload
@pytest.mark.requires("nested")
def test_sdpa_nested_backward_multi_seq(device, manifest, compare, input_gen):
    torch.manual_seed(42)
    seq_lens = [2, 4, 6, 8, 3]
    H, D = 4, 64
    dtype = torch.float32

    nt_q, nt_k, nt_v, q_vals, k_vals, v_vals, offsets = _make_nested_qkv(
        seq_lens, H, D, dtype, device, requires_grad=True
    )
    out = F.scaled_dot_product_attention(nt_q, nt_k, nt_v, is_causal=False)
    out.values().sum().backward()
    synchronize(device)

    # CPU reference: per-sequence forward + backward
    cpu_dq_parts, cpu_dk_parts, cpu_dv_parts = [], [], []
    for i, s in enumerate(seq_lens):
        start, end = offsets[i].item(), offsets[i + 1].item()
        q_cpu = q_vals[start:end].detach().clone().float().cpu().transpose(0, 1).unsqueeze(0).requires_grad_(True)
        k_cpu = k_vals[start:end].detach().clone().float().cpu().transpose(0, 1).unsqueeze(0).requires_grad_(True)
        v_cpu = v_vals[start:end].detach().clone().float().cpu().transpose(0, 1).unsqueeze(0).requires_grad_(True)
        ref_out = F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu, is_causal=False)
        ref_out.sum().backward()
        cpu_dq_parts.append(q_cpu.grad.squeeze(0).transpose(0, 1))
        cpu_dk_parts.append(k_cpu.grad.squeeze(0).transpose(0, 1))
        cpu_dv_parts.append(v_cpu.grad.squeeze(0).transpose(0, 1))

    cpu_dq = torch.cat(cpu_dq_parts, dim=0)
    cpu_dk = torch.cat(cpu_dk_parts, dim=0)
    cpu_dv = torch.cat(cpu_dv_parts, dim=0)

    dev_dq = q_vals.grad.float().cpu()
    dev_dk = k_vals.grad.float().cpu()
    dev_dv = v_vals.grad.float().cpu()

    assert (dev_dq - cpu_dq).abs().max().item() < 1e-4, "dQ mismatch (multi-seq)"
    assert (dev_dk - cpu_dk).abs().max().item() < 1e-4, "dK mismatch (multi-seq)"
    assert (dev_dv - cpu_dv).abs().max().item() < 1e-4, "dV mismatch (multi-seq)"


# ---------------------------------------------------------------------------
# 8. GQA with enable_gqa flag
# ---------------------------------------------------------------------------

@pytest.mark.workload
@pytest.mark.parametrize("Hq,Hk", [(8, 2), (8, 4), (32, 8), (4, 1)])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_sdpa_gqa_enable_gqa(Hq, Hk, D, causal, device, manifest, compare, input_gen):
    torch.manual_seed(42)
    B, S = 1, 32
    dtype = torch.float32

    Q = torch.randn(B, Hq, S, D, device=device, dtype=dtype)
    K = torch.randn(B, Hk, S, D, device=device, dtype=dtype)
    V = torch.randn(B, Hk, S, D, device=device, dtype=dtype)

    out = F.scaled_dot_product_attention(Q, K, V, is_causal=causal, enable_gqa=True)
    ref = F.scaled_dot_product_attention(Q.cpu(), K.cpu(), V.cpu(), is_causal=causal, enable_gqa=True)
    synchronize(device)
    compare(out, ref, category="gqa_sdpa", dtype=torch.float32)


# ---------------------------------------------------------------------------
# 9. GQA manual expand vs enable_gqa
# ---------------------------------------------------------------------------

@pytest.mark.workload
@pytest.mark.parametrize("Hq,Hk", [(8, 2), (8, 4), (32, 8), (4, 1)])
@pytest.mark.parametrize("D", [64, 128])
def test_sdpa_gqa_manual_expand(Hq, Hk, D, device, manifest, compare, input_gen):
    torch.manual_seed(42)
    B, S = 1, 32
    dtype = torch.float32

    Q = torch.randn(B, Hq, S, D, device=device, dtype=dtype)
    K = torch.randn(B, Hk, S, D, device=device, dtype=dtype)
    V = torch.randn(B, Hk, S, D, device=device, dtype=dtype)

    K_exp = K.repeat_interleave(Hq // Hk, dim=1)
    V_exp = V.repeat_interleave(Hq // Hk, dim=1)

    out = F.scaled_dot_product_attention(Q, K_exp, V_exp, is_causal=True)
    ref = F.scaled_dot_product_attention(Q.cpu(), K_exp.cpu(), V_exp.cpu(), is_causal=True)
    synchronize(device)
    compare(out, ref, category="gqa_sdpa", dtype=torch.float32)


# ---------------------------------------------------------------------------
# 10. GQA model configurations
# ---------------------------------------------------------------------------

@pytest.mark.workload
@pytest.mark.parametrize("config_name,Hq,Hk,D", [
    ("Llama3-8B", 32, 8, 128),
    ("Llama3-70B", 64, 8, 128),
    ("Mistral-7B", 32, 8, 128),
    ("Gemma2-9B", 16, 8, 128),
    ("Phi-3-small", 32, 8, 96),
    ("TinyLlama", 32, 4, 64),
])
def test_sdpa_gqa_model_configs(config_name, Hq, Hk, D, device, manifest, compare, input_gen):
    torch.manual_seed(42)
    B, S = 1, 64
    dtype = torch.float32

    Q = torch.randn(B, Hq, S, D, device=device, dtype=dtype)
    K = torch.randn(B, Hk, S, D, device=device, dtype=dtype)
    V = torch.randn(B, Hk, S, D, device=device, dtype=dtype)

    out = F.scaled_dot_product_attention(Q, K, V, is_causal=True, enable_gqa=True)
    ref = F.scaled_dot_product_attention(Q.cpu(), K.cpu(), V.cpu(), is_causal=True, enable_gqa=True)
    synchronize(device)
    compare(out, ref, category="gqa_sdpa", dtype=torch.float32)


# ---------------------------------------------------------------------------
# 11. MQA decode step (Hk=1, Sq=1)
# ---------------------------------------------------------------------------

@pytest.mark.workload
@pytest.mark.parametrize("Hq", [4, 8, 16, 32])
@pytest.mark.parametrize("D", [64, 128])
def test_sdpa_mqa_decode(Hq, D, device, manifest, compare, input_gen):
    # Shape coverage: Multi-Query Attention decode step (Hk=1, Sq=1).
    torch.manual_seed(42)
    B, Hk, Sq, Sk = 1, 1, 1, 128
    dtype = torch.float32

    Q = torch.randn(B, Hq, Sq, D, device=device, dtype=dtype)
    K = torch.randn(B, Hk, Sk, D, device=device, dtype=dtype)
    V = torch.randn(B, Hk, Sk, D, device=device, dtype=dtype)

    out = F.scaled_dot_product_attention(Q, K, V, is_causal=False, enable_gqa=True)
    ref = F.scaled_dot_product_attention(Q.cpu(), K.cpu(), V.cpu(), is_causal=False, enable_gqa=True)
    synchronize(device)
    compare(out, ref, category="gqa_sdpa", dtype=torch.float32)


# ---------------------------------------------------------------------------
# 12. GQA backward with enable_gqa
# ---------------------------------------------------------------------------

@pytest.mark.workload
def test_sdpa_gqa_enable_gqa_backward(device, manifest, compare, input_gen):
    torch.manual_seed(42)
    B, Hq, Hk, S, D = 1, 8, 2, 16, 64
    dtype = torch.float32

    Q = torch.randn(B, Hq, S, D, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(B, Hk, S, D, device=device, dtype=dtype)
    V = torch.randn(B, Hk, S, D, device=device, dtype=dtype)

    out = F.scaled_dot_product_attention(Q, K, V, is_causal=True, enable_gqa=True)
    out.sum().backward()
    synchronize(device)

    # CPU reference
    Q_cpu = Q.detach().clone().cpu().requires_grad_(True)
    K_cpu = K.detach().clone().cpu()
    V_cpu = V.detach().clone().cpu()
    ref = F.scaled_dot_product_attention(Q_cpu, K_cpu, V_cpu, is_causal=True, enable_gqa=True)
    ref.sum().backward()

    dev_dq = Q.grad.float().cpu()
    cpu_dq = Q_cpu.grad.float()
    assert (dev_dq - cpu_dq).abs().max().item() < 1e-4, "Q.grad mismatch in GQA backward"

