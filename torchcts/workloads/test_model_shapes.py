# Copyright (c) 2026 Kris Bailey. MIT License.

import pytest
import torch
import torch.nn.functional as F

from torchcts.core.device import synchronize


@pytest.mark.workload
@pytest.mark.parametrize(
    "Sq, Sk, H, D",
    [
        (1, 1500, 6, 64),
        (4, 1500, 6, 64),
        (1, 128, 4, 64),
        (1, 32, 1, 256),
    ],
)
def test_cross_attention_shapes(Sq, Sk, H, D, device, compare, input_gen):
    # Shape coverage: Whisper-style cross-attention (decode attending full encoder output).
    B = 1
    Q = input_gen((B, H, Sq, D), torch.float32, device)
    K = input_gen((B, H, Sk, D), torch.float32, device)
    V = input_gen((B, H, Sk, D), torch.float32, device)
    out = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
    ref = F.scaled_dot_product_attention(Q.cpu(), K.cpu(), V.cpu(), is_causal=False)
    synchronize(device)
    compare(out, ref, category="sdpa", dtype=torch.float32)


@pytest.mark.workload
def test_padding_mask_attention(device, compare, input_gen):
    # Shape coverage: BERT-style padding mask with variable-length sequences in same batch.
    B, H, S, D = 2, 4, 32, 64
    Q = input_gen((B, H, S, D), torch.float32, device)
    K = input_gen((B, H, S, D), torch.float32, device)
    V = input_gen((B, H, S, D), torch.float32, device)

    seq_lengths = [20, 32]
    attn_mask = torch.zeros(B, 1, S, S, dtype=torch.float32, device=device)
    for b in range(B):
        attn_mask[b, 0, :, seq_lengths[b]:] = float("-inf")
    attn_mask += torch.triu(
        torch.full((S, S), float("-inf"), device=device), diagonal=1
    )

    out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
    ref = F.scaled_dot_product_attention(
        Q.cpu(), K.cpu(), V.cpu(), attn_mask=attn_mask.cpu()
    )
    synchronize(device)
    compare(out, ref, category="sdpa", dtype=torch.float32)


@pytest.mark.workload
def test_relative_position_bias(device, compare):
    # Shape coverage: T5-style relative position bias added to attention scores.
    S, D, H = 32, 64, 4
    num_buckets = 16

    positions = torch.arange(S, device=device)
    rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)  # (S, S)
    # Simple bucketing: clamp and scale
    buckets = (
        rel_pos.clamp(-num_buckets // 2, num_buckets // 2 - 1) + num_buckets // 2
    )  # (S, S) in [0, num_buckets)

    bias_table = torch.randn(num_buckets, H, device=device)  # lookup table
    bias = bias_table[buckets.long()]  # (S, S, H)
    bias = bias.permute(2, 0, 1).unsqueeze(0).float()  # (1, H, S, S)

    Q, K, V = [torch.randn(1, H, S, D, device=device) for _ in range(3)]

    out = F.scaled_dot_product_attention(Q, K, V, attn_mask=bias)
    ref = F.scaled_dot_product_attention(
        Q.cpu(), K.cpu(), V.cpu(), attn_mask=bias.cpu()
    )
    synchronize(device)
    compare(out, ref, category="sdpa", dtype=torch.float32)


@pytest.mark.workload
@pytest.mark.parametrize(
    "K, N",
    [
        (128, 128),
        (4096, 4096),
        (4096, 1024),
        (4097, 4093),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gemv_m1_shapes(K, N, dtype, device, compare):
    # Shape coverage: M=1 GEMV decode-step pattern — exposes kernel padding bugs.
    torch.manual_seed(42)

    a = torch.randn(1, K, dtype=dtype, device=device)
    b = torch.randn(K, N, dtype=dtype, device=device)

    # NN layout
    out_nn = torch.mm(a, b)
    ref_nn = torch.mm(a.cpu(), b.cpu())
    synchronize(device)
    compare(out_nn, ref_nn, category="matmul", dtype=dtype)

    # NT layout
    b_raw = torch.randn(N, K, dtype=dtype, device=device)
    out_nt = torch.mm(a, b_raw.t())
    ref_nt = torch.mm(a.cpu(), b_raw.cpu().t())
    synchronize(device)
    compare(out_nt, ref_nt, category="noncontiguous_mm", dtype=dtype)


@pytest.mark.workload
def test_embedding_large_vocab(device, compare):
    # Shape coverage: Llama-scale embedding table (32000 x 4096) — stresses large index_select.
    torch.manual_seed(42)

    weight = torch.randn(32000, 4096)

    emb_dev = torch.nn.Embedding(32000, 4096).to(device)
    emb_dev.weight.data.copy_(weight)

    emb_cpu = torch.nn.Embedding(32000, 4096)
    emb_cpu.weight.data.copy_(weight)

    indices = torch.randint(0, 32000, (2, 16), dtype=torch.int64)

    out_dev = emb_dev(indices.to(device))
    out_cpu = emb_cpu(indices)
    synchronize(device)
    compare(out_dev, out_cpu, category="exact", dtype=torch.float32)


@pytest.mark.workload
def test_rope_position_encoding(device, compare):
    # Shape coverage: Rotary Position Embedding — standard cos/sin formula.
    D, S = 64, 32

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D, 2).float() / D))
    freqs = torch.outer(torch.arange(S).float(), inv_freq)  # (S, D//2)
    cos = freqs.cos()
    sin = freqs.sin()

    torch.manual_seed(42)
    x = torch.randn(1, 1, S, D)
    x1, x2 = x[..., : D // 2], x[..., D // 2 :]

    # CPU reference
    ref = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    # Device computation with same data
    out = torch.cat([x1.to(device) * cos.to(device) - x2.to(device) * sin.to(device),
                     x2.to(device) * cos.to(device) + x1.to(device) * sin.to(device)], dim=-1)

    synchronize(device)
    compare(out, ref, category="sdpa", dtype=torch.float32)


@pytest.mark.workload
def test_large_batch_matmul(device, compare, input_gen):
    # Shape coverage: large batch count matmul — stresses batched dispatch.
    a = input_gen((256, 128, 128), torch.float32, device)
    b = input_gen((256, 128, 128), torch.float32, device)
    out = torch.bmm(a, b)
    ref = torch.bmm(a.cpu(), b.cpu())
    synchronize(device)
    compare(out, ref, category="matmul", dtype=torch.float32)
