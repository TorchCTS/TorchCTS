"""Attention-family path-shape runners."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from torchcts.path_shapes.runners.common import compare_tensor, float_data, run_device_op, torch_dtype


def run_sdpa(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    q_cpu = float_data((s["batch"], s["heads"], s["sq"], s["head_dim"]), dtype)
    k_cpu = float_data((s["batch"], s["heads"], s["sk"], s["head_dim"]), dtype, scale=0.008)
    v_cpu = float_data((s["batch"], s["heads"], s["sk"], s["head_dim"]), dtype, scale=0.006)
    mask_cpu = None
    if s.get("mask") == "noncontiguous_bool":
        base = torch.ones((s["batch"], s["heads"], s["sk"], s["sq"]), dtype=torch.bool)
        mask_cpu = base.transpose(-1, -2)
    elif s.get("mask") == "bool":
        mask_cpu = torch.ones((s["batch"], s["heads"], s["sq"], s["sk"]), dtype=torch.bool)
    elif s.get("mask") == "float":
        mask_cpu = torch.zeros((s["batch"], s["heads"], s["sq"], s["sk"]), dtype=torch.float32)
        mask_cpu[..., -1] = -0.25

    def call(q, k, v, mask):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=bool(s.get("causal")))

    actual = run_device_op(
        case,
        device,
        lambda: call(q_cpu.to(device), k_cpu.to(device), v_cpu.to(device), mask_cpu.to(device) if mask_cpu is not None else None),
    )
    expected = call(q_cpu, k_cpu, v_cpu, mask_cpu)
    compare_tensor(actual, expected, compare, category="sdpa", dtype=dtype, device=device)
