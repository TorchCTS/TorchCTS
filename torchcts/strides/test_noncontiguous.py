# Copyright (c) 2026 Kris Bailey. MIT License.

import pytest
import torch
import torch.nn.functional as F

from torchcts.core.device import synchronize


# ---------------------------------------------------------------------------
# 1. Binary ops on non-contiguous layouts
# ---------------------------------------------------------------------------
@pytest.mark.medium
@pytest.mark.parametrize("op_name", ["add", "mul", "sub"])
@pytest.mark.parametrize("layout", ["transpose", "sliced", "permuted"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_binary_op_noncontiguous(op_name, layout, dtype, device, compare, input_gen):
    torch.manual_seed(42)
    op_fn = getattr(torch, op_name)

    if layout == "permuted":
        base_a = torch.randn(8, 16, 32, dtype=dtype, device=device)
        a = base_a.permute(2, 0, 1)
        base_b = torch.randn(8, 16, 32, dtype=dtype, device=device)
        b = base_b.permute(2, 0, 1)
    elif layout == "sliced":
        a = input_gen((32, 32), dtype, device, layout="sliced")
        b = input_gen((32, 32), dtype, device, layout="sliced")
    else:  # transpose
        a = input_gen((32, 32), dtype, device, layout="transpose")
        b = input_gen((32, 32), dtype, device, layout="transpose")

    out = op_fn(a, b)
    ref = op_fn(a.cpu(), b.cpu())
    synchronize(device)
    compare(out, ref, category="strided_reduction", dtype=dtype)


# ---------------------------------------------------------------------------
# 2. Binary op with mixed layouts (contiguous + transposed)
# ---------------------------------------------------------------------------
@pytest.mark.medium
def test_binary_op_mixed_layouts(device, compare, input_gen):
    torch.manual_seed(42)
    a = input_gen((32, 32), torch.float32, device, layout="contiguous")
    b = input_gen((32, 32), torch.float32, device, layout="transpose")

    out = torch.add(a, b)
    ref = torch.add(a.cpu(), b.cpu())
    synchronize(device)
    compare(out, ref, category="strided_reduction", dtype=torch.float32)


# ---------------------------------------------------------------------------
# 3. Unary ops on non-contiguous layouts
# ---------------------------------------------------------------------------
@pytest.mark.medium
@pytest.mark.parametrize("op_name", ["abs", "neg", "relu", "exp", "sin"])
@pytest.mark.parametrize("layout", ["transpose", "sliced", "permuted"])
def test_unary_op_noncontiguous(op_name, layout, device, compare):
    torch.manual_seed(42)

    if layout == "transpose":
        x = torch.randn(32, 64, device=device).T  # shape (64, 32)
    elif layout == "sliced":
        base = torch.randn(64, 64, device=device)
        x = base[::2, ::2]
    else:  # permuted
        base = torch.randn(4, 8, 16, device=device)
        x = base.permute(2, 0, 1)

    if op_name in ("exp", "sin"):
        x = x * 0.1

    if op_name == "relu":
        out = F.relu(x)
        ref = F.relu(x.cpu())
    else:
        fn = getattr(torch, op_name)
        out = fn(x)
        ref = fn(x.cpu())

    synchronize(device)
    compare(out, ref, category="elementwise", dtype=torch.float32)


# ---------------------------------------------------------------------------
# 4. Unary ops on strided slices with large strides
# ---------------------------------------------------------------------------
@pytest.mark.medium
@pytest.mark.parametrize("op_name", ["abs", "neg"])
def test_unary_op_strided_slice(op_name, device, compare):
    torch.manual_seed(42)
    base = torch.randn(96, 64, device=device)
    x = base[::3, ::2]  # shape (32, 32), strides (192, 2)
    assert not x.is_contiguous()

    fn = getattr(torch, op_name)
    out = fn(x)
    ref = fn(x.cpu())
    synchronize(device)
    compare(out, ref, category="elementwise", dtype=torch.float32)


# ---------------------------------------------------------------------------
# 5. Reductions on non-contiguous layouts
# ---------------------------------------------------------------------------
@pytest.mark.medium
@pytest.mark.parametrize("op_name", ["sum", "mean", "amax"])
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("layout", ["transpose", "permuted"])
def test_reduction_noncontiguous(op_name, dim, layout, device, compare):
    torch.manual_seed(42)

    if layout == "transpose":
        x = torch.randn(32, 64, device=device).T
    else:  # permuted
        x = torch.randn(4, 8, 16, device=device).permute(2, 0, 1)

    op_fn = getattr(torch, op_name)
    out = op_fn(x, dim=dim)
    ref = op_fn(x.cpu(), dim=dim)
    synchronize(device)
    compare(out, ref, category="strided_reduction", dtype=torch.float32)


# ---------------------------------------------------------------------------
# 6. Reduction with arbitrary (non-uniform) strides
# ---------------------------------------------------------------------------
@pytest.mark.medium
def test_reduction_arbitrary_stride(device, compare):
    torch.manual_seed(42)
    base = torch.randn(64, 64, device=device)
    x = base[::3, 1::2]  # shape ~(22, 32) with non-uniform strides
    assert not x.is_contiguous()

    out0 = torch.sum(x, dim=0)
    ref0 = torch.sum(x.cpu(), dim=0)
    synchronize(device)
    compare(out0, ref0, category="strided_reduction", dtype=torch.float32)

    out1 = torch.sum(x, dim=1)
    ref1 = torch.sum(x.cpu(), dim=1)
    synchronize(device)
    compare(out1, ref1, category="strided_reduction", dtype=torch.float32)


# ---------------------------------------------------------------------------
# 7. Copy from strided source to contiguous destination
# ---------------------------------------------------------------------------
@pytest.mark.medium
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_copy_strided_src_to_contiguous_dst(dtype, device, compare):
    torch.manual_seed(42)
    src_base = torch.randn(32, 32, dtype=dtype, device=device)
    src = src_base.T

    dst = torch.empty(32, 32, dtype=dtype, device=device)
    dst.copy_(src)
    synchronize(device)

    dst_cpu = torch.empty(32, 32, dtype=dtype)
    dst_cpu.copy_(src.cpu())
    compare(dst, dst_cpu, category="copy", dtype=dtype)


# ---------------------------------------------------------------------------
# 8. Copy from contiguous source to strided destination
# ---------------------------------------------------------------------------
@pytest.mark.medium
def test_copy_contiguous_src_to_strided_dst(device, compare):
    torch.manual_seed(42)
    base = torch.randn(32, 32, device=device)
    dst = base.T  # non-contiguous view

    src = torch.randn(32, 32, device=device)

    base_cpu = base.cpu().clone()
    dst_cpu = base_cpu.T
    src_cpu = src.cpu()

    dst.copy_(src)
    dst_cpu.copy_(src_cpu)
    synchronize(device)

    compare(base, base_cpu, category="copy", dtype=torch.float32)


# ---------------------------------------------------------------------------
# 9. Cross-dtype copy from strided source
# ---------------------------------------------------------------------------
@pytest.mark.medium
def test_copy_cross_dtype_strided(device, compare):
    torch.manual_seed(42)
    src_base = torch.randn(32, 32, dtype=torch.float32, device=device)
    src = src_base.T

    dst = torch.empty(32, 32, dtype=torch.bfloat16, device=device)
    dst.copy_(src)
    synchronize(device)

    dst_cpu = torch.empty(32, 32, dtype=torch.bfloat16)
    dst_cpu.copy_(src.cpu())
    compare(dst, dst_cpu, category="copy", dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# 10. Ops on permuted 4D tensors
# ---------------------------------------------------------------------------
@pytest.mark.medium
@pytest.mark.parametrize("op_name", ["add", "abs"])
@pytest.mark.parametrize(
    "perm", [(0, 2, 1, 3), (3, 2, 1, 0), (1, 0, 3, 2)]
)
def test_ops_on_permuted_4d(op_name, perm, device, compare):
    torch.manual_seed(42)
    base = torch.randn(2, 4, 8, 16, device=device)
    x = base.permute(*perm)

    if op_name == "add":
        y = torch.randn_like(x)
        out = torch.add(x, y)
        ref = torch.add(x.cpu(), y.cpu())
    else:  # abs
        out = torch.abs(x)
        ref = torch.abs(x.cpu())

    synchronize(device)
    compare(out, ref, category="elementwise", dtype=torch.float32)


# ---------------------------------------------------------------------------
# 11. Elementwise ops on as_strided views
# ---------------------------------------------------------------------------
@pytest.mark.medium
def test_as_strided_elementwise(device, compare):
    torch.manual_seed(42)
    base = torch.randn(1024, device=device)

    # Non-overlapping gapped view
    view1 = torch.as_strided(base, size=(8, 8), stride=(16, 1))
    out = torch.abs(view1)
    ref = torch.abs(view1.cpu())
    synchronize(device)
    compare(out, ref, category="elementwise", dtype=torch.float32)

    # Contiguous subset view
    view2 = torch.as_strided(base, size=(4, 4), stride=(4, 1))
    out2 = torch.neg(view2)
    ref2 = torch.neg(view2.cpu())
    synchronize(device)
    compare(out2, ref2, category="elementwise", dtype=torch.float32)


# ---------------------------------------------------------------------------
# 12. Expanded tensors with zero stride
# ---------------------------------------------------------------------------
@pytest.mark.medium
@pytest.mark.parametrize("op_name", ["add", "mul"])
def test_expanded_zero_stride(op_name, device, compare):
    torch.manual_seed(42)
    x = torch.randn(1, 64, device=device).expand(32, 64)  # stride-0 in dim 0
    y = torch.randn(32, 64, device=device)

    op_fn = getattr(torch, op_name)
    out = op_fn(x, y)
    ref = op_fn(x.cpu(), y.cpu())
    synchronize(device)
    compare(out, ref, category="elementwise", dtype=torch.float32)


# ---------------------------------------------------------------------------
# 13. Matrix multiply with sliced (non-contiguous) inputs
# ---------------------------------------------------------------------------
@pytest.mark.medium
@pytest.mark.parametrize(
    "M, K, N",
    [
        (1, 128, 128),   # GEMV with non-contiguous sliced operands
        (1, 256, 256),
        (7, 129, 127),
    ],
)
def test_mm_with_sliced_inputs(M, K, N, device, compare):
    torch.manual_seed(42)
    a_base = torch.randn(M * 2, K * 2, device=device)
    a = a_base[::2, ::2]
    assert not a.is_contiguous()

    b = torch.randn(K, N, device=device)
    out = torch.mm(a, b)
    ref = torch.mm(a.cpu(), b.cpu())
    synchronize(device)
    compare(out, ref, category="noncontiguous_mm", dtype=torch.float32)


# ---------------------------------------------------------------------------
# 14. addmm with transposed operands
# ---------------------------------------------------------------------------
@pytest.mark.medium
def test_addmm_transposed_operands(device, compare):
    torch.manual_seed(42)
    M, K, N = 32, 64, 48

    bias = torch.randn(M, N, device=device)
    a_raw = torch.randn(K, M, device=device)
    a = a_raw.T  # (M, K) non-contiguous
    b_raw = torch.randn(N, K, device=device)
    b = b_raw.T  # (K, N) non-contiguous

    out = torch.addmm(bias, a, b)
    ref = torch.addmm(bias.cpu(), a.cpu(), b.cpu())
    synchronize(device)
    compare(out, ref, category="noncontiguous_mm", dtype=torch.float32)
