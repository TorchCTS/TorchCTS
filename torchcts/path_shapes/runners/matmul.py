"""Matmul-family path-shape runners."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from torchcts.path_shapes.runners.common import compare_tensor, float_data, padded_2d, run_device_op, torch_dtype


def run_mm(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    shape = case["shape"]
    m, k, n = shape["m"], shape["k"], shape["n"]
    layout = case.get("layout", "nn")
    padded = case.get("stride_mode") == "padded"

    a_cpu = padded_2d(m, k, dtype) if padded else float_data((m, k), dtype)
    if layout == "nt":
        b_raw_cpu = padded_2d(n, k, dtype) if padded else float_data((n, k), dtype)
        b_cpu = b_raw_cpu.t()
    elif layout == "tn":
        a_raw_cpu = padded_2d(k, m, dtype) if padded else float_data((k, m), dtype)
        a_cpu = a_raw_cpu.t()
        b_cpu = float_data((k, n), dtype)
    else:
        b_cpu = padded_2d(k, n, dtype) if padded else float_data((k, n), dtype)

    actual = run_device_op(case, device, lambda: torch.mm(a_cpu.to(device), b_cpu.to(device)))
    expected = torch.mm(a_cpu, b_cpu)
    category = "noncontiguous_mm" if not a_cpu.is_contiguous() or not b_cpu.is_contiguous() else "matmul"
    compare_tensor(actual, expected, compare, category=category, dtype=dtype, device=device)


def run_bmm(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    shape = case["shape"]
    batch, m, k, n = shape["batch"], shape["m"], shape["k"], shape["n"]
    a_cpu = float_data((batch, m, k), dtype)
    if case.get("stride_mode") == "zero_stride_batch":
        b_cpu = float_data((1, k, n), dtype).expand(batch, -1, -1)
    else:
        b_cpu = float_data((batch, k, n), dtype)
    actual = run_device_op(case, device, lambda: torch.bmm(a_cpu.to(device), b_cpu.to(device)))
    expected = torch.bmm(a_cpu, b_cpu)
    compare_tensor(actual, expected, compare, category="matmul", dtype=dtype, device=device)


def run_matmul(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    shape = case["shape"]
    a_cpu = float_data(tuple(shape["a_shape"]), dtype)
    b_shape = tuple(shape["b_shape"])
    if case.get("stride_mode") == "broadcast_batch":
        b_cpu = float_data((1, *b_shape[-2:]), dtype).expand(*b_shape)
    else:
        b_cpu = float_data(b_shape, dtype)
    actual = run_device_op(case, device, lambda: torch.matmul(a_cpu.to(device), b_cpu.to(device)))
    expected = torch.matmul(a_cpu, b_cpu)
    compare_tensor(actual, expected, compare, category="matmul", dtype=dtype, device=device)


def run_addmm(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    shape = case["shape"]
    m, k, n = shape["m"], shape["k"], shape["n"]
    beta = float(shape.get("beta", 1.0))
    alpha = float(shape.get("alpha", 1.0))
    input_cpu = float_data((m, n), dtype, scale=0.004)
    mat1_cpu = float_data((m, k), dtype)
    mat2_cpu = float_data((k, n), dtype, scale=0.008)
    actual = run_device_op(
        case,
        device,
        lambda: torch.addmm(input_cpu.to(device), mat1_cpu.to(device), mat2_cpu.to(device), beta=beta, alpha=alpha),
    )
    expected = torch.addmm(input_cpu, mat1_cpu, mat2_cpu, beta=beta, alpha=alpha)
    compare_tensor(actual, expected, compare, category="matmul", dtype=dtype, device=device)


def run_linear(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    shape = case["shape"]
    x_cpu = float_data(tuple(shape["input_shape"]), dtype)
    weight_cpu = float_data((shape["out_features"], shape["in_features"]), dtype, scale=0.006)
    bias_cpu = float_data((shape["out_features"],), dtype, scale=0.002) if shape.get("bias", True) else None
    actual = run_device_op(
        case,
        device,
        lambda: F.linear(x_cpu.to(device), weight_cpu.to(device), bias_cpu.to(device) if bias_cpu is not None else None),
    )
    expected = F.linear(x_cpu, weight_cpu, bias_cpu)
    compare_tensor(actual, expected, compare, category="matmul", dtype=dtype, device=device)
