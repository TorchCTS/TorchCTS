"""Broadcasting-family path-shape runners."""

from __future__ import annotations

import torch

from torchcts.path_shapes.runners.common import compare_tensor, float_data, run_device_op, torch_dtype


def _operands(case: dict, dtype: torch.dtype):
    s = case["shape"]
    a_cpu = float_data(tuple(s["a_shape"]), dtype)
    b_cpu = float_data(tuple(s["b_shape"]), dtype, scale=0.006)
    if case.get("stride_mode") == "zero_stride":
        b_cpu = b_cpu.expand(tuple(s["expanded_b_shape"]))
    return a_cpu, b_cpu


def run_add(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    a_cpu, b_cpu = _operands(case, dtype)
    actual = run_device_op(case, device, lambda: a_cpu.to(device) + b_cpu.to(device))
    expected = a_cpu + b_cpu
    compare_tensor(actual, expected, compare, category="elementwise", dtype=dtype, device=device)


def run_mul(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    a_cpu, b_cpu = _operands(case, dtype)
    actual = run_device_op(case, device, lambda: a_cpu.to(device) * b_cpu.to(device))
    expected = a_cpu * b_cpu
    compare_tensor(actual, expected, compare, category="elementwise", dtype=dtype, device=device)


def run_where(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    a_cpu, b_cpu = _operands(case, dtype)
    mask = torch.zeros(tuple(case["shape"]["mask_shape"]), dtype=torch.bool)
    mask.reshape(-1)[::2] = True
    actual = run_device_op(case, device, lambda: torch.where(mask.to(device), a_cpu.to(device), b_cpu.to(device)))
    expected = torch.where(mask, a_cpu, b_cpu)
    compare_tensor(actual, expected, compare, category="elementwise", dtype=dtype, device=device)


def run_masked_fill(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    x_cpu = float_data(tuple(s["dims"]), dtype)
    mask = torch.zeros(tuple(s["mask_shape"]), dtype=torch.bool)
    mask.reshape(-1)[:: int(s.get("step", 2))] = True
    actual = run_device_op(case, device, lambda: x_cpu.to(device).masked_fill(mask.to(device), float(s.get("value", 3.0))))
    expected = x_cpu.masked_fill(mask, float(s.get("value", 3.0)))
    compare_tensor(actual, expected, compare, category="elementwise", dtype=dtype, device=device)
