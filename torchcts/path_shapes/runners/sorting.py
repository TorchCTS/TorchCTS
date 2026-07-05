"""Sorting-family path-shape runners."""

from __future__ import annotations

import torch

from torchcts.path_shapes.runners.common import compare_tensor, float_data, run_device_op, torch_dtype


def _input(case: dict, dtype: torch.dtype):
    s = case["shape"]
    x_cpu = float_data(tuple(s["dims"]), dtype)
    if s.get("ties"):
        x_cpu = x_cpu.clone()
        x_cpu.reshape(-1)[::3] = 1.0
    if case.get("stride_mode") == "transposed" and x_cpu.ndim == 2:
        x_cpu = x_cpu.t()
    return x_cpu


def run_sort(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    x_cpu = _input(case, dtype)
    kwargs = {"dim": s["dim"], "descending": bool(s.get("descending", False))}
    if "stable" in s:
        kwargs["stable"] = bool(s["stable"])
    actual_values, actual_indices = run_device_op(case, device, lambda: torch.sort(x_cpu.to(device), **kwargs))
    expected_values, expected_indices = torch.sort(x_cpu, **kwargs)
    compare_tensor(actual_values, expected_values, compare, category="exact", dtype=dtype, device=device)
    compare_tensor(actual_indices, expected_indices, compare, category="exact", dtype=torch.int64, device=device)


def run_topk(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    x_cpu = _input(case, dtype)
    kwargs = {"k": s["k"], "dim": s["dim"], "largest": bool(s.get("largest", True)), "sorted": bool(s.get("sorted", True))}
    actual_values, actual_indices = run_device_op(case, device, lambda: torch.topk(x_cpu.to(device), **kwargs))
    expected_values, expected_indices = torch.topk(x_cpu, **kwargs)
    compare_tensor(actual_values, expected_values, compare, category="exact", dtype=dtype, device=device)
    compare_tensor(actual_indices, expected_indices, compare, category="exact", dtype=torch.int64, device=device)


def run_kthvalue(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    x_cpu = _input(case, dtype)
    actual_values, actual_indices = run_device_op(case, device, lambda: torch.kthvalue(x_cpu.to(device), s["k"], dim=s["dim"]))
    expected_values, expected_indices = torch.kthvalue(x_cpu, s["k"], dim=s["dim"])
    compare_tensor(actual_values, expected_values, compare, category="exact", dtype=dtype, device=device)
    compare_tensor(actual_indices, expected_indices, compare, category="exact", dtype=torch.int64, device=device)
