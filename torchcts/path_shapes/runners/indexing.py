"""Indexing-family path-shape runners."""

from __future__ import annotations

import torch

from torchcts.path_shapes.runners.common import compare_tensor, float_data, run_device_op, torch_dtype


def run_index_select(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    x_cpu = float_data(tuple(s["dims"]), dtype)
    indices_cpu = torch.tensor(s["indices"], dtype=torch.int64)
    actual = run_device_op(case, device, lambda: torch.index_select(x_cpu.to(device), s["dim"], indices_cpu.to(device)))
    expected = torch.index_select(x_cpu, s["dim"], indices_cpu)
    compare_tensor(actual, expected, compare, category="exact", dtype=dtype, device=device)


def run_gather(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    x_cpu = float_data(tuple(s["dims"]), dtype)
    index_cpu = torch.tensor(s["index"], dtype=torch.int64)
    actual = run_device_op(case, device, lambda: torch.gather(x_cpu.to(device), s["dim"], index_cpu.to(device)))
    expected = torch.gather(x_cpu, s["dim"], index_cpu)
    compare_tensor(actual, expected, compare, category="exact", dtype=dtype, device=device)


def run_scatter_add(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    rows, cols = s["dims"]
    dim = s["dim"]
    x_cpu = torch.zeros((rows, cols), dtype=dtype)
    source_cpu = float_data((4, cols), dtype) if dim == 0 else float_data((rows, 4), dtype)
    if dim == 0:
        index_cpu = torch.tensor([0, 0, rows // 2, rows - 1], dtype=torch.int64).view(4, 1).expand_as(source_cpu)
    else:
        index_cpu = torch.tensor([0, 0, cols // 2, cols - 1], dtype=torch.int64).view(1, 4).expand_as(source_cpu)
    actual = run_device_op(case, device, lambda: x_cpu.to(device).scatter_add_(dim, index_cpu.to(device), source_cpu.to(device)))
    expected = x_cpu.scatter_add_(dim, index_cpu, source_cpu)
    compare_tensor(actual, expected, compare, category="elementwise", dtype=dtype, device=device)


def run_scatter_reduce(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    rows, cols = s["dims"]
    dim = s["dim"]
    x_cpu = torch.zeros((rows, cols), dtype=dtype)
    source_cpu = float_data((4, cols), dtype, offset=1.0) if dim == 0 else float_data((rows, 4), dtype, offset=1.0)
    if dim == 0:
        index_cpu = torch.tensor([0, 0, rows // 2, rows - 1], dtype=torch.int64).view(4, 1).expand_as(source_cpu)
    else:
        index_cpu = torch.tensor([0, 0, cols // 2, cols - 1], dtype=torch.int64).view(1, 4).expand_as(source_cpu)
    reduce = s.get("reduce", "sum")
    actual = run_device_op(case, device, lambda: x_cpu.to(device).scatter_reduce_(dim, index_cpu.to(device), source_cpu.to(device), reduce=reduce))
    expected = x_cpu.scatter_reduce_(dim, index_cpu, source_cpu, reduce=reduce)
    compare_tensor(actual, expected, compare, category="elementwise", dtype=dtype, device=device)


def run_take(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    x_cpu = float_data(tuple(s["dims"]), dtype)
    indices_cpu = torch.tensor(s["indices"], dtype=torch.int64)
    actual = run_device_op(case, device, lambda: torch.take(x_cpu.to(device), indices_cpu.to(device)))
    expected = torch.take(x_cpu, indices_cpu)
    compare_tensor(actual, expected, compare, category="exact", dtype=dtype, device=device)


def run_masked_select(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    x_cpu = float_data(tuple(s["dims"]), dtype)
    mask_cpu = torch.zeros(tuple(s["dims"]), dtype=torch.bool)
    mask_cpu.reshape(-1)[:: int(s.get("step", 3))] = True
    if case.get("stride_mode") == "transposed" and x_cpu.ndim == 2:
        x_cpu = x_cpu.t()
        mask_cpu = mask_cpu.t()
    actual = run_device_op(case, device, lambda: torch.masked_select(x_cpu.to(device), mask_cpu.to(device)))
    expected = torch.masked_select(x_cpu, mask_cpu)
    compare_tensor(actual, expected, compare, category="exact", dtype=dtype, device=device)
