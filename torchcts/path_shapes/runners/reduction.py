"""Reduction-family path-shape runners."""

from __future__ import annotations

import torch

from torchcts.path_shapes.runners.common import compare_output, compare_tensor, float_data, positive_data, run_device_op, torch_dtype


def _input(case: dict, dtype: torch.dtype):
    dims = tuple(case["shape"]["dims"])
    x_cpu = positive_data(dims, dtype) if case["runner"] == "reduction.prod" else float_data(dims, dtype)
    if case.get("layout") == "transposed" and len(dims) == 2:
        x_cpu = float_data((dims[1], dims[0]), dtype).t()
    elif case.get("stride_mode") == "expanded" and len(dims) == 2:
        x_cpu = float_data((dims[0], 1), dtype).expand(dims[0], dims[1])
    return x_cpu


def _compare_dtype(result, input_dtype):
    return result.dtype if not result.dtype.is_floating_point else input_dtype


def run_sum(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    x_cpu = _input(case, dtype)
    actual = run_device_op(case, device, lambda: torch.sum(x_cpu.to(device), dim=s["reduce_dim"], keepdim=s.get("keepdim", False)))
    expected = torch.sum(x_cpu, dim=s["reduce_dim"], keepdim=s.get("keepdim", False))
    compare_tensor(actual, expected, compare, category="reduction", dtype=_compare_dtype(expected, dtype), device=device)


def run_mean(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    x_cpu = _input(case, dtype)
    actual = run_device_op(case, device, lambda: torch.mean(x_cpu.to(device), dim=s["reduce_dim"], keepdim=s.get("keepdim", False)))
    expected = torch.mean(x_cpu, dim=s["reduce_dim"], keepdim=s.get("keepdim", False))
    compare_tensor(actual, expected, compare, category="reduction", dtype=dtype, device=device)


def run_amax(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    x_cpu = _input(case, dtype)
    actual = run_device_op(case, device, lambda: torch.amax(x_cpu.to(device), dim=s["reduce_dim"], keepdim=s.get("keepdim", False)))
    expected = torch.amax(x_cpu, dim=s["reduce_dim"], keepdim=s.get("keepdim", False))
    compare_tensor(actual, expected, compare, category="exact", dtype=dtype, device=device)


def run_argmax(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    x_cpu = _input(case, dtype).clone()
    reduce_dim = s["reduce_dim"]
    if x_cpu.ndim == 2 and x_cpu.shape[reduce_dim] > 6:
        if reduce_dim == 0:
            x_cpu[2, :] = 100.0
            x_cpu[5, :] = 100.0
        else:
            x_cpu[:, 2] = 100.0
            x_cpu[:, 5] = 100.0
    actual = run_device_op(case, device, lambda: torch.argmax(x_cpu.to(device), dim=reduce_dim))
    expected = torch.argmax(x_cpu, dim=reduce_dim)
    compare_output("argmax", actual, expected, compare, category="exact", dtype=torch.int64, device=device, input=x_cpu, args=(reduce_dim,), kwargs={"dim": reduce_dim})


def run_prod(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    x_cpu = _input(case, dtype)
    actual = run_device_op(case, device, lambda: torch.prod(x_cpu.to(device), dim=s["reduce_dim"], keepdim=s.get("keepdim", False)))
    expected = torch.prod(x_cpu, dim=s["reduce_dim"], keepdim=s.get("keepdim", False))
    compare_tensor(actual, expected, compare, category="reduction", dtype=dtype, device=device)
