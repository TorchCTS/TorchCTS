"""Spatial-family path-shape runners."""

from __future__ import annotations

import torch.nn.functional as F

from torchcts.path_shapes.runners.common import compare_tensor, float_data, maybe_channels_last, run_device_op, torch_dtype


def _image(case: dict, dtype):
    s = case["shape"]
    return maybe_channels_last(float_data((s["n"], s["c"], s["h"], s["w"]), dtype), case.get("layout", "nchw"))


def run_avg_pool2d(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    x_cpu = _image(case, dtype)
    kwargs = {
        "kernel_size": s["kernel"],
        "stride": s["stride"],
        "padding": s["padding"],
        "ceil_mode": bool(s.get("ceil_mode", False)),
        "count_include_pad": bool(s.get("count_include_pad", True)),
    }
    actual = run_device_op(case, device, lambda: F.avg_pool2d(x_cpu.to(device), **kwargs))
    expected = F.avg_pool2d(x_cpu, **kwargs)
    compare_tensor(actual, expected, compare, category="elementwise", dtype=dtype, device=device)


def run_max_pool2d(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    x_cpu = _image(case, dtype)
    kwargs = {"kernel_size": s["kernel"], "stride": s["stride"], "padding": s["padding"], "ceil_mode": bool(s.get("ceil_mode", False))}
    actual = run_device_op(case, device, lambda: F.max_pool2d(x_cpu.to(device), **kwargs))
    expected = F.max_pool2d(x_cpu, **kwargs)
    compare_tensor(actual, expected, compare, category="elementwise", dtype=dtype, device=device)


def run_adaptive_avg_pool2d(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    x_cpu = _image(case, dtype)
    output_size = tuple(s["output_size"])
    actual = run_device_op(case, device, lambda: F.adaptive_avg_pool2d(x_cpu.to(device), output_size))
    expected = F.adaptive_avg_pool2d(x_cpu, output_size)
    compare_tensor(actual, expected, compare, category="elementwise", dtype=dtype, device=device)


def run_interpolate(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    x_cpu = _image(case, dtype)
    kwargs = {"size": tuple(s["size"]), "mode": s["mode"]}
    if s["mode"] in {"linear", "bilinear", "bicubic", "trilinear"}:
        kwargs["align_corners"] = bool(s.get("align_corners", False))
    actual = run_device_op(case, device, lambda: F.interpolate(x_cpu.to(device), **kwargs))
    expected = F.interpolate(x_cpu, **kwargs)
    compare_tensor(actual, expected, compare, category="elementwise", dtype=dtype, device=device)
