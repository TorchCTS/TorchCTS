"""Convolution-family path-shape runners."""

from __future__ import annotations

import torch.nn.functional as F

from torchcts.path_shapes.runners.common import compare_tensor, float_data, maybe_channels_last, run_device_op, torch_dtype


def run_conv1d(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    x_cpu = float_data((s["n"], s["c"], s["length"]), dtype)
    weight_cpu = float_data((s["out_channels"], s["c"] // s["groups"], s["kernel"]), dtype, scale=0.005)
    bias_cpu = float_data((s["out_channels"],), dtype, scale=0.01)
    kwargs = {"stride": s["stride"], "padding": s["padding"], "dilation": s["dilation"], "groups": s["groups"]}
    actual = run_device_op(case, device, lambda: F.conv1d(x_cpu.to(device), weight_cpu.to(device), bias_cpu.to(device), **kwargs))
    expected = F.conv1d(x_cpu, weight_cpu, bias_cpu, **kwargs)
    compare_tensor(actual, expected, compare, category="conv", dtype=dtype, device=device)


def run_conv2d(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    x_cpu = maybe_channels_last(float_data((s["n"], s["c"], s["h"], s["w"]), dtype), case.get("layout", "nchw"))
    weight_cpu = float_data((s["out_channels"], s["c"] // s["groups"], s["kernel"], s["kernel"]), dtype, scale=0.005)
    bias_cpu = float_data((s["out_channels"],), dtype, scale=0.01)
    kwargs = {"stride": s["stride"], "padding": s["padding"], "dilation": s["dilation"], "groups": s["groups"]}
    actual = run_device_op(case, device, lambda: F.conv2d(x_cpu.to(device), weight_cpu.to(device), bias_cpu.to(device), **kwargs))
    expected = F.conv2d(x_cpu, weight_cpu, bias_cpu, **kwargs)
    compare_tensor(actual, expected, compare, category="conv", dtype=dtype, device=device)


def run_conv3d(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    x_cpu = float_data((s["n"], s["c"], s["d"], s["h"], s["w"]), dtype)
    weight_cpu = float_data((s["out_channels"], s["c"] // s["groups"], s["kernel"], s["kernel"], s["kernel"]), dtype, scale=0.004)
    bias_cpu = float_data((s["out_channels"],), dtype, scale=0.01)
    kwargs = {"stride": s["stride"], "padding": s["padding"], "dilation": s["dilation"], "groups": s["groups"]}
    actual = run_device_op(case, device, lambda: F.conv3d(x_cpu.to(device), weight_cpu.to(device), bias_cpu.to(device), **kwargs))
    expected = F.conv3d(x_cpu, weight_cpu, bias_cpu, **kwargs)
    compare_tensor(actual, expected, compare, category="conv", dtype=dtype, device=device)


def run_conv_transpose2d(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    x_cpu = float_data((s["n"], s["c"], s["h"], s["w"]), dtype)
    weight_cpu = float_data((s["c"], s["out_channels"] // s["groups"], s["kernel"], s["kernel"]), dtype, scale=0.005)
    bias_cpu = float_data((s["out_channels"],), dtype, scale=0.01)
    kwargs = {
        "stride": s["stride"],
        "padding": s["padding"],
        "output_padding": s.get("output_padding", 0),
        "groups": s["groups"],
        "dilation": s["dilation"],
    }
    actual = run_device_op(case, device, lambda: F.conv_transpose2d(x_cpu.to(device), weight_cpu.to(device), bias_cpu.to(device), **kwargs))
    expected = F.conv_transpose2d(x_cpu, weight_cpu, bias_cpu, **kwargs)
    compare_tensor(actual, expected, compare, category="conv", dtype=dtype, device=device)
