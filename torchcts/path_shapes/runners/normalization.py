"""Normalization-family path-shape runners."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from torchcts.path_shapes.runners.common import compare_tensor, float_data, maybe_channels_last, run_device_op, torch_dtype


def run_layer_norm(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    dims = tuple(s["dims"])
    normalized_shape = tuple(s["normalized_shape"])
    x_cpu = float_data(dims, dtype)
    weight_cpu = float_data(normalized_shape, dtype, scale=0.02) + 1.0
    bias_cpu = float_data(normalized_shape, dtype, scale=0.01)
    actual = run_device_op(
        case,
        device,
        lambda: F.layer_norm(x_cpu.to(device), normalized_shape, weight_cpu.to(device), bias_cpu.to(device), eps=s.get("eps", 1e-5)),
    )
    expected = F.layer_norm(x_cpu, normalized_shape, weight_cpu, bias_cpu, eps=s.get("eps", 1e-5))
    compare_tensor(actual, expected, compare, category="norm", dtype=dtype, device=device)


def run_group_norm(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    x_cpu = maybe_channels_last(float_data((s["n"], s["c"], s["h"], s["w"]), dtype), case.get("layout", "nchw"))
    weight_cpu = float_data((s["c"],), dtype, scale=0.02) + 1.0
    bias_cpu = float_data((s["c"],), dtype, scale=0.01)
    actual = run_device_op(case, device, lambda: F.group_norm(x_cpu.to(device), s["groups"], weight_cpu.to(device), bias_cpu.to(device), eps=s.get("eps", 1e-5)))
    expected = F.group_norm(x_cpu, s["groups"], weight_cpu, bias_cpu, eps=s.get("eps", 1e-5))
    compare_tensor(actual, expected, compare, category="norm", dtype=dtype, device=device)


def run_batch_norm(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    x_cpu = maybe_channels_last(float_data((s["n"], s["c"], s["h"], s["w"]), dtype), case.get("layout", "nchw"))
    weight_cpu = float_data((s["c"],), dtype, scale=0.02) + 1.0
    bias_cpu = float_data((s["c"],), dtype, scale=0.01)
    running_mean = torch.zeros((s["c"],), dtype=dtype)
    running_var = torch.ones((s["c"],), dtype=dtype)
    training = bool(s.get("training", False))
    actual = run_device_op(
        case,
        device,
        lambda: F.batch_norm(
            x_cpu.to(device),
            running_mean.to(device),
            running_var.to(device),
            weight_cpu.to(device),
            bias_cpu.to(device),
            training=training,
            momentum=0.1,
            eps=s.get("eps", 1e-5),
        ),
    )
    expected = F.batch_norm(x_cpu, running_mean, running_var, weight_cpu, bias_cpu, training=training, momentum=0.1, eps=s.get("eps", 1e-5))
    compare_tensor(actual, expected, compare, category="norm", dtype=dtype, device=device)
