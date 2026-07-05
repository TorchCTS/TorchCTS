"""Composed model-pattern path-shape runners."""

from __future__ import annotations

import torch.nn.functional as F

from torchcts.path_shapes.runners.common import compare_tensor, float_data, maybe_channels_last, run_device_op, torch_dtype


def run_vision_block(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    x_cpu = maybe_channels_last(float_data((s["n"], s["c"], s["h"], s["w"]), dtype), case.get("layout", "nchw"))
    weight_cpu = float_data((s["out_channels"], s["c"], 3, 3), dtype, scale=0.005)
    bias_cpu = float_data((s["out_channels"],), dtype, scale=0.01)

    def block(x, weight, bias):
        return F.relu(F.conv2d(x, weight, bias, padding=1))

    actual = run_device_op(case, device, lambda: block(x_cpu.to(device), weight_cpu.to(device), bias_cpu.to(device)))
    expected = block(x_cpu, weight_cpu, bias_cpu)
    compare_tensor(actual, expected, compare, category="workload_e2e", dtype=dtype, device=device)


def run_depthwise_separable(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    x_cpu = maybe_channels_last(float_data((s["n"], s["c"], s["h"], s["w"]), dtype), case.get("layout", "nchw"))
    depthwise_weight_cpu = float_data((s["c"], 1, 3, 3), dtype, scale=0.005)
    pointwise_weight_cpu = float_data((s["out_channels"], s["c"], 1, 1), dtype, scale=0.004)
    depthwise_bias_cpu = float_data((s["c"],), dtype, scale=0.01)
    pointwise_bias_cpu = float_data((s["out_channels"],), dtype, scale=0.01)

    def block(x, depthwise_weight, pointwise_weight, depthwise_bias, pointwise_bias):
        x = F.conv2d(x, depthwise_weight, depthwise_bias, padding=1, groups=s["c"])
        x = F.relu(x)
        return F.conv2d(x, pointwise_weight, pointwise_bias)

    actual = run_device_op(
        case,
        device,
        lambda: block(
            x_cpu.to(device),
            depthwise_weight_cpu.to(device),
            pointwise_weight_cpu.to(device),
            depthwise_bias_cpu.to(device),
            pointwise_bias_cpu.to(device),
        ),
    )
    expected = block(x_cpu, depthwise_weight_cpu, pointwise_weight_cpu, depthwise_bias_cpu, pointwise_bias_cpu)
    compare_tensor(actual, expected, compare, category="workload_e2e", dtype=dtype, device=device)


def run_patch_embedding(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    x_cpu = float_data((s["n"], s["c"], s["h"], s["w"]), dtype)
    weight_cpu = float_data((s["embed_dim"], s["c"], s["kernel"], s["kernel"]), dtype, scale=0.004)
    bias_cpu = float_data((s["embed_dim"],), dtype, scale=0.01)

    def patch_embed(x, weight, bias):
        out = F.conv2d(x, weight, bias, stride=s["stride"])
        return out.flatten(2).transpose(1, 2)

    actual = run_device_op(case, device, lambda: patch_embed(x_cpu.to(device), weight_cpu.to(device), bias_cpu.to(device)))
    expected = patch_embed(x_cpu, weight_cpu, bias_cpu)
    compare_tensor(actual, expected, compare, category="workload_e2e", dtype=dtype, device=device)


def run_transformer_block_fragment(case: dict, device: str, compare) -> None:
    from torchcts.path_shapes.runners.attention import run_sdpa

    run_sdpa(case, device, compare)


def run_residual_norm(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    x_cpu = float_data(tuple(s["dims"]), dtype)
    residual_cpu = float_data(tuple(s["dims"]), dtype, scale=0.006)
    normalized_shape = tuple(s["normalized_shape"])
    weight_cpu = float_data(normalized_shape, dtype, scale=0.02) + 1.0
    bias_cpu = float_data(normalized_shape, dtype, scale=0.01)

    def block(x, residual, weight, bias):
        return F.layer_norm(x + residual, normalized_shape, weight, bias)

    actual = run_device_op(
        case,
        device,
        lambda: block(x_cpu.to(device), residual_cpu.to(device), weight_cpu.to(device), bias_cpu.to(device)),
    )
    expected = block(x_cpu, residual_cpu, weight_cpu, bias_cpu)
    compare_tensor(actual, expected, compare, category="workload_e2e", dtype=dtype, device=device)
