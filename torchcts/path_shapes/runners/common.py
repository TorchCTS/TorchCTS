"""Shared helpers for path-shape runners."""

from __future__ import annotations

import math
from typing import Callable

import pytest
import torch

from torchcts.core.device import synchronize
from torchcts.core.non_unique_output_compare import compare_non_unique_output_if_applicable


def torch_dtype(dtype_name: str) -> torch.dtype:
    prefix = "torch."
    if not dtype_name.startswith(prefix):
        raise ValueError(f"expected torch dtype name, got {dtype_name!r}")
    value = getattr(torch, dtype_name[len(prefix):])
    if not isinstance(value, torch.dtype):
        raise ValueError(f"expected torch dtype name, got {dtype_name!r}")
    return value


def float_data(shape, dtype: torch.dtype, *, device: str = "cpu", scale: float = 0.01, offset: float = 0.0) -> torch.Tensor:
    count = math.prod(tuple(shape))
    base = torch.arange(count, dtype=torch.float32, device=device)
    base = (base - (count / 2)) * scale + offset
    return base.reshape(tuple(shape)).to(dtype)


def positive_data(shape, dtype: torch.dtype, *, device: str = "cpu", scale: float = 0.01) -> torch.Tensor:
    return float_data(shape, dtype, device=device, scale=scale, offset=1.25).abs() + 0.25


def maybe_channels_last(tensor: torch.Tensor, layout: str | None) -> torch.Tensor:
    if layout == "channels_last":
        return tensor.contiguous(memory_format=torch.channels_last)
    return tensor


def padded_2d(rows: int, cols: int, dtype: torch.dtype, device: str = "cpu", *, pad: int = 3) -> torch.Tensor:
    base = float_data((rows, cols + pad), dtype, device=device)
    return base[:, :cols]


def case_device_expectation(case: dict, device: str) -> str:
    expectations = case.get("device_expectation") or {}
    return str(expectations.get(device) or expectations.get("default") or "should_pass")


def maybe_skip_for_device(case: dict, device: str) -> None:
    expectation = case_device_expectation(case, device)
    if expectation == "not_applicable":
        pytest.skip(f"path-shape case {case['case_id']} is not applicable on {device}")


def run_device_op(case: dict, device: str, fn: Callable[[], torch.Tensor | tuple]) -> torch.Tensor | tuple:
    maybe_skip_for_device(case, device)
    try:
        return fn()
    except (NotImplementedError, RuntimeError) as exc:
        expectation = case_device_expectation(case, device)
        message = str(exc)
        unsupported_markers = (
            "not implemented",
            "not available",
            "unsupported",
            "not support",
            "is not currently implemented",
            "could not run",
        )
        if expectation in {"may_skip", "not_applicable"} or any(marker in message.lower() for marker in unsupported_markers):
            if expectation in {"may_skip", "not_applicable"}:
                pytest.skip(f"path-shape case {case['case_id']} unsupported on {device}: {message.splitlines()[0]}")
        raise


def compare_tensor(actual, expected, compare, *, category: str, dtype: torch.dtype, device: str) -> None:
    synchronize(device)
    compare(actual, expected, category=category, dtype=dtype)


def compare_output(
    op_name: str,
    actual,
    expected,
    compare,
    *,
    category: str,
    dtype: torch.dtype,
    device: str,
    input=None,
    args=None,
    kwargs=None,
) -> None:
    synchronize(device)
    if compare_non_unique_output_if_applicable(
        op_name,
        actual,
        expected,
        input=input,
        args=args,
        kwargs=kwargs,
        input_condition="clean",
        category=category,
        dtype=dtype,
        compare=compare,
    ):
        return
    compare(actual, expected, category=category, dtype=dtype)
