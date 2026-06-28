# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies or substantial portions of the Software.

from __future__ import annotations

import torch


def assert_same_tensor(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    """Assert exact tensor equality after normalizing both tensors to CPU."""

    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(f"{label} shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}")
    if actual.dtype != expected.dtype:
        raise AssertionError(f"{label} dtype mismatch: {actual.dtype} vs {expected.dtype}")
    if not torch.equal(actual.detach().cpu(), expected.detach().cpu()):
        raise AssertionError(f"{label} value mismatch")


def assert_close_tensor(
    actual: torch.Tensor,
    expected: torch.Tensor,
    label: str,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    """Assert tolerant tensor equality after normalizing both tensors to CPU."""

    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(f"{label} shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}")
    if actual.dtype != expected.dtype:
        raise AssertionError(f"{label} dtype mismatch: {actual.dtype} vs {expected.dtype}")
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    if not torch.allclose(actual_cpu, expected_cpu, rtol=rtol, atol=atol, equal_nan=True):
        diff = (actual_cpu - expected_cpu).abs().max().item()
        raise AssertionError(f"{label} value mismatch; max abs diff {diff}")


def assert_out_identity(actual, out, label: str) -> None:
    """Assert an out= dispatcher returned the exact provided output object."""

    if actual is not out:
        raise AssertionError(f"{label} did not return the provided out tensor")
    if isinstance(actual, torch.Tensor) and isinstance(out, torch.Tensor):
        if actual.data_ptr() != out.data_ptr():
            raise AssertionError(f"{label} returned a tensor with different storage than out")
