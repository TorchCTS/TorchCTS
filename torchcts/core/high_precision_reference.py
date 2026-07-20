# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software.

from __future__ import annotations

import math

import mpmath as mp
import torch


MAX_HIGH_PRECISION_ELEMENTS = 64
REFERENCE_DECIMAL_DIGITS = 80


def _bounded_tensors(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    broadcast = torch.broadcast_tensors(*(tensor.detach().cpu() for tensor in tensors))
    if not broadcast or broadcast[0].numel() > MAX_HIGH_PRECISION_ELEMENTS:
        raise ValueError(
            f"high-precision reference is limited to {MAX_HIGH_PRECISION_ELEMENTS} elements"
        )
    return broadcast


def _tensor_from_values(values, template: torch.Tensor) -> torch.Tensor:
    return torch.tensor(values, dtype=template.dtype).reshape(template.shape)


def dirichlet_grad_reference(
    x: torch.Tensor,
    alpha: torch.Tensor,
    total: torch.Tensor,
    *,
    dps: int = REFERENCE_DECIMAL_DIGITS,
) -> torch.Tensor:
    x_cpu, alpha_cpu, total_cpu = _bounded_tensors(x, alpha, total)
    values = []
    with mp.workdps(dps):
        for x_value, alpha_value, total_value in zip(
            x_cpu.reshape(-1).tolist(),
            alpha_cpu.reshape(-1).tolist(),
            total_cpu.reshape(-1).tolist(),
        ):
            xv = mp.mpf(x_value)
            av = mp.mpf(alpha_value)
            bv = mp.mpf(total_value) - av
            if not (0 < xv < 1 and av > 0 and bv > 0):
                values.append(math.nan)
                continue
            derivative_cdf = mp.diff(lambda shape: mp.betainc(shape, bv, 0, xv, regularized=True), av)
            density = xv ** (av - 1) * (1 - xv) ** (bv - 1) / mp.beta(av, bv)
            # aten::_dirichlet_grad returns the implicit Beta derivative in the
            # normalization used by Dirichlet's pathwise backward.  That
            # primitive includes an additional 1 / (1 - x) factor; returning
            # the ordinary Beta quantile derivative here would silently test a
            # different contract.
            values.append(float(-derivative_cdf / (density * (1 - xv))))
    return _tensor_from_values(values, x_cpu)


def standard_gamma_grad_reference(
    alpha: torch.Tensor,
    output: torch.Tensor,
    *,
    dps: int = REFERENCE_DECIMAL_DIGITS,
) -> torch.Tensor:
    alpha_cpu, output_cpu = _bounded_tensors(alpha, output)
    values = []
    with mp.workdps(dps):
        for alpha_value, output_value in zip(alpha_cpu.reshape(-1).tolist(), output_cpu.reshape(-1).tolist()):
            av = mp.mpf(alpha_value)
            xv = mp.mpf(output_value)
            if not (av > 0 and xv > 0):
                values.append(math.nan)
                continue
            derivative_cdf = mp.diff(
                lambda shape: mp.gammainc(shape, 0, xv, regularized=True),
                av,
            )
            density = xv ** (av - 1) * mp.exp(-xv) / mp.gamma(av)
            values.append(float(-derivative_cdf / density))
    return _tensor_from_values(values, alpha_cpu)


def i0_reference(input_tensor: torch.Tensor, *, dps: int = REFERENCE_DECIMAL_DIGITS) -> torch.Tensor:
    (input_cpu,) = _bounded_tensors(input_tensor)
    values = []
    with mp.workdps(dps):
        for value in input_cpu.reshape(-1).tolist():
            if math.isinf(value):
                values.append(math.inf)
            elif math.isnan(value):
                values.append(math.nan)
            else:
                values.append(float(mp.besseli(0, mp.mpf(value))))
    # PyTorch promotes integral and boolean i0 inputs to float32. Preserve
    # that public result contract instead of casting back to the input dtype.
    output_dtype = input_cpu.dtype if input_cpu.dtype.is_floating_point else torch.float32
    return torch.tensor(values, dtype=output_dtype).reshape(input_cpu.shape)


def polygamma_reference(
    order: int,
    input_tensor: torch.Tensor,
    *,
    dps: int = REFERENCE_DECIMAL_DIGITS,
) -> torch.Tensor:
    (input_cpu,) = _bounded_tensors(input_tensor)
    values = []
    with mp.workdps(dps):
        for value in input_cpu.reshape(-1).tolist():
            value = float(value)
            if math.isnan(value):
                values.append(math.nan)
            elif math.isinf(value):
                values.append(0.0 if value > 0 else math.nan)
            elif value <= 0 and value.is_integer():
                values.append(math.inf if order % 2 == 1 else math.nan)
            else:
                values.append(float(mp.polygamma(order, mp.mpf(value))))
    output_dtype = input_cpu.dtype if input_cpu.dtype.is_floating_point else torch.float32
    return torch.tensor(values, dtype=output_dtype).reshape(input_cpu.shape)


def regularized_gamma_reference(
    shape: torch.Tensor,
    value: torch.Tensor,
    *,
    upper: bool,
    dps: int = REFERENCE_DECIMAL_DIGITS,
) -> torch.Tensor:
    shape_cpu, value_cpu = _bounded_tensors(shape, value)
    values = []
    with mp.workdps(dps):
        for shape_value, input_value in zip(
            shape_cpu.reshape(-1).tolist(), value_cpu.reshape(-1).tolist()
        ):
            av = mp.mpf(shape_value)
            xv = mp.mpf(input_value)
            if not (av > 0 and xv >= 0):
                values.append(math.nan)
                continue
            if upper:
                result = mp.gammainc(av, xv, mp.inf, regularized=True)
            else:
                result = mp.gammainc(av, 0, xv, regularized=True)
            values.append(float(result))
    return _tensor_from_values(values, shape_cpu)


def kaiser_window_reference(
    length: int,
    periodic: bool,
    beta: float,
    dtype: torch.dtype,
    *,
    dps: int = REFERENCE_DECIMAL_DIGITS,
) -> torch.Tensor:
    calculation_length = int(length) + 1 if periodic else int(length)
    if calculation_length <= 0:
        return torch.empty((0,), dtype=dtype)
    if calculation_length == 1:
        return torch.ones((1,), dtype=dtype)
    center = mp.mpf(calculation_length - 1) / 2
    values = []
    with mp.workdps(dps):
        beta_mp = mp.mpf(beta)
        denominator = mp.besseli(0, beta_mp) if mp.isfinite(beta_mp) else None
        for index in range(calculation_length):
            if mp.isinf(beta_mp):
                values.append(1.0 if mp.mpf(index) == center else 0.0)
                continue
            position = (mp.mpf(index) - center) / center
            argument = beta_mp * mp.sqrt(max(mp.mpf(0), 1 - position * position))
            values.append(float(mp.besseli(0, argument) / denominator))
    if periodic:
        values = values[:-1]
    real_dtype = torch.float32
    if dtype in {torch.float64, torch.complex128}:
        real_dtype = torch.float64
    elif dtype in {torch.float16, torch.complex32}:
        real_dtype = torch.float16
    elif dtype == torch.bfloat16:
        real_dtype = torch.bfloat16
    result = torch.tensor(values, dtype=real_dtype)
    return result.to(dtype)
