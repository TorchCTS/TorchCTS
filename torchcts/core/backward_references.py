# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies or substantial portions of the Software.

"""TorchCTS-owned backward contracts for unreliable CPU autograd paths."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch

from torchcts.core.reference_oracles import (
    segment_reduce_prod_backward_reference,
    segment_reduce_prod_reference,
)


@dataclass(frozen=True)
class BackwardContract:
    reference_id: str
    populate_gradients: Callable[[object, tuple, dict], bool]


_BACKWARD_REFERENCE_OPS = frozenset(
    {
        "linalg.cond",
        "linalg.vander",
        "__rpow__",
        "nn.functional.group_norm",
        "sparse.sampled_addmm",
        "renorm",
        "_segment_reduce",
    }
)


def has_opinfo_backward_reference(op_name: str, dtype: torch.dtype) -> bool:
    if op_name not in _BACKWARD_REFERENCE_OPS:
        return False
    if op_name == "_segment_reduce":
        return dtype in {torch.float16, torch.bfloat16}
    if op_name in {"__rpow__", "sparse.sampled_addmm", "renorm"}:
        return dtype in {torch.complex64, torch.complex128}
    return dtype.is_floating_point or dtype.is_complex


def _wide_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype.is_complex:
        return torch.complex128
    return torch.float64


def _sum_to_shape(value: torch.Tensor, shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
    shape = tuple(shape)
    while value.ndim > len(shape):
        value = value.sum(dim=0)
    for dim, size in enumerate(shape):
        if size == 1 and value.shape[dim] != 1:
            value = value.sum(dim=dim, keepdim=True)
    return value.reshape(shape)


def vander_backward_reference(
    input_tensor: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    columns: int | None = None,
    increasing: bool = True,
) -> torch.Tensor:
    if columns is None:
        columns = input_tensor.shape[-1] if input_tensor.ndim else 1
    exponents = list(range(columns))
    if not increasing:
        exponents.reverse()
    wide_input = input_tensor.detach().cpu().to(_wide_dtype(input_tensor.dtype))
    wide_grad = grad_output.detach().cpu().to(_wide_dtype(grad_output.dtype))
    result = torch.zeros_like(wide_input)
    for column, exponent in enumerate(exponents):
        if exponent:
            derivative = exponent * wide_input.pow(exponent - 1)
            result = result + wide_grad[..., column] * derivative.conj()
    return result.to(input_tensor.dtype)


def complex_rpow_backward_reference(
    exponent: torch.Tensor,
    base: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    wide_dtype = torch.complex128
    exponent_wide, base_wide = torch.broadcast_tensors(
        exponent.detach().cpu().to(wide_dtype),
        base.detach().cpu().to(wide_dtype),
    )
    grad_wide = torch.broadcast_to(grad_output.detach().cpu().to(wide_dtype), exponent_wide.shape)
    output_wide = torch.exp(exponent_wide * torch.log(base_wide))
    exponent_grad = grad_wide * (output_wide * torch.log(base_wide)).conj()
    base_grad = grad_wide * (output_wide * exponent_wide / base_wide).conj()
    return (
        _sum_to_shape(exponent_grad, exponent.shape).to(exponent.dtype),
        _sum_to_shape(base_grad, base.shape).to(base.dtype),
    )


def condition_number_backward_reference(
    input_tensor: torch.Tensor,
    p=None,
    grad_output: torch.Tensor | None = None,
) -> torch.Tensor:
    wide_dtype = _wide_dtype(input_tensor.dtype)
    matrix = input_tensor.detach().cpu().to(wide_dtype)
    if grad_output is None:
        grad_output = torch.ones(matrix.shape[:-2], dtype=matrix.real.dtype)
    grad_output = grad_output.detach().cpu().to(matrix.real.dtype)

    if p is None or p in {2, -2}:
        u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
        if p is None or p == 2:
            numerator_index, denominator_index = 0, -1
        else:
            numerator_index, denominator_index = -1, 0
        numerator = singular_values[..., numerator_index]
        denominator = singular_values[..., denominator_index]
        numerator_outer = (
            u[..., :, numerator_index].unsqueeze(-1)
            * vh[..., numerator_index, :].unsqueeze(-2)
        )
        denominator_outer = (
            u[..., :, denominator_index].unsqueeze(-1)
            * vh[..., denominator_index, :].unsqueeze(-2)
        )
        gradient = (
            numerator_outer / denominator[..., None, None]
            - denominator_outer
            * (numerator / denominator.square())[..., None, None]
        )
        gradient = gradient * grad_output[..., None, None]
        return gradient.to(input_tensor.dtype)

    # Other norm families are expressed from their public mathematical
    # definition.  Rebuilding the inverse and norm graph avoids linalg.cond's
    # invalidated saved tensors while retaining the documented subgradients.
    reference_input = matrix.detach().requires_grad_(True)
    inverse = torch.linalg.inv(reference_input)
    first = torch.linalg.matrix_norm(reference_input, ord=p, dim=(-2, -1))
    second = torch.linalg.matrix_norm(inverse, ord=p, dim=(-2, -1))
    (gradient,) = torch.autograd.grad(first * second, reference_input, grad_output)
    return gradient.to(input_tensor.dtype)


def group_norm_backward_reference(
    input_tensor: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None,
    grad_output: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    wide_dtype = _wide_dtype(input_tensor.dtype)
    x = input_tensor.detach().cpu().to(wide_dtype)
    dy = grad_output.detach().cpu().to(wide_dtype)
    batch, channels = x.shape[:2]
    spatial_shape = x.shape[2:]
    group_width = channels // int(num_groups)
    group_size = group_width * math.prod(spatial_shape)
    grouped = x.reshape(batch, int(num_groups), group_size)
    mean = grouped.mean(dim=-1, keepdim=True)
    variance = (grouped - mean).square().mean(dim=-1, keepdim=True)
    inverse_std = torch.rsqrt(variance + eps)
    normalized = ((grouped - mean) * inverse_std).reshape_as(x)

    if weight is None:
        weighted_dy = dy
        weight_grad = None
    else:
        weight_wide = weight.detach().cpu().to(wide_dtype)
        broadcast_shape = (1, channels, *(1 for _ in spatial_shape))
        weighted_dy = dy * weight_wide.reshape(broadcast_shape)
        reduction_dims = (0, *range(2, x.ndim))
        weight_grad = (dy * normalized).sum(dim=reduction_dims).to(weight.dtype)

    grouped_dy = weighted_dy.reshape(batch, int(num_groups), group_size)
    grouped_normalized = normalized.reshape(batch, int(num_groups), group_size)
    input_grad = inverse_std / group_size * (
        group_size * grouped_dy
        - grouped_dy.sum(dim=-1, keepdim=True)
        - grouped_normalized
        * (grouped_dy * grouped_normalized).sum(dim=-1, keepdim=True)
    )
    reduction_dims = (0, *range(2, x.ndim))
    bias_grad = dy.sum(dim=reduction_dims)
    return (
        input_grad.reshape_as(x).to(input_tensor.dtype),
        weight_grad,
        bias_grad.to(input_tensor.dtype),
    )


def sampled_addmm_backward_reference(
    sparse_input: torch.Tensor,
    matrix1: torch.Tensor,
    matrix2: torch.Tensor,
    *,
    alpha=1,
    beta=1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if sparse_input.layout != torch.sparse_csr:
        raise ValueError(f"sampled_addmm reference requires sparse CSR input, got {sparse_input.layout}")
    wide_dtype = _wide_dtype(matrix1.dtype)
    mask = torch.sparse_csr_tensor(
        sparse_input.crow_indices().cpu(),
        sparse_input.col_indices().cpu(),
        torch.ones_like(sparse_input.values(), device="cpu", dtype=wide_dtype),
        size=sparse_input.shape,
    ).to_dense()
    left = matrix1.detach().cpu().to(wide_dtype)
    right = matrix2.detach().cpu().to(wide_dtype)
    alpha_conj = torch.as_tensor(alpha, dtype=wide_dtype).conj()
    beta_conj = torch.as_tensor(beta, dtype=wide_dtype).conj()
    input_grad = mask * beta_conj
    left_grad = alpha_conj * (mask @ right.mH)
    right_grad = alpha_conj * (left.mH @ mask)
    return (
        input_grad.to(sparse_input.dtype),
        left_grad.to(matrix1.dtype),
        right_grad.to(matrix2.dtype),
    )


def renorm_inf_backward_reference(
    input_tensor: torch.Tensor,
    dim: int,
    maxnorm: float,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    wide_dtype = _wide_dtype(input_tensor.dtype)
    x = input_tensor.detach().cpu().to(wide_dtype)
    grad = grad_output.detach().cpu().to(wide_dtype)
    dim = dim % x.ndim
    moved = x.movedim(dim, 0).reshape(x.shape[dim], -1)
    moved_grad = grad.movedim(dim, 0).reshape_as(moved)
    result = torch.empty_like(moved)
    for row_index, (row, row_grad) in enumerate(zip(moved, moved_grad)):
        magnitudes = row.abs()
        norm = magnitudes.max()
        if norm <= maxnorm:
            result[row_index] = row_grad
            continue
        maxima = magnitudes == norm
        if int(maxima.sum()) != 1:
            raise ValueError("renorm infinity reference requires a unique maximum per scaled slice")
        eps = torch.as_tensor(1e-7, dtype=norm.dtype)
        denominator = norm + eps
        scale = maxnorm / denominator
        row_result = row_grad * scale
        maximum_index = int(maxima.nonzero(as_tuple=False)[0, 0])
        real_scale_sensitivity = torch.real((row_grad.conj() * row).sum())
        row_result[maximum_index] += (
            -maxnorm
            * real_scale_sensitivity
            * row[maximum_index]
            / (denominator.square() * norm)
        )
        result[row_index] = row_result
    restored = result.reshape(x.movedim(dim, 0).shape).movedim(0, dim)
    return restored.to(input_tensor.dtype)


def _set_grad(tensor, gradient) -> None:
    if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
        if tensor.layout == torch.sparse_csr and gradient.layout == torch.strided:
            crow = tensor.crow_indices()
            columns = tensor.col_indices()
            rows = torch.repeat_interleave(
                torch.arange(tensor.shape[0], device=crow.device),
                crow[1:] - crow[:-1],
            )
            gradient = torch.sparse_csr_tensor(
                crow,
                columns,
                gradient[rows, columns],
                size=tensor.shape,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        tensor.grad = gradient.detach().to(dtype=tensor.dtype, device=tensor.device)


def resolve_opinfo_backward_reference(op_name, sample, dtype) -> BackwardContract | None:
    if (
        op_name == "_segment_reduce"
        and dtype in {torch.float16, torch.bfloat16}
        and sample.args
        and sample.args[0] == "prod"
    ):
        def populate(data, _args, kwargs):
            reference_output = segment_reduce_prod_reference(
                data,
                lengths=kwargs.get("lengths"),
                offsets=kwargs.get("offsets"),
                axis=int(kwargs.get("axis", 0)),
                initial=kwargs.get("initial"),
            )
            gradient = segment_reduce_prod_backward_reference(
                torch.ones_like(reference_output),
                data,
                lengths=kwargs.get("lengths"),
                offsets=kwargs.get("offsets"),
                axis=int(kwargs.get("axis", 0)),
                initial=kwargs.get("initial"),
            )
            _set_grad(data, gradient)
            return True

        return BackwardContract("segment_prod_exclusive_f32", populate)

    if op_name == "linalg.cond" and isinstance(sample.input, torch.Tensor):
        p = sample.args[0] if sample.args else sample.kwargs.get("p")

        def populate(input_tensor, _args, _kwargs):
            _set_grad(input_tensor, condition_number_backward_reference(input_tensor, p))
            return True

        return BackwardContract("linalg_cond_recomputed", populate)

    if op_name == "linalg.vander" and isinstance(sample.input, torch.Tensor):
        columns = sample.kwargs.get("N", sample.args[0] if sample.args else None)

        def populate(input_tensor, _args, _kwargs):
            columns_value = (
                input_tensor.shape[-1] if columns is None and input_tensor.ndim else
                1 if columns is None else
                int(columns)
            )
            grad_output = torch.ones(
                (*input_tensor.shape, columns_value), dtype=input_tensor.dtype, device="cpu"
            )
            _set_grad(
                input_tensor,
                vander_backward_reference(
                    input_tensor, grad_output, columns=columns_value, increasing=True
                ),
            )
            return True

        return BackwardContract("linalg_vander_column_power", populate)

    if op_name == "__rpow__" and dtype in {torch.complex64, torch.complex128} and sample.args:
        def populate(exponent, args, _kwargs):
            base = args[0]
            shape = torch.broadcast_shapes(exponent.shape, base.shape)
            grad_output = torch.ones(shape, dtype=torch.complex128)
            exponent_grad, base_grad = complex_rpow_backward_reference(
                exponent, base, grad_output
            )
            _set_grad(exponent, exponent_grad)
            _set_grad(base, base_grad)
            return True

        return BackwardContract("complex_rpow_c128", populate)

    if op_name == "nn.functional.group_norm" and isinstance(sample.input, torch.Tensor):
        num_groups = int(sample.args[0])

        def populate(input_tensor, _args, kwargs):
            weight = kwargs.get("weight")
            bias = kwargs.get("bias")
            eps = kwargs.get("eps", 1e-5)
            if input_tensor.numel() == 0:
                return False
            input_grad, weight_grad, bias_grad = group_norm_backward_reference(
                input_tensor,
                num_groups,
                weight,
                torch.ones_like(input_tensor),
                eps,
            )
            _set_grad(input_tensor, input_grad)
            if weight_grad is not None:
                _set_grad(weight, weight_grad)
            if bias is not None:
                _set_grad(bias, bias_grad.to(bias.dtype))
            return True

        return BackwardContract("group_norm_explicit_reduction", populate)

    if op_name == "sparse.sampled_addmm" and dtype.is_complex and len(sample.args) >= 2:
        def populate(sparse_input, args, kwargs):
            matrix1, matrix2 = args[:2]
            input_grad, left_grad, right_grad = sampled_addmm_backward_reference(
                sparse_input,
                matrix1,
                matrix2,
                alpha=kwargs.get("alpha", 1),
                beta=kwargs.get("beta", 1),
            )
            _set_grad(sparse_input, input_grad)
            _set_grad(matrix1, left_grad)
            _set_grad(matrix2, right_grad)
            return True

        return BackwardContract("sampled_addmm_conjugate_vjp", populate)

    if op_name == "renorm" and dtype.is_complex and len(sample.args) >= 3:
        p, dim, maxnorm = sample.args[:3]
        if float(p) == math.inf:
            def populate(input_tensor, _args, _kwargs):
                try:
                    gradient = renorm_inf_backward_reference(
                        input_tensor,
                        int(dim),
                        float(maxnorm),
                        torch.ones_like(input_tensor),
                    )
                except ValueError:
                    return False
                _set_grad(input_tensor, gradient)
                return True

            return BackwardContract("renorm_inf_unique_max", populate)

    return None
