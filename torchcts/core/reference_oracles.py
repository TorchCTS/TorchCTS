# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies or substantial portions of the Software.

from __future__ import annotations

import cmath
import itertools
import math

import torch
import torch.nn.functional as F


_LOW_PRECISION_REAL_DTYPES = frozenset({torch.float16, torch.bfloat16})


def quantized_opmath_tensor(
    tensor: torch.Tensor,
    *,
    opmath_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Freeze public input precision before widening for reference arithmetic."""

    return tensor.detach().cpu().to(tensor.dtype).to(opmath_dtype)


def cast_public_result(result: torch.Tensor, public_dtype: torch.dtype) -> torch.Tensor:
    """Perform the single public narrowing step of a higher-opmath reference."""

    return result.to(public_dtype)


def _complex32_dtype() -> torch.dtype | None:
    value = getattr(torch, "complex32", None)
    return value if isinstance(value, torch.dtype) else None


def _matmul_reference_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().cpu().resolve_conj().resolve_neg()
    complex32 = _complex32_dtype()
    if complex32 is not None and tensor.dtype == complex32:
        return tensor.to(torch.complex64)
    if tensor.dtype == torch.complex64:
        return tensor.to(torch.complex128)
    if tensor.dtype in {
        torch.float16,
        torch.bfloat16,
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
    }:
        return tensor.to(torch.float32)
    if tensor.dtype in {
        getattr(torch, "uint16", None),
        getattr(torch, "uint32", None),
        getattr(torch, "uint64", None),
    }:
        return tensor.to(torch.int64)
    return tensor


def _matmul_reference_result(result: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if result.dtype == dtype:
        return result
    return result.to(dtype)


def _matmul_reference_dtype(args: tuple, kwargs: dict | None) -> torch.dtype:
    def _find(value) -> torch.dtype | None:
        if isinstance(value, torch.Tensor):
            return value.dtype
        if isinstance(value, (list, tuple)):
            for item in value:
                dtype = _find(item)
                if dtype is not None:
                    return dtype
        return None

    for item in args:
        dtype = _find(item)
        if dtype is not None:
            return dtype
    for item in (kwargs or {}).values():
        dtype = _find(item)
        if dtype is not None:
            return dtype
    raise ValueError("matmul reference requires at least one tensor argument")


def _matmul_reference_base(dispatcher_name: str) -> str:
    name = dispatcher_name.removeprefix("aten::")
    base = name.split(".", 1)[0]
    return base.rstrip("_")


def _python_complex_matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_cpu = left.detach().cpu()
    right_cpu = right.detach().cpu()
    left_vector = left_cpu.ndim == 1
    right_vector = right_cpu.ndim == 1
    if left_vector:
        left_cpu = left_cpu.unsqueeze(0)
    if right_vector:
        right_cpu = right_cpu.unsqueeze(-1)
    batch_shape = torch.broadcast_shapes(left_cpu.shape[:-2], right_cpu.shape[:-2])
    left_cpu = left_cpu.expand(*batch_shape, *left_cpu.shape[-2:])
    right_cpu = right_cpu.expand(*batch_shape, *right_cpu.shape[-2:])
    if left_cpu.shape[-1] != right_cpu.shape[-2]:
        raise ValueError("matmul inner dimensions do not agree")
    output = torch.empty(
        (*batch_shape, left_cpu.shape[-2], right_cpu.shape[-1]),
        dtype=torch.promote_types(left_cpu.dtype, right_cpu.dtype),
    )
    batch_indices = itertools.product(*(range(size) for size in batch_shape)) if batch_shape else [()]
    for batch_index in batch_indices:
        for row in range(left_cpu.shape[-2]):
            for column in range(right_cpu.shape[-1]):
                terms = [
                    complex(left_cpu[batch_index + (row, inner)].item())
                    * complex(right_cpu[batch_index + (inner, column)].item())
                    for inner in range(left_cpu.shape[-1])
                ]
                accumulated = terms[0]
                for term in terms[1:]:
                    accumulated += term
                output[batch_index + (row, column)] = accumulated
    if left_vector:
        output = output.squeeze(-2)
    if right_vector:
        output = output.squeeze(-1)
    return output


def _semantic_matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_wide = _matmul_reference_tensor(left)
    right_wide = _matmul_reference_tensor(right)
    if left_wide.is_complex() and (
        not bool(torch.isfinite(torch.view_as_real(left_wide)).all())
        or not bool(torch.isfinite(torch.view_as_real(right_wide)).all())
    ):
        return _python_complex_matmul(left_wide, right_wide)
    return torch.matmul(left_wide, right_wide)


def _semantic_add(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.is_complex() or right.is_complex():
        return _python_complex_tensor_op(
            left,
            right,
            lambda a, b: a + b,
            owner_dtype=torch.promote_types(left.dtype, right.dtype),
        )
    return left + right


def _semantic_scale(value: torch.Tensor, coefficient) -> torch.Tensor:
    if value.is_complex():
        coefficient_value = coefficient
        if isinstance(coefficient, torch.Tensor) and coefficient.numel() == 1:
            coefficient_value = coefficient.detach().cpu().item()
        if isinstance(coefficient_value, (int, float)):
            if coefficient_value == 1:
                return value.clone()
            lanes = torch.view_as_real(value)
            return torch.view_as_complex((lanes * coefficient_value).contiguous())
        if isinstance(coefficient_value, complex) and coefficient_value.imag == 0:
            if coefficient_value.real == 1:
                return value.clone()
            lanes = torch.view_as_real(value)
            return torch.view_as_complex((lanes * coefficient_value.real).contiguous())
        return _python_complex_tensor_op(
            value,
            coefficient,
            lambda a, b: a * b,
            owner_dtype=value.dtype,
        )
    return value * coefficient


def _semantic_sum_first_dim(value: torch.Tensor) -> torch.Tensor:
    if not value.is_complex():
        return value.sum(dim=0)
    result = value[0]
    for index in range(1, value.shape[0]):
        result = _semantic_add(result, value[index])
    return result


def _determinate_sum_terms(terms: list[float]) -> tuple[float, bool]:
    return _determinate_real_linear_sum(terms, [1.0] * len(terms))


def _complex_matmul_determinate(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    left_cpu = _matmul_reference_tensor(left)
    right_cpu = _matmul_reference_tensor(right)
    left_vector = left_cpu.ndim == 1
    right_vector = right_cpu.ndim == 1
    if left_vector:
        left_cpu = left_cpu.unsqueeze(0)
    if right_vector:
        right_cpu = right_cpu.unsqueeze(-1)
    batch_shape = torch.broadcast_shapes(left_cpu.shape[:-2], right_cpu.shape[:-2])
    left_cpu = left_cpu.expand(*batch_shape, *left_cpu.shape[-2:])
    right_cpu = right_cpu.expand(*batch_shape, *right_cpu.shape[-2:])
    if left_cpu.shape[-1] != right_cpu.shape[-2]:
        raise ValueError("matmul inner dimensions do not agree")
    output_shape = (*batch_shape, left_cpu.shape[-2], right_cpu.shape[-1])
    output = torch.empty(output_shape, dtype=torch.complex128)
    mask = torch.empty((*output_shape, 2), dtype=torch.bool)
    batch_indices = itertools.product(*(range(size) for size in batch_shape)) if batch_shape else [()]
    for batch_index in batch_indices:
        for row in range(left_cpu.shape[-2]):
            for column in range(right_cpu.shape[-1]):
                real_terms: list[float] = []
                imaginary_terms: list[float] = []
                for inner in range(left_cpu.shape[-1]):
                    left_value = complex(left_cpu[batch_index + (row, inner)].item())
                    right_value = complex(right_cpu[batch_index + (inner, column)].item())
                    real_terms.extend((
                        left_value.real * right_value.real,
                        -(left_value.imag * right_value.imag),
                    ))
                    imaginary_terms.extend((
                        left_value.real * right_value.imag,
                        left_value.imag * right_value.real,
                    ))
                real, real_is_determinate = _determinate_sum_terms(real_terms)
                imaginary, imaginary_is_determinate = _determinate_sum_terms(imaginary_terms)
                output[batch_index + (row, column)] = complex(real, imaginary)
                mask[batch_index + (row, column, 0)] = real_is_determinate
                mask[batch_index + (row, column, 1)] = imaginary_is_determinate
    if left_vector:
        output = output.squeeze(-2)
        mask = mask.squeeze(-3)
    if right_vector:
        output = output.squeeze(-1)
        mask = mask.squeeze(-2)
    return output, mask


def _scalar_coefficient(value, name: str) -> complex:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar")
        value = value.detach().cpu().item()
    return complex(value)


def _scale_determinate_complex(
    value: torch.Tensor,
    mask: torch.Tensor,
    coefficient: complex,
) -> tuple[torch.Tensor, torch.Tensor]:
    lanes = torch.view_as_real(value.to(torch.complex128))
    output_lanes = torch.empty_like(lanes)
    output_mask = torch.empty_like(mask)
    flattened_lanes = lanes.reshape(-1, 2)
    flattened_mask = mask.reshape(-1, 2)
    flattened_output = output_lanes.reshape(-1, 2)
    flattened_output_mask = output_mask.reshape(-1, 2)
    for index, (value_lanes, value_mask) in enumerate(
        zip(flattened_lanes, flattened_mask)
    ):
        real_terms = (
            (float(value_lanes[0]), coefficient.real, bool(value_mask[0])),
            (float(value_lanes[1]), -coefficient.imag, bool(value_mask[1])),
        )
        imaginary_terms = (
            (float(value_lanes[0]), coefficient.imag, bool(value_mask[0])),
            (float(value_lanes[1]), coefficient.real, bool(value_mask[1])),
        )
        for lane, terms in enumerate((real_terms, imaginary_terms)):
            if not all(term_mask or term_coefficient == 0 for _, term_coefficient, term_mask in terms):
                flattened_output[index, lane] = math.nan
                flattened_output_mask[index, lane] = False
                continue
            result, is_determinate = _determinate_real_linear_sum(
                [component for component, _, _ in terms],
                [term_coefficient for _, term_coefficient, _ in terms],
            )
            flattened_output[index, lane] = result
            flattened_output_mask[index, lane] = is_determinate
    return torch.view_as_complex(output_lanes.contiguous()), output_mask


def _combine_determinate_complex_terms(
    terms: list[tuple[torch.Tensor, torch.Tensor, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    values = [torch.view_as_real(value.to(torch.complex128)) for value, _, _ in terms]
    masks = [mask.to(torch.bool) for _, mask, _ in terms]
    broadcast_values = torch.broadcast_tensors(*values)
    broadcast_masks = torch.broadcast_tensors(*masks)
    output_lanes = torch.empty_like(broadcast_values[0])
    output_mask = torch.empty_like(broadcast_masks[0])
    flattened_values = [value.reshape(-1) for value in broadcast_values]
    flattened_masks = [mask.reshape(-1) for mask in broadcast_masks]
    coefficients = [coefficient for _, _, coefficient in terms]
    for lane in range(output_lanes.numel()):
        if not all(bool(mask[lane]) for mask in flattened_masks):
            output_lanes.reshape(-1)[lane] = math.nan
            output_mask.reshape(-1)[lane] = False
            continue
        result, is_determinate = _determinate_real_linear_sum(
            [float(value[lane]) for value in flattened_values],
            coefficients,
        )
        output_lanes.reshape(-1)[lane] = result
        output_mask.reshape(-1)[lane] = is_determinate
    return torch.view_as_complex(output_lanes.contiguous()), output_mask


def _reduce_determinate_complex_first_dim(
    value: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if value.shape[0] == 0:
        raise ValueError("determinate complex reduction requires a nonempty batch")
    return _combine_determinate_complex_terms([
        (value[index], mask[index], 1.0) for index in range(value.shape[0])
    ])


def _masked_matrix_multiply_determinate(
    left: torch.Tensor,
    left_mask: torch.Tensor,
    right: torch.Tensor,
    right_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("chain_matmul determinate reference requires rank-2 matrices")
    if left.shape[1] != right.shape[0]:
        raise ValueError("chain_matmul inner dimensions do not agree")

    output = torch.empty((left.shape[0], right.shape[1]), dtype=torch.complex128)
    output_mask = torch.empty((*output.shape, 2), dtype=torch.bool)
    for row in range(left.shape[0]):
        for column in range(right.shape[1]):
            real_terms: list[float] = []
            imaginary_terms: list[float] = []
            real_dependencies = True
            imaginary_dependencies = True
            for inner in range(left.shape[1]):
                left_value = complex(left[row, inner].item())
                right_value = complex(right[inner, column].item())
                left_known = left_mask[row, inner]
                right_known = right_mask[inner, column]
                real_dependencies = real_dependencies and bool(
                    left_known[0] and right_known[0] and left_known[1] and right_known[1]
                )
                imaginary_dependencies = imaginary_dependencies and bool(
                    left_known[0] and right_known[1] and left_known[1] and right_known[0]
                )
                real_terms.extend((
                    left_value.real * right_value.real,
                    -(left_value.imag * right_value.imag),
                ))
                imaginary_terms.extend((
                    left_value.real * right_value.imag,
                    left_value.imag * right_value.real,
                ))

            if real_dependencies:
                real, real_known = _determinate_sum_terms(real_terms)
            else:
                real, real_known = math.nan, False
            if imaginary_dependencies:
                imaginary, imaginary_known = _determinate_sum_terms(imaginary_terms)
            else:
                imaginary, imaginary_known = math.nan, False
            output[row, column] = complex(real, imaginary)
            output_mask[row, column, 0] = real_known
            output_mask[row, column, 1] = imaginary_known
    return output, output_mask


def _chain_matmul_determinate(
    matrices: list[torch.Tensor],
    public_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 2 <= len(matrices) <= 5:
        raise ValueError("chain_matmul determinate reference supports two through five matrices")

    candidates: dict[tuple[int, int], list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for index, matrix in enumerate(matrices):
        value = _matmul_reference_tensor(matrix).to(torch.complex128)
        if value.ndim != 2:
            raise ValueError("chain_matmul determinate reference requires rank-2 matrices")
        mask = torch.ones((*value.shape, 2), dtype=torch.bool)
        candidates[index, index + 1] = [(value, mask)]

    operation_budget = 0
    for length in range(2, len(matrices) + 1):
        for start in range(len(matrices) - length + 1):
            stop = start + length
            interval_candidates: list[tuple[torch.Tensor, torch.Tensor]] = []
            for split in range(start + 1, stop):
                for left, left_mask in candidates[start, split]:
                    for right, right_mask in candidates[split, stop]:
                        operation_budget += 4 * left.shape[0] * left.shape[1] * right.shape[1]
                        if operation_budget > 1_000_000:
                            raise ValueError("chain_matmul determinate reference exceeded its scalar-operation budget")
                        interval_candidates.append(
                            _masked_matrix_multiply_determinate(
                                left,
                                left_mask,
                                right,
                                right_mask,
                            )
                        )
            candidates[start, stop] = interval_candidates

    final_candidates = candidates[0, len(matrices)]
    cast_values = [value.to(public_dtype) for value, _ in final_candidates]
    result = cast_values[0].clone()
    result_lanes = torch.view_as_real(result)
    final_mask = torch.stack([mask for _, mask in final_candidates]).all(dim=0)
    candidate_lanes = [torch.view_as_real(value) for value in cast_values]
    for lane_index in range(result_lanes.numel()):
        if not bool(final_mask.reshape(-1)[lane_index]):
            continue
        lane_values = [float(lanes.reshape(-1)[lane_index]) for lanes in candidate_lanes]
        first = lane_values[0]
        if math.isnan(first):
            agree = all(math.isnan(value) for value in lane_values)
        elif math.isinf(first):
            agree = all(value == first for value in lane_values)
        else:
            agree = all(value == first for value in lane_values)
        final_mask.reshape(-1)[lane_index] = agree
    return result, final_mask


def matmul_family_determinate_reference(
    dispatcher_name: str,
    args: tuple,
    kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return complex matmul results only where expanded real lanes are unique."""

    kwargs = dict(kwargs or {})
    dtype = _matmul_reference_dtype(args, kwargs)
    if not dtype.is_complex:
        raise ValueError("matmul determinate-lane reference requires a complex dtype")
    base = _matmul_reference_base(dispatcher_name)

    if base in {"matmul", "linalg_matmul", "mm", "bmm"}:
        if len(args) < 2:
            raise ValueError(f"{dispatcher_name} reference requires two tensor arguments")
        result, mask = _complex_matmul_determinate(args[0], args[1])
    elif base in {"addmm", "addbmm", "baddbmm"}:
        if len(args) < 3:
            raise ValueError(f"{dispatcher_name} reference requires three tensor arguments")
        product, product_mask = _complex_matmul_determinate(args[1], args[2])
        if base == "addbmm":
            product, product_mask = _reduce_determinate_complex_first_dim(
                product, product_mask
            )
        additive = args[0].detach().cpu().to(torch.complex128)
        additive_mask = torch.ones((*additive.shape, 2), dtype=torch.bool)
        additive, additive_mask = _scale_determinate_complex(
            additive,
            additive_mask,
            _scalar_coefficient(kwargs.get("beta", 1), "beta"),
        )
        product, product_mask = _scale_determinate_complex(
            product,
            product_mask,
            _scalar_coefficient(kwargs.get("alpha", 1), "alpha"),
        )
        result, mask = _combine_determinate_complex_terms([
            (additive, additive_mask, 1.0),
            (product, product_mask, 1.0),
        ])
    elif base == "chain_matmul":
        if not args or not isinstance(args[0], (list, tuple)):
            raise ValueError(f"{dispatcher_name} reference requires a tensor list")
        result, mask = _chain_matmul_determinate(list(args[0]), dtype)
    elif base == "linear":
        if len(args) < 2:
            raise ValueError(f"{dispatcher_name} reference requires input and weight")
        result, mask = _complex_matmul_determinate(
            args[0], args[1].transpose(-2, -1)
        )
        bias = args[2] if len(args) >= 3 else kwargs.get("bias")
        if bias is not None:
            bias = bias.detach().cpu().to(torch.complex128)
            bias_mask = torch.ones((*bias.shape, 2), dtype=torch.bool)
            result, mask = _combine_determinate_complex_terms([
                (result, mask, 1.0),
                (bias, bias_mask, 1.0),
            ])
    else:
        raise ValueError(f"{dispatcher_name} is not a supported determinate matmul surface")
    return result.to(dtype), mask


def matmul_family_reference(dispatcher_name: str, args: tuple, kwargs: dict | None = None) -> torch.Tensor:
    """Return a CPU reference for TorchCTS matmul-family generated samples.

    This path is intentionally independent from the same public CPU kernels used
    by PyTorch for the tested dispatcher.  It covers dtypes such as
    ``torch.complex32`` where the dtype exists and device backends may implement
    it, but the local CPU build has no native matmul kernel.
    """

    kwargs = dict(kwargs or {})
    dtype = _matmul_reference_dtype(args, kwargs)
    base = _matmul_reference_base(dispatcher_name)

    if base in {"matmul", "linalg_matmul", "mm"}:
        if len(args) < 2:
            raise ValueError(f"{dispatcher_name} reference requires two tensor arguments")
        result = _semantic_matmul(args[0], args[1])
        return _matmul_reference_result(result, dtype)

    if base == "bmm":
        if len(args) < 2:
            raise ValueError(f"{dispatcher_name} reference requires batch1 and batch2")
        result = _semantic_matmul(args[0], args[1])
        return _matmul_reference_result(result, dtype)

    if base == "addmm":
        if len(args) < 3:
            raise ValueError(f"{dispatcher_name} reference requires input, mat1, and mat2")
        beta = kwargs.get("beta", 1)
        alpha = kwargs.get("alpha", 1)
        input_value = _matmul_reference_tensor(args[0])
        product = _semantic_matmul(args[1], args[2])
        return _matmul_reference_result(
            _semantic_add(_semantic_scale(input_value, beta), _semantic_scale(product, alpha)), dtype
        )

    if base == "addbmm":
        if len(args) < 3:
            raise ValueError(f"{dispatcher_name} reference requires input, batch1, and batch2")
        beta = kwargs.get("beta", 1)
        alpha = kwargs.get("alpha", 1)
        input_value = _matmul_reference_tensor(args[0])
        product = _semantic_sum_first_dim(_semantic_matmul(args[1], args[2]))
        return _matmul_reference_result(
            _semantic_add(_semantic_scale(input_value, beta), _semantic_scale(product, alpha)), dtype
        )

    if base == "baddbmm":
        if len(args) < 3:
            raise ValueError(f"{dispatcher_name} reference requires input, batch1, and batch2")
        beta = kwargs.get("beta", 1)
        alpha = kwargs.get("alpha", 1)
        input_value = _matmul_reference_tensor(args[0])
        product = _semantic_matmul(args[1], args[2])
        return _matmul_reference_result(
            _semantic_add(_semantic_scale(input_value, beta), _semantic_scale(product, alpha)), dtype
        )

    if base == "chain_matmul":
        if not args or not isinstance(args[0], (list, tuple)) or len(args[0]) < 2:
            raise ValueError(f"{dispatcher_name} reference requires a tensor list with at least two matrices")
        matrices = [_matmul_reference_tensor(matrix) for matrix in args[0]]
        result = matrices[0]
        for matrix in matrices[1:]:
            result = _semantic_matmul(result, matrix)
        return _matmul_reference_result(result, dtype)

    if base == "linear":
        if len(args) < 2:
            raise ValueError(f"{dispatcher_name} reference requires input and weight")
        input_value = _matmul_reference_tensor(args[0])
        weight = _matmul_reference_tensor(args[1])
        bias = args[2] if len(args) >= 3 else kwargs.get("bias")
        result = _semantic_matmul(input_value, weight.transpose(-2, -1))
        if bias is not None:
            result = _semantic_add(result, _matmul_reference_tensor(bias))
        return _matmul_reference_result(result, dtype)

    raise ValueError(f"{dispatcher_name} is not a supported matmul-family reference surface")


def soft_margin_loss_reference(
    input_tensor: torch.Tensor,
    target: torch.Tensor,
    reduction: str | int = "mean",
) -> torch.Tensor:
    """Stable soft-margin loss with one final cast to the public dtype."""

    dtype = input_tensor.dtype
    x = quantized_opmath_tensor(input_tensor)
    y = quantized_opmath_tensor(target)
    result = torch.nn.functional.softplus(-(x * y))
    if reduction in ("none", 0):
        pass
    elif reduction in ("mean", 1, None):
        result = result.mean()
    elif reduction in ("sum", 2):
        result = result.sum()
    else:
        raise ValueError(f"unsupported soft-margin reduction {reduction!r}")
    return cast_public_result(result, dtype)


def soft_margin_loss_backward_reference(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    target: torch.Tensor,
    reduction: str | int = "mean",
) -> torch.Tensor:
    """Stable derivative of softplus(-target * input) in f32 opmath."""

    dtype = input_tensor.dtype
    grad = quantized_opmath_tensor(grad_output)
    x = quantized_opmath_tensor(input_tensor)
    y = quantized_opmath_tensor(target)
    derivative = -y * torch.sigmoid(-(x * y))
    if reduction in ("mean", 1, None):
        derivative = derivative / x.numel()
    elif reduction not in ("none", 0, "sum", 2):
        raise ValueError(f"unsupported soft-margin reduction {reduction!r}")
    return cast_public_result(derivative * grad, dtype)


def segment_reduce_prod_backward_reference(
    grad: torch.Tensor,
    data: torch.Tensor,
    *,
    lengths: torch.Tensor | None = None,
    offsets: torch.Tensor | None = None,
    axis: int = 0,
    initial=None,
) -> torch.Tensor:
    """Exclusive-product derivative with explicit zero handling in f32 opmath."""

    axis = int(axis) % data.ndim
    if lengths is None and offsets is None:
        raise ValueError("segment-product reference requires lengths or offsets")
    data_f32 = quantized_opmath_tensor(data)
    grad_f32 = quantized_opmath_tensor(grad)
    result = torch.zeros_like(data_f32)
    initial_f32 = 1.0 if initial is None else float(initial)
    prefix_shape = tuple(data.shape[:axis])
    segment_count = (
        int(lengths.shape[-1])
        if lengths is not None
        else int(offsets.shape[-1]) - 1
    )
    expected_grad_shape = (*prefix_shape, segment_count, *data.shape[axis + 1 :])
    if tuple(grad_f32.shape) != expected_grad_shape:
        raise ValueError(
            f"segment grad shape {tuple(grad_f32.shape)} does not match {expected_grad_shape}"
        )
    prefix_indices = (
        itertools.product(*(range(size) for size in prefix_shape))
        if prefix_shape
        else [()]
    )
    for prefix in prefix_indices:
        if lengths is not None:
            row_lengths = lengths.detach().cpu().to(torch.long)[prefix].reshape(-1)
            stops_tensor = torch.cumsum(row_lengths, dim=0)
            starts = torch.cat((torch.zeros(1, dtype=torch.long), stops_tensor[:-1])).tolist()
            stops = stops_tensor.tolist()
        else:
            row_offsets = offsets.detach().cpu().to(torch.long)[prefix].reshape(-1)
            starts = row_offsets[:-1].tolist()
            stops = row_offsets[1:].tolist()
        for segment, (start, stop) in enumerate(zip(starts, stops)):
            data_index = (*prefix, slice(start, stop), *(
                slice(None) for _ in data.shape[axis + 1 :]
            ))
            values = data_f32[data_index]
            if values.shape[0] == 0:
                continue
            grad_index = (*prefix, segment, *(
                slice(None) for _ in data.shape[axis + 1 :]
            ))
            segment_grad = grad_f32[grad_index]
            zero_mask = values == 0
            zero_count = zero_mask.sum(dim=0)
            nonzero_product = torch.where(
                zero_mask, torch.ones_like(values), values
            ).prod(dim=0) * initial_f32
            exclusive = torch.zeros_like(values)
            no_zeros = zero_count == 0
            one_zero = zero_count == 1
            if bool(no_zeros.any()):
                total = values.prod(dim=0) * initial_f32
                exclusive = torch.where(no_zeros.unsqueeze(0), total / values, exclusive)
            if bool(one_zero.any()):
                exclusive = torch.where(
                    zero_mask & one_zero.unsqueeze(0),
                    nonzero_product.unsqueeze(0),
                    exclusive,
                )
            result[data_index] = exclusive * segment_grad
    return cast_public_result(result, data.dtype)


def segment_reduce_prod_reference(
    data: torch.Tensor,
    *,
    lengths: torch.Tensor | None = None,
    offsets: torch.Tensor | None = None,
    axis: int = 0,
    initial=None,
) -> torch.Tensor:
    """Segment products accumulated once in f32 and narrowed once."""

    axis = int(axis) % data.ndim
    if lengths is None and offsets is None:
        raise ValueError("segment-product reference requires lengths or offsets")
    data_f32 = quantized_opmath_tensor(data)
    initial_f32 = 1.0 if initial is None else float(initial)
    prefix_shape = tuple(data.shape[:axis])
    segment_count = (
        int(lengths.shape[-1])
        if lengths is not None
        else int(offsets.shape[-1]) - 1
    )
    result = torch.empty(
        (*prefix_shape, segment_count, *data.shape[axis + 1 :]),
        dtype=data_f32.dtype,
    )
    prefix_indices = (
        itertools.product(*(range(size) for size in prefix_shape))
        if prefix_shape
        else [()]
    )
    for prefix in prefix_indices:
        if lengths is not None:
            row_lengths = lengths.detach().cpu().to(torch.long)[prefix].reshape(-1)
            stops_tensor = torch.cumsum(row_lengths, dim=0)
            starts = torch.cat((torch.zeros(1, dtype=torch.long), stops_tensor[:-1])).tolist()
            stops = stops_tensor.tolist()
        else:
            row_offsets = offsets.detach().cpu().to(torch.long)[prefix].reshape(-1)
            starts = row_offsets[:-1].tolist()
            stops = row_offsets[1:].tolist()
        for segment, (start, stop) in enumerate(zip(starts, stops)):
            data_index = (*prefix, slice(start, stop), *(
                slice(None) for _ in data.shape[axis + 1 :]
            ))
            result_index = (*prefix, segment, *(
                slice(None) for _ in data.shape[axis + 1 :]
            ))
            result[result_index] = data_f32[data_index].prod(dim=0) * initial_f32
    return cast_public_result(result, data.dtype)


def linear_backward_reference(
    input_tensor: torch.Tensor,
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return CPU autograd reference gradients for linear backward."""

    ref_input = input_tensor.detach().cpu().clone().requires_grad_(True)
    ref_weight = weight.detach().cpu().clone().requires_grad_(True)
    ref_bias = None if bias is None else bias.detach().cpu().clone().requires_grad_(True)
    torch.nn.functional.linear(ref_input, ref_weight, ref_bias).backward(grad_output.detach().cpu())
    return ref_input.grad, ref_weight.grad, None if ref_bias is None else ref_bias.grad


def max_pool2d_backward_reference(
    input_tensor: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode: bool,
) -> torch.Tensor:
    """Return CPU autograd reference input gradient for max_pool2d backward."""

    ref_input = input_tensor.detach().cpu().clone().requires_grad_(True)
    torch.nn.functional.max_pool2d(
        ref_input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    ).backward(grad_output.detach().cpu())
    return ref_input.grad


def pack_int4_values(values: torch.Tensor, *, even_k_in_high_bits: bool) -> torch.Tensor:
    """Pack 0..15 int4 values from [out_features, in_features] into bytes."""

    values = values.detach().cpu()
    if values.dim() != 2:
        raise ValueError(f"int4 values must be 2-D, got shape {tuple(values.shape)}")
    if values.shape[1] % 2:
        raise ValueError("int4 values must have an even in_features dimension")
    if torch.any((values < 0) | (values > 15)):
        raise ValueError("int4 values must be in the inclusive range [0, 15]")

    values_i16 = values.to(torch.int16)
    if even_k_in_high_bits:
        high = values_i16[:, ::2]
        low = values_i16[:, 1::2]
    else:
        high = values_i16[:, 1::2]
        low = values_i16[:, ::2]
    return ((high << 4) | (low & 0x0F)).to(torch.uint8).contiguous()


def unpack_int4_values(packed: torch.Tensor, *, even_k_in_high_bits: bool) -> torch.Tensor:
    """Unpack byte-packed int4 values into [out_features, in_features]."""

    packed_i32 = packed.detach().cpu().to(torch.int32)
    if packed_i32.dim() != 2:
        raise ValueError(f"packed int4 values must be 2-D, got shape {tuple(packed_i32.shape)}")
    high = (packed_i32 >> 4) & 0x0F
    low = packed_i32 & 0x0F
    values = torch.empty((packed_i32.shape[0], packed_i32.shape[1] * 2), dtype=torch.int32)
    if even_k_in_high_bits:
        values[:, ::2] = high
        values[:, 1::2] = low
    else:
        values[:, ::2] = low
        values[:, 1::2] = high
    return values


def tinygemm_int4_dequantize_reference(
    values: torch.Tensor,
    scales_and_zeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Return TinyGEMM-style dequantized int4 weights.

    The MPS/TinyGEMM path interprets each group as
    ``(q - 8) * scale + zero`` with scales/zeros shaped
    ``[num_groups, out_features, 2]``.
    """

    values_f32 = values.detach().cpu().to(torch.float32)
    qparams = scales_and_zeros.detach().cpu().to(torch.float32)
    if values_f32.dim() != 2:
        raise ValueError(f"int4 values must be 2-D, got shape {tuple(values_f32.shape)}")
    if qparams.dim() != 3 or qparams.shape[-1] != 2:
        raise ValueError(f"scales_and_zeros must have shape [groups, out_features, 2], got {tuple(qparams.shape)}")
    out_features, in_features = values_f32.shape
    if group_size <= 0 or in_features % group_size:
        raise ValueError(f"group_size must divide in_features, got group_size={group_size} in_features={in_features}")
    num_groups = in_features // group_size
    if tuple(qparams.shape[:2]) != (num_groups, out_features):
        raise ValueError(
            "scales_and_zeros leading shape must be "
            f"({num_groups}, {out_features}), got {tuple(qparams.shape[:2])}"
        )

    result = torch.empty_like(values_f32)
    for group_index in range(num_groups):
        start = group_index * group_size
        end = start + group_size
        scale = qparams[group_index, :, 0].reshape(out_features, 1)
        zero = qparams[group_index, :, 1].reshape(out_features, 1)
        result[:, start:end] = (values_f32[:, start:end] - 8.0) * scale + zero
    return result


def tinygemm_int4_matmul_reference(
    input_tensor: torch.Tensor,
    values: torch.Tensor,
    scales_and_zeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Return a CPU reference for MPS/TinyGEMM int4 packed-weight matmul."""

    dequantized = tinygemm_int4_dequantize_reference(values, scales_and_zeros, group_size)
    return input_tensor.detach().cpu().to(torch.float32) @ dequantized.T


def unpack_dynamic_int4_weight_bytes(
    packed_weights: torch.Tensor,
    *,
    in_features: int,
    out_features: int,
) -> torch.Tensor:
    """Unpack byte-packed dynamic int4 weights to logical unsigned nibbles.

    ``aten::_dyn_quant_pack_4bit_weight`` expects two logical 4-bit weights per
    input byte. The low nibble maps to the even input feature and the high nibble
    maps to the following odd input feature. For odd ``in_features`` the final
    byte's high nibble is padding and is ignored.
    """

    weights = packed_weights.detach().cpu()
    if weights.dtype != torch.uint8:
        raise ValueError(f"dynamic int4 weights must be uint8, got {weights.dtype}")
    if in_features <= 0 or out_features <= 0:
        raise ValueError(f"in_features and out_features must be positive, got {in_features}, {out_features}")

    bytes_per_row = (in_features + 1) // 2
    expected_numel = out_features * bytes_per_row
    if weights.numel() != expected_numel:
        raise ValueError(
            "dynamic int4 weights must contain exactly "
            f"{expected_numel} bytes for shape ({out_features}, {in_features}), got {weights.numel()}"
        )

    byte_rows = weights.reshape(out_features, bytes_per_row).to(torch.int16)
    low = byte_rows & 0x0F
    high = (byte_rows >> 4) & 0x0F
    values = torch.empty((out_features, bytes_per_row * 2), dtype=torch.int16)
    values[:, 0::2] = low
    values[:, 1::2] = high
    return values[:, :in_features].to(torch.float32)


def dynamic_int4_dequantize_reference(
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    *,
    block_size: int,
    in_features: int,
    out_features: int,
) -> torch.Tensor:
    """Return dequantized dynamic int4 weights with symmetric ``q - 8`` semantics."""

    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if in_features % block_size:
        raise ValueError(f"block_size must divide in_features, got {block_size} and {in_features}")
    num_groups = in_features // block_size

    qvalues = unpack_dynamic_int4_weight_bytes(
        packed_weights,
        in_features=in_features,
        out_features=out_features,
    )
    scale_values = scales.detach().cpu().to(torch.float32).reshape(-1)
    expected_scales = out_features * num_groups
    if scale_values.numel() != expected_scales:
        raise ValueError(
            "dynamic int4 scales must contain exactly "
            f"{expected_scales} values for ({out_features}, {num_groups}), got {scale_values.numel()}"
        )
    scale_rows = scale_values.reshape(out_features, num_groups)

    dequantized = torch.empty((out_features, in_features), dtype=torch.float32)
    centered = qvalues - 8.0
    for group_index in range(num_groups):
        start = group_index * block_size
        end = start + block_size
        dequantized[:, start:end] = centered[:, start:end] * scale_rows[:, group_index].reshape(out_features, 1)
    return dequantized


def dynamic_int4_matmul_reference(
    input_tensor: torch.Tensor,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    block_size: int,
    in_features: int,
    out_features: int,
) -> torch.Tensor:
    """Return a CPU reference for dynamic 4-bit packed-weight matmul."""

    if input_tensor.dim() != 2:
        raise ValueError(f"dynamic int4 input must be 2-D, got shape {tuple(input_tensor.shape)}")
    if input_tensor.shape[1] != in_features:
        raise ValueError(f"input last dimension must be {in_features}, got {input_tensor.shape[1]}")

    weights = dynamic_int4_dequantize_reference(
        packed_weights,
        scales,
        block_size=block_size,
        in_features=in_features,
        out_features=out_features,
    )
    result = input_tensor.detach().cpu().to(torch.float32) @ weights.T
    if bias is not None:
        bias_cpu = bias.detach().cpu().to(torch.float32)
        if tuple(bias_cpu.shape) != (out_features,):
            raise ValueError(f"dynamic int4 bias must have shape ({out_features},), got {tuple(bias_cpu.shape)}")
        result = result + bias_cpu
    return result.to(input_tensor.dtype)


def complex_unit_alpha_add_sub_reference(
    operation: str,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Compute complex add/sub/rsub without multiplying by ``1+0j``."""

    left_cpu, right_cpu = torch.broadcast_tensors(left.detach().cpu(), right.detach().cpu())
    if not left_cpu.is_complex() or not right_cpu.is_complex():
        raise ValueError("unit-alpha complex reference requires two complex tensors")
    if operation == "add":
        real = left_cpu.real + right_cpu.real
        imag = left_cpu.imag + right_cpu.imag
    elif operation == "sub":
        real = left_cpu.real - right_cpu.real
        imag = left_cpu.imag - right_cpu.imag
    elif operation == "rsub":
        real = right_cpu.real - left_cpu.real
        imag = right_cpu.imag - left_cpu.imag
    else:
        raise ValueError(f"unsupported unit-alpha operation {operation!r}")
    return torch.complex(real, imag).to(left_cpu.dtype)


def _nonnegative_integral_exponent_mask(exponent: torch.Tensor) -> torch.Tensor:
    exponent_cpu = exponent.detach().cpu()
    if not exponent_cpu.is_complex():
        return torch.zeros(exponent_cpu.shape, dtype=torch.bool)
    real = exponent_cpu.real
    return (
        torch.isfinite(real)
        & torch.isfinite(exponent_cpu.imag)
        & (exponent_cpu.imag == 0)
        & (real >= 0)
        & (real == torch.trunc(real))
    )


def has_nonnegative_integral_complex_exponent(exponent: torch.Tensor) -> bool:
    return bool(_nonnegative_integral_exponent_mask(exponent).any())


def _complex_integer_power_scalar(base: torch.Tensor, exponent: int) -> torch.Tensor:
    if exponent < 0:
        raise ValueError("integer-power reference accepts non-negative exponents only")
    if exponent == 0:
        return torch.ones((), dtype=base.dtype)
    if exponent == 1:
        return base.clone()

    result = None
    factor = base
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = factor.clone() if result is None else torch.mul(result, factor)
        remaining >>= 1
        if remaining:
            factor = torch.mul(factor, factor)
    return result


def complex_tensor_integer_power_reference(
    base: torch.Tensor,
    exponent: torch.Tensor,
    native_result: torch.Tensor,
) -> torch.Tensor:
    """Patch exact non-negative integer exponent lanes in a native power result."""

    base_cpu, exponent_cpu = torch.broadcast_tensors(base.detach().cpu(), exponent.detach().cpu())
    if not base_cpu.is_complex() or not exponent_cpu.is_complex():
        raise ValueError("complex tensor-power reference requires complex base and exponent tensors")
    result = native_result.detach().cpu().clone()
    if tuple(result.shape) != tuple(base_cpu.shape):
        raise ValueError(
            f"native power result shape {tuple(result.shape)} does not match broadcast shape {tuple(base_cpu.shape)}"
        )
    mask = _nonnegative_integral_exponent_mask(exponent_cpu).reshape(-1)
    base_flat = base_cpu.reshape(-1)
    exponent_flat = exponent_cpu.reshape(-1)
    result_flat = result.reshape(-1)
    for index in mask.nonzero(as_tuple=False).reshape(-1).tolist():
        exponent_value = int(exponent_flat[index].real.item())
        corrected = _complex_integer_power_scalar(
            base_flat[index].to(result.dtype),
            exponent_value,
        )
        result_flat[index] = corrected
    return result


def complex_l1_loss_reference(
    input_tensor: torch.Tensor,
    target: torch.Tensor,
    reduction,
) -> torch.Tensor:
    """Compute L1 loss using independent real and imaginary subtraction lanes."""

    input_cpu, target_cpu = torch.broadcast_tensors(input_tensor.detach().cpu(), target.detach().cpu())
    if not (input_cpu.is_complex() or target_cpu.is_complex()):
        raise ValueError("complex L1 reference requires at least one complex operand")
    real_dtype = (
        torch.float64
        if any(dtype in {torch.float64, torch.complex128} for dtype in (input_cpu.dtype, target_cpu.dtype))
        else torch.float32
    )

    def lanes(value):
        if value.is_complex():
            return value.real.to(real_dtype), value.imag.to(real_dtype)
        return value.to(real_dtype), torch.zeros_like(value, dtype=real_dtype)

    input_real, input_imag = lanes(input_cpu)
    target_real, target_imag = lanes(target_cpu)
    magnitude = torch.hypot(input_real - target_real, input_imag - target_imag)
    if reduction in ("none", 0):
        return magnitude
    if reduction in ("mean", 1, None):
        return magnitude.mean()
    if reduction in ("sum", 2):
        return magnitude.sum()
    raise ValueError(f"unsupported L1 reduction {reduction!r}")


def binary_cross_entropy_with_logits_reference(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    pos_weight: torch.Tensor | None = None,
    reduction="mean",
) -> torch.Tensor:
    """Stable BCE-with-logits reference with coefficient-aware infinities."""

    logits_cpu, target_cpu = torch.broadcast_tensors(
        logits.detach().cpu(),
        target.detach().cpu(),
    )
    if not logits_cpu.is_floating_point() or not target_cpu.is_floating_point():
        raise ValueError("BCE-with-logits reference requires floating logits and target")
    if not bool(torch.isfinite(target_cpu).all()) or not bool(
        ((target_cpu >= 0) & (target_cpu <= 1)).all()
    ):
        raise ValueError("BCE-with-logits target must be finite and within [0, 1]")

    opmath_dtype = torch.float64 if logits_cpu.dtype == torch.float64 else torch.float32
    x = logits_cpu.to(opmath_dtype)
    y = target_cpu.to(opmath_dtype)
    positive_scale = y
    if pos_weight is not None:
        positive_scale = positive_scale * pos_weight.detach().cpu().to(opmath_dtype)
    negative_scale = 1 - y

    positive_loss = torch.nn.functional.softplus(-x)
    negative_loss = torch.nn.functional.softplus(x)
    positive_term = torch.where(
        positive_scale == 0,
        torch.zeros_like(positive_loss),
        positive_scale * positive_loss,
    )
    negative_term = torch.where(
        negative_scale == 0,
        torch.zeros_like(negative_loss),
        negative_scale * negative_loss,
    )
    result = positive_term + negative_term
    if weight is not None:
        weight_cpu = weight.detach().cpu().to(opmath_dtype)
        result = torch.where(weight_cpu == 0, torch.zeros_like(result), result * weight_cpu)

    result = result.to(logits_cpu.dtype)
    if reduction in ("none", 0):
        return result
    if reduction in ("mean", 1, None):
        return result.mean()
    if reduction in ("sum", 2):
        return result.sum()
    raise ValueError(f"unsupported BCE-with-logits reduction {reduction!r}")


def complex_log2_reference(input_tensor: torch.Tensor) -> torch.Tensor:
    input_cpu = input_tensor.detach().cpu()
    if not input_cpu.is_complex():
        raise ValueError("complex log2 reference requires a complex tensor")
    natural = torch.log(input_cpu)
    scale = 1.0 / math.log(2.0)
    return torch.complex(natural.real * scale, natural.imag * scale).to(input_cpu.dtype)


def _map_python_complex(input_tensor: torch.Tensor, fn) -> torch.Tensor:
    input_cpu = input_tensor.detach().cpu()
    if not input_cpu.is_complex():
        raise ValueError("complex scalar reference requires a complex tensor")
    values = [fn(complex(value)) for value in input_cpu.reshape(-1).tolist()]
    return torch.tensor(values, dtype=input_cpu.dtype).reshape(input_cpu.shape)


def complex_sigmoid_reference(input_tensor: torch.Tensor) -> torch.Tensor:
    def scalar(value: complex) -> complex:
        if value.real == -math.inf:
            return complex(0.0, math.copysign(0.0, value.imag))
        if value.real == math.inf:
            return complex(1.0, math.copysign(0.0, value.imag))
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            return complex(math.nan, math.nan)
        if value.real < 0:
            exponential = cmath.exp(value)
            return exponential / (1 + exponential)
        return 1 / (1 + cmath.exp(-value))

    return _map_python_complex(input_tensor, scalar)


def complex_rsqrt_reference(input_tensor: torch.Tensor) -> torch.Tensor:
    def scalar(value: complex) -> complex:
        root = cmath.sqrt(value)
        if root == 0:
            return complex(math.inf, math.copysign(0.0, -root.imag))
        return 1 / root

    return _map_python_complex(input_tensor, scalar)


def complex_expm1_reference(input_tensor: torch.Tensor) -> torch.Tensor:
    def scalar(value: complex) -> complex:
        if value.real == -math.inf:
            return complex(-1.0, math.copysign(0.0, value.imag))
        if value.real == math.inf and not math.isfinite(value.imag):
            return complex(math.inf, math.nan)
        if not math.isfinite(value.imag):
            return complex(math.nan, math.nan)
        if value.real == math.inf:
            return complex(
                math.copysign(math.inf, math.cos(value.imag)),
                math.copysign(math.inf, math.sin(value.imag)),
            )
        return cmath.exp(value) - 1

    return _map_python_complex(input_tensor, scalar)


def complex_ldexp_reference(input_tensor: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    input_cpu, exponent_cpu = torch.broadcast_tensors(input_tensor.detach().cpu(), exponent.detach().cpu())
    if not input_cpu.is_complex() or not exponent_cpu.is_complex():
        raise ValueError("complex ldexp reference requires complex input and exponent")

    def scalar(value: complex, power: complex) -> complex:
        phase = math.log(2.0) * power.imag
        if power.real == -math.inf and math.isfinite(value.real) and math.isfinite(value.imag):
            return complex(math.copysign(0.0, value.real), math.copysign(0.0, value.imag))
        if power.real == math.inf and math.isfinite(phase) and math.isfinite(value.real) and math.isfinite(value.imag):
            rotated = value * complex(math.cos(phase), math.sin(phase))

            def divergent(component: float) -> float:
                return math.copysign(math.inf, component) if component != 0 else math.copysign(0.0, component)

            return complex(divergent(rotated.real), divergent(rotated.imag))
        if power.real == math.inf and not math.isfinite(phase):
            return value * complex(math.inf, math.nan)
        if not math.isfinite(power.real) or not math.isfinite(phase):
            return complex(math.nan, math.nan)
        factor = cmath.exp(complex(math.log(2.0) * power.real, phase))
        return value * factor

    values = [
        scalar(complex(value), complex(power))
        for value, power in zip(input_cpu.reshape(-1).tolist(), exponent_cpu.reshape(-1).tolist())
    ]
    return torch.tensor(values, dtype=input_cpu.dtype).reshape(input_cpu.shape)


def complex_integral_ldexp_reference(
    input_tensor: torch.Tensor,
    exponent: torch.Tensor,
) -> torch.Tensor:
    """Scale complex lanes independently for the integral-exponent overload."""

    input_cpu, exponent_cpu = torch.broadcast_tensors(
        input_tensor.detach().cpu(), exponent.detach().cpu()
    )
    if not input_cpu.is_complex():
        raise ValueError("integral-exponent complex ldexp requires complex input")
    if exponent_cpu.dtype == torch.bool or exponent_cpu.is_floating_point() or exponent_cpu.is_complex():
        raise ValueError("integral-exponent complex ldexp requires an integral exponent")

    real = wide_ldexp_reference(input_cpu.real, exponent_cpu)
    imag = wide_ldexp_reference(input_cpu.imag, exponent_cpu)
    return torch.complex(real, imag).to(input_cpu.dtype)


def complex_mul_reference(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_cpu, right_cpu = torch.broadcast_tensors(left.detach().cpu(), right.detach().cpu())
    values = [
        complex(a) * complex(b)
        for a, b in zip(left_cpu.reshape(-1).tolist(), right_cpu.reshape(-1).tolist())
    ]
    return torch.tensor(values, dtype=torch.promote_types(left_cpu.dtype, right_cpu.dtype)).reshape(left_cpu.shape)


def _python_complex_tensor_op(left, right, fn, *, owner_dtype: torch.dtype) -> torch.Tensor:
    def normalize(value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().to(owner_dtype)
        return torch.tensor(value, dtype=owner_dtype)

    left_cpu, right_cpu = torch.broadcast_tensors(normalize(left), normalize(right))
    values = [
        fn(complex(a), complex(b))
        for a, b in zip(left_cpu.reshape(-1).tolist(), right_cpu.reshape(-1).tolist())
    ]
    return torch.tensor(values, dtype=owner_dtype).reshape(left_cpu.shape)


def _python_complex_scalar_scale(value, scale, *, owner_dtype: torch.dtype) -> torch.Tensor:
    scale_cpu = scale.detach().cpu() if isinstance(scale, torch.Tensor) else torch.tensor(scale)
    if scale_cpu.numel() != 1:
        return _python_complex_tensor_op(
            value,
            scale,
            lambda left, right: left * right,
            owner_dtype=owner_dtype,
        )

    scalar = complex(scale_cpu.item())
    if scalar.imag != 0:
        return _python_complex_tensor_op(
            value,
            scalar,
            lambda left, right: left * right,
            owner_dtype=owner_dtype,
        )

    value_cpu = (
        value.detach().cpu().to(owner_dtype)
        if isinstance(value, torch.Tensor)
        else torch.tensor(value, dtype=owner_dtype)
    )
    real_scale = float(scalar.real)
    if real_scale == 1.0:
        return value_cpu.clone()
    values = [
        complex(complex(item).real * real_scale, complex(item).imag * real_scale)
        for item in value_cpu.reshape(-1).tolist()
    ]
    return torch.tensor(values, dtype=owner_dtype).reshape(value_cpu.shape)


def foreach_complex_compound_reference(
    dispatcher_name: str,
    inputs: list[torch.Tensor],
    args: tuple,
    kwargs: dict,
) -> list[torch.Tensor]:
    """C99-style scalar semantics for exceptional foreach compound rows."""

    name = dispatcher_name.lower()

    def per_item(value, index):
        if isinstance(value, (list, tuple)):
            return value[index]
        if isinstance(value, torch.Tensor) and value.ndim == 1 and value.numel() == len(inputs):
            return value[index]
        return value

    results = []
    for index, input_tensor in enumerate(inputs):
        owner_dtype = input_tensor.dtype
        if "addcdiv" in name or "addcmul" in name:
            tensor1 = per_item(args[0], index)
            tensor2 = per_item(args[1], index)
            scale = per_item(args[2] if len(args) > 2 else kwargs.get("value", 1), index)
            combined = _python_complex_tensor_op(
                tensor1,
                tensor2,
                (lambda left, right: left / right) if "addcdiv" in name else (lambda left, right: left * right),
                owner_dtype=owner_dtype,
            )
            scaled = _python_complex_tensor_op(
                combined,
                scale,
                lambda left, right: left * right,
                owner_dtype=owner_dtype,
            )
            results.append(
                _python_complex_tensor_op(
                    input_tensor,
                    scaled,
                    lambda left, right: left + right,
                    owner_dtype=owner_dtype,
                )
            )
        elif "lerp" in name:
            endpoint = per_item(args[0], index)
            weight = per_item(args[1], index)
            delta = _python_complex_tensor_op(
                endpoint,
                input_tensor,
                lambda left, right: left - right,
                owner_dtype=owner_dtype,
            )
            scaled = _python_complex_tensor_op(
                delta,
                weight,
                lambda left, right: left * right,
                owner_dtype=owner_dtype,
            )
            results.append(
                _python_complex_tensor_op(
                    input_tensor,
                    scaled,
                    lambda left, right: left + right,
                    owner_dtype=owner_dtype,
                )
            )
        else:
            other = args[0] if ".tensor" in name else per_item(args[0], index)
            alpha = kwargs.get("alpha", 1)
            scaled = _python_complex_scalar_scale(other, alpha, owner_dtype=owner_dtype)
            operation = (lambda left, right: left - right) if "sub" in name else (lambda left, right: left + right)
            results.append(
                _python_complex_tensor_op(
                    input_tensor,
                    scaled,
                    operation,
                    owner_dtype=owner_dtype,
                )
            )
    return results


def complex_cumprod_reference(input_tensor: torch.Tensor, dim: int) -> torch.Tensor:
    input_cpu = input_tensor.detach().cpu()
    dim = dim % input_cpu.ndim
    rows = input_cpu.movedim(dim, -1).reshape(-1, input_cpu.shape[dim])
    output_rows = []
    for row in rows.tolist():
        if not row:
            output_rows.append([])
            continue
        accumulated = complex(row[0])
        result_row = [accumulated]
        for value in row[1:]:
            accumulated = accumulated * complex(value)
            result_row.append(accumulated)
        output_rows.append(result_row)
    result = torch.tensor(output_rows, dtype=input_cpu.dtype).reshape(input_cpu.movedim(dim, -1).shape)
    return result.movedim(-1, dim)


def complex_gradient_reference(
    input_tensor: torch.Tensor,
    *,
    spacing=1.0,
    dim=None,
    edge_order: int = 1,
):
    """Apply real spacing independently to real and imaginary gradient lanes."""

    input_cpu = input_tensor.detach().cpu()
    real = torch.gradient(input_cpu.real, spacing=spacing, dim=dim, edge_order=edge_order)
    imag = torch.gradient(input_cpu.imag, spacing=spacing, dim=dim, edge_order=edge_order)
    return tuple(torch.complex(real_item, imag_item).to(input_cpu.dtype) for real_item, imag_item in zip(real, imag))


def complex_covariance_reference(
    input_tensor: torch.Tensor,
    *,
    correction: int = 1,
    fweights: torch.Tensor | None = None,
    aweights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Complex covariance with real normalization and an explicitly real diagonal."""

    values = input_tensor.detach().cpu()
    if values.ndim == 1:
        values = values.unsqueeze(0)
    observations = values.shape[-1]
    weights = torch.ones(observations, dtype=values.real.dtype)
    if fweights is not None:
        weights = weights * fweights.detach().cpu().to(weights.dtype)
    if aweights is not None:
        weights = weights * aweights.detach().cpu().to(weights.dtype)
    weight_sum = weights.sum()
    average = (values * weights).sum(dim=-1, keepdim=True) / weight_sum
    centered = values - average
    if aweights is None:
        normalization = weight_sum - correction
    else:
        normalization = weight_sum - correction * (weights * aweights.detach().cpu().to(weights.dtype)).sum() / weight_sum
    result = (centered * weights) @ centered.mH
    result = torch.complex(result.real / normalization, result.imag / normalization)
    diagonal = torch.diagonal(result, dim1=-2, dim2=-1)
    result = result.clone()
    result.diagonal(dim1=-2, dim2=-1).copy_(
        torch.complex(diagonal.real, torch.zeros_like(diagonal.real))
    )
    return result.squeeze() if input_tensor.ndim == 1 else result


def _determinate_real_linear_sum(
    components: list[float],
    coefficients: list[float],
) -> tuple[float, bool]:
    finite_terms: list[float] = []
    has_nan = False
    has_positive_infinity = False
    has_negative_infinity = False
    for component, coefficient in zip(components, coefficients):
        if coefficient == 0.0:
            continue
        term = component * coefficient
        if math.isnan(term):
            has_nan = True
        elif term == math.inf:
            has_positive_infinity = True
        elif term == -math.inf:
            has_negative_infinity = True
        else:
            finite_terms.append(term)
    determinate = not has_nan and not (
        has_positive_infinity and has_negative_infinity
    )
    if not determinate:
        return math.nan, False
    if has_positive_infinity:
        return math.inf, True
    if has_negative_infinity:
        return -math.inf, True
    return math.fsum(finite_terms), True


def laguerre_polynomial_reference(input_tensor: torch.Tensor, degree) -> torch.Tensor:
    input_cpu, degree_cpu = torch.broadcast_tensors(
        input_tensor.detach().cpu(),
        degree.detach().cpu() if isinstance(degree, torch.Tensor) else torch.tensor(degree),
    )
    result = torch.empty_like(input_cpu)
    for index in range(result.numel()):
        x = input_cpu.reshape(-1)[index]
        order = int(degree_cpu.reshape(-1)[index].item())
        if order < 0:
            value = torch.zeros_like(x)
        elif order == 0:
            value = torch.ones_like(x)
        else:
            previous = torch.ones_like(x)
            current = 1 - x
            for n in range(2, order + 1):
                previous, current = current, ((2 * n - 1 - x) * current - (n - 1) * previous) / n
            value = current
        result.reshape(-1)[index] = value
    return result


def shifted_chebyshev_polynomial_reference(
    family: str,
    input_tensor: torch.Tensor,
    degree,
) -> torch.Tensor:
    input_cpu, degree_cpu = torch.broadcast_tensors(
        input_tensor.detach().cpu(),
        degree.detach().cpu() if isinstance(degree, torch.Tensor) else torch.tensor(degree),
    )
    family = family.lower()
    result = torch.empty_like(input_cpu)
    for index in range(result.numel()):
        x = input_cpu.reshape(-1)[index]
        order = int(degree_cpu.reshape(-1)[index].item())
        if order < 0:
            value = torch.zeros_like(x)
        elif order == 0:
            value = torch.ones_like(x)
        else:
            shifted = 2 * x - 1
            if family == "t":
                current = shifted
            elif family == "u":
                current = 2 * shifted
            elif family == "v":
                current = 2 * shifted - 1
            elif family == "w":
                current = 2 * shifted + 1
            else:
                raise ValueError(f"unknown shifted Chebyshev family {family!r}")
            previous = torch.ones_like(x)
            for _n in range(2, order + 1):
                previous, current = current, 2 * shifted * current - previous
            value = current
        result.reshape(-1)[index] = value
    return result


def lanczos3_coefficient(distance: float) -> float:
    """Radius-three Lanczos coefficient with exact integer sinc zeros."""

    if abs(distance) >= 3:
        return 0.0
    if distance == 0:
        return 1.0
    if float(distance).is_integer():
        return 0.0
    return (
        math.sin(math.pi * distance) / (math.pi * distance)
        * math.sin(math.pi * distance / 3) / (math.pi * distance / 3)
    )


def _lanczos_indices_weights(
    input_size: int,
    output_size: int,
    align_corners: bool,
    scale_factor: float | None,
) -> list[list[tuple[int, float]]]:
    if align_corners:
        scale = (input_size - 1) / (output_size - 1) if output_size > 1 else 0.0
    else:
        scale = 1.0 / scale_factor if scale_factor is not None and scale_factor > 0 else input_size / output_size
    support = 3.0 * scale if scale >= 1.0 else 3.0
    inverse_scale = 1.0 / scale if scale >= 1.0 else 1.0
    result: list[list[tuple[int, float]]] = []
    for output_index in range(output_size):
        center = scale * (output_index + 0.5)
        first = max(int(center - support + 0.5), 0)
        count = min(int(center + support + 0.5), input_size) - first
        weights = [
            lanczos3_coefficient((source - center + 0.5) * inverse_scale)
            for source in range(first, first + count)
        ]
        total = math.fsum(weights)
        if total != 0.0:
            weights = [weight / total for weight in weights]
        result.append(list(zip(range(first, first + count), weights)))
    return result


def lanczos2d_aa_reference(
    input_tensor: torch.Tensor,
    output_size,
    align_corners: bool,
    scales_h: float | None = None,
    scales_w: float | None = None,
) -> torch.Tensor:
    """Small separable Lanczos-3 oracle with mathematical sinc zeros.

    Zero-weight terms are omitted instead of evaluated as ``0 * value``.  This
    distinction is observable for nonfinite inputs and is the contract this
    oracle exists to preserve.
    """

    input_cpu = input_tensor.detach().cpu()
    if input_cpu.ndim != 4:
        raise ValueError(f"Lanczos2d reference requires NCHW rank 4, got {input_cpu.ndim}")
    output_height, output_width = (int(output_size[0]), int(output_size[1]))
    height_weights = _lanczos_indices_weights(
        input_cpu.shape[-2], output_height, bool(align_corners), scales_h
    )
    width_weights = _lanczos_indices_weights(
        input_cpu.shape[-1], output_width, bool(align_corners), scales_w
    )
    result = torch.empty(
        (*input_cpu.shape[:-2], output_height, output_width),
        dtype=input_cpu.dtype,
    )
    for batch in range(input_cpu.shape[0]):
        for channel in range(input_cpu.shape[1]):
            for output_y, y_terms in enumerate(height_weights):
                for output_x, x_terms in enumerate(width_weights):
                    terms = []
                    for input_y, y_weight in y_terms:
                        if y_weight == 0.0:
                            continue
                        for input_x, x_weight in x_terms:
                            weight = y_weight * x_weight
                            if weight == 0.0:
                                continue
                            terms.append(float(input_cpu[batch, channel, input_y, input_x]) * weight)
                    if any(math.isnan(term) for term in terms):
                        value = math.nan
                    else:
                        has_positive_infinity = any(term == math.inf for term in terms)
                        has_negative_infinity = any(term == -math.inf for term in terms)
                        if has_positive_infinity and has_negative_infinity:
                            value = math.nan
                        elif has_positive_infinity:
                            value = math.inf
                        elif has_negative_infinity:
                            value = -math.inf
                        else:
                            value = math.fsum(terms)
                    result[batch, channel, output_y, output_x] = value
    return result


def float_to_uint8_reference(input_tensor: torch.Tensor) -> torch.Tensor:
    """PyTorch's documented signed-truncate-then-narrow byte conversion."""

    input_cpu = input_tensor.detach().cpu()
    values = []
    for value in input_cpu.reshape(-1).tolist():
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("float-to-byte reference requires finite inputs")
            integer = math.trunc(value)
        else:
            integer = int(value)
        values.append(integer % 256)
    return torch.tensor(values, dtype=torch.uint8).reshape(input_cpu.shape)


def unsigned_negation_reference(input_tensor: torch.Tensor) -> torch.Tensor:
    if input_tensor.dtype not in {torch.uint8, torch.uint16, torch.uint32, torch.uint64}:
        raise ValueError(f"unsigned negation reference requires unsigned input, got {input_tensor.dtype}")
    modulus = 1 << torch.iinfo(input_tensor.dtype).bits
    values = [(-int(value)) % modulus for value in input_tensor.detach().cpu().reshape(-1).tolist()]
    return torch.tensor(values, dtype=input_tensor.dtype).reshape(input_tensor.shape)


def saturate_weight_to_fp16_reference(weight: torch.Tensor) -> torch.Tensor:
    result = weight.detach().cpu().clone(memory_format=torch.preserve_format)
    return torch.clamp(result, min=-65504.0, max=65504.0)


def weight_int8pack_mm_reference(
    input_tensor: torch.Tensor,
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    if input_tensor.ndim != 2 or packed_weight.ndim != 2:
        raise ValueError("int8pack matmul reference requires rank-2 input and weight")
    if input_tensor.shape[1] != packed_weight.shape[1]:
        raise ValueError("int8pack matmul inner dimensions must match")
    input_f32 = input_tensor.detach().cpu().float()
    weight_f32 = packed_weight.detach().cpu().float()
    scales_f32 = scales.detach().cpu().float()
    result = torch.zeros(
        input_f32.shape[0], packed_weight.shape[0], dtype=torch.float32
    )
    for index in range(input_f32.shape[1]):
        result = result + input_f32[:, index : index + 1] * weight_f32[:, index].unsqueeze(0)
    result = result * scales_f32.unsqueeze(0)
    return result.to(input_tensor.dtype)


def gcd_integer_reference(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_cpu, right_cpu = torch.broadcast_tensors(left.detach().cpu(), right.detach().cpu())
    info = torch.iinfo(left_cpu.dtype)
    values = []
    for left_value, right_value in zip(left_cpu.reshape(-1).tolist(), right_cpu.reshape(-1).tolist()):
        result = math.gcd(int(left_value), int(right_value))
        if result > info.max:
            result -= 1 << info.bits
        values.append(result)
    return torch.tensor(values, dtype=left_cpu.dtype).reshape(left_cpu.shape)


def histc_integer_count_reference(
    input_tensor: torch.Tensor,
    bins: int,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    values = input_tensor.detach().cpu().reshape(-1).tolist()
    minimum = float(minimum)
    maximum = float(maximum)
    if minimum == maximum:
        finite = [float(value) for value in values if math.isfinite(float(value))]
        if finite:
            minimum, maximum = min(finite), max(finite)
    counts = [0] * int(bins)
    if maximum > minimum:
        for value in values:
            value = float(value)
            if not math.isfinite(value) or value < minimum or value > maximum:
                continue
            if value == maximum:
                index = bins - 1
            else:
                index = int((value - minimum) / (maximum - minimum) * bins)
            if 0 <= index < bins:
                counts[index] += 1
    return torch.tensor(counts, dtype=torch.int64).to(input_tensor.dtype)


def im2col_reference(
    input_tensor: torch.Tensor,
    kernel_size,
    dilation,
    padding,
    stride,
) -> torch.Tensor:
    input_cpu = input_tensor.detach().cpu()
    kernel_h, kernel_w = map(int, kernel_size)
    dilation_h, dilation_w = map(int, dilation)
    padding_h, padding_w = map(int, padding)
    stride_h, stride_w = map(int, stride)
    batch, channels, height, width = input_cpu.shape
    output_h = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    output_w = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    result = torch.zeros(
        (batch, channels * kernel_h * kernel_w, output_h * output_w),
        dtype=input_cpu.dtype,
    )
    for n in range(batch):
        for c in range(channels):
            for ky in range(kernel_h):
                for kx in range(kernel_w):
                    row = (c * kernel_h + ky) * kernel_w + kx
                    for oy in range(output_h):
                        iy = oy * stride_h - padding_h + ky * dilation_h
                        for ox in range(output_w):
                            ix = ox * stride_w - padding_w + kx * dilation_w
                            if 0 <= iy < height and 0 <= ix < width:
                                result[n, row, oy * output_w + ox] = input_cpu[n, c, iy, ix]
    return result


def col2im_reference(
    columns: torch.Tensor,
    output_size,
    kernel_size,
    dilation,
    padding,
    stride,
) -> torch.Tensor:
    columns_cpu = columns.detach().cpu()
    output_h, output_w = map(int, output_size)
    kernel_h, kernel_w = map(int, kernel_size)
    dilation_h, dilation_w = map(int, dilation)
    padding_h, padding_w = map(int, padding)
    stride_h, stride_w = map(int, stride)
    batch = columns_cpu.shape[0]
    channels = columns_cpu.shape[1] // (kernel_h * kernel_w)
    blocks_h = (output_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    blocks_w = (output_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    result = torch.zeros((batch, channels, output_h, output_w), dtype=columns_cpu.dtype)
    for n in range(batch):
        for c in range(channels):
            for ky in range(kernel_h):
                for kx in range(kernel_w):
                    row = (c * kernel_h + ky) * kernel_w + kx
                    for oy in range(blocks_h):
                        iy = oy * stride_h - padding_h + ky * dilation_h
                        for ox in range(blocks_w):
                            ix = ox * stride_w - padding_w + kx * dilation_w
                            if 0 <= iy < output_h and 0 <= ix < output_w:
                                result[n, c, iy, ix] += columns_cpu[n, row, oy * blocks_w + ox]
    return result


def wide_ldexp_reference(input_tensor: torch.Tensor, exponents: torch.Tensor) -> torch.Tensor:
    input_cpu, exponent_cpu = torch.broadcast_tensors(
        input_tensor.detach().cpu(), exponents.detach().cpu()
    )
    values = []
    for value, exponent in zip(input_cpu.reshape(-1).tolist(), exponent_cpu.reshape(-1).tolist()):
        value = float(value)
        exponent = int(exponent)
        if value == 0 or not math.isfinite(value):
            values.append(math.ldexp(value, 0))
        elif exponent > 2**31 - 1:
            values.append(math.copysign(math.inf, value))
        elif exponent < -(2**31):
            values.append(math.copysign(0.0, value))
        else:
            try:
                values.append(math.ldexp(value, exponent))
            except OverflowError:
                values.append(math.copysign(math.inf, value))
    return torch.tensor(values, dtype=input_cpu.dtype).reshape(input_cpu.shape)


def logit_backward_reference(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    eps: float | None,
) -> torch.Tensor:
    grad_cpu, input_cpu = torch.broadcast_tensors(
        grad_output.detach().cpu(), input_tensor.detach().cpu()
    )
    if eps is not None:
        outside = (input_cpu < eps) | (input_cpu > 1 - eps)
        result = grad_cpu / (input_cpu * (1 - input_cpu))
        return torch.where(outside, torch.zeros_like(result), result)
    return grad_cpu / (input_cpu * (1 - input_cpu))


def grid_sampler_backward_f32_reference(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    grid: torch.Tensor,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
    output_mask=(True, True),
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = input_tensor.dtype
    if input_tensor.ndim == 4:
        op = torch.ops.aten.grid_sampler_2d_backward.default
    elif input_tensor.ndim == 5:
        op = torch.ops.aten.grid_sampler_3d_backward.default
    else:
        raise ValueError(f"grid sampler reference requires rank-4/5 input, got rank {input_tensor.ndim}")
    result = op(
        quantized_opmath_tensor(grad_output),
        quantized_opmath_tensor(input_tensor),
        quantized_opmath_tensor(grid),
        interpolation_mode,
        padding_mode,
        align_corners,
        list(output_mask),
    )
    return tuple(item.to(dtype) for item in result)


def grid_sampler_3d_backward_f32_reference(*args, **kwargs):
    """Compatibility alias for callers predating the shared 2-D/3-D oracle."""

    return grid_sampler_backward_f32_reference(*args, **kwargs)


def grid_sampler_forward_f32_reference(
    input_tensor: torch.Tensor,
    grid: torch.Tensor,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
) -> torch.Tensor:
    if input_tensor.ndim == 4:
        op = torch.ops.aten.grid_sampler_2d.default
    elif input_tensor.ndim == 5:
        op = torch.ops.aten.grid_sampler_3d.default
    else:
        raise ValueError(f"grid sampler reference requires rank-4/5 input, got rank {input_tensor.ndim}")
    result = op(
        quantized_opmath_tensor(input_tensor),
        quantized_opmath_tensor(grid),
        interpolation_mode,
        padding_mode,
        align_corners,
    )
    return cast_public_result(result, input_tensor.dtype)


def conv_transpose_f32_reference(
    op_name: str,
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    **kwargs,
) -> torch.Tensor:
    functions = {
        "nn.functional.conv_transpose1d": F.conv_transpose1d,
        "nn.functional.conv_transpose2d": F.conv_transpose2d,
        "nn.functional.conv_transpose3d": F.conv_transpose3d,
    }
    if op_name not in functions:
        raise ValueError(f"unsupported transpose-convolution operation {op_name!r}")
    bias_f32 = None if bias is None else quantized_opmath_tensor(bias)
    result = functions[op_name](
        quantized_opmath_tensor(input_tensor),
        quantized_opmath_tensor(weight),
        bias_f32,
        **kwargs,
    )
    return cast_public_result(result, input_tensor.dtype)


def conv_transpose3d_f32_reference(input_tensor, weight, bias, **kwargs):
    """Compatibility alias for the former single-rank reference."""

    return conv_transpose_f32_reference(
        "nn.functional.conv_transpose3d", input_tensor, weight, bias, **kwargs
    )


def matrix_exp_f64_reference(input_tensor: torch.Tensor) -> torch.Tensor:
    """Independent f64 scaling-and-squaring Taylor reference for small matrices."""

    if input_tensor.ndim < 2 or input_tensor.shape[-1] != input_tensor.shape[-2]:
        raise ValueError("matrix exponential reference requires square matrices")
    matrix = quantized_opmath_tensor(input_tensor, opmath_dtype=torch.float64)
    norm = torch.linalg.matrix_norm(matrix, ord=1, dim=(-2, -1))
    max_norm = float(norm.max().item()) if norm.numel() else 0.0
    squarings = max(0, int(math.ceil(math.log2(max_norm / 0.5)))) if max_norm > 0.5 else 0
    scaled = matrix / (2 ** squarings)
    identity = torch.eye(scaled.shape[-1], dtype=scaled.dtype).expand_as(scaled)
    result = identity.clone()
    term = identity.clone()
    for order in range(1, 81):
        term = torch.matmul(term, scaled) / order
        result = result + term
        term_norm = float(torch.linalg.matrix_norm(term, ord=1, dim=(-2, -1)).max().item())
        result_norm = float(torch.linalg.matrix_norm(result, ord=1, dim=(-2, -1)).max().item())
        if term_norm <= torch.finfo(torch.float64).eps * max(1.0, result_norm):
            break
    else:
        raise RuntimeError("matrix exponential Taylor reference did not converge")
    for _ in range(squarings):
        result = torch.matmul(result, result)
    return cast_public_result(result, input_tensor.dtype)


def embedding_bag_scale_grad_by_freq_reference(
    grad: torch.Tensor,
    indices: torch.Tensor,
    offset2bag: torch.Tensor,
    num_weights: int,
    *,
    padding_idx: int = -1,
) -> torch.Tensor:
    """Dense sum-mode EmbeddingBag backward with each row's global frequency."""

    grad_cpu = grad.detach().cpu()
    indices_cpu = indices.detach().cpu().to(torch.long).reshape(-1)
    bags_cpu = offset2bag.detach().cpu().to(torch.long).reshape(-1)
    if indices_cpu.numel() != bags_cpu.numel():
        raise ValueError("indices and offset2bag must have the same number of elements")
    if grad_cpu.dim() != 2:
        raise ValueError(f"embedding-bag grad must be 2-D, got {tuple(grad_cpu.shape)}")
    result = torch.zeros((num_weights, grad_cpu.shape[1]), dtype=grad_cpu.dtype)
    counts = torch.bincount(indices_cpu[indices_cpu != padding_idx], minlength=num_weights)
    for position, row_tensor in enumerate(indices_cpu):
        row = int(row_tensor.item())
        if row == padding_idx:
            continue
        if row < 0 or row >= num_weights:
            raise ValueError(f"embedding index {row} is outside [0, {num_weights})")
        frequency = int(counts[row].item())
        bag = int(bags_cpu[position].item())
        result[row] += grad_cpu[bag] / frequency
    return result


_COMPLEX_CONV_FUNCTIONS = {
    "nn.functional.conv1d": (F.conv1d, False),
    "nn.functional.conv2d": (F.conv2d, False),
    "nn.functional.conv3d": (F.conv3d, False),
    "nn.functional.conv_transpose1d": (F.conv_transpose1d, True),
    "nn.functional.conv_transpose2d": (F.conv_transpose2d, True),
    "nn.functional.conv_transpose3d": (F.conv_transpose3d, True),
}


def complex_convolution_reference(
    op_name: str,
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    kwargs: dict,
) -> torch.Tensor:
    """Fast direct four-real-convolution reference for complex convolution."""

    if op_name not in _COMPLEX_CONV_FUNCTIONS:
        raise ValueError(f"unsupported complex convolution operation {op_name!r}")
    if not input_tensor.is_complex() or not weight.is_complex():
        raise ValueError("complex convolution reference requires complex input and weight")
    fn, _transposed = _COMPLEX_CONV_FUNCTIONS[op_name]
    x = input_tensor.detach().cpu()
    w = weight.detach().cpu()
    call_kwargs = dict(kwargs)
    rr = fn(x.real, w.real, None, **call_kwargs)
    ii = fn(x.imag, w.imag, None, **call_kwargs)
    ri = fn(x.real, w.imag, None, **call_kwargs)
    ir = fn(x.imag, w.real, None, **call_kwargs)
    real = rr - ii
    imag = ri + ir
    if bias is not None:
        bias_cpu = bias.detach().cpu()
        unbatched = x.dim() + 1 == w.dim()
        shape = (bias_cpu.numel(),) + (1,) * (real.dim() - 1) if unbatched else (
            1,
            bias_cpu.numel(),
            *((1,) * (real.dim() - 2)),
        )
        real = real + bias_cpu.real.reshape(shape)
        imag = imag + bias_cpu.imag.reshape(shape)
    return torch.complex(real, imag).to(x.dtype)


def _expand_conv_arg(value, dims: int, default: int) -> tuple[int, ...]:
    if value is None:
        return (default,) * dims
    if isinstance(value, int):
        return (value,) * dims
    values = tuple(int(item) for item in value)
    if len(values) == 1:
        return values * dims
    if len(values) != dims:
        raise ValueError(f"expected {dims} convolution values, got {values}")
    return values


def slow_complex_convolution_reference(
    op_name: str,
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    kwargs: dict,
) -> torch.Tensor:
    """Small-shape term-wise proof oracle for ``complex_convolution_reference``."""

    if op_name not in _COMPLEX_CONV_FUNCTIONS:
        raise ValueError(f"unsupported complex convolution operation {op_name!r}")
    _fn, transposed = _COMPLEX_CONV_FUNCTIONS[op_name]
    x = input_tensor.detach().cpu()
    w = weight.detach().cpu()
    unbatched = x.dim() + 1 == w.dim()
    if unbatched:
        x = x.unsqueeze(0)
    dims = w.dim() - 2
    stride = _expand_conv_arg(kwargs.get("stride"), dims, 1)
    dilation = _expand_conv_arg(kwargs.get("dilation"), dims, 1)
    output_padding = _expand_conv_arg(kwargs.get("output_padding"), dims, 0)
    groups = int(kwargs.get("groups", 1))
    padding_arg = kwargs.get("padding", 0)
    same_padding = False
    if isinstance(padding_arg, str):
        if padding_arg == "valid":
            padding = (0,) * dims
        elif padding_arg == "same":
            if transposed:
                raise ValueError("transposed convolution does not accept padding='same'")
            same_padding = True
            padding = tuple(
                dilation[i] * (w.shape[i + 2] - 1) // 2
                for i in range(dims)
            )
        else:
            raise ValueError(f"unsupported padding string {padding_arg!r}")
    else:
        padding = _expand_conv_arg(padding_arg, dims, 0)

    in_spatial = tuple(x.shape[2:])
    kernel = tuple(w.shape[2:])
    if transposed:
        out_spatial = tuple(
            (in_spatial[i] - 1) * stride[i] - 2 * padding[i]
            + dilation[i] * (kernel[i] - 1) + output_padding[i] + 1
            for i in range(dims)
        )
        out_channels = w.shape[1] * groups
    else:
        if same_padding:
            out_spatial = in_spatial
        else:
            out_spatial = tuple(
                (in_spatial[i] + 2 * padding[i] - dilation[i] * (kernel[i] - 1) - 1)
                // stride[i] + 1
                for i in range(dims)
            )
        out_channels = w.shape[0]
    result = torch.empty((x.shape[0], out_channels, *out_spatial), dtype=x.dtype)
    in_channels_per_group = x.shape[1] // groups
    out_channels_per_group = out_channels // groups
    for batch in range(x.shape[0]):
        for out_channel in range(out_channels):
            group = out_channel // out_channels_per_group
            out_channel_in_group = out_channel % out_channels_per_group
            for out_coord in itertools.product(*(range(size) for size in out_spatial)):
                accumulator = complex(bias[out_channel].item()) if bias is not None else 0j
                for in_channel_in_group in range(in_channels_per_group):
                    in_channel = group * in_channels_per_group + in_channel_in_group
                    for kernel_coord in itertools.product(*(range(size) for size in kernel)):
                        input_coord = []
                        valid = True
                        for axis in range(dims):
                            if transposed:
                                coord = out_coord[axis] + padding[axis] - kernel_coord[axis] * dilation[axis]
                                if coord < 0 or coord % stride[axis]:
                                    valid = False
                                    break
                                coord //= stride[axis]
                            else:
                                coord = out_coord[axis] * stride[axis] - padding[axis] + kernel_coord[axis] * dilation[axis]
                            if coord < 0 or coord >= in_spatial[axis]:
                                valid = False
                            input_coord.append(coord)
                        if transposed and not valid:
                            continue
                        input_value = x[(batch, in_channel, *input_coord)] if valid else torch.zeros((), dtype=x.dtype)
                        weight_value = w[
                            (in_channel, out_channel_in_group, *kernel_coord)
                            if transposed
                            else (out_channel, in_channel_in_group, *kernel_coord)
                        ]
                        accumulator += complex(torch.mul(input_value, weight_value).item())
                        if x.dtype == torch.complex64:
                            accumulator = complex(torch.tensor(accumulator, dtype=torch.complex64).item())
                result[(batch, out_channel, *out_coord)] = accumulator
    return result.squeeze(0) if unbatched else result
