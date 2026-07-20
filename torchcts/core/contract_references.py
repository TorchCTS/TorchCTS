# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies or substantial portions of the Software.
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Closed routing for permanent TorchCTS-owned semantic references."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number

import torch

from torchcts.core.high_precision_reference import (
    dirichlet_grad_reference,
    i0_reference,
    polygamma_reference,
    standard_gamma_grad_reference,
)
from torchcts.core.reference_oracles import (
    binary_cross_entropy_with_logits_reference,
    complex_convolution_reference,
    complex_covariance_reference,
    complex_cumprod_reference,
    complex_expm1_reference,
    foreach_complex_compound_reference,
    complex_gradient_reference,
    complex_l1_loss_reference,
    complex_integral_ldexp_reference,
    complex_ldexp_reference,
    complex_log2_reference,
    complex_mul_reference,
    complex_rsqrt_reference,
    complex_sigmoid_reference,
    complex_tensor_integer_power_reference,
    complex_unit_alpha_add_sub_reference,
    conv_transpose_f32_reference,
    grid_sampler_forward_f32_reference,
    float_to_uint8_reference,
    has_nonnegative_integral_complex_exponent,
    laguerre_polynomial_reference,
    lanczos2d_aa_reference,
    matmul_family_reference,
    matrix_exp_f64_reference,
    soft_margin_loss_reference,
    shifted_chebyshev_polynomial_reference,
)


@dataclass(frozen=True)
class ContractReference:
    value: object
    reference_id: str
    category: str


class ContractReferenceError(RuntimeError):
    pass


_SPECIAL_CONDITIONS = frozenset({"has_nan", "has_inf"})
_COMPLEX32 = getattr(torch, "complex32", None)
_STANDARD_COMPLEX_DTYPES = frozenset({torch.complex64, torch.complex128})
_UNIT_ALPHA_DTYPES = frozenset(
    dtype for dtype in (_COMPLEX32, torch.complex64, torch.complex128) if dtype is not None
)
_ALL_COMPLEX_DTYPES = _UNIT_ALPHA_DTYPES
_INTEGER_POWER_DTYPES = _UNIT_ALPHA_DTYPES

_SHIFTED_CHEBYSHEV_OPS = {
    "special.shifted_chebyshev_polynomial_t": "t",
    "special.shifted_chebyshev_polynomial_u": "u",
    "special.shifted_chebyshev_polynomial_v": "v",
    "special.shifted_chebyshev_polynomial_w": "w",
}

_OPINFO_ADD_SUB = {
    "add": "add",
    "sub": "sub",
    "rsub": "rsub",
    "__rsub__": "rsub",
}

_OPINFO_POWER = {
    "pow": "aten::pow.Tensor_Tensor",
    "float_power": "aten::float_power.Tensor_Tensor",
}

_OPINFO_MATMUL = {
    "matmul": "aten::matmul",
    "mm": "aten::mm",
    "bmm": "aten::bmm",
    "addmm": "aten::addmm",
    "addbmm": "aten::addbmm",
    "baddbmm": "aten::baddbmm",
    "nn.functional.linear": "aten::linear",
}

_GENERATED_ADD_SUB = {
    "aten::add.Tensor": "add",
    "aten::subtract.Tensor": "sub",
    "aten::add_.Tensor": "add",
    "aten::add.out": "add",
    "aten::sub.Tensor": "sub",
    "aten::sub_.Tensor": "sub",
    "aten::subtract_.Tensor": "sub",
    "aten::sub.out": "sub",
    "aten::subtract.out": "sub",
    "aten::rsub.Tensor": "rsub",
    "aten::rsub.Tensor_out": "rsub",
}

_GENERATED_POWER = frozenset({
    "aten::float_power.Tensor_Tensor",
    "aten::float_power.Tensor_Tensor_out",
    "aten::float_power_.Tensor",
    "aten::pow.Tensor_Tensor",
    "aten::pow_.Tensor",
    "aten::pow.Tensor_Tensor_out",
})

_GENERATED_LOG2 = frozenset({
    "aten::log2_",
    "aten::log2.out",
    "aten::_foreach_log2",
    "aten::_foreach_log2_",
    "aten::_foreach_log2.out",
})

_GENERATED_I0 = frozenset({
    "aten::i0",
    "aten::i0.out",
    "aten::special_i0",
    "aten::special_i0.out",
    "aten::special_modified_bessel_i0",
    "aten::special_modified_bessel_i0.out",
})

_OPINFO_COMPLEX_UNARY = {
    "sigmoid": ("complex_sigmoid_stable", complex_sigmoid_reference),
    "rsqrt": ("complex_rsqrt_c99", complex_rsqrt_reference),
    "expm1": ("complex_expm1_lane_aware", complex_expm1_reference),
}

_GENERATED_COMPLEX_UNARY = {
    "aten::sigmoid.out": ("complex_sigmoid_stable", complex_sigmoid_reference),
    "aten::sigmoid_": ("complex_sigmoid_stable", complex_sigmoid_reference),
    "aten::_foreach_sigmoid": ("complex_sigmoid_stable", complex_sigmoid_reference),
    "aten::_foreach_sigmoid_": ("complex_sigmoid_stable", complex_sigmoid_reference),
    "aten::_foreach_sigmoid.out": ("complex_sigmoid_stable", complex_sigmoid_reference),
    "aten::rsqrt.out": ("complex_rsqrt_c99", complex_rsqrt_reference),
    "aten::rsqrt_": ("complex_rsqrt_c99", complex_rsqrt_reference),
    "aten::_foreach_rsqrt": ("complex_rsqrt_c99", complex_rsqrt_reference),
    "aten::_foreach_rsqrt_": ("complex_rsqrt_c99", complex_rsqrt_reference),
    "aten::_foreach_rsqrt.out": ("complex_rsqrt_c99", complex_rsqrt_reference),
    "aten::expm1.out": ("complex_expm1_lane_aware", complex_expm1_reference),
    "aten::expm1_": ("complex_expm1_lane_aware", complex_expm1_reference),
    "aten::special_expm1": ("complex_expm1_lane_aware", complex_expm1_reference),
    "aten::special_expm1.out": ("complex_expm1_lane_aware", complex_expm1_reference),
    "aten::_foreach_expm1": ("complex_expm1_lane_aware", complex_expm1_reference),
    "aten::_foreach_expm1_": ("complex_expm1_lane_aware", complex_expm1_reference),
    "aten::_foreach_expm1.out": ("complex_expm1_lane_aware", complex_expm1_reference),
}

_FOREACH_COMPLEX_ADD_SUB_SCHEMAS = frozenset({
    "aten::_foreach_add.List",
    "aten::_foreach_add.List_out",
    "aten::_foreach_add.ScalarList",
    "aten::_foreach_add.ScalarList_out",
    "aten::_foreach_add.Scalar_out",
    "aten::_foreach_add.Tensor",
    "aten::_foreach_add.Tensor_out",
    "aten::_foreach_add_.List",
    "aten::_foreach_add_.Scalar",
    "aten::_foreach_add_.ScalarList",
    "aten::_foreach_add_.Tensor",
    "aten::_foreach_sub.List",
    "aten::_foreach_sub.List_out",
    "aten::_foreach_sub.Scalar",
    "aten::_foreach_sub.ScalarList",
    "aten::_foreach_sub.ScalarList_out",
    "aten::_foreach_sub.Scalar_out",
    "aten::_foreach_sub_.List",
    "aten::_foreach_sub_.Scalar",
    "aten::_foreach_sub_.ScalarList",
})

_FOREACH_COMPLEX_TENSOR_MUL_SCHEMAS = frozenset({
    "aten::_foreach_mul.Tensor",
    "aten::_foreach_mul.Tensor_out",
    "aten::_foreach_mul_.Tensor",
})

_FOREACH_COMPLEX_EXISTING_COMPOUND_SCHEMAS = frozenset({
    "aten::_foreach_addcdiv.Scalar",
    "aten::_foreach_addcdiv.ScalarList",
    "aten::_foreach_addcdiv.ScalarList_out",
    "aten::_foreach_addcdiv.Scalar_out",
    "aten::_foreach_addcdiv.Tensor",
    "aten::_foreach_addcdiv.Tensor_out",
    "aten::_foreach_addcdiv_.Scalar",
    "aten::_foreach_addcdiv_.ScalarList",
    "aten::_foreach_addcdiv_.Tensor",
    "aten::_foreach_addcmul.Scalar",
    "aten::_foreach_addcmul.ScalarList",
    "aten::_foreach_addcmul.ScalarList_out",
    "aten::_foreach_addcmul.Scalar_out",
    "aten::_foreach_addcmul.Tensor",
    "aten::_foreach_addcmul.Tensor_out",
    "aten::_foreach_addcmul_.Scalar",
    "aten::_foreach_addcmul_.ScalarList",
    "aten::_foreach_addcmul_.Tensor",
    "aten::_foreach_lerp.List",
    "aten::_foreach_lerp.List_out",
    "aten::_foreach_lerp.ScalarList",
    "aten::_foreach_lerp.ScalarList_out",
    "aten::_foreach_lerp.Scalar_out",
    "aten::_foreach_lerp_.List",
    "aten::_foreach_lerp_.Scalar",
    "aten::_foreach_lerp_.ScalarList",
})

_COMPLEX_CONV_OPS = frozenset({
    "nn.functional.conv1d",
    "nn.functional.conv2d",
    "nn.functional.conv3d",
    "nn.functional.conv_transpose1d",
    "nn.functional.conv_transpose2d",
    "nn.functional.conv_transpose3d",
})
_TRANSPOSED_COMPLEX_CONV_OPS = frozenset({
    "nn.functional.conv_transpose1d",
    "nn.functional.conv_transpose2d",
    "nn.functional.conv_transpose3d",
})
_ORDINARY_CONV_PARAMETERS = (
    ("weight", None),
    ("bias", None),
    ("stride", 1),
    ("padding", 0),
    ("dilation", 1),
    ("groups", 1),
)
_TRANSPOSED_CONV_PARAMETERS = (
    ("weight", None),
    ("bias", None),
    ("stride", 1),
    ("padding", 0),
    ("output_padding", 0),
    ("groups", 1),
    ("dilation", 1),
)


def _condition(value) -> str:
    return str(value).lower()


def _is_real_unit_alpha(value) -> bool:
    if isinstance(value, torch.Tensor):
        return False
    if not isinstance(value, Number):
        return False
    candidate = complex(value)
    return candidate.real == 1.0 and candidate.imag == 0.0


def _build(reference_id: str, category: str, builder) -> ContractReference:
    try:
        value = builder()
    except ContractReferenceError:
        raise
    except Exception as exc:
        raise ContractReferenceError(
            f"Permanent contract reference {reference_id} failed: {type(exc).__name__}: {exc}"
        ) from exc
    return ContractReference(value=value, reference_id=reference_id, category=category)


def _unit_alpha_reference(operation, sample, dtype):
    if not sample.args or not isinstance(sample.input, torch.Tensor) or not isinstance(sample.args[0], torch.Tensor):
        return None
    if not _is_real_unit_alpha(sample.kwargs.get("alpha", 1)):
        return None
    if not sample.input.is_complex() or not sample.args[0].is_complex():
        return None
    if sample.input.dtype != dtype or sample.args[0].dtype != dtype:
        return None
    return _build(
        "complex_unit_alpha_add_sub",
        "elementwise",
        lambda: complex_unit_alpha_add_sub_reference(operation, sample.input, sample.args[0]),
    )


def _power_reference(dispatcher_name, sample, dtype):
    if not sample.args or not isinstance(sample.input, torch.Tensor) or not isinstance(sample.args[0], torch.Tensor):
        return None
    exponent = sample.args[0]
    if not sample.input.is_complex() or not exponent.is_complex():
        return None
    if sample.input.dtype != dtype or exponent.dtype != dtype:
        return None
    if dispatcher_name == "aten::float_power_.Tensor" and dtype != torch.complex128:
        return None
    if not has_nonnegative_integral_complex_exponent(exponent):
        return None

    def build():
        if "float_power" in dispatcher_name:
            native = torch.float_power(sample.input.detach().cpu(), exponent.detach().cpu())
        elif dtype == _COMPLEX32:
            # CPU tensor power is not implemented for complex32.  The wider
            # calculation supplies only the non-integral lanes; exact integer
            # lanes are replaced below by exponentiation by squaring.
            native = torch.pow(
                sample.input.detach().cpu().to(torch.complex64),
                exponent.detach().cpu().to(torch.complex64),
            ).to(dtype)
        else:
            native = torch.pow(sample.input.detach().cpu(), exponent.detach().cpu())
        return complex_tensor_integer_power_reference(sample.input, exponent, native)

    return _build("complex_tensor_integer_power", "elementwise", build)


def _log2_value(value):
    if isinstance(value, torch.Tensor):
        return complex_log2_reference(value)
    if isinstance(value, list):
        return [_log2_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_log2_value(item) for item in value)
    raise ValueError(f"complex log2 reference received {type(value).__name__}")


def _complex_unary_value(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, list):
        return [_complex_unary_value(item, fn) for item in value]
    if isinstance(value, tuple):
        return tuple(_complex_unary_value(item, fn) for item in value)
    raise ValueError(f"complex unary reference received {type(value).__name__}")


def _complex_gradient_from_sample(sample):
    input_cpu = sample.input.detach().cpu()
    args = tuple(
        value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for value in sample.args
    )
    kwargs = {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in sample.kwargs.items()
    }
    real = torch.gradient(input_cpu.real, *args, **kwargs)
    imag = torch.gradient(input_cpu.imag, *args, **kwargs)
    return tuple(
        torch.complex(real_item, imag_item).to(input_cpu.dtype)
        for real_item, imag_item in zip(real, imag)
    )


def _is_complex_tensor_tree(value, dtype) -> bool:
    if isinstance(value, torch.Tensor):
        return value.is_complex() and value.dtype == dtype
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_is_complex_tensor_tree(item, dtype) for item in value)
    return False


def _l1_operand_dtypes_match(input_tensor, target, dtype) -> bool:
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    allowed = {dtype, real_dtype}
    return (
        input_tensor.dtype in allowed
        and target.dtype in allowed
        and (input_tensor.dtype == dtype or target.dtype == dtype)
    )


def _normalize_conv_arguments(op_name, sample):
    if not isinstance(sample.input, torch.Tensor):
        return None
    parameters = (
        _TRANSPOSED_CONV_PARAMETERS
        if op_name in _TRANSPOSED_COMPLEX_CONV_OPS
        else _ORDINARY_CONV_PARAMETERS
    )
    if len(sample.args) > len(parameters):
        return None

    parameter_names = {name for name, _default in parameters}
    if not set(sample.kwargs).issubset(parameter_names):
        return None

    values = {}
    for index, value in enumerate(sample.args):
        name = parameters[index][0]
        if name in sample.kwargs:
            return None
        values[name] = value
    for name, default in parameters[len(sample.args):]:
        values[name] = sample.kwargs.get(name, default)

    weight = values["weight"]
    bias = values["bias"]
    if not isinstance(weight, torch.Tensor):
        return None
    if bias is not None and not isinstance(bias, torch.Tensor):
        return None
    kwargs = {
        name: values[name]
        for name, _default in parameters
        if name not in {"weight", "bias"}
    }
    return weight, bias, kwargs


def _normalize_bce_with_logits_arguments(sample):
    if not isinstance(sample.input, torch.Tensor) or not sample.args:
        return None
    target = sample.args[0]
    if not isinstance(target, torch.Tensor):
        return None
    weight = sample.kwargs.get("weight")
    pos_weight = sample.kwargs.get("pos_weight")
    reduction = sample.kwargs.get("reduction", "mean")
    if len(sample.args) > 1:
        weight = sample.args[1]
    if len(sample.args) > 2:
        pos_weight = sample.args[2]
    if len(sample.args) > 3:
        reduction = sample.args[3]
    if weight is not None and not isinstance(weight, torch.Tensor):
        return None
    if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
        return None
    return target, weight, pos_weight, reduction


def _normalize_grid_arguments(sample):
    if not isinstance(sample.input, torch.Tensor) or not sample.args:
        return None
    grid = sample.args[0]
    if not isinstance(grid, torch.Tensor):
        return None
    interpolation_mode = sample.kwargs.get("interpolation_mode", sample.kwargs.get("mode", 0))
    padding_mode = sample.kwargs.get("padding_mode", 0)
    align_corners = sample.kwargs.get("align_corners", False)
    if len(sample.args) > 1:
        interpolation_mode = sample.args[1]
    if len(sample.args) > 2:
        padding_mode = sample.args[2]
    if len(sample.args) > 3:
        align_corners = sample.args[3]
    mode_names = {"bilinear": 0, "nearest": 1, "bicubic": 2}
    padding_names = {"zeros": 0, "border": 1, "reflection": 2}
    interpolation_mode = mode_names.get(interpolation_mode, interpolation_mode)
    padding_mode = padding_names.get(padding_mode, padding_mode)
    return grid, int(interpolation_mode), int(padding_mode), bool(align_corners)


def _normalize_soft_margin_arguments(sample):
    if not isinstance(sample.input, torch.Tensor) or not sample.args:
        return None
    target = sample.args[0]
    if not isinstance(target, torch.Tensor):
        return None
    reduction = sample.kwargs.get("reduction", "mean")
    if len(sample.args) > 1:
        reduction = sample.args[1]
    return target, reduction


def resolve_opinfo_forward_reference(
    op_name,
    sample,
    dtype,
    input_condition,
) -> ContractReference | None:
    condition = _condition(input_condition)

    if op_name == "special.laguerre_polynomial_l" and condition in _SPECIAL_CONDITIONS:
        if isinstance(sample.input, torch.Tensor) and sample.args:
            return _build(
                "laguerre_initialized_recurrence",
                "elementwise",
                lambda: laguerre_polynomial_reference(sample.input, sample.args[0]),
            )

    if op_name in _SHIFTED_CHEBYSHEV_OPS and condition in _SPECIAL_CONDITIONS:
        if isinstance(sample.input, torch.Tensor) and sample.args:
            return _build(
                "shifted_chebyshev_initialized_recurrence",
                "elementwise",
                lambda: shifted_chebyshev_polynomial_reference(
                    _SHIFTED_CHEBYSHEV_OPS[op_name], sample.input, sample.args[0]
                ),
            )

    if op_name in {"i0", "special.i0"} and isinstance(sample.input, torch.Tensor):
        return _build("i0_mpmath", "elementwise", lambda: i0_reference(sample.input))

    if (
        op_name in {"polygamma", "special.polygamma"}
        and isinstance(sample.input, torch.Tensor)
        and sample.args
        and int(sample.args[0]) == 1
    ):
        return _build(
            "polygamma_mpmath_poles",
            "elementwise",
            lambda: polygamma_reference(int(sample.args[0]), sample.input),
        )

    if (
        op_name in _OPINFO_MATMUL
        and dtype in _STANDARD_COMPLEX_DTYPES
        and condition == "clean"
    ):
        if isinstance(sample.input, torch.Tensor):
            return _build(
                "complex_matmul_wide_semantic",
                "matmul",
                lambda: matmul_family_reference(
                    _OPINFO_MATMUL[op_name],
                    (sample.input, *sample.args),
                    sample.kwargs,
                ),
            )

    if op_name in _OPINFO_COMPLEX_UNARY and dtype in _STANDARD_COMPLEX_DTYPES and condition in _SPECIAL_CONDITIONS:
        if _is_complex_tensor_tree(sample.input, dtype):
            reference_id, fn = _OPINFO_COMPLEX_UNARY[op_name]
            return _build(reference_id, "elementwise", lambda: _complex_unary_value(sample.input, fn))

    if op_name == "ldexp" and dtype in _STANDARD_COMPLEX_DTYPES and condition in _SPECIAL_CONDITIONS:
        if sample.args and isinstance(sample.input, torch.Tensor) and isinstance(sample.args[0], torch.Tensor):
            if sample.input.is_complex() and sample.args[0].is_complex():
                return _build(
                    "complex_ldexp_phase_scale",
                    "elementwise",
                    lambda: complex_ldexp_reference(sample.input, sample.args[0]),
                )

    if op_name == "cumprod" and dtype in _STANDARD_COMPLEX_DTYPES and condition in _SPECIAL_CONDITIONS:
        dim = sample.args[0] if sample.args else sample.kwargs.get("dim", 0)
        return _build(
            "complex_cumprod_inclusive",
            "reduction",
            lambda: complex_cumprod_reference(sample.input, int(dim)),
        )

    if op_name == "gradient" and dtype in _STANDARD_COMPLEX_DTYPES and condition in _SPECIAL_CONDITIONS:
        return _build(
            "complex_gradient_real_spacing",
            "elementwise",
            lambda: _complex_gradient_from_sample(sample),
        )

    if op_name == "cov" and dtype in _STANDARD_COMPLEX_DTYPES and condition in _SPECIAL_CONDITIONS:
        return _build(
            "complex_covariance_real_diagonal",
            "reduction",
            lambda: complex_covariance_reference(
                sample.input,
                correction=int(sample.kwargs.get("correction", 1)),
                fweights=sample.kwargs.get("fweights"),
                aweights=sample.kwargs.get("aweights"),
            ),
        )

    if condition == "clean" and dtype in {torch.float16, torch.bfloat16} and op_name in {
        "grid_sampler_2d",
        "grid_sampler_3d",
        "nn.functional.grid_sample",
    }:
        normalized = _normalize_grid_arguments(sample)
        if normalized is not None:
            grid, interpolation_mode, padding_mode, align_corners = normalized
            return _build(
                "grid_sampler_f32",
                "grid_sample",
                lambda: grid_sampler_forward_f32_reference(
                    sample.input,
                    grid,
                    interpolation_mode,
                    padding_mode,
                    align_corners,
                ),
            )

    if condition == "clean" and dtype in {torch.float16, torch.bfloat16} and op_name == "addbmm":
        return _build(
            "addbmm_f32_opmath",
            "matmul",
            lambda: matmul_family_reference(
                "aten::addbmm",
                (sample.input, *sample.args),
                sample.kwargs,
            ),
        )

    if condition == "clean" and dtype in {torch.float16, torch.bfloat16} and op_name == "nn.functional.soft_margin_loss":
        normalized = _normalize_soft_margin_arguments(sample)
        if normalized is not None:
            target, reduction = normalized
            return _build(
                "soft_margin_f32_opmath",
                "loss",
                lambda: soft_margin_loss_reference(sample.input, target, reduction),
            )

    if condition == "clean" and dtype in {torch.float16, torch.bfloat16, torch.float32} and op_name in {
        "linalg.matrix_exp",
        "matrix_exp",
    }:
        if isinstance(sample.input, torch.Tensor) and not sample.args and not sample.kwargs:
            return _build(
                "matrix_exp_f64_scaling_squaring",
                "linalg",
                lambda: matrix_exp_f64_reference(sample.input),
            )

    if (
        op_name == "nn.functional.binary_cross_entropy_with_logits"
        and condition in _SPECIAL_CONDITIONS
    ):
        normalized = _normalize_bce_with_logits_arguments(sample)
        if normalized is None:
            return None
        target, weight, pos_weight, reduction = normalized
        return _build(
            "bce_with_logits_stable",
            "loss",
            lambda: binary_cross_entropy_with_logits_reference(
                sample.input,
                target,
                weight,
                pos_weight,
                reduction,
            ),
        )

    if op_name in _OPINFO_ADD_SUB and dtype in _UNIT_ALPHA_DTYPES and condition in _SPECIAL_CONDITIONS:
        return _unit_alpha_reference(_OPINFO_ADD_SUB[op_name], sample, dtype)

    if op_name in _OPINFO_POWER and dtype in _INTEGER_POWER_DTYPES and condition in _SPECIAL_CONDITIONS:
        return _power_reference(_OPINFO_POWER[op_name], sample, dtype)

    if op_name == "nn.functional.l1_loss" and dtype in _STANDARD_COMPLEX_DTYPES and condition in _SPECIAL_CONDITIONS:
        if (
            not isinstance(sample.input, torch.Tensor)
            or not sample.args
            or not isinstance(sample.args[0], torch.Tensor)
            or not (sample.input.is_complex() or sample.args[0].is_complex())
            or not _l1_operand_dtypes_match(sample.input, sample.args[0], dtype)
        ):
            return None
        reduction = sample.kwargs.get("reduction", "mean")
        return _build(
            "complex_l1_loss",
            "loss",
            lambda: complex_l1_loss_reference(sample.input, sample.args[0], reduction),
        )

    if op_name == "log2" and dtype in _STANDARD_COMPLEX_DTYPES and condition in _SPECIAL_CONDITIONS:
        if not _is_complex_tensor_tree(sample.input, dtype):
            return None
        return _build(
            "complex_log2",
            "elementwise",
            lambda: complex_log2_reference(sample.input),
        )

    if op_name in _TRANSPOSED_COMPLEX_CONV_OPS and dtype in {torch.float16, torch.bfloat16} and condition == "clean":
        normalized = _normalize_conv_arguments(op_name, sample)
        if normalized is None:
            return None
        weight, bias, conv_kwargs = normalized
        if sample.input.dtype != dtype or weight.dtype != dtype:
            return None
        if bias is not None and bias.dtype != dtype:
            return None
        return _build(
            "conv_transpose_f32_opmath",
            "conv",
            lambda: conv_transpose_f32_reference(op_name, sample.input, weight, bias, **conv_kwargs),
        )

    if op_name in _COMPLEX_CONV_OPS and dtype in _STANDARD_COMPLEX_DTYPES and condition in _SPECIAL_CONDITIONS:
        normalized = _normalize_conv_arguments(op_name, sample)
        if normalized is None:
            return None
        weight, bias, conv_kwargs = normalized
        if weight.dtype != dtype or sample.input.dtype != dtype:
            return None
        if bias is not None and (not isinstance(bias, torch.Tensor) or bias.dtype != dtype):
            return None
        return _build(
            "complex_convolution_four_real",
            "conv",
            lambda: complex_convolution_reference(
                op_name,
                sample.input,
                weight,
                bias,
                conv_kwargs,
            ),
        )

    return None


def resolve_generated_forward_reference(
    dispatcher_name,
    sample,
    dtype,
    input_condition,
) -> ContractReference | None:
    condition = _condition(input_condition)

    if (
        "special_laguerre_polynomial_l" in dispatcher_name
        and condition in _SPECIAL_CONDITIONS
        and isinstance(sample.input, torch.Tensor)
        and sample.args
    ):
        return _build(
            "laguerre_initialized_recurrence",
            "elementwise",
            lambda: laguerre_polynomial_reference(sample.input, sample.args[0]),
        )

    if condition in _SPECIAL_CONDITIONS and isinstance(sample.input, torch.Tensor) and sample.args:
        for token, family in (
            ("special_shifted_chebyshev_polynomial_t", "t"),
            ("special_shifted_chebyshev_polynomial_u", "u"),
            ("special_shifted_chebyshev_polynomial_v", "v"),
            ("special_shifted_chebyshev_polynomial_w", "w"),
        ):
            if token in dispatcher_name:
                return _build(
                    "shifted_chebyshev_initialized_recurrence",
                    "elementwise",
                    lambda family=family: shifted_chebyshev_polynomial_reference(
                        family, sample.input, sample.args[0]
                    ),
                )

    if dispatcher_name == "aten::_cast_Byte" and isinstance(sample.input, torch.Tensor):
        if sample.input.dtype.is_floating_point:
            return _build(
                "float_to_uint8_signed_truncate",
                "copy",
                lambda: float_to_uint8_reference(sample.input),
            )

    if "_upsample_lanczos2d_aa" in dispatcher_name and condition in _SPECIAL_CONDITIONS:
        if isinstance(sample.input, torch.Tensor) and len(sample.args) >= 2:
            output_size, align_corners = sample.args[:2]
            scale_arguments = sample.args[2:]
            if len(scale_arguments) == 1 and isinstance(scale_arguments[0], (list, tuple)):
                scales_h, scales_w = scale_arguments[0]
            else:
                scales_h = scale_arguments[0] if len(scale_arguments) > 0 else None
                scales_w = scale_arguments[1] if len(scale_arguments) > 1 else None
            return _build(
                "lanczos3_exact_zero",
                "elementwise",
                lambda: lanczos2d_aa_reference(
                    sample.input,
                    output_size,
                    align_corners,
                    scales_h,
                    scales_w,
                ),
            )

    if dispatcher_name in {"aten::_dirichlet_grad", "aten::_dirichlet_grad.out"}:
        if isinstance(sample.input, torch.Tensor) and len(sample.args) >= 2:
            return _build(
                "dirichlet_grad_mpmath",
                "elementwise",
                lambda: dirichlet_grad_reference(sample.input, sample.args[0], sample.args[1]),
            )

    if dispatcher_name in {"aten::_standard_gamma_grad", "aten::_standard_gamma_grad.out"}:
        if isinstance(sample.input, torch.Tensor) and sample.args:
            return _build(
                "standard_gamma_grad_mpmath",
                "elementwise",
                lambda: standard_gamma_grad_reference(sample.input, sample.args[0]),
            )

    if dispatcher_name in _GENERATED_I0:
        if isinstance(sample.input, torch.Tensor):
            return _build("i0_mpmath", "elementwise", lambda: i0_reference(sample.input))

    if "polygamma" in dispatcher_name:
        if isinstance(sample.input, torch.Tensor) and sample.args:
            tensor, order = sample.input, int(sample.args[0])
        elif isinstance(sample.input, int) and sample.args and isinstance(sample.args[0], torch.Tensor):
            tensor, order = sample.args[0], int(sample.input)
        else:
            tensor = None
        if tensor is not None and order == 1:
            return _build(
                "polygamma_mpmath_poles",
                "elementwise",
                lambda: polygamma_reference(order, tensor),
            )

    if (
        dispatcher_name in _FOREACH_COMPLEX_TENSOR_MUL_SCHEMAS
        and dtype in _ALL_COMPLEX_DTYPES
        and condition in _SPECIAL_CONDITIONS
        and isinstance(sample.input, list)
        and sample.args
        and isinstance(sample.args[0], torch.Tensor)
    ):
        return _build(
            "foreach_complex_c99_tensor_mul",
            "elementwise",
            lambda: [
                complex_mul_reference(input_tensor, sample.args[0])
                for input_tensor in sample.input
            ],
        )

    if (
        dispatcher_name in _FOREACH_COMPLEX_ADD_SUB_SCHEMAS
        and dtype in _ALL_COMPLEX_DTYPES
        and condition in _SPECIAL_CONDITIONS
        and isinstance(sample.input, list)
    ):
        return _build(
            "foreach_complex_real_alpha_add_sub",
            "elementwise",
            lambda: foreach_complex_compound_reference(
                dispatcher_name,
                sample.input,
                sample.args,
                sample.kwargs,
            ),
        )

    if (
        dispatcher_name in _FOREACH_COMPLEX_EXISTING_COMPOUND_SCHEMAS
        and dtype in _STANDARD_COMPLEX_DTYPES
        and condition in _SPECIAL_CONDITIONS
        and isinstance(sample.input, list)
    ):
        return _build(
            "foreach_complex_c99_compound",
            "elementwise",
            lambda: foreach_complex_compound_reference(
                dispatcher_name,
                sample.input,
                sample.args,
                sample.kwargs,
            ),
        )

    if dispatcher_name in _GENERATED_COMPLEX_UNARY and dtype in _STANDARD_COMPLEX_DTYPES and condition in _SPECIAL_CONDITIONS:
        if _is_complex_tensor_tree(sample.input, dtype):
            reference_id, fn = _GENERATED_COMPLEX_UNARY[dispatcher_name]
            return _build(reference_id, "elementwise", lambda: _complex_unary_value(sample.input, fn))

    if dispatcher_name in {"aten::ldexp.Tensor", "aten::ldexp_", "aten::ldexp.out"} and dtype in _ALL_COMPLEX_DTYPES and condition in _SPECIAL_CONDITIONS:
        if sample.args and isinstance(sample.input, torch.Tensor) and isinstance(sample.args[0], torch.Tensor):
            if sample.input.is_complex() and sample.args[0].is_complex():
                return _build(
                    "complex_ldexp_phase_scale",
                    "elementwise",
                    lambda: complex_ldexp_reference(sample.input, sample.args[0]),
                )
            if (
                sample.input.is_complex()
                and sample.args[0].dtype != torch.bool
                and not sample.args[0].is_floating_point()
                and not sample.args[0].is_complex()
            ):
                return _build(
                    "complex_ldexp_integral_lane_scale",
                    "elementwise",
                    lambda: complex_integral_ldexp_reference(sample.input, sample.args[0]),
                )

    if dispatcher_name in {"aten::cumprod_", "aten::cumprod.out"} and dtype in _STANDARD_COMPLEX_DTYPES and condition in _SPECIAL_CONDITIONS:
        dim = sample.args[0] if sample.args else sample.kwargs.get("dim", 0)
        return _build(
            "complex_cumprod_inclusive",
            "reduction",
            lambda: complex_cumprod_reference(sample.input, int(dim)),
        )

    if condition == "clean" and dtype in {torch.float16, torch.bfloat16} and dispatcher_name in {
        "aten::soft_margin_loss",
        "aten::soft_margin_loss.out",
    }:
        normalized = _normalize_soft_margin_arguments(sample)
        if normalized is not None:
            target, reduction = normalized
            return _build(
                "soft_margin_f32_opmath",
                "loss",
                lambda: soft_margin_loss_reference(sample.input, target, reduction),
            )

    if condition == "clean" and dtype in {torch.float16, torch.bfloat16, torch.float32} and dispatcher_name in {
        "aten::linalg_matrix_exp",
        "aten::linalg_matrix_exp.out",
    }:
        if isinstance(sample.input, torch.Tensor):
            return _build(
                "matrix_exp_f64_scaling_squaring",
                "linalg",
                lambda: matrix_exp_f64_reference(sample.input),
            )

    if condition not in _SPECIAL_CONDITIONS:
        return None

    if dispatcher_name == "aten::binary_cross_entropy_with_logits.out":
        normalized = _normalize_bce_with_logits_arguments(sample)
        if normalized is None:
            return None
        target, weight, pos_weight, reduction = normalized
        return _build(
            "bce_with_logits_stable",
            "loss",
            lambda: binary_cross_entropy_with_logits_reference(
                sample.input,
                target,
                weight,
                pos_weight,
                reduction,
            ),
        )

    if dispatcher_name in _GENERATED_ADD_SUB and dtype in _UNIT_ALPHA_DTYPES:
        return _unit_alpha_reference(_GENERATED_ADD_SUB[dispatcher_name], sample, dtype)

    if dispatcher_name in _GENERATED_POWER and dtype in _INTEGER_POWER_DTYPES:
        return _power_reference(dispatcher_name, sample, dtype)

    if dispatcher_name == "aten::l1_loss" and dtype in _STANDARD_COMPLEX_DTYPES:
        if (
            not isinstance(sample.input, torch.Tensor)
            or not sample.args
            or not isinstance(sample.args[0], torch.Tensor)
            or not (sample.input.is_complex() or sample.args[0].is_complex())
            or not _l1_operand_dtypes_match(sample.input, sample.args[0], dtype)
        ):
            return None
        reduction = sample.args[1] if len(sample.args) > 1 else sample.kwargs.get("reduction", 1)
        return _build(
            "complex_l1_loss",
            "loss",
            lambda: complex_l1_loss_reference(sample.input, sample.args[0], reduction),
        )

    if dispatcher_name in _GENERATED_LOG2 and dtype in _STANDARD_COMPLEX_DTYPES:
        if not _is_complex_tensor_tree(sample.input, dtype):
            return None
        return _build(
            "complex_log2",
            "elementwise",
            lambda: _log2_value(sample.input),
        )

    return None
