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

from torchcts.core.reference_oracles import (
    complex_convolution_reference,
    complex_l1_loss_reference,
    complex_log2_reference,
    complex_tensor_integer_power_reference,
    complex_unit_alpha_add_sub_reference,
    conv_transpose3d_f32_reference,
    has_nonnegative_integral_complex_exponent,
)


@dataclass(frozen=True)
class ContractReference:
    value: object
    reference_id: str
    category: str


class ContractReferenceError(RuntimeError):
    pass


_SPECIAL_CONDITIONS = frozenset({"has_nan", "has_inf"})
_COMPLEX_DTYPES = frozenset({torch.complex64, torch.complex128})

_OPINFO_ADD_SUB = {
    "add": "add",
    "sub": "sub",
    "rsub": "rsub",
}

_GENERATED_ADD_SUB = {
    "aten::subtract.Tensor": "sub",
    "aten::add_.Tensor": "add",
    "aten::add.out": "add",
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


def resolve_opinfo_forward_reference(
    op_name,
    sample,
    dtype,
    input_condition,
) -> ContractReference | None:
    condition = _condition(input_condition)

    if op_name in _OPINFO_ADD_SUB and dtype in _COMPLEX_DTYPES and condition in _SPECIAL_CONDITIONS:
        return _unit_alpha_reference(_OPINFO_ADD_SUB[op_name], sample, dtype)

    if op_name == "nn.functional.l1_loss" and dtype in _COMPLEX_DTYPES and condition in _SPECIAL_CONDITIONS:
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

    if op_name == "log2" and dtype in _COMPLEX_DTYPES and condition in _SPECIAL_CONDITIONS:
        if not _is_complex_tensor_tree(sample.input, dtype):
            return None
        return _build(
            "complex_log2",
            "elementwise",
            lambda: complex_log2_reference(sample.input),
        )

    if op_name == "nn.functional.conv_transpose3d" and dtype == torch.bfloat16 and condition == "clean":
        normalized = _normalize_conv_arguments(op_name, sample)
        if normalized is None:
            return None
        weight, bias, conv_kwargs = normalized
        if sample.input.dtype != dtype or weight.dtype != dtype:
            return None
        if bias is not None and bias.dtype != dtype:
            return None
        return _build(
            "bf16_conv_transpose3d_f32",
            "conv",
            lambda: conv_transpose3d_f32_reference(sample.input, weight, bias, **conv_kwargs),
        )

    if op_name in _COMPLEX_CONV_OPS and dtype in _COMPLEX_DTYPES and condition in _SPECIAL_CONDITIONS:
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
    if dtype not in _COMPLEX_DTYPES or condition not in _SPECIAL_CONDITIONS:
        return None

    if dispatcher_name in _GENERATED_ADD_SUB:
        return _unit_alpha_reference(_GENERATED_ADD_SUB[dispatcher_name], sample, dtype)

    if dispatcher_name in _GENERATED_POWER:
        return _power_reference(dispatcher_name, sample, dtype)

    if dispatcher_name == "aten::l1_loss":
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

    if dispatcher_name in _GENERATED_LOG2:
        if not _is_complex_tensor_tree(sample.input, dtype):
            return None
        return _build(
            "complex_log2",
            "elementwise",
            lambda: _log2_value(sample.input),
        )

    return None
