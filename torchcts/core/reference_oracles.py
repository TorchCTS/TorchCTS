# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies or substantial portions of the Software.

from __future__ import annotations

import torch


def _complex32_dtype() -> torch.dtype | None:
    value = getattr(torch, "complex32", None)
    return value if isinstance(value, torch.dtype) else None


def _matmul_reference_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    complex32 = _complex32_dtype()
    if complex32 is not None and tensor.dtype == complex32:
        return tensor.to(torch.complex64)
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
        result = torch.matmul(_matmul_reference_tensor(args[0]), _matmul_reference_tensor(args[1]))
        return _matmul_reference_result(result, dtype)

    if base == "bmm":
        if len(args) < 2:
            raise ValueError(f"{dispatcher_name} reference requires batch1 and batch2")
        result = torch.bmm(_matmul_reference_tensor(args[0]), _matmul_reference_tensor(args[1]))
        return _matmul_reference_result(result, dtype)

    if base == "addmm":
        if len(args) < 3:
            raise ValueError(f"{dispatcher_name} reference requires input, mat1, and mat2")
        beta = kwargs.get("beta", 1)
        alpha = kwargs.get("alpha", 1)
        input_value = _matmul_reference_tensor(args[0])
        product = torch.matmul(_matmul_reference_tensor(args[1]), _matmul_reference_tensor(args[2]))
        return _matmul_reference_result(input_value * beta + product * alpha, dtype)

    if base == "addbmm":
        if len(args) < 3:
            raise ValueError(f"{dispatcher_name} reference requires input, batch1, and batch2")
        beta = kwargs.get("beta", 1)
        alpha = kwargs.get("alpha", 1)
        input_value = _matmul_reference_tensor(args[0])
        product = torch.bmm(_matmul_reference_tensor(args[1]), _matmul_reference_tensor(args[2])).sum(dim=0)
        return _matmul_reference_result(input_value * beta + product * alpha, dtype)

    if base == "baddbmm":
        if len(args) < 3:
            raise ValueError(f"{dispatcher_name} reference requires input, batch1, and batch2")
        beta = kwargs.get("beta", 1)
        alpha = kwargs.get("alpha", 1)
        input_value = _matmul_reference_tensor(args[0])
        product = torch.bmm(_matmul_reference_tensor(args[1]), _matmul_reference_tensor(args[2]))
        return _matmul_reference_result(input_value * beta + product * alpha, dtype)

    if base == "chain_matmul":
        if not args or not isinstance(args[0], (list, tuple)) or len(args[0]) < 2:
            raise ValueError(f"{dispatcher_name} reference requires a tensor list with at least two matrices")
        matrices = [_matmul_reference_tensor(matrix) for matrix in args[0]]
        result = matrices[0]
        for matrix in matrices[1:]:
            result = torch.matmul(result, matrix)
        return _matmul_reference_result(result, dtype)

    if base == "linear":
        if len(args) < 2:
            raise ValueError(f"{dispatcher_name} reference requires input and weight")
        input_value = _matmul_reference_tensor(args[0])
        weight = _matmul_reference_tensor(args[1])
        bias = args[2] if len(args) >= 3 else kwargs.get("bias")
        result = torch.matmul(input_value, weight.transpose(-2, -1))
        if bias is not None:
            result = result + _matmul_reference_tensor(bias)
        return _matmul_reference_result(result, dtype)

    raise ValueError(f"{dispatcher_name} is not a supported matmul-family reference surface")


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
