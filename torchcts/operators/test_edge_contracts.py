# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies or substantial portions of the Software.

import math

import pytest
import torch

from torchcts.core.device import synchronize
from torchcts.core.high_precision_reference import (
    kaiser_window_reference,
    regularized_gamma_reference,
)
from torchcts.core.reference_oracles import (
    col2im_reference,
    gcd_integer_reference,
    histc_integer_count_reference,
    im2col_reference,
    logit_backward_reference,
    wide_ldexp_reference,
)


_SEMANTIC_ORACLE = pytest.mark.cpu_contract_exempt(
    "TorchCTS-owned semantic oracle intentionally replaces a broken or absent CPU path"
)


@pytest.mark.smoke
@pytest.mark.covers("aten::im2col.out", surface="out_variant")
@pytest.mark.covers("aten::col2im.out", surface="out_variant")
def test_im2col_col2im_out_input_alias_uses_pre_resize_values(device, compare):
    kernel = [2, 2]
    dilation = [1, 1]
    padding = [0, 0]
    stride = [1, 1]
    image_cpu = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
    expected_columns = im2col_reference(image_cpu, kernel, dilation, padding, stride)
    expected_image = col2im_reference(
        expected_columns, [3, 4], kernel, dilation, padding, stride
    )
    if torch.device(device).type == "cpu":
        return

    aliased_image = image_cpu.to(device)
    returned = torch.ops.aten.im2col.out(
        aliased_image,
        kernel,
        dilation,
        padding,
        stride,
        out=aliased_image,
    )
    synchronize(device)
    assert returned is aliased_image
    compare(aliased_image, expected_columns, category="copy", dtype=torch.float32)

    aliased_columns = expected_columns.to(device)
    returned = torch.ops.aten.col2im.out(
        aliased_columns,
        [3, 4],
        kernel,
        dilation,
        padding,
        stride,
        out=aliased_columns,
    )
    synchronize(device)
    assert returned is aliased_columns
    compare(aliased_columns, expected_image, category="copy", dtype=torch.float32)


@pytest.mark.smoke
@_SEMANTIC_ORACLE
@pytest.mark.covers("aten::gcd")
@pytest.mark.parametrize("dtype", [torch.int8, torch.int16])
def test_compact_gcd_signed_minimum_uses_unsigned_magnitude(dtype, device):
    minimum = torch.iinfo(dtype).min
    odd = 31 if dtype == torch.int8 else 7
    left = torch.tensor([minimum, minimum, minimum], dtype=dtype)
    right = torch.tensor([odd, -odd, 0], dtype=dtype)
    expected = gcd_integer_reference(left, right)
    assert torch.equal(expected[:2], torch.ones(2, dtype=dtype))
    assert expected[-1] == minimum
    if torch.device(device).type == "cpu":
        return
    actual = torch.gcd(left.to(device), right.to(device))
    synchronize(device)
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.smoke
@_SEMANTIC_ORACLE
@pytest.mark.covers("aten::histc")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_histc_casts_completed_integer_counts_once(dtype, device):
    input_cpu = torch.cat(
        [
            torch.full((4062,), 0.5, dtype=dtype),
            torch.full((35,), 2.0, dtype=dtype),
        ]
    )
    expected = histc_integer_count_reference(input_cpu, 1, 0.0, 1.0)
    assert int(expected.item()) in {4062, 4064}
    if torch.device(device).type == "cpu":
        return
    actual = torch.histc(input_cpu.to(device), bins=1, min=0.0, max=1.0)
    synchronize(device)
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.smoke
@_SEMANTIC_ORACLE
@pytest.mark.covers("aten::igamma")
@pytest.mark.covers("aten::igammac")
@pytest.mark.parametrize("dtype", [torch.float64])
def test_igamma_tiny_shape_tails_use_high_precision_reference(dtype, device):
    shape = torch.tensor([1e-300, 1e-300], dtype=dtype)
    value = torch.tensor([1e-300, 1.0], dtype=dtype)
    expected_lower = regularized_gamma_reference(shape, value, upper=False)
    expected_upper = regularized_gamma_reference(shape, value, upper=True)
    if torch.device(device).type == "cpu":
        return
    actual_lower = torch.igamma(shape.to(device), value.to(device)).cpu()
    actual_upper = torch.igammac(shape.to(device), value.to(device)).cpu()
    synchronize(device)
    for actual, expected in zip(actual_lower, expected_lower):
        assert abs(float(actual) - float(expected)) <= max(abs(float(expected)) * 5e-13, 1e-320)
    for actual, expected in zip(actual_upper, expected_upper):
        assert abs(float(actual) - float(expected)) <= max(abs(float(expected)) * 5e-13, 1e-320)


@pytest.mark.smoke
@_SEMANTIC_ORACLE
@pytest.mark.covers("aten::dot")
@pytest.mark.covers("aten::vdot")
@pytest.mark.parametrize("dtype", [torch.complex64])
def test_long_strided_complex_dot_uses_c128_residual(dtype, device):
    length = 500_002
    generator = torch.Generator().manual_seed(67)
    left_source = torch.complex(
        torch.rand(2 * length, generator=generator) * 18 - 9,
        torch.rand(2 * length, generator=generator) * 18 - 9,
    ).to(dtype)
    right_source = torch.complex(
        torch.rand(2 * length, generator=generator) * 18 - 9,
        torch.rand(2 * length, generator=generator) * 18 - 9,
    ).to(dtype)
    left = left_source[::2]
    right = right_source[::2]
    reference_left = left.to(torch.complex128)
    reference_right = right.to(torch.complex128)
    scale = (reference_left.abs() * reference_right.abs()).sum().item()
    residual_bound = max(0.01, 0.25 * torch.finfo(torch.float32).eps * scale)
    expected_dot = torch.dot(reference_left, reference_right)
    expected_vdot = torch.vdot(reference_left, reference_right)
    if torch.device(device).type == "cpu":
        return

    left_dev = left_source.to(device)[::2]
    right_dev = right_source.to(device)[::2]
    assert left_dev.stride(0) == 2 and right_dev.stride(0) == 2
    actual_dot = torch.dot(left_dev, right_dev).cpu().to(torch.complex128)
    actual_vdot = torch.vdot(left_dev, right_dev).cpu().to(torch.complex128)
    synchronize(device)
    assert abs(actual_dot - expected_dot).item() <= residual_bound
    assert abs(actual_vdot - expected_vdot).item() <= residual_bound


@pytest.mark.smoke
@_SEMANTIC_ORACLE
@pytest.mark.covers("aten::flip")
@pytest.mark.parametrize("dtype", [torch.complex32])
def test_complex32_flip_uses_raw_element_bits(dtype, device):
    raw = torch.tensor(
        [
            0x00000000,
            0x80000000 - 2**32,
            0x7C017E00,
            0xFC007C00 - 2**32,
            0x3555B555,
            0x00010002,
        ],
        dtype=torch.int32,
    ).reshape(2, 3)
    expected_bits = torch.flip(raw, [0, 1])
    if torch.device(device).type == "cpu":
        return
    input_dev = raw.to(device).view(dtype)
    actual_bits = torch.flip(input_dev, [0, 1]).view(torch.int32).cpu()
    synchronize(device)
    assert torch.equal(actual_bits, expected_bits)


@pytest.mark.smoke
@_SEMANTIC_ORACLE
@pytest.mark.covers("aten::kaiser_window")
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.complex32, torch.complex64],
)
def test_kaiser_large_and_infinite_beta_use_stable_limit(dtype, device, compare):
    expected_large = kaiser_window_reference(9, False, 100.0, dtype)
    expected_infinite = kaiser_window_reference(9, False, math.inf, dtype)
    assert torch.isfinite(torch.view_as_real(expected_large) if dtype.is_complex else expected_large).all()
    assert expected_infinite[4] == 1
    assert torch.count_nonzero(expected_infinite) == 1
    if torch.device(device).type == "cpu":
        return
    actual_large = torch.kaiser_window(
        9, periodic=False, beta=100.0, dtype=dtype, device=device
    )
    actual_infinite = torch.kaiser_window(
        9, periodic=False, beta=math.inf, dtype=dtype, device=device
    )
    synchronize(device)
    compare(actual_large, expected_large, category="elementwise", dtype=dtype)
    compare(actual_infinite, expected_infinite, category="exact", dtype=dtype)


@pytest.mark.smoke
@_SEMANTIC_ORACLE
@pytest.mark.covers("aten::maximum")
@pytest.mark.covers("aten::minimum")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_strided_extrema_preserve_first_signed_zero_on_ties(dtype, device):
    left_base = torch.tensor(
        [[-0.0, 0.0, -0.0, 0.0], [0.0, -0.0, 0.0, -0.0]], dtype=dtype
    )
    right_base = -left_base
    left = left_base.t()
    right = right_base.t()
    assert not left.is_contiguous() and not right.is_contiguous()
    expected_sign = torch.signbit(left)
    if torch.device(device).type == "cpu":
        return
    left_dev = left_base.to(device).t()
    right_dev = right_base.to(device).t()
    for operator in (torch.maximum, torch.minimum):
        actual = operator(left_dev, right_dev)
        synchronize(device)
        assert torch.equal(torch.signbit(actual).cpu(), expected_sign)


@pytest.mark.smoke
@_SEMANTIC_ORACLE
@pytest.mark.covers("aten::ldexp.Tensor")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ldexp_wide_int64_exponents_do_not_wrap(dtype, device):
    values = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=dtype)
    exponents = torch.tensor([2**32, 2**32, -(2**32), -(2**32)], dtype=torch.int64)
    expected = wide_ldexp_reference(values, exponents)
    assert torch.isposinf(expected[0]) and torch.isneginf(expected[1])
    assert expected[2] == 0 and not torch.signbit(expected[2])
    assert expected[3] == 0 and torch.signbit(expected[3])
    if torch.device(device).type == "cpu":
        return
    actual = torch.ldexp(values.to(device), exponents.to(device)).cpu()
    synchronize(device)
    assert torch.equal(torch.isinf(actual), torch.isinf(expected))
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))


@pytest.mark.smoke
@_SEMANTIC_ORACLE
@pytest.mark.covers("aten::logit_backward")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("length", [1, 7, 8, 9, 15, 16, 17])
def test_logit_backward_negative_zero_is_simd_width_independent(dtype, length, device):
    input_cpu = torch.full((length,), -0.0, dtype=dtype)
    grad_cpu = torch.ones(length, dtype=dtype)
    expected = logit_backward_reference(grad_cpu, input_cpu, None)
    assert torch.isneginf(expected).all()
    if torch.device(device).type == "cpu":
        return
    actual = torch.ops.aten.logit_backward.default(
        grad_cpu.to(device), input_cpu.to(device), None
    ).cpu()
    synchronize(device)
    assert torch.isneginf(actual).all()
