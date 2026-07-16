# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software.

import math
from types import SimpleNamespace

import pytest
import torch

import torchcts.core.contract_references as contract_references
from torchcts.core.contract_references import (
    ContractReferenceError,
    resolve_generated_forward_reference,
    resolve_opinfo_forward_reference,
)
from torchcts.core.reference_oracles import (
    complex_convolution_reference,
    complex_l1_loss_reference,
    complex_log2_reference,
    complex_tensor_integer_power_reference,
    complex_unit_alpha_add_sub_reference,
    embedding_bag_scale_grad_by_freq_reference,
    grid_sampler_3d_backward_f32_reference,
    slow_complex_convolution_reference,
)


pytestmark = pytest.mark.covers_category("selftest")


def _sample(input_value, *args, **kwargs):
    return SimpleNamespace(input=input_value, args=args, kwargs=kwargs)


def _assert_complex_special_equal(actual, expected, *, rtol=2e-4, atol=2e-4):
    actual_lanes = torch.view_as_real(actual)
    expected_lanes = torch.view_as_real(expected)
    assert torch.equal(torch.isnan(actual_lanes), torch.isnan(expected_lanes))
    assert torch.equal(torch.isinf(actual_lanes), torch.isinf(expected_lanes))
    inf_mask = torch.isinf(actual_lanes)
    assert torch.equal(
        torch.signbit(actual_lanes[inf_mask]),
        torch.signbit(expected_lanes[inf_mask]),
    )
    finite_mask = torch.isfinite(actual_lanes) & torch.isfinite(expected_lanes)
    torch.testing.assert_close(
        actual_lanes[finite_mask],
        expected_lanes[finite_mask],
        rtol=rtol,
        atol=atol,
    )


def test_complex_unit_alpha_reference_keeps_lanes_independent():
    left = torch.complex(
        torch.tensor([0.0, 1.0]),
        torch.tensor([0.5, float("inf")]),
    ).to(torch.complex64)
    right = torch.complex(
        torch.tensor([-float("inf"), 2.0]),
        torch.tensor([0.25, 3.0]),
    ).to(torch.complex64)

    added = complex_unit_alpha_add_sub_reference("add", left, right)
    subtracted = complex_unit_alpha_add_sub_reference("sub", left, right)
    reversed_subtraction = complex_unit_alpha_add_sub_reference("rsub", left, right)

    assert torch.isneginf(added.real[0]) and added.imag[0] == 0.75
    assert torch.isposinf(subtracted.real[0]) and subtracted.imag[0] == 0.25
    assert torch.isneginf(reversed_subtraction.real[0])
    assert reversed_subtraction.imag[0] == -0.25


def test_unit_alpha_router_is_closed_and_semantic():
    dtype = torch.complex64
    left = torch.tensor([1 + 2j], dtype=dtype)
    right = torch.tensor([3 + 4j], dtype=dtype)

    assert resolve_opinfo_forward_reference(
        "add", _sample(left, right, alpha=2), dtype, "has_inf"
    ) is None
    assert resolve_opinfo_forward_reference(
        "add", _sample(left, right), dtype, "clean"
    ) is None
    assert resolve_opinfo_forward_reference(
        "multiply", _sample(left, right), dtype, "has_inf"
    ) is None
    matched = resolve_generated_forward_reference(
        "aten::add_.Tensor", _sample(left, right), dtype, "has_inf"
    )
    assert matched is not None
    assert matched.reference_id == "complex_unit_alpha_add_sub"


def test_complex_integer_power_reference():
    base = torch.complex(
        torch.tensor([-float("inf"), 1.0682]),
        torch.tensor([0.0625, float("inf")]),
    ).to(torch.complex64)
    exponent = torch.full(base.shape, complex(2, 0), dtype=torch.complex64)
    native = torch.pow(base, exponent)

    result = complex_tensor_integer_power_reference(base, exponent, native)
    expected = torch.pow(base, 2)

    torch.testing.assert_close(result, expected, equal_nan=True)


def test_complex_integer_power_leaves_general_exponents_native():
    base = torch.tensor([2 + 1j] * 8, dtype=torch.complex64)
    exponent = torch.tensor(
        [
            -1 + 0j,
            1.5 + 0j,
            2 + 1j,
            complex(float("nan"), 0),
            complex(float("inf"), 0),
            complex(2, float("nan")),
            complex(2, float("inf")),
            complex(-float("inf"), 0),
        ],
        dtype=torch.complex64,
    )
    native = torch.pow(base, exponent)

    result = complex_tensor_integer_power_reference(base, exponent, native)

    torch.testing.assert_close(result, native, equal_nan=True)


def test_complex_log2_reference_preserves_finite_phase():
    value = torch.complex(
        torch.tensor([float("inf"), -float("inf")]),
        torch.tensor([1.0, 1.0]),
    ).to(torch.complex64)

    result = complex_log2_reference(value)

    assert torch.isposinf(result.real).all()
    assert torch.isfinite(result.imag).all()
    torch.testing.assert_close(
        result.imag,
        torch.tensor([0.0, math.pi / math.log(2.0)]),
    )


def test_matched_reference_failure_never_falls_back(monkeypatch):
    value = torch.tensor([complex(float("inf"), 1.0)], dtype=torch.complex64)

    def fail(_value):
        raise RuntimeError("intentional oracle failure")

    monkeypatch.setattr(contract_references, "complex_log2_reference", fail)

    with pytest.raises(ContractReferenceError, match="complex_log2"):
        resolve_opinfo_forward_reference(
            "log2",
            _sample(value),
            torch.complex64,
            "has_inf",
        )


def test_grid_sampler_f32_reference_matches_f32_autograd():
    dtype = torch.bfloat16
    input_tensor = torch.linspace(-1.0, 1.0, 24).reshape(1, 1, 2, 3, 4).to(dtype)
    grid = torch.tensor(
        [[[[[-1.25, -0.5, 0.25], [0.75, 1.2, -0.8]]]]],
        dtype=dtype,
    )
    grad_output = torch.tensor([[[[[0.75, -1.25]]]]], dtype=dtype)

    actual = grid_sampler_3d_backward_f32_reference(
        grad_output,
        input_tensor,
        grid,
        0,
        2,
        True,
    )
    input_f32 = input_tensor.float().requires_grad_(True)
    grid_f32 = grid.float().requires_grad_(True)
    output_f32 = torch.ops.aten.grid_sampler_3d.default(
        input_f32,
        grid_f32,
        0,
        2,
        True,
    )
    expected_f32 = torch.autograd.grad(
        output_f32,
        (input_f32, grid_f32),
        grad_outputs=grad_output.float(),
    )
    expected = tuple(item.to(dtype) for item in expected_f32)

    for actual_item, expected_item in zip(actual, expected):
        torch.testing.assert_close(actual_item, expected_item, rtol=0, atol=0)


def test_complex_l1_reference_uses_independent_lanes():
    input_tensor = torch.tensor(
        [3 + 4j, complex(float("inf"), 1.0)],
        dtype=torch.complex64,
    )
    target = torch.tensor([0.0, 2.0], dtype=torch.float32)

    result = complex_l1_loss_reference(input_tensor, target, "none")

    assert result.dtype == torch.float32
    assert result[0] == 5.0
    assert torch.isposinf(result[1])


def test_embedding_bag_frequency_reference_uses_each_rows_count():
    grad = torch.tensor([[1.0, -0.5], [-0.25, 2.0]])
    indices = torch.tensor([1, 1, 2, 1, 2, 4])
    offset2bag = torch.tensor([0, 0, 0, 1, 1, 1])
    expected = torch.tensor(
        [
            [0.0, 0.0],
            [1.75 / 3.0, 1.0 / 3.0],
            [0.375, 0.75],
            [0.0, 0.0],
            [-0.25, 2.0],
        ]
    )

    result = embedding_bag_scale_grad_by_freq_reference(
        grad,
        indices,
        offset2bag,
        5,
    )

    torch.testing.assert_close(result, expected)
    assert torch.equal(result[[0, 3]], torch.zeros((2, 2), dtype=result.dtype))


def test_complex_convolution_reference_matches_independent_scalar_reference():
    generator = torch.Generator().manual_seed(0)
    dtype = torch.complex64
    input_shape = (1, 4, 3, 3)
    weight_shape = (4, 2, 2, 2)
    kwargs = {
        "groups": 2,
        "padding": (1, 1),
        "stride": (2, 2),
        "dilation": (2, 2),
        "output_padding": (1, 1),
    }
    input_tensor = torch.complex(
        torch.randn(input_shape, generator=generator),
        torch.randn(input_shape, generator=generator),
    ).to(dtype)
    weight = torch.complex(
        torch.randn(weight_shape, generator=generator),
        torch.randn(weight_shape, generator=generator),
    ).to(dtype)
    bias = torch.complex(
        torch.randn(4, generator=generator),
        torch.randn(4, generator=generator),
    ).to(dtype)
    input_tensor.reshape(-1)[0] = complex(float("inf"), 0.25)

    fast = complex_convolution_reference(
        "nn.functional.conv_transpose2d",
        input_tensor,
        weight,
        bias,
        kwargs,
    )
    slow = slow_complex_convolution_reference(
        "nn.functional.conv_transpose2d",
        input_tensor,
        weight,
        bias,
        kwargs,
    )

    _assert_complex_special_equal(fast, slow)
