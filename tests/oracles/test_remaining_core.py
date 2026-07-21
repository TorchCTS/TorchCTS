import struct

import pytest
import torch

from tests.oracles.schema import decode_tensor, encode_tensor
from tests.oracles.test_fixed_values import _case
from torchcts.core.high_precision_reference import (
    dirichlet_grad_reference,
    i0_reference,
    kaiser_window_reference,
    polygamma_reference,
    regularized_gamma_reference,
    standard_gamma_grad_reference,
)
from torchcts.core.reference_oracles import (
    binary_cross_entropy_with_logits_reference,
    complex_covariance_reference,
    complex_cumprod_reference,
    complex_expm1_reference,
    complex_gradient_reference,
    complex_integral_ldexp_reference,
    complex_ldexp_reference,
    complex_mul_reference,
    complex_rsqrt_reference,
    complex_sigmoid_reference,
    conv_transpose3d_f32_reference,
    conv_transpose_f32_reference,
    foreach_complex_compound_reference,
    grid_sampler_backward_f32_reference,
    grid_sampler_forward_f32_reference,
    has_nonnegative_integral_complex_exponent,
    lanczos2d_aa_reference,
    lanczos3_coefficient,
    matrix_exp_f64_reference,
    wide_ldexp_reference,
)


def _assert_exact(result, expected):
    assert encode_tensor(result) == expected


def _ordered_bits(value, *, width):
    code, integer_code = ("f", "I") if width == 32 else ("d", "Q")
    bits = struct.unpack(">" + integer_code, struct.pack(">" + code, float(value)))[0]
    sign = 1 << (width - 1)
    mask = (1 << width) - 1
    return (mask - bits) if bits & sign else bits + sign


def _assert_ulp(result, expected_record, ceiling):
    expected = decode_tensor(expected_record)
    assert result.dtype == expected.dtype
    assert result.shape == expected.shape
    if result.is_complex():
        result_values = torch.view_as_real(result).reshape(-1).tolist()
        expected_values = torch.view_as_real(expected).reshape(-1).tolist()
        width = 64 if result.dtype == torch.complex128 else 32
    else:
        result_values = result.reshape(-1).tolist()
        expected_values = expected.reshape(-1).tolist()
        width = 64 if result.dtype == torch.float64 else 32
    distances = [
        abs(_ordered_bits(actual, width=width) - _ordered_bits(wanted, width=width))
        for actual, wanted in zip(result_values, expected_values)
    ]
    assert max(distances, default=0) <= ceiling


@pytest.mark.oracle_contract(id="cp-complex_arith-remaining-core", validation_class="V1_FIXED_VALUE")
def test_remaining_complex_arithmetic_matches_scalar_records():
    case = _case("cp-complex_arith-remaining-core")
    inputs, expected = case["inputs"], case["expected"]
    for predicate_case in inputs["exponent_cases"]:
        assert has_nonnegative_integral_complex_exponent(
            decode_tensor(predicate_case["input"])
        ) is predicate_case["expected"]
    left = decode_tensor(inputs["left"])
    right = decode_tensor(inputs["right"])
    _assert_exact(complex_mul_reference(left, right), expected["multiplied"])
    base = decode_tensor(inputs["foreach_base"])
    other = decode_tensor(inputs["foreach_other"])
    add = foreach_complex_compound_reference(
        "aten::_foreach_add.Tensor", [base], (other,), {"alpha": 0.5}
    )
    _assert_exact(add[0], expected["foreach_add_alpha_half"])
    addcmul = foreach_complex_compound_reference(
        "aten::_foreach_addcmul.Scalar", [base], (left[:2], right[:2], 1), {}
    )
    _assert_exact(addcmul[0], expected["foreach_addcmul"])


@pytest.mark.oracle_contract(id="cp-complex_loss-remaining-core", validation_class="V1_FIXED_VALUE")
def test_bce_with_logits_matches_high_precision_stable_formula():
    case = _case("cp-complex_loss-remaining-core")
    inputs, expected = case["inputs"], case["expected"]
    args = (
        decode_tensor(inputs["logits"]),
        decode_tensor(inputs["target"]),
        decode_tensor(inputs["weight"]),
        decode_tensor(inputs["pos_weight"]),
    )
    ceiling = case["comparison"]["ulp_ceiling"]
    for reduction in ("none", "sum", "mean"):
        _assert_ulp(
            binary_cross_entropy_with_logits_reference(*args, reduction=reduction),
            expected[f"expected_{reduction}"],
            ceiling,
        )


@pytest.mark.oracle_contract(id="cp-complex_unary-remaining-core", validation_class="V1_FIXED_VALUE")
def test_complex_unary_references_match_high_precision_records():
    case = _case("cp-complex_unary-remaining-core")
    value = decode_tensor(case["inputs"]["input"])
    ceiling = case["comparison"]["ulp_ceiling"]
    for function, name in (
        (complex_sigmoid_reference, "sigmoid"),
        (complex_rsqrt_reference, "rsqrt"),
        (complex_expm1_reference, "expm1"),
    ):
        _assert_ulp(function(value), case["expected"][name], ceiling)


@pytest.mark.oracle_contract(id="cp-ldexp_cumprod-remaining-core", validation_class="V1_FIXED_VALUE")
def test_ldexp_and_cumprod_references_match_independent_records():
    case = _case("cp-ldexp_cumprod-remaining-core")
    inputs, expected = case["inputs"], case["expected"]
    value = decode_tensor(inputs["input"])
    ceiling = case["comparison"]["ulp_ceiling"]
    _assert_ulp(
        complex_ldexp_reference(value, decode_tensor(inputs["complex_exponent"])),
        expected["complex_expected"],
        ceiling,
    )
    _assert_exact(
        complex_integral_ldexp_reference(value, decode_tensor(inputs["integral_exponent"])),
        expected["integral_expected"],
    )
    _assert_exact(complex_cumprod_reference(value, 0), expected["cumprod_expected"])
    _assert_exact(
        wide_ldexp_reference(
            decode_tensor(inputs["wide_input"]), decode_tensor(inputs["wide_exponent"])
        ),
        expected["wide_expected"],
    )


@pytest.mark.oracle_contract(id="cp-grad_cov-remaining-core", validation_class="V1_FIXED_VALUE")
def test_complex_gradient_and_covariance_match_exact_formula_records():
    case = _case("cp-grad_cov-remaining-core")
    inputs, expected = case["inputs"], case["expected"]
    gradient = complex_gradient_reference(
        decode_tensor(inputs["gradient_input"]),
        spacing=inputs["spacing"],
        edge_order=inputs["edge_order"],
    )
    assert len(gradient) == 1
    _assert_exact(gradient[0], expected["gradient_expected"])
    _assert_exact(
        complex_covariance_reference(
            decode_tensor(inputs["covariance_input"]), correction=inputs["correction"]
        ),
        expected["covariance_expected"],
    )


@pytest.mark.oracle_contract(id="cp-lanczos-remaining-core", validation_class="V1_FIXED_VALUE")
def test_lanczos_references_match_arbitrary_precision_sinc_records():
    case = _case("cp-lanczos-remaining-core")
    inputs, expected = case["inputs"], case["expected"]
    distances = decode_tensor(inputs["distances"])
    coefficients = torch.tensor(
        [lanczos3_coefficient(float(value)) for value in distances], dtype=torch.float64
    )
    ceiling = case["comparison"]["ulp_ceiling"]
    _assert_ulp(coefficients, expected["coefficients"], ceiling)
    _assert_ulp(
        lanczos2d_aa_reference(
            decode_tensor(inputs["input"]),
            inputs["output_size"],
            inputs["align_corners"],
        ),
        expected["output"],
        ceiling,
    )


@pytest.mark.oracle_contract(id="cp-grid-remaining-core", validation_class="V1_FIXED_VALUE")
def test_grid_references_match_manual_bilinear_records():
    case = _case("cp-grid-remaining-core")
    inputs, expected = case["inputs"], case["expected"]
    input_tensor = decode_tensor(inputs["input"])
    grid = decode_tensor(inputs["grid"])
    args = (
        inputs["interpolation_mode"], inputs["padding_mode"], inputs["align_corners"]
    )
    _assert_exact(grid_sampler_forward_f32_reference(input_tensor, grid, *args), expected["forward"])
    grad_input, grad_grid = grid_sampler_backward_f32_reference(
        decode_tensor(inputs["grad_output"]), input_tensor, grid, *args
    )
    _assert_exact(grad_input, expected["grad_input"])
    _assert_exact(grad_grid, expected["grad_grid"])


@pytest.mark.oracle_contract(id="cp-conv-remaining-core", validation_class="V1_FIXED_VALUE")
def test_transpose_convolution_references_match_scatter_records():
    case = _case("cp-conv-remaining-core")
    inputs, expected = case["inputs"], case["expected"]
    _assert_exact(
        conv_transpose_f32_reference(
            "nn.functional.conv_transpose1d",
            decode_tensor(inputs["transpose1d_input"]),
            decode_tensor(inputs["transpose1d_weight"]),
            decode_tensor(inputs["transpose1d_bias"]),
            **inputs["transpose1d_kwargs"],
        ),
        expected["transpose1d_expected"],
    )
    _assert_exact(
        conv_transpose3d_f32_reference(
            decode_tensor(inputs["transpose3d_input"]),
            decode_tensor(inputs["transpose3d_weight"]),
            decode_tensor(inputs["transpose3d_bias"]),
            **inputs["transpose3d_kwargs"],
        ),
        expected["transpose3d_expected"],
    )


@pytest.mark.oracle_contract(id="cp-matrixexp-remaining-core", validation_class="V1_FIXED_VALUE")
def test_matrix_exp_reference_matches_high_precision_diagonal_record():
    case = _case("cp-matrixexp-remaining-core")
    _assert_ulp(
        matrix_exp_f64_reference(decode_tensor(case["inputs"]["input"])),
        case["expected"]["expected"],
        case["comparison"]["ulp_ceiling"],
    )


@pytest.mark.oracle_contract(id="cp-special-remaining-core", validation_class="V1_FIXED_VALUE")
def test_high_precision_references_match_frozen_mpmath_records():
    case = _case("cp-special-remaining-core")
    inputs, expected = case["inputs"], case["expected"]
    ceiling = case["comparison"]["ulp_ceiling"]
    checks = [
        (
            dirichlet_grad_reference(
                decode_tensor(inputs["dirichlet_x"]),
                decode_tensor(inputs["dirichlet_alpha"]),
                decode_tensor(inputs["dirichlet_total"]),
            ),
            expected["dirichlet_expected"],
        ),
        (
            standard_gamma_grad_reference(
                decode_tensor(inputs["gamma_alpha"]),
                decode_tensor(inputs["gamma_output"]),
            ),
            expected["gamma_grad_expected"],
        ),
        (i0_reference(decode_tensor(inputs["i0_input"])), expected["i0_expected"]),
        (
            polygamma_reference(
                inputs["polygamma_order"], decode_tensor(inputs["polygamma_input"])
            ),
            expected["polygamma_expected"],
        ),
        (
            regularized_gamma_reference(
                decode_tensor(inputs["regularized_shape"]),
                decode_tensor(inputs["regularized_value"]),
                upper=False,
            ),
            expected["regularized_lower"],
        ),
        (
            regularized_gamma_reference(
                decode_tensor(inputs["regularized_shape"]),
                decode_tensor(inputs["regularized_value"]),
                upper=True,
            ),
            expected["regularized_upper"],
        ),
        (
            kaiser_window_reference(
                inputs["kaiser_length"],
                inputs["kaiser_periodic"],
                inputs["kaiser_beta"],
                torch.float64,
            ),
            expected["kaiser_expected"],
        ),
    ]
    for result, expected_record in checks:
        _assert_ulp(result, expected_record, ceiling)
