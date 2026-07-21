import pytest
import torch

from tests.oracles.schema import decode_tensor, encode_tensor
from tests.oracles.test_fixed_values import _case
from torchcts.core.reference_oracles import (
    cast_public_result,
    col2im_reference,
    dynamic_int4_dequantize_reference,
    dynamic_int4_matmul_reference,
    histc_integer_count_reference,
    im2col_reference,
    laguerre_polynomial_reference,
    linear_backward_reference,
    logit_backward_reference,
    matmul_family_determinate_reference,
    matmul_family_reference,
    max_pool2d_backward_reference,
    pack_int4_values,
    quantized_opmath_tensor,
    saturate_weight_to_fp16_reference,
    segment_reduce_prod_backward_reference,
    segment_reduce_prod_reference,
    shifted_chebyshev_polynomial_reference,
    soft_margin_loss_backward_reference,
    soft_margin_loss_reference,
    tinygemm_int4_dequantize_reference,
    tinygemm_int4_matmul_reference,
    unpack_dynamic_int4_weight_bytes,
    unpack_int4_values,
    weight_int8pack_mm_reference,
)


def _assert_exact(result, expected):
    assert encode_tensor(result) == expected


def _ordered_float32_bits(tensor):
    unsigned = tensor.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return torch.where(
        (unsigned & 0x80000000) != 0,
        0xFFFFFFFF - unsigned,
        unsigned + 0x80000000,
    )


def _assert_float32_ulp(result, expected_record, ceiling):
    expected = decode_tensor(expected_record)
    assert result.dtype == expected.dtype == torch.float32
    assert result.shape == expected.shape
    distance = (_ordered_float32_bits(result) - _ordered_float32_bits(expected)).abs()
    assert int(distance.max()) <= ceiling


@pytest.mark.oracle_contract(id="cp-cast-exact-core", validation_class="V1_FIXED_VALUE")
def test_cast_helpers_match_independently_rounded_records():
    case = _case("cp-cast-exact-core")
    inputs, expected = case["inputs"], case["expected"]
    _assert_exact(
        quantized_opmath_tensor(decode_tensor(inputs["opmath_input"])),
        expected["opmath_expected"],
    )
    _assert_exact(
        cast_public_result(decode_tensor(inputs["public_input"]), torch.bfloat16),
        expected["public_bfloat16_expected"],
    )
    _assert_exact(
        saturate_weight_to_fp16_reference(decode_tensor(inputs["saturation_input"])),
        expected["saturation_expected"],
    )


@pytest.mark.oracle_contract(id="cp-matmul-exact-core", validation_class="V1_FIXED_VALUE")
def test_matmul_references_match_exact_complex_term_expansion():
    case = _case("cp-matmul-exact-core")
    inputs, expected = case["inputs"], case["expected"]
    args = (decode_tensor(inputs["left"]), decode_tensor(inputs["right"]))
    _assert_exact(
        matmul_family_reference(inputs["dispatcher"], args),
        expected["expected"],
    )
    value, determinate = matmul_family_determinate_reference(inputs["dispatcher"], args)
    _assert_exact(value, expected["expected"])
    _assert_exact(determinate, expected["determinate_mask"])


@pytest.mark.oracle_contract(id="cp-softmargin-exact-core", validation_class="V1_FIXED_VALUE")
def test_softmargin_references_match_high_precision_formula_records():
    case = _case("cp-softmargin-exact-core")
    inputs, expected = case["inputs"], case["expected"]
    value = decode_tensor(inputs["input"])
    target = decode_tensor(inputs["target"])
    grad = decode_tensor(inputs["grad"])
    ceiling = case["comparison"]["ulp_ceiling"]
    for reduction in ("none", "sum", "mean"):
        _assert_float32_ulp(
            soft_margin_loss_reference(value, target, reduction),
            expected[f"loss_{reduction}"],
            ceiling,
        )
    for reduction in ("none", "mean"):
        _assert_float32_ulp(
            soft_margin_loss_backward_reference(grad, value, target, reduction),
            expected[f"backward_{reduction}"],
            ceiling,
        )


@pytest.mark.oracle_contract(id="cp-segment-exact-core", validation_class="V1_FIXED_VALUE")
def test_segment_product_references_match_zero_aware_exact_records():
    case = _case("cp-segment-exact-core")
    inputs, expected = case["inputs"], case["expected"]
    data = decode_tensor(inputs["data"])
    grad = decode_tensor(inputs["grad"])
    initial = inputs["initial"]
    for boundaries in ("lengths", "offsets"):
        kwargs = {boundaries: decode_tensor(inputs[boundaries]), "initial": initial}
        _assert_exact(segment_reduce_prod_reference(data, **kwargs), expected["expected_forward"])
        _assert_exact(
            segment_reduce_prod_backward_reference(grad, data, **kwargs),
            expected["expected_backward"],
        )
    no_zero = decode_tensor(inputs["no_zero_data"])
    no_zero_lengths = decode_tensor(inputs["no_zero_lengths"])
    no_zero_grad = torch.ones_like(decode_tensor(expected["no_zero_forward"]))
    _assert_exact(
        segment_reduce_prod_reference(no_zero, lengths=no_zero_lengths),
        expected["no_zero_forward"],
    )
    _assert_exact(
        segment_reduce_prod_backward_reference(
            no_zero_grad, no_zero, lengths=no_zero_lengths
        ),
        expected["no_zero_backward"],
    )


@pytest.mark.oracle_contract(id="cp-linear_bwd-exact-core", validation_class="V1_FIXED_VALUE")
def test_linear_backward_reference_matches_manual_jacobian_products():
    case = _case("cp-linear_bwd-exact-core")
    inputs, expected = case["inputs"], case["expected"]
    gradients = linear_backward_reference(
        decode_tensor(inputs["input"]),
        decode_tensor(inputs["grad_output"]),
        decode_tensor(inputs["weight"]),
        decode_tensor(inputs["bias"]),
    )
    for result, name in zip(gradients, ("input", "weight", "bias")):
        _assert_exact(result, expected[f"expected_{name}_grad"])


@pytest.mark.oracle_contract(id="cp-pool-exact-core", validation_class="V1_FIXED_VALUE")
def test_pool_backward_reference_matches_frozen_unique_winners():
    case = _case("cp-pool-exact-core")
    inputs = case["inputs"]
    result = max_pool2d_backward_reference(
        decode_tensor(inputs["input"]),
        decode_tensor(inputs["grad_output"]),
        **inputs["parameters"],
    )
    _assert_exact(result, case["expected"]["expected_input_grad"])


@pytest.mark.oracle_contract(id="cp-int4-exact-core", validation_class="V1_FIXED_VALUE")
def test_int4_references_match_nibble_and_affine_formula_records():
    case = _case("cp-int4-exact-core")
    inputs, expected = case["inputs"], case["expected"]
    logical = decode_tensor(inputs["logical"])
    input_tensor = decode_tensor(inputs["input"])
    qparams = decode_tensor(inputs["qparams"])
    group_size = inputs["group_size"]
    for high, expected_name in ((True, "packed_even_high"), (False, "packed_even_low")):
        packed = pack_int4_values(logical, even_k_in_high_bits=high)
        _assert_exact(packed, expected[expected_name])
        _assert_exact(
            unpack_int4_values(packed, even_k_in_high_bits=high),
            inputs["logical"],
        )
    _assert_exact(
        tinygemm_int4_dequantize_reference(logical, qparams, group_size),
        expected["expected_dequant"],
    )
    _assert_exact(
        tinygemm_int4_matmul_reference(input_tensor, logical, qparams, group_size),
        expected["expected_matmul"],
    )
    dynamic_packed = decode_tensor(inputs["dynamic_packed"])
    dynamic_scales = decode_tensor(inputs["dynamic_scales"])
    dynamic_bias = decode_tensor(inputs["dynamic_bias"])
    out_features, in_features = logical.shape
    dynamic_kwargs = {
        "block_size": group_size,
        "in_features": in_features,
        "out_features": out_features,
    }
    _assert_exact(
        unpack_dynamic_int4_weight_bytes(
            dynamic_packed, in_features=in_features, out_features=out_features
        ),
        expected["dynamic_unpacked"],
    )
    _assert_exact(
        dynamic_int4_dequantize_reference(dynamic_packed, dynamic_scales, **dynamic_kwargs),
        expected["expected_dynamic_dequant"],
    )
    _assert_exact(
        dynamic_int4_matmul_reference(
            input_tensor, dynamic_packed, dynamic_scales, dynamic_bias, **dynamic_kwargs
        ),
        expected["expected_dynamic_matmul"],
    )


@pytest.mark.oracle_contract(id="cp-quant-exact-core", validation_class="V1_FIXED_VALUE")
def test_int8pack_reference_matches_exact_dot_product_record():
    case = _case("cp-quant-exact-core")
    inputs = case["inputs"]
    result = weight_int8pack_mm_reference(
        decode_tensor(inputs["input"]),
        decode_tensor(inputs["weight"]),
        decode_tensor(inputs["scales"]),
    )
    _assert_exact(result, case["expected"]["expected"])


@pytest.mark.oracle_contract(id="cp-histc-exact-core", validation_class="V1_FIXED_VALUE")
def test_histogram_reference_matches_boundary_inclusion_records():
    case = _case("cp-histc-exact-core")
    inputs = case["inputs"]
    for bins in (1, 2, 4):
        result = histc_integer_count_reference(
            decode_tensor(inputs["input"]), bins, inputs["minimum"], inputs["maximum"]
        )
        _assert_exact(result, case["expected"][f"bins_{bins}"])


@pytest.mark.oracle_contract(id="cp-im2col-exact-core", validation_class="V1_FIXED_VALUE")
def test_im2col_references_match_index_mapping_records():
    case = _case("cp-im2col-exact-core")
    inputs, expected = case["inputs"], case["expected"]
    parameters = inputs["parameters"]
    columns = im2col_reference(decode_tensor(inputs["input"]), **parameters)
    _assert_exact(columns, expected["columns"])
    _assert_exact(
        col2im_reference(columns, inputs["output_size"], **parameters),
        expected["reconstructed"],
    )


@pytest.mark.oracle_contract(id="cp-logit-exact-core", validation_class="V1_FIXED_VALUE")
def test_logit_backward_reference_matches_high_precision_formula_records():
    case = _case("cp-logit-exact-core")
    inputs, expected = case["inputs"], case["expected"]
    grad = decode_tensor(inputs["grad"])
    value = decode_tensor(inputs["input"])
    ceiling = case["comparison"]["ulp_ceiling"]
    _assert_float32_ulp(logit_backward_reference(grad, value, None), expected["expected_none"], ceiling)
    _assert_float32_ulp(
        logit_backward_reference(grad, value, inputs["epsilon"]),
        expected["expected_eps"],
        ceiling,
    )


@pytest.mark.oracle_contract(id="cp-polynomial-exact-core", validation_class="V1_FIXED_VALUE")
def test_polynomial_references_match_symbolic_recurrence_records():
    case = _case("cp-polynomial-exact-core")
    inputs, expected = case["inputs"], case["expected"]
    value = decode_tensor(inputs["input"])
    degree = decode_tensor(inputs["degrees"])
    _assert_exact(laguerre_polynomial_reference(value, degree), expected["laguerre"])
    for family in ("t", "u", "v", "w"):
        _assert_exact(
            shifted_chebyshev_polynomial_reference(family, value, degree),
            expected["shifted"][family],
        )
