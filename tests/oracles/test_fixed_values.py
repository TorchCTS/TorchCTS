from pathlib import Path
from functools import lru_cache

import pytest
import torch

from tests.oracles.schema import decode_tensor, encode_tensor, load_case_manifest
from torchcts.core.reference_oracles import (
    complex_convolution_reference,
    complex_l1_loss_reference,
    complex_log2_reference,
    complex_tensor_integer_power_reference,
    complex_unit_alpha_add_sub_reference,
    embedding_bag_scale_grad_by_freq_reference,
    float_to_uint8_reference,
    gcd_integer_reference,
    grid_sampler_3d_backward_f32_reference,
    slow_complex_convolution_reference,
    unsigned_negation_reference,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = REPO_ROOT / "evidence/oracles"


@lru_cache(maxsize=1)
def _cases():
    cases = load_case_manifest(
        REPO_ROOT / "tests/oracles/cases/manifest.json",
        source_catalog_path=EVIDENCE_ROOT / "sources.json",
        repo_root=REPO_ROOT,
        evidence_root=EVIDENCE_ROOT,
    )
    return {case.case_id: case.payload for case in cases}


def _case(case_id):
    return _cases()[case_id]


@pytest.mark.oracle_contract(id="cp-integer-c17", validation_class="V1_FIXED_VALUE")
def test_integer_references_match_frozen_c17_records():
    case = _case("cp-integer-c17")
    inputs = case["inputs"]
    expected = case["expected"]

    byte_result = float_to_uint8_reference(decode_tensor(inputs["float_to_uint8"]))
    assert encode_tensor(byte_result) == expected["float_to_uint8"]

    gcd_result = gcd_integer_reference(
        decode_tensor(inputs["gcd_left"]),
        decode_tensor(inputs["gcd_right"]),
    )
    assert encode_tensor(gcd_result) == expected["gcd"]

    for dtype_name, input_record in inputs["unsigned"].items():
        result = unsigned_negation_reference(decode_tensor(input_record))
        assert encode_tensor(result) == expected["unsigned"][dtype_name]


@pytest.mark.oracle_contract(id="cp-complex-arithmetic-migration", validation_class="V1_FIXED_VALUE")
def test_complex_arithmetic_references_match_frozen_lane_records():
    case = _case("cp-complex-arithmetic-migration")
    inputs = case["inputs"]
    expected = case["expected"]
    unit = inputs["unit_alpha"]
    left = decode_tensor(unit["left"])
    right = decode_tensor(unit["right"])

    for operation in ("add", "sub", "rsub"):
        result = complex_unit_alpha_add_sub_reference(operation, left, right)
        assert encode_tensor(result) == expected["unit_alpha"][operation]

    power = inputs["integer_power"]
    result = complex_tensor_integer_power_reference(
        decode_tensor(power["base"]),
        decode_tensor(power["exponent"]),
        decode_tensor(power["native_placeholder"]),
    )
    assert encode_tensor(result) == expected["integer_power"]

    passthrough = inputs["general_power_passthrough"]
    result = complex_tensor_integer_power_reference(
        decode_tensor(passthrough["base"]),
        decode_tensor(passthrough["exponent"]),
        decode_tensor(passthrough["native_sentinel"]),
    )
    assert encode_tensor(result) == expected["general_power_passthrough"]


@pytest.mark.oracle_contract(id="cp-complex-unary-log2-migration", validation_class="V1_FIXED_VALUE")
def test_complex_log2_reference_matches_frozen_symbolic_phase():
    case = _case("cp-complex-unary-log2-migration")
    result = complex_log2_reference(decode_tensor(case["inputs"]["input"]))

    assert encode_tensor(result) == case["expected"]["output"]


@pytest.mark.oracle_contract(id="cp-complex-loss-l1-migration", validation_class="V1_FIXED_VALUE")
def test_complex_l1_reference_matches_frozen_lane_formula():
    case = _case("cp-complex-loss-l1-migration")
    inputs = case["inputs"]
    result = complex_l1_loss_reference(
        decode_tensor(inputs["input"]),
        decode_tensor(inputs["target"]),
        inputs["reduction"],
    )

    assert encode_tensor(result) == case["expected"]["output"]


@pytest.mark.oracle_contract(id="cp-grid-3d-backward-migration", validation_class="V1_FIXED_VALUE")
def test_grid_sampler_backward_reference_matches_exact_rational_vjp():
    case = _case("cp-grid-3d-backward-migration")
    inputs = case["inputs"]
    grad_input, grad_grid = grid_sampler_3d_backward_f32_reference(
        decode_tensor(inputs["grad_output"]),
        decode_tensor(inputs["input"]),
        decode_tensor(inputs["grid"]),
        inputs["interpolation_mode"],
        inputs["padding_mode"],
        inputs["align_corners"],
    )

    assert encode_tensor(grad_input) == case["expected"]["grad_input"]
    assert encode_tensor(grad_grid) == case["expected"]["grad_grid"]


@pytest.mark.oracle_contract(id="cp-embedding-frequency-migration", validation_class="V1_FIXED_VALUE")
def test_embedding_frequency_reference_matches_exact_rational_record_within_one_ulp():
    case = _case("cp-embedding-frequency-migration")
    inputs = case["inputs"]
    result = embedding_bag_scale_grad_by_freq_reference(
        decode_tensor(inputs["grad"]),
        decode_tensor(inputs["indices"]),
        decode_tensor(inputs["offset2bag"]),
        inputs["num_weights"],
    )
    expected = decode_tensor(case["expected"]["output"])

    assert result.dtype == expected.dtype
    assert result.shape == expected.shape
    bit_distance = (
        result.contiguous().view(torch.int32).to(torch.int64)
        - expected.contiguous().view(torch.int32).to(torch.int64)
    ).abs()
    assert int(bit_distance.max()) <= case["comparison"]["ulp_ceiling"]


@pytest.mark.oracle_contract(id="cp-complex-convolution-migration", validation_class="V1_FIXED_VALUE")
def test_complex_convolution_references_match_frozen_scalar_term_map():
    case = _case("cp-complex-convolution-migration")
    inputs = case["inputs"]
    args = (
        inputs["op_name"],
        decode_tensor(inputs["input"]),
        decode_tensor(inputs["weight"]),
        decode_tensor(inputs["bias"]),
        inputs["kwargs"],
    )

    assert encode_tensor(complex_convolution_reference(*args)) == case["expected"]["output"]
    assert encode_tensor(slow_complex_convolution_reference(*args)) == case["expected"]["output"]
