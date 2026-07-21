import pytest
import torch

from tests.oracles.schema import decode_tensor, encode_tensor
from tests.oracles.test_fixed_values import _case
from tests.oracles.test_remaining_core import _assert_ulp
from torchcts.core.backward_references import (
    complex_rpow_backward_reference,
    condition_number_backward_reference,
    group_norm_backward_reference,
    has_opinfo_backward_reference,
    renorm_inf_backward_reference,
    sampled_addmm_backward_reference,
    vander_backward_reference,
)


def _assert_exact(result, expected):
    assert encode_tensor(result) == expected


@pytest.mark.oracle_contract(id="cp-linalg_bwd-backward", validation_class="V1_FIXED_VALUE")
def test_linalg_backward_references_match_matrix_calculus_records():
    case = _case("cp-linalg_bwd-backward")
    inputs, expected = case["inputs"], case["expected"]
    _assert_exact(
        vander_backward_reference(
            decode_tensor(inputs["vander_input"]),
            decode_tensor(inputs["vander_grad"]),
            columns=inputs["vander_columns"],
            increasing=inputs["vander_increasing"],
        ),
        expected["vander_expected"],
    )
    exponent_grad, base_grad = complex_rpow_backward_reference(
        decode_tensor(inputs["rpow_exponent"]),
        decode_tensor(inputs["rpow_base"]),
        decode_tensor(inputs["rpow_grad"]),
    )
    ceiling = case["comparison"]["ulp_ceiling"]
    _assert_ulp(exponent_grad, expected["rpow_exponent_expected"], ceiling)
    _assert_ulp(base_grad, expected["rpow_base_expected"], ceiling)
    _assert_exact(
        condition_number_backward_reference(
            decode_tensor(inputs["cond_input"]),
            inputs["cond_p"],
            decode_tensor(inputs["cond_grad"]),
        ),
        expected["cond_expected"],
    )


@pytest.mark.oracle_contract(id="cp-norm_sparse_bwd-backward", validation_class="V1_FIXED_VALUE")
def test_norm_and_sparse_backward_references_match_formula_records():
    case = _case("cp-norm_sparse_bwd-backward")
    inputs, expected = case["inputs"], case["expected"]
    ceiling = case["comparison"]["ulp_ceiling"]
    input_grad, weight_grad, bias_grad = group_norm_backward_reference(
        decode_tensor(inputs["group_input"]),
        inputs["group_count"],
        decode_tensor(inputs["group_weight"]),
        decode_tensor(inputs["group_grad"]),
        inputs["group_eps"],
    )
    _assert_ulp(input_grad, expected["group_input_expected"], ceiling)
    _assert_ulp(weight_grad, expected["group_weight_expected"], ceiling)
    _assert_ulp(bias_grad, expected["group_bias_expected"], ceiling)

    with torch.sparse.check_sparse_tensor_invariants():
        sparse = torch.sparse_csr_tensor(
            decode_tensor(inputs["sampled_crow"]),
            decode_tensor(inputs["sampled_col"]),
            decode_tensor(inputs["sampled_values"]),
            size=inputs["sampled_shape"],
        )
    gradients = sampled_addmm_backward_reference(
        sparse,
        decode_tensor(inputs["sampled_matrix1"]),
        decode_tensor(inputs["sampled_matrix2"]),
        alpha=inputs["sampled_alpha"],
        beta=inputs["sampled_beta"],
    )
    for result, name in zip(gradients, ("input", "left", "right")):
        _assert_exact(result, expected[f"sampled_{name}_expected"])

    renorm = renorm_inf_backward_reference(
        decode_tensor(inputs["renorm_input"]),
        inputs["renorm_dim"],
        inputs["renorm_maxnorm"],
        decode_tensor(inputs["renorm_grad"]),
    )
    _assert_ulp(renorm, expected["renorm_expected"], ceiling)


@pytest.mark.oracle_contract(id="cp-routing-backward", validation_class="V3_ROUTING")
def test_backward_reference_availability_matches_reviewed_routing_table():
    case = _case("cp-routing-backward")
    expected = case["expected"]["has_reference"]
    for row, wanted in zip(case["inputs"]["has_reference"], expected):
        dtype = getattr(torch, row["dtype"].removeprefix("torch."))
        assert has_opinfo_backward_reference(row["op"], dtype) is wanted
