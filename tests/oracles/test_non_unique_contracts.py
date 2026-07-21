import pytest
import torch

from tests.oracles.test_fixed_values import _case
from torchcts.core.non_unique_output_compare import (
    NON_UNIQUE_OUTPUT_CONTRACTS,
    compare_non_unique_output_if_applicable,
    contract_for_op_name,
)


def _compare(actual, expected, **_metadata):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.oracle_contract(id="cp-non-unique-contracts", validation_class="V2_ADMISSIBILITY")
def test_all_non_unique_contracts_match_reviewed_dispositions_and_routes():
    case = _case("cp-non-unique-contracts")
    reviewed = case["admissible_results"]["contracts"]
    runtime = {contract.family: contract for contract in NON_UNIQUE_OUTPUT_CONTRACTS}
    assert len(reviewed) == len(runtime) == 28
    for record in reviewed:
        contract = runtime[record["family"]]
        assert contract.ambiguity_type == record["ambiguity_type"]
        assert contract.status == record["status"]
        assert (contract.checker.__name__ if contract.checker else None) == record["checker"]
        assert contract_for_op_name(record["representative_op"]).family == record["family"]


@pytest.mark.oracle_contract(id="cp-non-unique-contracts", validation_class="V2_ADMISSIBILITY")
def test_representative_legal_alternates_are_accepted_by_contract_family():
    tied = torch.tensor([5.0, 5.0, 1.0])
    assert compare_non_unique_output_if_applicable(
        "sort",
        (torch.tensor([1.0, 5.0, 5.0]), torch.tensor([2, 1, 0])),
        (torch.tensor([1.0, 5.0, 5.0]), torch.tensor([2, 0, 1])),
        input=tied,
        kwargs={"dim": 0, "stable": False},
        category="default",
        dtype=torch.float32,
        compare=_compare,
    )
    assert compare_non_unique_output_if_applicable(
        "argmax",
        torch.tensor(1),
        torch.tensor(0),
        input=tied,
        category="default",
        dtype=torch.float32,
        compare=_compare,
    )

    modal = torch.tensor([1.0, 1.0, 2.0, 2.0])
    assert compare_non_unique_output_if_applicable(
        "mode",
        (torch.tensor(2.0), torch.tensor(2)),
        (torch.tensor(1.0), torch.tensor(0)),
        input=modal,
        kwargs={"dim": 0},
        category="default",
        dtype=torch.float32,
        compare=_compare,
    )

    pool_input = torch.tensor([[[5.0, 5.0]]])
    assert compare_non_unique_output_if_applicable(
        "max_pool1d",
        (torch.tensor([[[5.0]]]), torch.tensor([[[1]]])),
        (torch.tensor([[[5.0]]]), torch.tensor([[[0]]])),
        input=pool_input,
        category="default",
        dtype=torch.float32,
        compare=_compare,
    )
    writers = torch.tensor([[[1.0, 2.0]]])
    destinations = torch.tensor([[[0, 0]]])
    assert compare_non_unique_output_if_applicable(
        "max_unpool1d",
        torch.tensor([[[2.0, 0.0]]]),
        torch.tensor([[[1.0, 0.0]]]),
        input=writers,
        args=(destinations,),
        category="default",
        dtype=torch.float32,
        compare=_compare,
    )

    rrelu_input = torch.tensor([-2.0, 0.0, 3.0])
    assert compare_non_unique_output_if_applicable(
        "nn.functional.rrelu",
        torch.tensor([-0.5, 0.0, 3.0]),
        torch.tensor([-0.4, 0.0, 3.0]),
        input=rrelu_input,
        args=(0.1, 0.3, True),
        category="default",
        dtype=torch.float32,
        compare=_compare,
    )
    assert compare_non_unique_output_if_applicable(
        "rand",
        torch.tensor([0.25, 0.75]),
        torch.tensor([0.1, 0.2]),
        category="default",
        dtype=torch.float32,
        compare=_compare,
    )
    assert compare_non_unique_output_if_applicable(
        "empty",
        torch.tensor([float("nan"), 123.0]),
        torch.zeros(2),
        category="default",
        dtype=torch.float32,
        compare=_compare,
    )


@pytest.mark.oracle_contract(id="cp-non-unique-contracts", validation_class="V2_ADMISSIBILITY")
def test_linalg_contract_accepts_sign_equivalent_qr_and_svd_factors():
    matrix = torch.diag(torch.tensor([3.0, 2.0]))
    identity = torch.eye(2)
    negative_identity = -identity
    assert compare_non_unique_output_if_applicable(
        "linalg.qr",
        (negative_identity, -matrix),
        (identity, matrix),
        input=matrix,
        category="default",
        dtype=torch.float32,
        compare=_compare,
    )
    singular_values = torch.tensor([3.0, 2.0])
    assert compare_non_unique_output_if_applicable(
        "linalg.svd",
        (negative_identity, singular_values, negative_identity),
        (identity, singular_values, identity),
        input=matrix,
        category="default",
        dtype=torch.float32,
        compare=_compare,
    )
