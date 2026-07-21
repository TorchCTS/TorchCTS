import pytest

from scripts.oracle_fixtures.oracle_inventory import (
    DIRECT_ORACLE_CASE_PACKS,
    direct_validation_class,
)
from tests.oracles.test_fixed_values import _case
from torchcts.core.oracles import all_oracle_specs, oracle_spec_for


def _record(spec):
    metadata = spec.metadata()
    return {
        "surface": spec.surface,
        "oracle_id": spec.oracle_id,
        "coverage_status": spec.coverage_status,
        "coverage_kind": spec.coverage_kind,
        "runner": spec.runner,
        "backend_gate": spec.backend_gate,
        "contract_status": metadata["contract_status"],
        "contract_ref": spec.contract_ref,
        "case_pack": DIRECT_ORACLE_CASE_PACKS[spec.oracle_id],
        "validation_class": direct_validation_class(spec.oracle_id, spec.coverage_status),
    }


@pytest.mark.oracle_contract(id="cp-direct-dispositions", validation_class="V6_DISPOSITION")
def test_every_direct_surface_matches_its_reviewed_disposition():
    case = _case("cp-direct-dispositions")
    reviewed = case["admissible_results"]["records"]
    runtime = [_record(spec) for spec in all_oracle_specs()]
    assert runtime == reviewed
    assert len(runtime) == case["admissible_results"]["counts"]["surfaces"] == 243
    assert len({row["oracle_id"] for row in runtime}) == 63
    assert len({
        (
            row["oracle_id"], row["coverage_status"], row["runner"],
            row["backend_gate"], row["contract_ref"],
        )
        for row in runtime
    }) == 70
    assert len({row["surface"] for row in runtime}) == len(runtime)
    for row in runtime:
        assert _record(oracle_spec_for(row["surface"])) == row


@pytest.mark.oracle_contract(id="cp-direct-dispositions", validation_class="V6_DISPOSITION")
def test_pending_and_excluded_direct_surfaces_cannot_masquerade_as_accepted():
    records = _case("cp-direct-dispositions")["admissible_results"]["records"]
    for row in records:
        if row["coverage_status"].startswith(("pending_", "excluded_")):
            assert row["validation_class"] == "V6_DISPOSITION"
            assert row["contract_status"] in {"candidate", "blocked", "excluded"}
        else:
            assert row["coverage_status"].startswith("covered_")
            assert row["contract_status"] == "accepted"
