from pathlib import Path

from scripts.oracle_fixtures.inventory import (
    REVIEWED_LOCAL_PATH,
    SNAPSHOT_PATH,
    _canonical,
    _check_reviewed_local,
    build_inventory,
)


EXPECTED_COUNTS = {
    "reference_functions": 56,
    "high_precision_functions": 6,
    "backward_functions": 8,
    "fft_functions": 4,
    "forward_reference_ids": 31,
    "backward_reference_ids": 7,
    "non_unique_contracts": 28,
    "direct_oracle_ids": 63,
    "direct_status_groups": 70,
    "direct_surfaces": 243,
    "call_sites": 102,
    "semantic_marked_tests": 9,
    "suspicious_local_expected": 9,
}


def test_oracle_inventory_matches_reviewed_snapshot():
    inventory = build_inventory()

    assert inventory["counts"] == EXPECTED_COUNTS
    assert not _check_reviewed_local(inventory)
    assert Path(REVIEWED_LOCAL_PATH).is_file()
    assert Path(SNAPSHOT_PATH).read_text(encoding="utf-8") == _canonical(inventory)


def test_every_direct_surface_publishes_validation_metadata():
    inventory = build_inventory()
    direct_specs = inventory["inventory"]["direct_specs"]

    assert len(direct_specs) == 243
    assert all(row["case_pack"].startswith("CP-") for row in direct_specs)
    assert all(row["validation_class"].startswith("V") for row in direct_specs)
