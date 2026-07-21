import copy
import json
from pathlib import Path
import shutil

import pytest

from tests.oracles.schema import (
    CaseRecord,
    FixtureValidationError,
    load_case_manifest,
    load_generator_catalog,
    load_source_catalog,
    validate_then_run,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CASES_ROOT = REPO_ROOT / "tests/oracles/cases"
MANIFEST = CASES_ROOT / "manifest.json"
EVIDENCE_ROOT = REPO_ROOT / "evidence/oracles"
SOURCES = EVIDENCE_ROOT / "sources.json"


@pytest.mark.oracle_contract(id="fixture-provenance-complete", validation_class="V4_PROPERTY")
def test_accepted_manifest_and_source_catalog_validate_completely():
    cases = load_case_manifest(
        MANIFEST,
        source_catalog_path=SOURCES,
        repo_root=REPO_ROOT,
        evidence_root=EVIDENCE_ROOT,
    )
    sources = load_source_catalog(SOURCES, evidence_root=EVIDENCE_ROOT)
    assert [case.case_id for case in cases] == [
        "cp-cast-exact-core",
        "cp-complex-arithmetic-migration",
        "cp-complex-convolution-migration",
        "cp-complex-loss-l1-migration",
        "cp-complex-unary-log2-migration",
        "cp-complex_arith-remaining-core",
        "cp-complex_loss-remaining-core",
        "cp-complex_unary-remaining-core",
        "cp-conv-remaining-core",
        "cp-direct-dispositions",
        "cp-embedding-frequency-migration",
        "cp-fft-contracts",
        "cp-grad_cov-remaining-core",
        "cp-grid-3d-backward-migration",
        "cp-grid-remaining-core",
        "cp-histc-exact-core",
        "cp-im2col-exact-core",
        "cp-int4-exact-core",
        "cp-integer-c17",
        "cp-lanczos-remaining-core",
        "cp-ldexp_cumprod-remaining-core",
        "cp-linalg_bwd-backward",
        "cp-linear_bwd-exact-core",
        "cp-logit-exact-core",
        "cp-matmul-exact-core",
        "cp-matrixexp-remaining-core",
        "cp-non-unique-contracts",
        "cp-norm_sparse_bwd-backward",
        "cp-polynomial-exact-core",
        "cp-pool-exact-core",
        "cp-quant-exact-core",
        "cp-routing-backward",
        "cp-segment-exact-core",
        "cp-softmargin-exact-core",
        "cp-special-remaining-core",
    ]
    assert "SRC-IEEE-C" in sources
    assert "SRC-PYTORCH-SOURCE" in sources


@pytest.mark.oracle_contract(id="fixture-digest-fail-closed", validation_class="V4_PROPERTY")
def test_digest_corruption_stops_before_candidate_execution(tmp_path):
    copied_cases = tmp_path / "cases"
    shutil.copytree(CASES_ROOT, copied_cases)
    case_path = copied_cases / "reference/cp-integer-c17.json"
    case_path.write_text(case_path.read_text(encoding="ascii") + "\n", encoding="ascii")
    calls = []

    with pytest.raises(FixtureValidationError, match="file digest mismatch"):
        validate_then_run(
            copied_cases / "manifest.json",
            source_catalog_path=SOURCES,
            repo_root=REPO_ROOT,
            evidence_root=EVIDENCE_ROOT,
            candidate=calls.append,
        )

    assert calls == []


@pytest.mark.oracle_contract(id="fixture-source-authority", validation_class="V4_PROPERTY")
def test_comparator_cannot_assert_a_field_no_source_authorizes():
    sources = load_source_catalog(SOURCES, evidence_root=EVIDENCE_ROOT)
    generators = load_generator_catalog(
        EVIDENCE_ROOT / "generators.json", repo_root=REPO_ROOT
    )
    payload = json.loads((CASES_ROOT / "reference/cp-integer-c17.json").read_text(encoding="ascii"))
    payload["comparison"]["asserts"].append("vendor_opaque_bytes")

    with pytest.raises(FixtureValidationError, match="lack source authority"):
        CaseRecord.from_mapping(
            payload,
            sources=sources,
            generator_paths=generators,
            repo_root=REPO_ROOT,
            evidence_root=EVIDENCE_ROOT,
        )


@pytest.mark.oracle_contract(id="fixture-orphan-rejection", validation_class="V4_PROPERTY")
def test_unlisted_fixture_is_rejected(tmp_path):
    copied_cases = tmp_path / "cases"
    shutil.copytree(CASES_ROOT, copied_cases)
    orphan = copied_cases / "reference/orphan.json"
    orphan.write_text("{}\n", encoding="ascii")

    with pytest.raises(FixtureValidationError, match="orphan fixture"):
        load_case_manifest(
            copied_cases / "manifest.json",
            source_catalog_path=SOURCES,
            repo_root=REPO_ROOT,
            evidence_root=EVIDENCE_ROOT,
        )
