# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies or substantial portions of the Software.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from torchcts.core import backend_evidence_collection as collection
from torchcts.core.backend_evidence import (
    ExpandedBackendEvidence,
    add_source_observations,
    canonical_backend,
    load_backend_evidence_store,
    normalize_observation,
    normalize_source_environment,
    stores_semantically_equal,
    validate_expanded_store,
    write_backend_evidence_store,
)


pytestmark = pytest.mark.covers_category("selftest")


def _source(source_id: str, *backends: str) -> dict:
    return {
        "architecture": "aarch64",
        "backends": sorted(backends),
        "collected_at": "2026-07-02T19:15:35Z",
        "device_capabilities": [],
        "operating_system": {"name": "Linux", "release": "6.11"},
        "python_version": "3.12.3",
        "pytorch_version": "2.12.1+cu130",
        "requested_device": "cuda",
        "runtime_modifications": [],
        "source_id": source_id,
        "torch_build": {"cuda_version": "13.0", "hip_version": None},
        "torchcts_version": "0.1.0",
    }


def _passing_contract(value: int = 1) -> dict:
    return {
        "oracle": {
            "contract_ref": "docs/coverage/contract-evidence.md#example",
            "oracle_id": "example_backend_pack",
            "runner": "example_runner",
        },
        "oracle_result": {"ok": True, "value": value},
    }


def test_store_groups_universal_contracts_and_preserves_reviewed_promotion(tmp_path: Path):
    store = ExpandedBackendEvidence()
    for index in (1, 2):
        source_id = f"source-{index:010d}"
        add_source_observations(
            store,
            source=_source(source_id, "cuda"),
            records=[("cuda", "aten::example", _passing_contract())],
        )
    store.promotions = {"cuda": {"aten::example": {"source-0000000001"}}}

    destination = tmp_path / "backend-evidence"
    write_backend_evidence_store(store, destination)
    record = json.loads((destination / "cuda.jsonl").read_text().strip())

    assert "sources" not in record
    assert record["promotion_sources"] == ["source-0000000001"]
    reread = load_backend_evidence_store(destination, verify_canonical=True)
    assert stores_semantically_equal(store, reread)


def test_focused_source_forces_explicit_scope_without_changing_promotion(tmp_path: Path):
    store = ExpandedBackendEvidence()
    add_source_observations(
        store,
        source=_source("source-0000000001", "cuda"),
        records=[("cuda", "aten::first", _passing_contract())],
    )
    store.promotions = {"cuda": {"aten::first": {"source-0000000001"}}}
    add_source_observations(
        store,
        source=_source("source-0000000002", "cuda"),
        records=[("cuda", "aten::second", _passing_contract(2))],
    )

    destination = tmp_path / "backend-evidence"
    write_backend_evidence_store(store, destination)
    records = [json.loads(line) for line in (destination / "cuda.jsonl").read_text().splitlines()]

    first = next(record for record in records if record["surface"] == "aten::first")
    assert first["sources"] == ["source-0000000001"]
    assert first["promotion_sources"] == ["source-0000000001"]


def test_different_contracts_produce_disjoint_source_scopes(tmp_path: Path):
    store = ExpandedBackendEvidence()
    for index, value in ((1, 1), (2, 2), (3, 1)):
        source_id = f"source-{index:010d}"
        add_source_observations(
            store,
            source=_source(source_id, "cuda"),
            records=[("cuda", "aten::example", _passing_contract(value))],
        )

    destination = tmp_path / "backend-evidence"
    write_backend_evidence_store(store, destination)
    records = [json.loads(line) for line in (destination / "cuda.jsonl").read_text().splitlines()]

    assert len(records) == 2
    assert {tuple(record["sources"]) for record in records} == {
        ("source-0000000001", "source-0000000003"),
        ("source-0000000002",),
    }


def test_writing_regroups_deterministically_and_removes_stale_files(tmp_path: Path):
    stores = []
    for order in ((1, 2), (2, 1)):
        store = ExpandedBackendEvidence()
        for index in order:
            source_id = f"source-{index:010d}"
            add_source_observations(
                store,
                source=_source(source_id, "cuda"),
                records=[("cuda", "aten::example", _passing_contract())],
            )
        stores.append(store)

    first = tmp_path / "first"
    second = tmp_path / "second"
    write_backend_evidence_store(stores[0], first)
    (first / "stale.jsonl").write_text("stale\n")
    write_backend_evidence_store(stores[0], first)
    write_backend_evidence_store(stores[1], second)

    assert not (first / "stale.jsonl").exists()
    assert {
        path.name: path.read_bytes()
        for path in first.iterdir()
    } == {
        path.name: path.read_bytes()
        for path in second.iterdir()
    }


def test_observation_sanitization_removes_machine_local_paths():
    environment = {
        "cwd": "/home/alice/torchcts",
        "hostname": "workstation-alice",
        "python_executable": "/home/alice/torchcts/.venv/bin/python",
    }
    record = {
        "backend_gate": "cuda",
        "dispatch": {
            "dispatch_dump_table": {
                "ok": True,
                "value": (
                    "Meta: registered at /home/alice/torchcts/.venv/lib/python3.12/"
                    "site-packages/torch/_meta_registrations.py:53 [kernel]"
                ),
            }
        },
        "exclusion": {
            "_source_path": "/home/alice/torchcts/torchcts/coverage_exclusions.json",
            "category": "backend_specific_internal",
        },
        "oracle": {
            "promotion_backend": "cuda",
            "promotion_evidence": "legacy-promotion-reference",
        },
        "oracle_result": {
            "error_message": "failed on workstation-alice",
            "ok": False,
            "traceback": (
                '  File "/home/alice/torchcts/torchcts/core/oracles.py", line 10, in run\n'
                '  File "/home/alice/torchcts/.venv/lib/python3.12/site-packages/torch/foo.py", line 2\n'
            ),
        },
        "surface": "aten::example",
    }

    backend, surface, contract = normalize_observation(record, environment)
    serialized = json.dumps(contract, sort_keys=True)

    assert backend == "cuda"
    assert surface == "aten::example"
    assert "/home/" not in serialized
    assert "workstation-alice" not in serialized
    assert ".venv" not in serialized
    assert "_source_path" not in serialized
    assert "torchcts/core/oracles.py" in serialized
    assert "torch/_meta_registrations.py" in serialized


def test_source_normalization_keeps_capabilities_but_drops_machine_identity():
    environment = {
        "cwd": "/home/alice/torchcts",
        "device": {
            "cuda_available": True,
            "cuda_devices": [
                {
                    "index": 0,
                    "major": 8,
                    "minor": 9,
                    "multi_processor_count": 20,
                    "name": "NVIDIA GB10 (LD_PRELOAD fake sm_89)",
                    "total_memory": 128,
                }
            ],
            "cudnn_available": True,
            "cudnn_version": 91000,
            "mps_available": False,
            "requested_device": "cuda",
            "torch_cuda": "13.0",
            "torch_hip": None,
            "torch_version": "2.12.1+cu130",
        },
        "generated_at": "2026-07-02T19:15:35Z",
        "hostname": "workstation-alice",
        "machine": "aarch64",
        "platform": "Linux-6.11.0-aarch64-with-glibc2.39",
        "python": "3.12.3 (main)",
        "python_executable": "/home/alice/torchcts/.venv/bin/python",
        "torch_config": "installed under /home/alice/torchcts/.venv",
        "torchcts_version": "0.1.0",
    }

    source = normalize_source_environment(
        environment,
        source_id="source-0000000001",
        backends=["cuda"],
    )
    serialized = json.dumps(source, sort_keys=True)

    assert source["device_capabilities"][0]["name"] == "NVIDIA GB10"
    assert source["runtime_modifications"] == ["sm89-guard-bypass"]
    assert "/home/" not in serialized
    assert "workstation-alice" not in serialized
    assert "torch_config" not in source


def test_cpu_source_omits_unrelated_accelerator_capabilities():
    environment = {
        "device": {
            "cuda_available": True,
            "cuda_devices": [
                {
                    "index": 0,
                    "major": 12,
                    "minor": 1,
                    "multi_processor_count": 48,
                    "name": "NVIDIA GB10",
                    "total_memory": 128,
                }
            ],
            "requested_device": "cpu",
            "torch_version": "2.12.1+cu130",
        },
        "generated_at": "2026-07-02T19:15:35Z",
        "machine": "aarch64",
        "platform": "Linux-6.11.0-aarch64-with-glibc2.39",
        "python": "3.12.3",
        "torchcts_version": "0.1.0",
    }

    source = normalize_source_environment(
        environment,
        source_id="source-0000000001",
        backends=["cpu-build", "fbgemm"],
    )

    assert source["device_capabilities"] == []


def test_validation_rejects_absolute_paths_anywhere():
    store = ExpandedBackendEvidence(
        sources={
            "source-0000000001": {
                **_source("source-0000000001", "cuda"),
                "note": "/home/alice/private.txt",
            }
        },
        observations={"cuda": {"aten::example": {"source-0000000001": _passing_contract()}}},
    )

    with pytest.raises(ValueError, match="POSIX absolute path"):
        validate_expanded_store(store)


def test_verification_rejects_noncanonical_source_serialization(tmp_path: Path):
    store = ExpandedBackendEvidence()
    add_source_observations(
        store,
        source=_source("source-0000000001", "cuda"),
        records=[("cuda", "aten::example", _passing_contract())],
    )
    destination = tmp_path / "backend-evidence"
    write_backend_evidence_store(store, destination)
    source = json.loads((destination / "sources.jsonl").read_text())
    (destination / "sources.jsonl").write_text(json.dumps(source) + "\n")

    with pytest.raises(ValueError, match="sources are not canonically serialized"):
        load_backend_evidence_store(destination, verify_canonical=True)


def test_backend_keys_are_semantic_and_explicit():
    assert canonical_backend("cpu_build") == "cpu-build"
    assert canonical_backend("privateuseone") == "privateuse1"
    with pytest.raises(ValueError, match="explicit backend"):
        canonical_backend("any")


def test_cpu_build_gate_is_valid_on_cpu_even_when_the_build_has_other_backends():
    assert collection._oracle_gate_can_run_on_device("cpu_build", "cpu") is True
    assert collection._oracle_gate_can_run_on_device("fbgemm", "cpu") is True
    assert collection._oracle_gate_can_run_on_device("cuda", "cpu") is False


def test_target_selection_requires_explicit_backend_without_surfaces():
    audit = {
        "entries": [
            {
                "coverage_kind": "backend_pack",
                "name": "aten::example",
                "pending_review": {"backend_gate": "cuda"},
            }
        ]
    }

    with pytest.raises(ValueError, match="requires --backend-gate"):
        collection._select_targets(audit)


def test_incompatible_backend_fails_before_collection(monkeypatch, tmp_path: Path):
    audit = {
        "entries": [
            {
                "coverage_kind": "backend_pack",
                "name": "aten::example",
                "pending_review": {"backend_gate": "cpu_build"},
            }
        ]
    }
    monkeypatch.setattr(collection.coverage, "build_audit", lambda: audit)
    monkeypatch.setattr(collection, "all_oracle_specs", lambda: ())
    monkeypatch.setattr(
        collection,
        "_backend_pack_evidence",
        lambda *args, **kwargs: pytest.fail("collection ran before backend preflight"),
    )

    with pytest.raises(ValueError, match="cannot run on requested device"):
        collection.collect_backend_evidence(
            store=tmp_path / "backend-evidence",
            device="cuda",
            backend_gates=["cpu-build"],
        )

    assert not (tmp_path / "backend-evidence").exists()


def test_privateuse1_requires_explicit_backend_identity(monkeypatch, tmp_path: Path):
    audit = {
        "entries": [
            {
                "coverage_kind": "backend_pack",
                "name": "aten::example",
                "pending_review": {"backend_gate": "privateuse1"},
            }
        ]
    }
    monkeypatch.setattr(collection.coverage, "build_audit", lambda: audit)
    monkeypatch.setattr(collection, "all_oracle_specs", lambda: ())

    with pytest.raises(ValueError, match="requires the registered backend identity"):
        collection.collect_backend_evidence(
            store=tmp_path / "backend-evidence",
            device="privateuseone",
            backend_gates=["privateuse1"],
        )

    assert not (tmp_path / "backend-evidence").exists()


def test_collection_writes_only_the_canonical_store_and_persists_failures(
    monkeypatch,
    tmp_path: Path,
):
    audit_entry = {
        "coverage_kind": "backend_pack",
        "name": "aten::example",
        "pending_review": {"backend_gate": "cpu_build"},
        "status": "pending_backend_pack",
    }
    raw_record = {
        "audit_coverage_kind": "backend_pack",
        "audit_status": "pending_backend_pack",
        "backend_gate": "cpu_build",
        "coverage_kind": "backend_pack",
        "coverage_status": "pending_backend_pack",
        "dispatch": {},
        "exclusion": None,
        "oracle": None,
        "oracle_result": {"ok": False, "error_message": "expected test failure"},
        "pending_review": {"backend_gate": "cpu_build"},
        "schema": {},
        "semantic_level": 5,
        "surface": "aten::example",
        "surface_kind": "functional_data",
        "variant_kind": "functional",
    }
    environment = {
        "cwd": "/home/alice/torchcts",
        "device": {
            "cuda_available": False,
            "mps_available": False,
            "requested_device": "cpu",
            "torch_cuda": "13.0",
            "torch_hip": None,
            "torch_version": "2.12.1+cu130",
        },
        "generated_at": "2026-07-02T19:15:35Z",
        "hostname": "workstation-alice",
        "machine": "aarch64",
        "platform": "Linux-6.11.0-aarch64-with-glibc2.39",
        "python": "3.12.3",
        "python_executable": "/home/alice/torchcts/.venv/bin/python",
        "torchcts_version": "0.1.0",
    }
    monkeypatch.setattr(collection.coverage, "build_audit", lambda: {"entries": [audit_entry]})
    monkeypatch.setattr(collection, "all_oracle_specs", lambda: ())
    monkeypatch.setattr(
        collection,
        "_backend_pack_evidence",
        lambda *args, **kwargs: {"metadata": {}, "records": [raw_record]},
    )
    monkeypatch.setattr(collection, "_environment_record", lambda device: environment)

    destination = tmp_path / "backend-evidence"
    result = collection.collect_backend_evidence(
        store=destination,
        device="cpu",
        surfaces=["aten::example"],
        backend_gates=["cpu-build"],
    )

    assert result["oracle_failure_count"] == 1
    assert {path.name for path in destination.iterdir()} == {
        "cpu-build.jsonl",
        "manifest.json",
        "sources.jsonl",
    }
    assert not list(tmp_path.rglob("*.tar.gz"))
    store = load_backend_evidence_store(destination, verify_canonical=True)
    assert store.observations["cpu-build"]["aten::example"]["source-0000000001"]["oracle_result"]["ok"] is False

    exit_code = collection.run_backend_evidence_collection_command(
        store=destination,
        device="cpu",
        surfaces=["aten::example"],
        backend_gates=["cpu-build"],
        run_oracles=True,
        run_pending_candidates=False,
        fail_on_oracle_failure=True,
    )
    reread = load_backend_evidence_store(destination, verify_canonical=True)
    assert exit_code == 1
    assert len(reread.sources) == 2
    assert reread.observations["cpu-build"]["aten::example"]["source-0000000002"]["oracle_result"]["ok"] is False
