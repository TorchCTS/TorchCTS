# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies or substantial portions of the Software.

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torchcts.core.dtype_contracts as dtype_contracts_module
import torchcts.op_metadata as op_metadata_module


REPO_ROOT = Path(__file__).resolve().parents[2]
if not (REPO_ROOT / "scripts" / "collect_pytorch_version_matrix.py").exists():
    pytest.skip("PyTorch version matrix selftests require a source checkout", allow_module_level=True)

pytestmark = pytest.mark.covers_category("selftest")


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _dispatcher_artifact(version: str, entries: list[dict], path: Path) -> Path:
    payload = {
        "version": 1,
        "artifact_kind": "torch_dispatcher_inventory",
        "collection": {
            "torch_version": version,
            "normalized_torch_version": version,
            "wheel_family": "cpu",
            "python_executable": sys.executable,
            "python_version": "test",
            "platform": "test",
            "repo_root": str(REPO_ROOT),
            "repo_commit": "test",
            "generated_at": "2026-06-30T00:00:00Z",
        },
        "entries": entries,
        "errors": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _dtype_artifact(version: str, contracts: dict, path: Path) -> Path:
    payload = {
        "version": 1,
        "artifact_kind": "torch_dtype_contract_probe",
        "collection": {
            "torch_version": version,
            "normalized_torch_version": version,
            "wheel_family": "cpu",
            "python_executable": sys.executable,
            "python_version": "test",
            "platform": "test",
            "repo_root": str(REPO_ROOT),
            "repo_commit": "test",
            "generated_at": "2026-06-30T00:00:00Z",
            "probe_layers": ["source"],
            "selected_dtypes": ["torch.float32"],
            "version_rule": version,
        },
        "contracts": contracts,
        "probe_counts": {},
        "contract_counts": {},
        "source_extraction": {},
        "errors": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _entry(name: str, schema: str, *, dispatch=None, surface_kind="functional_data") -> dict:
    base_overload = name.removeprefix("aten::").split(".", 1)
    base = base_overload[0]
    overload = base_overload[1] if len(base_overload) == 2 else ""
    return {
        "name": name,
        "base_name": base,
        "overload": overload,
        "schema": schema,
        "args": [{"name": "self", "type": "Tensor", "tensor": True}],
        "returns": [{"name": "", "type": "Tensor", "tensor": True}],
        "surface_kind": surface_kind,
        "variant_kind": "functional",
        "dispatch": dispatch or {"CPU": True, "PrivateUse1": False},
    }


def _schema_range(name: str, schema: str, min_version: str, max_version: str | None = None) -> dict:
    entry = _entry(name, schema)
    return {
        "min": min_version,
        "max": max_version,
        "schema_hash": f"test:{min_version}:{max_version}",
        "schema": entry["schema"],
        "args": entry["args"],
        "returns": entry["returns"],
        "surface_kind": entry["surface_kind"],
        "variant_kind": entry["variant_kind"],
        "base_name": entry["base_name"],
        "overload": entry["overload"],
    }


def test_collect_ops_default_root_is_task_specific_scratch_directory():
    collect_ops = _load_script("collect_pytorch_ops")

    path = collect_ops.default_output_path("2.7.0", "cpu")

    assert path == REPO_ROOT / "scratch" / "pytorch-2.7-compat" / "matrix" / "raw" / "torch-2.7.0-cpu-dispatcher.json"


def test_matrix_selection_resolves_latest_aliases_and_deduplicates_versions():
    matrix_runner = _load_script("collect_pytorch_version_matrix")
    matrix = {
        "source_url": "https://example.test/pytorch",
        "families": {"cpu": {"index_url": "https://example.test/cpu"}},
        "selections": {
            "test": [
                {"version": "2.8.0", "family": "cpu"},
                {"alias": "latest:2.8", "minor": "2.8", "family": "cpu"},
                {"version": "2.9.0", "family": "cpu"},
            ]
        },
    }

    jobs = matrix_runner.plan_jobs(
        matrix,
        selection="test",
        latest_resolver=lambda minor, _source_url: "2.8.0" if minor == "2.8" else "bad",
    )

    assert [(job.version, job.family) for job in jobs] == [("2.8.0", "cpu"), ("2.9.0", "cpu")]
    assert jobs[0].labels == ("2.8.0", "latest:2.8")
    assert jobs[0].index_url == "https://example.test/cpu"


def test_matrix_runner_writes_manifest_without_touching_real_venvs_in_dry_run(tmp_path):
    matrix_runner = _load_script("collect_pytorch_version_matrix")
    jobs = [matrix_runner.MatrixJob(version="2.7.0", family="cpu", index_url="", labels=("2.7.0",))]

    records = matrix_runner.run_matrix(jobs, matrix_root=tmp_path, dry_run=True)

    manifest = json.loads((tmp_path / "run-manifest.json").read_text(encoding="utf-8"))
    assert records[0]["status"] == "dry_run"
    assert manifest["runs"][0]["venv"].endswith("venvs/torch-2.7.0-cpu")
    assert manifest["runs"][0]["artifacts"][0].endswith("raw/torch-2.7.0-cpu-dispatcher.json")
    assert not (tmp_path / "venvs").exists()


def test_matrix_runner_uses_task_specific_output_paths_with_fake_commands(tmp_path):
    matrix_runner = _load_script("collect_pytorch_version_matrix")
    jobs = [matrix_runner.MatrixJob(version="2.7.0", family="cpu", index_url="https://example.test/cpu")]
    commands = []

    def fake_venv_creator(path, python_executable):
        assert python_executable == "/fake/python3.10"
        bin_dir = path / ("Scripts" if sys.platform == "win32" else "bin")
        bin_dir.mkdir(parents=True)
        (bin_dir / ("python.exe" if sys.platform == "win32" else "python")).write_text("", encoding="utf-8")

    def fake_runner(cmd, *, cwd, env, log_file):
        commands.append(cmd)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("fake\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="")

    records = matrix_runner.run_matrix(
        jobs,
        matrix_root=tmp_path,
        keep_venvs=True,
        python_executable="/fake/python3.10",
        command_runner=fake_runner,
        venv_creator=fake_venv_creator,
    )

    assert records[0]["status"] == "passed"
    assert records[0]["python_executable"] == "/fake/python3.10"
    assert commands[1][-2:] == ["--index-url", "https://example.test/cpu"]
    assert commands[2][-2:] == ["--out", str(tmp_path / "raw" / "torch-2.7.0-cpu-dispatcher.json")]
    assert (tmp_path / "venvs" / "torch-2.7.0-cpu").exists()


def test_matrix_runner_can_collect_dtype_contract_artifacts_with_fake_commands(tmp_path):
    matrix_runner = _load_script("collect_pytorch_version_matrix")
    jobs = [matrix_runner.MatrixJob(version="2.7.0", family="cpu", index_url="")]
    commands = []

    def fake_venv_creator(path, python_executable):
        bin_dir = path / ("Scripts" if sys.platform == "win32" else "bin")
        bin_dir.mkdir(parents=True)
        (bin_dir / ("python.exe" if sys.platform == "win32" else "python")).write_text("", encoding="utf-8")

    def fake_runner(cmd, *, cwd, env, log_file):
        commands.append(cmd)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("fake\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="")

    records = matrix_runner.run_matrix(
        jobs,
        matrix_root=tmp_path,
        keep_venvs=True,
        artifacts=("dispatcher", "dtype-contracts"),
        dtype_layers=("source",),
        dtype_values=("torch.float32",),
        dtype_limit=3,
        command_runner=fake_runner,
        venv_creator=fake_venv_creator,
    )

    assert records[0]["status"] == "passed"
    assert records[0]["artifact_types"] == ["dispatcher", "dtype-contracts"]
    assert records[0]["artifacts"] == [
        str(tmp_path / "raw" / "torch-2.7.0-cpu-dispatcher.json"),
        str(tmp_path / "raw" / "torch-2.7.0-cpu-dtype-contracts-layers-source-dtypes-float32-limit-3.json"),
    ]
    assert commands[2][-4:] == ["pytest>=7.0.0", "psutil>=5.0.0", "numpy", "expecttest"]
    dtype_command = commands[-1]
    assert str(REPO_ROOT / "scripts" / "collect_pytorch_dtype_contracts.py") in dtype_command
    assert ["--layer", "source"] == dtype_command[dtype_command.index("--layer"):dtype_command.index("--layer") + 2]
    assert ["--dtypes", "torch.float32"] == dtype_command[dtype_command.index("--dtypes"):dtype_command.index("--dtypes") + 2]
    assert ["--limit", "3"] == dtype_command[dtype_command.index("--limit"):dtype_command.index("--limit") + 2]


def test_reducer_compresses_schema_ranges_and_tracks_runtime_absence(tmp_path):
    reducer = _load_script("reduce_pytorch_op_inventory")
    op_v1 = _entry("aten::sample.Tensor", "aten::sample.Tensor(Tensor self) -> Tensor")
    op_v2 = _entry("aten::sample.Tensor", "aten::sample.Tensor(Tensor self, *, bool flag=False) -> Tensor")
    new_op = _entry("aten::newop", "aten::newop(Tensor self) -> Tensor")
    old_op = _entry("aten::oldop", "aten::oldop(Tensor self) -> Tensor")
    paths = [
        _dispatcher_artifact("2.7.0", [op_v1, old_op], tmp_path / "torch-2.7.0-cpu-dispatcher.json"),
        _dispatcher_artifact("2.8.0", [op_v1], tmp_path / "torch-2.8.0-cpu-dispatcher.json"),
        _dispatcher_artifact("2.9.0", [op_v2, new_op], tmp_path / "torch-2.9.0-cpu-dispatcher.json"),
    ]

    reduced = reducer.reduce_artifacts([reducer.load_artifact(path) for path in paths])

    sample = reduced["ops"]["aten::sample.Tensor"]
    assert sample["introduced"] == "2.7.0"
    assert sample["removed"] is None
    assert [record["min"] for record in sample["schema_ranges"]] == ["2.7.0", "2.9.0"]
    assert sample["schema_ranges"][0]["max"] == "2.9.0"
    assert reduced["ops"]["aten::oldop"]["removed"] == "2.8.0"
    assert reduced["ops"]["aten::newop"]["versions_missing"] == ["2.7.0", "2.8.0"]
    assert reduced["ops"]["aten::newop"]["collection_status_by_version"]["2.8.0"] == "absent"
    assert {"kind": "schema_changed", "name": "aten::sample.Tensor"} in reduced["warnings"]


def test_reducer_keeps_dispatch_evidence_non_gating(tmp_path):
    reducer = _load_script("reduce_pytorch_op_inventory")
    paths = [
        _dispatcher_artifact(
            "2.7.0",
            [_entry("aten::dispatchy", "aten::dispatchy(Tensor self) -> Tensor", dispatch={"CPU": True, "PrivateUse1": False})],
            tmp_path / "torch-2.7.0-cpu-dispatcher.json",
        ),
        _dispatcher_artifact(
            "2.8.0",
            [_entry("aten::dispatchy", "aten::dispatchy(Tensor self) -> Tensor", dispatch={"CPU": True, "PrivateUse1": True})],
            tmp_path / "torch-2.8.0-cpu-dispatcher.json",
        ),
    ]

    reduced = reducer.reduce_artifacts([reducer.load_artifact(path) for path in paths])

    op = reduced["ops"]["aten::dispatchy"]
    assert op["versions_seen"] == ["2.7.0", "2.8.0"]
    assert len(op["schema_ranges"]) == 1
    assert len(op["dispatch_evidence_ranges"]) == 2
    assert all(record["non_gating"] is True for record in op["dispatch_evidence_ranges"])
    assert reduced["metadata"]["dispatch_evidence_non_gating"] is True


def test_reducer_can_omit_dispatch_evidence_and_preserve_legacy_fields(tmp_path):
    reducer = _load_script("reduce_pytorch_op_inventory")
    paths = [
        _dispatcher_artifact(
            "2.7.0",
            [_entry("aten::legacy_hint", "aten::legacy_hint(Tensor self) -> Tensor")],
            tmp_path / "torch-2.7.0-cpu-dispatcher.json",
        ),
    ]
    legacy_metadata = {
        "version": 1,
        "ops": {
            "aten::legacy_hint": {
                "category": "elementwise_unary",
                "pytorch_dtypes": ["f32", "f64"],
                "signature": "legacy",
            },
            "aten::legacy_static": {
                "base_op": "legacy_static",
                "category": "other",
                "overload": "",
                "pytorch_dtypes": ["f32"],
                "signature": "aten::legacy_static(Tensor self) -> Tensor",
            }
        },
    }

    reduced = reducer.reduce_artifacts(
        [reducer.load_artifact(path) for path in paths],
        include_dispatch_evidence=False,
        legacy_metadata=legacy_metadata,
    )

    op = reduced["ops"]["aten::legacy_hint"]
    assert "dispatch_evidence_ranges" not in op
    assert op["category"] == "elementwise_unary"
    assert op["pytorch_dtypes"] == ["f32", "f64"]
    assert reduced["ops"]["aten::legacy_static"]["legacy_static_only"] is True
    assert reduced["metadata"]["legacy_static_op_count"] == 1
    assert reduced["metadata"]["dispatch_evidence_non_gating"] is False
    assert reduced["metadata"]["legacy_fields_preserved"] == ["category", "pytorch_dtypes"]


def test_dtype_contract_collector_wraps_generator_payload():
    collector = _load_script("collect_pytorch_dtype_contracts")
    contract_payload = {
        "metadata": {
            "version_rule": "2.7.0",
            "selected_dtypes": ["torch.float32"],
            "last_run_probe_counts": {"source_seeded_ops": 1},
            "contract_counts": {"source_expected": 1},
            "source_extraction": {"op_metadata_seeded_ops": 1},
        },
        "contracts": {
            "aten::sample": {
                "2.7.0": {
                    "source_expected": {"*": ["torch.float32"]},
                }
            }
        },
    }

    artifact = collector.build_raw_artifact(
        contract_payload=contract_payload,
        torch_version="2.7.0+cpu",
        family="cpu",
        layers=["source"],
    )

    assert artifact["artifact_kind"] == "torch_dtype_contract_probe"
    assert artifact["collection"]["normalized_torch_version"] == "2.7.0"
    assert artifact["probe_counts"] == {"source_seeded_ops": 1}
    assert artifact["contracts"]["aten::sample"]["2.7.0"]["source_expected"]["*"] == ["torch.float32"]


def test_dtype_contract_reducer_marks_version_entries_as_replacements(tmp_path):
    reducer = _load_script("reduce_pytorch_dtype_contracts")
    artifacts = [
        _dtype_artifact(
            "2.7.0",
            {
                "aten::sample": {
                    "2.7.0": {
                        "cpu_supported": {"forward:clean": ["torch.float32"]},
                        "evidence": {"source": "test"},
                    }
                }
            },
            tmp_path / "torch-2.7.0-cpu-dtype-contracts.json",
        ),
        _dtype_artifact(
            "2.8.0",
            {
                "aten::sample": {
                    "2.8.0": {
                        "cpu_unsupported": {"forward:clean": ["torch.float32"]},
                        "evidence": {"source": "test"},
                    }
                }
            },
            tmp_path / "torch-2.8.0-cpu-dtype-contracts.json",
        ),
    ]

    reduced = reducer.build_expanded_evidence([reducer.load_artifact(path) for path in artifacts])

    sample = reduced["contracts"]["aten::sample"]
    assert sample["2.7.0"]["replace_contract"] is True
    assert sample["2.8.0"]["replace_contract"] is True
    assert sample["2.7.0"]["cpu_supported"]["forward:clean"] == ["torch.float32"]
    assert sample["2.8.0"]["cpu_unsupported"]["forward:clean"] == ["torch.float32"]
    assert reduced["metadata"]["version_entry_semantics"] == "replace_contract"


def test_dtype_contract_reducer_emits_compact_profiles_and_ranges(tmp_path):
    reducer = _load_script("reduce_pytorch_dtype_contracts")
    artifacts = [
        _dtype_artifact(
            "2.7.0",
            {
                "aten::sample": {
                    "2.7.0": {
                        "cpu_supported": {"forward:clean": ["torch.float32"]},
                        "probe_details": {"forward:clean": {"torch.float32": "large detail"}},
                        "evidence": {"source": "test"},
                    }
                }
            },
            tmp_path / "torch-2.7.0-cpu-dtype-contracts.json",
        ),
        _dtype_artifact(
            "2.8.0",
            {
                "aten::sample": {
                    "2.8.0": {
                        "cpu_supported": {"forward:clean": ["torch.float32"]},
                        "probe_details": {"forward:clean": {"torch.float32": "different large detail"}},
                        "evidence": {"source": "test"},
                    }
                }
            },
            tmp_path / "torch-2.8.0-cpu-dtype-contracts.json",
        ),
    ]

    reduced = reducer.reduce_artifacts([reducer.load_artifact(path) for path in artifacts])

    assert reduced["version"] == 2
    assert reduced["format"] == "runtime_profile_ranges"
    assert reduced["metadata"]["dependency_upper_bound"] == "2.8.1"
    assert reduced["metadata"]["profile_count"] == 1
    assert reduced["contracts"]["aten::sample"] == [["2.7.0", "2.8.0", "p000001"]]
    assert "probe_details" not in reduced["profiles"]["p000001"]
    assert reduced["profiles"]["p000001"]["cpu_supported"]["forward:clean"] == ["torch.float32"]


def test_dtype_contract_reducer_breaks_ranges_on_missing_versions(tmp_path):
    reducer = _load_script("reduce_pytorch_dtype_contracts")
    artifacts = [
        _dtype_artifact(
            "2.7.0",
            {"aten::sample": {"2.7.0": {"cpu_supported": {"forward:clean": ["torch.float32"]}}}},
            tmp_path / "torch-2.7.0-cpu-dtype-contracts.json",
        ),
        _dtype_artifact(
            "2.8.0",
            {},
            tmp_path / "torch-2.8.0-cpu-dtype-contracts.json",
        ),
        _dtype_artifact(
            "2.9.0",
            {"aten::sample": {"2.9.0": {"cpu_supported": {"forward:clean": ["torch.float32"]}}}},
            tmp_path / "torch-2.9.0-cpu-dtype-contracts.json",
        ),
    ]

    reduced = reducer.reduce_artifacts([reducer.load_artifact(path) for path in artifacts])

    assert reduced["contracts"]["aten::sample"] == [
        ["2.7.0", "2.7.0", "p000001"],
        ["2.9.0", "2.9.0", "p000001"],
    ]


def test_dtype_contract_evidence_jsonl_round_trips_full_entries(tmp_path):
    reducer = _load_script("reduce_pytorch_dtype_contracts")
    artifact = _dtype_artifact(
        "2.7.0",
        {
            "aten::sample": {
                "2.7.0": {
                    "cpu_unknown": {"forward:clean": ["torch.float32"]},
                    "probe_details": {"forward:clean": {"torch.float32": "detail"}},
                    "source_probe_mismatches": [{"kind": "source_declared_but_probe_unknown"}],
                    "evidence": {"source": "test"},
                }
            }
        },
        tmp_path / "torch-2.7.0-cpu-dtype-contracts.json",
    )
    evidence_path = tmp_path / "evidence.jsonl"
    expanded = reducer.build_expanded_evidence([reducer.load_artifact(artifact)])

    reducer.write_evidence_jsonl(evidence_path, expanded)
    loaded = reducer.load_evidence_jsonl(evidence_path)

    entry = loaded["contracts"]["aten::sample"]["2.7.0"]
    assert entry["probe_details"]["forward:clean"]["torch.float32"] == "detail"
    assert entry["source_probe_mismatches"][0]["kind"] == "source_declared_but_probe_unknown"


def test_compact_dtype_contract_runtime_accepts_collected_local_build_and_rejects_future(monkeypatch):
    payload = {
        "version": 2,
        "format": "runtime_profile_ranges",
        "metadata": {
            "contract_authority": "versioned_cpu_probe",
            "collected_versions": ["2.7.0"],
            "min_validated_version": "2.7.0",
            "max_validated_version": "2.7.0",
            "dependency_upper_bound": "2.7.1",
        },
        "profiles": {
            "p000001": {
                "cpu_supported": {"forward:clean": ["torch.float32"]},
                "cpu_unsupported": {},
                "cpu_unknown": {},
                "cpu_pending": {},
                "oracle_supported": {},
                "source_expected": {"*": ["torch.float32"]},
            }
        },
        "contracts": {
            "aten::sample": [["2.7.0", "2.7.0", "p000001"]],
        },
    }
    monkeypatch.setattr(dtype_contracts_module, "load_dtype_contracts", lambda: payload)
    monkeypatch.setattr(dtype_contracts_module.torch, "__version__", "2.7.0+cpu")

    collected = dtype_contracts_module.contract_disposition("sample", "torch.float32")

    assert collected.allowed
    assert collected.status == dtype_contracts_module.CPU_SUPPORTED
    assert collected.evidence["contract_profiles"][0]["profile_id"] == "p000001"

    monkeypatch.setattr(dtype_contracts_module.torch, "__version__", "2.8.0")
    future = dtype_contracts_module.contract_disposition("sample", "torch.float32")

    assert not future.allowed
    assert future.status == dtype_contracts_module.NOT_RECORDED
    assert "not in the TorchCTS validated PyTorch matrix" in future.detail


def test_dtype_contract_reducer_merges_multiple_layers_for_same_version(tmp_path):
    reducer = _load_script("reduce_pytorch_dtype_contracts")
    artifacts = [
        _dtype_artifact(
            "2.7.0",
            {
                "aten::sample": {
                    "2.7.0": {
                        "source_expected": {"*": ["torch.float32", "torch.float64"]},
                        "evidence": {"source": "source_layer"},
                    }
                }
            },
            tmp_path / "torch-2.7.0-cpu-dtype-contracts.json",
        ),
        _dtype_artifact(
            "2.7.0",
            {
                "aten::sample": {
                    "2.7.0": {
                        "cpu_supported": {"forward:clean": ["torch.float32"]},
                        "probe_details": {"forward:clean": {"torch.float32": "ok"}},
                        "evidence": {"source": "probe_layer"},
                    }
                }
            },
            tmp_path / "torch-2.7.0-cpu-dtype-contracts-probe.json",
        ),
    ]

    reduced = reducer.build_expanded_evidence([reducer.load_artifact(path) for path in artifacts])

    entry = reduced["contracts"]["aten::sample"]["2.7.0"]
    assert entry["replace_contract"] is True
    assert entry["source_expected"]["*"] == ["torch.float32", "torch.float64"]
    assert entry["cpu_supported"]["forward:clean"] == ["torch.float32"]
    assert entry["probe_details"]["forward:clean"]["torch.float32"] == "ok"
    assert entry["evidence"]["sources"] == ["probe_layer", "source_layer"]


def test_dtype_contract_reducer_preserves_existing_versions_and_replaces_artifact_version(tmp_path):
    reducer = _load_script("reduce_pytorch_dtype_contracts")
    artifact = _dtype_artifact(
        "2.7.0",
        {
            "aten::sample": {
                "2.7.0": {
                    "cpu_supported": {"forward:clean": ["torch.float32"]},
                    "evidence": {"source": "fresh_probe"},
                }
            }
        },
        tmp_path / "torch-2.7.0-cpu-dtype-contracts.json",
    )
    existing = {
        "version": 1,
        "contracts": {
            "aten::sample": {
                "2.7.0": {
                    "cpu_unsupported": {"forward:clean": ["torch.float32"]},
                    "evidence": {"source": "stale_probe"},
                },
                "2.12": {
                    "cpu_supported": {"forward:clean": ["torch.float64"]},
                    "evidence": {"source": "existing_probe"},
                },
            }
        },
    }

    reduced = reducer.build_expanded_evidence(
        [reducer.load_artifact(artifact)],
        existing_contracts=existing,
    )

    sample = reduced["contracts"]["aten::sample"]
    assert set(sample) == {"2.7.0", "2.12"}
    assert sample["2.7.0"]["cpu_supported"]["forward:clean"] == ["torch.float32"]
    assert sample["2.7.0"]["cpu_unsupported"] == {}
    assert sample["2.12"]["cpu_supported"]["forward:clean"] == ["torch.float64"]
    assert reduced["metadata"]["collected_versions"] == ["2.7.0", "2.12"]
    assert reduced["metadata"]["input_artifact_versions"] == ["2.7.0"]
    assert reduced["metadata"]["preserved_versions"] == ["2.12"]


def test_dtype_contract_generator_filters_runtime_unavailable_metadata(monkeypatch):
    generator = _load_script("generate_op_dtype_contracts")
    monkeypatch.setattr(
        generator,
        "load_op_metadata",
        lambda: {
            "ops": {
                "aten::live": {"pytorch_dtypes": ["torch.float32"]},
                "aten::future": {"pytorch_dtypes": ["torch.float32"]},
            }
        },
    )
    monkeypatch.setattr(generator, "op_available_in_runtime", lambda name: name == "aten::live")

    data = {"version": 1, "metadata": {}, "contracts": {}}
    count = generator._seed_source_metadata(data, version_rule="2.7.0")

    assert count == 1
    assert sorted(data["contracts"]) == ["aten::live"]


def test_dtype_contract_generator_filters_runtime_unavailable_generated_cases(monkeypatch):
    generator = _load_script("generate_op_dtype_contracts")
    import torchcts.generated.generated_cases as generated_cases

    monkeypatch.setattr(
        generated_cases,
        "GENERATED_CASES",
        {
            "cases_by_surface": {
                "functional_data": [
                    {
                        "name": "aten::live",
                        "status": "covered_generated",
                        "generated": {"strategy": {"strategy": "manual"}},
                    },
                    {
                        "name": "aten::future",
                        "status": "covered_generated",
                        "generated": {"strategy": {"strategy": "manual"}},
                    },
                ]
            }
        },
    )
    monkeypatch.setattr(generator, "op_available_in_runtime", lambda name: name == "aten::live")

    assert [entry["name"] for entry in generator._iter_generated_entries()] == ["aten::live"]


def test_op_metadata_helpers_select_runtime_schema_ranges_and_absence():
    metadata = {
        "version": 2,
        "ops": {
            "aten::sample.Tensor": {
                "introduced": "2.7.0",
                "removed": None,
                "versions_seen": ["2.7.0", "2.8.0", "2.9.0"],
                "versions_missing": [],
                "schema_ranges": [
                    _schema_range(
                        "aten::sample.Tensor",
                        "aten::sample.Tensor(Tensor self) -> Tensor",
                        "2.7.0",
                        "2.9.0",
                    ),
                    _schema_range(
                        "aten::sample.Tensor",
                        "aten::sample.Tensor(Tensor self, *, bool flag=False) -> Tensor",
                        "2.9.0",
                        None,
                    ),
                ],
            },
            "aten::newer": {
                "introduced": "2.9.0",
                "removed": None,
                "versions_seen": ["2.9.0"],
                "versions_missing": ["2.7.0", "2.8.0"],
                "schema_ranges": [
                    _schema_range(
                        "aten::newer",
                        "aten::newer(Tensor self) -> Tensor",
                        "2.9.0",
                        None,
                    ),
                ],
            },
        },
    }

    assert op_metadata_module.op_available_in_runtime("aten::sample.Tensor", "2.8.0", metadata=metadata)
    assert not op_metadata_module.op_available_in_runtime("aten::newer", "2.8.0", metadata=metadata)
    assert op_metadata_module.op_available_in_runtime("aten::newer", "2.9.0", metadata=metadata)

    selected = op_metadata_module.schema_record_for_runtime("sample.Tensor", "2.9.0", metadata=metadata)
    assert selected["schema"] == "aten::sample.Tensor(Tensor self, *, bool flag=False) -> Tensor"

    unavailable = op_metadata_module.runtime_unavailable_op_entries(
        metadata=metadata,
        runtime_version="2.8.0",
        live_names={"aten::sample.Tensor"},
    )
    assert [entry["name"] for entry in unavailable] == ["aten::newer"]
    assert unavailable[0]["runtime_availability"]["status"] == "unavailable_in_pytorch_runtime"


def test_get_op_metadata_returns_v1_compatible_runtime_facade(monkeypatch):
    metadata = {
        "version": 2,
        "ops": {
            "aten::sample.Tensor": {
                "introduced": "2.7.0",
                "removed": None,
                "category": "elementwise_unary",
                "pytorch_dtypes": ["f32"],
                "versions_seen": ["2.7.0", "2.9.0"],
                "versions_missing": ["2.8.0"],
                "schema_ranges": [
                    _schema_range(
                        "aten::sample.Tensor",
                        "aten::sample.Tensor(Tensor self) -> Tensor",
                        "2.7.0",
                        "2.9.0",
                    ),
                    _schema_range(
                        "aten::sample.Tensor",
                        "aten::sample.Tensor(Tensor self, *, bool flag=False) -> Tensor",
                        "2.9.0",
                        None,
                    ),
                ],
            },
        },
    }
    monkeypatch.setattr(op_metadata_module, "load_op_metadata", lambda: metadata)
    monkeypatch.setattr(op_metadata_module.torch, "__version__", "2.9.0")

    record = op_metadata_module.get_op_metadata("sample.Tensor")

    assert record["category"] == "elementwise_unary"
    assert record["pytorch_dtypes"] == ["f32"]
    assert record["signature"] == "aten::sample.Tensor(Tensor self, *, bool flag=False) -> Tensor"
    assert record["schema"] == "aten::sample.Tensor(Tensor self, *, bool flag=False) -> Tensor"
    assert record["base_op"] == "sample"
    assert record["base_name"] == "sample"
    assert record["overload"] == "Tensor"
    assert record["surface_kind"] == "functional_data"
    assert record["variant_kind"] == "functional"
    assert record["variant"] == "functional"
    assert record["args"][0]["name"] == "self"


def test_op_metadata_helpers_preserve_v1_availability_behavior():
    metadata = {
        "version": 1,
        "ops": {
            "aten::legacy": {
                "base_op": "legacy",
                "overload": "",
                "signature": "aten::legacy(Tensor self) -> Tensor",
            }
        },
    }

    assert op_metadata_module.op_available_in_runtime("legacy", "2.7.0", metadata=metadata)
    assert op_metadata_module.schema_record_for_runtime("aten::legacy", "2.7.0", metadata=metadata)["schema"] == (
        "aten::legacy(Tensor self) -> Tensor"
    )
    assert op_metadata_module.runtime_unavailable_op_entries(metadata=metadata, runtime_version="2.7.0") == []


def test_op_metadata_helpers_do_not_claim_future_unvalidated_versions():
    metadata = {
        "version": 2,
        "metadata": {"collected_versions": ["2.7.0", "2.8.0"]},
        "ops": {
            "aten::sample": {
                "introduced": "2.7.0",
                "removed": None,
                "versions_seen": ["2.7.0", "2.8.0"],
                "versions_missing": [],
                "schema_ranges": [
                    _schema_range("aten::sample", "aten::sample(Tensor self) -> Tensor", "2.7.0", None),
                ],
            },
        },
    }

    assert op_metadata_module.schema_record_for_runtime("aten::sample", "2.8.0", metadata=metadata)
    assert op_metadata_module.schema_record_for_runtime("aten::sample", "2.9.0", metadata=metadata) is None


def test_version_hole_checker_detects_missing_stable_patch(tmp_path):
    checker = _load_script("check_pytorch_version_holes")
    matrix = {
        "families": {"cpu": {"index_url": ""}},
        "selections": {
            "test": [
                {"version": "2.7.0", "family": "cpu"},
                {"version": "2.7.2", "family": "cpu"},
            ],
        },
    }

    result = checker.find_version_holes(
        matrix,
        selection="test",
        available_versions={"2.7.0", "2.7.1", "2.7.2"},
    )

    assert not result["ok"]
    assert result["families"]["cpu"]["unresolved_holes"] == ["2.7.1"]


def test_version_hole_checker_respects_exclusions():
    checker = _load_script("check_pytorch_version_holes")
    matrix = {
        "families": {"cpu": {"index_url": ""}},
        "selections": {
            "test": [
                {"version": "2.7.0", "family": "cpu"},
                {"version": "2.7.2", "family": "cpu"},
            ],
        },
    }

    result = checker.find_version_holes(
        matrix,
        selection="test",
        available_versions={"2.7.0", "2.7.1", "2.7.2"},
        exclusions={("2.7.1", "cpu")},
    )

    assert result["ok"]
    assert result["families"]["cpu"]["excluded_holes"] == ["2.7.1"]


def test_dtype_contract_verifier_detects_unknown_profile_reference():
    verifier = _load_script("verify_pytorch_dtype_contract_artifacts")
    runtime = {
        "version": 2,
        "format": "runtime_profile_ranges",
        "metadata": {
            "collected_versions": ["2.7.0"],
            "min_validated_version": "2.7.0",
            "max_validated_version": "2.7.0",
            "dependency_upper_bound": "2.7.1",
            "profile_count": 0,
            "range_count": 1,
            "contract_count": 1,
        },
        "profiles": {},
        "contracts": {"aten::sample": [["2.7.0", "2.7.0", "p000001"]]},
    }

    errors = verifier.validate_runtime_schema(runtime)

    assert any("unknown profile" in error for error in errors)


def test_op_metadata_helpers_skip_v2_legacy_static_only_records_for_runtime_unavailable():
    metadata = {
        "version": 2,
        "ops": {
            "aten::legacy_static": {
                "legacy_static_only": True,
                "base_op": "legacy_static",
                "overload": "",
                "signature": "aten::legacy_static(Tensor self) -> Tensor",
                "category": "other",
                "pytorch_dtypes": ["f32"],
            }
        },
    }

    assert op_metadata_module.op_available_in_runtime("aten::legacy_static", "2.7.0", metadata=metadata)
    assert op_metadata_module.runtime_unavailable_op_entries(metadata=metadata, runtime_version="2.7.0") == []
