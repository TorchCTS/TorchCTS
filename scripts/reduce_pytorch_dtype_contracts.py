#!/usr/bin/env python3
"""Reduce raw dtype-contract artifacts into version-aware contract metadata."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_PREVIEW_PATH = REPO_ROOT / "scratch" / "pytorch-2.7-compat" / "matrix" / "reduced" / "op_dtype_contracts.preview.json"
DEFAULT_EVIDENCE_PREVIEW_PATH = REPO_ROOT / "scratch" / "pytorch-2.7-compat" / "matrix" / "reduced" / "op_dtype_contract_evidence.preview.jsonl"
TRACKED_RUNTIME_OUTPUT = REPO_ROOT / "torchcts" / "op_dtype_contracts.json"
TRACKED_EVIDENCE_OUTPUT = REPO_ROOT / "data" / "pytorch-version-matrix" / "op_dtype_contract_evidence.jsonl"

RUNTIME_BUCKETS = (
    "cpu_supported",
    "cpu_unsupported",
    "cpu_unknown",
    "cpu_pending",
    "oracle_supported",
    "source_expected",
)
EVIDENCE_ONLY_KEYS = {
    "evidence",
    "probe_details",
    "replace_contract",
    "source_probe_mismatches",
}


def parse_version_parts(version: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", str(version))
    if not match:
        raise ValueError(f"Expected semantic version, got {version!r}")
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch) if patch is not None else 0


def load_artifact(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("artifact_kind") != "torch_dtype_contract_probe":
        raise ValueError(f"{path} is not a torch_dtype_contract_probe artifact")
    collection = data.get("collection") or {}
    if not collection.get("normalized_torch_version"):
        raise ValueError(f"{path} is missing collection.normalized_torch_version")
    if not isinstance(data.get("contracts"), dict):
        raise ValueError(f"{path} is missing contracts object")
    data["_path"] = str(path)
    return data


def repo_relative(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def next_patch_upper_bound(version: str) -> str:
    major, minor, patch = parse_version_parts(version)
    return f"{major}.{minor}.{patch + 1}"


def _entry_for_artifact_version(versioned_contract: dict[str, Any], version: str) -> dict[str, Any] | None:
    if version in versioned_contract and isinstance(versioned_contract[version], dict):
        return dict(versioned_contract[version])
    if len(versioned_contract) == 1:
        value = next(iter(versioned_contract.values()))
        return dict(value) if isinstance(value, dict) else None
    return None


def _contract_counter(contracts: dict[str, Any]) -> dict[str, int]:
    counts = Counter()
    for versions in contracts.values():
        if not isinstance(versions, dict):
            continue
        for entry in versions.values():
            if not isinstance(entry, dict):
                continue
            for key in ("cpu_supported", "cpu_unsupported", "cpu_unknown", "cpu_pending", "oracle_supported", "source_expected"):
                for dtypes in (entry.get(key) or {}).values():
                    counts[key] += len(dtypes or ())
            counts["source_probe_mismatches"] += len(entry.get("source_probe_mismatches") or ())
            if entry.get("source_expected"):
                counts["source_expected_ops"] += 1
                for dtypes in (entry.get("source_expected") or {}).values():
                    counts["source_expected_entries"] += len(dtypes or ())
    return dict(sorted(counts.items()))


def _merge_condition_map(target: dict[str, Any], source: Any) -> None:
    if not isinstance(source, dict):
        return
    for condition, dtypes in source.items():
        values = set(target.get(str(condition), []) or [])
        values.update(str(dtype) for dtype in dtypes or ())
        target[str(condition)] = sorted(values)


def _merge_probe_details(target: dict[str, Any], source: Any) -> None:
    if not isinstance(source, dict):
        return
    for condition, records in source.items():
        if not isinstance(records, dict):
            continue
        target.setdefault(str(condition), {}).update(records)


def _append_unique_dicts(target: list[Any], source: Any) -> None:
    for item in source or ():
        if item not in target:
            target.append(item)


def _merge_evidence(target: dict[str, Any], source: Any) -> None:
    if not isinstance(source, dict):
        return
    sources = set(target.get("sources", []) or [])
    if target.get("source"):
        sources.add(str(target["source"]))
    if source.get("source"):
        sources.add(str(source["source"]))
    target.update(source)
    if sources:
        target["sources"] = sorted(sources)


def merge_contract_entry(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    merged = target
    merged["replace_contract"] = True
    for key in ("cpu_supported", "cpu_unsupported", "cpu_unknown", "cpu_pending", "oracle_supported", "source_expected"):
        _merge_condition_map(merged.setdefault(key, {}), source.get(key))
    _merge_probe_details(merged.setdefault("probe_details", {}), source.get("probe_details"))
    _append_unique_dicts(merged.setdefault("source_probe_mismatches", []), source.get("source_probe_mismatches"))
    _merge_evidence(merged.setdefault("evidence", {}), source.get("evidence"))
    for key, value in source.items():
        if key in {
            "cpu_supported",
            "cpu_unsupported",
            "cpu_unknown",
            "cpu_pending",
            "oracle_supported",
            "source_expected",
            "probe_details",
            "source_probe_mismatches",
            "evidence",
            "replace_contract",
        }:
            continue
        merged.setdefault(key, value)
    return merged


def load_existing_contracts(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _copy_existing_contracts(existing_contracts: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    for op_name, versioned in ((existing_contracts or {}).get("contracts") or {}).items():
        if not isinstance(versioned, dict):
            continue
        copied_versions = {
            str(version): dict(entry)
            for version, entry in versioned.items()
            if isinstance(entry, dict)
        }
        if copied_versions:
            contracts[str(op_name)] = copied_versions
    return contracts


def _version_keys(contracts: dict[str, dict[str, Any]]) -> list[str]:
    versions = {
        version
        for versioned in contracts.values()
        for version in versioned
    }
    return sorted(versions, key=parse_version_parts)


def build_expanded_evidence(
    artifacts: list[dict[str, Any]],
    *,
    existing_contracts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    versions = sorted(
        {artifact["collection"]["normalized_torch_version"] for artifact in artifacts},
        key=parse_version_parts,
    )
    contracts = _copy_existing_contracts(existing_contracts)
    existing_versions = _version_keys(contracts)
    preserved_versions = [version for version in existing_versions if version not in versions]
    replaced_artifact_entries: set[tuple[str, str]] = set()
    warnings: list[dict[str, str]] = []

    for artifact in sorted(artifacts, key=lambda item: parse_version_parts(item["collection"]["normalized_torch_version"])):
        version = artifact["collection"]["normalized_torch_version"]
        if artifact.get("errors"):
            warnings.append({
                "kind": "artifact_errors",
                "version": version,
                "detail": f"{len(artifact['errors'])} error(s)",
            })
        for op_name, versioned_contract in (artifact.get("contracts") or {}).items():
            if not isinstance(versioned_contract, dict):
                continue
            entry = _entry_for_artifact_version(versioned_contract, version)
            if entry is None:
                warnings.append({
                    "kind": "missing_version_entry",
                    "version": version,
                    "name": str(op_name),
                })
                continue
            evidence = entry.setdefault("evidence", {})
            if isinstance(evidence, dict):
                evidence.setdefault("version_rule", version)
                evidence.setdefault("source", "pytorch_version_matrix.dtype_contract_probe")
            op_key = str(op_name)
            replace_key = (op_key, version)
            if replace_key not in replaced_artifact_entries:
                contracts.setdefault(op_key, {})[version] = {}
                replaced_artifact_entries.add(replace_key)
            target = contracts.setdefault(op_key, {}).setdefault(version, {})
            merge_contract_entry(target, entry)

    all_versions = sorted(set(versions).union(preserved_versions), key=parse_version_parts)
    return {
        "version": 2,
        "format": "expanded_evidence",
        "metadata": {
            "contract_authority": "versioned_cpu_probe",
            "generated_by": "scripts/reduce_pytorch_dtype_contracts.py",
            "collected_versions": all_versions,
            "input_artifact_versions": versions,
            "min_validated_version": all_versions[0] if all_versions else None,
            "max_validated_version": all_versions[-1] if all_versions else None,
            "dependency_upper_bound": next_patch_upper_bound(all_versions[-1]) if all_versions else None,
            "preserved_versions": preserved_versions,
            "artifact_paths": [repo_relative(artifact.get("_path")) for artifact in artifacts],
            "contract_count": len(contracts),
            "version_entry_semantics": "replace_contract",
            "contract_counts": _contract_counter(contracts),
        },
        "contracts": dict(sorted(contracts.items())),
        "warnings": warnings,
    }


def _normalize_condition_map(value: Any) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    if not isinstance(value, dict):
        return normalized
    for condition, dtypes in value.items():
        if not dtypes:
            continue
        if isinstance(dtypes, str):
            dtype_values = [dtypes]
        else:
            dtype_values = [str(dtype) for dtype in dtypes if dtype is not None]
        if dtype_values:
            normalized[str(condition)] = sorted(set(dtype_values))
    return dict(sorted(normalized.items()))


def runtime_profile_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        bucket: _normalize_condition_map(entry.get(bucket))
        for bucket in RUNTIME_BUCKETS
    }


def canonical_profile(profile: dict[str, Any]) -> str:
    return json.dumps(profile, sort_keys=True, separators=(",", ":"))


def build_profiles(expanded_contracts: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    canonical_to_profile: dict[str, dict[str, Any]] = {}
    for versioned in (expanded_contracts.get("contracts") or {}).values():
        if not isinstance(versioned, dict):
            continue
        for entry in versioned.values():
            if not isinstance(entry, dict):
                continue
            profile = runtime_profile_from_entry(entry)
            canonical_to_profile.setdefault(canonical_profile(profile), profile)

    profiles: dict[str, dict[str, Any]] = {}
    canonical_to_id: dict[str, str] = {}
    for index, canonical in enumerate(sorted(canonical_to_profile), start=1):
        profile_id = f"p{index:06d}"
        canonical_to_id[canonical] = profile_id
        profiles[profile_id] = canonical_to_profile[canonical]
    return profiles, canonical_to_id


def _range_record(start: str, end: str, profile_id: str) -> list[str]:
    return [start, end, profile_id]


def compact_runtime_from_expanded(expanded: dict[str, Any]) -> dict[str, Any]:
    contracts = expanded.get("contracts") or {}
    metadata = expanded.get("metadata") or {}
    versions = sorted(metadata.get("collected_versions") or _version_keys(contracts), key=parse_version_parts)
    profiles, canonical_to_id = build_profiles(expanded)
    compact_contracts: dict[str, list[list[str]]] = {}
    range_count = 0

    for op_name, versioned in sorted(contracts.items()):
        if not isinstance(versioned, dict):
            continue
        ranges: list[list[str]] = []
        current_start: str | None = None
        current_end: str | None = None
        current_profile_id: str | None = None
        for version in versions:
            entry = versioned.get(version)
            if not isinstance(entry, dict):
                if current_start is not None and current_end is not None and current_profile_id is not None:
                    ranges.append(_range_record(current_start, current_end, current_profile_id))
                current_start = None
                current_end = None
                current_profile_id = None
                continue
            profile = runtime_profile_from_entry(entry)
            profile_id = canonical_to_id[canonical_profile(profile)]
            if current_start is None:
                current_start = version
                current_end = version
                current_profile_id = profile_id
                continue
            if profile_id == current_profile_id:
                current_end = version
                continue
            if current_end is not None and current_profile_id is not None:
                ranges.append(_range_record(current_start, current_end, current_profile_id))
            current_start = version
            current_end = version
            current_profile_id = profile_id
        if current_start is not None and current_end is not None and current_profile_id is not None:
            ranges.append(_range_record(current_start, current_end, current_profile_id))
        if ranges:
            range_count += len(ranges)
            compact_contracts[str(op_name)] = ranges

    runtime_metadata = {
        "contract_authority": metadata.get("contract_authority") or "versioned_cpu_probe",
        "generated_by": "scripts/reduce_pytorch_dtype_contracts.py",
        "collected_versions": versions,
        "min_validated_version": versions[0] if versions else None,
        "max_validated_version": versions[-1] if versions else None,
        "dependency_upper_bound": next_patch_upper_bound(versions[-1]) if versions else None,
        "evidence_artifact": str(TRACKED_EVIDENCE_OUTPUT.relative_to(REPO_ROOT)).replace("\\", "/"),
        "profile_count": len(profiles),
        "range_count": range_count,
        "contract_count": len(compact_contracts),
        "contract_counts": metadata.get("contract_counts") or {},
    }

    return {
        "version": 2,
        "format": "runtime_profile_ranges",
        "metadata": runtime_metadata,
        "profiles": profiles,
        "contracts": compact_contracts,
        "warnings": list(expanded.get("warnings") or []),
    }


def reduce_artifacts(
    artifacts: list[dict[str, Any]],
    *,
    existing_contracts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expanded = build_expanded_evidence(artifacts, existing_contracts=existing_contracts)
    return compact_runtime_from_expanded(expanded)


def write_json(path: Path, payload: dict[str, Any], *, compact: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    else:
        text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def write_evidence_jsonl(path: Path, expanded: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "record_kind": "metadata",
                "version": 2,
                "format": "expanded_evidence_jsonl",
                "metadata": expanded.get("metadata") or {},
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    ]
    for op_name, versions in sorted((expanded.get("contracts") or {}).items()):
        lines.append(json.dumps(
            {
                "record_kind": "op_contract_evidence",
                "op": op_name,
                "versions": versions,
            },
            sort_keys=True,
            separators=(",", ":"),
        ))
    for warning in expanded.get("warnings") or []:
        lines.append(json.dumps(
            {
                "record_kind": "warning",
                "warning": warning,
            },
            sort_keys=True,
            separators=(",", ":"),
        ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_evidence_jsonl(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    contracts: dict[str, Any] = {}
    warnings: list[Any] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        kind = record.get("record_kind")
        if kind == "metadata":
            metadata = record.get("metadata") or {}
        elif kind == "op_contract_evidence":
            op_name = record.get("op")
            versions = record.get("versions")
            if not isinstance(op_name, str) or not isinstance(versions, dict):
                raise ValueError(f"{path}:{line_number}: invalid op evidence record")
            contracts[op_name] = versions
        elif kind == "warning":
            warnings.append(record.get("warning"))
        else:
            raise ValueError(f"{path}:{line_number}: unknown record_kind {kind!r}")
    return {
        "version": 2,
        "format": "expanded_evidence",
        "metadata": metadata,
        "contracts": dict(sorted(contracts.items())),
        "warnings": warnings,
    }


def verify_runtime_equivalence(expanded: dict[str, Any], runtime: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    profiles = runtime.get("profiles") or {}
    runtime_contracts = runtime.get("contracts") or {}
    versions = runtime.get("metadata", {}).get("collected_versions") or []
    for op_name, versioned in (expanded.get("contracts") or {}).items():
        ranges = runtime_contracts.get(op_name) or []
        for version in versions:
            entry = versioned.get(version) if isinstance(versioned, dict) else None
            expected = runtime_profile_from_entry(entry) if isinstance(entry, dict) else None
            actual = None
            for range_record in ranges:
                if not isinstance(range_record, list) or len(range_record) != 3:
                    continue
                start, end, profile_id = range_record
                if parse_version_parts(str(start)) <= parse_version_parts(str(version)) <= parse_version_parts(str(end)):
                    actual = profiles.get(profile_id)
                    break
            if expected != actual:
                errors.append(f"{op_name} {version}: compact runtime profile mismatch")
                if len(errors) >= 20:
                    return errors
    return errors


def runtime_size(path: Path) -> int:
    return path.stat().st_size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="Backward-compatible alias for --runtime-out.")
    parser.add_argument("--runtime-out", type=Path, default=DEFAULT_RUNTIME_PREVIEW_PATH)
    parser.add_argument("--evidence-out", type=Path, default=DEFAULT_EVIDENCE_PREVIEW_PATH)
    parser.add_argument("--update-tracked", action="store_true")
    parser.add_argument("--existing-contracts", type=Path, default=TRACKED_RUNTIME_OUTPUT)
    parser.add_argument("--no-existing-contracts", action="store_true")
    parser.add_argument("--verify-equivalence", action="store_true")
    parser.add_argument("--max-runtime-bytes", type=int, default=0)
    args = parser.parse_args(argv)

    runtime_out = TRACKED_RUNTIME_OUTPUT if args.update_tracked else (args.out or args.runtime_out)
    evidence_out = TRACKED_EVIDENCE_OUTPUT if args.update_tracked else args.evidence_out
    existing_contracts = {} if args.no_existing_contracts else load_existing_contracts(args.existing_contracts)
    expanded = build_expanded_evidence(
        [load_artifact(path) for path in args.artifacts],
        existing_contracts=existing_contracts,
    )
    runtime = compact_runtime_from_expanded(expanded)
    if args.verify_equivalence:
        errors = verify_runtime_equivalence(expanded, runtime)
        if errors:
            for error in errors:
                print(error)
            return 1
    write_json(runtime_out, runtime, compact=True)
    write_evidence_jsonl(evidence_out, expanded)
    print(f"Wrote compact dtype contracts: {runtime_out}")
    print(f"Wrote dtype contract evidence: {evidence_out}")
    if args.max_runtime_bytes and runtime_size(runtime_out) > args.max_runtime_bytes:
        print(f"Runtime artifact is {runtime_size(runtime_out)} bytes, above limit {args.max_runtime_bytes}")
        return 1
    if runtime["warnings"]:
        print(f"Warnings: {len(runtime['warnings'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
