#!/usr/bin/env python3
"""Verify compact dtype contracts against source evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import reduce_pytorch_dtype_contracts as reducer


DEFAULT_RUNTIME = REPO_ROOT / "torchcts" / "op_dtype_contracts.json"
DEFAULT_EVIDENCE = REPO_ROOT / "data" / "pytorch-version-matrix" / "op_dtype_contract_evidence.jsonl"


def _version_key(version: str) -> tuple[int, int, int]:
    return reducer.parse_version_parts(version)


def _json_canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load_runtime(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def validate_runtime_schema(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if data.get("version") != 2:
        errors.append("runtime version must be 2")
    if data.get("format") != "runtime_profile_ranges":
        errors.append("runtime format must be runtime_profile_ranges")

    metadata = data.get("metadata")
    profiles = data.get("profiles")
    contracts = data.get("contracts")
    if not isinstance(metadata, dict):
        errors.append("runtime metadata must be an object")
        metadata = {}
    if not isinstance(profiles, dict):
        errors.append("runtime profiles must be an object")
        profiles = {}
    if not isinstance(contracts, dict):
        errors.append("runtime contracts must be an object")
        contracts = {}

    versions = metadata.get("collected_versions")
    if not isinstance(versions, list) or not all(isinstance(item, str) for item in versions):
        errors.append("metadata.collected_versions must be a string list")
        versions = []
    else:
        sorted_versions = sorted(versions, key=_version_key)
        if versions != sorted_versions:
            errors.append("metadata.collected_versions must be sorted")
        if len(set(versions)) != len(versions):
            errors.append("metadata.collected_versions must not contain duplicates")
        if versions:
            if metadata.get("min_validated_version") != versions[0]:
                errors.append("metadata.min_validated_version must equal first collected version")
            if metadata.get("max_validated_version") != versions[-1]:
                errors.append("metadata.max_validated_version must equal last collected version")
            if metadata.get("dependency_upper_bound") != reducer.next_patch_upper_bound(versions[-1]):
                errors.append("metadata.dependency_upper_bound must be next patch after max validated version")

    version_set = set(versions)
    profile_ids = set(profiles)
    for profile_id, profile in profiles.items():
        if not isinstance(profile_id, str) or not profile_id.startswith("p"):
            errors.append(f"profile id {profile_id!r} is not deterministic pNNNNNN form")
        if not isinstance(profile, dict):
            errors.append(f"profile {profile_id} must be an object")
            continue
        for forbidden in reducer.EVIDENCE_ONLY_KEYS:
            if forbidden in profile:
                errors.append(f"profile {profile_id} contains evidence-only key {forbidden}")

    expected_profile_count = len(profiles)
    if metadata.get("profile_count") != expected_profile_count:
        errors.append("metadata.profile_count does not match profiles length")

    actual_range_count = 0
    for op_name, ranges in contracts.items():
        if not isinstance(op_name, str):
            errors.append("contract op keys must be strings")
        if not isinstance(ranges, list):
            errors.append(f"{op_name}: ranges must be a list")
            continue
        previous_end: tuple[int, int, int] | None = None
        for record in ranges:
            actual_range_count += 1
            if not isinstance(record, list) or len(record) != 3:
                errors.append(f"{op_name}: range must be [start,end,profile]")
                continue
            start, end, profile_id = (str(record[0]), str(record[1]), str(record[2]))
            if start not in version_set or end not in version_set:
                errors.append(f"{op_name}: range {record!r} references uncollected version")
                continue
            start_key = _version_key(start)
            end_key = _version_key(end)
            if start_key > end_key:
                errors.append(f"{op_name}: range {record!r} has start after end")
            if previous_end is not None and start_key <= previous_end:
                errors.append(f"{op_name}: ranges overlap or are unsorted")
            previous_end = end_key
            if profile_id not in profile_ids:
                errors.append(f"{op_name}: range {record!r} references unknown profile")

    if metadata.get("range_count") != actual_range_count:
        errors.append("metadata.range_count does not match actual range count")
    if metadata.get("contract_count") not in (None, len(contracts)):
        errors.append("metadata.contract_count does not match contracts length")
    return errors


def verify_artifacts(runtime_path: Path, evidence_path: Path, *, max_runtime_bytes: int = 0) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    runtime = load_runtime(runtime_path)
    evidence = reducer.load_evidence_jsonl(evidence_path)
    errors.extend(validate_runtime_schema(runtime))

    recomputed = reducer.compact_runtime_from_expanded(evidence)
    if _json_canonical(recomputed) != _json_canonical(runtime):
        errors.append("compact runtime does not match runtime recomputed from source evidence")

    errors.extend(reducer.verify_runtime_equivalence(evidence, runtime))

    runtime_bytes = runtime_path.stat().st_size
    if max_runtime_bytes and runtime_bytes > max_runtime_bytes:
        errors.append(f"runtime artifact is {runtime_bytes} bytes, above limit {max_runtime_bytes}")

    summary = {
        "runtime": str(runtime_path),
        "evidence": str(evidence_path),
        "runtime_bytes": runtime_bytes,
        "ops": len(runtime.get("contracts") or {}),
        "versions": len(runtime.get("metadata", {}).get("collected_versions") or []),
        "profiles": len(runtime.get("profiles") or {}),
        "ranges": runtime.get("metadata", {}).get("range_count"),
        "evidence_records": len(evidence.get("contracts") or {}),
    }
    return errors, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--max-runtime-bytes", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        errors, summary = verify_artifacts(
            args.runtime,
            args.evidence,
            max_runtime_bytes=args.max_runtime_bytes,
        )
    except Exception as exc:
        errors = [f"{type(exc).__name__}: {exc}"]
        summary = {}

    if args.json:
        print(json.dumps({"ok": not errors, "errors": errors, "summary": summary}, indent=2, sort_keys=True))
    else:
        print("PyTorch dtype contract artifact verification")
        for key, value in summary.items():
            print(f"{key}: {value}")
        if errors:
            print("errors:")
            for error in errors:
                print(f"- {error}")
        else:
            print("ok: true")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
