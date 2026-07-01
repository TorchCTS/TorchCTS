#!/usr/bin/env python3
"""Reduce raw PyTorch dispatcher inventories into op-keyed metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREVIEW_PATH = REPO_ROOT / "scratch" / "pytorch-2.7-compat" / "matrix" / "reduced" / "op_metadata.preview.json"
TRACKED_OUTPUT = REPO_ROOT / "torchcts" / "op_metadata.json"


def parse_version_parts(version: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(version))
    if not match:
        raise ValueError(f"Expected semantic version, got {version!r}")
    return tuple(int(part) for part in match.groups())


def next_patch_upper_bound(version: str) -> str:
    major, minor, patch = parse_version_parts(version)
    return f"{major}.{minor}.{patch + 1}"


def repo_relative(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def load_artifact(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("artifact_kind") != "torch_dispatcher_inventory":
        raise ValueError(f"{path} is not a torch_dispatcher_inventory artifact")
    collection = data.get("collection") or {}
    if not collection.get("normalized_torch_version"):
        raise ValueError(f"{path} is missing collection.normalized_torch_version")
    if not isinstance(data.get("entries"), list):
        raise ValueError(f"{path} is missing entries list")
    data["_path"] = str(path)
    return data


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def schema_fingerprint(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": entry.get("schema"),
        "args": entry.get("args") or [],
        "returns": entry.get("returns") or [],
        "surface_kind": entry.get("surface_kind"),
        "variant_kind": entry.get("variant_kind"),
        "base_name": entry.get("base_name"),
        "overload": entry.get("overload") or "",
    }


def dispatch_fingerprint(entry: dict[str, Any]) -> dict[str, Any]:
    return dict(entry.get("dispatch") or {})


def load_legacy_metadata(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _range_record(start: str, end: str | None, fingerprint: dict[str, Any]) -> dict[str, Any]:
    return {
        "min": start,
        "max": end,
        "schema_hash": stable_hash(fingerprint),
        **fingerprint,
    }


def _dispatch_range_record(start: str, end: str | None, dispatch: dict[str, Any]) -> dict[str, Any]:
    return {
        "min": start,
        "max": end,
        "dispatch_hash": stable_hash(dispatch),
        "dispatch": dispatch,
        "non_gating": True,
    }


def compress_schema_ranges(
    versions: list[str],
    entries_by_version: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    ranges: list[dict[str, Any]] = []
    current_start: str | None = None
    current_fingerprint: dict[str, Any] | None = None
    for version in versions:
        entry = entries_by_version.get(version)
        fingerprint = schema_fingerprint(entry) if entry else None
        if fingerprint is None:
            if current_start is not None and current_fingerprint is not None:
                ranges.append(_range_record(current_start, version, current_fingerprint))
                current_start = None
                current_fingerprint = None
            continue
        if current_start is None:
            current_start = version
            current_fingerprint = fingerprint
            continue
        if fingerprint != current_fingerprint:
            ranges.append(_range_record(current_start, version, current_fingerprint))
            current_start = version
            current_fingerprint = fingerprint
    if current_start is not None and current_fingerprint is not None:
        ranges.append(_range_record(current_start, None, current_fingerprint))
    return ranges


def compress_dispatch_ranges(
    versions: list[str],
    entries_by_version: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    ranges: list[dict[str, Any]] = []
    current_start: str | None = None
    current_dispatch: dict[str, Any] | None = None
    for version in versions:
        entry = entries_by_version.get(version)
        dispatch = dispatch_fingerprint(entry) if entry else None
        if dispatch is None:
            if current_start is not None and current_dispatch is not None:
                ranges.append(_dispatch_range_record(current_start, version, current_dispatch))
                current_start = None
                current_dispatch = None
            continue
        if current_start is None:
            current_start = version
            current_dispatch = dispatch
            continue
        if dispatch != current_dispatch:
            ranges.append(_dispatch_range_record(current_start, version, current_dispatch))
            current_start = version
            current_dispatch = dispatch
    if current_start is not None and current_dispatch is not None:
        ranges.append(_dispatch_range_record(current_start, None, current_dispatch))
    return ranges


def first_removed_version(versions: list[str], seen: set[str]) -> str | None:
    observed_present = False
    for version in versions:
        if version in seen:
            observed_present = True
            continue
        if observed_present:
            return version
    return None


def was_reintroduced(versions: list[str], seen: set[str]) -> bool:
    observed_present = False
    observed_missing_after_present = False
    for version in versions:
        if version in seen:
            if observed_missing_after_present:
                return True
            observed_present = True
        elif observed_present:
            observed_missing_after_present = True
    return False


def _legacy_fields_for(name: str, legacy_metadata: dict[str, Any]) -> dict[str, Any]:
    legacy_entry = (legacy_metadata.get("ops") or {}).get(name, {})
    if not isinstance(legacy_entry, dict):
        return {}
    return {
        key: legacy_entry[key]
        for key in ("category", "pytorch_dtypes")
        if key in legacy_entry
    }


def _legacy_static_record(name: str, legacy_entry: dict[str, Any]) -> dict[str, Any]:
    record = {
        "legacy_static_only": True,
        "runtime_versioned": False,
        "introduced": None,
        "removed": None,
        "versions_seen": [],
        "versions_missing": [],
        "collection_status_by_version": {},
        "schema_ranges": [],
        "base_op": legacy_entry.get("base_op"),
        "overload": legacy_entry.get("overload") or "",
        "signature": legacy_entry.get("signature") or "",
    }
    for key in ("category", "pytorch_dtypes", "variant"):
        if key in legacy_entry:
            record[key] = legacy_entry[key]
    return record


def reduce_artifacts(
    artifacts: list[dict[str, Any]],
    *,
    include_dispatch_evidence: bool = True,
    legacy_metadata: dict[str, Any] | None = None,
    preserve_legacy_static_ops: bool = True,
) -> dict[str, Any]:
    versions = sorted(
        {artifact["collection"]["normalized_torch_version"] for artifact in artifacts},
        key=parse_version_parts,
    )
    artifacts_by_version = {
        artifact["collection"]["normalized_torch_version"]: artifact for artifact in artifacts
    }
    op_names = sorted({
        entry["name"]
        for artifact in artifacts
        for entry in artifact.get("entries", [])
        if entry.get("name")
    })

    warnings: list[dict[str, str]] = []
    ops: dict[str, Any] = {}
    for name in op_names:
        entries_by_version = {}
        for version in versions:
            artifact = artifacts_by_version[version]
            entry = next((candidate for candidate in artifact.get("entries", []) if candidate.get("name") == name), None)
            if entry is not None:
                entries_by_version[version] = entry

        seen_versions = sorted(entries_by_version, key=parse_version_parts)
        missing_versions = [version for version in versions if version not in entries_by_version]
        seen_set = set(seen_versions)
        schema_ranges = compress_schema_ranges(versions, entries_by_version)
        dispatch_ranges = compress_dispatch_ranges(versions, entries_by_version) if include_dispatch_evidence else []
        if was_reintroduced(versions, seen_set):
            warnings.append({"kind": "op_reintroduced", "name": name})
        if len({record["schema_hash"] for record in schema_ranges}) > 1:
            warnings.append({"kind": "schema_changed", "name": name})
        surface_kinds = {
            range_record.get("surface_kind")
            for range_record in schema_ranges
            if range_record.get("surface_kind")
        }
        if len(surface_kinds) > 1:
            warnings.append({"kind": "surface_kind_changed", "name": name})

        op_record = {
            "introduced": seen_versions[0],
            "removed": first_removed_version(versions, seen_set),
            "versions_seen": seen_versions,
            "versions_missing": missing_versions,
            "collection_status_by_version": {
                version: "present" if version in seen_set else "absent"
                for version in versions
            },
            "schema_ranges": schema_ranges,
        }
        if include_dispatch_evidence:
            op_record["dispatch_evidence_ranges"] = dispatch_ranges
        op_record.update(_legacy_fields_for(name, legacy_metadata or {}))
        ops[name] = op_record

    legacy_only_count = 0
    if preserve_legacy_static_ops and legacy_metadata:
        for name, legacy_entry in sorted((legacy_metadata.get("ops") or {}).items()):
            if name in ops or not isinstance(legacy_entry, dict):
                continue
            ops[name] = _legacy_static_record(name, legacy_entry)
            legacy_only_count += 1

    return {
        "version": 2,
        "metadata": {
            "source": "multi_version_pytorch_dispatcher_inventory",
            "collected_versions": versions,
            "min_validated_version": versions[0] if versions else None,
            "max_validated_version": versions[-1] if versions else None,
            "dependency_upper_bound": next_patch_upper_bound(versions[-1]) if versions else None,
            "artifact_paths": [repo_relative(artifact.get("_path")) for artifact in artifacts],
            "op_count": len(ops),
            "collected_op_count": len(op_names),
            "legacy_static_op_count": legacy_only_count,
            "dispatch_evidence_non_gating": bool(include_dispatch_evidence),
            "legacy_fields_preserved": ["category", "pytorch_dtypes"] if legacy_metadata else [],
        },
        "ops": ops,
        "warnings": warnings,
    }


def write_json(path: Path, payload: dict[str, Any], *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    else:
        text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_PREVIEW_PATH)
    parser.add_argument("--update-tracked", action="store_true")
    parser.add_argument("--legacy-metadata", type=Path, default=TRACKED_OUTPUT)
    parser.add_argument("--preserve-legacy-static-ops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-dispatch-evidence", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)

    out = TRACKED_OUTPUT if args.update_tracked else args.out
    include_dispatch_evidence = args.include_dispatch_evidence
    if include_dispatch_evidence is None:
        include_dispatch_evidence = not args.update_tracked
    payload = reduce_artifacts(
        [load_artifact(path) for path in args.artifacts],
        include_dispatch_evidence=include_dispatch_evidence,
        legacy_metadata=load_legacy_metadata(args.legacy_metadata),
        preserve_legacy_static_ops=args.preserve_legacy_static_ops,
    )
    write_json(out, payload, compact=args.compact)
    print(f"Wrote reduced op metadata: {out}")
    if payload["warnings"]:
        print(f"Warnings: {len(payload['warnings'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
