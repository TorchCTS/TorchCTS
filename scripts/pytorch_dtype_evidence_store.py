#!/usr/bin/env python3
"""Read, write, and validate categorized PyTorch dtype evidence stores."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from scripts import pytorch_dtype_evidence_taxonomy as taxonomy


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAT_VERSION = 3
FORMAT_NAME = "categorized_scoped_contract_evidence"
MANIFEST_NAME = "manifest.json"
LOCAL_EVIDENCE_PATH_REPLACEMENTS = (
    (str(REPO_ROOT / "scratch" / "pytorch-2.7-compat" / "matrix"), "<matrix-workdir>"),
    (str(REPO_ROOT / "scratch"), "<torchcts-workdir>"),
    ("scratch/pytorch-2.7-compat/matrix", "<matrix-workdir>"),
)
COUNT_BUCKETS = (
    "cpu_supported",
    "cpu_unsupported",
    "cpu_unknown",
    "cpu_pending",
    "oracle_supported",
    "source_expected",
)


def parse_version_parts(version: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.(\d+))?", str(version))
    if not match:
        raise ValueError(f"Expected semantic version, got {version!r}")
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch) if patch is not None else 0


def next_patch_upper_bound(version: str) -> str:
    major, minor, patch = parse_version_parts(version)
    return f"{major}.{minor}.{patch + 1}"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sanitize_tracked_evidence(value: Any) -> Any:
    if isinstance(value, str):
        sanitized = value
        for original, replacement in LOCAL_EVIDENCE_PATH_REPLACEMENTS:
            sanitized = sanitized.replace(original, replacement)
        return sanitized
    if isinstance(value, list):
        return [sanitize_tracked_evidence(item) for item in value]
    if isinstance(value, dict):
        return {str(key): sanitize_tracked_evidence(item) for key, item in value.items()}
    return value


def resolve_manifest_path(path: Path) -> Path:
    path = Path(path)
    return path if path.name == MANIFEST_NAME else path / MANIFEST_NAME


def normalize_contract(entry: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(entry)
    normalized.pop("replace_contract", None)
    evidence = normalized.get("evidence")
    if isinstance(evidence, dict):
        evidence.pop("pytorch_version", None)
        evidence.pop("version_rule", None)
    return sanitize_tracked_evidence(normalized)


def restore_structural_fields(
    contract: dict[str, Any],
    *,
    version: str,
    replace_contract: bool,
) -> dict[str, Any]:
    entry = deepcopy(contract)
    entry["replace_contract"] = replace_contract
    evidence = entry.get("evidence")
    if isinstance(evidence, dict):
        evidence["pytorch_version"] = version
        evidence["version_rule"] = version
    return entry


def normalize_expanded_evidence(expanded: dict[str, Any]) -> dict[str, Any]:
    contracts: dict[str, dict[str, Any]] = {}
    for op, versioned in (expanded.get("contracts") or {}).items():
        if not isinstance(versioned, dict):
            continue
        contracts[str(op)] = {
            str(version): normalize_contract(entry)
            for version, entry in versioned.items()
            if isinstance(entry, dict)
        }
    return {
        "collected_versions": sorted(
            [str(version) for version in (expanded.get("metadata") or {}).get("collected_versions") or []],
            key=parse_version_parts,
        ),
        "contracts": dict(sorted(contracts.items())),
        "warnings": list(expanded.get("warnings") or []),
    }


def expanded_contract_counts(contracts: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for versioned in contracts.values():
        if not isinstance(versioned, dict):
            continue
        for entry in versioned.values():
            if not isinstance(entry, dict):
                continue
            for key in COUNT_BUCKETS:
                for dtypes in (entry.get(key) or {}).values():
                    counts[key] += len(dtypes or ())
            mismatches = entry.get("source_probe_mismatches") or ()
            counts["source_probe_mismatches"] += len(mismatches)
            if entry.get("source_expected"):
                counts["source_expected_ops"] += 1
                for dtypes in (entry.get("source_expected") or {}).values():
                    counts["source_expected_entries"] += len(dtypes or ())
    return dict(sorted(counts.items()))


def load_legacy_evidence_jsonl(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] | None = None
    contracts: dict[str, Any] = {}
    warnings: list[Any] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        kind = record.get("record_kind")
        if kind == "metadata":
            if metadata is not None:
                raise ValueError(f"{path}:{line_number}: duplicate metadata record")
            if record.get("version") != 2 or record.get("format") != "expanded_evidence_jsonl":
                raise ValueError(f"{path}:{line_number}: unsupported legacy evidence header")
            metadata = record.get("metadata") or {}
        elif kind == "op_contract_evidence":
            op = record.get("op")
            versions = record.get("versions")
            if not isinstance(op, str) or not isinstance(versions, dict):
                raise ValueError(f"{path}:{line_number}: invalid op evidence record")
            if op in contracts:
                raise ValueError(f"{path}:{line_number}: duplicate evidence for {op}")
            contracts[op] = versions
        elif kind == "warning":
            warnings.append(record.get("warning"))
        else:
            raise ValueError(f"{path}:{line_number}: unknown record_kind {kind!r}")
    if metadata is None:
        raise ValueError(f"{path}: missing legacy evidence metadata record")
    return {
        "version": 2,
        "format": "expanded_evidence",
        "metadata": metadata,
        "contracts": dict(sorted(contracts.items())),
        "warnings": warnings,
    }


def _group_records(
    expanded: dict[str, Any],
    classifications: dict[str, taxonomy.Classification],
) -> dict[str, list[dict[str, Any]]]:
    metadata = expanded.get("metadata") or {}
    collected_versions = sorted(
        [str(version) for version in metadata.get("collected_versions") or []],
        key=parse_version_parts,
    )
    if not collected_versions:
        raise ValueError("Expanded evidence has no metadata.collected_versions")
    collected_set = set(collected_versions)
    by_category: dict[str, list[dict[str, Any]]] = {}

    for op, versioned in sorted((expanded.get("contracts") or {}).items()):
        if not isinstance(op, str) or not isinstance(versioned, dict) or not versioned:
            raise ValueError(f"{op!r}: evidence must be a non-empty version object")
        unknown_versions = set(versioned) - collected_set
        if unknown_versions:
            raise ValueError(f"{op}: evidence contains uncollected versions {sorted(unknown_versions)!r}")

        grouped: dict[str, tuple[dict[str, Any], list[str]]] = {}
        for version in collected_versions:
            entry = versioned.get(version)
            if not isinstance(entry, dict):
                continue
            contract = normalize_contract(entry)
            key = canonical_json(contract)
            if key not in grouped:
                grouped[key] = (contract, [])
            grouped[key][1].append(version)

        if not grouped:
            raise ValueError(f"{op}: operator has no evidence in collected versions")

        records: list[dict[str, Any]] = []
        only_group = next(iter(grouped.values())) if len(grouped) == 1 else None
        if only_group is not None and only_group[1] == collected_versions:
            records.append({"op": op, "contract": only_group[0]})
        else:
            variants = sorted(grouped.values(), key=lambda item: parse_version_parts(item[1][0]))
            for contract, versions in variants:
                records.append({"op": op, "versions": versions, "contract": contract})

        category = classifications[op].category
        by_category.setdefault(category, []).extend(records)

    for records in by_category.values():
        records.sort(
            key=lambda record: (
                record["op"],
                parse_version_parts((record.get("versions") or collected_versions)[0]),
            )
        )
    return by_category


def _manifest_metadata(expanded: dict[str, Any]) -> dict[str, Any]:
    metadata = expanded.get("metadata") or {}
    kept = {
        key: metadata[key]
        for key in ("artifact_names", "contract_authority", "generated_by", "input_artifact_versions")
        if key in metadata
    }
    return sanitize_tracked_evidence(kept)


def write_evidence_store(
    output: Path,
    expanded: dict[str, Any],
    *,
    metadata_categories: dict[str, str] | None = None,
    audit_report: Path | None = None,
) -> Path:
    manifest_path = resolve_manifest_path(output)
    store_root = manifest_path.parent
    if store_root.exists() and any(store_root.iterdir()):
        raise ValueError(f"Refusing to write evidence store into non-empty directory {store_root}")
    store_root.mkdir(parents=True, exist_ok=True)

    contracts = expanded.get("contracts") or {}
    classifications = taxonomy.classify_operators(set(contracts), metadata_categories)
    if audit_report is not None:
        taxonomy.write_audit_report(audit_report, classifications)
    by_category = _group_records(expanded, classifications)

    category_entries: list[dict[str, Any]] = []
    for category in sorted(by_category):
        relative_path = taxonomy.category_path(category)
        target = store_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [canonical_json(record) for record in by_category[category]]
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        category_entries.append({
            "category": category,
            "path": relative_path,
            "record_count": len(lines),
        })

    versions = sorted(
        [str(version) for version in (expanded.get("metadata") or {}).get("collected_versions") or []],
        key=parse_version_parts,
    )
    manifest = {
        "categories": category_entries,
        "collected_versions": versions,
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "metadata": _manifest_metadata(expanded),
        "replace_contract": True,
        "taxonomy_version": taxonomy.TAXONOMY_VERSION,
        "warnings": sanitize_tracked_evidence(list(expanded.get("warnings") or [])),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_evidence_store(manifest_path, verify_canonical=True, metadata_categories=metadata_categories)
    return manifest_path


def _replace_directory_with_rollback(source: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}.backup")
    if backup.exists():
        shutil.rmtree(backup)
    had_destination = destination.exists()
    if had_destination:
        os.replace(destination, backup)
    try:
        os.replace(source, destination)
    except Exception:
        if had_destination and backup.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def write_evidence_store_atomic(
    output: Path,
    expanded: dict[str, Any],
    *,
    metadata_categories: dict[str, str] | None = None,
    audit_report: Path | None = None,
) -> Path:
    destination = resolve_manifest_path(output).parent
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        write_evidence_store(
            temp_root,
            expanded,
            metadata_categories=metadata_categories,
            audit_report=audit_report,
        )
        _replace_directory_with_rollback(temp_root, destination)
    except Exception:
        if temp_root.exists():
            shutil.rmtree(temp_root)
        raise
    return destination / MANIFEST_NAME


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid manifest JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"{path}: manifest must be a JSON object")
    if manifest.get("format_version") != FORMAT_VERSION or manifest.get("format") != FORMAT_NAME:
        raise ValueError(
            f"{path}: unsupported evidence format "
            f"{manifest.get('format')!r} version {manifest.get('format_version')!r}"
        )
    if manifest.get("taxonomy_version") != taxonomy.TAXONOMY_VERSION:
        raise ValueError(f"{path}: unsupported taxonomy version {manifest.get('taxonomy_version')!r}")
    return manifest


def load_evidence_store(
    path: Path,
    *,
    verify_canonical: bool = False,
    metadata_categories: dict[str, str] | None = None,
) -> dict[str, Any]:
    manifest_path = resolve_manifest_path(path)
    manifest = _load_manifest(manifest_path)
    store_root = manifest_path.parent

    versions = manifest.get("collected_versions")
    if not isinstance(versions, list) or not versions or not all(isinstance(item, str) for item in versions):
        raise ValueError(f"{manifest_path}: collected_versions must be a non-empty string list")
    if versions != sorted(versions, key=parse_version_parts) or len(versions) != len(set(versions)):
        raise ValueError(f"{manifest_path}: collected_versions must be uniquely semver-sorted")
    version_set = set(versions)
    replace_contract = manifest.get("replace_contract")
    if not isinstance(replace_contract, bool):
        raise ValueError(f"{manifest_path}: replace_contract must be a boolean")
    manifest_metadata = manifest.get("metadata")
    if not isinstance(manifest_metadata, dict):
        raise ValueError(f"{manifest_path}: metadata must be an object")
    warnings = manifest.get("warnings")
    if not isinstance(warnings, list):
        raise ValueError(f"{manifest_path}: warnings must be a list")

    categories = manifest.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"{manifest_path}: categories must be a non-empty list")
    declared_paths: set[str] = set()
    declared_categories: set[str] = set()
    previous_category: str | None = None
    contracts: dict[str, dict[str, Any]] = {}
    op_categories: dict[str, str] = {}
    op_modes: dict[str, str] = {}
    op_contract_keys: dict[str, set[str]] = {}

    for category_entry in categories:
        if not isinstance(category_entry, dict):
            raise ValueError(f"{manifest_path}: category entries must be objects")
        category = category_entry.get("category")
        relative_path = category_entry.get("path")
        record_count = category_entry.get("record_count")
        if not isinstance(category, str) or category not in taxonomy.CATEGORIES:
            raise ValueError(f"{manifest_path}: unknown category {category!r}")
        if category in declared_categories:
            raise ValueError(f"{manifest_path}: duplicate category {category!r}")
        if previous_category is not None and category <= previous_category:
            raise ValueError(f"{manifest_path}: categories must be sorted")
        previous_category = category
        declared_categories.add(category)
        expected_path = taxonomy.category_path(category)
        if relative_path != expected_path:
            raise ValueError(f"{manifest_path}: category {category} must use path {expected_path}")
        if relative_path in declared_paths:
            raise ValueError(f"{manifest_path}: duplicate category path {relative_path!r}")
        declared_paths.add(relative_path)
        if type(record_count) is not int or record_count <= 0:
            raise ValueError(f"{manifest_path}: {category} record_count must be positive")

        category_path = store_root / relative_path
        if not category_path.is_file():
            raise ValueError(f"{manifest_path}: missing category file {relative_path}")
        raw_text = category_path.read_text(encoding="utf-8")
        if not raw_text.endswith("\n"):
            raise ValueError(f"{category_path}: category file must end with a newline")
        raw_lines = raw_text.splitlines()
        if len(raw_lines) != record_count:
            raise ValueError(
                f"{category_path}: manifest says {record_count} records, found {len(raw_lines)}"
            )

        previous_sort_key: tuple[str, tuple[int, int, int]] | None = None
        for line_number, raw_line in enumerate(raw_lines, start=1):
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{category_path}:{line_number}: invalid JSON: {exc}") from exc
            if verify_canonical and raw_line != canonical_json(record):
                raise ValueError(f"{category_path}:{line_number}: record is not canonically serialized")
            if not isinstance(record, dict) or set(record) not in ({"op", "contract"}, {"op", "versions", "contract"}):
                raise ValueError(f"{category_path}:{line_number}: invalid evidence record shape")
            op = record.get("op")
            contract = record.get("contract")
            if not isinstance(op, str) or not isinstance(contract, dict):
                raise ValueError(f"{category_path}:{line_number}: op and contract have invalid types")
            if "replace_contract" in contract:
                raise ValueError(f"{category_path}:{line_number}: contract repeats replace_contract")
            evidence = contract.get("evidence")
            if isinstance(evidence, dict) and ({"pytorch_version", "version_rule"} & set(evidence)):
                raise ValueError(f"{category_path}:{line_number}: contract repeats structural version fields")

            scoped_versions = record.get("versions")
            is_universal = scoped_versions is None
            if not is_universal:
                if not isinstance(scoped_versions, list) or not scoped_versions or not all(isinstance(item, str) for item in scoped_versions):
                    raise ValueError(f"{category_path}:{line_number}: versions must be a non-empty string list")
                if scoped_versions != sorted(scoped_versions, key=parse_version_parts) or len(scoped_versions) != len(set(scoped_versions)):
                    raise ValueError(f"{category_path}:{line_number}: versions must be uniquely semver-sorted")
                unknown = set(scoped_versions) - version_set
                if unknown:
                    raise ValueError(f"{category_path}:{line_number}: unknown versions {sorted(unknown)!r}")
                if scoped_versions == versions:
                    raise ValueError(
                        f"{category_path}:{line_number}: all-version contract must omit versions"
                    )
            else:
                scoped_versions = versions

            sort_key = (op, parse_version_parts(scoped_versions[0]))
            if previous_sort_key is not None and sort_key <= previous_sort_key:
                raise ValueError(f"{category_path}:{line_number}: records must be sorted and unique")
            previous_sort_key = sort_key

            prior_category = op_categories.setdefault(op, category)
            if prior_category != category:
                raise ValueError(f"{op}: operator appears in multiple categories")
            versioned = contracts.setdefault(op, {})
            mode = "universal" if is_universal else "scoped"
            prior_mode = op_modes.setdefault(op, mode)
            if prior_mode != mode or (mode == "universal" and versioned):
                raise ValueError(f"{op}: universal and scoped records cannot be combined")
            contract_key = canonical_json(contract)
            contract_keys = op_contract_keys.setdefault(op, set())
            if contract_key in contract_keys:
                raise ValueError(f"{op}: identical contracts must be grouped into one record")
            contract_keys.add(contract_key)
            for version in scoped_versions:
                if version in versioned:
                    raise ValueError(f"{op}: overlapping evidence scope at {version}")
                versioned[version] = restore_structural_fields(
                    contract,
                    version=version,
                    replace_contract=replace_contract,
                )

    actual_paths = {
        item.relative_to(store_root).as_posix()
        for item in store_root.rglob("*")
        if item.is_file() and item != manifest_path
    }
    if actual_paths != declared_paths:
        missing = sorted(declared_paths - actual_paths)
        extra = sorted(actual_paths - declared_paths)
        raise ValueError(f"{manifest_path}: category file mismatch; missing={missing!r}, extra={extra!r}")

    metadata_category_map = metadata_categories if metadata_categories is not None else taxonomy.load_metadata_categories()
    classifications = taxonomy.classify_operators(set(contracts), metadata_category_map)
    for op, category in op_categories.items():
        if classifications[op].category != category:
            raise ValueError(
                f"{op}: record is stored in {category}, classifier requires {classifications[op].category}"
            )

    for op, versioned in contracts.items():
        if not versioned:
            raise ValueError(f"{op}: operator has no applicable versions")

    metadata = dict(manifest_metadata)
    metadata.update({
        "collected_versions": versions,
        "contract_count": len(contracts),
        "contract_counts": expanded_contract_counts(contracts),
        "dependency_upper_bound": next_patch_upper_bound(versions[-1]),
        "max_validated_version": versions[-1],
        "min_validated_version": versions[0],
        "version_entry_semantics": "replace_contract",
    })
    return {
        "version": 2,
        "format": "expanded_evidence",
        "metadata": metadata,
        "contracts": dict(sorted(contracts.items())),
        "warnings": list(warnings),
    }


def load_evidence(
    path: Path,
    *,
    verify_canonical: bool = False,
    metadata_categories: dict[str, str] | None = None,
) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir() or path.name == MANIFEST_NAME:
        return load_evidence_store(
            path,
            verify_canonical=verify_canonical,
            metadata_categories=metadata_categories,
        )
    return load_legacy_evidence_jsonl(path)


def evidence_store_bytes(path: Path) -> int:
    root = resolve_manifest_path(path).parent
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())


def category_file_sizes(path: Path) -> dict[str, int]:
    manifest_path = resolve_manifest_path(path)
    manifest = _load_manifest(manifest_path)
    return {
        entry["category"]: (manifest_path.parent / entry["path"]).stat().st_size
        for entry in manifest["categories"]
    }
