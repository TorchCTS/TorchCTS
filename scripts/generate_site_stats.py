#!/usr/bin/env python3
"""Generate a Markdown stats source for website copy and AI-assisted site work."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as _datetime
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

import torchcts
from torchcts.core.coverage import (
    COVERED_STATUSES,
    DISPATCH_KEYS,
    EXCLUDED_STATUSES,
    PENDING_STATUSES,
    build_audit,
)


DEFAULT_OUTPUT = REPO_ROOT / "docs" / "site-stats.md"
SITE_STATS_COLLECTION_ENV = "TORCHCTS_SITE_STATS_COLLECTION_JSON"
SEMANTIC_LEVELS = tuple(str(level) for level in range(1, 9))


def _pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.0%"
    return f"{numerator / denominator * 100:.1f}%"


def _table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def _counter_rows(counter: Counter, *, limit: int | None = None) -> list[list[object]]:
    items = sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))
    if limit is not None:
        items = items[:limit]
    return [[key, value] for key, value in items]


def _sorted_mapping_rows(mapping: dict, *, numeric_keys: bool = False) -> list[list[object]]:
    def sort_key(item):
        key, _value = item
        if numeric_keys:
            try:
                return int(key)
            except (TypeError, ValueError):
                return key
        return str(key)

    return [[key, value] for key, value in sorted(mapping.items(), key=sort_key)]


def _repo_relative(path: str) -> str:
    return path.replace("\\", "/")


def _pyproject_version() -> str:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    return match.group(1) if match else "unknown"


def _installed_version() -> str:
    try:
        return importlib.metadata.version("torchcts")
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _version_provenance() -> dict:
    module_version = getattr(torchcts, "__version__", "missing")
    installed_version = _installed_version()
    pyproject_version = _pyproject_version()
    versions_agree = (
        module_version != "missing"
        and module_version == installed_version
        and module_version == pyproject_version
    )
    return {
        "module_version": module_version,
        "installed_version": installed_version,
        "pyproject_version": pyproject_version,
        "import_path": str(Path(getattr(torchcts, "__file__", "") or "namespace package")),
        "versions_agree": versions_agree,
    }


def _suite_from_nodeid(nodeid: str) -> str:
    path = nodeid.split("::", 1)[0]
    parts = Path(path).parts
    if len(parts) >= 2 and parts[0] == "torchcts":
        return parts[1]
    return "unknown"


def _file_from_nodeid(nodeid: str) -> str:
    return _repo_relative(nodeid.split("::", 1)[0])


def _function_from_nodeid(nodeid: str) -> str:
    tail = nodeid.split("::", 1)[1] if "::" in nodeid else nodeid
    return tail.split("[", 1)[0]


def _test_kind_from_suite(suite: str) -> str:
    if suite == "opinfo":
        return "opinfo"
    if suite == "generated":
        return "generated"
    if suite == "selftest":
        return "selftest"
    return "handwritten"


def _has_level_override(extra_pytest_args: list[str]) -> bool:
    return any(
        arg == "--level"
        or arg == "--level-exact"
        or arg == "--level-range"
        or arg.startswith("--level=")
        or arg.startswith("--level-exact=")
        or arg.startswith("--level-range=")
        for arg in extra_pytest_args
    )


def _collect_pytest_nodes(extra_pytest_args: list[str]) -> dict:
    effective_extra_args = list(extra_pytest_args)
    if not _has_level_override(effective_extra_args):
        effective_extra_args.extend(["--level", "8"])
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "torchcts",
        "--validation",
        *effective_extra_args,
    ]
    command_display = [
        "python",
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "torchcts",
        "--validation",
        *effective_extra_args,
    ]
    with tempfile.TemporaryDirectory(prefix="torchcts-site-stats-") as tmpdir:
        metadata_path = Path(tmpdir) / "collection.json"
        env = dict(os.environ)
        env[SITE_STATS_COLLECTION_ENV] = str(metadata_path)
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=env,
        )
        structured_collection = None
        if metadata_path.exists():
            structured_collection = json.loads(metadata_path.read_text(encoding="utf-8"))
    output = result.stdout + result.stderr
    nodes = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.startswith("torchcts/") and "::" in line
    ]
    collected_from_summary = None
    match = re.search(r"(\d+) tests collected", output)
    if match:
        collected_from_summary = int(match.group(1))
    if result.returncode != 0:
        tail = "\n".join(output.splitlines()[-80:])
        raise RuntimeError(
            "pytest collection failed with exit code "
            f"{result.returncode}\nCommand: {' '.join(command)}\n{tail}"
        )
    return {
        "command": command,
        "command_display": command_display,
        "nodes": nodes,
        "node_count": len(nodes),
        "collected_from_summary": collected_from_summary,
        "structured_collection": structured_collection,
    }


def _fallback_records_from_nodes(nodes: list[str]) -> list[dict]:
    records = []
    for nodeid in nodes:
        level_match = re.search(r"\[L([1-8])\]", nodeid)
        records.append({
            "nodeid": nodeid,
            "file": _file_from_nodeid(nodeid),
            "suite": _suite_from_nodeid(nodeid),
            "test_kind": _test_kind_from_suite(_suite_from_nodeid(nodeid)),
            "function": _function_from_nodeid(nodeid),
            "semantic_level": int(level_match.group(1)) if level_match else None,
            "capability": None,
            "dtype": None,
            "dtype_fields": {},
            "dispatcher_name": None,
            "coverage_id": None,
            "coverage_kind": None,
            "surface_kind": None,
            "variant_kind": None,
            "strategy": None,
            "strategy_family": None,
            "decision": "executable",
            "skip_reason": None,
            "skip_detail": None,
        })
    return records


def _structured_records(collection: dict | None, nodes: list[str]) -> list[dict]:
    if collection:
        records = collection.get("records")
        if isinstance(records, list):
            return [record for record in records if isinstance(record, dict)]
    return _fallback_records_from_nodes(nodes)


def _level_key(value) -> str:
    if value is None:
        return "unknown"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _field_presence(value) -> str:
    return "present" if value else "missing"


def _collection_stats(nodes: list[str], structured_collection: dict | None = None) -> dict:
    records = _structured_records(structured_collection, nodes)
    suites = Counter()
    kinds = Counter()
    files = Counter()
    functions = Counter()
    suite_function_pairs = Counter()
    decisions = Counter()
    skip_reasons = Counter()
    capabilities = Counter()
    dtype_values = Counter()
    dtype_field_values = Counter()
    semantic_levels = Counter()
    semantic_decisions = {level: Counter() for level in SEMANTIC_LEVELS}
    generated_strategy_counts = Counter()
    generated_family_counts = Counter()
    coverage_kind_counts = Counter()
    surface_kind_counts = Counter()
    variant_kind_counts = Counter()
    dispatcher_presence_counts = Counter()
    coverage_id_presence_counts = Counter()
    parameterized = 0
    visible_dtype_tokens = Counter()
    visible_level_tokens = Counter()

    for record in records:
        nodeid = str(record.get("nodeid") or "")
        suite = record.get("suite") or _suite_from_nodeid(nodeid)
        function = record.get("function") or _function_from_nodeid(nodeid)
        decision = record.get("decision") or "unknown"
        level = _level_key(record.get("semantic_level"))
        suites[suite] += 1
        kinds[record.get("test_kind") or _test_kind_from_suite(suite)] += 1
        files[record.get("file") or _file_from_nodeid(nodeid)] += 1
        functions[function] += 1
        suite_function_pairs[f"{suite}::{function}"] += 1
        decisions[decision] += 1
        if record.get("skip_reason"):
            skip_reasons[str(record["skip_reason"])] += 1
        capabilities[record.get("capability") or "none"] += 1
        semantic_levels[level] += 1
        if level in semantic_decisions:
            semantic_decisions[level][decision] += 1
        dtype_value = record.get("dtype")
        if dtype_value:
            dtype_values[str(dtype_value)] += 1
        for field_name, value in (record.get("dtype_fields") or {}).items():
            if value:
                dtype_field_values[f"{field_name}:{value}"] += 1
                dtype_values[str(value)] += 1
        if record.get("strategy"):
            generated_strategy_counts[str(record["strategy"])] += 1
        if record.get("strategy_family"):
            generated_family_counts[str(record["strategy_family"])] += 1
        coverage_kind_counts[record.get("coverage_kind") or "none"] += 1
        surface_kind_counts[record.get("surface_kind") or "none"] += 1
        variant_kind_counts[record.get("variant_kind") or "none"] += 1
        dispatcher_presence_counts[_field_presence(record.get("dispatcher_name"))] += 1
        coverage_id_presence_counts[_field_presence(record.get("coverage_id"))] += 1
        if "[" in nodeid and nodeid.endswith("]"):
            parameterized += 1
        for token in re.findall(r"torch\.[A-Za-z0-9_]+", nodeid):
            visible_dtype_tokens[token] += 1
        for token in re.findall(r"\[L([1-8])\]", nodeid):
            visible_level_tokens[f"L{token}"] += 1

    return {
        "total": len(records),
        "stdout_node_count": len(nodes),
        "structured_metadata": structured_collection is not None,
        "parameterized": parameterized,
        "unparameterized": len(records) - parameterized,
        "suites": suites,
        "kinds": kinds,
        "files": files,
        "functions": functions,
        "suite_function_pairs": suite_function_pairs,
        "decisions": decisions,
        "skip_reasons": skip_reasons,
        "capabilities": capabilities,
        "dtype_values": dtype_values,
        "dtype_field_values": dtype_field_values,
        "semantic_levels": semantic_levels,
        "semantic_decisions": semantic_decisions,
        "generated_strategy_counts": generated_strategy_counts,
        "generated_family_counts": generated_family_counts,
        "coverage_kind_counts": coverage_kind_counts,
        "surface_kind_counts": surface_kind_counts,
        "variant_kind_counts": variant_kind_counts,
        "dispatcher_presence_counts": dispatcher_presence_counts,
        "coverage_id_presence_counts": coverage_id_presence_counts,
        "visible_dtype_tokens": visible_dtype_tokens,
        "visible_level_tokens": visible_level_tokens,
    }


def _coverage_stats(audit: dict) -> dict:
    entries = audit["entries"]
    metadata = audit["metadata"]
    status_counts = Counter(metadata.get("status_counts", {}))
    total = int(metadata.get("total_aten_overloads", len(entries)))
    not_backend_relevant = status_counts.get("not_backend_relevant", 0)
    runtime_unavailable = status_counts.get("unavailable_in_pytorch_runtime", 0)
    backend_relevant = total - not_backend_relevant - runtime_unavailable
    covered = sum(status_counts.get(status, 0) for status in COVERED_STATUSES)
    pending = sum(status_counts.get(status, 0) for status in PENDING_STATUSES)
    excluded = sum(status_counts.get(status, 0) for status in EXCLUDED_STATUSES)
    unknown = int(metadata.get("unknown_count", status_counts.get("unknown", 0)))
    status_family_counts = Counter({
        "covered": covered,
        "pending": pending,
        "excluded": excluded,
        "unknown": unknown,
        "not_backend_relevant": not_backend_relevant,
        "runtime_unavailable": runtime_unavailable,
    })

    variant_counts = Counter()
    tensor_io_counts = Counter()
    dispatch_counts = Counter()
    generated_strategy_counts = Counter()
    generated_family_counts = Counter()
    handwritten_marker_suite_counts = Counter()
    coverage_marker_suite_counts = Counter()
    category_marker_suite_counts = Counter()
    pending_required_closure_counts = Counter()
    pending_next_family_counts = Counter()
    pending_source_category_counts = Counter()
    exclusion_category_counts = Counter()
    exclusion_match_counts = Counter()
    exclusion_surface_counts = Counter()
    source_combo_counts = Counter()

    for entry in entries:
        variant_counts[entry.get("variant_kind") or "unknown"] += 1

        has_args = bool(entry.get("has_tensor_args"))
        has_returns = bool(entry.get("has_tensor_returns"))
        if has_args and has_returns:
            tensor_io_counts["tensor_args_and_returns"] += 1
        elif has_args:
            tensor_io_counts["tensor_args_only"] += 1
        elif has_returns:
            tensor_io_counts["tensor_returns_only"] += 1
        else:
            tensor_io_counts["no_tensor_io"] += 1

        for key in DISPATCH_KEYS:
            if entry.get("dispatch", {}).get(key):
                dispatch_counts[key] += 1

        generated = entry.get("generated") or {}
        strategy = generated.get("strategy") or {}
        if generated.get("covered"):
            generated_strategy_counts[strategy.get("strategy") or "unknown"] += 1
            generated_family_counts[strategy.get("family") or "unknown"] += 1

        handwritten = entry.get("handwritten") or {}
        if handwritten.get("covered"):
            markers = handwritten.get("markers") or []
            for marker in markers:
                coverage_marker_suite_counts[_suite_from_nodeid(marker.get("nodeid", ""))] += 1

        source_bits = []
        for key in ("opinfo", "handwritten", "generated"):
            if (entry.get(key) or {}).get("covered"):
                source_bits.append(key)
        if entry.get("oracle"):
            source_bits.append("oracle")
        if entry.get("exclusion"):
            source_bits.append("exclusion")
        if entry.get("pending_review"):
            source_bits.append("pending_review")
        source_combo_counts["+".join(source_bits) if source_bits else "none"] += 1

        pending_review = entry.get("pending_review") or {}
        if pending_review:
            pending_required_closure_counts[pending_review.get("required_closure") or "unknown"] += 1
            pending_next_family_counts[pending_review.get("next_family") or "unknown"] += 1
            pending_source_category_counts[pending_review.get("source_category") or "unknown"] += 1

        exclusion = entry.get("exclusion") or {}
        if exclusion:
            exclusion_category_counts[exclusion.get("category") or "unknown"] += 1
            exclusion_match_counts[exclusion.get("match") or "unknown"] += 1
            exclusion_surface_counts[exclusion.get("surface") or "unknown"] += 1

    for marker in audit.get("coverage_markers", []):
        handwritten_marker_suite_counts[_suite_from_nodeid(marker.get("nodeid", ""))] += 1
    for marker in audit.get("category_markers", []):
        category_marker_suite_counts[_suite_from_nodeid(marker.get("nodeid", ""))] += 1

    return {
        "total": total,
        "backend_relevant": backend_relevant,
        "covered": covered,
        "pending": pending,
        "excluded": excluded,
        "runtime_unavailable": runtime_unavailable,
        "unknown": unknown,
        "coverage_pct": _pct(covered, backend_relevant),
        "status_family_counts": status_family_counts,
        "status_counts": status_counts,
        "surface_counts": Counter(metadata.get("surface_counts", {})),
        "coverage_kind_counts": Counter(metadata.get("coverage_kind_counts", {})),
        "semantic_level_counts": metadata.get("semantic_level_counts", {}),
        "semantic_level_status_counts": metadata.get("semantic_level_status_counts", {}),
        "semantic_level_surface_counts": metadata.get("semantic_level_surface_counts", {}),
        "semantic_level_descriptions": metadata.get("semantic_level_descriptions", {}),
        "generated_case_depth": metadata.get("generated_case_depth", {}),
        "pending_blocker_counts": Counter(metadata.get("pending_blocker_counts", {})),
        "pending_backend_gate_counts": Counter(metadata.get("pending_backend_gate_counts", {})),
        "variant_counts": variant_counts,
        "tensor_io_counts": tensor_io_counts,
        "dispatch_counts": dispatch_counts,
        "generated_strategy_counts": generated_strategy_counts,
        "generated_family_counts": generated_family_counts,
        "coverage_marker_suite_counts": coverage_marker_suite_counts,
        "category_marker_suite_counts": category_marker_suite_counts,
        "pending_required_closure_counts": pending_required_closure_counts,
        "pending_next_family_counts": pending_next_family_counts,
        "pending_source_category_counts": pending_source_category_counts,
        "exclusion_category_counts": exclusion_category_counts,
        "exclusion_match_counts": exclusion_match_counts,
        "exclusion_surface_counts": exclusion_surface_counts,
        "source_combo_counts": source_combo_counts,
    }


def _known_crash_stats() -> dict:
    path = REPO_ROOT / "torchcts" / "known_segfaults.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("known_segfaults", [])
    backend_counts = Counter()
    match_counts = Counter()
    scope_counts = Counter()
    classification_counts = Counter()
    signal_counts = Counter()
    constraint_key_counts = Counter()
    constrained_count = 0

    for entry in entries:
        backend_counts[entry.get("backend") or "unknown"] += 1
        match_counts[entry.get("match") or "unknown"] += 1
        scope_counts[entry.get("evidence_scope") or "unknown"] += 1
        classification_counts[entry.get("classification") or "unknown"] += 1
        signal_counts[entry.get("expected_signal") or "unknown"] += 1
        constraints = entry.get("constraints") or {}
        if constraints:
            constrained_count += 1
            for key in constraints:
                constraint_key_counts[key] += 1

    return {
        "count": len(entries),
        "backend_counts": backend_counts,
        "match_counts": match_counts,
        "scope_counts": scope_counts,
        "classification_counts": classification_counts,
        "signal_counts": signal_counts,
        "constrained_count": constrained_count,
        "constraint_key_counts": constraint_key_counts,
    }


def _dtype_contract_stats() -> dict:
    path = REPO_ROOT / "torchcts" / "op_dtype_contracts.json"
    if not path.exists():
        return {"exists": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    contracts = data.get("contracts") or {}
    metadata = data.get("metadata") or {}
    profiles = data.get("profiles") or {}
    bucket_dtype_counts = Counter()
    version_rule_counts = Counter()
    source_condition_counts = Counter()
    mismatch_counts = Counter()
    evidence_path = REPO_ROOT / "data" / "pytorch-version-matrix" / "op_dtype_contract_evidence.jsonl"
    evidence_records = 0
    evidence_warnings = 0

    if data.get("format") == "runtime_profile_ranges":
        for ranges in contracts.values():
            if not isinstance(ranges, list):
                continue
            for range_record in ranges:
                if isinstance(range_record, list) and len(range_record) == 3:
                    version_rule_counts[f"{range_record[0]}..{range_record[1]}"] += 1
        for profile in profiles.values():
            if not isinstance(profile, dict):
                continue
            for bucket in ("cpu_supported", "cpu_unsupported", "cpu_unknown", "cpu_pending", "oracle_supported"):
                for dtypes in (profile.get(bucket) or {}).values():
                    for dtype in dtypes or ():
                        bucket_dtype_counts[f"{bucket}:{dtype}"] += 1
            for condition, dtypes in (profile.get("source_expected") or {}).items():
                source_condition_counts[str(condition)] += len(dtypes or ())
        if evidence_path.exists():
            for line in evidence_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("record_kind") == "op_contract_evidence":
                    evidence_records += 1
                    for version_entry in (record.get("versions") or {}).values():
                        for mismatch in version_entry.get("source_probe_mismatches") or ():
                            if isinstance(mismatch, dict):
                                mismatch_counts[mismatch.get("kind") or "unknown"] += 1
                            else:
                                mismatch_counts[str(mismatch)] += 1
                elif record.get("record_kind") == "warning":
                    evidence_warnings += 1

        return {
            "exists": True,
            "format": data.get("format"),
            "metadata": metadata,
            "contract_count": len(contracts),
            "contract_counts": metadata.get("contract_counts") or {},
            "last_run_probe_counts": metadata.get("last_run_probe_counts") or {},
            "source_extraction": metadata.get("source_extraction") or {},
            "version_rule_counts": version_rule_counts,
            "bucket_dtype_counts": bucket_dtype_counts,
            "source_condition_counts": source_condition_counts,
            "mismatch_counts": mismatch_counts,
            "profile_count": len(profiles),
            "range_count": metadata.get("range_count", 0),
            "artifact_size_bytes": path.stat().st_size,
            "collected_versions": metadata.get("collected_versions") or [],
            "max_validated_version": metadata.get("max_validated_version"),
            "dependency_upper_bound": metadata.get("dependency_upper_bound"),
            "evidence_exists": evidence_path.exists(),
            "evidence_records": evidence_records,
            "evidence_warnings": evidence_warnings,
        }

    for versions in contracts.values():
        if not isinstance(versions, dict):
            continue
        for version_rule, entry in versions.items():
            if not isinstance(entry, dict):
                continue
            version_rule_counts[str(version_rule)] += 1
            for bucket in ("cpu_supported", "cpu_unsupported", "cpu_unknown", "cpu_pending", "oracle_supported"):
                for dtypes in (entry.get(bucket) or {}).values():
                    for dtype in dtypes or ():
                        bucket_dtype_counts[f"{bucket}:{dtype}"] += 1
            for condition, dtypes in (entry.get("source_expected") or {}).items():
                source_condition_counts[str(condition)] += len(dtypes or ())
            for mismatch in entry.get("source_probe_mismatches") or ():
                if isinstance(mismatch, dict):
                    mismatch_counts[mismatch.get("kind") or "unknown"] += 1
                else:
                    mismatch_counts[str(mismatch)] += 1

    return {
        "exists": True,
        "metadata": metadata,
        "contract_count": len(contracts),
        "contract_counts": metadata.get("contract_counts") or {},
        "last_run_probe_counts": metadata.get("last_run_probe_counts") or {},
        "source_extraction": metadata.get("source_extraction") or {},
        "version_rule_counts": version_rule_counts,
        "bucket_dtype_counts": bucket_dtype_counts,
        "source_condition_counts": source_condition_counts,
        "mismatch_counts": mismatch_counts,
    }


def _append_counter_section(lines: list[str], title: str, counter: Counter, *, limit: int | None = None) -> None:
    lines.append(f"## {title}")
    lines.append("")
    if counter:
        rows = _counter_rows(counter, limit=limit)
        lines.extend(_table(["Name", "Count"], rows))
    else:
        lines.append("No entries.")
    lines.append("")


def _append_mapping_section(lines: list[str], title: str, mapping: dict, *, numeric_keys: bool = False) -> None:
    lines.append(f"## {title}")
    lines.append("")
    if mapping:
        lines.extend(_table(["Name", "Count"], _sorted_mapping_rows(mapping, numeric_keys=numeric_keys)))
    else:
        lines.append("No entries.")
    lines.append("")


def _append_semantic_level_count_table(lines: list[str], title: str, counts: dict | Counter) -> None:
    lines.append(f"## {title}")
    lines.append("")
    lines.extend(_table(
        ["Level", "Count"],
        [[level, counts.get(level, counts.get(int(level), 0))] for level in SEMANTIC_LEVELS],
    ))
    lines.append("")


def _append_semantic_level_counter_sections(lines: list[str], title: str, counters: dict) -> None:
    lines.append(f"## {title}")
    lines.append("")
    for level in SEMANTIC_LEVELS:
        lines.append(f"### Level {level}")
        lines.append("")
        counts = counters.get(level) or counters.get(int(level)) or {}
        if counts:
            lines.extend(_table(["Name", "Count"], _sorted_mapping_rows(counts)))
        else:
            lines.append("No entries.")
        lines.append("")


def _semantic_level_overview_rows(collection_stats: dict | None, coverage: dict, generated_depth: dict) -> list[list[object]]:
    coverage_counts = coverage["semantic_level_counts"]
    generated_counts = generated_depth.get("by_semantic_level", {})
    descriptions = coverage["semantic_level_descriptions"]
    rows = []
    for level in SEMANTIC_LEVELS:
        decisions = (collection_stats or {}).get("semantic_decisions", {}).get(level, Counter())
        pytest_nodes = (collection_stats or {}).get("semantic_levels", {}).get(level, "not collected")
        rows.append([
            level,
            pytest_nodes,
            decisions.get("executable", "not collected" if collection_stats is None else 0),
            decisions.get("pytest_skip_marked", "not collected" if collection_stats is None else 0),
            decisions.get("structured_deselected", "not collected" if collection_stats is None else 0),
            coverage_counts.get(level, coverage_counts.get(int(level), 0)),
            generated_counts.get(level, generated_counts.get(int(level), 0)),
            descriptions.get(level, ""),
        ])
    return rows


def render_markdown(*, audit: dict, collection: dict | None, include_collect: bool) -> str:
    coverage = _coverage_stats(audit)
    known_crashes = _known_crash_stats()
    dtype_contracts = _dtype_contract_stats()
    collection_stats = (
        _collection_stats(collection["nodes"], collection.get("structured_collection")) if collection else None
    )
    version_info = _version_provenance()
    generated_depth = coverage["generated_case_depth"]
    now = _datetime.datetime.now(_datetime.timezone.utc).isoformat().replace("+00:00", "Z")

    lines: list[str] = []
    lines.append("# TorchCTS Site Stats")
    lines.append("")
    lines.append("This generated file is a statistics source for website copy and AI-assisted site updates.")
    lines.append("It describes the current checkout and installed PyTorch build; it is not a backend pass/fail report.")
    lines.append("")
    lines.extend(_table(
        ["Field", "Value"],
        [
            ["Generated at", now],
            ["TorchCTS version", version_info["module_version"]],
            ["Installed TorchCTS metadata version", version_info["installed_version"]],
            ["pyproject.toml version", version_info["pyproject_version"]],
            ["TorchCTS versions agree", version_info["versions_agree"]],
            ["TorchCTS import path", version_info["import_path"]],
            ["PyTorch version", torch.__version__],
            ["Python version", platform.python_version()],
            ["Platform", platform.platform()],
            ["Coverage audit timestamp", audit["metadata"].get("generated_at")],
            ["Pytest collection included", "yes" if include_collect else "no"],
        ],
    ))
    lines.append("")
    if not version_info["versions_agree"]:
        lines.append("> Warning: TorchCTS version sources disagree. Reinstall the editable package or check import precedence before publishing version-sensitive site copy.")
        lines.append("")

    headline_rows = [
        ["Pytest nodes collected", collection_stats["total"] if collection_stats else "not collected"],
        ["Pytest executable nodes", collection_stats["decisions"].get("executable", 0) if collection_stats else "not collected"],
        ["Pytest skip-marked nodes", collection_stats["decisions"].get("pytest_skip_marked", 0) if collection_stats else "not collected"],
        ["Structured deselected nodes", collection_stats["decisions"].get("structured_deselected", 0) if collection_stats else "not collected"],
        ["ATen overloads inventoried", coverage["total"]],
        ["Backend-relevant overloads", coverage["backend_relevant"]],
        ["Covered backend-relevant overloads", coverage["covered"]],
        ["Dispatcher coverage", coverage["coverage_pct"]],
        ["Unknown tensor-touching surfaces", coverage["unknown"]],
        ["Pending surfaces", coverage["pending"]],
        ["Excluded surfaces", coverage["excluded"]],
        ["Runtime-unavailable overloads", coverage["runtime_unavailable"]],
        ["Generated coverage surfaces", coverage["coverage_kind_counts"].get("generated", 0)],
        ["Generated dispatcher semantic cases", generated_depth.get("generated_semantic_cases", 0)],
        ["Required generated dispatcher semantic cases", generated_depth.get("required_generated_semantic_cases", 0)],
        ["Known crash isolation rules", known_crashes["count"]],
        ["CPU dtype contract records", dtype_contracts.get("contract_count", "not found")],
    ]
    lines.append("## Headline Stats")
    lines.append("")
    lines.extend(_table(["Metric", "Value"], headline_rows))
    lines.append("")

    lines.append("## Semantic Level Overview")
    lines.append("")
    lines.append("This table combines pytest collection inventory with dispatcher coverage inventory. Level 7 and 8 currently live primarily in handwritten workload, multi-device, and stress tests, so generated-dispatcher counts can be zero while pytest nodes are nonzero.")
    lines.append("")
    lines.extend(_table(
        [
            "Level",
            "Pytest nodes",
            "Executable nodes",
            "Pytest skip-marked nodes",
            "Structured deselected nodes",
            "Coverage surfaces",
            "Generated dispatcher cases",
            "Description",
        ],
        _semantic_level_overview_rows(collection_stats, coverage, generated_depth),
    ))
    lines.append("")

    if collection_stats:
        lines.append("## Pytest Collection Summary")
        lines.append("")
        command = " ".join(collection.get("command_display", collection["command"]))
        summary_count = collection.get("collected_from_summary")
        rows = [
            ["Collection command", f"`{command}`"],
            ["Structured collection metadata", "yes" if collection_stats["structured_metadata"] else "no"],
            ["Structured records parsed", collection_stats["total"]],
            ["Node IDs parsed from stdout", collection_stats["stdout_node_count"]],
            ["Pytest summary count", summary_count if summary_count is not None else "not found"],
            ["Parameterized node IDs", collection_stats["parameterized"]],
            ["Unparameterized node IDs", collection_stats["unparameterized"]],
        ]
        lines.extend(_table(["Metric", "Value"], rows))
        lines.append("")
        _append_counter_section(lines, "Pytest Collection Decisions", collection_stats["decisions"])
        _append_counter_section(lines, "Pytest Collection Skip Reasons", collection_stats["skip_reasons"])
        _append_counter_section(lines, "Pytest Nodes By Suite", collection_stats["suites"])
        _append_counter_section(lines, "Pytest Nodes By Test Kind", collection_stats["kinds"])
        _append_counter_section(lines, "Pytest Nodes By File", collection_stats["files"])
        _append_counter_section(lines, "Top Pytest Test Functions", collection_stats["functions"], limit=75)
        _append_counter_section(lines, "Top Suite And Function Pairs", collection_stats["suite_function_pairs"], limit=75)
        _append_counter_section(lines, "Visible Dtype Tokens In Node IDs", collection_stats["visible_dtype_tokens"])
        _append_counter_section(lines, "Visible Generated Level Tokens In Node IDs", collection_stats["visible_level_tokens"])
        _append_counter_section(lines, "Collection Nodes By Capability", collection_stats["capabilities"])
        _append_counter_section(lines, "Collection Nodes By Dtype", collection_stats["dtype_values"])
        _append_counter_section(lines, "Collection Dtype Field Counts", collection_stats["dtype_field_values"])
        _append_counter_section(lines, "Collection Nodes By Coverage Kind", collection_stats["coverage_kind_counts"])
        _append_counter_section(lines, "Collection Nodes By Surface Kind", collection_stats["surface_kind_counts"])
        _append_counter_section(lines, "Collection Nodes By Variant Kind", collection_stats["variant_kind_counts"])
        _append_counter_section(lines, "Collection Generated Nodes By Strategy", collection_stats["generated_strategy_counts"])
        _append_counter_section(lines, "Collection Generated Nodes By Strategy Family", collection_stats["generated_family_counts"])
        _append_counter_section(lines, "Collection Dispatcher Name Presence", collection_stats["dispatcher_presence_counts"])
        _append_counter_section(lines, "Collection Coverage ID Presence", collection_stats["coverage_id_presence_counts"])
        _append_semantic_level_count_table(lines, "Pytest Nodes By Semantic Level", collection_stats["semantic_levels"])
        _append_semantic_level_counter_sections(lines, "Pytest Collection Decisions By Semantic Level", collection_stats["semantic_decisions"])
    else:
        lines.append("## Pytest Collection Summary")
        lines.append("")
        lines.append("Pytest collection was skipped. Re-run without `--no-collect` to include test-node breakdowns.")
        lines.append("")

    lines.append("## Dispatcher Coverage Summary")
    lines.append("")
    lines.extend(_table(
        ["Metric", "Value"],
        [
            ["ATen overloads", coverage["total"]],
            ["Backend-relevant overloads", coverage["backend_relevant"]],
            ["Covered backend-relevant overloads", coverage["covered"]],
            ["Coverage percent", coverage["coverage_pct"]],
            ["Pending surfaces", coverage["pending"]],
            ["Excluded surfaces", coverage["excluded"]],
            ["Runtime-unavailable overloads", coverage["runtime_unavailable"]],
            ["Unknown surfaces", coverage["unknown"]],
        ],
    ))
    lines.append("")

    _append_counter_section(lines, "Coverage Status Counts", coverage["status_counts"])
    _append_counter_section(lines, "Coverage Status Family Counts", coverage["status_family_counts"])
    _append_counter_section(lines, "Coverage Kind Counts", coverage["coverage_kind_counts"])
    _append_counter_section(lines, "Surface Kind Counts", coverage["surface_counts"])
    _append_counter_section(lines, "Variant Kind Counts", coverage["variant_counts"])
    _append_counter_section(lines, "Tensor Input And Return Shape Counts", coverage["tensor_io_counts"])
    _append_counter_section(lines, "Dispatch Key Availability Counts", coverage["dispatch_counts"])
    _append_counter_section(lines, "Coverage Source Combination Counts", coverage["source_combo_counts"])

    _append_semantic_level_count_table(lines, "Coverage Surfaces By Semantic Level", coverage["semantic_level_counts"])
    lines.append("## Semantic Level Descriptions")
    lines.append("")
    lines.extend(_table(
        ["Level", "Description"],
        [
            [level, coverage["semantic_level_descriptions"].get(level, "")]
            for level in SEMANTIC_LEVELS
        ],
    ))
    lines.append("")

    _append_semantic_level_counter_sections(lines, "Coverage Surfaces By Semantic Level And Status", coverage["semantic_level_status_counts"])
    _append_semantic_level_counter_sections(lines, "Coverage Surfaces By Semantic Level And Surface Kind", coverage["semantic_level_surface_counts"])

    lines.append("## Generated Coverage Depth")
    lines.append("")
    lines.extend(_table(
        ["Metric", "Value"],
        [
            ["Generated surfaces with case plans", generated_depth.get("generated_surfaces_with_case_plan", 0)],
            ["Generated semantic cases", generated_depth.get("generated_semantic_cases", 0)],
            ["Required generated semantic cases", generated_depth.get("required_generated_semantic_cases", 0)],
            ["Optional generated semantic cases", generated_depth.get("optional_generated_semantic_cases", 0)],
        ],
    ))
    lines.append("")
    _append_mapping_section(lines, "Generated Dispatcher Cases By Strategy", generated_depth.get("by_strategy", {}))
    _append_semantic_level_count_table(lines, "Generated Dispatcher Cases By Semantic Level", generated_depth.get("by_semantic_level", {}))
    _append_counter_section(lines, "Generated Covered Surfaces By Strategy", coverage["generated_strategy_counts"])
    _append_counter_section(lines, "Generated Covered Surfaces By Strategy Family", coverage["generated_family_counts"])

    lines.append("## CPU Dtype Contract Stats")
    lines.append("")
    if dtype_contracts.get("exists"):
        contract_counts = dtype_contracts["contract_counts"]
        source_extraction = dtype_contracts["source_extraction"]
        lines.extend(_table(
            ["Metric", "Value"],
            [
                ["Contracted dispatcher entries", dtype_contracts["contract_count"]],
                ["Runtime contract format", dtype_contracts.get("format", "expanded")],
                ["Runtime contract artifact bytes", dtype_contracts.get("artifact_size_bytes", "unknown")],
                ["Runtime dtype profiles", dtype_contracts.get("profile_count", "unknown")],
                ["Runtime profile ranges", dtype_contracts.get("range_count", "unknown")],
                ["Collected PyTorch versions", ", ".join(dtype_contracts.get("collected_versions") or [])],
                ["Max validated PyTorch version", dtype_contracts.get("max_validated_version") or "unknown"],
                ["PyTorch dependency upper bound", dtype_contracts.get("dependency_upper_bound") or "unknown"],
                ["Source evidence present", dtype_contracts.get("evidence_exists", False)],
                ["Source evidence op records", dtype_contracts.get("evidence_records", 0)],
                ["Source evidence warnings", dtype_contracts.get("evidence_warnings", 0)],
                ["CPU-supported dtype cases", contract_counts.get("cpu_supported", 0)],
                ["CPU-unsupported dtype cases", contract_counts.get("cpu_unsupported", 0)],
                ["CPU-pending dtype cases", contract_counts.get("cpu_pending", 0)],
                ["CPU-unknown dtype cases", contract_counts.get("cpu_unknown", 0)],
                ["Oracle-supported dtype cases", contract_counts.get("oracle_supported", 0)],
                ["Source-expected ops", contract_counts.get("source_expected_ops", 0)],
                ["Source-expected dtype entries", contract_counts.get("source_expected_entries", 0)],
                ["Source/probe mismatches", contract_counts.get("source_probe_mismatches", 0)],
                ["Local PyTorch source available", source_extraction.get("pytorch_src_available", False)],
                ["Local PyTorch ufunc source entries", source_extraction.get("pytorch_src_ufunc_inner_loop_seeded_ops", 0)],
            ],
        ))
        lines.append("")
        _append_mapping_section(lines, "CPU Dtype Contract Last Run Probe Counts", dtype_contracts["last_run_probe_counts"])
        _append_counter_section(lines, "CPU Dtype Contract Version Rules", dtype_contracts["version_rule_counts"])
        _append_counter_section(lines, "CPU Dtype Contract Buckets By Dtype", dtype_contracts["bucket_dtype_counts"])
        _append_counter_section(lines, "CPU Dtype Contract Source Conditions", dtype_contracts["source_condition_counts"])
        _append_counter_section(lines, "CPU Dtype Contract Source Probe Mismatches", dtype_contracts["mismatch_counts"])
    else:
        lines.append("No dtype contract artifact found.")
        lines.append("")

    lines.append("## Marker And Source Coverage Stats")
    lines.append("")
    lines.extend(_table(
        ["Metric", "Value"],
        [
            ["Coverage markers discovered", len(audit.get("coverage_markers", []))],
            ["Category markers discovered", len(audit.get("category_markers", []))],
            ["Unmapped hand-authored tests", len(audit.get("unmapped_tests", []))],
            ["Audit warnings", len(audit.get("warnings", []))],
            ["Audit errors", len(audit.get("errors", []))],
        ],
    ))
    lines.append("")
    _append_counter_section(lines, "Coverage Markers By Suite", coverage["coverage_marker_suite_counts"])
    _append_counter_section(lines, "Category Markers By Suite", coverage["category_marker_suite_counts"])

    _append_counter_section(lines, "Pending Blocker Counts", coverage["pending_blocker_counts"])
    _append_counter_section(lines, "Pending Backend Gate Counts", coverage["pending_backend_gate_counts"])
    _append_counter_section(lines, "Pending Required Closure Counts", coverage["pending_required_closure_counts"])
    _append_counter_section(lines, "Pending Next Family Counts", coverage["pending_next_family_counts"])
    _append_counter_section(lines, "Pending Source Category Counts", coverage["pending_source_category_counts"])

    _append_counter_section(lines, "Exclusion Category Counts", coverage["exclusion_category_counts"])
    _append_counter_section(lines, "Exclusion Match Counts", coverage["exclusion_match_counts"])
    _append_counter_section(lines, "Exclusion Surface Counts", coverage["exclusion_surface_counts"])

    lines.append("## Known Crash Isolation Stats")
    lines.append("")
    lines.extend(_table(
        ["Metric", "Value"],
        [
            ["Rules", known_crashes["count"]],
            ["Rules with constraints", known_crashes["constrained_count"]],
        ],
    ))
    lines.append("")
    _append_counter_section(lines, "Known Crash Rules By Backend", known_crashes["backend_counts"])
    _append_counter_section(lines, "Known Crash Rules By Match Mode", known_crashes["match_counts"])
    _append_counter_section(lines, "Known Crash Rules By Evidence Scope", known_crashes["scope_counts"])
    _append_counter_section(lines, "Known Crash Rules By Classification", known_crashes["classification_counts"])
    _append_counter_section(lines, "Known Crash Rules By Expected Signal", known_crashes["signal_counts"])
    _append_counter_section(lines, "Known Crash Constraint Key Counts", known_crashes["constraint_key_counts"])

    lines.append("## Website Interpretation Notes")
    lines.append("")
    lines.append("- Use coverage and collection numbers as current-checkout statistics, not universal PyTorch promises.")
    lines.append("- Pytest collection stats describe TorchCTS test inventory and selection, not backend pass/fail results.")
    lines.append("- `executable`, `pytest_skip_marked`, and `structured_deselected` are distinct collection decisions.")
    lines.append("- `unknown=0` means TorchCTS has an explicit disposition for every tensor-touching backend-relevant ATen surface in this audit.")
    lines.append("- Pending backend-pack counts are intentional hardware/build gates, not claimed coverage.")
    lines.append("- Known crash rules are subprocess isolation policy only; they do not skip, xfail, or downgrade failures.")
    lines.append("- Re-run this script after changing tests, generated coverage, coverage exclusions, known crash rules, or PyTorch versions.")
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Markdown output path. Default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--no-collect",
        action="store_true",
        help="Skip pytest collect-only stats and generate coverage/ledger stats only.",
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Extra argument to append to the pytest collect-only command. Repeatable.",
    )
    args = parser.parse_args(argv)

    audit = build_audit()
    collection = None
    if not args.no_collect:
        collection = _collect_pytest_nodes(args.pytest_arg)

    markdown = render_markdown(
        audit=audit,
        collection=collection,
        include_collect=not args.no_collect,
    )
    output = args.output
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(f"Wrote site stats to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
