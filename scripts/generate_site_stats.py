#!/usr/bin/env python3
"""Generate a Markdown stats source for website copy and AI-assisted site work."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as _datetime
import json
import platform
import re
import subprocess
import sys
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
    items = counter.most_common()
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


def _collect_pytest_nodes(extra_pytest_args: list[str]) -> dict:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "torchcts",
        "--validation",
        *extra_pytest_args,
    ]
    command_display = [
        "python",
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "torchcts",
        "--validation",
        *extra_pytest_args,
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
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
    }


def _collection_stats(nodes: list[str]) -> dict:
    suites = Counter()
    kinds = Counter()
    files = Counter()
    functions = Counter()
    suite_function_pairs = Counter()
    parameterized = 0
    visible_dtype_tokens = Counter()
    visible_level_tokens = Counter()

    for nodeid in nodes:
        suite = _suite_from_nodeid(nodeid)
        function = _function_from_nodeid(nodeid)
        suites[suite] += 1
        kinds[_test_kind_from_suite(suite)] += 1
        files[_file_from_nodeid(nodeid)] += 1
        functions[function] += 1
        suite_function_pairs[f"{suite}::{function}"] += 1
        if "[" in nodeid and nodeid.endswith("]"):
            parameterized += 1
        for token in re.findall(r"torch\.[A-Za-z0-9_]+", nodeid):
            visible_dtype_tokens[token] += 1
        for token in re.findall(r"\[L([1-8])\]", nodeid):
            visible_level_tokens[f"L{token}"] += 1

    return {
        "total": len(nodes),
        "parameterized": parameterized,
        "unparameterized": len(nodes) - parameterized,
        "suites": suites,
        "kinds": kinds,
        "files": files,
        "functions": functions,
        "suite_function_pairs": suite_function_pairs,
        "visible_dtype_tokens": visible_dtype_tokens,
        "visible_level_tokens": visible_level_tokens,
    }


def _coverage_stats(audit: dict) -> dict:
    entries = audit["entries"]
    metadata = audit["metadata"]
    status_counts = Counter(metadata.get("status_counts", {}))
    total = int(metadata.get("total_aten_overloads", len(entries)))
    not_backend_relevant = status_counts.get("not_backend_relevant", 0)
    backend_relevant = total - not_backend_relevant
    covered = sum(status_counts.get(status, 0) for status in COVERED_STATUSES)
    pending = sum(status_counts.get(status, 0) for status in PENDING_STATUSES)
    excluded = sum(status_counts.get(status, 0) for status in EXCLUDED_STATUSES)

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
        "unknown": int(metadata.get("unknown_count", status_counts.get("unknown", 0))),
        "coverage_pct": _pct(covered, backend_relevant),
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
    bucket_dtype_counts = Counter()
    version_rule_counts = Counter()
    source_condition_counts = Counter()
    mismatch_counts = Counter()

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


def render_markdown(*, audit: dict, collection: dict | None, include_collect: bool) -> str:
    coverage = _coverage_stats(audit)
    known_crashes = _known_crash_stats()
    dtype_contracts = _dtype_contract_stats()
    collection_stats = _collection_stats(collection["nodes"]) if collection else None
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
            ["TorchCTS version", torchcts.__version__],
            ["PyTorch version", torch.__version__],
            ["Python version", platform.python_version()],
            ["Platform", platform.platform()],
            ["Coverage audit timestamp", audit["metadata"].get("generated_at")],
            ["Pytest collection included", "yes" if include_collect else "no"],
        ],
    ))
    lines.append("")

    headline_rows = [
        ["Pytest nodes collected", collection_stats["total"] if collection_stats else "not collected"],
        ["ATen overloads inventoried", coverage["total"]],
        ["Backend-relevant overloads", coverage["backend_relevant"]],
        ["Covered backend-relevant overloads", coverage["covered"]],
        ["Dispatcher coverage", coverage["coverage_pct"]],
        ["Unknown tensor-touching surfaces", coverage["unknown"]],
        ["Pending surfaces", coverage["pending"]],
        ["Excluded surfaces", coverage["excluded"]],
        ["Generated coverage surfaces", coverage["coverage_kind_counts"].get("generated", 0)],
        ["Generated semantic cases", generated_depth.get("generated_semantic_cases", 0)],
        ["Required generated semantic cases", generated_depth.get("required_generated_semantic_cases", 0)],
        ["Known crash isolation rules", known_crashes["count"]],
        ["CPU dtype contract records", dtype_contracts.get("contract_count", "not found")],
    ]
    lines.append("## Headline Stats")
    lines.append("")
    lines.extend(_table(["Metric", "Value"], headline_rows))
    lines.append("")

    if collection_stats:
        lines.append("## Pytest Collection Summary")
        lines.append("")
        command = " ".join(collection.get("command_display", collection["command"]))
        summary_count = collection.get("collected_from_summary")
        rows = [
            ["Collection command", f"`{command}`"],
            ["Node IDs parsed", collection_stats["total"]],
            ["Pytest summary count", summary_count if summary_count is not None else "not found"],
            ["Parameterized node IDs", collection_stats["parameterized"]],
            ["Unparameterized node IDs", collection_stats["unparameterized"]],
        ]
        lines.extend(_table(["Metric", "Value"], rows))
        lines.append("")
        _append_counter_section(lines, "Pytest Nodes By Suite", collection_stats["suites"])
        _append_counter_section(lines, "Pytest Nodes By Test Kind", collection_stats["kinds"])
        _append_counter_section(lines, "Pytest Nodes By File", collection_stats["files"])
        _append_counter_section(lines, "Top Pytest Test Functions", collection_stats["functions"], limit=75)
        _append_counter_section(lines, "Top Suite And Function Pairs", collection_stats["suite_function_pairs"], limit=75)
        _append_counter_section(lines, "Visible Dtype Tokens In Node IDs", collection_stats["visible_dtype_tokens"])
        _append_counter_section(lines, "Visible Generated Level Tokens In Node IDs", collection_stats["visible_level_tokens"])
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
            ["Unknown surfaces", coverage["unknown"]],
        ],
    ))
    lines.append("")

    _append_counter_section(lines, "Coverage Status Counts", coverage["status_counts"])
    _append_counter_section(lines, "Coverage Kind Counts", coverage["coverage_kind_counts"])
    _append_counter_section(lines, "Surface Kind Counts", coverage["surface_counts"])
    _append_counter_section(lines, "Variant Kind Counts", coverage["variant_counts"])
    _append_counter_section(lines, "Tensor Input And Return Shape Counts", coverage["tensor_io_counts"])
    _append_counter_section(lines, "Dispatch Key Availability Counts", coverage["dispatch_counts"])
    _append_counter_section(lines, "Coverage Source Combination Counts", coverage["source_combo_counts"])

    _append_mapping_section(lines, "Semantic Level Counts", coverage["semantic_level_counts"], numeric_keys=True)
    lines.append("## Semantic Level Descriptions")
    lines.append("")
    lines.extend(_table(
        ["Level", "Description"],
        _sorted_mapping_rows(coverage["semantic_level_descriptions"], numeric_keys=True),
    ))
    lines.append("")

    lines.append("## Semantic Level By Status")
    lines.append("")
    for level, counts in sorted(coverage["semantic_level_status_counts"].items(), key=lambda item: int(item[0])):
        lines.append(f"### Level {level}")
        lines.append("")
        lines.extend(_table(["Status", "Count"], _sorted_mapping_rows(counts)))
        lines.append("")

    lines.append("## Semantic Level By Surface Kind")
    lines.append("")
    for level, counts in sorted(coverage["semantic_level_surface_counts"].items(), key=lambda item: int(item[0])):
        lines.append(f"### Level {level}")
        lines.append("")
        lines.extend(_table(["Surface kind", "Count"], _sorted_mapping_rows(counts)))
        lines.append("")

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
    _append_mapping_section(lines, "Generated Semantic Cases By Strategy", generated_depth.get("by_strategy", {}))
    _append_mapping_section(lines, "Generated Semantic Cases By Semantic Level", generated_depth.get("by_semantic_level", {}), numeric_keys=True)
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

    lines.append("## Notes For Website Use")
    lines.append("")
    lines.append("- Use coverage and collection numbers as current-checkout statistics, not universal PyTorch promises.")
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
