# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import json
import sys
import datetime
import re
import torch
import psutil
from torchcts.core.manifest_schema import CAPABILITY_ORDER, KNOWN_CAPABILITIES

_BACKEND_MANIFEST_DECLINED_SKIP_REASONS = frozenset({
    "dtype_not_supported",
    "dtype_regex_filtered",
    "dtype_not_listed",
    "capability_not_declared",
    "op_excluded",
    "container_format_not_supported",
    "custom_container_decoder_not_declared",
    "generated_no_manifest_enabled_dtypes",
    "resource_limit",
    "device_count",
    "no_device_module",
    "set_device_not_supported",
    "oom_not_recoverable",
})
_SELECTION_SKIP_REASONS = frozenset({
    "semantic_level_gt_requested",
    "semantic_level_out_of_range",
    "opinfo_no_selected_cases",
    "backend_gate_mismatch",
    "path_shape_no_selected_cases",
})
_CONTRACT_SKIP_REASONS = frozenset({
    "cpu_contract_unsupported",
    "cpu_contract_unknown",
    "cpu_contract_pending",
})
_COVERAGE_DEBT_SKIP_REASONS = frozenset({
    "coverage_unknown",
    "coverage_excluded",
    "coverage_strategy_pending",
    "pending_property",
    "backend_not_available",
    "unavailable_in_pytorch_runtime",
    "excluded",
    "excluded_framework_plumbing",
    "excluded_deprecated_or_removed",
    "excluded_unsupported_public_api",
    "excluded_distributed_scope",
    "excluded_host_storage",
})
_BACKEND_PACK_DEBT_SKIP_REASONS = frozenset({
    "pending_backend_pack",
})
_KNOWN_UNSAFE_SKIP_REASONS = frozenset({
    "framework_bug",
    "known_backend_crash",
})
_SEMANTIC_RUNTIME_SKIP_PREFIXES = (
    "cpu_contract_unsupported",
    "cpu_contract_unknown",
    "cpu_contract_pending",
    "coverage_strategy_pending",
    "pending_property",
    "pending_backend_pack",
    "backend_not_available",
    "framework_bug",
)
_BACKEND_MANIFEST_ASSERTION_BAR_WIDTH = 37
_MIN_FULL_RUN_MANIFEST_TEST_COUNT = 19000


def _skip_detail(record):
    return str(record.get("detail") or record.get("skip_detail") or record.get("error_message") or "")


def _effective_skip_reason(record):
    reason = record.get("skip_reason", "unknown")
    if reason == "runtime_skip":
        detail = _skip_detail(record).strip()
        for prefix in _SEMANTIC_RUNTIME_SKIP_PREFIXES:
            if detail == prefix or detail.startswith(f"{prefix}:"):
                return prefix
    return reason


def _backend_manifest_assertion_from_data(metadata, results, skips_dict):
    stored = metadata.get("backend_manifest_assertion")
    if isinstance(stored, dict) and stored.get("total_runnable_test_count") is not None:
        try:
            asserted = int(stored.get("asserted_test_count", 0) or 0)
            total = int(stored.get("total_runnable_test_count", 0) or 0)
        except (TypeError, ValueError):
            asserted = 0
            total = 0
        declined = max(total - asserted, 0)
        fraction = asserted / total if total else 0.0
        return {
            "asserted_test_count": asserted,
            "declined_test_count": declined,
            "total_runnable_test_count": total,
            "asserted_fraction": fraction,
            "asserted_percent": fraction * 100.0,
            "progress_bar_width": int(stored.get("progress_bar_width") or _BACKEND_MANIFEST_ASSERTION_BAR_WIDTH),
        }

    declined = sum(
        1
        for record in skips_dict.values()
        if record.get("skip_reason") in _BACKEND_MANIFEST_DECLINED_SKIP_REASONS
    )
    asserted = len(results)
    total = asserted + declined
    fraction = asserted / total if total else 0.0
    return {
        "asserted_test_count": asserted,
        "declined_test_count": declined,
        "total_runnable_test_count": total,
        "asserted_fraction": fraction,
        "asserted_percent": fraction * 100.0,
        "progress_bar_width": _BACKEND_MANIFEST_ASSERTION_BAR_WIDTH,
    }


def _backend_manifest_assertion_lines(summary, metadata=None):
    metadata = metadata or {}
    asserted = int(summary.get("asserted_test_count", 0) or 0)
    total = int(summary.get("total_runnable_test_count", 0) or 0)
    unavailable_reasons = []
    if metadata.get("session_completed") is False:
        unavailable_reasons.append(
            "This run did not complete; use a completed run before publishing backend support percentages."
        )
    if metadata.get("collect_only") is True:
        unavailable_reasons.append("This was a collection-only session; no tests executed.")
    if total < _MIN_FULL_RUN_MANIFEST_TEST_COUNT:
        unavailable_reasons.append(
            f"A full-run scorecard requires at least {_MIN_FULL_RUN_MANIFEST_TEST_COUNT} runnable tests."
        )
        unavailable_reasons.append(
            "This is a partial, interrupted, or failed collection report."
        )
    if unavailable_reasons:
        lines = [
            "  Backend manifest support scorecard unavailable for this run.",
            f"  Scored {asserted} out of {total} selected/declined tests.",
        ]
        lines.extend(f"  {reason}" for reason in unavailable_reasons)
        lines.append("  No backend support percentage is shown.")
        lines.append("")
        return lines
    fraction = float(summary.get("asserted_fraction", 0.0) or 0.0)
    percent = float(summary.get("asserted_percent", fraction * 100.0) or 0.0)
    width = int(summary.get("progress_bar_width") or _BACKEND_MANIFEST_ASSERTION_BAR_WIDTH)
    width = max(width, 1)
    filled = min(width, max(0, int((width * fraction) + 0.5)))
    bar = "[" + ("█" * filled) + ("░" * (width - filled)) + "]"
    return [
        "  Backend asserts via manifest.py that it supports:",
        f"  {asserted} out of {total} tests / {percent:.2f}%",
        f"  {bar}",
        "",
    ]


def get_hardware_key(device_name, manifest=None):
    if device_name == "cuda" and torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0).replace(" ", "_")
        except:
            name = "cuda"
        name = re.sub(r'[^a-zA-Z0-9_]', '', name)
        
        # Get memory from manifest or device properties
        mem_gb = 0
        if manifest and "hardware" in manifest and "device_memory_gb" in manifest.get("hardware", {}):
            mem_gb = manifest["hardware"]["device_memory_gb"][0]
        else:
            try:
                mem_gb = int(torch.cuda.get_device_properties(0).total_memory / (1024**3))
            except:
                pass
        return f"{name}_{mem_gb}gb"
        
    elif device_name == "mps":
        import platform
        cpu_brand = ""
        try:
            import subprocess
            cpu_brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        except:
            cpu_brand = platform.processor() or "Apple_Silicon"
        cpu_brand = cpu_brand.replace(" ", "_").replace("(", "").replace(")", "").replace("@", "")
        cpu_brand = re.sub(r'[^a-zA-Z0-9_]', '', cpu_brand)
        
        mem_gb = 0
        if manifest and "hardware" in manifest and "system_memory_gb" in manifest.get("hardware", {}):
            mem_gb = manifest["hardware"]["system_memory_gb"]
        else:
            try:
                mem_gb = int(psutil.virtual_memory().total / (1024**3))
            except:
                pass
        return f"{cpu_brand}_{mem_gb}gb"
        
    else:
        # Default fallback
        import platform
        node = platform.node().replace(" ", "_")
        node = re.sub(r'[^a-zA-Z0-9_]', '', node)
        
        mem_gb = 0
        if manifest and "hardware" in manifest and "system_memory_gb" in manifest.get("hardware", {}):
            mem_gb = manifest["hardware"]["system_memory_gb"]
        else:
            try:
                mem_gb = int(psutil.virtual_memory().total / (1024**3))
            except:
                pass
        return f"{device_name}_{node}_{mem_gb}gb"

def build_report(current_data, baseline_data=None, include_skips=False):
    metadata = current_data.get("metadata", {})
    device = metadata.get("device_name", "unknown")
    hw_key = metadata.get("hardware_key", "unknown")
    pytorch_version = metadata.get("pytorch_version", torch.__version__)
    timestamp = metadata.get("timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"))
    elapsed = str(datetime.timedelta(seconds=int(metadata.get("elapsed_sec", 0))))
    collect_only = metadata.get("collect_only", False)

    results = current_data.get("results", {})
    skips_dict = current_data.get("skips", {})
    backend_manifest_assertion = _backend_manifest_assertion_from_data(metadata, results, skips_dict)
    manifest_skip_reasons = _BACKEND_MANIFEST_DECLINED_SKIP_REASONS
    selection_skip_reasons = _SELECTION_SKIP_REASONS
    coverage_skip_reasons = _COVERAGE_DEBT_SKIP_REASONS
    contract_skip_reasons = _CONTRACT_SKIP_REASONS
    backend_pack_skip_reasons = _BACKEND_PACK_DEBT_SKIP_REASONS
    known_unsafe_skip_reasons = _KNOWN_UNSAFE_SKIP_REASONS
    runtime_skip_reasons = {
        "runtime_skip",
    }
    dtype_skip_reasons = {
        "dtype_not_supported",
        "dtype_regex_filtered",
        "dtype_not_listed",
        "cpu_contract_unsupported",
        "cpu_contract_unknown",
        "cpu_contract_pending",
    }

    not_run_bucket_labels = {
        "manifest": "Ops with not-run cases (manifest)",
        "selection": "Ops with not-run cases (selection)",
        "coverage": "Ops with not-run cases (coverage)",
        "contract": "Ops with not-run cases (CPU contract)",
        "backend_pack": "Ops with not-run cases (backend pack)",
        "known_unsafe": "Ops with not-run cases (known unsafe)",
        "runtime": "Ops with not-run cases (runtime)",
        "other": "Ops with not-run cases (other)",
    }

    def not_run_bucket_for_reason(reason):
        if reason in manifest_skip_reasons:
            return "manifest"
        if reason in selection_skip_reasons:
            return "selection"
        if reason in coverage_skip_reasons:
            return "coverage"
        if reason in contract_skip_reasons:
            return "contract"
        if reason in backend_pack_skip_reasons:
            return "backend_pack"
        if reason in known_unsafe_skip_reasons:
            return "known_unsafe"
        if reason in runtime_skip_reasons:
            return "runtime"
        return "other"

    def suite_for(nodeid, res):
        if res.get("suite"):
            return res["suite"]
        if "test_opinfo_" in nodeid:
            return "opinfo"
        for suite_name in (
            "operators",
            "training",
            "compiler",
            "device_api",
            "autograd",
            "memory",
            "dtypes",
            "strides",
            "workloads",
            "rng",
            "serialization",
            "errors",
            "stress",
            "multi_device",
        ):
            if f"{suite_name}/" in nodeid:
                return suite_name
        return "custom"

    def test_kind_for(nodeid, res):
        return res.get("test_kind") or ("opinfo" if "test_opinfo_" in nodeid else "handwritten")

    def capability_for(nodeid, res):
        explicit = res.get("capability")
        if explicit:
            for cap_name in explicit.split(","):
                if cap_name in KNOWN_CAPABILITIES:
                    return cap_name

        suite_name = suite_for(nodeid, res)
        return {
            "opinfo": "inference",
            "operators": "inference",
            "autograd": "training",
            "training": "training",
            "compiler": "compile",
            "serialization": "serialization",
            "rng": "rng",
            "device_api": "device_api",
            "strides": "channels_last",
            "multi_device": "multi_device",
        }.get(suite_name)

    # ── Operator Coverage Calculations ──
    # OpInfo breadth coverage only
    all_opinfo_ops_tested = set()
    passed_ops = set()
    failed_ops = set()
    skipped_ops_by_bucket = {bucket: set() for bucket in not_run_bucket_labels}

    for nodeid, res in results.items():
        if res.get("status") == "SKIP":
            continue
        if test_kind_for(nodeid, res) != "opinfo":
            continue
        op_name = res.get("op")
        if not op_name:
            continue
        status = res.get("status")
        all_opinfo_ops_tested.add(op_name)
        if status == "PASS":
            passed_ops.add(op_name)
        elif status in ("FAIL", "ERROR"):
            failed_ops.add(op_name)

    for nodeid, res in skips_dict.items():
        if test_kind_for(nodeid, res) != "opinfo":
            continue
        op_name = res.get("op")
        if not op_name:
            continue
        bucket = not_run_bucket_for_reason(_effective_skip_reason(res))
        skipped_ops_by_bucket[bucket].add(op_name)

    skipped_ops_all = set()
    for ops in skipped_ops_by_bucket.values():
        skipped_ops_all.update(ops)

    total_ops_discovered = len(all_opinfo_ops_tested | skipped_ops_all)
    num_pass = len(passed_ops)
    num_fail = len(failed_ops)
    num_skips_by_bucket = {bucket: len(ops) for bucket, ops in skipped_ops_by_bucket.items()}

    def pct(n):
        return f"{n / (total_ops_discovered or 1) * 100:.1f}%"

    # ── Capability Results ──
    # We group tests by suite/capabilities.
    capability_counts = {
        cap: {"pass": 0, "total": 0, "declined": False, "skipped": False}
        for cap in CAPABILITY_ORDER
    }

    # Inspect non-executed records to distinguish manifest declines from runtime skips.
    for nodeid, res in skips_dict.items():
        reason = _effective_skip_reason(res)
        cap = capability_for(nodeid, res)
        if reason in _BACKEND_MANIFEST_DECLINED_SKIP_REASONS:
            if cap in capability_counts:
                capability_counts[cap]["declined"] = True
        elif reason not in _BACKEND_MANIFEST_DECLINED_SKIP_REASONS:
            if cap in capability_counts:
                capability_counts[cap]["skipped"] = True

    # Count passes and totals from results
    # IEEE 754 compliance tracking (NaN/Inf tiers)
    ieee754_pass = 0
    ieee754_fail = 0
    ieee754_skip = 0
    quality_warnings = 0

    for nodeid, res in results.items():
        if res.get("is_plumbing", False):
            continue
        status = res.get("status")
        if status == "SKIP":
            continue

        input_cond = res.get("input_condition")

        # Track quality warnings
        if res.get("quality_warning"):
            quality_warnings += 1

        # NaN/Inf tier tests go to IEEE 754 section, not capability counts
        if input_cond and input_cond != "clean":
            if status == "PASS":
                ieee754_pass += 1
            elif status in ("FAIL", "ERROR"):
                ieee754_fail += 1
            continue

        cap_matched = capability_for(nodeid, res)
        if not res.get("capability"):
            if res.get("suite") == "strides" and "channels_last" in nodeid:
                cap_matched = "channels_last"
            if "sparse" in nodeid:
                cap_matched = "sparse"
            if "test_mixed_precision" in nodeid:
                cap_matched = "autocast"
            if "fp8" in nodeid:
                cap_matched = "fp8"

        if cap_matched:
            capability_counts[cap_matched]["total"] += 1
            if status == "PASS":
                capability_counts[cap_matched]["pass"] += 1

    # ── Dtype Coverage ──
    # Group results by dtype
    dtype_counts = {}
    for nodeid, res in results.items():
        if res.get("status") == "SKIP":
            continue
        dt = res.get("dtype")
        if not dt:
            continue
        # clean representation: e.g. torch.float32 or float32
        dt = dt.replace("torch.", "")
        if dt not in dtype_counts:
            dtype_counts[dt] = {"pass": 0, "total": 0, "fail": 0, "skip": 0, "declined": 0}
        dtype_counts[dt]["total"] += 1
        if res.get("status") == "PASS":
            dtype_counts[dt]["pass"] += 1
        else:
            dtype_counts[dt]["fail"] += 1
    for nodeid, res in skips_dict.items():
        reason = _effective_skip_reason(res)
        if reason not in dtype_skip_reasons:
            continue
        dt = res.get("dtype")
        if not dt:
            continue
        dt = dt.replace("torch.", "")
        if dt not in dtype_counts:
            dtype_counts[dt] = {"pass": 0, "total": 0, "fail": 0, "skip": 0, "declined": 0}
        if reason in _BACKEND_MANIFEST_DECLINED_SKIP_REASONS:
            dtype_counts[dt]["declined"] += 1
        else:
            dtype_counts[dt]["skip"] += 1

    # ── Semantic Level Coverage ──
    semantic_counts = {}
    requested_level = metadata.get("semantic_level")
    requested_selection = metadata.get("semantic_level_selection") or {}
    for source in (results, skips_dict):
        for _nodeid, res in source.items():
            level = res.get("semantic_level")
            if level is None:
                continue
            level = int(level)
            if requested_level is None:
                requested_level = res.get("requested_level")
            semantic_counts.setdefault(level, {"pass": 0, "fail": 0, "skip": 0, "deselected": 0, "total": 0})
            semantic_counts[level]["total"] += 1
            status = res.get("status")
            if status == "PASS":
                semantic_counts[level]["pass"] += 1
            elif status in ("FAIL", "ERROR"):
                semantic_counts[level]["fail"] += 1
            elif (
                _effective_skip_reason(res) in _BACKEND_MANIFEST_DECLINED_SKIP_REASONS
                or _effective_skip_reason(res) in selection_skip_reasons
                or _effective_skip_reason(res) in contract_skip_reasons
                or _effective_skip_reason(res) in coverage_skip_reasons
                or _effective_skip_reason(res) in backend_pack_skip_reasons
                or _effective_skip_reason(res) in known_unsafe_skip_reasons
            ):
                semantic_counts[level]["deselected"] += 1
            else:
                semantic_counts[level]["skip"] += 1

    # ── Failures List ──
    failures_summary = []
    for nodeid, res in results.items():
        if res.get("status") in ("FAIL", "ERROR"):
            op = res.get("op") or nodeid.split("[")[0].split(".")[-1]
            dt = (res.get("dtype") or "unknown").replace("torch.", "")
            maxerr = res.get("maxerr")
            err_msg = res.get("error_message", "")
            
            if maxerr is not None:
                failures_summary.append(f"  {op:<22} {dt:<9} maxerr={maxerr:<7}")
            else:
                # Truncate exception message
                msg_summary = err_msg.split("\n")[0][:40]
                failures_summary.append(f"  {op:<22} {dt:<9} {res.get('status')}: {msg_summary}")
            # Append diagnostic hint if available
            diag = res.get("diagnosis")
            if diag:
                failures_summary.append(f"    ↳ Hint: {diag['likely_cause']}")

    # ── Regressions ──
    regressions_text = []
    baseline_results = {}
    if baseline_data:
        baseline_results = baseline_data.get("results", {})
        baseline_time = baseline_data.get("metadata", {}).get("timestamp", "unknown")
        
        new_failures = []
        fixed = []
        precision_degraded = []
        
        for nodeid, res in results.items():
            status = res.get("status")
            op = res.get("op") or nodeid
            dt = (res.get("dtype") or "unknown").replace("torch.", "")
            maxerr = res.get("maxerr")
            
            base_res = baseline_results.get(nodeid)
            if base_res:
                base_status = base_res.get("status")
                base_maxerr = base_res.get("maxerr")
                
                if base_status == "PASS" and status in ("FAIL", "ERROR"):
                    new_failures.append(f"     {op} [{dt}] PASS → {status}")
                elif base_status in ("FAIL", "ERROR") and status == "PASS":
                    fixed.append(f"     {op} [{dt}] {base_status} → PASS")
                elif base_status == "PASS" and status == "PASS" and maxerr is not None and base_maxerr is not None:
                    # check if degraded >2x
                    if maxerr > 0 and base_maxerr > 0 and maxerr >= base_maxerr * 2:
                        factor = maxerr / base_maxerr
                        precision_degraded.append(f"     {op} [{dt}] maxerr {base_maxerr:.4f} → {maxerr:.4f} ({factor:.1f}×)")
            else:
                # new test
                pass

        if new_failures or fixed or precision_degraded:
            regressions_text.append(f"  REGRESSIONS SINCE LAST RUN ({baseline_time})")
            regressions_text.append(f"  " + "─" * 50)
            if new_failures:
                regressions_text.append(f"  ⚠️  {len(new_failures)} new failures:")
                regressions_text.extend(new_failures)
                regressions_text.append("")
            if fixed:
                regressions_text.append(f"  ✨ {len(fixed)} fixed:")
                regressions_text.extend(fixed)
                regressions_text.append("")
            if precision_degraded:
                regressions_text.append(f"  📉 {len(precision_degraded)} precision degraded:")
                regressions_text.extend(precision_degraded)
                regressions_text.append("")

    # Construct the summary output
    summary_lines = []
    summary_lines.extend(_backend_manifest_assertion_lines(backend_manifest_assertion, metadata))
    summary_lines.append("=" * 60)
    summary_lines.append(f"  Backend: {device:<10} | Hardware: {hw_key}")
    summary_lines.append(f"  PyTorch: {pytorch_version:<10} | Run: {timestamp}")
    summary_lines.append(f"  Duration: {elapsed}")
    summary_lines.append("=" * 60)
    summary_lines.append("")
    if collect_only:
        summary_lines.append("  Collection-only session: no tests executed.")
        summary_lines.append("")
    summary_lines.append("  OPERATOR COVERAGE")
    summary_lines.append("  " + "─" * 17)
    summary_lines.append("  Op categories overlap; an op can pass one case and skip another.")
    summary_lines.append(f"  OpInfo ops represented:    {total_ops_discovered}")
    summary_lines.append(f"  Ops with PASS coverage:    {num_pass:<4} ({pct(num_pass)})")
    summary_lines.append(f"  Ops with FAIL/ERROR:       {num_fail:<4} ({pct(num_fail)})")
    for bucket in ("manifest", "selection", "coverage", "contract", "backend_pack", "known_unsafe", "runtime"):
        count = num_skips_by_bucket[bucket]
        summary_lines.append(f"  {not_run_bucket_labels[bucket]}: {count:<4} ({pct(count)})")
    other_count = num_skips_by_bucket["other"]
    if other_count:
        summary_lines.append(f"  {not_run_bucket_labels['other']}:    {other_count:<4} ({pct(other_count)})")
    summary_lines.append("")
    
    if regressions_text:
        summary_lines.extend(regressions_text)
        summary_lines.append("")

    summary_lines.append("  CAPABILITY RESULTS")
    summary_lines.append("  " + "─" * 18)
    for cap, stats in capability_counts.items():
        if stats["total"] > 0:
            indicator = "✅" if stats["pass"] == stats["total"] else "❌"
            summary_lines.append(f"  {indicator}  {cap:<15} {stats['pass']}/{stats['total']} passed")
        elif stats["declined"]:
            summary_lines.append(f"  ⬚  {cap:<15} DECLINED")
        elif stats["skipped"]:
            summary_lines.append(f"  ⬚  {cap:<15} SKIPPED")
        else:
            summary_lines.append(f"  ⬚  {cap:<15} {stats['pass']}/{stats['total']} passed")
    summary_lines.append("")

    summary_lines.append("  DTYPE COVERAGE")
    summary_lines.append("  " + "─" * 14)
    # Print dtypes in a grid
    dt_keys = sorted(list(dtype_counts.keys()))
    for i in range(0, len(dt_keys), 2):
        chunk = dt_keys[i:i+2]
        line_parts = []
        for dt in chunk:
            stats = dtype_counts[dt]
            if stats["total"] == 0 and (stats["skip"] or stats["declined"]):
                ind = "⬚"
            else:
                ind = "✅" if stats["fail"] == 0 else "❌"
            skip_text = f" skip={stats['skip']}" if stats["skip"] else ""
            declined_text = f" declined={stats['declined']}" if stats["declined"] else ""
            line_parts.append(f"  {dt:<10} {stats['pass']}/{stats['total']} {ind}{skip_text}{declined_text}")
        summary_lines.append("  ".join(line_parts))
    summary_lines.append("")

    if semantic_counts:
        requested_text = requested_selection.get("label")
        if not requested_text:
            requested_text = f"requested <= {requested_level}" if requested_level is not None else "requested unknown"
        summary_lines.append("  SEMANTIC LEVELS")
        summary_lines.append("  " + "─" * 15)
        summary_lines.append(f"  {requested_text}")
        for level in sorted(semantic_counts):
            stats = semantic_counts[level]
            parts = [
                f"pass={stats['pass']:<4}",
                f"fail={stats['fail']:<4}",
            ]
            if stats["skip"] or not stats["deselected"]:
                parts.append(f"skip={stats['skip']:<4}")
            if stats["deselected"]:
                parts.append(f"deselected={stats['deselected']:<4}")
            parts.append(f"total={stats['total']}")
            summary_lines.append(f"  L{level:<2} " + " ".join(parts))
        summary_lines.append("")

    # IEEE 754 Compliance section (NaN/Inf tiers)
    ieee754_total = ieee754_pass + ieee754_fail
    if ieee754_total > 0:
        ieee754_indicator = "✅" if ieee754_fail == 0 else "❌"
        summary_lines.append("  IEEE 754 COMPLIANCE")
        summary_lines.append("  " + "─" * 19)
        summary_lines.append(f"  {ieee754_indicator}  NaN/Inf propagation  {ieee754_pass}/{ieee754_total} passed")
        summary_lines.append("")

    # Quality warnings
    if quality_warnings > 0:
        summary_lines.append(f"  QUALITY WARNINGS: {quality_warnings} tests passed at usable tolerance but failed golden tier")
        summary_lines.append("")

    if num_fail > 0:
        summary_lines.append(f"  FAILURES ({num_fail})")
        summary_lines.append("  " + "─" * 12)
        summary_lines.extend(failures_summary[:20]) # Limit to 20 in summary
        if len(failures_summary) > 20:
            summary_lines.append(f"  ... and {len(failures_summary) - 20} more failures")
        summary_lines.append("")

    scorecard_str = "\n".join(summary_lines)

    # ── Markdown Detail Section ──
    md_lines = []
    md_lines.append(f"# Validator Scorecard for {device}")
    md_lines.append("")
    md_lines.append("```")
    md_lines.append(scorecard_str)
    md_lines.append("```")
    md_lines.append("")

    if num_fail > 0:
        md_lines.append("## Per-Test Failure Details")
        md_lines.append("")
        for nodeid, res in results.items():
            if res.get("status") in ("FAIL", "ERROR"):
                op = res.get("op") or nodeid
                dt = (res.get("dtype") or "unknown").replace("torch.", "")
                cat = res.get("category", "unknown")
                maxerr = res.get("maxerr")
                cosim = res.get("cosim")
                shapes = res.get("shapes", "unknown")
                duration = f"{res.get('duration_ms', 0):.0f}ms"
                err_msg = res.get("error_message", "")
                
                md_lines.append("---")
                md_lines.append(f"### {res.get('status')}: {op} [{dt}]")
                md_lines.append("")
                md_lines.append(f"- **Category**:    {cat}")
                md_lines.append(f"- **Input Shapes**: {shapes}")
                md_lines.append(f"- **Duration**:     {duration}")
                if maxerr is not None:
                    md_lines.append(f"- **Max Error**:    {maxerr:.6f}")
                if cosim is not None:
                    md_lines.append(f"- **Cosim**:        {cosim:.6f}")
                md_lines.append("")
                md_lines.append("**Traceback / Error Details**:")
                md_lines.append("```")
                md_lines.append(err_msg)
                md_lines.append("```")
                md_lines.append("")
                
                # Diagnostic hint
                diag = res.get("diagnosis")
                if diag:
                    md_lines.append("> [!CAUTION]")
                    md_lines.append(f"> **Likely Cause**: {diag['likely_cause']}")
                    md_lines.append(f"> **Remediation**: {diag['remediation']}")
                    md_lines.append("")
                
                # Check baseline for regression notes
                if baseline_data:
                    base_res = baseline_results.get(nodeid)
                    if base_res and base_res.get("status") == "PASS":
                        md_lines.append(f"> [!WARNING]")
                        md_lines.append(f"> **REGRESSION**: This test PASSED in the previous run.")
                        if base_res.get("maxerr") is not None:
                            md_lines.append(f"> Previous maxerr: {base_res.get('maxerr'):.6f}")
                        md_lines.append("")

    def not_run_record_label(reason):
        if reason in _BACKEND_MANIFEST_DECLINED_SKIP_REASONS:
            return "declines"
        if reason in selection_skip_reasons:
            return "deselections"
        if reason in contract_skip_reasons:
            return "contract-blocked"
        if reason in coverage_skip_reasons:
            return "coverage debt"
        if reason in backend_pack_skip_reasons:
            return "backend-pack debt"
        if reason in known_unsafe_skip_reasons:
            return "known unsafe"
        return "skips"

    # ── Not-Run Audit Section ──
    if include_skips and skips_dict:
        md_lines.append("## Not-Run Audit")
        md_lines.append("")
        
        # Group non-executed records by reason
        reason_groups = {}
        for nodeid, res in skips_dict.items():
            reason = _effective_skip_reason(res)
            if reason not in reason_groups:
                reason_groups[reason] = []
            reason_groups[reason].append(res)
            
        md_lines.append("### Not Run By Reason:")
        for reason, items in reason_groups.items():
            md_lines.append(f"- **{reason}**: {len(items)} {not_run_record_label(reason)}")
        md_lines.append("")

        dtype_not_run_groups = {}
        for _nodeid, res in skips_dict.items():
            reason = _effective_skip_reason(res)
            if reason not in dtype_skip_reasons:
                continue
            dt = (res.get("dtype") or "unknown").replace("torch.", "")
            label = not_run_record_label(reason)
            dtype_not_run_groups.setdefault(
                dt,
                {
                    "declines": 0,
                    "deselections": 0,
                    "contract-blocked": 0,
                    "coverage debt": 0,
                    "backend-pack debt": 0,
                    "known unsafe": 0,
                    "skips": 0,
                },
            )
            dtype_not_run_groups[dt][label] += 1
        if dtype_not_run_groups:
            md_lines.append("### Dtype Not Run:")
            for dt, counts in sorted(dtype_not_run_groups.items()):
                parts = [
                    f"{count} {label}"
                    for label, count in (
                        ("declines", counts["declines"]),
                        ("deselections", counts["deselections"]),
                        ("contract-blocked", counts["contract-blocked"]),
                        ("coverage debt", counts["coverage debt"]),
                        ("backend-pack debt", counts["backend-pack debt"]),
                        ("known unsafe", counts["known unsafe"]),
                        ("skips", counts["skips"]),
                    )
                    if count
                ]
                md_lines.append(f"- **{dt}**: {', '.join(parts)}")
            md_lines.append("")

        md_lines.append("### Full Not-Run List:")
        md_lines.append("| Test Name | Reason | Detail |")
        md_lines.append("|---|---|---|")
        for nodeid, res in skips_dict.items():
            op = res.get("op") or nodeid.split("[")[0].split(".")[-1]
            dt = (res.get("dtype") or "").replace("torch.", "")
            reason = _effective_skip_reason(res)
            detail = _skip_detail(res).replace("\n", " ")
            md_lines.append(f"| `{nodeid.split('/')[-1]}` | `{reason}` | {detail} |")
        md_lines.append("")

    while md_lines and not md_lines[-1]:
        md_lines.pop()
    markdown_report = "\n".join(md_lines)
    markdown_report = "\n".join(line.rstrip() for line in markdown_report.splitlines())
    return scorecard_str, markdown_report

def generate_report_cli(from_file=None):
    if from_file:
        file_to_load = os.path.abspath(from_file)
        results_dir = os.path.dirname(file_to_load) or os.getcwd()
    else:
        results_dir = os.path.join(os.getcwd(), "results")
        # Find latest file
        if not os.path.exists(results_dir):
            print(f"Results directory '{results_dir}' does not exist.", file=sys.stderr)
            return 1
        files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith("_latest.json")]
        if not files:
            print("No latest results JSON found in ./results/", file=sys.stderr)
            return 1
        # take the most recently modified latest.json file
        file_to_load = max(files, key=os.path.getmtime)

    try:
        with open(file_to_load, "r", encoding="utf-8") as f:
            current_data = json.load(f)
    except Exception as e:
        print(f"Error loading results file {file_to_load}: {e}", file=sys.stderr)
        return 1

    # Load baseline if history exists
    # Previous run is the latest run in history, or the latest file from history dir
    baseline_data = None
    hw_key = current_data.get("metadata", {}).get("hardware_key", "unknown")
    current_timestamp = current_data.get("metadata", {}).get("timestamp")
    history_dir = os.path.join(results_dir, f"{hw_key}_history")
    if os.path.exists(history_dir):
        history_files = [os.path.join(history_dir, f) for f in os.listdir(history_dir) if f.endswith(".json")]
        if history_files:
            # Sort by file mtime to find the previous one
            history_files.sort(key=os.path.getmtime, reverse=True)
            for hf in history_files:
                if os.path.abspath(hf) != file_to_load:
                    try:
                        with open(hf, "r", encoding="utf-8") as f:
                            candidate_data = json.load(f)
                        candidate_timestamp = candidate_data.get("metadata", {}).get("timestamp")
                        if current_timestamp and candidate_timestamp == current_timestamp:
                            continue
                        if candidate_data.get("metadata", {}).get("session_completed") is False:
                            continue
                        baseline_data = candidate_data
                        break
                    except:
                        pass

    scorecard, markdown = build_report(current_data, baseline_data, include_skips=True)
    
    # Print scorecard to stdout
    try:
        print(scorecard)
    except UnicodeEncodeError:
        try:
            sys.stdout.buffer.write(scorecard.encode(sys.stdout.encoding or "utf-8", errors="replace"))
            sys.stdout.flush()
        except Exception:
            pass
    
    # Save markdown report
    report_path = os.path.join(results_dir, f"{hw_key}_report.md")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"Report saved to: {report_path}")
    except Exception as e:
        print(f"Failed to write markdown report: {e}", file=sys.stderr)
        return 1

    return 0
