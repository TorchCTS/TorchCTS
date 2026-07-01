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
    manifest_skip_reasons = {
        "op_excluded",
        "dtype_not_supported",
        "dtype_regex_filtered",
        "dtype_not_listed",
        "capability_not_declared",
    }
    selection_skip_reasons = {
        "semantic_level_gt_requested",
        "semantic_level_out_of_range",
    }
    coverage_skip_reasons = {
        "coverage_unknown",
        "coverage_excluded",
        "coverage_strategy_pending",
        "pending_backend_pack",
        "backend_not_available",
        "coverage_capability_disabled",
    }
    contract_skip_reasons = {
        "cpu_contract_unsupported",
        "cpu_contract_unknown",
        "cpu_contract_pending",
    }
    runtime_skip_reasons = {
        "runtime_skip",
        "unavailable_in_pytorch_runtime",
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
        "manifest": "Ops not run (manifest)",
        "selection": "Ops not run (selection)",
        "coverage": "Ops not run (coverage)",
        "contract": "Ops not run (CPU contract)",
        "runtime": "Ops not run (runtime)",
        "other": "Ops not run (other)",
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
        bucket = not_run_bucket_for_reason(res.get("skip_reason"))
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
        cap: {"pass": 0, "total": 0, "skipped": False}
        for cap in CAPABILITY_ORDER
    }

    # Inspect skips to see if capability skipped
    for nodeid, res in skips_dict.items():
        # if a capability was not declared, mark it skipped
        reason = res.get("skip_reason")
        cap = capability_for(nodeid, res)
        if reason == "capability_not_declared":
            if cap in capability_counts:
                capability_counts[cap]["skipped"] = True
        elif reason == "device_count" and cap == "multi_device":
            capability_counts["multi_device"]["skipped"] = True

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
            dtype_counts[dt] = {"pass": 0, "total": 0, "fail": 0, "skip": 0}
        dtype_counts[dt]["total"] += 1
        if res.get("status") == "PASS":
            dtype_counts[dt]["pass"] += 1
        else:
            dtype_counts[dt]["fail"] += 1
    for nodeid, res in skips_dict.items():
        if res.get("skip_reason") not in dtype_skip_reasons:
            continue
        dt = res.get("dtype")
        if not dt:
            continue
        dt = dt.replace("torch.", "")
        if dt not in dtype_counts:
            dtype_counts[dt] = {"pass": 0, "total": 0, "fail": 0, "skip": 0}
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
            semantic_counts.setdefault(level, {"pass": 0, "fail": 0, "skip": 0, "total": 0})
            semantic_counts[level]["total"] += 1
            status = res.get("status")
            if status == "PASS":
                semantic_counts[level]["pass"] += 1
            elif status in ("FAIL", "ERROR"):
                semantic_counts[level]["fail"] += 1
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
    summary_lines.append(f"  OpInfo ops discovered:     {total_ops_discovered}")
    summary_lines.append(f"  Ops tested (PASS):         {num_pass:<4} ({pct(num_pass)})")
    summary_lines.append(f"  Ops tested (FAIL):         {num_fail:<4} ({pct(num_fail)})")
    for bucket in ("manifest", "selection", "coverage", "contract", "runtime"):
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
        if stats["skipped"]:
            summary_lines.append(f"  ⬚  {cap:<15} SKIPPED")
        else:
            indicator = "✅" if stats["pass"] == stats["total"] and stats["total"] > 0 else "❌"
            if stats["total"] == 0:
                indicator = "⬚"
            summary_lines.append(f"  {indicator}  {cap:<15} {stats['pass']}/{stats['total']} passed")
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
            if stats["total"] == 0 and stats["skip"]:
                ind = "⬚"
            else:
                ind = "✅" if stats["fail"] == 0 else "❌"
            skip_text = f" skip={stats['skip']}" if stats["skip"] else ""
            line_parts.append(f"  {dt:<10} {stats['pass']}/{stats['total']} {ind}{skip_text}")
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
            summary_lines.append(
                f"  L{level:<2} pass={stats['pass']:<4} fail={stats['fail']:<4} skip={stats['skip']:<4} total={stats['total']}"
            )
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

    # ── Skip Audit Section ──
    if include_skips and skips_dict:
        md_lines.append("## Skip Audit")
        md_lines.append("")
        
        # Group skips by reason
        reason_groups = {}
        for nodeid, res in skips_dict.items():
            reason = res.get("skip_reason", "unknown")
            if reason not in reason_groups:
                reason_groups[reason] = []
            reason_groups[reason].append(res)
            
        md_lines.append("### Skips By Reason:")
        for reason, items in reason_groups.items():
            md_lines.append(f"- **{reason}**: {len(items)} skips")
        md_lines.append("")

        dtype_skip_groups = {}
        for _nodeid, res in skips_dict.items():
            if res.get("skip_reason") not in dtype_skip_reasons:
                continue
            dt = (res.get("dtype") or "unknown").replace("torch.", "")
            dtype_skip_groups[dt] = dtype_skip_groups.get(dt, 0) + 1
        if dtype_skip_groups:
            md_lines.append("### Dtype Skips:")
            for dt, count in sorted(dtype_skip_groups.items()):
                md_lines.append(f"- **{dt}**: {count} skips")
            md_lines.append("")

        md_lines.append("### Full Skip List:")
        md_lines.append("| Test Name | Reason | Detail |")
        md_lines.append("|---|---|---|")
        for nodeid, res in skips_dict.items():
            op = res.get("op") or nodeid.split("[")[0].split(".")[-1]
            dt = (res.get("dtype") or "").replace("torch.", "")
            reason = res.get("skip_reason")
            detail = res.get("detail", "").replace("\n", " ")
            md_lines.append(f"| `{nodeid.split('/')[-1]}` | `{reason}` | {detail} |")
        md_lines.append("")

    markdown_report = "\n".join(md_lines)
    return scorecard_str, markdown_report

def generate_report_cli(from_file=None):
    results_dir = os.path.join(os.getcwd(), "results")
    if from_file:
        file_to_load = from_file
    else:
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
    history_dir = os.path.join(results_dir, f"{hw_key}_history")
    if os.path.exists(history_dir):
        history_files = [os.path.join(history_dir, f) for f in os.listdir(history_dir) if f.endswith(".json")]
        if history_files:
            # Sort by file mtime to find the previous one
            history_files.sort(key=os.path.getmtime, reverse=True)
            for hf in history_files:
                if hf != file_to_load:
                    try:
                        with open(hf, "r", encoding="utf-8") as f:
                            baseline_data = json.load(f)
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
