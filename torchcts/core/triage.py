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

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Iterable


TRIAGE_CLASSIFICATIONS = (
    "confirmed_mps_crash",
    "confirmed_mps_wrong_value",
    "confirmed_mps_missing_kernel",
    "confirmed_mps_unsupported_dtype_or_layout",
    "manifest_overclaim",
    "local_toolchain_or_environment",
    "torchcts_invalid_sample",
    "torchcts_bad_oracle",
    "torchcts_bad_assertion",
    "torchcts_bad_generated_strategy",
    "cpu_oracle_failure",
    "expected_unsupported",
    "needs_more_evidence",
)

DEFAULT_TRIAGE_DIR = Path("results") / "mps_triage"
DEFAULT_MPS_CRASH_NODES = (
    "torchcts/autograd/test_backward.py::test_internal_dispatcher_surface[grid2_cpu_fallback_default]",
    "torchcts/autograd/test_backward.py::test_internal_dispatcher_surface[grid2_cpu_fallback_out]",
    "torchcts/autograd/test_backward.py::test_internal_dispatcher_surface[grid2_cpu_fallback_backward]",
)
FAILURE_STATUSES = {"FAIL", "ERROR"}


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def tail_text(text: str | None, limit: int = 12000) -> str:
    if not text:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    elif not isinstance(text, str):
        text = str(text)
    if len(text) <= limit:
        return text
    return text[-limit:]


def stable_id(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    if len(slug) > 80:
        slug = slug[:80].rstrip("_")
    return f"{slug}_{digest}" if slug else digest


def signal_name(returncode: int | None) -> str | None:
    if returncode is None:
        return None
    signum = None
    if returncode < 0:
        signum = -returncode
    elif returncode >= 128:
        signum = returncode - 128
    if signum is None:
        return None
    try:
        return signal.Signals(signum).name
    except Exception:
        return f"SIG{signum}"


def is_process_crash(returncode: int | None, stderr: str = "", stdout: str = "") -> bool:
    if returncode is None:
        return False
    if signal_name(returncode) in {"SIGSEGV", "SIGABRT", "SIGBUS", "SIGILL"}:
        return True
    haystack = f"{stderr}\n{stdout}"
    return "Fatal Python error" in haystack or "Segmentation fault" in haystack


def command_string(command: Iterable[str]) -> str:
    return " ".join(str(part) for part in command)


def load_result_file(path: str | Path | None) -> tuple[Path | None, dict]:
    if path is None:
        resolved = find_default_mps_result()
        if resolved is None:
            return None, {}
        path = resolved
    result_path = Path(path)
    if not result_path.exists():
        return result_path, {}
    return result_path, json.loads(result_path.read_text(encoding="utf-8"))


def find_default_mps_result(results_dir: str | Path = "results") -> Path | None:
    candidates = []
    for path in Path(results_dir).glob("*_latest.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("metadata", {}).get("device_name") == "mps":
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _hardware_key_from_result(path: Path | None, data: dict) -> str | None:
    key = data.get("metadata", {}).get("hardware_key")
    if key:
        return str(key)
    if path and path.name.endswith("_latest.json"):
        return path.name[: -len("_latest.json")]
    return None


def freeze_input_snapshot(result_path: Path | None, data: dict, triage_dir: Path) -> list[str]:
    snapshot_dir = triage_dir / "input_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    if result_path and result_path.exists():
        target = snapshot_dir / result_path.name
        if result_path.resolve() != target.resolve():
            shutil.copy2(result_path, target)
        copied.append(str(target))

    key = _hardware_key_from_result(result_path, data)
    if key:
        base_dir = result_path.parent if result_path else Path("results")
        for suffix in ("_report.md", "_runlog.txt"):
            source = base_dir / f"{key}{suffix}"
            if source.exists():
                target = snapshot_dir / source.name
                if source.resolve() != target.resolve():
                    shutil.copy2(source, target)
                copied.append(str(target))
    return copied


def parse_runlog_crash_candidates(runlog_path: str | Path | None, limit: int = 3) -> list[str]:
    if not runlog_path:
        return []
    path = Path(runlog_path)
    if not path.exists():
        return []
    candidates: list[str] = []
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        match = re.match(r"\s*[\d.]+s\s+(.+)$", line)
        if not match:
            continue
        nodeid = match.group(1).strip()
        if nodeid and nodeid not in candidates:
            candidates.append(nodeid)
        if len(candidates) >= limit:
            break
    return candidates


def build_triage_queue(
    result_data: dict | None,
    *,
    include_crashers: bool = False,
    nodes_file: str | Path | None = None,
    runlog_path: str | Path | None = None,
) -> list[str]:
    ordered: list[str] = []

    def add(nodeid: str | None) -> None:
        if nodeid and nodeid not in ordered:
            ordered.append(nodeid)

    if include_crashers:
        for nodeid in DEFAULT_MPS_CRASH_NODES:
            add(nodeid)
        for nodeid in parse_runlog_crash_candidates(runlog_path):
            add(nodeid)

    if nodes_file:
        path = Path(nodes_file)
        if path.exists():
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if line and not line.startswith("#"):
                    add(line)

    for nodeid, record in sorted((result_data or {}).get("results", {}).items()):
        if record.get("status") in FAILURE_STATUSES:
            add(nodeid)
    return ordered


def _classification(reason: str, classification: str) -> dict:
    if classification not in TRIAGE_CLASSIFICATIONS:
        classification = "needs_more_evidence"
    return {"classification": classification, "classification_reason": reason}


def _is_grid_cpu_fallback_backward_wrong_value(nodeid: str, text: str) -> bool:
    combined = f"{nodeid}\n{text}".lower()
    return (
        "grid2_cpu_fallback_backward" in combined
        and "tensor-likes are not close" in combined
        and "mismatched elements" in combined
    )


def classify_record(record: dict | None, subprocess_result: dict | None = None) -> dict:
    record = record or {}
    message = str(record.get("error_message") or "")
    error_type = str(record.get("error_type") or "")
    nodeid = str(record.get("nodeid") or "")
    if subprocess_result:
        if subprocess_result.get("timed_out"):
            return _classification("subprocess timed out before evidence was complete", "needs_more_evidence")
        returncode = subprocess_result.get("returncode")
        subprocess_text = "\n".join(
            part
            for part in [
                message,
                subprocess_result.get("stderr_tail") or "",
                subprocess_result.get("stdout_tail") or "",
            ]
            if part
        )
        process_crash = is_process_crash(
            returncode,
            subprocess_result.get("stderr_tail", ""),
            subprocess_result.get("stdout_tail", ""),
        )
        if _is_grid_cpu_fallback_backward_wrong_value(nodeid, subprocess_text):
            return _classification(
                "MPS grid-sampler CPU-fallback backward returned values that disagree with the CPU oracle; subprocess isolation is still required because the child exits with SIGSEGV after reporting the mismatch",
                "confirmed_mps_wrong_value",
            )
        if process_crash:
            return _classification(
                f"fresh subprocess crashed with returncode={returncode} signal={subprocess_result.get('signal')}",
                "confirmed_mps_crash",
            )
        message = subprocess_text

    text = message.lower()
    combined = f"{nodeid}\n{text}"

    if _is_grid_cpu_fallback_backward_wrong_value(nodeid, text):
        return _classification(
            "MPS grid-sampler CPU-fallback backward returned values that disagree with the CPU oracle; subprocess isolation is still required because the child exits with SIGSEGV after reporting the mismatch",
            "confirmed_mps_wrong_value",
        )
    if error_type == "ProcessCrash" or "fatal python error" in text or "segmentation fault" in text:
        return _classification("process crash evidence was recorded", "confirmed_mps_crash")
    if "libc++.1.dylib" in text or ("torchinductor" in text and "dlopen" in text):
        return _classification("torch.compile failed while loading a generated local dynamic library", "local_toolchain_or_environment")
    if "module 'torch' has no attribute 'stream'" in text:
        return _classification("test uses a stream API that is not exposed through this PyTorch MPS build", "torchcts_bad_assertion")
    if "stream api unavailable" in text or "stream context api unavailable" in text:
        return _classification("manifest enables streams but this PyTorch build exposes no usable stream context API", "manifest_overclaim")
    if "doesn't support synchronizing streams" in text or "does not support synchronizing streams" in text:
        return _classification("manifest enables streams but this PyTorch build cannot synchronize streams", "manifest_overclaim")
    if "event api unavailable" in text or "event timing api unavailable" in text:
        return _classification("manifest enables events but this PyTorch build exposes no usable event API", "manifest_overclaim")
    if "empty_permuted" in combined and ("not close" in text or "mismatched" in text):
        return _classification("empty tensor output contains uninitialized values and is not a valid value oracle", "torchcts_bad_generated_strategy")
    if "sparse csr tensors do not have is_contiguous" in text:
        return _classification("TorchCTS attempted dense-style IEEE sample mutation on a sparse CSR tensor", "torchcts_bad_generated_strategy")
    if "trying to convert float8" in text or "undefined type float8" in text:
        return _classification("manifest enables FP8 but this MPS build rejects FP8 tensors", "manifest_overclaim")
    if "nyi: named tensors only support cpu, cuda, xpu or privateuseone tensors" in text:
        return _classification("manifest enables named tensors but this PyTorch MPS build rejects named tensor metadata", "manifest_overclaim")
    if (
        "nestedtensorimpl storage must be either" in text
        and "privateuseone" in text
        and "mps" in text
    ):
        return _classification("manifest enables nested tensors but this PyTorch MPS build rejects nested tensor storage", "manifest_overclaim")
    if "device type of values (mps) must be one of cpu, cuda, xpu, meta or privateuse1" in text:
        return _classification("MPS rejects compressed sparse tensor values for this dispatcher surface", "confirmed_mps_unsupported_dtype_or_layout")
    if "values and compressed tensor instance need to be on the same device" in text:
        return _classification("MPS compressed sparse constructor path rejects the value/layout device combination", "confirmed_mps_unsupported_dtype_or_layout")
    if "expected n % 32 == 0 && k % 32 == 0" in text:
        return _classification("MPS int8 packed matmul kernel rejects shapes outside its 32-aligned implementation limit", "confirmed_mps_unsupported_dtype_or_layout")
    if "promotion for uint16, uint32, uint64 types is not supported" in text:
        return _classification("MPS rejects unsigned integer promotion for a CPU-valid dispatcher sample", "confirmed_mps_unsupported_dtype_or_layout")
    if "float64 dtype" in text or "mps doesn't support complex" in text or "does not have support for that dtype" in text:
        return _classification("manifest enables a dtype/layout rejected by this MPS build", "manifest_overclaim")
    if "arange_cpu" in text and "not implemented for 'uint16'" in text:
        return _classification("TorchCTS CPU oracle/sample construction used a uint16 arange path unavailable in PyTorch CPU", "torchcts_bad_oracle")
    if "equal_cpu" in text and "not implemented for 'complexhalf'" in text:
        return _classification("TorchCTS reproducibility assertion used CPU torch.equal on complex-half tensors", "torchcts_bad_assertion")
    if "expected a 'cpu' device type for generator" in text:
        return _classification("TorchCTS generated a target-device generator for a dispatcher surface that requires a CPU generator", "torchcts_invalid_sample")
    if "high must be a scalar tensor and on cpu" in text:
        return _classification("TorchCTS generated a device tensor bound for a dispatcher surface that requires a CPU scalar tensor bound", "torchcts_invalid_sample")
    if "normal expects mean to be non-complex" in text:
        return _classification("MPS rejects a complex normal sample that the CPU oracle accepts", "confirmed_mps_unsupported_dtype_or_layout")
    if "only supports cpu, cuda and xpu device type" in text and "mps" in text:
        return _classification("valid dispatcher path reports no MPS implementation for this surface", "confirmed_mps_missing_kernel")
    if "expected tensor to have cpu backend" in text and "mps backend" in text:
        return _classification("valid dispatcher path routes to a CPU-only implementation when invoked with MPS tensors", "confirmed_mps_missing_kernel")
    if "nan_to_num_mps" in text and "not implemented" in text:
        return _classification("valid dispatcher path reports missing MPS nan_to_num kernel support for this dtype", "confirmed_mps_missing_kernel")
    if "dispatchstub: missing kernel for mps" in text or ("could not run" in text and "mps" in text):
        return _classification("valid dispatcher path reports missing MPS kernel support", "confirmed_mps_missing_kernel")
    if "expected exception" in text and "not raised" in text:
        return _classification("MPS accepted an invalid OpInfo error sample that PyTorch metadata says should be rejected", "confirmed_mps_wrong_value")
    if "cpu raised runtimeerror but device succeeded" in text:
        return _classification("MPS accepted an input that the CPU oracle rejected for the same OpInfo sample", "confirmed_mps_wrong_value")
    if "_fused_rms_norm" in combined and "actual nonetype vs expected tensor" in text:
        return _classification("MPS _fused_rms_norm omitted the schema-declared inverse-rms tensor return", "confirmed_mps_wrong_value")
    if "test_low_level_misc_dispatcher_helpers" in combined and "tensor-likes are not close" in text:
        return _classification("MPS _saturate_weight_to_fp16 leaves out-of-range values unclamped compared with the CPU oracle", "confirmed_mps_wrong_value")
    if (
        "propagation mismatch" in text
        and any(
            surface in combined
            for surface in (
                "_safe_softmax",
                "_softmax.out",
                "has_nan-softmax",
                "softmax.int",
                "_logcumsumexp",
                "logcumsumexp",
                "prod.dim_int",
            )
        )
    ):
        return _classification("MPS NaN/Inf propagation disagrees with the CPU oracle for this reproduced generated reduction sample", "confirmed_mps_wrong_value")
    if (
        any(op in combined for op in ("std", "std_mean", "var", "var_mean"))
        and ("shape mismatch after comparison normalization" in text or "dtype mismatch" in text)
    ):
        return _classification("MPS complex variance/std reduction returned a tensor shape or dtype that disagrees with the CPU oracle", "confirmed_mps_wrong_value")
    if "stride_and_storage_offset_metadata" in combined and ("stride" in text or "storage_offset" in text):
        return _classification("MPS non-contiguous view metadata disagrees with the CPU stride/storage-offset oracle", "confirmed_mps_wrong_value")
    if "cholesky" in combined and "not positive-definite" in text:
        return _classification("MPS complex Cholesky rejected an OpInfo sample that the CPU oracle factorized successfully", "confirmed_mps_wrong_value")
    if "conj_physical" in combined and "expected self.is_complex() to be true" in text:
        return _classification("MPS conj_physical rejects a real-valued tensor sample that the CPU dispatcher accepts as an identity operation", "confirmed_mps_wrong_value")
    if "propagation mismatch" in text or "inf sign mismatch" in text:
        return _classification("MPS NaN/Inf propagation disagrees with the CPU oracle for an IEEE-enabled comparison sample", "confirmed_mps_wrong_value")
    if "alias mismatch" in text:
        return _classification("MPS aliasing semantics disagree with the CPU oracle for this view/alias sample", "confirmed_mps_wrong_value")
    if "shape mismatch" in text:
        return _classification("MPS output shape disagrees with the CPU oracle for this generated sample", "confirmed_mps_wrong_value")
    if "integer/index values differ" in text:
        return _classification("MPS integer/index output values disagree with the CPU oracle for this generated sample", "confirmed_mps_wrong_value")
    if "returned a different object than the provided out tensor" in text:
        return _classification("MPS out= dispatcher did not return the provided output tensor object", "confirmed_mps_wrong_value")
    if "expected out tensor to have dtype" in text and "got" in text:
        return _classification("MPS out= dispatcher used an output dtype that disagrees with the CPU oracle", "confirmed_mps_wrong_value")
    if "produced values below" in text or "produced values >=" in text:
        return _classification("MPS RNG output violates the CPU-validated output domain", "confirmed_mps_wrong_value")
    if "scalars are not close" in text:
        return _classification("MPS scalar result disagrees with the CPU oracle for this generated sample", "confirmed_mps_wrong_value")
    if "shape '" in text and "is invalid for input of size" in text:
        return _classification("MPS rejected a shape tensor sample accepted by the CPU oracle", "confirmed_mps_wrong_value")
    if "placeholder tensor is empty" in text or "internal assert failed" in text:
        return _classification("MPS raised an internal assertion for a CPU-validated dispatcher sample", "confirmed_mps_wrong_value")
    if "tensor-likes are not close" in text or "tensor-likes are not equal" in text:
        return _classification("MPS tensor values disagree with the CPU oracle for this comparison sample", "confirmed_mps_wrong_value")
    if "coverage_strategy_pending" in text:
        return _classification("TorchCTS intentionally has no executable strategy for this surface yet", "expected_unsupported")
    if error_type == "AssertionError":
        return _classification("value/property mismatch needs CPU oracle and standalone repro validation", "needs_more_evidence")
    if record.get("status") in FAILURE_STATUSES:
        return _classification("failure has not yet been adjudicated with a minimal repro", "needs_more_evidence")
    return _classification("record is not a failure", "expected_unsupported")


def classify_records(result_data: dict) -> dict[str, dict]:
    classifications = {}
    for nodeid, record in sorted((result_data or {}).get("results", {}).items()):
        if record.get("status") not in FAILURE_STATUSES:
            continue
        enriched = dict(record)
        enriched["nodeid"] = nodeid
        classifications[nodeid] = classify_record(enriched)
    return classifications


def _adjudication_family(nodeid: str, record: dict) -> str:
    dispatcher = str(record.get("dispatcher_name") or record.get("op") or "")
    haystack = f"{nodeid}\n{dispatcher}".lower()
    if "batch_norm" in haystack and "test_autograd_backward" in haystack:
        return "generated_autograd_backward_batchnorm"
    if "_foreach_" in haystack or "test_foreach_fused" in haystack:
        return "foreach_fused"
    if "test_functional_variants" in haystack:
        return "generated_functional_variants"
    if "test_factories" in haystack:
        return "factories"
    return "other"


def build_adjudication_queue(
    result_data: dict | None,
    classifications: dict[str, dict],
    *,
    result_path: str | Path | None = None,
) -> dict:
    groups: dict[str, list[dict]] = {}
    results = (result_data or {}).get("results", {})
    for nodeid, classification in sorted(classifications.items()):
        if classification.get("classification") != "needs_more_evidence":
            continue
        record = dict(results.get(nodeid, {}))
        family = _adjudication_family(nodeid, record)
        groups.setdefault(family, []).append(
            {
                "nodeid": nodeid,
                "current_classification": classification.get("classification"),
                "classification_reason": classification.get("classification_reason"),
                "error_type": record.get("error_type"),
                "failure_stage": record.get("failure_stage"),
                "dispatcher_name": record.get("dispatcher_name"),
                "schema": record.get("schema"),
                "strategy": record.get("strategy"),
                "strategy_family": record.get("strategy_family"),
                "dtype": record.get("dtype"),
                "semantic_level": record.get("semantic_level"),
                "result_artifact_path": str(result_path) if result_path else None,
                "standalone_repro_exists": False,
            }
        )
    return {
        "generated_at": _utc_now(),
        "groups": {
            family: {
                "count": len(items),
                "items": items,
            }
            for family, items in sorted(groups.items())
        },
    }


def run_subprocess_command(
    command: list[str],
    *,
    cwd: str | Path = ".",
    timeout: float = 120.0,
    env: dict[str, str] | None = None,
) -> dict:
    start = time.time()
    timed_out = False
    stdout = ""
    stderr = ""
    returncode = None
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        returncode = completed.returncode
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
    duration_ms = (time.time() - start) * 1000
    return {
        "command": command,
        "command_string": command_string(command),
        "returncode": returncode,
        "signal": signal_name(returncode),
        "timed_out": timed_out,
        "duration_ms": duration_ms,
        "stdout_tail": tail_text(stdout),
        "stderr_tail": tail_text(stderr),
    }


def run_pytest_node(
    nodeid: str,
    *,
    device: str = "mps",
    level: int = 8,
    timeout: float = 120.0,
    triage_dir: str | Path = DEFAULT_TRIAGE_DIR,
    cwd: str | Path = ".",
) -> dict:
    triage_path = Path(triage_dir)
    logs_dir = triage_path / "logs"
    pytest_results_dir = triage_path / "pytest_results"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pytest_results_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pytest",
        nodeid,
        "--device",
        device,
        "--level",
        str(level),
        "--tb=short",
        "--results-dir",
        str(pytest_results_dir),
        "--known-segfault-policy",
        "off",
    ]
    env = os.environ.copy()
    env["TORCHCTS_NON_INTERACTIVE"] = "1"
    env["_TORCHCTS_TRIAGE_CHILD"] = "1"
    result = run_subprocess_command(command, cwd=cwd, timeout=timeout, env=env)
    result["nodeid"] = nodeid
    result["status"] = "ERROR" if result["returncode"] else "PASS"
    result["classification"] = classify_record({"nodeid": nodeid, "status": result["status"]}, result)
    log_id = stable_id(nodeid)
    stdout_path = logs_dir / f"{log_id}.stdout.txt"
    stderr_path = logs_dir / f"{log_id}.stderr.txt"
    stdout_path.write_text(result["stdout_tail"], encoding="utf-8")
    stderr_path.write_text(result["stderr_tail"], encoding="utf-8")
    result["stdout_path"] = str(stdout_path)
    result["stderr_path"] = str(stderr_path)
    return result


def _probe_program(body: str) -> str:
    indented = textwrap.indent(body.strip(), "    ")
    return f"""
import sys
import torch

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    print("UNSUPPORTED: MPS is not available")
    sys.exit(2)

try:
{indented}
    if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    print("SUPPORTED")
except Exception as exc:
    print(f"UNSUPPORTED: {{type(exc).__name__}}: {{exc}}")
    sys.exit(2)
"""


MPS_SUPPORT_PROBES: dict[str, str] = {
    "dtype_float64": 'torch.zeros(1, dtype=torch.float64, device="mps")',
    "dtype_bfloat16": 'torch.zeros(1, dtype=torch.bfloat16, device="mps") + 1',
    "dtype_complex64": 'torch.zeros(1, dtype=torch.complex64, device="mps") + 1',
    "dtype_complex128": 'torch.zeros(1, dtype=torch.complex128, device="mps") + 1',
    "dtype_fp8_e4m3fn": 'torch.zeros(1, dtype=torch.float8_e4m3fn, device="mps")',
    "dtype_fp8_e5m2": 'torch.zeros(1, dtype=torch.float8_e5m2, device="mps")',
    "quantized": 'torch.quantize_per_tensor(torch.arange(4, dtype=torch.float32), 0.1, 3, torch.quint8).to("mps")',
    "streams_events": 's = torch.Stream(device="mps")\ne = torch.Event(device="mps")\nwith s:\n    x = torch.ones(2, device="mps") + 1\ne.record(s)\ns.synchronize()\nassert e.query()\nassert x.device.type == "mps"',
    "pinned_memory": 'torch.empty(4, pin_memory=True)',
    "foreach": 'a = [torch.ones(2, device="mps")]\nb = [torch.ones(2, device="mps")]\ntorch._foreach_add(a, b)',
    "sparse": 'torch.sparse_coo_tensor(torch.tensor([[0], [0]]), torch.tensor([1.0]), (1, 1), device="mps")',
    "sparse_compressed": 'torch.sparse_csr_tensor(torch.tensor([0, 1], device="mps"), torch.tensor([0], device="mps"), torch.tensor([1.0], device="mps"), size=(1, 1))',
    "nested": 'torch.nested.nested_tensor([torch.ones(2), torch.ones(3)], device="mps")',
    "named_tensor": 'torch.ones((2, 3), device="mps").refine_names("rows", "cols")',
    "compile_training": 'def f(x):\n    y = (x * x).sum()\n    return y\ncf = torch.compile(f)\nx = torch.ones(4, device="mps", requires_grad=True)\ny = cf(x)\ny.backward()',
    "native_batch_norm": 'x = torch.randn(2, 3, 4, 4, device="mps")\nw = torch.ones(3, device="mps")\nb = torch.zeros(3, device="mps")\ntorch.ops.aten._native_batch_norm_legit.no_stats(x, w, b, True, 0.1, 1e-5)',
    "grid_sampler": 'x = torch.randn(1, 1, 3, 3, device="mps")\ng = torch.zeros(1, 2, 2, 2, device="mps")\ntorch.ops.aten.grid_sampler_2d(x, g, 0, 0, False)',
    "linalg_complex": 'torch.linalg.det(torch.eye(2, dtype=torch.complex64, device="mps"))',
}


def run_mps_support_probe(
    *,
    timeout: float = 30.0,
    triage_dir: str | Path = DEFAULT_TRIAGE_DIR,
    cwd: str | Path = ".",
) -> dict:
    probes: dict[str, dict] = {}
    env = os.environ.copy()
    env["TORCHCTS_NON_INTERACTIVE"] = "1"
    for name, body in sorted(MPS_SUPPORT_PROBES.items()):
        command = [sys.executable, "-c", _probe_program(body)]
        result = run_subprocess_command(command, cwd=cwd, timeout=timeout, env=env)
        if result["timed_out"]:
            status = "inconclusive"
        elif is_process_crash(result["returncode"], result["stderr_tail"], result["stdout_tail"]):
            status = "crashes"
        elif result["returncode"] == 0:
            status = "supported"
        elif result["returncode"] == 2:
            status = "unsupported"
        else:
            status = "inconclusive"
        probes[name] = {
            "status": status,
            "returncode": result["returncode"],
            "signal": result["signal"],
            "stdout_tail": result["stdout_tail"],
            "stderr_tail": result["stderr_tail"],
        }
    payload = {
        "generated_at": _utc_now(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "probes": probes,
    }
    _json_dump(Path(triage_dir) / "manifest_support_probe.json", payload)
    return payload


GRID_REPRO_SCRIPT = r'''
import argparse
import torch


def sync():
    if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def compare(actual, expected):
    if isinstance(actual, tuple):
        for a, e in zip(actual, expected):
            compare(a, e)
        return
    sync()
    torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), rtol=1e-4, atol=1e-4)


def segment_backward_default():
    data_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
    lengths_cpu = torch.tensor([2, 3], dtype=torch.long)
    out_cpu = torch.ops.aten.segment_reduce.default(data_cpu, "sum", lengths=lengths_cpu)
    expected = torch.ops.aten._segment_reduce_backward.default(torch.ones_like(out_cpu), out_cpu, data_cpu, "sum", lengths=lengths_cpu)
    data = data_cpu.to("mps")
    lengths = lengths_cpu.to("mps")
    out = torch.ops.aten.segment_reduce.default(data, "sum", lengths=lengths)
    actual = torch.ops.aten._segment_reduce_backward.default(torch.ones_like(out), out, data, "sum", lengths=lengths)
    compare(actual, expected)


def segment_backward_out():
    data_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
    lengths_cpu = torch.tensor([2, 3], dtype=torch.long)
    out_cpu = torch.ops.aten.segment_reduce.default(data_cpu, "sum", lengths=lengths_cpu)
    expected = torch.ops.aten._segment_reduce_backward.default(torch.ones_like(out_cpu), out_cpu, data_cpu, "sum", lengths=lengths_cpu)
    data = data_cpu.to("mps")
    lengths = lengths_cpu.to("mps")
    out = torch.ops.aten.segment_reduce.default(data, "sum", lengths=lengths)
    target = torch.empty_like(data)
    actual = torch.ops.aten._segment_reduce_backward.out(torch.ones_like(out), out, data, "sum", lengths=lengths, out=target)
    compare(actual, expected)


def grid2_inputs():
    input_cpu = torch.linspace(-1.0, 1.0, 9, dtype=torch.float32).reshape(1, 1, 3, 3)
    grid_cpu = torch.tensor([[[[-0.5, -0.5], [0.5, -0.5]], [[-0.5, 0.5], [0.5, 0.5]]]], dtype=torch.float32)
    grad_cpu = torch.linspace(0.1, 0.4, 4, dtype=torch.float32).reshape(1, 1, 2, 2)
    return input_cpu, grid_cpu, grad_cpu


def grid2_cpu_fallback_default():
    input_cpu, grid_cpu, _ = grid2_inputs()
    expected = torch.ops.aten._grid_sampler_2d_cpu_fallback.default(input_cpu, grid_cpu, 0, 0, False)
    actual = torch.ops.aten._grid_sampler_2d_cpu_fallback.default(input_cpu.to("mps"), grid_cpu.to("mps"), 0, 0, False)
    compare(actual, expected)


def grid2_cpu_fallback_out():
    input_cpu, grid_cpu, _ = grid2_inputs()
    expected = torch.ops.aten._grid_sampler_2d_cpu_fallback.default(input_cpu, grid_cpu, 0, 0, False)
    out = torch.empty_like(expected, device="mps")
    actual = torch.ops.aten._grid_sampler_2d_cpu_fallback.out(input_cpu.to("mps"), grid_cpu.to("mps"), 0, 0, False, out=out)
    compare(actual, expected)


def grid2_cpu_fallback_backward():
    input_cpu, grid_cpu, grad_cpu = grid2_inputs()
    expected = torch.ops.aten._grid_sampler_2d_cpu_fallback_backward.default(grad_cpu, input_cpu, grid_cpu, 0, 0, False)
    actual = torch.ops.aten._grid_sampler_2d_cpu_fallback_backward.default(grad_cpu.to("mps"), input_cpu.to("mps"), grid_cpu.to("mps"), 0, 0, False)
    compare(actual, expected)


def grid2_backward_default():
    input_cpu, grid_cpu, grad_cpu = grid2_inputs()
    expected = torch.ops.aten.grid_sampler_2d_backward.default(grad_cpu, input_cpu, grid_cpu, 0, 0, False, [True, True])
    actual = torch.ops.aten.grid_sampler_2d_backward.default(grad_cpu.to("mps"), input_cpu.to("mps"), grid_cpu.to("mps"), 0, 0, False, [True, True])
    compare(actual, expected)


def grid2_backward_out():
    input_cpu, grid_cpu, grad_cpu = grid2_inputs()
    expected = torch.ops.aten.grid_sampler_2d_backward.default(grad_cpu, input_cpu, grid_cpu, 0, 0, False, [True, True])
    input_mps = input_cpu.to("mps")
    grid_mps = grid_cpu.to("mps")
    out0 = torch.empty_like(input_mps)
    out1 = torch.empty_like(grid_mps)
    actual = torch.ops.aten.grid_sampler_2d_backward.out(grad_cpu.to("mps"), input_mps, grid_mps, 0, 0, False, [True, True], out0=out0, out1=out1)
    compare(actual, expected)


def grid3_backward_default():
    input_cpu = torch.linspace(-1.0, 1.0, 27, dtype=torch.float32).reshape(1, 1, 3, 3, 3)
    grid_cpu = torch.zeros(1, 2, 2, 2, 3, dtype=torch.float32)
    grad_cpu = torch.linspace(0.1, 0.8, 8, dtype=torch.float32).reshape(1, 1, 2, 2, 2)
    expected = torch.ops.aten.grid_sampler_3d_backward.default(grad_cpu, input_cpu, grid_cpu, 0, 0, False, [True, True])
    actual = torch.ops.aten.grid_sampler_3d_backward.default(grad_cpu.to("mps"), input_cpu.to("mps"), grid_cpu.to("mps"), 0, 0, False, [True, True])
    compare(actual, expected)


CASES = {
    "segment_backward_default": segment_backward_default,
    "segment_backward_out": segment_backward_out,
    "grid2_cpu_fallback_default": grid2_cpu_fallback_default,
    "grid2_cpu_fallback_out": grid2_cpu_fallback_out,
    "grid2_cpu_fallback_backward": grid2_cpu_fallback_backward,
    "grid2_backward_default": grid2_backward_default,
    "grid2_backward_out": grid2_backward_out,
    "grid3_backward_default": grid3_backward_default,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    args = parser.parse_args()
    print(f"running {args.case}")
    CASES[args.case]()
    sync()
    print("ok")
'''


FACTORY_OUT_REPRO_SCRIPT = r'''
import argparse
import torch


def sync():
    if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def compare(actual, expected):
    sync()
    torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), rtol=0, atol=0)


def range_out_control():
    expected = torch.empty_strided((7,), (1,), dtype=torch.float32, device="cpu")
    returned_cpu = torch.ops.aten.range.out(0, 6, 1, out=expected)
    assert returned_cpu is expected

    actual = torch.empty_strided((7,), (1,), dtype=torch.float32, device="mps")
    returned = torch.ops.aten.range.out(0, 6, 1, out=actual)
    assert returned is actual
    compare(actual, expected)


def range_out_():
    expected = torch.empty_strided((7,), (1,), dtype=torch.float32, device="cpu")
    returned_cpu = torch.ops.aten.range.out_(0, 6, out=expected)
    assert returned_cpu is expected

    actual = torch.empty_strided((7,), (1,), dtype=torch.float32, device="mps")
    returned = torch.ops.aten.range.out_(0, 6, out=actual)
    assert returned is actual
    sync()
    print("metadata", tuple(actual.shape), actual.dtype, actual.device, actual.stride())
    compare(actual, expected)


CASES = {
    "range_out_control": range_out_control,
    "range_out_": range_out_,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    args = parser.parse_args()
    print(f"running {args.case}")
    CASES[args.case]()
    sync()
    print("ok")
'''


def write_repro_scripts(triage_dir: str | Path = DEFAULT_TRIAGE_DIR) -> dict[str, str]:
    repro_dir = Path(triage_dir) / "repros"
    repro_dir.mkdir(parents=True, exist_ok=True)
    scripts = {
        "factory_out": repro_dir / "mps_factory_out_repro.py",
        "grid_segment_dense_split": repro_dir / "mps_grid_segment_dense_split_repro.py",
    }
    scripts["factory_out"].write_text(FACTORY_OUT_REPRO_SCRIPT.lstrip(), encoding="utf-8")
    scripts["grid_segment_dense_split"].write_text(GRID_REPRO_SCRIPT.lstrip(), encoding="utf-8")
    return {name: str(path) for name, path in scripts.items()}


def run_known_repros(
    *,
    triage_dir: str | Path = DEFAULT_TRIAGE_DIR,
    timeout: float = 120.0,
    cwd: str | Path = ".",
) -> list[dict]:
    scripts = write_repro_scripts(triage_dir)
    runs: list[dict] = []
    grid_cases = [
        "segment_backward_default",
        "segment_backward_out",
        "grid2_cpu_fallback_default",
        "grid2_cpu_fallback_out",
        "grid2_cpu_fallback_backward",
        "grid2_backward_default",
        "grid2_backward_out",
        "grid3_backward_default",
    ]
    for case in grid_cases:
        command = [sys.executable, scripts["grid_segment_dense_split"], "--case", case]
        result = run_subprocess_command(command, cwd=cwd, timeout=timeout, env={**os.environ, "TORCHCTS_NON_INTERACTIVE": "1"})
        result["repro"] = "grid_segment_dense_split"
        result["case"] = case
        status = "ERROR" if result["timed_out"] or result["returncode"] else "PASS"
        result["status"] = status
        result["classification"] = classify_record({"nodeid": case, "status": status}, result)
        runs.append(result)
    factory_cases = [
        "range_out_control",
        "range_out_",
    ]
    for case in factory_cases:
        command = [sys.executable, scripts["factory_out"], "--case", case]
        result = run_subprocess_command(command, cwd=cwd, timeout=timeout, env={**os.environ, "TORCHCTS_NON_INTERACTIVE": "1"})
        result["repro"] = "factory_out"
        result["case"] = case
        status = "ERROR" if result["timed_out"] or result["returncode"] else "PASS"
        result["status"] = status
        result["classification"] = classify_record({"nodeid": case, "status": status}, result)
        runs.append(result)
    _json_dump(Path(triage_dir) / "crashes.json", {"generated_at": _utc_now(), "runs": runs})
    return runs


def repro_counts(repro_runs: list[dict]) -> dict[str, int]:
    counts = {
        "confirmed_mps_crash": 0,
        "pass": 0,
        "timeout": 0,
        "needs_more_evidence": 0,
    }
    for run in repro_runs:
        if run.get("timed_out"):
            counts["timeout"] += 1
            continue
        if run.get("status") == "PASS":
            counts["pass"] += 1
            continue
        classification = (run.get("classification") or {}).get("classification", "needs_more_evidence")
        counts[classification] = counts.get(classification, 0) + 1
    return counts


def active_known_segfault_summary(
    *,
    cwd: str | Path = ".",
    torch_version: str | None = None,
    hardware_key: str | None = None,
) -> list[dict]:
    try:
        import torch
        from torchcts.core.known_segfaults import active_known_segfaults, load_known_segfaults

        entries = load_known_segfaults(cwd)
        active = active_known_segfaults(
            entries,
            backend="mps",
            torch_version=torch_version or torch.__version__,
            hardware_key=hardware_key or "any",
        )
        return [
            {
                "id": entry["id"],
                "nodeid": entry["nodeid"],
                "dispatcher": entry["dispatcher"],
                "expected_signal": entry["expected_signal"],
                "repro": entry["repro"],
                "reason": entry["reason"],
                "review_after": entry["review_after"],
            }
            for entry in active
        ]
    except Exception:
        return []


def render_summary(payload: dict) -> str:
    classifications = payload.get("classifications", {})
    counts: dict[str, int] = {}
    for item in classifications.values():
        key = item.get("classification", "needs_more_evidence")
        counts[key] = counts.get(key, 0) + 1
    lines = [
        "# MPS Triage Summary",
        "",
        f"Generated: {payload.get('generated_at')}",
        f"Input result: {payload.get('input_result') or 'none'}",
        f"Queued nodes: {len(payload.get('queue', []))}",
        "",
        "## Classification Counts",
        "",
    ]
    if counts:
        for key in sorted(counts):
            lines.append(f"- `{key}`: {counts[key]}")
    else:
        lines.append("- No failing records were classified.")
    lines.extend(["", "## Known Segfaults", ""])
    known_entries = payload.get("known_segfaults", [])
    if known_entries:
        lines.append(f"- active entries: {len(known_entries)}")
        for entry in known_entries:
            lines.append(
                f"- `{entry.get('id')}`: `{entry.get('dispatcher')}` "
                f"expected_signal=`{entry.get('expected_signal')}`"
            )
    else:
        lines.append("- No active known segfault entries.")
    lines.extend(["", "## Standalone Repros", ""])
    repro_runs = payload.get("repro_runs", [])
    if repro_runs:
        counts = payload.get("repro_counts") or repro_counts(repro_runs)
        for key in sorted(counts):
            lines.append(f"- `{key}`: {counts[key]}")
        lines.append("")
        for run in repro_runs:
            cls = (run.get("classification") or {}).get("classification", "needs_more_evidence")
            lines.append(
                f"- `{run.get('repro')}` / `{run.get('case')}`: status={run.get('status')} "
                f"returncode={run.get('returncode')} signal={run.get('signal')} classification=`{cls}`"
            )
    else:
        lines.append("- No standalone repros were run.")
    lines.extend(["", "## Subprocess Runs", ""])
    runs = payload.get("subprocess_runs", [])
    if runs:
        for run in runs:
            cls = run.get("classification", {}).get("classification", "needs_more_evidence")
            lines.append(
                f"- `{run.get('nodeid')}`: returncode={run.get('returncode')} "
                f"signal={run.get('signal')} timed_out={run.get('timed_out')} classification=`{cls}`"
            )
    else:
        lines.append("- No pytest node subprocess runs were requested.")
    lines.extend(["", "## Repro Scripts", ""])
    scripts = payload.get("repro_scripts", {})
    if scripts:
        for name, path in sorted(scripts.items()):
            lines.append(f"- `{name}`: `{path}`")
    else:
        lines.append("- No repro scripts were written.")
    return "\n".join(lines) + "\n"


def run_mps_triage(
    *,
    from_file: str | Path | None = None,
    include_crashers: bool = False,
    nodes_file: str | Path | None = None,
    triage_dir: str | Path = DEFAULT_TRIAGE_DIR,
    timeout: float = 120.0,
    level: int = 8,
    run_nodes: bool = True,
    repros_only: bool = False,
    cwd: str | Path = ".",
) -> dict:
    triage_path = Path(triage_dir)
    triage_path.mkdir(parents=True, exist_ok=True)
    if repros_only and from_file is None:
        result_path, result_data = None, {}
        copied = []
    else:
        result_path, result_data = load_result_file(from_file)
        copied = freeze_input_snapshot(result_path, result_data, triage_path)
    hardware_key = _hardware_key_from_result(result_path, result_data)
    runlog_path = None
    if result_path and hardware_key:
        candidate = result_path.parent / f"{hardware_key}_runlog.txt"
        if candidate.exists():
            runlog_path = candidate

    queue = [] if repros_only else build_triage_queue(
        result_data,
        include_crashers=include_crashers,
        nodes_file=nodes_file,
        runlog_path=runlog_path,
    )
    classifications = classify_records(result_data)
    support_probe = None if repros_only else run_mps_support_probe(timeout=min(timeout, 30.0), triage_dir=triage_path, cwd=cwd)
    repro_scripts = write_repro_scripts(triage_path)
    repro_runs = (
        run_known_repros(triage_dir=triage_path, timeout=timeout, cwd=cwd)
        if repros_only or (include_crashers and run_nodes)
        else []
    )

    subprocess_runs: list[dict] = []
    if run_nodes and not repros_only:
        for nodeid in queue:
            run = run_pytest_node(nodeid, device="mps", level=level, timeout=timeout, triage_dir=triage_path, cwd=cwd)
            subprocess_runs.append(run)
            if run.get("classification"):
                classifications[nodeid] = run["classification"]

    known_segfaults = active_known_segfault_summary(
        cwd=cwd,
        torch_version=(result_data or {}).get("metadata", {}).get("pytorch_version"),
        hardware_key=hardware_key,
    )
    adjudication_queue = build_adjudication_queue(
        result_data,
        classifications,
        result_path=result_path,
    )

    payload = {
        "generated_at": _utc_now(),
        "input_result": str(result_path) if result_path else None,
        "input_snapshot": copied,
        "queue": queue,
        "classifications": classifications,
        "support_probe": support_probe,
        "known_segfaults": known_segfaults,
        "repro_scripts": repro_scripts,
        "repro_runs": repro_runs,
        "repro_counts": repro_counts(repro_runs),
        "subprocess_runs": subprocess_runs,
        "adjudication_queue": adjudication_queue,
    }
    _json_dump(triage_path / "classifications.json", classifications)
    _json_dump(triage_path / "failures.json", {"generated_at": payload["generated_at"], "runs": subprocess_runs})
    _json_dump(triage_path / "adjudication_queue.json", adjudication_queue)
    _json_dump(triage_path / "triage.json", payload)
    summary = render_summary(payload)
    (triage_path / "summary.md").write_text(summary, encoding="utf-8")
    return payload
