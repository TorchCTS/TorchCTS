#!/usr/bin/env python3
"""Materialize TorchCTS CPU dtype contracts from CPU probes.

The contract file is intentionally evidence-based:

* source metadata is informative;
* CPU probe results decide executable dtype cases;
* unknown probe results are recorded and kept out of executable tests.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torchcts.core.dtype_contracts import (
    CPU_PENDING,
    CPU_SUPPORTED_BUT_MISSING_FROM_SOURCE,
    SOURCE_DECLARED_BUT_PROBE_UNKNOWN,
    SOURCE_EXPECTED_BUT_CPU_UNSUPPORTED,
    dtype_from_contract_str,
    dtype_to_contract_str,
    phase_condition_key,
)
from torchcts.op_metadata import load_op_metadata, op_available_in_runtime


CONTRACT_PATH = REPO_ROOT / "torchcts" / "op_dtype_contracts.json"

DEFAULT_DTYPE_UNIVERSE = (
    "torch.bool",
    "torch.uint8",
    "torch.int8",
    "torch.int16",
    "torch.int32",
    "torch.int64",
    "torch.float16",
    "torch.bfloat16",
    "torch.float32",
    "torch.float64",
    "torch.complex32",
    "torch.complex64",
    "torch.complex128",
)

UFUNC_DTYPE_TOKENS = {
    "AllAndComplex": (
        "torch.uint8",
        "torch.int8",
        "torch.int16",
        "torch.int32",
        "torch.int64",
        "torch.float32",
        "torch.float64",
        "torch.complex64",
        "torch.complex128",
    ),
    "BFloat16": ("torch.bfloat16",),
    "Half": ("torch.float16",),
    "ComplexHalf": ("torch.complex32",),
    "Bool": ("torch.bool",),
}


OPINFO_PROBE_CODE = r"""
import json
import sys

import torch

from torchcts.core.dtype_contracts import is_deterministic_cpu_unsupported
from torchcts.core.opinfo_adapter import InputCondition, is_cpu_reference_failure, prepare_sample, str_to_dtype

op_name, dtype_str, phase = sys.argv[1:4]
dtype = str_to_dtype(dtype_str)
if dtype is None:
    print(json.dumps({"status": "unknown", "detail": f"unknown dtype {dtype_str}"}))
    raise SystemExit(0)

try:
    import torch.testing._internal.common_methods_invocations as cmi

    op = next((candidate for candidate in cmi.op_db if candidate.name == op_name), None)
    if op is None:
        print(json.dumps({"status": "unknown", "detail": f"missing OpInfo {op_name}"}))
        raise SystemExit(0)
    samples = op.sample_inputs("cpu", dtype, requires_grad=(phase == "backward"))
    raw_sample = next(iter(samples), None)
    if raw_sample is None:
        print(json.dumps({"status": "unknown", "detail": "sample_inputs produced no CPU sample"}))
        raise SystemExit(0)
    sample = prepare_sample(raw_sample, InputCondition.CLEAN, op_name=op_name)
    op.op(sample.input, *sample.args, **sample.kwargs)
    print(json.dumps({"status": "supported", "detail": "clean CPU sample executed"}))
except BaseException as exc:
    status = "unsupported" if is_deterministic_cpu_unsupported(exc) or is_cpu_reference_failure(exc) else "unknown"
    print(json.dumps({"status": status, "detail": f"{type(exc).__name__}: {exc}"}))
"""


GENERATED_PROBE_CODE = r"""
import json
import sys

from torchcts.core.opinfo_adapter import str_to_dtype
from torchcts.generated.coverage_helpers import probe_generated_clean_cpu_contract
from torchcts.generated.generated_cases import GENERATED_CASES

entry_name, dtype_str = sys.argv[1:3]
dtype = str_to_dtype(dtype_str)
if dtype is None:
    print(json.dumps({"status": "unknown", "detail": f"unknown dtype {dtype_str}"}))
    raise SystemExit(0)

entry = None
for candidates in GENERATED_CASES.get("cases_by_surface", {}).values():
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("name") == entry_name:
            entry = candidate
            break
    if entry is not None:
        break
if entry is None:
    print(json.dumps({"status": "unknown", "detail": f"missing generated entry {entry_name}"}))
    raise SystemExit(0)

print(json.dumps(probe_generated_clean_cpu_contract(entry, dtype, {}, enforce_recorded_contract=False)))
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _version_rule() -> str:
    parts = torch.__version__.split("+", 1)[0].split(".")
    return ".".join(parts[:2])


def _load_contracts(path: Path = CONTRACT_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"version": 1, "metadata": {}, "contracts": {}}


def _write_contracts(data: dict, path: Path = CONTRACT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _selected_dtypes(raw_values: list[str] | None) -> tuple[str, ...]:
    if not raw_values:
        values = DEFAULT_DTYPE_UNIVERSE
    else:
        expanded: list[str] = []
        for raw in raw_values:
            expanded.extend(item.strip() for item in raw.split(",") if item.strip())
        values = tuple(expanded)
    normalized = []
    for value in values:
        dtype_str = dtype_to_contract_str(value)
        if dtype_str and dtype_from_contract_str(dtype_str) is not None and dtype_str not in normalized:
            normalized.append(dtype_str)
    return tuple(normalized)


def _contract_key(op_name: str) -> str:
    return op_name if op_name.startswith("aten::") else f"aten::{op_name}"


def _source_key_for_phase(phase: str | None) -> str:
    return f"{phase}:*" if phase else "*"


def _version_entry(data: dict, op_name: str, version_rule: str) -> dict:
    return data.setdefault("contracts", {}).setdefault(_contract_key(op_name), {}).setdefault(version_rule, {})


def _set_source_expected(
    data: dict,
    op_name: str,
    source_expected: Iterable[str],
    *,
    version_rule: str,
    phase: str | None,
) -> None:
    dtypes = sorted({dtype for dtype in source_expected if dtype})
    if not dtypes:
        return
    entry = _version_entry(data, op_name, version_rule)
    source_map = entry.setdefault("source_expected", {})
    key = _source_key_for_phase(phase)
    source_map[key] = sorted(set(source_map.get(key, [])) | set(dtypes))


def _remove_dtype_from_probe_buckets(entry: dict, dtype_str: str, condition_key: str) -> None:
    for bucket in ("cpu_supported", "cpu_unsupported", "cpu_unknown", "cpu_pending"):
        condition_map = entry.setdefault(bucket, {})
        for key in ("clean", condition_key):
            values = set(condition_map.get(key, []))
            values.discard(dtype_str)
            if values:
                condition_map[key] = sorted(values)
            else:
                condition_map.pop(key, None)


def _record_probe_result(
    data: dict,
    op_name: str,
    dtype_str: str,
    phase: str,
    source_expected: Iterable[str],
    result: dict,
    version_rule: str,
    *,
    evidence_source: str,
) -> None:
    source_values = sorted({dtype for dtype in source_expected if dtype})
    entry = _version_entry(data, op_name, version_rule)
    _set_source_expected(data, op_name, source_values, version_rule=version_rule, phase=phase)

    bucket = {
        "supported": "cpu_supported",
        "unsupported": "cpu_unsupported",
        "unknown": "cpu_unknown",
        "pending": "cpu_pending",
    }.get(result.get("status"), "cpu_unknown")

    condition_key = phase_condition_key("clean", phase)
    _remove_dtype_from_probe_buckets(entry, dtype_str, condition_key)
    values = set(entry.setdefault(bucket, {}).setdefault(condition_key, []))
    values.add(dtype_str)
    entry[bucket][condition_key] = sorted(values)
    entry.setdefault("probe_details", {}).setdefault(condition_key, {})[dtype_str] = {
        "status": result.get("status", "unknown"),
        "detail": result.get("detail", ""),
        "source": evidence_source,
    }

    mismatch = None
    if result.get("status") == "supported" and dtype_str not in source_values:
        mismatch = CPU_SUPPORTED_BUT_MISSING_FROM_SOURCE
    elif result.get("status") == "unsupported" and dtype_str in source_values:
        mismatch = SOURCE_EXPECTED_BUT_CPU_UNSUPPORTED
    elif result.get("status") == "unknown" and dtype_str in source_values:
        mismatch = SOURCE_DECLARED_BUT_PROBE_UNKNOWN
    if mismatch:
        records = entry.setdefault("source_probe_mismatches", [])
        record = {
            "kind": mismatch,
            "dtype": dtype_str,
            "phase": phase,
            "detail": result.get("detail", ""),
        }
        if record not in records:
            records.append(record)

    evidence = entry.setdefault("evidence", {})
    evidence.update(
        {
            "source": evidence_source,
            "pytorch_version": torch.__version__,
            "last_probe_detail": result.get("detail", ""),
        }
    )
    phases = set(evidence.get("phases", []))
    phases.add(phase)
    evidence["phases"] = sorted(phases)


def _clear_phase_entries(data: dict, *, phase: str, version_rule: str, source_prefix: str | None = None) -> None:
    prefix = f"{phase}:"
    for versions in data.get("contracts", {}).values():
        if not isinstance(versions, dict):
            continue
        version_entry = versions.get(version_rule)
        if not isinstance(version_entry, dict):
            continue
        evidence = version_entry.get("evidence") or {}
        if source_prefix and not str(evidence.get("source", "")).startswith(source_prefix):
            continue
        for bucket in (
            "cpu_supported",
            "cpu_unsupported",
            "cpu_unknown",
            "cpu_pending",
            "oracle_supported",
            "source_expected",
            "probe_details",
        ):
            condition_map = version_entry.get(bucket)
            if not isinstance(condition_map, dict):
                continue
            for condition in list(condition_map):
                if str(condition).startswith(prefix):
                    condition_map.pop(condition, None)
        mismatches = version_entry.get("source_probe_mismatches")
        if isinstance(mismatches, list):
            version_entry["source_probe_mismatches"] = [
                record
                for record in mismatches
                if not (isinstance(record, dict) and record.get("phase") == phase)
            ]


def _run_probe(code: str, args: list[str], timeout: float) -> dict:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code, *args],
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "unknown", "detail": f"probe timed out after {timeout:g}s"}
    output = (proc.stdout or "").strip().splitlines()
    if not output:
        detail = (proc.stderr or "").strip() or f"probe exited {proc.returncode}"
        return {"status": "unknown", "detail": detail[-1000:]}
    try:
        return json.loads(output[-1])
    except json.JSONDecodeError:
        detail = "\n".join(output[-5:])
        return {"status": "unknown", "detail": detail[-1000:]}


def _normalize_probe_result(result: dict) -> dict:
    if not isinstance(result, dict):
        return {"status": "unknown", "detail": f"probe returned {type(result).__name__}"}
    status = result.get("status", "unknown")
    detail = str(result.get("detail", ""))
    lower = detail.lower()
    if status != "unknown":
        return result

    unsupported_markers = (
        "sample rejected dtype",
        "not implemented for",
        "not supported",
        "unsupported",
        "input dtype should be",
        "input dtype must be",
        "expects floating point dtype",
        "expects floating point dtypes",
        "expected input to have floating point",
        "requires a floating point",
        "requires floating point",
        "dtype should be",
        "dtype must be",
        "could not infer output dtype",
        "low precision dtypes not supported",
        "expected both inputs to be half, float or double",
        "expected a tensor with 2 or more dimensions of float, double, cfloat or cdouble types",
        "must be same dtype",
        "should have the same dtype",
        "not yet implemented for complex",
        "operation's result requires dtype",
        "normal expects standard deviation to be non-complex",
        "input tensor may not be a boolean tensor",
        "unrecognized scalartype: complexha",
        "sample rejected dtype",
    )
    if any(marker in lower for marker in unsupported_markers):
        normalized = dict(result)
        normalized["status"] = "unsupported"
        return normalized

    pending_markers = (
        "sample generation failed",
        "expects a tensor with <= 2 dimensions",
        "value cannot be converted to type uint8_t without overflow",
        "out of bounds for bool",
        "should be convertible without narrowing to the specified dtype",
        "takes ",
        "positional argument",
        "missing value for argument",
        "expected a value of type",
        "dimension out of range",
        "need input of dimension",
        "none zero group size expected",
        "same dtype as input",
        "not enough expected outputs",
        "expected at most",
        "padding mode",
        "no opinfo sample",
    )
    if any(marker in lower for marker in pending_markers):
        normalized = dict(result)
        normalized["status"] = "pending"
        if not normalized.get("detail"):
            normalized["detail"] = "pending sample/probe construction evidence"
        return normalized
    return result


def _opinfo_source_dtypes(op, phase: str) -> list[str]:
    from torchcts.core.opinfo_adapter import DIFFERENTIABLE

    attr = "backward_dtypes" if phase == "backward" else "dtypes"
    dtypes = getattr(op, attr, ())
    if phase == "backward":
        dtypes = [dtype for dtype in dtypes if dtype in DIFFERENTIABLE]
    return sorted({dtype_to_contract_str(dtype) for dtype in dtypes if dtype_to_contract_str(dtype)})


def _opinfo_contract_in_scope(op, phase: str) -> bool:
    from torchcts.core.opinfo_adapter import SKIP_OPS, _NO_GENERIC_BACKWARD_ORACLE_OPS

    if op.name in SKIP_OPS:
        return False
    if phase == "backward":
        if op.name in _NO_GENERIC_BACKWARD_ORACLE_OPS:
            return False
        if not getattr(op, "supports_autograd", True):
            return False
    return True


def _seed_source_metadata(data: dict, *, version_rule: str) -> int:
    count = 0
    for name, metadata in (load_op_metadata().get("ops") or {}).items():
        if not op_available_in_runtime(name):
            continue
        dtypes = [dtype_to_contract_str(dtype) for dtype in metadata.get("pytorch_dtypes", ()) or ()]
        dtypes = [dtype for dtype in dtypes if dtype]
        if not dtypes:
            continue
        _set_source_expected(data, name, dtypes, version_rule=version_rule, phase=None)
        entry = _version_entry(data, name, version_rule)
        entry.setdefault("evidence", {}).update(
            {
                "source": "op_metadata.pytorch_dtypes",
                "execution_authority": "none_source_expectation_only",
                "pytorch_version": version_rule,
            }
        )
        count += 1
    return count


def _dispatcher_name_from_native_func(func: str) -> str | None:
    match = re.match(r"([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)?)\(", str(func))
    if not match:
        return None
    return f"aten::{match.group(1)}"


def _source_dtypes_from_ufunc_loop(loop: dict) -> list[str]:
    dtypes = set()
    for value in (loop or {}).values():
        for group in re.findall(r"\(([^)]*)\)", str(value)):
            for token in (part.strip() for part in group.split(",")):
                dtypes.update(UFUNC_DTYPE_TOKENS.get(token, ()))
    return sorted(dtypes)


def _seed_pytorch_source_ufunc_metadata(data: dict, *, version_rule: str, pytorch_src: Path) -> int:
    native_yaml = pytorch_src / "aten" / "src" / "ATen" / "native" / "native_functions.yaml"
    if not native_yaml.exists():
        return 0
    try:
        import yaml

        records = yaml.safe_load(native_yaml.read_text(encoding="utf-8")) or []
    except Exception:
        return 0
    count = 0
    for record in records:
        if not isinstance(record, dict) or "ufunc_inner_loop" not in record:
            continue
        dispatcher_name = _dispatcher_name_from_native_func(str(record.get("func", "")))
        dtypes = _source_dtypes_from_ufunc_loop(record.get("ufunc_inner_loop") or {})
        if not dispatcher_name or not dtypes:
            continue
        if not op_available_in_runtime(dispatcher_name):
            continue
        _set_source_expected(data, dispatcher_name, dtypes, version_rule=version_rule, phase=None)
        entry = _version_entry(data, dispatcher_name, version_rule)
        evidence = entry.setdefault("evidence", {})
        layers = set(evidence.get("source_layers", []))
        layers.add("pytorch_src.native_functions.ufunc_inner_loop")
        evidence["source_layers"] = sorted(layers)
        evidence.setdefault("execution_authority", "none_source_expectation_only")
        evidence["pytorch_source_tree"] = str(pytorch_src)
        count += 1
    return count


def _iter_opinfo_probe_cases(phase: str, selected_dtypes: tuple[str, ...]):
    import torch.testing._internal.common_methods_invocations as cmi

    for op in cmi.op_db:
        if not _opinfo_contract_in_scope(op, phase):
            continue
        source_expected = _opinfo_source_dtypes(op, phase)
        for dtype_str in source_expected:
            if dtype_str in selected_dtypes:
                yield op.name, dtype_str, source_expected


def _iter_generated_entries():
    from torchcts.generated.generated_cases import GENERATED_CASES

    seen = set()
    for candidates in GENERATED_CASES.get("cases_by_surface", {}).values():
        for entry in candidates:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            strategy = (entry.get("generated") or {}).get("strategy") or {}
            if not name or not strategy or name in seen:
                continue
            if not op_available_in_runtime(name):
                continue
            if entry.get("status") not in {"covered_generated", "unknown"}:
                continue
            seen.add(name)
            yield entry


def _source_expected_for_generated(entry: dict) -> list[str]:
    metadata = load_op_metadata().get("ops", {}).get(entry["name"], {})
    return sorted({dtype_to_contract_str(dtype) for dtype in metadata.get("pytorch_dtypes", ()) or () if dtype_to_contract_str(dtype)})


def _iter_generated_probe_cases(selected_dtypes: tuple[str, ...]):
    for entry in _iter_generated_entries():
        source_expected = _source_expected_for_generated(entry)
        for dtype_str in selected_dtypes:
            yield entry["name"], dtype_str, source_expected


def _contract_counts(data: dict) -> dict[str, int]:
    counts = Counter()
    source_expected_ops = 0
    source_expected_entries = 0
    for versions in (data.get("contracts") or {}).values():
        if not isinstance(versions, dict):
            continue
        for entry in versions.values():
            if not isinstance(entry, dict):
                continue
            source_map = entry.get("source_expected") or {}
            if source_map:
                source_expected_ops += 1
                for dtypes in source_map.values():
                    source_expected_entries += len(dtypes or ())
            for bucket in ("cpu_supported", "cpu_unsupported", "cpu_unknown", "cpu_pending", "oracle_supported"):
                for dtypes in (entry.get(bucket) or {}).values():
                    counts[bucket] += len(dtypes or ())
            counts["source_probe_mismatches"] += len(entry.get("source_probe_mismatches") or ())
    counts["source_expected_ops"] = source_expected_ops
    counts["source_expected_entries"] = source_expected_entries
    return dict(sorted(counts.items()))


def _source_tree_metadata(pytorch_src: Path, *, op_metadata_seed_count: int, ufunc_seed_count: int) -> dict:
    common_methods = pytorch_src / "torch" / "testing" / "_internal" / "common_methods_invocations.py"
    native_yaml = pytorch_src / "aten" / "src" / "ATen" / "native" / "native_functions.yaml"
    return {
        "op_metadata_seeded_ops": op_metadata_seed_count,
        "installed_opinfo_source_used": True,
        "pytorch_src_path": str(pytorch_src),
        "pytorch_src_available": pytorch_src.exists(),
        "pytorch_src_common_methods_invocations": common_methods.exists(),
        "pytorch_src_native_functions_yaml": native_yaml.exists(),
        "pytorch_src_ufunc_inner_loop_seeded_ops": ufunc_seed_count,
        "note": (
            "PyTorch source metadata is investigation evidence only. "
            "Executable dtype cases come from CPU probes or explicit reviewed oracle entries."
        ),
    }


def _record_metadata(
    data: dict,
    *,
    args,
    counts: Counter,
    selected_dtypes: tuple[str, ...],
    source_seed_count: int,
    ufunc_seed_count: int,
) -> None:
    data["metadata"] = {
        "contract_authority": "cpu_probe",
        "generated_at": _utc_now(),
        "generated_by": "scripts/generate_op_dtype_contracts.py",
        "pytorch_version": torch.__version__,
        "version_rule": args.version_rule,
        "last_refreshed_layers": args.layer,
        "selected_dtypes": list(selected_dtypes),
        "last_run_probe_counts": dict(sorted(counts.items())),
        "contract_counts": _contract_counts(data),
        "source_extraction": _source_tree_metadata(
            Path(args.pytorch_src),
            op_metadata_seed_count=source_seed_count,
            ufunc_seed_count=ufunc_seed_count,
        ),
    }


def _run_opinfo_layer(data: dict, *, phase: str, selected_dtypes: tuple[str, ...], args, counts: Counter) -> int:
    _clear_phase_entries(data, phase=phase, version_rule=args.version_rule, source_prefix=None)
    probed = 0
    for op_name, dtype_str, source_expected in _iter_opinfo_probe_cases(phase, selected_dtypes):
        result = _run_probe(OPINFO_PROBE_CODE, [op_name, dtype_str, phase], args.timeout) if args.isolated else _probe_opinfo_inprocess(op_name, dtype_str, phase)
        result = _normalize_probe_result(result)
        _record_probe_result(
            data,
            op_name,
            dtype_str,
            phase,
            source_expected,
            result,
            args.version_rule,
            evidence_source="bounded_cpu_probe.opinfo",
        )
        counts[f"opinfo_{phase}_{result.get('status', 'unknown')}"] += 1
        probed += 1
        _maybe_report_progress(args, probed, f"opinfo {phase}")
        if args.limit and probed >= args.limit:
            break
    return probed


def _probe_opinfo_inprocess(op_name: str, dtype_str: str, phase: str) -> dict:
    from torchcts.core.dtype_contracts import is_deterministic_cpu_unsupported
    from torchcts.core.opinfo_adapter import InputCondition, is_cpu_reference_failure, prepare_sample, str_to_dtype
    import torch.testing._internal.common_methods_invocations as cmi

    dtype = str_to_dtype(dtype_str)
    if dtype is None:
        return {"status": "unknown", "detail": f"unknown dtype {dtype_str}"}
    op = next((candidate for candidate in cmi.op_db if candidate.name == op_name), None)
    if op is None:
        return {"status": "unknown", "detail": f"missing OpInfo {op_name}"}
    try:
        samples = op.sample_inputs("cpu", dtype, requires_grad=(phase == "backward"))
        raw_sample = next(iter(samples), None)
        if raw_sample is None:
            return {"status": "unknown", "detail": "sample_inputs produced no CPU sample"}
        sample = prepare_sample(raw_sample, InputCondition.CLEAN, op_name=op.name)
        op.op(sample.input, *sample.args, **sample.kwargs)
        return {"status": "supported", "detail": "clean CPU sample executed"}
    except BaseException as exc:
        status = "unsupported" if is_deterministic_cpu_unsupported(exc) or is_cpu_reference_failure(exc) else "unknown"
        return {"status": status, "detail": f"{type(exc).__name__}: {exc}"}


def _run_generated_layer(data: dict, *, selected_dtypes: tuple[str, ...], args, counts: Counter) -> int:
    _clear_phase_entries(data, phase="forward", version_rule=args.version_rule, source_prefix="bounded_cpu_probe.generated")
    probed = 0
    for op_name, dtype_str, source_expected in _iter_generated_probe_cases(selected_dtypes):
        result = _run_probe(GENERATED_PROBE_CODE, [op_name, dtype_str], args.timeout) if args.isolated else _probe_generated_inprocess(op_name, dtype_str)
        result = _normalize_probe_result(result)
        _record_probe_result(
            data,
            op_name,
            dtype_str,
            "forward",
            source_expected,
            result,
            args.version_rule,
            evidence_source="bounded_cpu_probe.generated",
        )
        counts[f"generated_{result.get('status', 'unknown')}"] += 1
        probed += 1
        _maybe_report_progress(args, probed, "generated")
        if args.limit and probed >= args.limit:
            break
    return probed


def _probe_generated_inprocess(op_name: str, dtype_str: str) -> dict:
    from torchcts.core.opinfo_adapter import str_to_dtype
    from torchcts.generated.coverage_helpers import probe_generated_clean_cpu_contract

    dtype = str_to_dtype(dtype_str)
    if dtype is None:
        return {"status": "unknown", "detail": f"unknown dtype {dtype_str}"}
    entry = _generated_entry_map().get(op_name)
    if entry is not None:
        return probe_generated_clean_cpu_contract(entry, dtype, {}, enforce_recorded_contract=False)
    return {"status": "unknown", "detail": f"missing generated entry {op_name}"}


_GENERATED_ENTRY_MAP: dict[str, dict] | None = None


def _generated_entry_map() -> dict[str, dict]:
    global _GENERATED_ENTRY_MAP
    if _GENERATED_ENTRY_MAP is None:
        _GENERATED_ENTRY_MAP = {entry["name"]: entry for entry in _iter_generated_entries()}
    return _GENERATED_ENTRY_MAP


def _maybe_report_progress(args, probed: int, label: str) -> None:
    if not args.quiet:
        return
    if args.summary_every and probed % args.summary_every == 0:
        print(f"Probed {probed} {label} case(s)", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtypes", action="append", help="dtype or comma-separated dtypes to probe")
    parser.add_argument(
        "--layer",
        action="append",
        choices=("source", "opinfo-forward", "opinfo-backward", "generated", "all"),
        default=None,
        help="contract layer to refresh; defaults to all",
    )
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--isolated", action="store_true", help="probe each case in a subprocess with --timeout")
    parser.add_argument("--limit", type=int, default=0, help="limit probes per selected probing layer")
    parser.add_argument("--quiet", action="store_true", help="suppress per-probe output")
    parser.add_argument("--summary-every", type=int, default=250, help="print progress every N probes when --quiet is set")
    parser.add_argument("--version-rule", default=_version_rule())
    parser.add_argument("--out", type=Path, default=CONTRACT_PATH, help="contract output path")
    parser.add_argument(
        "--pytorch-src",
        default=str(REPO_ROOT.parent / "pytorch-src"),
        help="optional local PyTorch source checkout for informative source metadata",
    )
    args = parser.parse_args(argv)

    layers = args.layer or ["all"]
    if "all" in layers:
        layers = ["source", "opinfo-forward", "opinfo-backward", "generated"]
    args.layer = layers

    selected_dtypes = _selected_dtypes(args.dtypes)
    data = _load_contracts(args.out)
    counts: Counter = Counter()

    source_seed_count = 0
    ufunc_seed_count = 0
    if "source" in layers:
        source_seed_count = _seed_source_metadata(data, version_rule=args.version_rule)
        ufunc_seed_count = _seed_pytorch_source_ufunc_metadata(
            data,
            version_rule=args.version_rule,
            pytorch_src=Path(args.pytorch_src),
        )
        counts["source_seeded_ops"] = source_seed_count
        counts["pytorch_src_ufunc_seeded_ops"] = ufunc_seed_count
    else:
        previous_source = ((data.get("metadata") or {}).get("source_extraction") or {})
        source_seed_count = int(previous_source.get("op_metadata_seeded_ops") or 0)
        ufunc_seed_count = int(previous_source.get("pytorch_src_ufunc_inner_loop_seeded_ops") or 0)

    if "opinfo-forward" in layers:
        probed = _run_opinfo_layer(data, phase="forward", selected_dtypes=selected_dtypes, args=args, counts=counts)
        counts["opinfo_forward_total"] = probed

    if "opinfo-backward" in layers:
        probed = _run_opinfo_layer(data, phase="backward", selected_dtypes=selected_dtypes, args=args, counts=counts)
        counts["opinfo_backward_total"] = probed

    if "generated" in layers:
        probed = _run_generated_layer(data, selected_dtypes=selected_dtypes, args=args, counts=counts)
        counts["generated_total"] = probed

    _record_metadata(
        data,
        args=args,
        counts=counts,
        selected_dtypes=selected_dtypes,
        source_seed_count=source_seed_count,
        ufunc_seed_count=ufunc_seed_count,
    )
    _write_contracts(data, args.out)
    print(f"Wrote {args.out}")
    for key, value in sorted(counts.items()):
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
