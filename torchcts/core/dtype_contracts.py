# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
from importlib import resources
from pathlib import Path
from typing import Any

import torch

from torchcts.core.pytorch_compat import (
    is_runtime_version_validated,
    normalize_torch_version,
    unvalidated_version_message,
)
from torchcts.core.version_rules import parse_torch_version
from torchcts.op_metadata import get_op_metadata


CONTRACT_RESOURCE = "op_dtype_contracts.json"

CPU_SUPPORTED = "cpu_supported"
CPU_UNSUPPORTED = "cpu_unsupported"
CPU_UNKNOWN = "cpu_unknown"
CPU_PENDING = "cpu_pending"
ORACLE_SUPPORTED = "oracle_supported"
NOT_RECORDED = "not_recorded"

SOURCE_EXPECTED_BUT_CPU_UNSUPPORTED = "source_expected_but_cpu_unsupported"
CPU_SUPPORTED_BUT_MISSING_FROM_SOURCE = "cpu_supported_but_missing_from_source"
SOURCE_DECLARED_BUT_PROBE_UNKNOWN = "source_declared_but_probe_unknown"
COMPACT_FORMAT = "runtime_profile_ranges"

_ALIASES = {
    "f16": "torch.float16",
    "f32": "torch.float32",
    "f64": "torch.float64",
    "bf16": "torch.bfloat16",
    "c32": "torch.complex32",
    "c64": "torch.complex64",
    "c128": "torch.complex128",
    "i8": "torch.int8",
    "i16": "torch.int16",
    "i32": "torch.int32",
    "i64": "torch.int64",
    "u8": "torch.uint8",
    "u16": "torch.uint16",
    "u32": "torch.uint32",
    "u64": "torch.uint64",
    "bool": "torch.bool",
}


@dataclass(frozen=True)
class ContractDisposition:
    allowed: bool
    status: str
    skip_reason: str | None = None
    detail: str = ""
    source_expected: tuple[str, ...] = ()
    mismatches: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)


def dtype_to_contract_str(dtype: torch.dtype | str | None) -> str | None:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return str(dtype)
    text = str(dtype)
    if text in _ALIASES:
        return _ALIASES[text]
    if text.startswith("torch."):
        return text
    candidate = f"torch.{text}"
    if hasattr(torch, text):
        return candidate
    return text


def dtype_from_contract_str(dtype: torch.dtype | str | None) -> torch.dtype | None:
    text = dtype_to_contract_str(dtype)
    if text is None or not text.startswith("torch."):
        return dtype if isinstance(dtype, torch.dtype) else None
    name = text.removeprefix("torch.")
    value = getattr(torch, name, None)
    return value if isinstance(value, torch.dtype) else None


def _normalize_dtype_sequence(values: Any) -> tuple[str, ...]:
    if not values:
        return ()
    if isinstance(values, str):
        values = (values,)
    normalized = []
    for value in values:
        dtype_str = dtype_to_contract_str(value)
        if dtype_str is not None:
            normalized.append(dtype_str)
    return tuple(sorted(set(normalized)))


def _normalize_op_name(op_name: str | None) -> str | None:
    if not op_name:
        return None
    text = str(op_name)
    if text.startswith("aten::"):
        return text
    return f"aten::{text}"


@lru_cache(maxsize=1)
def load_dtype_contracts() -> dict:
    try:
        try:
            text = resources.files("torchcts").joinpath(CONTRACT_RESOURCE).read_text(encoding="utf-8")
        except FileNotFoundError:
            text = Path(__file__).resolve().parents[1].joinpath(CONTRACT_RESOURCE).read_text(encoding="utf-8")
        data = json.loads(text)
    except FileNotFoundError:
        data = {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        return {}
    if data.get("version") != 2 or data.get("format") != COMPACT_FORMAT:
        return {}
    data.setdefault("contracts", {})
    data.setdefault("profiles", {})
    data.setdefault("metadata", {})
    return data


def _op_contract_ranges(dispatcher_name: str | None) -> list:
    name = _normalize_op_name(dispatcher_name)
    if name is None:
        return []
    contracts = load_dtype_contracts().get("contracts", {})
    entry = contracts.get(name, {})
    return entry if isinstance(entry, list) else []


def _merge_condition_map(target: dict[str, set[str]], source: Any) -> None:
    if not isinstance(source, dict):
        return
    for condition, dtypes in source.items():
        target.setdefault(str(condition), set()).update(_normalize_dtype_sequence(dtypes))


def _empty_merged_contract() -> dict:
    return {
        "cpu_supported": {},
        "cpu_unsupported": {},
        "cpu_unknown": {},
        "cpu_pending": {},
        "oracle_supported": {},
        "source_expected": {},
        "evidence": [],
        "runtime_status": {},
    }


def _version_in_inclusive_range(runtime_version: str, start: str, end: str) -> bool:
    runtime = parse_torch_version(runtime_version)
    minimum = parse_torch_version(start)
    maximum = parse_torch_version(end)
    if runtime is None or minimum is None or maximum is None:
        return False
    return minimum <= runtime <= maximum


def _merged_contract(dispatcher_name: str | None, runtime_version: str | None = None) -> dict:
    merged = _empty_merged_contract()
    data = load_dtype_contracts()
    metadata = data.get("metadata") or {}
    normalized_version = normalize_torch_version(runtime_version)
    status = {
        "runtime_version": str(runtime_version or torch.__version__),
        "normalized_runtime_version": normalized_version,
        "validated": is_runtime_version_validated(metadata, runtime_version),
    }
    merged["runtime_status"] = status
    if not status["validated"] or normalized_version is None:
        return merged

    profiles = data.get("profiles") or {}
    for range_record in _op_contract_ranges(dispatcher_name):
        if not isinstance(range_record, list) or len(range_record) != 3:
            continue
        start, end, profile_id = (str(range_record[0]), str(range_record[1]), str(range_record[2]))
        if not _version_in_inclusive_range(normalized_version, start, end):
            continue
        profile = profiles.get(profile_id)
        if not isinstance(profile, dict):
            continue
        for key in ("cpu_supported", "cpu_unsupported", "cpu_unknown", "cpu_pending", "oracle_supported", "source_expected"):
            _merge_condition_map(merged[key], profile.get(key))
        merged["evidence"].append({
            "contract_authority": metadata.get("contract_authority"),
            "profile_id": profile_id,
            "range": [start, end],
            "runtime_version": normalized_version,
        })
    return merged


def source_expected_dtypes(dispatcher_name: str | None, *, opinfo_dtypes: Any = None) -> tuple[str, ...]:
    values = set(_normalize_dtype_sequence(opinfo_dtypes))
    name = _normalize_op_name(dispatcher_name)
    if name is not None:
        contract = _merged_contract(name)
        contract_values = set()
        for dtypes in contract.get("source_expected", {}).values():
            contract_values.update(dtypes)
        if contract_values:
            values.update(contract_values)
        else:
            metadata = get_op_metadata(name)
            values.update(_normalize_dtype_sequence(metadata.get("pytorch_dtypes")))
    return tuple(sorted(values))


def phase_condition_key(input_condition: str = "clean", phase: str = "forward") -> str:
    return f"{phase}:{input_condition}"


def _contains_dtype(condition_map: dict[str, set[str]], dtype_str: str, input_condition: str, phase: str) -> bool:
    keys = (
        phase_condition_key(input_condition, phase),
        input_condition,
        f"{phase}:*",
        "*",
        phase_condition_key("clean", phase),
        "clean",
    )
    return any(dtype_str in condition_map.get(key, set()) for key in keys)


def contract_disposition(
    dispatcher_name: str | None,
    dtype: torch.dtype | str,
    *,
    input_condition: str = "clean",
    phase: str = "forward",
    opinfo_dtypes: Any = None,
) -> ContractDisposition:
    dtype_str = dtype_to_contract_str(dtype)
    source_expected = source_expected_dtypes(dispatcher_name, opinfo_dtypes=opinfo_dtypes)
    if dtype_str is None:
        return ContractDisposition(False, CPU_UNKNOWN, "cpu_contract_unknown", "dtype could not be normalized")

    contract = _merged_contract(dispatcher_name)
    evidence = {
        "source_expected": list(source_expected),
        "runtime_status": contract.get("runtime_status") or {},
        "contract_profiles": list(contract.get("evidence") or ()),
    }

    if _contains_dtype(contract["oracle_supported"], dtype_str, input_condition, phase):
        return ContractDisposition(True, ORACLE_SUPPORTED, source_expected=source_expected, evidence=evidence)
    if _contains_dtype(contract["cpu_supported"], dtype_str, input_condition, phase):
        mismatches = () if dtype_str in source_expected else (CPU_SUPPORTED_BUT_MISSING_FROM_SOURCE,)
        return ContractDisposition(True, CPU_SUPPORTED, source_expected=source_expected, mismatches=mismatches, evidence=evidence)
    if _contains_dtype(contract["cpu_unsupported"], dtype_str, input_condition, phase):
        mismatches = (SOURCE_EXPECTED_BUT_CPU_UNSUPPORTED,) if dtype_str in source_expected else ()
        return ContractDisposition(
            False,
            CPU_UNSUPPORTED,
            "cpu_contract_unsupported",
            f"{dtype_str} is not supported by the PyTorch CPU contract for {dispatcher_name}",
            source_expected=source_expected,
            mismatches=mismatches,
            evidence=evidence,
        )
    if _contains_dtype(contract["cpu_unknown"], dtype_str, input_condition, phase):
        mismatches = (SOURCE_DECLARED_BUT_PROBE_UNKNOWN,) if dtype_str in source_expected else ()
        return ContractDisposition(
            False,
            CPU_UNKNOWN,
            "cpu_contract_unknown",
            f"{dtype_str} has unknown PyTorch CPU contract status for {dispatcher_name}",
            source_expected=source_expected,
            mismatches=mismatches,
            evidence=evidence,
        )
    if _contains_dtype(contract["cpu_pending"], dtype_str, input_condition, phase):
        return ContractDisposition(
            False,
            CPU_PENDING,
            "cpu_contract_pending",
            f"{dtype_str} has pending PyTorch CPU contract probe evidence for {dispatcher_name}",
            source_expected=source_expected,
            evidence=evidence,
        )

    runtime_status = contract.get("runtime_status") or {}
    if runtime_status and not runtime_status.get("validated"):
        return ContractDisposition(
            False,
            NOT_RECORDED,
            "cpu_contract_unknown",
            unvalidated_version_message(load_dtype_contracts().get("metadata") or {}, runtime_status.get("runtime_version")),
            source_expected=source_expected,
            evidence=evidence,
        )

    return ContractDisposition(
        False,
        NOT_RECORDED,
        "cpu_contract_unknown",
        f"{dtype_str} has no recorded PyTorch CPU contract for {dispatcher_name}",
        source_expected=source_expected,
        evidence=evidence,
    )


def is_deterministic_cpu_unsupported(exc: BaseException) -> bool:
    message = str(exc).lower()
    fragments = (
        "not implemented for",
        "not implemented",
        "could not run",
        "unsupported dtype",
        "unsupported datatype",
        "unsupported scalar type",
        "not supported for",
        "not currently supported",
        "is not supported",
        "not intended to support",
        "does not support",
        "only supports",
        "only support",
        "expected scalar type",
        "expected a floating point or complex tensor",
        "following scalar types",
        "must be either float or double dtype",
        "must be a floating point",
        "requires a floating point",
        "requires a floating point or complex",
        "dtype must be a floating point",
        "expects floating point dtype",
        "expects floating point dtypes",
        "expected input to have floating point",
        "input dtype should be either floating point or complex",
        "input dtype must be either a floating point or complex dtype",
        "could not infer output dtype",
        "low precision dtypes not supported",
        "integer division with addcdiv is no longer supported",
        "handles only",
        "expected out tensor to have dtype",
        "expected both inputs to be half, float or double",
        "expected both inputs to be half, float or double tensors",
        "expected a tensor with 2 or more dimensions of float, double, cfloat or cdouble types",
        "can't be cast to the desired output type",
        "cannot be cast to the desired output type",
        "operation's result requires dtype",
        "normal expects standard deviation to be non-complex",
        "input tensor may not be a boolean tensor",
        "not available for",
        "requires a floating point or complex tensor",
        "iscomplextype(typemetatoscalartype(dtype())) internal assert failed",
        "unrecognized scalartype: complexha",
    )
    return isinstance(exc, (NotImplementedError, RuntimeError, TypeError)) and any(
        fragment in message for fragment in fragments
    )


def disposition_from_cpu_probe(
    dispatcher_name: str | None,
    dtype: torch.dtype | str,
    *,
    supported: bool | None,
    detail: str = "",
    input_condition: str = "clean",
    phase: str = "forward",
    opinfo_dtypes: Any = None,
) -> ContractDisposition:
    dtype_str = dtype_to_contract_str(dtype)
    source_expected = source_expected_dtypes(dispatcher_name, opinfo_dtypes=opinfo_dtypes)
    evidence = {"source_expected": list(source_expected)}
    if dtype_str is None:
        return ContractDisposition(False, CPU_UNKNOWN, "cpu_contract_unknown", detail, source_expected=source_expected, evidence=evidence)
    if supported is True:
        mismatches = () if dtype_str in source_expected else (CPU_SUPPORTED_BUT_MISSING_FROM_SOURCE,)
        return ContractDisposition(True, CPU_SUPPORTED, source_expected=source_expected, mismatches=mismatches, evidence=evidence)
    if supported is False:
        mismatches = (SOURCE_EXPECTED_BUT_CPU_UNSUPPORTED,) if dtype_str in source_expected else ()
        return ContractDisposition(
            False,
            CPU_UNSUPPORTED,
            "cpu_contract_unsupported",
            detail or f"{dtype_str} is not supported by the PyTorch CPU contract for {dispatcher_name}",
            source_expected=source_expected,
            mismatches=mismatches,
            evidence=evidence,
        )
    mismatches = (SOURCE_DECLARED_BUT_PROBE_UNKNOWN,) if dtype_str in source_expected else ()
    return ContractDisposition(
        False,
        CPU_UNKNOWN,
        "cpu_contract_unknown",
        detail or f"{dtype_str} has unknown PyTorch CPU contract status for {dispatcher_name}",
        source_expected=source_expected,
        mismatches=mismatches,
        evidence=evidence,
    )


def mismatch_counts() -> dict[str, int]:
    return {
        SOURCE_EXPECTED_BUT_CPU_UNSUPPORTED: 0,
        CPU_SUPPORTED_BUT_MISSING_FROM_SOURCE: 0,
        SOURCE_DECLARED_BUT_PROBE_UNKNOWN: 0,
    }


__all__ = [
    "CPU_SUPPORTED",
    "CPU_SUPPORTED_BUT_MISSING_FROM_SOURCE",
    "CPU_PENDING",
    "CPU_UNKNOWN",
    "CPU_UNSUPPORTED",
    "ContractDisposition",
    "NOT_RECORDED",
    "ORACLE_SUPPORTED",
    "SOURCE_DECLARED_BUT_PROBE_UNKNOWN",
    "SOURCE_EXPECTED_BUT_CPU_UNSUPPORTED",
    "contract_disposition",
    "disposition_from_cpu_probe",
    "dtype_from_contract_str",
    "dtype_to_contract_str",
    "is_deterministic_cpu_unsupported",
    "load_dtype_contracts",
    "mismatch_counts",
    "phase_condition_key",
    "source_expected_dtypes",
]
