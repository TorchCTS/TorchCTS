# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
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
import json
from pathlib import Path
from typing import Iterable

from torchcts.core.version_rules import parse_torch_version, version_in_range


PACKAGED_KNOWN_SEGFAULTS = Path(__file__).resolve().parents[1] / "known_segfaults.json"
PROJECT_KNOWN_SEGFAULTS = "known_segfaults.json"

VALID_SIGNALS = {"SIGSEGV", "SIGABRT", "SIGBUS", "SIGILL"}
VALID_MATCHES = {"nodeid"}
VALID_CLASSIFICATIONS = {"confirmed_mps_crash"}

REQUIRED_ENTRY_KEYS = {
    "id",
    "backend",
    "match",
    "nodeid",
    "dispatcher",
    "classification",
    "expected_signal",
    "repro",
    "reason",
    "owner",
    "pytorch_min",
    "pytorch_max",
    "hardware",
    "review_after",
}

ALLOWED_ENTRY_KEYS = set(REQUIRED_ENTRY_KEYS)
REQUIRED_REPRO_KEYS = {"script", "case"}
ALLOWED_REPRO_KEYS = set(REQUIRED_REPRO_KEYS)


class KnownSegfaultError(ValueError):
    """Raised when a known-segfault ledger is malformed."""


def _parse_date(value: str, path: str, entry_id: str) -> _datetime.date:
    if not isinstance(value, str):
        raise KnownSegfaultError(f"{path}: entry {entry_id} review_after must be YYYY-MM-DD")
    try:
        return _datetime.date.fromisoformat(value)
    except ValueError as exc:
        raise KnownSegfaultError(f"{path}: entry {entry_id} review_after must be YYYY-MM-DD") from exc


def _normalize_payload_item(payload_item) -> tuple[str, dict]:
    if isinstance(payload_item, tuple) and len(payload_item) == 2:
        path, payload = payload_item
        return str(path), payload
    return "<memory>", payload_item


def _validate_entry(entry: dict, path: str, seen_ids: set[str]) -> dict:
    if not isinstance(entry, dict):
        raise KnownSegfaultError(f"{path}: each known_segfaults entry must be an object")

    entry_id = entry.get("id", "<missing>")
    missing = sorted(REQUIRED_ENTRY_KEYS - set(entry))
    if missing:
        raise KnownSegfaultError(f"{path}: entry {entry_id} missing required field(s): {', '.join(missing)}")

    unknown = sorted(set(entry) - ALLOWED_ENTRY_KEYS)
    if unknown:
        raise KnownSegfaultError(f"{path}: entry {entry_id} has unknown field(s): {', '.join(unknown)}")

    if not isinstance(entry["id"], str) or not entry["id"].strip():
        raise KnownSegfaultError(f"{path}: entry id must be a non-empty string")
    if entry["id"] in seen_ids:
        raise KnownSegfaultError(f"{path}: duplicate known segfault id {entry['id']!r}")
    seen_ids.add(entry["id"])

    for key in ("backend", "nodeid", "dispatcher", "reason", "owner", "hardware"):
        if not isinstance(entry[key], str) or not entry[key].strip():
            raise KnownSegfaultError(f"{path}: entry {entry['id']} field {key} must be a non-empty string")

    if entry["match"] not in VALID_MATCHES:
        raise KnownSegfaultError(f"{path}: entry {entry['id']} match must be one of {sorted(VALID_MATCHES)}")
    if entry["classification"] not in VALID_CLASSIFICATIONS:
        raise KnownSegfaultError(
            f"{path}: entry {entry['id']} classification must be confirmed_mps_crash"
        )
    if entry["expected_signal"] not in VALID_SIGNALS:
        raise KnownSegfaultError(f"{path}: entry {entry['id']} expected_signal must be one of {sorted(VALID_SIGNALS)}")

    if entry["pytorch_min"] is not None and parse_torch_version(entry["pytorch_min"]) is None:
        raise KnownSegfaultError(f"{path}: entry {entry['id']} pytorch_min is invalid")
    if entry["pytorch_max"] is not None and parse_torch_version(entry["pytorch_max"]) is None:
        raise KnownSegfaultError(f"{path}: entry {entry['id']} pytorch_max is invalid")

    _parse_date(entry["review_after"], path, entry["id"])

    repro = entry["repro"]
    if not isinstance(repro, dict):
        raise KnownSegfaultError(f"{path}: entry {entry['id']} repro must be an object")
    missing_repro = sorted(REQUIRED_REPRO_KEYS - set(repro))
    if missing_repro:
        raise KnownSegfaultError(
            f"{path}: entry {entry['id']} repro missing required field(s): {', '.join(missing_repro)}"
        )
    unknown_repro = sorted(set(repro) - ALLOWED_REPRO_KEYS)
    if unknown_repro:
        raise KnownSegfaultError(
            f"{path}: entry {entry['id']} repro has unknown field(s): {', '.join(unknown_repro)}"
        )
    for key in REQUIRED_REPRO_KEYS:
        if not isinstance(repro[key], str) or not repro[key].strip():
            raise KnownSegfaultError(f"{path}: entry {entry['id']} repro.{key} must be a non-empty string")

    normalized = dict(entry)
    normalized["source_path"] = path
    return normalized


def validate_known_segfaults(payloads: Iterable[dict | tuple[str | Path, dict]]) -> list[dict]:
    entries: list[dict] = []
    seen_ids: set[str] = set()
    for payload_item in payloads:
        path, payload = _normalize_payload_item(payload_item)
        if not isinstance(payload, dict):
            raise KnownSegfaultError(f"{path}: known-segfault ledger must be a JSON object")
        unknown_top = sorted(set(payload) - {"version", "known_segfaults"})
        if unknown_top:
            raise KnownSegfaultError(f"{path}: unknown top-level field(s): {', '.join(unknown_top)}")
        if payload.get("version") != 1:
            raise KnownSegfaultError(f"{path}: version must be 1")
        raw_entries = payload.get("known_segfaults")
        if not isinstance(raw_entries, list):
            raise KnownSegfaultError(f"{path}: known_segfaults must be a list")
        for entry in raw_entries:
            entries.append(_validate_entry(entry, path, seen_ids))
    return entries


def _read_payload(path: Path) -> tuple[str, dict]:
    try:
        return str(path), json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise KnownSegfaultError(f"{path}: malformed JSON: {exc}") from exc


def load_known_segfaults(project_root: str | Path = ".") -> list[dict]:
    payloads: list[tuple[str, dict]] = []
    if PACKAGED_KNOWN_SEGFAULTS.exists():
        payloads.append(_read_payload(PACKAGED_KNOWN_SEGFAULTS))

    project_path = Path(project_root) / PROJECT_KNOWN_SEGFAULTS
    if project_path.exists():
        payloads.append(_read_payload(project_path))

    return validate_known_segfaults(payloads)


def active_known_segfaults(
    entries: Iterable[dict],
    *,
    backend: str,
    torch_version: str,
    hardware_key: str,
) -> list[dict]:
    active: list[dict] = []
    for entry in entries:
        if entry["backend"] != backend:
            continue
        if entry["hardware"] != "any" and entry["hardware"] != hardware_key:
            continue
        if not version_in_range(torch_version, entry["pytorch_min"], entry["pytorch_max"]):
            continue
        active.append(dict(entry))
    return active


def canonicalize_nodeid(nodeid: str) -> str:
    """Return a package-relative node id stable across source and wheel installs."""

    text = str(nodeid).replace("\\", "/")
    path, sep, suffix = text.partition("::")
    while path.startswith("./"):
        path = path[2:]
    marker = "/torchcts/"
    if marker in path:
        path = "torchcts/" + path.rsplit(marker, 1)[1]
    return f"{path}{sep}{suffix}"


def match_known_segfault(item, active_entries: Iterable[dict]) -> dict | None:
    nodeid = canonicalize_nodeid(getattr(item, "nodeid", str(item)))
    for entry in active_entries:
        if entry.get("match") == "nodeid" and canonicalize_nodeid(entry.get("nodeid", "")) == nodeid:
            return dict(entry)
    return None


def expired_known_segfault_warnings(
    entries: Iterable[dict],
    *,
    today: _datetime.date | None = None,
) -> list[str]:
    today = today or _datetime.date.today()
    warnings: list[str] = []
    for entry in entries:
        review_after = _parse_date(entry["review_after"], entry.get("source_path", "<memory>"), entry["id"])
        if review_after < today:
            warnings.append(
                f"known segfault {entry['id']} review_after={entry['review_after']} has expired"
            )
    return warnings
