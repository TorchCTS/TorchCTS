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

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any

_VERSION_PREFIX_RE = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?")
_NON_VERSION_EXACT_SORT_PARTS = (10**9, 10**9, 10**9)


@dataclass(frozen=True)
class VersionRuleKey:
    parts: tuple[int, int, int]
    specificity: int
    exact_only: bool
    text: str


def parse_torch_version(version: str | None) -> tuple[int, int, int] | None:
    if version is None:
        return None
    text = str(version)
    match = _VERSION_PREFIX_RE.match(text)
    if match is None:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch) if patch is not None else 0


def version_in_range(torch_version: str, pytorch_min: str | None, pytorch_max: str | None) -> bool:
    current = parse_torch_version(torch_version)
    if current is None:
        return False
    min_version = parse_torch_version(pytorch_min)
    max_version = parse_torch_version(pytorch_max)
    if min_version is not None and current < min_version:
        return False
    if max_version is not None and current > max_version:
        return False
    return True


def parse_version_rule_key(version: str | None) -> VersionRuleKey | None:
    """Parse a version-policy key.

    Bare numeric keys are cumulative:
      - 2.12 applies to every 2.12 patch and future versions until removed.
      - 2.12.1 applies to 2.12.1 and future versions until removed.

    Keys with a suffix or local build tag are exact-only policy overlays:
      - 2.12.1+cpu applies only to runtime 2.12.1+cpu.
      - 2.13.0.dev20260628 applies only to that exact runtime string.
    """
    if version is None:
        return None
    text = str(version)
    match = _VERSION_PREFIX_RE.match(text)
    if match is None:
        return None

    major, minor, patch = match.groups()
    suffix = text[match.end():]
    parts = (int(major), int(minor), int(patch) if patch is not None else 0)
    specificity = 2 if patch is None else 3
    return VersionRuleKey(parts=parts, specificity=specificity, exact_only=bool(suffix), text=text)


def iter_version_rule_entries(data: Mapping[str, Any], runtime_version: str) -> tuple[tuple[str, Any], ...]:
    """Return version-policy entries that apply to a runtime, in application order."""

    runtime = parse_version_rule_key(runtime_version)
    runtime_text = str(runtime_version)
    if runtime is None:
        value = data.get(runtime_text)
        return ((runtime_text, value),) if value is not None else ()

    entries = []
    for key, value in data.items():
        parsed = parse_version_rule_key(key)
        if parsed is None:
            if key == runtime_text:
                entries.append((_NON_VERSION_EXACT_SORT_PARTS, 0, key, value))
            continue

        if parsed.exact_only:
            if key == runtime_text:
                entries.append((parsed.parts, parsed.specificity, key, value))
            continue
        if parsed.parts <= runtime.parts:
            entries.append((parsed.parts, parsed.specificity, key, value))

    entries.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple((key, value) for _, _, key, value in entries)


def _item_set(value: Any) -> set:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    return set(value)


def add_remove_items(value: Any, *, default_add_key: str = "add") -> tuple[set, set]:
    if isinstance(value, dict):
        add = value.get("add", value.get(default_add_key, ()))
        remove = value.get("remove", ())
    else:
        add = value
        remove = ()
    return _item_set(add), _item_set(remove)


def cumulative_versioned_set(data: Mapping[str, Any], runtime_version: str, *, default_add_key: str = "ops") -> frozenset:
    values = set()
    for _, value in iter_version_rule_entries(data, runtime_version):
        add, remove = add_remove_items(value, default_add_key=default_add_key)
        values.update(add)
        values.difference_update(remove)
    return frozenset(values)
