# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software.

"""Normalize structural result paths while preserving diagnostic text."""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


_POSIX_PATH_RE = re.compile(r"(?<![A-Za-z0-9_.:/-])/(?!/)[^\s\"'<>|`]+")
_WINDOWS_PATH_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(?:"
    r"[A-Z]:[\\/][^\s\"'<>|`]+"
    r"|\\\\[^\\/\s\"'<>|`]+[\\/][^\s\"'<>|`]+"
    r")"
)
_FILE_URI_RE = re.compile(r"(?i)file://(?P<path>/(?!/)[^\s\"'<>|`]+|[A-Z]:[\\/][^\s\"'<>|`]+)")
_PUBLIC_URI_RE = re.compile(r"(?i)\b(?!file:)[a-z][a-z0-9+.-]*://[^\s\"'<>|`]+")
_LINE_SUFFIX_RE = re.compile(r"^(?P<path>.*?)(?P<suffix>:\d+(?::\d+)?)$")
_TRAILING_PUNCTUATION = ",.;)]}"

_STRUCTURAL_RESULT_FIELDS = frozenset(
    {
        "command",
        "command_args",
        "command_display",
        "command_string",
        "copied_inputs",
        "cwd",
        "executable",
        "file",
        "files",
        "input_result",
        "input_snapshot",
        "nodeid",
        "python_executable",
        "repro_scripts",
        "rootdir",
    }
)
_STRUCTURAL_RESULT_FIELD_SUFFIXES = (
    "_command",
    "_command_args",
    "_file",
    "_files",
    "_nodeid",
    "_path",
    "_paths",
)
_STRUCTURAL_KEY_CONTAINERS = frozenset({"results"})


def _default_roots() -> tuple[Path, ...]:
    candidates = [Path.cwd(), Path(__file__).resolve().parents[2]]
    roots: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved not in roots:
            roots.append(resolved)
    return tuple(roots)


def _split_token(token: str) -> tuple[str, str, str]:
    trailing = ""
    while token and token[-1] in _TRAILING_PUNCTUATION:
        trailing = token[-1] + trailing
        token = token[:-1]
    suffix = ""
    match = _LINE_SUFFIX_RE.match(token)
    if match:
        token = match.group("path")
        suffix = match.group("suffix")
    return token, suffix, trailing


def _relative_to_roots(path: str, roots: Iterable[Path]) -> str | None:
    candidate = Path(path)
    for root in roots:
        try:
            relative = candidate.relative_to(root)
        except (OSError, ValueError):
            continue
        if relative.parts:
            return relative.as_posix()
    return None


def normalize_absolute_path(value: str, *, roots: Iterable[Path] | None = None) -> str:
    """Return a stable relative presentation for one absolute filesystem path."""

    token, suffix, trailing = _split_token(str(value).replace("\\", "/"))
    path = token
    root_values = tuple(roots) if roots is not None else _default_roots()

    parts = PurePosixPath(path).parts
    basename = PurePosixPath(path).name or "path"
    parent = PurePosixPath(path).parent.name.lower()
    if parent in {"bin", "scripts"} and re.fullmatch(r"python(?:w)?(?:\d+(?:\.\d+)*)?(?:\.exe)?", basename, re.I):
        normalized = "python"
    elif "/site-packages/" in path:
        normalized = path.rsplit("/site-packages/", 1)[1]
    elif "/dist-packages/" in path:
        normalized = path.rsplit("/dist-packages/", 1)[1]
    elif "/pytorch/" in path:
        normalized = "pytorch/" + path.rsplit("/pytorch/", 1)[1]
    else:
        normalized = _relative_to_roots(path, root_values)
        if normalized is None:
            torchcts_indexes = [index for index, part in enumerate(parts) if part == "torchcts"]
            if torchcts_indexes:
                normalized = "/".join(parts[torchcts_indexes[-1] :])
            elif path.startswith(tempfile.gettempdir().replace("\\", "/") + "/"):
                normalized = f"temp/{basename}"
            elif path in {"/dev/null", "NUL"}:
                normalized = "device/null"
            else:
                normalized = f"external/{basename}"
    return normalized.lstrip("/") + suffix + trailing


def sanitize_text(value: str, *, roots: Iterable[Path] | None = None) -> str:
    """Replace absolute POSIX, Windows, UNC, and file-URI paths in text."""

    text = str(value)
    root_values = tuple(roots) if roots is not None else _default_roots()
    public_uris: list[str] = []

    def preserve_public_uri(match: re.Match[str]) -> str:
        placeholder = f"RESULTURIPRESERVE{len(public_uris)}TOKEN"
        public_uris.append(match.group(0))
        return placeholder

    def replace_uri(match: re.Match[str]) -> str:
        return normalize_absolute_path(match.group("path"), roots=root_values)

    def replace_path(match: re.Match[str]) -> str:
        return normalize_absolute_path(match.group(0), roots=root_values)

    text = _PUBLIC_URI_RE.sub(preserve_public_uri, text)
    text = _FILE_URI_RE.sub(replace_uri, text)
    text = _WINDOWS_PATH_RE.sub(replace_path, text)
    text = _POSIX_PATH_RE.sub(replace_path, text)
    for index, uri in enumerate(public_uris):
        text = text.replace(f"RESULTURIPRESERVE{index}TOKEN", uri)
    return text


def is_structural_result_field(key: str | None) -> bool:
    """Return whether a result field represents path or command structure."""

    if key is None:
        return False
    normalized = str(key).lower()
    return normalized in _STRUCTURAL_RESULT_FIELDS or normalized.endswith(
        _STRUCTURAL_RESULT_FIELD_SUFFIXES
    )


def sanitize_result_payload(
    value: Any,
    *,
    roots: Iterable[Path] | None = None,
    _field: str | None = None,
    _structural: bool = False,
) -> Any:
    """Normalize structural paths without altering free-form diagnostic text."""

    root_values = tuple(roots) if roots is not None else _default_roots()
    structural = _structural or is_structural_result_field(_field)
    if isinstance(value, dict):
        sanitized: dict[Any, Any] = {}
        sanitize_keys = _field in _STRUCTURAL_KEY_CONTAINERS
        for key, item in value.items():
            sanitized_key = (
                sanitize_text(key, roots=root_values)
                if sanitize_keys and isinstance(key, str)
                else key
            )
            if sanitized_key in sanitized:
                raise ValueError(f"Result path sanitization produced duplicate key {sanitized_key!r}")
            sanitized[sanitized_key] = sanitize_result_payload(
                item,
                roots=root_values,
                _field=str(key),
                _structural=structural,
            )
        return sanitized
    if isinstance(value, list):
        return [
            sanitize_result_payload(
                item,
                roots=root_values,
                _field=_field,
                _structural=structural,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            sanitize_result_payload(
                item,
                roots=root_values,
                _field=_field,
                _structural=structural,
            )
            for item in value
        )
    if isinstance(value, Path):
        return sanitize_text(str(value), roots=root_values)
    if isinstance(value, str) and structural:
        return sanitize_text(value, roots=root_values)
    return value


def structural_absolute_paths(
    value: Any,
    *,
    _field: str | None = None,
    _structural: bool = False,
) -> list[str]:
    """Return absolute paths found in structural fields, excluding diagnostics."""

    structural = _structural or is_structural_result_field(_field)
    matches: list[str] = []
    if isinstance(value, dict):
        inspect_keys = _field in _STRUCTURAL_KEY_CONTAINERS
        for key, item in value.items():
            if inspect_keys and isinstance(key, str):
                matches.extend(absolute_paths_in_text(key))
            matches.extend(
                structural_absolute_paths(
                    item,
                    _field=str(key),
                    _structural=structural,
                )
            )
    elif isinstance(value, (list, tuple)):
        for item in value:
            matches.extend(
                structural_absolute_paths(
                    item,
                    _field=_field,
                    _structural=structural,
                )
            )
    elif isinstance(value, Path):
        matches.extend(absolute_paths_in_text(str(value)))
    elif isinstance(value, str) and structural:
        matches.extend(absolute_paths_in_text(value))
    return matches


def absolute_paths_in_text(value: str) -> list[str]:
    """Return absolute path tokens still present in a serialized artifact."""

    text = str(value)
    text = _PUBLIC_URI_RE.sub("PUBLICURIPRESERVED", text)
    matches = [match.group(0) for match in _FILE_URI_RE.finditer(text)]
    matches.extend(match.group(0) for match in _WINDOWS_PATH_RE.finditer(text))
    matches.extend(match.group(0) for match in _POSIX_PATH_RE.finditer(text))
    return matches
