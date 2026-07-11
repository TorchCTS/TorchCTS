# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software.

"""Lossless compact storage and references for TorchCTS result artifacts."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path, PurePosixPath
import string
import tempfile
from typing import Any

from torchcts.core.result_sanitization import sanitize_result_payload


COMPACT_RESULT_FORMAT = "torchcts_compact_results"
COMPACT_RESULT_FORMAT_VERSION = 1
RESULT_REFERENCE_FORMAT = "torchcts_result_reference"
RESULT_REFERENCE_FORMAT_VERSION = 1

_TOKEN_ALPHABET = string.digits + string.ascii_lowercase + string.ascii_uppercase
_STRING_REFERENCE_KEY = "$"
_LITERAL_KEY_ESCAPE = "~"
_MAX_REFERENCE_DEPTH = 8


class ResultArtifactError(ValueError):
    """Raised when a result artifact is malformed or unsafe to resolve."""


def _compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_size(value: Any) -> int:
    return len(_compact_json(value).encode("utf-8"))


def _token(index: int) -> str:
    if index < 0:
        raise ValueError("token index cannot be negative")
    base = len(_TOKEN_ALPHABET)
    encoded = ""
    while True:
        index, remainder = divmod(index, base)
        encoded = _TOKEN_ALPHABET[remainder] + encoded
        if index == 0:
            return encoded


def _collect_frequencies(value: Any, keys: Counter[str], strings: Counter[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ResultArtifactError(f"result object key is not a string: {key!r}")
            keys[key] += 1
            _collect_frequencies(item, keys, strings)
    elif isinstance(value, list):
        for item in value:
            _collect_frequencies(item, keys, strings)
    elif isinstance(value, str):
        strings[value] += 1


def _build_key_table(frequencies: Counter[str]) -> list[str]:
    ranked = sorted(
        frequencies,
        key=lambda key: (
            -(frequencies[key] * _json_size(key)),
            -frequencies[key],
            key,
        ),
    )
    table: list[str] = []
    for key in ranked:
        token = _token(len(table))
        original_size = _json_size(key)
        token_size = _json_size(token)
        table_cost = original_size + 1
        if frequencies[key] * (original_size - token_size) > table_cost:
            table.append(key)
    return table


def _build_string_table(frequencies: Counter[str]) -> list[str]:
    ranked = sorted(
        frequencies,
        key=lambda value: (
            -(frequencies[value] * _json_size(value)),
            -frequencies[value],
            value,
        ),
    )
    table: list[str] = []
    for value in ranked:
        index = len(table)
        original_size = _json_size(value)
        reference_size = _json_size({_STRING_REFERENCE_KEY: index})
        table_cost = original_size + 1
        if frequencies[value] * original_size > (
            frequencies[value] * reference_size + table_cost
        ):
            table.append(value)
    return table


def encode_compact_result(payload: dict) -> dict:
    """Encode an expanded result payload into a self-describing compact object."""

    if not isinstance(payload, dict):
        raise ResultArtifactError("result payload must be a JSON object")
    key_frequencies: Counter[str] = Counter()
    string_frequencies: Counter[str] = Counter()
    _collect_frequencies(payload, key_frequencies, string_frequencies)
    key_table = _build_key_table(key_frequencies)
    string_table = _build_string_table(string_frequencies)
    key_tokens = {key: _token(index) for index, key in enumerate(key_table)}
    reserved_tokens = set(key_tokens.values())
    string_indexes = {value: index for index, value in enumerate(string_table)}

    def encode(value: Any) -> Any:
        if isinstance(value, dict):
            encoded: dict[str, Any] = {}
            for key, item in value.items():
                if key in key_tokens:
                    encoded_key = key_tokens[key]
                elif (
                    key in reserved_tokens
                    or key == _STRING_REFERENCE_KEY
                    or key.startswith(_LITERAL_KEY_ESCAPE)
                ):
                    encoded_key = _LITERAL_KEY_ESCAPE + key
                else:
                    encoded_key = key
                encoded[encoded_key] = encode(item)
            return encoded
        if isinstance(value, list):
            return [encode(item) for item in value]
        if isinstance(value, str) and value in string_indexes:
            return {_STRING_REFERENCE_KEY: string_indexes[value]}
        return value

    return {
        "format": COMPACT_RESULT_FORMAT,
        "format_version": COMPACT_RESULT_FORMAT_VERSION,
        "key_table": key_table,
        "string_table": string_table,
        "payload": encode(payload),
    }


def decode_compact_result(encoded: dict) -> dict:
    """Expand a self-describing compact result object losslessly."""

    if encoded.get("format") != COMPACT_RESULT_FORMAT:
        raise ResultArtifactError(f"unknown compact result format: {encoded.get('format')!r}")
    if encoded.get("format_version") != COMPACT_RESULT_FORMAT_VERSION:
        raise ResultArtifactError(
            f"unsupported compact result format version: {encoded.get('format_version')!r}"
        )
    key_table = encoded.get("key_table")
    string_table = encoded.get("string_table")
    if not isinstance(key_table, list) or not all(
        isinstance(key, str) for key in key_table
    ):
        raise ResultArtifactError("compact result key_table must contain strings")
    if len(set(key_table)) != len(key_table):
        raise ResultArtifactError("compact result key_table contains duplicates")
    if not isinstance(string_table, list) or not all(
        isinstance(value, str) for value in string_table
    ):
        raise ResultArtifactError("compact result string_table must contain strings")
    token_keys = {_token(index): key for index, key in enumerate(key_table)}

    def decode(value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) == {_STRING_REFERENCE_KEY}:
                index = value[_STRING_REFERENCE_KEY]
                if not isinstance(index, int) or isinstance(index, bool):
                    raise ResultArtifactError(
                        "compact result string reference must be an integer"
                    )
                if index < 0 or index >= len(string_table):
                    raise ResultArtifactError(
                        f"compact result string reference is out of range: {index}"
                    )
                return string_table[index]
            decoded: dict[str, Any] = {}
            for encoded_key, item in value.items():
                if encoded_key.startswith(_LITERAL_KEY_ESCAPE):
                    key = encoded_key[1:]
                else:
                    key = token_keys.get(encoded_key, encoded_key)
                if key in decoded:
                    raise ResultArtifactError(f"compact result decodes duplicate key {key!r}")
                decoded[key] = decode(item)
            return decoded
        if isinstance(value, list):
            return [decode(item) for item in value]
        return value

    payload = decode(encoded.get("payload"))
    if not isinstance(payload, dict):
        raise ResultArtifactError("compact result payload must decode to an object")
    return payload


def dumps_result_artifact(payload: dict, *, optimize_tables: bool = True) -> str:
    """Return deterministic JSON for an expanded result payload.

    Completed artifacts use table optimization. Frequently rewritten partial
    artifacts can disable it to avoid rebuilding frequency tables per test.
    """

    sanitized = sanitize_result_payload(payload)
    expanded_text = _compact_json(sanitized)
    if not optimize_tables:
        return expanded_text + "\n"
    encoded_text = _compact_json(encode_compact_result(sanitized))
    stored_text = min(
        expanded_text,
        encoded_text,
        key=lambda text: len(text.encode("utf-8")),
    )
    return stored_text + "\n"


def loads_result_artifact(text: str) -> dict:
    """Load an expanded legacy artifact or compact artifact from JSON text."""

    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ResultArtifactError(f"invalid result JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ResultArtifactError("result artifact must contain a JSON object")
    if value.get("format") == COMPACT_RESULT_FORMAT:
        return decode_compact_result(value)
    if value.get("format") == RESULT_REFERENCE_FORMAT:
        raise ResultArtifactError("result references require a filesystem path to resolve")
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(
                path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_result_artifact(
    path: str | Path,
    payload: dict,
    *,
    optimize_tables: bool = True,
) -> None:
    """Atomically write an expanded result payload in compact format."""

    _atomic_write_text(
        Path(path),
        dumps_result_artifact(payload, optimize_tables=optimize_tables),
    )


def _reference_target(reference_path: Path, value: dict) -> Path:
    if value.get("format_version") != RESULT_REFERENCE_FORMAT_VERSION:
        raise ResultArtifactError(
            f"unsupported result reference version: {value.get('format_version')!r}"
        )
    artifact = value.get("artifact")
    if not isinstance(artifact, str) or not artifact:
        raise ResultArtifactError("result reference artifact must be a non-empty relative path")
    pure = PurePosixPath(artifact)
    if pure.is_absolute() or ".." in pure.parts:
        raise ResultArtifactError(f"result reference escapes its directory: {artifact!r}")
    return reference_path.parent.joinpath(*pure.parts)


def resolve_result_artifact_path(path: str | Path) -> Path:
    """Resolve a latest-reference chain to its canonical result artifact."""

    current = Path(path)
    seen: set[Path] = set()
    for _ in range(_MAX_REFERENCE_DEPTH + 1):
        resolved = current.resolve()
        if resolved in seen:
            raise ResultArtifactError(f"cyclic result reference at {current}")
        seen.add(resolved)
        try:
            value = json.loads(current.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ResultArtifactError(f"invalid result JSON in {current}: {exc}") from exc
        if not isinstance(value, dict) or value.get("format") != RESULT_REFERENCE_FORMAT:
            return current
        current = _reference_target(current, value)
    raise ResultArtifactError(f"result reference depth exceeds {_MAX_REFERENCE_DEPTH}")


def load_result_artifact(path: str | Path) -> dict:
    """Load legacy, compact, or referenced result data into the expanded model."""

    resolved = resolve_result_artifact_path(path)
    return loads_result_artifact(resolved.read_text(encoding="utf-8"))


def write_result_reference(path: str | Path, artifact: str | Path) -> None:
    """Atomically point a latest artifact at one canonical completed run."""

    reference_path = Path(path)
    artifact_path = Path(artifact)
    relative = os.path.relpath(artifact_path, start=reference_path.parent)
    pure = PurePosixPath(Path(relative).as_posix())
    if pure.is_absolute() or ".." in pure.parts:
        raise ResultArtifactError("canonical result must be below the latest reference directory")
    payload = {
        "format": RESULT_REFERENCE_FORMAT,
        "format_version": RESULT_REFERENCE_FORMAT_VERSION,
        "artifact": pure.as_posix(),
    }
    _atomic_write_text(reference_path, _compact_json(payload) + "\n")
