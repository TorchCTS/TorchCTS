"""Strict, development-only schema for frozen TorchCTS oracle records.

This module intentionally lives outside the ``torchcts`` package.  Runtime
conformance never imports fixture data, and wheels do not contain it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping


SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Z0-9][A-Z0-9._-]*$")
_CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_VALIDATION_CLASSES = {
    "V1_FIXED_VALUE",
    "V2_ADMISSIBILITY",
    "V3_ROUTING",
    "V4_PROPERTY",
    "V5_BACKEND_SEMANTIC",
    "V6_DISPOSITION",
}
_ORACLE_KINDS = {
    "value_reference",
    "permanent_reference",
    "legality_contract",
    "fft_contract",
    "direct_oracle",
    "test_local_semantic",
}


class FixtureValidationError(ValueError):
    """A frozen fixture, manifest, source, or evidence record is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one byte representation used for every stored digest."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def canonical_json_text(value: Any) -> str:
    return canonical_json_bytes(value).decode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def resolve_under(root: Path, relative_path: str, *, must_exist: bool = True) -> Path:
    """Resolve a fixture-owned path and reject traversal and symlink escapes."""

    if not isinstance(relative_path, str) or not relative_path:
        raise FixtureValidationError("artifact path must be a non-empty string")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise FixtureValidationError(f"artifact path escapes its root: {relative_path!r}")
    root_resolved = root.resolve(strict=True)
    try:
        resolved = (root_resolved / relative).resolve(strict=must_exist)
        resolved.relative_to(root_resolved)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise FixtureValidationError(
            f"artifact path escapes its root or does not exist: {relative_path!r}"
        ) from error
    return resolved


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FixtureValidationError(f"{label} must be an object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise FixtureValidationError(f"{label} must be a non-empty string")
    return value


def _require_string_list(value: Any, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not allow_empty and not value):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise FixtureValidationError(f"{label} must be {qualifier} of strings")
    if any(not isinstance(item, str) or not item for item in value):
        raise FixtureValidationError(f"{label} must contain only non-empty strings")
    if len(value) != len(set(value)):
        raise FixtureValidationError(f"{label} contains duplicate values")
    return tuple(value)


def _verify_artifact(
    artifact: Mapping[str, Any],
    *,
    root: Path,
    label: str,
) -> Path:
    path_text = _require_string(artifact.get("path"), f"{label}.path")
    digest = _require_string(artifact.get("sha256"), f"{label}.sha256")
    if not _SHA256_RE.fullmatch(digest):
        raise FixtureValidationError(f"{label}.sha256 is not a SHA-256 digest")
    path = resolve_under(root, path_text)
    actual = sha256_file(path)
    if actual != digest:
        raise FixtureValidationError(
            f"{label}: digest mismatch for {path_text}: stored {digest}, calculated {actual}"
        )
    return path


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    kind: str
    title: str
    version: str
    revision: str
    url: str
    authority_for: tuple[str, ...]
    not_authority_for: tuple[str, ...]
    raw_evidence: tuple[Mapping[str, Any], ...]

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        evidence_root: Path,
    ) -> "SourceRecord":
        record = _require_mapping(value, "source")
        if record.get("schema_version") != SCHEMA_VERSION:
            raise FixtureValidationError("source: unsupported schema_version")
        source_id = _require_string(record.get("source_id"), "source.source_id")
        if not _ID_RE.fullmatch(source_id):
            raise FixtureValidationError(f"source: invalid source_id {source_id!r}")
        authority = _require_string_list(record.get("authority_for"), "source.authority_for")
        excluded = _require_string_list(
            record.get("not_authority_for"),
            "source.not_authority_for",
            allow_empty=True,
        )
        overlap = sorted(set(authority) & set(excluded))
        if overlap:
            raise FixtureValidationError(
                f"source {source_id}: authority_for and not_authority_for overlap: {overlap}"
            )
        raw = record.get("raw_evidence")
        if not isinstance(raw, list):
            raise FixtureValidationError(f"source {source_id}: raw_evidence must be a list")
        for index, artifact in enumerate(raw):
            _verify_artifact(
                _require_mapping(artifact, f"source {source_id}.raw_evidence[{index}]"),
                root=evidence_root,
                label=f"source {source_id}.raw_evidence[{index}]",
            )
        return cls(
            source_id=source_id,
            kind=_require_string(record.get("kind"), f"source {source_id}.kind"),
            title=_require_string(record.get("title"), f"source {source_id}.title"),
            version=_require_string(record.get("version"), f"source {source_id}.version"),
            revision=_require_string(record.get("revision"), f"source {source_id}.revision"),
            url=_require_string(record.get("url"), f"source {source_id}.url"),
            authority_for=authority,
            not_authority_for=excluded,
            raw_evidence=tuple(raw),
        )


def load_generator_catalog(path: Path, *, repo_root: Path) -> frozenset[str]:
    """Validate each independent generator once and return its stable path."""

    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FixtureValidationError(f"cannot read generator catalog {path}: {error}") from error
    catalog = _require_mapping(payload, "generator catalog")
    if catalog.get("schema_version") != SCHEMA_VERSION:
        raise FixtureValidationError("generator catalog: unsupported schema_version")
    rows = catalog.get("generators")
    if not isinstance(rows, list):
        raise FixtureValidationError("generator catalog: generators must be a list")
    result: set[str] = set()
    for index, row in enumerate(rows):
        artifact = _require_mapping(row, f"generator catalog.generators[{index}]")
        generator_path = _require_string(
            artifact.get("path"), f"generator catalog.generators[{index}].path"
        )
        _verify_artifact(
            artifact,
            root=repo_root,
            label=f"generator catalog.generators[{index}]",
        )
        if generator_path in result:
            raise FixtureValidationError(
                f"generator catalog: duplicate generator {generator_path!r}"
            )
        result.add(generator_path)
    return frozenset(result)


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    oracle_kind: str
    validation_class: str
    case_pack: str
    oracle_ids: tuple[str, ...]
    implementation_entry_points: tuple[str, ...]
    dispatcher_surfaces: tuple[str, ...]
    source_ids: tuple[str, ...]
    comparison_asserts: tuple[str, ...]
    payload: Mapping[str, Any]

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        sources: Mapping[str, SourceRecord],
        generator_paths: frozenset[str],
        repo_root: Path,
        evidence_root: Path,
    ) -> "CaseRecord":
        record = _require_mapping(value, "case")
        if record.get("schema_version") != SCHEMA_VERSION:
            raise FixtureValidationError("case: unsupported schema_version")
        case_id = _require_string(record.get("case_id"), "case.case_id")
        if not _CASE_ID_RE.fullmatch(case_id):
            raise FixtureValidationError(f"case: invalid case_id {case_id!r}")
        kind = _require_string(record.get("oracle_kind"), f"case {case_id}.oracle_kind")
        if kind not in _ORACLE_KINDS:
            raise FixtureValidationError(f"case {case_id}: unknown oracle_kind {kind!r}")
        validation_class = _require_string(
            record.get("validation_class"), f"case {case_id}.validation_class"
        )
        if validation_class not in _VALIDATION_CLASSES:
            raise FixtureValidationError(
                f"case {case_id}: unknown validation_class {validation_class!r}"
            )
        case_pack = _require_string(record.get("case_pack"), f"case {case_id}.case_pack")
        if not case_pack.startswith("CP-"):
            raise FixtureValidationError(f"case {case_id}: case_pack must start with CP-")
        _require_mapping(record.get("applicability"), f"case {case_id}.applicability")
        if "inputs" not in record:
            raise FixtureValidationError(f"case {case_id}: missing inputs")
        outputs = [key for key in ("expected", "admissible_results", "invariants") if key in record]
        if len(outputs) != 1:
            raise FixtureValidationError(
                f"case {case_id}: exactly one of expected, admissible_results, or invariants is required"
            )
        comparison = _require_mapping(record.get("comparison"), f"case {case_id}.comparison")
        asserts = _require_string_list(
            comparison.get("asserts"), f"case {case_id}.comparison.asserts"
        )
        source_ids = _require_string_list(record.get("source_ids"), f"case {case_id}.source_ids")
        missing_sources = sorted(set(source_ids) - set(sources))
        if missing_sources:
            raise FixtureValidationError(f"case {case_id}: unknown source_ids {missing_sources}")
        authorized = set().union(*(sources[source_id].authority_for for source_id in source_ids))
        uncovered = sorted(set(asserts) - authorized)
        if uncovered:
            raise FixtureValidationError(
                f"case {case_id}: comparator assertions lack source authority: {uncovered}"
            )
        generator = _require_string(record.get("generator"), f"case {case_id}.generator")
        if generator not in generator_paths:
            raise FixtureValidationError(
                f"case {case_id}: generator {generator!r} is not in the generator catalog"
            )
        raw = _require_string_list(record.get("raw_evidence"), f"case {case_id}.raw_evidence")
        authorized_artifacts = {
            artifact.get("path")
            for source_id in source_ids
            for artifact in sources[source_id].raw_evidence
        }
        for path_text in raw:
            if path_text not in authorized_artifacts:
                raise FixtureValidationError(
                    f"case {case_id}: raw evidence {path_text!r} is not owned by a named source"
                )
        review = _require_mapping(record.get("review"), f"case {case_id}.review")
        _require_string_list(review.get("reviewed_by"), f"case {case_id}.review.reviewed_by")
        _require_string(review.get("reviewed_at"), f"case {case_id}.review.reviewed_at")
        conclusion = _require_string(review.get("conclusion"), f"case {case_id}.review.conclusion")
        if conclusion not in {
            "implementation_correct",
            "torchcts_bug",
            "source_ambiguity",
            "fixture_bug",
        }:
            raise FixtureValidationError(f"case {case_id}: invalid review conclusion {conclusion!r}")
        return cls(
            case_id=case_id,
            oracle_kind=kind,
            validation_class=validation_class,
            case_pack=case_pack,
            oracle_ids=_require_string_list(record.get("oracle_ids"), f"case {case_id}.oracle_ids"),
            implementation_entry_points=_require_string_list(
                record.get("implementation_entry_points"),
                f"case {case_id}.implementation_entry_points",
            ),
            dispatcher_surfaces=_require_string_list(
                record.get("dispatcher_surfaces"),
                f"case {case_id}.dispatcher_surfaces",
                allow_empty=True,
            ),
            source_ids=source_ids,
            comparison_asserts=asserts,
            payload=record,
        )


def load_source_catalog(path: Path, *, evidence_root: Path) -> dict[str, SourceRecord]:
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FixtureValidationError(f"cannot read source catalog {path}: {error}") from error
    catalog = _require_mapping(payload, "source catalog")
    if catalog.get("schema_version") != SCHEMA_VERSION:
        raise FixtureValidationError("source catalog: unsupported schema_version")
    rows = catalog.get("sources")
    if not isinstance(rows, list):
        raise FixtureValidationError("source catalog: sources must be a list")
    result: dict[str, SourceRecord] = {}
    for row in rows:
        source = SourceRecord.from_mapping(row, evidence_root=evidence_root)
        if source.source_id in result:
            raise FixtureValidationError(f"source catalog: duplicate source_id {source.source_id}")
        result[source.source_id] = source
    return result


def load_case_manifest(
    manifest_path: Path,
    *,
    source_catalog_path: Path,
    repo_root: Path,
    evidence_root: Path,
) -> list[CaseRecord]:
    """Validate the whole corpus before returning any candidate-executable record."""

    sources = load_source_catalog(source_catalog_path, evidence_root=evidence_root)
    generator_paths = load_generator_catalog(
        evidence_root / "generators.json", repo_root=repo_root
    )
    try:
        payload = json.loads(manifest_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FixtureValidationError(f"cannot read case manifest {manifest_path}: {error}") from error
    manifest = _require_mapping(payload, "case manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise FixtureValidationError("case manifest: unsupported schema_version")
    rows = manifest.get("cases")
    if not isinstance(rows, list):
        raise FixtureValidationError("case manifest: cases must be a list")
    cases_root = manifest_path.parent
    result: list[CaseRecord] = []
    paths: set[Path] = set()
    ids: set[str] = set()
    for index, row_value in enumerate(rows):
        row = _require_mapping(row_value, f"case manifest.cases[{index}]")
        path_text = _require_string(row.get("path"), f"case manifest.cases[{index}].path")
        path = resolve_under(cases_root, path_text)
        if path.suffix != ".json":
            raise FixtureValidationError(f"case manifest: case path is not JSON: {path_text}")
        expected_sha = _require_string(row.get("sha256"), f"case manifest.cases[{index}].sha256")
        if not _SHA256_RE.fullmatch(expected_sha) or sha256_file(path) != expected_sha:
            raise FixtureValidationError(f"case manifest: file digest mismatch for {path_text}")
        try:
            case_payload = json.loads(path.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise FixtureValidationError(f"cannot read case {path_text}: {error}") from error
        case = CaseRecord.from_mapping(
            case_payload,
            sources=sources,
            generator_paths=generator_paths,
            repo_root=repo_root,
            evidence_root=evidence_root,
        )
        listed_id = _require_string(row.get("case_id"), f"case manifest.cases[{index}].case_id")
        if case.case_id != listed_id:
            raise FixtureValidationError(
                f"case manifest: {path_text} contains {case.case_id!r}, expected {listed_id!r}"
            )
        if path in paths or case.case_id in ids:
            raise FixtureValidationError(f"case manifest: duplicate case or path for {case.case_id}")
        paths.add(path)
        ids.add(case.case_id)
        result.append(case)
    ignored = {
        manifest_path.resolve(),
        (cases_root / "inventory.json").resolve(),
        (cases_root / "reviewed_local_expected.json").resolve(),
    }
    actual_case_paths = {
        path.resolve()
        for path in cases_root.rglob("*.json")
        if path.resolve() not in ignored
    }
    orphans = sorted(path.relative_to(cases_root).as_posix() for path in actual_case_paths - paths)
    if orphans:
        raise FixtureValidationError(f"case manifest: orphan fixture files: {orphans}")
    return sorted(result, key=lambda case: case.case_id)


def validate_then_run(
    manifest_path: Path,
    *,
    source_catalog_path: Path,
    repo_root: Path,
    evidence_root: Path,
    candidate: Callable[[CaseRecord], Any],
) -> list[Any]:
    """Fail closed: no candidate callback runs until every record is validated."""

    cases = load_case_manifest(
        manifest_path,
        source_catalog_path=source_catalog_path,
        repo_root=repo_root,
        evidence_root=evidence_root,
    )
    return [candidate(case) for case in cases]


def _torch():
    import torch

    return torch


def _hex_values(tensor: Any, width: int) -> list[str]:
    return [f"0x{int(value):0{width}x}" for value in tensor.reshape(-1).tolist()]


def encode_tensor(tensor: Any) -> dict[str, Any]:
    """Encode dense, quantized, sparse COO, or nested tensor data without float JSON."""

    torch = _torch()
    value = tensor.detach().cpu()
    common = {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "layout": str(value.layout),
    }
    if getattr(value, "is_nested", False):
        return {
            **common,
            "encoding": "nested_tensors",
            "constituents": [encode_tensor(item) for item in value.unbind()],
        }
    if value.is_quantized:
        result = {
            **common,
            "strides": list(value.stride()),
            "storage_offset": int(value.storage_offset()),
            "encoding": "quantized_integer_representation",
            "integer_representation": encode_tensor(value.int_repr()),
            "qscheme": str(value.qscheme()),
        }
        if value.qscheme() in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            result.update(scale=float(value.q_scale()), zero_point=int(value.q_zero_point()))
        else:
            result.update(
                scales=encode_tensor(value.q_per_channel_scales()),
                zero_points=encode_tensor(value.q_per_channel_zero_points()),
                axis=int(value.q_per_channel_axis()),
            )
        return result
    if value.layout == torch.sparse_coo:
        coalesced = value.coalesce()
        return {
            **common,
            "encoding": "sparse_coo",
            "indices": encode_tensor(coalesced.indices()),
            "values": encode_tensor(coalesced.values()),
            "coalesced": bool(value.is_coalesced()),
        }
    if value.layout != torch.strided:
        raise FixtureValidationError(f"unsupported tensor layout for frozen encoding: {value.layout}")
    common.update(
        strides=list(value.stride()),
        storage_offset=int(value.storage_offset()),
    )
    logical = value.contiguous()
    float_bits = {
        torch.float16: (torch.uint16, 4),
        torch.bfloat16: (torch.uint16, 4),
        torch.float32: (torch.uint32, 8),
        torch.float64: (torch.uint64, 16),
    }
    if logical.dtype in float_bits:
        integer_dtype, width = float_bits[logical.dtype]
        return {
            **common,
            "encoding": "ieee754_bits",
            "values": _hex_values(logical.view(integer_dtype), width),
        }
    complex_bits = {
        getattr(torch, "complex32", object()): (torch.float16, torch.uint16, 4),
        torch.complex64: (torch.float32, torch.uint32, 8),
        torch.complex128: (torch.float64, torch.uint64, 16),
    }
    if logical.dtype in complex_bits:
        _, integer_dtype, width = complex_bits[logical.dtype]
        lanes = torch.view_as_real(logical)
        return {
            **common,
            "encoding": "complex_ieee754_bits",
            "real": _hex_values(lanes[..., 0].contiguous().view(integer_dtype), width),
            "imag": _hex_values(lanes[..., 1].contiguous().view(integer_dtype), width),
        }
    integer_info = {
        torch.int8: (8, True),
        torch.uint8: (8, False),
        torch.int16: (16, True),
        torch.uint16: (16, False),
        torch.int32: (32, True),
        torch.uint32: (32, False),
        torch.int64: (64, True),
        torch.uint64: (64, False),
    }
    if logical.dtype in integer_info:
        bits, signed = integer_info[logical.dtype]
        return {
            **common,
            "encoding": "integer_decimal",
            "bit_width": bits,
            "signed": signed,
            "values": [str(int(item)) for item in logical.reshape(-1).tolist()],
        }
    if logical.dtype == torch.bool:
        return {
            **common,
            "encoding": "boolean",
            "values": [bool(item) for item in logical.reshape(-1).tolist()],
        }
    raise FixtureValidationError(f"unsupported tensor dtype for frozen encoding: {logical.dtype}")


def _dtype_from_name(name: str) -> Any:
    torch = _torch()
    if not name.startswith("torch."):
        raise FixtureValidationError(f"invalid torch dtype name {name!r}")
    dtype = getattr(torch, name.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise FixtureValidationError(f"unknown torch dtype name {name!r}")
    return dtype


def _restore_layout(logical: Any, record: Mapping[str, Any]) -> Any:
    torch = _torch()
    shape = tuple(int(item) for item in record["shape"])
    strides = tuple(int(item) for item in record["strides"])
    offset = int(record["storage_offset"])
    if any(stride < 0 for stride in strides) or offset < 0:
        raise FixtureValidationError("negative strides and offsets are not supported")
    if logical.numel() == 0:
        return torch.empty_strided(shape, strides, dtype=logical.dtype)
    storage_size = offset + 1 + sum((size - 1) * stride for size, stride in zip(shape, strides))
    base = torch.zeros(storage_size, dtype=logical.dtype)
    result = torch.as_strided(base, shape, strides, offset)
    result.copy_(logical.reshape(shape))
    return result


def decode_tensor(record: Mapping[str, Any]) -> Any:
    """Decode a canonical tensor record and restore its strided metadata."""

    torch = _torch()
    encoding = record.get("encoding")
    dtype = _dtype_from_name(_require_string(record.get("dtype"), "tensor.dtype"))
    shape = tuple(int(item) for item in record.get("shape", []))
    count = 1
    for size in shape:
        count *= size
    if encoding == "nested_tensors":
        return torch.nested.nested_tensor([decode_tensor(item) for item in record["constituents"]])
    if encoding == "sparse_coo":
        result = torch.sparse_coo_tensor(
            decode_tensor(record["indices"]),
            decode_tensor(record["values"]),
            size=shape,
            dtype=dtype,
        )
        return result.coalesce() if record.get("coalesced") else result
    if encoding == "quantized_integer_representation":
        integer = decode_tensor(record["integer_representation"])
        qscheme = record["qscheme"]
        if qscheme in {"torch.per_tensor_affine", "torch.per_tensor_symmetric"}:
            return torch._make_per_tensor_quantized_tensor(integer, record["scale"], record["zero_point"])
        return torch._make_per_channel_quantized_tensor(
            integer,
            decode_tensor(record["scales"]),
            decode_tensor(record["zero_points"]),
            record["axis"],
        )
    float_bits = {
        torch.float16: (torch.uint16, 4),
        torch.bfloat16: (torch.uint16, 4),
        torch.float32: (torch.uint32, 8),
        torch.float64: (torch.uint64, 16),
    }
    if encoding == "ieee754_bits" and dtype in float_bits:
        integer_dtype, width = float_bits[dtype]
        values = record.get("values")
        if not isinstance(values, list) or len(values) != count:
            raise FixtureValidationError("tensor IEEE value count does not match shape")
        if any(not isinstance(item, str) or not re.fullmatch(rf"0x[0-9a-f]{{{width}}}", item) for item in values):
            raise FixtureValidationError("tensor IEEE values are not canonical fixed-width hex")
        logical = torch.tensor([int(item, 16) for item in values], dtype=integer_dtype).view(dtype)
        return _restore_layout(logical.reshape(shape), record)
    complex_bits = {
        getattr(torch, "complex32", object()): (torch.float16, torch.uint16, 4),
        torch.complex64: (torch.float32, torch.uint32, 8),
        torch.complex128: (torch.float64, torch.uint64, 16),
    }
    if encoding == "complex_ieee754_bits" and dtype in complex_bits:
        lane_dtype, integer_dtype, width = complex_bits[dtype]
        real, imag = record.get("real"), record.get("imag")
        if not isinstance(real, list) or not isinstance(imag, list) or len(real) != count or len(imag) != count:
            raise FixtureValidationError("tensor complex lane count does not match shape")
        pattern = re.compile(rf"0x[0-9a-f]{{{width}}}")
        if any(not isinstance(item, str) or not pattern.fullmatch(item) for item in real + imag):
            raise FixtureValidationError("tensor complex lanes are not canonical fixed-width hex")
        real_tensor = torch.tensor([int(item, 16) for item in real], dtype=integer_dtype).view(lane_dtype)
        imag_tensor = torch.tensor([int(item, 16) for item in imag], dtype=integer_dtype).view(lane_dtype)
        lanes = torch.stack((real_tensor, imag_tensor), dim=-1)
        logical = torch.view_as_complex(lanes).reshape(shape)
        return _restore_layout(logical, record)
    if encoding == "integer_decimal":
        values = record.get("values")
        if not isinstance(values, list) or len(values) != count or any(
            not isinstance(item, str) or not re.fullmatch(r"-?(0|[1-9][0-9]*)", item)
            for item in values
        ):
            raise FixtureValidationError("tensor integer values are not canonical decimals")
        logical = torch.tensor([int(item) for item in values], dtype=dtype).reshape(shape)
        return _restore_layout(logical, record)
    if encoding == "boolean":
        values = record.get("values")
        if not isinstance(values, list) or len(values) != count or any(type(item) is not bool for item in values):
            raise FixtureValidationError("tensor boolean values are invalid")
        logical = torch.tensor(values, dtype=torch.bool).reshape(shape)
        return _restore_layout(logical, record)
    raise FixtureValidationError(f"unsupported tensor encoding {encoding!r} for {dtype}")


def exact_tensor_bits_equal(actual: Any, expected_record: Mapping[str, Any]) -> bool:
    """Compare tensor values and metadata using the canonical bit representation."""

    return encode_tensor(actual) == dict(expected_record)
