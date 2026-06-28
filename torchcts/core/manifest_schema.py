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

from dataclasses import dataclass, field
import os
import re

import torch

from torchcts.core.quantized_decoders import KNOWN_CONTAINER_FORMATS, validate_decoder_spec
from torchcts.core.semantic_levels import validate_semantic_level, SemanticLevelError
from torchcts.core.tolerances import DEFAULT_TOLERANCES, Tol, TieredTol, normalize_tolerance_overrides


CAPABILITY_ORDER = (
    "inference",
    "training",
    "serialization",
    "rng",
    "device_generator",
    "rng_distributions",
    "double_backward",
    "gradcheck",
    "gradient_checkpointing",
    "autocast",
    "fused_optimizer",
    "dataloader",
    "module_hooks",
    "channels_last",
    "sparse",
    "nested",
    "named_tensor",
    "foreach",
    "fp8",
    "quantized_container_plumbing",
    "native_quantization",
    "custom_quantized_decode",
    "compile",
    "pinned_memory",
    "streams",
    "events",
    "deterministic",
    "guard_alloc",
    "device_api",
    "multi_device",
    "ieee754",
)
KNOWN_CAPABILITIES = frozenset(CAPABILITY_ORDER)

KNOWN_TOP_LEVEL_KEYS = frozenset({
    "manifest_version",
    "device_name",
    "backend_import",
    "supported_dtypes",
    "device_count",
    "ieee754_seed",
    "max_samples",
    "max_samples_ieee754",
    "semantic_level",
    "hardware",
    "resource_limits",
    "capabilities",
    "skip_ops",
    "tolerance_overrides",
    "supported_container_formats",
    "custom_container_decoders",
    "custom_test_dirs",
    "show_traceback",
})

_REQUIRED_TOP_LEVEL_KEYS = ("manifest_version", "device_name", "capabilities")
_KNOWN_HARDWARE_KEYS = frozenset({
    "memory_model",
    "device_memory_gb",
    "system_memory_gb",
    "oom_recoverable",
})
_KNOWN_RESOURCE_LIMIT_KEYS = frozenset({
    "max_device_memory_mb",
    "max_system_memory_mb",
    "max_tensor_size_mb",
    "cleanup_threshold_pct",
})
_KNOWN_TOLERANCE_CATEGORIES = frozenset(category for category, _ in DEFAULT_TOLERANCES)
_VALID_MEMORY_MODELS = frozenset({"discrete", "unified"})


@dataclass
class ManifestValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self):
        return not self.errors


def dtype_from_manifest_key(key):
    if isinstance(key, torch.dtype):
        return key
    if isinstance(key, str):
        name = key.strip()
        if name.startswith("torch."):
            name = name[len("torch."):]
        value = getattr(torch, name, None)
        if isinstance(value, torch.dtype):
            return value
    return None


def _is_positive_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _is_nonnegative_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_bool_map(result, section_name, mapping, known_keys=None, allow_ieee754_patterns=False):
    if not isinstance(mapping, dict):
        result.errors.append(f"'{section_name}' must be a dictionary")
        return

    for key, value in mapping.items():
        if known_keys is not None and key not in known_keys:
            result.errors.append(f"Unknown {section_name} key '{key}'")
            continue
        if allow_ieee754_patterns and key == "ieee754":
            if isinstance(value, bool):
                continue
            if isinstance(value, str):
                try:
                    re.compile(value)
                except re.error as exc:
                    result.errors.append(f"capabilities.ieee754 regex is invalid: {exc}")
                continue
            if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
                for pattern in value:
                    try:
                        re.compile(pattern)
                    except re.error as exc:
                        result.errors.append(f"capabilities.ieee754 regex is invalid: {exc}")
                continue
        if not isinstance(value, bool):
            result.errors.append(f"'{section_name}.{key}' must be a boolean")


def _validate_supported_dtypes(result, supported_dtypes):
    if not isinstance(supported_dtypes, dict):
        result.errors.append("'supported_dtypes' must be a dictionary")
        return

    for key, value in supported_dtypes.items():
        if dtype_from_manifest_key(key) is None:
            result.errors.append(f"Invalid supported_dtypes key {key!r}")
        if isinstance(value, str):
            try:
                re.compile(value)
            except re.error as exc:
                result.errors.append(f"supported_dtypes[{key!r}] regex is invalid: {exc}")
        elif not isinstance(value, bool):
            result.errors.append(f"supported_dtypes[{key!r}] must be a boolean or regex string")


def _validate_hardware(result, manifest):
    hardware = manifest.get("hardware", {})
    if not isinstance(hardware, dict):
        result.errors.append("'hardware' must be a dictionary")
        return

    for key in hardware:
        if key not in _KNOWN_HARDWARE_KEYS:
            result.errors.append(f"Unknown hardware key '{key}'")

    memory_model = hardware.get("memory_model")
    if memory_model is not None and memory_model not in _VALID_MEMORY_MODELS:
        result.errors.append("'hardware.memory_model' must be 'discrete' or 'unified'")

    system_memory = hardware.get("system_memory_gb")
    if system_memory is not None and system_memory != "auto" and not _is_positive_number(system_memory):
        result.errors.append("'hardware.system_memory_gb' must be 'auto' or a positive number")

    device_memory = hardware.get("device_memory_gb")
    if device_memory is not None and device_memory != "auto":
        valid_list = (
            isinstance(device_memory, list)
            and device_memory
            and all(_is_positive_number(value) for value in device_memory)
        )
        if not valid_list:
            result.errors.append("'hardware.device_memory_gb' must be 'auto' or a non-empty list of positive numbers")

    if "oom_recoverable" in hardware and not isinstance(hardware["oom_recoverable"], bool):
        result.errors.append("'hardware.oom_recoverable' must be a boolean")


def _validate_resource_limits(result, manifest):
    limits = manifest.get("resource_limits", {})
    if not isinstance(limits, dict):
        result.errors.append("'resource_limits' must be a dictionary")
        return

    for key, value in limits.items():
        if key not in _KNOWN_RESOURCE_LIMIT_KEYS:
            result.errors.append(f"Unknown resource_limits key '{key}'")
            continue
        if key == "cleanup_threshold_pct":
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 100:
                result.errors.append("'resource_limits.cleanup_threshold_pct' must be an integer from 1 to 100")
        elif value is not None and not _is_positive_number(value):
            result.errors.append(f"'resource_limits.{key}' must be null or a positive number")


def _validate_tolerance_overrides(result, manifest):
    overrides = manifest.get("tolerance_overrides", {})
    if not isinstance(overrides, dict):
        result.errors.append("'tolerance_overrides' must be a dictionary")
        return

    try:
        normalized = normalize_tolerance_overrides(overrides)
    except ValueError as exc:
        result.errors.append(f"Invalid tolerance_overrides: {exc}")
        return

    for (category, dtype), value in normalized.items():
        if category not in _KNOWN_TOLERANCE_CATEGORIES:
            result.errors.append(f"Invalid tolerance_overrides category '{category}'")
        if dtype_from_manifest_key(dtype) is None:
            result.errors.append(f"Invalid tolerance_overrides dtype {dtype!r}")
        if isinstance(value, TieredTol):
            continue
        if isinstance(value, Tol):
            continue
        if isinstance(value, dict):
            unknown = set(value) - {"rtol", "atol"}
            if unknown:
                result.errors.append(f"Invalid tolerance override keys for '{category}:{dtype}': {sorted(unknown)}")
            for tol_key in ("rtol", "atol"):
                if tol_key in value and not isinstance(value[tol_key], (int, float)):
                    result.errors.append(f"tolerance_overrides {category}:{dtype} {tol_key} must be numeric")
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            if not all(isinstance(item, (int, float)) for item in value):
                result.errors.append(f"tolerance_overrides {category}:{dtype} tuple values must be numeric")
        else:
            result.errors.append(f"Invalid tolerance override value for '{category}:{dtype}'")


def _validate_custom_test_dirs(result, manifest, base_dir):
    custom_dirs = manifest.get("custom_test_dirs", [])
    if not isinstance(custom_dirs, list) or not all(isinstance(item, str) for item in custom_dirs):
        result.errors.append("'custom_test_dirs' must be a list of strings")
        return

    if base_dir:
        for path in custom_dirs:
            abs_path = path if os.path.isabs(path) else os.path.join(base_dir, path)
            if not os.path.isdir(abs_path):
                result.warnings.append(f"custom_test_dirs entry '{path}' does not exist")


def _validate_quantized_sections(result, manifest):
    formats = manifest.get("supported_container_formats", {})
    if not isinstance(formats, dict):
        result.errors.append("'supported_container_formats' must be a dictionary")
        return

    _validate_bool_map(result, "supported_container_formats", formats, KNOWN_CONTAINER_FORMATS)

    decoders = manifest.get("custom_container_decoders", {})
    if not isinstance(decoders, dict):
        result.errors.append("'custom_container_decoders' must be a dictionary")
        return

    for fmt, spec in decoders.items():
        if fmt not in KNOWN_CONTAINER_FORMATS:
            result.errors.append(f"Unknown custom_container_decoders key '{fmt}'")
            continue
        if not formats.get(fmt, False):
            result.errors.append(f"custom_container_decoders.{fmt} requires supported_container_formats.{fmt}=True")
        decoder_error = validate_decoder_spec(spec)
        if decoder_error:
            result.errors.append(f"Invalid custom_container_decoders.{fmt}: {decoder_error}")

    caps = manifest.get("capabilities", {})
    if isinstance(caps, dict):
        if caps.get("custom_quantized_decode", False) and not decoders:
            result.errors.append(
                "'capabilities.custom_quantized_decode' requires at least one custom_container_decoders entry"
            )
        if decoders and not caps.get("custom_quantized_decode", False):
            result.warnings.append(
                "custom_container_decoders is set but capabilities.custom_quantized_decode is false"
            )


def validate_manifest(manifest, base_dir=None):
    result = ManifestValidationResult()

    if not isinstance(manifest, dict):
        result.errors.append("'manifest' must be a dictionary")
        return result

    for key in _REQUIRED_TOP_LEVEL_KEYS:
        if key not in manifest:
            result.errors.append(f"Missing required key '{key}'")

    for key in manifest:
        if key not in KNOWN_TOP_LEVEL_KEYS:
            result.errors.append(f"Unknown top-level manifest key '{key}'")

    if manifest.get("manifest_version") != 1:
        result.warnings.append(f"Unexpected manifest_version {manifest.get('manifest_version')!r}; expected 1")

    if "device_name" in manifest and not isinstance(manifest["device_name"], str):
        result.errors.append("'device_name' must be a string")
    if "backend_import" in manifest and manifest["backend_import"] is not None and not isinstance(manifest["backend_import"], str):
        result.errors.append("'backend_import' must be null or a string")
    if "device_count" in manifest and (not isinstance(manifest["device_count"], int) or isinstance(manifest["device_count"], bool) or manifest["device_count"] < 1):
        result.errors.append("'device_count' must be a positive integer")
    for key in ("ieee754_seed", "max_samples", "max_samples_ieee754"):
        if key in manifest and not _is_nonnegative_int(manifest[key]):
            result.errors.append(f"'{key}' must be a non-negative integer")
    if "semantic_level" in manifest:
        try:
            validate_semantic_level(manifest["semantic_level"])
        except SemanticLevelError as exc:
            result.errors.append(str(exc))
    if "show_traceback" in manifest and not isinstance(manifest["show_traceback"], bool):
        result.errors.append("'show_traceback' must be a boolean")

    if "supported_dtypes" in manifest:
        _validate_supported_dtypes(result, manifest["supported_dtypes"])

    capabilities = manifest.get("capabilities", {})
    _validate_bool_map(result, "capabilities", capabilities, KNOWN_CAPABILITIES, allow_ieee754_patterns=True)

    if "skip_ops" in manifest:
        skip_ops = manifest["skip_ops"]
        if not isinstance(skip_ops, list) or not all(isinstance(item, str) for item in skip_ops):
            result.errors.append("'skip_ops' must be a list of strings")

    _validate_hardware(result, manifest)
    _validate_resource_limits(result, manifest)
    _validate_tolerance_overrides(result, manifest)
    _validate_quantized_sections(result, manifest)
    _validate_custom_test_dirs(result, manifest, base_dir)

    return result
