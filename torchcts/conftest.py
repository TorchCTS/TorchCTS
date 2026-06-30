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
import sys
import json
import glob
import time
import datetime
import inspect
import subprocess
import warnings
import traceback
import faulthandler
from contextlib import contextmanager
import pytest
import torch

# Configure stdout/stderr encoding/errors to handle unicode properly
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# Enable faulthandler so segfaults print a traceback instead of silent death
faulthandler.enable()


@contextmanager
def _without_stale_conda_env_for_venv():
    """Hide inherited Conda markers when running inside a non-Conda venv.

    PyTorch Inductor checks CONDA_PREFIX during compile setup and may shell out
    to `conda list` even when the active interpreter is a regular virtualenv.
    If the shell has stale Conda markers, that subprocess can run the wrong
    root Conda Python and pollute TorchCTS output.
    """
    active_prefix = os.path.abspath(sys.prefix)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if sys.prefix == sys.base_prefix or not conda_prefix:
        yield
        return
    if os.path.abspath(conda_prefix) == active_prefix:
        yield
        return

    keys = [
        key for key in os.environ
        if key.startswith("CONDA")
    ]
    saved = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        yield
    finally:
        for key in keys:
            if saved[key] is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = saved[key]

from torchcts.core.device import (
    get_device_backend,
    synchronize,
    empty_cache,
    memory_allocated
)
from torchcts.core.report import get_hardware_key
from torchcts.core.opinfo_adapter import (
    consume_pending_manifest_skips,
    dtype_manifest_disposition,
    str_to_dtype,
    dtype_to_str,
)
from torchcts.core.dtype_contracts import NOT_RECORDED, contract_disposition, load_dtype_contracts
from torchcts.core.comparer import clear_metrics, get_metrics
from torchcts.core.input_gen import refresh_shared_data
from torchcts.core.runtime_evidence import (
    harness_probe_failure_key,
    record_harness_probe_failure,
)
from torchcts.core.semantic_levels import (
    DEFAULT_REQUESTED_SEMANTIC_LEVEL,
    SemanticLevelError,
    SemanticLevelSelection,
    generated_level_for_entry,
    marker_value_to_level,
    normalize_level_selection,
    suite_default_level,
    validate_semantic_level,
)

# Global session variables
_MANIFEST = {}
_DEVICE_NAME = "cpu"
_HARDWARE_KEY = "unknown"
_RESULTS_DIR = "./results"
_START_TIME = 0
_SESSION_RESULTS = {}
_SESSION_SKIPS = {}
_SESSION_PROBE_FAILURES = []
_SESSION_PROBE_FAILURE_KEYS = set()
_DTYPE_MANIFEST_SKIP_REASONS = frozenset({
    "dtype_not_supported",
    "dtype_regex_filtered",
    "dtype_not_listed",
})
_DTYPE_CONTRACT_SKIP_REASONS = frozenset({
    "cpu_contract_unsupported",
    "cpu_contract_unknown",
    "cpu_contract_pending",
})
_SEMANTIC_SELECTION_SKIP_REASONS = frozenset({
    "semantic_level_gt_requested",
    "semantic_level_out_of_range",
})
_DTYPE_PARAM_NAMES = ("dtype", "src_dtype", "dst_dtype", "autocast_dtype")
_HANDWRITTEN_CONTRACT_ALIASES = {
    # Python API names used by handwritten tests whose dtype contract is stored
    # under namespaced or overload-specific dispatcher metadata.
    "add": ("aten::add.Tensor",),
    "sub": ("aten::sub.Tensor",),
    "mul": ("aten::mul.Tensor",),
    "div": ("aten::div.Tensor",),
    "add_": ("aten::add_.Scalar",),
    "sub_": ("aten::sub_.Scalar",),
    "mul_": ("aten::mul_.Scalar",),
    "div_": ("aten::div_.Scalar",),
    "relu": ("aten::relu.out",),
    "gelu": ("aten::gelu.out",),
    "softmax": ("aten::softmax.int",),
    "log_softmax": ("aten::log_softmax.int",),
    "mse_loss": ("aten::mse_loss.out",),
    "smooth_l1": ("aten::smooth_l1_loss.out",),
    "smooth_l1_loss": ("aten::smooth_l1_loss.out",),
    "huber": ("aten::huber_loss.out",),
    "huber_loss": ("aten::huber_loss.out",),
    "binary_cross_entropy_with_logits": ("aten::binary_cross_entropy_with_logits.out",),
    "conv1d": ("aten::nn.functional.conv1d",),
    "conv2d": ("aten::nn.functional.conv2d",),
    "conv3d": ("aten::nn.functional.conv3d",),
    "conv_transpose2d": ("aten::nn.functional.conv_transpose2d",),
    "layer_norm": ("aten::nn.functional.layer_norm",),
    "group_norm": ("aten::nn.functional.group_norm",),
    "batch_norm": ("aten::nn.functional.batch_norm",),
    "instance_norm": ("aten::nn.functional.instance_norm",),
    "linear": ("aten::linear.out",),
    "scaled_dot_product_attention": ("aten::nn.functional.scaled_dot_product_attention",),
    "fft": ("aten::fft.fft",),
    "ifft": ("aten::fft.ifft",),
    "rfft": ("aten::fft.rfft",),
    "irfft": ("aten::fft.irfft",),
    "solve": ("aten::linalg.solve",),
    "inv": ("aten::linalg.inv",),
    "det": ("aten::linalg.det",),
    "matmul": ("aten::matmul",),
    "relu_add": ("aten::add.Tensor", "aten::relu.out"),
    "gelu_mul": ("aten::mul.Tensor", "aten::gelu.out"),
    "aten::fft_fft": ("aten::fft.fft",),
    "aten::fft_ifft": ("aten::fft.ifft",),
    "aten::fft_rfft": ("aten::fft.rfft",),
    "aten::fft_irfft": ("aten::fft.irfft",),
    "aten::linalg_cholesky": ("aten::linalg.cholesky",),
    "aten::linalg_qr": ("aten::linalg.qr",),
    "aten::linalg_svd": ("aten::linalg.svd",),
    "aten::_conj": ("aten::conj",),
    "aten::_to_copy": ("aten::to.dtype",),
}
_HANDWRITTEN_FUNCTION_CONTRACT_ALIASES = {
    "test_compile_dynamic_batch_linear": ("linear",),
    "test_compile_matmul": ("mm",),
    "test_compile_softmax": ("softmax",),
    "test_compile_layer_norm": ("native_layer_norm",),
    "test_compile_conv2d": ("conv2d",),
    "test_compile_chained_ops": ("mm", "relu", "add", "gelu", "sum"),
    "test_compile_training_optimizer": ("linear",),
    "test_compile_multi_step_convergence": ("linear", "mse_loss"),
    "test_allocator_tracking_and_cache": ("randn",),
    "test_oom_recovery": ("empty", "fill_"),
    "test_determinism_stale_buffers": ("randn", "mm", "silu", "native_layer_norm", "softmax"),
    "test_guard_alloc_canary": ("randn", "add"),
    "test_save_load_roundtrip": ("randn", "to"),
    "test_scale_mismatch_numerics": ("logsumexp", "cumsum"),
    "test_dtype_min_max": ("scalar_tensor",),
    "test_zero_element_and_scalar_tensors": ("empty", "add", "scalar_tensor"),
    "test_large_allocations": ("empty", "fill_"),
    "test_checkpoint_roundtrip": ("linear",),
    "test_dataloader_pin_memory": ("randn", "to"),
    "test_gradient_accumulation": ("mm", "sum"),
    "test_lr_schedulers": ("randn",),
    "test_module_hooks": ("linear", "sum"),
    "test_gradient_checkpointing": ("linear", "sum"),
    "test_optimizer_pipelines": ("linear", "sum"),
    "test_fused_optimizer_pipelines": ("linear", "sum"),
    "test_autocast_precisions": ("mm",),
    "test_lora_forward_backward": ("linear", "matmul", "add", "mul", "sum"),
    "test_gemv_m1_shapes": ("mm",),
    "test_sdpa_causal": ("scaled_dot_product_attention",),
    "test_sdpa_nested_forward": ("scaled_dot_product_attention",),
    "test_sdpa_nested_backward_dtypes": ("scaled_dot_product_attention",),
}
_HANDWRITTEN_PARAM_CONTRACT_ALIASES = {
    ("model_name", "linear"): ("linear",),
    ("model_name", "mlp"): ("linear", "relu"),
    ("model_name", "conv"): ("conv2d", "adaptive_avg_pool2d", "linear", "relu"),
    ("model_name", "norm"): ("native_layer_norm", "linear"),
    ("model_name", "gpt2"): ("linear", "native_layer_norm", "gelu"),
    ("model_name", "qwen"): ("linear", "native_layer_norm", "silu"),
    ("component", "cnn"): ("conv2d", "batch_norm", "relu"),
    ("component", "vit"): ("conv2d",),
    ("block_type", "gpt2"): ("linear", "native_layer_norm", "gelu"),
    ("clipping_method", "norm"): ("norm",),
    ("clipping_method", "value"): ("clamp",),
}
_BASELINE_RESULTS = {}
_SHOW_SKIPS = False
_REPORT_SKIPS = False
_SUBPROCESS_MODE = False
_KNOWN_SEGFAULT_POLICY = "isolate"
_KNOWN_SEGFAULTS_ACTIVE = []
_KNOWN_SEGFAULT_WARNINGS = []
_KNOWN_SEGFAULT_AUDIT = False
_ADAPTIVE_ISOLATION_MODE = "auto"
_ADAPTIVE_ISOLATION_LOAD = None
_ADAPTIVE_ISOLATION_ACTIVE = {}
_ADAPTIVE_ISOLATION_REJECTED = []
_ADAPTIVE_ISOLATION_WARNINGS = []
_MEMORY_MODE = "balanced"
_CLEANUP_THRESHOLD = 80
_RUN_LOG_FH = None
_REQUESTED_SEMANTIC_LEVEL = DEFAULT_REQUESTED_SEMANTIC_LEVEL
_SEMANTIC_LEVEL_SELECTION = SemanticLevelSelection("cumulative", 1, DEFAULT_REQUESTED_SEMANTIC_LEVEL)
_SESSION_COMPLETED = False

# pytest-xdist parallel execution support
_XDIST_WORKER_ID = os.environ.get("PYTEST_XDIST_WORKER")  # e.g. "gw0", "gw1" or None
_IS_XDIST_WORKER = _XDIST_WORKER_ID is not None

_RUNTIME_UNSUPPORTED_PATTERNS_UNIVERSAL = [
    # PyTorch dispatcher: op not registered for backend
    r"Could not run '.*' with arguments from the '.*' backend",
    r"Could not run '.*' from the '.*' device",
    # Backend dtype conversion limitations (any backend)
    r"Cannot convert a \S+ Tensor to float64 dtype",
    # Backend op/dtype restrictions (any backend — matches "MPS does not support",
    # "CUDA does not support", "XPU does not support", etc.)
    r"does not support .* for non-float",
    r"device does not support .* for non-float",
    # Backend type restrictions (any backend)
    r"doesn't support complex types",
    r"only supports floats",
    r"currently supports float32 only",
    r"Only float is supported",
    # tensor_split device mismatch (framework limitation)
    r"tensor_split expected .* to be on cpu, but it's on",
    # Generic kernel/dtype restrictions
    r"not supported on (?:MPS|CUDA|XPU)",
    r"Convolution is supported only for Floating types",
    r"Failed to create function state object for:",
    r"does not support automatic differentiation for outputs with complex dtype",
    r"memory format option is only supported by strided tensors",
    r"only supports floating-point dtypes",
    r"value cannot be converted to type .* without overflow",
]

_RUNTIME_UNSUPPORTED_PATTERNS_MPS = [
    r"Adaptive pool MPS: input sizes must be divisible by output sizes",
    r"grid_sampler_3d: Unsupported Nearest interpolation",
    r"linalg_inv: not supported for complex types yet",
    r"cholesky_inverse: MPS only supports float type",
    r"index_reduce for MPS does not support torch\.long",
    r"ConvTranspose 3D with BF16 or FP16 types is not supported on MPS",
    r"the MPS framework doesn't support",
]

# Effective pattern list — built during pytest_configure based on device
_RUNTIME_UNSUPPORTED_PATTERNS = list(_RUNTIME_UNSUPPORTED_PATTERNS_UNIVERSAL)

_MAX_DEVICE_MEM = None
_MAX_TENSOR_SIZE = None
_COLLECT_ONLY = False
_ARTIFACT_WRITES_ENABLED = True
_ACTUAL_DEVICE_COUNT = 1
_SITE_STATS_COLLECTION_ENV = "TORCHCTS_SITE_STATS_COLLECTION_JSON"


def _site_stats_collection_enabled():
    return bool(os.environ.get(_SITE_STATS_COLLECTION_ENV))


def _is_child_process():
    return (
        os.environ.get("_TORCHCTS_SUBPROCESS") == "1"
        or os.environ.get("_TORCH_CTS_SUBPROCESS") == "1"
        or os.environ.get("_BACKEND_VALIDATOR_SUBPROCESS") == "1"
    )


def _text_tail(text, limit=12000):
    if not text:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    elif not isinstance(text, str):
        text = str(text)
    if len(text) <= limit:
        return text
    return text[-limit:]


def _atomic_json_dump(path, payload):
    """Write JSON without exposing a partially-written target file."""

    target = os.fspath(path)
    directory = os.path.dirname(target) or "."
    os.makedirs(directory, exist_ok=True)
    temp_path = os.path.join(
        directory,
        f".{os.path.basename(target)}.{os.getpid()}.{time.time_ns()}.tmp",
    )
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, target)
        try:
            dir_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    except Exception:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise


def _signal_name(returncode):
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
        import signal

        return signal.Signals(signum).name
    except Exception:
        return f"SIG{signum}"


def _is_process_crash(returncode, stderr="", stdout=""):
    sig = _signal_name(returncode)
    if sig in {"SIGSEGV", "SIGABRT", "SIGBUS", "SIGILL"}:
        return True
    haystack = f"{stderr or ''}\n{stdout or ''}"
    return "Fatal Python error" in haystack or "Segmentation fault" in haystack


def _known_segfault_match_for_item(item):
    if _KNOWN_SEGFAULT_POLICY != "isolate" or not _KNOWN_SEGFAULTS_ACTIVE:
        return None
    try:
        from torchcts.core.known_segfaults import KnownSegfaultError, match_known_segfault

        return match_known_segfault(
            item,
            _KNOWN_SEGFAULTS_ACTIVE,
            metadata=_extract_result_metadata(item),
        )
    except KnownSegfaultError as exc:
        pytest.exit(f"Invalid known segfault match: {exc}", returncode=1)
    except Exception as exc:
        pytest.exit(f"Failed to match known segfault policy for {item.nodeid}: {exc}", returncode=1)
        return None


def _known_segfault_result_fields(match, *, resolved=False, actual_signal=None):
    if not match:
        return {}
    fields = {
        "isolation_source": "known_segfault",
        "known_segfault_id": match["id"],
        "known_segfault_dispatcher": match["dispatcher"],
        "known_segfault_classification": match.get("classification"),
        "known_segfault_reason": match["reason"],
        "known_segfault_expected_signal": match["expected_signal"],
        "known_segfault_repro": dict(match["repro"]),
        "known_segfault_match": match.get("matched_by") or match.get("match"),
        "known_segfault_evidence_scope": match.get("evidence_scope"),
        "known_segfault_constraints": dict(match.get("constraints") or {}),
        "known_segfault_matched_nodeid": match.get("matched_nodeid"),
        "known_segfault_matched_metadata": dict(match.get("matched_metadata") or {}),
    }
    if resolved:
        fields["known_segfault_resolved"] = True
    if actual_signal is not None and actual_signal != match["expected_signal"]:
        fields["known_segfault_unexpected_signal"] = actual_signal
    return fields


def _known_segfault_process_classification(match, stdout="", stderr=""):
    if not match:
        return None
    text = f"{stdout or ''}\n{stderr or ''}".lower()
    if (
        "_grid_sampler_2d_cpu_fallback_backward" in str(match.get("dispatcher", ""))
        and "tensor-likes are not close" in text
        and "mismatched elements" in text
    ):
        return "confirmed_backend_wrong_value"
    return "confirmed_backend_crash"


def _canonical_collection_path(value):
    from torchcts.core.known_segfaults import canonicalize_nodeid

    text = canonicalize_nodeid(str(value).split("::", 1)[0]).rstrip("/")
    return text or "."


def _path_selects_file(selected_path, candidate_file):
    selected = _canonical_collection_path(selected_path)
    candidate = _canonical_collection_path(candidate_file)
    if selected in {".", "torchcts"}:
        return True
    return candidate == selected or candidate.startswith(f"{selected}/")


def _known_segfault_descriptor_for_item(item):
    from torchcts.core.known_segfaults import canonicalize_nodeid

    canonical_nodeid = canonicalize_nodeid(item.nodeid)
    return {
        "nodeid": item.nodeid,
        "canonical_nodeid": canonical_nodeid,
        "file": canonical_nodeid.split("::", 1)[0],
        "metadata": _extract_result_metadata(item),
    }


def _selected_collection_args(config):
    return [str(arg) for arg in getattr(config, "args", []) if str(arg) and not str(arg).startswith("-")]


def _known_segfault_entry_in_scope(config, entry, descriptors):
    from torchcts.core.known_segfaults import canonicalize_nodeid, entry_matches

    primary_matches = [
        descriptor
        for descriptor in descriptors
        if entry_matches(entry, descriptor["canonical_nodeid"], descriptor["metadata"], include_constraints=False)
    ]
    if primary_matches:
        dtype_constraints = (entry.get("constraints") or {}).get("dtype") or []
        if dtype_constraints and not any(
            str(descriptor["metadata"].get("dtype")) in dtype_constraints
            for descriptor in primary_matches
        ):
            return False
        return True

    suite_option = config.getoption("--suite") if hasattr(config, "getoption") else None
    suite_constraints = (entry.get("constraints") or {}).get("suite") or []

    selected_args = _selected_collection_args(config)
    if suite_option and suite_option in suite_constraints and not selected_args:
        return True

    selected_nodes = [arg for arg in selected_args if "::" in arg]
    if selected_nodes:
        entry_nodeid = entry.get("nodeid")
        if not entry_nodeid:
            return False
        canonical_entry = canonicalize_nodeid(entry_nodeid)
        return any(canonical_entry.startswith(canonicalize_nodeid(arg)) for arg in selected_nodes)

    selected_paths = [_canonical_collection_path(arg) for arg in selected_args]
    if not selected_paths:
        return True
    if suite_option is None and (len(selected_paths) >= 8 or any(path in {".", "torchcts"} for path in selected_paths)):
        return True

    if entry.get("nodeid"):
        entry_file = canonicalize_nodeid(entry["nodeid"]).split("::", 1)[0]
        if any(_path_selects_file(path, entry_file) for path in selected_paths):
            return True

    if suite_constraints:
        for suite in suite_constraints:
            suite_path = f"torchcts/{suite}"
            if any(_path_selects_file(path, suite_path) for path in selected_paths):
                return True
    return False


def _validate_known_segfault_collection(config, entries, items):
    from torchcts.core.known_segfaults import (
        KnownSegfaultError,
        best_known_segfault_match,
        entry_matches,
    )

    descriptors = [_known_segfault_descriptor_for_item(item) for item in items]
    errors = []
    for descriptor in descriptors:
        try:
            best_known_segfault_match(
                descriptor["canonical_nodeid"],
                entries,
                metadata=descriptor["metadata"],
            )
        except KnownSegfaultError as exc:
            errors.append(str(exc))

    stale_entries = []
    for entry in entries:
        if not _known_segfault_entry_in_scope(config, entry, descriptors):
            continue
        if not any(entry_matches(entry, d["canonical_nodeid"], d["metadata"]) for d in descriptors):
            stale_entries.append(entry)

    if stale_entries:
        errors.append(
            "Known segfault ledger contains stale in-scope rule(s):\n"
            + "\n".join(
                "  - {id}: match={match} dispatcher={dispatcher} evidence_scope={scope} constraints={constraints}".format(
                    id=entry["id"],
                    match=entry["match"],
                    dispatcher=entry["dispatcher"],
                    scope=entry["evidence_scope"],
                    constraints=entry.get("constraints") or {},
                )
                for entry in stale_entries
            )
        )

    if errors:
        pytest.exit("\n".join(errors), returncode=1)
    return descriptors


def _print_known_segfault_audit(config, entries, descriptors):
    from torchcts.core.known_segfaults import entry_matches

    verbose = getattr(getattr(config, "option", None), "verbose", 0) or 0
    print(f"\nKnown segfault audit: {len(entries)} active rule(s)")
    for entry in entries:
        matches = [
            descriptor
            for descriptor in descriptors
            if entry_matches(entry, descriptor["canonical_nodeid"], descriptor["metadata"])
        ]
        print(
            f"  - {entry['id']}: match={entry['match']} "
            f"evidence_scope={entry['evidence_scope']} matched={len(matches)}"
        )
        if verbose:
            for descriptor in matches:
                print(f"      {descriptor['canonical_nodeid']}")


def _adaptive_isolation_match_for_item(item):
    if not _ADAPTIVE_ISOLATION_ACTIVE:
        return None
    try:
        from torchcts.core.known_segfaults import canonicalize_nodeid

        return _ADAPTIVE_ISOLATION_ACTIVE.get(canonicalize_nodeid(item.nodeid))
    except Exception:
        return None


def _adaptive_isolation_result_fields(match, *, known_segfault_match=None, resolved=False):
    if not match:
        return {}
    fields = {
        "isolation_source": "known_segfault" if known_segfault_match else match["isolation_source"],
        "adaptive_isolation_source": match["isolation_source"],
        "adaptive_isolation_reason": match["reason"],
        "adaptive_isolation_evidence_path": match["evidence_path"],
        "adaptive_isolation_prior_status": match.get("prior_status"),
        "adaptive_isolation_prior_signal": match.get("prior_signal"),
    }
    if match.get("prior_error_type") is not None:
        fields["adaptive_isolation_prior_error_type"] = match.get("prior_error_type")
    if match.get("prior_timestamp") is not None:
        fields["adaptive_isolation_prior_timestamp"] = match.get("prior_timestamp")
    if resolved:
        fields["adaptive_isolation_resolved"] = True
    return fields


def _finalize_adaptive_isolation_for_collection(config, items):
    global _ADAPTIVE_ISOLATION_ACTIVE, _ADAPTIVE_ISOLATION_REJECTED
    if (
        _ADAPTIVE_ISOLATION_MODE != "auto"
        or _ADAPTIVE_ISOLATION_LOAD is None
        or _is_child_process()
        or _COLLECT_ONLY
        or _SHOW_SKIPS
    ):
        _ADAPTIVE_ISOLATION_ACTIVE = {}
        return

    from torchcts.core.adaptive_isolation import (
        build_adaptive_isolation_artifact,
        filter_candidates_for_collection,
    )

    accepted, rejected = filter_candidates_for_collection(
        _ADAPTIVE_ISOLATION_LOAD,
        [item.nodeid for item in items],
    )
    _ADAPTIVE_ISOLATION_ACTIVE = accepted
    _ADAPTIVE_ISOLATION_REJECTED = rejected

    if not _IS_XDIST_WORKER:
        if accepted:
            print(f"Adaptive isolation: {len(accepted)} node(s) from previous crash/hang evidence")
            if getattr(config.option, "verbose", 0):
                for candidate in accepted.values():
                    print(f"  - {candidate['nodeid']} ({candidate['isolation_source']})")
        if _ARTIFACT_WRITES_ENABLED:
            artifact = build_adaptive_isolation_artifact(
                hardware_key=_HARDWARE_KEY,
                device_name=_DEVICE_NAME,
                torch_version=torch.__version__,
                mode=_ADAPTIVE_ISOLATION_MODE,
                accepted=accepted,
                rejected=rejected,
                warnings=_ADAPTIVE_ISOLATION_WARNINGS,
                artifacts_considered=_ADAPTIVE_ISOLATION_LOAD.artifacts_considered,
            )
            path = os.path.join(_RESULTS_DIR, f"{_HARDWARE_KEY}_adaptive_isolation.json")
            _atomic_json_dump(path, artifact)


def _failure_stage_for_exception(error_type, error_message):
    message = (error_message or "").lower()
    if error_type == "AssertionError":
        return "comparison"
    if "cpu oracle" in message or ("expected" in message and "cpu" in message):
        return "cpu_oracle"
    if "copy" in message and "cpu" in message:
        return "sync_or_copy"
    if "failed on mps" in message or "mps" in message:
        return "mps_execution"
    return "pytest_call"


def _canonical_suite_for_item(item):
    filepath = str(item.fspath).replace("\\", "/")
    for suite_name in (
        "opinfo",
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
        "selftest",
        "stress",
        "multi_device",
        "generated",
    ):
        token = f"/{suite_name}/"
        if token in filepath:
            return suite_name
    return "custom"


def _runtime_skip_reason(err_msg: str, previous_skip: dict | None, item) -> str:
    """Classify a pytest.skip raised during execution."""

    if previous_skip:
        return previous_skip["skip_reason"]
    if "coverage_unknown" in err_msg:
        return "coverage_unknown"
    if "coverage_excluded" in err_msg:
        return "coverage_excluded"
    if "backend_not_available" in err_msg:
        return "backend_not_available"
    if "coverage_strategy_pending" in err_msg:
        return "coverage_strategy_pending"
    return "runtime_skip"


def _runtime_unsupported_pattern_match(message: str) -> str | None:
    import re

    for pattern in _RUNTIME_UNSUPPORTED_PATTERNS:
        if re.search(pattern, message or ""):
            return pattern
    return None


def _probe_failure_tail(text, limit=1200):
    text = "" if text is None else str(text)
    return text if len(text) <= limit else text[-limit:]


def _harness_probe_artifact_path():
    if not _RESULTS_DIR or not _HARDWARE_KEY:
        return None
    return os.path.join(
        _RESULTS_DIR,
        f"{_HARDWARE_KEY}_harness_probe_failures_{os.getpid()}.jsonl",
    )


def _warn_probe_failure_once(record):
    if _COLLECT_ONLY or _SHOW_SKIPS or _KNOWN_SEGFAULT_AUDIT:
        return
    message = _probe_failure_tail(record.get("error_message"), limit=300).replace("\n", " | ")
    print(
        "Warning: declared {kind} {name!r} failed diagnostic probe "
        "({stage}); tests will still run. {etype}: {message}".format(
            kind=record.get("probe_kind"),
            name=record.get("name"),
            stage=record.get("stage"),
            etype=record.get("error_type"),
            message=message,
        ),
        file=sys.stderr,
    )


def _record_session_probe_failure(probe_kind, name, exc_or_result, *, stage):
    record = record_harness_probe_failure(probe_kind, name, exc_or_result, stage=stage)
    if record is None:
        return None
    key = harness_probe_failure_key(record)
    if key in _SESSION_PROBE_FAILURE_KEYS:
        return record
    _SESSION_PROBE_FAILURE_KEYS.add(key)
    _SESSION_PROBE_FAILURES.append(record)
    _warn_probe_failure_once(record)
    return record


def _apply_declared_dtype_probes(supported_dtypes, device_name):
    records = []
    for dt, val in list(supported_dtypes.items()):
        if not val:
            continue
        probe_dtype = dt if isinstance(dt, torch.dtype) else str_to_dtype(str(dt))
        dtype_name = dtype_to_str(probe_dtype) if probe_dtype is not None else str(dt)
        if probe_dtype is None:
            records.append(
                _record_session_probe_failure(
                    "dtype",
                    dtype_name,
                    ValueError(f"Could not resolve manifest dtype key {dt!r}."),
                    stage="declared_dtype_probe",
                )
            )
            continue
        try:
            torch.zeros(1, dtype=probe_dtype, device=device_name)
        except Exception as exc:
            records.append(
                _record_session_probe_failure(
                    "dtype",
                    dtype_name,
                    exc,
                    stage="declared_dtype_probe",
                )
            )
    return [record for record in records if record is not None]


def _capability_probe_accepts_backend_import(probe_func):
    try:
        params = inspect.signature(probe_func).parameters
    except (TypeError, ValueError):
        return False
    if "backend_import" in params:
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())


def _apply_declared_capability_probes(caps, device_name, probe_func=None, backend_import=None):
    if probe_func is None:
        from torchcts.core.device import probe_capability_result

        probe_func = probe_capability_result
    probe_accepts_backend_import = _capability_probe_accepts_backend_import(probe_func)
    records = []
    for cap in ["pinned_memory", "sparse", "nested", "named_tensor", "fp8"]:
        if not caps.get(cap, False):
            continue
        if backend_import is not None and probe_accepts_backend_import:
            probe = probe_func(device_name, cap, backend_import=backend_import)
        else:
            probe = probe_func(device_name, cap)
        if probe.supported:
            continue
        records.append(
            _record_session_probe_failure(
                "capability",
                cap,
                probe,
                stage="declared_capability_probe",
            )
        )
    return [record for record in records if record is not None]


def _extract_result_metadata(item):
    metadata = {
        "suite": _canonical_suite_for_item(item),
        "test_kind": "opinfo" if "/opinfo/" in str(item.fspath).replace("\\", "/") else "handwritten",
        "capability": None,
        "is_plumbing": False,
        "is_conformance": False,
        "op": None,
        "dtype": None,
        "shapes": None,
        "golden_pass": True,
        "usable_pass": True,
        "quality_warning": False,
        "requested_level": _REQUESTED_SEMANTIC_LEVEL,
        "semantic_level_selection": _SEMANTIC_LEVEL_SELECTION.to_metadata(),
        "dispatcher_name": None,
        "schema": None,
        "strategy": None,
        "strategy_family": None,
        "sample_descriptor": None,
    }

    filepath = str(item.fspath)
    if "test_quantized.py" in filepath and item.name.startswith("test_custom_quantized_decoder"):
        metadata["is_conformance"] = True
    elif "test_quantized.py" in filepath or "test_guard_alloc.py" in filepath:
        metadata["is_plumbing"] = True
    else:
        metadata["is_conformance"] = True

    req_caps = sorted(get_required_capabilities(item))
    if req_caps:
        metadata["capability"] = ",".join(req_caps)

    if hasattr(item, "callspec"):
        params = item.callspec.params
        if "op" in params:
            op_param = params["op"]
            metadata["op"] = getattr(op_param, "name", str(op_param))
        elif "op_name" in params:
            metadata["op"] = params["op_name"]

        if isinstance(params.get("entry"), dict):
            coverage_entry = params["entry"]
            strategy = (coverage_entry.get("generated") or {}).get("strategy") or {}
            metadata["op"] = coverage_entry.get("name") or metadata["op"]
            metadata["dispatcher_name"] = coverage_entry.get("name")
            metadata["schema"] = coverage_entry.get("schema")
            metadata["strategy"] = strategy.get("strategy")
            metadata["strategy_family"] = strategy.get("family")
            metadata["coverage_status"] = coverage_entry.get("status")
            metadata["coverage_id"] = coverage_entry.get("name")
            metadata["surface_kind"] = coverage_entry.get("surface_kind")
            metadata["variant_kind"] = coverage_entry.get("variant_kind")

        if "dtype" in params:
            metadata["dtype"] = dtype_to_str(params["dtype"])
        elif "dtype_str" in params:
            metadata["dtype"] = params["dtype_str"]

        if "sample_input" in params:
            sample = params["sample_input"]
            if hasattr(sample, "input") and isinstance(sample.input, torch.Tensor):
                shapes = [list(sample.input.shape)]
                if hasattr(sample, "args"):
                    for arg in sample.args:
                        if isinstance(arg, torch.Tensor):
                            shapes.append(list(arg.shape))
                metadata["shapes"] = shapes

    # Input condition tier (opinfo tests)
    if hasattr(item, "callspec") and "input_condition" in item.callspec.params:
        metadata["input_condition"] = item.callspec.params["input_condition"]

    covers = []
    coverage_surfaces = []
    for marker in item.iter_markers(name="covers"):
        for arg in marker.args:
            if isinstance(arg, str):
                covers.append(arg)
        surface = marker.kwargs.get("surface")
        if isinstance(surface, str):
            coverage_surfaces.append(surface)

    categories = []
    for marker in item.iter_markers(name="covers_category"):
        for arg in marker.args:
            if isinstance(arg, str):
                categories.append(arg)
    try:
        from torchcts.core.coverage import coverage_categories_for_path
        categories.extend(coverage_categories_for_path(str(item.fspath)))
    except Exception:
        pass

    metadata["covers"] = sorted(set(covers))
    metadata["covers_categories"] = sorted(set(categories))
    if not metadata.get("dispatcher_name") and len(metadata["covers"]) == 1:
        metadata["dispatcher_name"] = metadata["covers"][0]
    contract_surfaces = _contract_surfaces_for_item(item)
    if not metadata.get("dispatcher_name") and len(contract_surfaces) == 1:
        metadata["dispatcher_name"] = contract_surfaces[0]
    metadata["cpu_contract_surfaces"] = contract_surfaces
    metadata["cpu_contract_exempt_reason"] = _cpu_contract_exempt_reason(item)
    metadata["cpu_contract_dtype_gates"] = [
        {"surface": surface, "dtype": dtype_to_str(dtype)}
        for surface, dtype in _fixed_dtype_contract_gates_for_item(item)
    ]
    if metadata["test_kind"] == "opinfo":
        metadata["coverage_kind"] = "opinfo"
    elif metadata["suite"] == "generated":
        metadata["coverage_kind"] = "generated"
    elif categories and not covers:
        metadata["coverage_kind"] = "category"
    else:
        metadata["coverage_kind"] = "handwritten"
    if not metadata.get("surface_kind"):
        metadata["surface_kind"] = sorted(set(coverage_surfaces))[0] if coverage_surfaces else None
    metadata.setdefault("variant_kind", None)
    if not metadata.get("coverage_id"):
        metadata["coverage_id"] = ",".join(metadata["covers"]) if metadata["covers"] else None
    metadata.setdefault("coverage_status", None)
    metadata.update(_semantic_level_for_item(item))
    metadata["requested_level"] = _REQUESTED_SEMANTIC_LEVEL
    metadata["semantic_level_selection"] = _SEMANTIC_LEVEL_SELECTION.to_metadata()

    return metadata


def _skip_record_for_item(item, skip_reason, detail, extra=None):
    metadata = _extract_result_metadata(item)
    record = {
        "suite": metadata["suite"],
        "test_kind": metadata["test_kind"],
        "capability": metadata["capability"],
        "is_plumbing": metadata["is_plumbing"],
        "is_conformance": metadata["is_conformance"],
        "op": metadata["op"] or item.name,
        "dtype": metadata["dtype"],
        "covers": metadata["covers"],
        "covers_categories": metadata["covers_categories"],
        "coverage_kind": metadata["coverage_kind"],
        "surface_kind": metadata["surface_kind"],
        "variant_kind": metadata["variant_kind"],
        "coverage_id": metadata["coverage_id"],
        "coverage_status": metadata["coverage_status"],
        "cpu_contract_surfaces": metadata["cpu_contract_surfaces"],
        "cpu_contract_exempt_reason": metadata["cpu_contract_exempt_reason"],
        "cpu_contract_dtype_gates": metadata["cpu_contract_dtype_gates"],
        "input_condition": metadata.get("input_condition"),
        "semantic_level": metadata["semantic_level"],
        "requested_level": metadata["requested_level"],
        "semantic_level_selection": metadata["semantic_level_selection"],
        "level_reason": metadata["level_reason"],
        "level_source": metadata["level_source"],
        "semantic_skip_reason": skip_reason if skip_reason.startswith("semantic_") else None,
        "skip_reason": skip_reason,
        "detail": detail,
    }
    if extra:
        record.update(extra)
    return record


def _site_stats_dtype_fields_for_item(item):
    if not hasattr(item, "callspec"):
        return {}
    fields = {}
    for name in ("dtype", "dtype_str", "src_dtype", "dst_dtype", "autocast_dtype"):
        if name not in item.callspec.params:
            continue
        value = item.callspec.params[name]
        if isinstance(value, torch.dtype):
            fields[name] = dtype_to_str(value)
        else:
            fields[name] = str(value)
    return fields


def _site_stats_collection_record(item, decision, *, skip_reason=None, skip_detail=None):
    metadata = _extract_result_metadata(item)
    node_path, _, _tail = item.nodeid.partition("::")
    dtype_fields = _site_stats_dtype_fields_for_item(item)
    suite = metadata["suite"]
    test_kind = metadata["test_kind"]
    if suite == "generated":
        test_kind = "generated"
    elif suite == "selftest":
        test_kind = "selftest"
    return {
        "nodeid": item.nodeid,
        "file": node_path,
        "suite": suite,
        "test_kind": test_kind,
        "function": (getattr(item, "originalname", None) or item.name.split("[", 1)[0]),
        "semantic_level": metadata["semantic_level"],
        "capability": metadata["capability"],
        "dtype": metadata["dtype"],
        "dtype_fields": dtype_fields,
        "dispatcher_name": metadata["dispatcher_name"],
        "coverage_id": metadata["coverage_id"],
        "coverage_kind": metadata["coverage_kind"],
        "surface_kind": metadata["surface_kind"],
        "variant_kind": metadata["variant_kind"],
        "strategy": metadata["strategy"],
        "strategy_family": metadata["strategy_family"],
        "decision": decision,
        "skip_reason": skip_reason,
        "skip_detail": skip_detail,
    }


def _write_site_stats_collection_records(config, records):
    path = os.environ.get(_SITE_STATS_COLLECTION_ENV)
    if not path:
        return
    payload = {
        "metadata": {
            "schema_version": 1,
            "device_name": _DEVICE_NAME,
            "pytorch_version": torch.__version__,
            "requested_level": _REQUESTED_SEMANTIC_LEVEL,
            "semantic_level_selection": _SEMANTIC_LEVEL_SELECTION.to_metadata(),
            "dtype_filter": list(_MANIFEST.get("dtype_filter") or []),
            "collection_args": [str(arg) for arg in getattr(config, "args", [])],
        },
        "records": records,
    }
    _atomic_json_dump(path, payload)


def _merge_pending_manifest_skips(*, include_opinfo):
    pending = consume_pending_manifest_skips()
    if not include_opinfo:
        return
    default_level = suite_default_level("opinfo")
    for nodeid, record in pending.items():
        record.setdefault("requested_level", _REQUESTED_SEMANTIC_LEVEL)
        record.setdefault("semantic_level_selection", _SEMANTIC_LEVEL_SELECTION.to_metadata())
        record.setdefault("semantic_level", default_level.level)
        record.setdefault("level_reason", default_level.reason)
        record.setdefault("level_source", default_level.source)
        _SESSION_SKIPS[nodeid] = record


def _get_runtime_device_count(device_name):
    if device_name in ("cpu", "meta"):
        return 1
    try:
        from torchcts.core.device import get_device_module

        if device_name == "cuda" and torch.cuda.is_available():
            return max(torch.cuda.device_count(), 1)
        if device_name == "mps":
            return 1
        if device_name == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "device_count"):
            return max(torch.xpu.device_count(), 1)

        mod = get_device_module(device_name)
        if mod is not None and hasattr(mod, "device_count"):
            return max(int(mod.device_count()), 1)
    except Exception:
        pass
    return None

def pytest_addoption(parser):
    group = parser.getgroup("torchcts", "TorchCTS Options")
    group.addoption("--device", default="auto", help="Target device name (e.g. mps, cuda, auto)")
    group.addoption("--dtype", action="append", help="Override supported dtypes (can be specified multiple times)")
    group.addoption("--suite", choices=["opinfo", "operators", "training", "compiler", "device_api", "autograd", "memory", "custom", "dtypes", "strides", "workloads", "rng", "serialization", "errors", "stress", "multi_device", "adversarial", "generated"], help="Limit test collection to a specific suite")
    group.addoption("--memory-mode", default="balanced", choices=["conservative", "balanced", "performance"], help="Memory cleanup cadence")
    group.addoption("--max-device-memory", type=int, help="Cap maximum device memory allowed (MB)")
    group.addoption("--max-tensor-size", type=int, help="Cap maximum single tensor size allowed (MB)")
    group.addoption("--level", type=int, help="Run semantic test cases with semantic_level <= LEVEL (1-8)")
    group.addoption("--level-exact", type=int, help="Run only semantic test cases with semantic_level == LEVEL (1-8)")
    group.addoption("--level-range", help="Run only semantic test cases in inclusive MIN:MAX level range")
    group.addoption("--show-skips", action="store_true", help="Dry-run: print skips and exit")
    group.addoption("--report-skips", action="store_true", help="Include skip audit in report")
    group.addoption("--results-dir", default="./results", help="Directory to save JSON/Markdown results")
    group.addoption("--non-interactive", action="store_true", help="Error instead of prompting in auto device selection")
    group.addoption("--subprocess-per-test", action="store_true", help="Run each test in a separate subprocess for crash isolation")
    group.addoption("--subprocess-timeout", type=float, default=120.0, help="Seconds allowed for each subprocess-isolated test")
    group.addoption(
        "--known-segfault-policy",
        choices=["isolate", "off"],
        default=None,
        help="Isolate known backend segfault tests without skipping them; defaults to isolate",
    )
    group.addoption(
        "--known-segfault-audit",
        action="store_true",
        help="Collect tests, validate active known-segfault rules, print rule matches, and exit",
    )
    group.addoption(
        "--adaptive-isolation",
        choices=["auto", "off"],
        default="auto",
        help="Isolate tests with matching prior crash/hang evidence without skipping them",
    )
    group.addoption("--validation", action="store_true", help="Validate harness and CPU-compatible tests without probing an accelerator")

def load_manifest():
    manifest_py = os.path.join(os.getcwd(), "manifest.py")
    if os.path.exists(manifest_py):
        import importlib.util
        spec = importlib.util.spec_from_file_location("manifest", manifest_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "manifest", {})

    pyproject_toml = os.path.join(os.getcwd(), "pyproject.toml")
    if os.path.exists(pyproject_toml):
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                return {}
        try:
            with open(pyproject_toml, "rb") as f:
                data = tomllib.load(f)
                toml_manifest = data.get("tool", {}).get("torchcts", {}) or data.get("tool", {}).get("torch-cts", {}) or data.get("tool", {}).get("backend-validator", {})
                if "supported_dtypes" in toml_manifest:
                    resolved_dtypes = {}
                    for k, v in toml_manifest["supported_dtypes"].items():
                        dt = str_to_dtype(k)
                        if dt:
                            resolved_dtypes[dt] = v
                    toml_manifest["supported_dtypes"] = resolved_dtypes
                return toml_manifest
        except Exception:
            pass
    return {}

def get_required_capabilities(item):
    reqs = set()
    for m in item.iter_markers(name="requires"):
        for arg in m.args:
            reqs.add(arg)
    return reqs


def _marker_string_args(item, marker_name):
    values = []
    for marker in item.iter_markers(name=marker_name):
        for arg in marker.args:
            if isinstance(arg, str):
                values.append(arg)
    return values


def _cpu_contract_exempt_reason(item):
    markers = list(item.iter_markers(name="cpu_contract_exempt"))
    if not markers:
        return None
    marker = markers[0]
    if marker.args and isinstance(marker.args[0], str):
        return marker.args[0]
    reason = marker.kwargs.get("reason")
    return str(reason) if reason else "handwritten test intentionally has no dispatcher dtype contract"


def _contract_surface_exists(surface):
    if not surface:
        return False
    name = str(surface)
    if not name.startswith("aten::"):
        name = f"aten::{name}"
    contracts = load_dtype_contracts().get("contracts", {})
    return name in contracts


def _contract_surface_has_executable_evidence(surface):
    if not surface:
        return False
    name = str(surface)
    if not name.startswith("aten::"):
        name = f"aten::{name}"
    versions = load_dtype_contracts().get("contracts", {}).get(name, {})
    if not isinstance(versions, dict):
        return False
    for version_entry in versions.values():
        if not isinstance(version_entry, dict):
            continue
        if any(
            version_entry.get(bucket)
            for bucket in (
                "cpu_supported",
                "cpu_unsupported",
                "cpu_unknown",
                "cpu_pending",
                "oracle_supported",
            )
        ):
            return True
    return False


def _candidate_contract_surfaces(surface):
    if not surface:
        return ()
    text = str(surface)
    alias = _HANDWRITTEN_CONTRACT_ALIASES.get(text)
    if alias is not None:
        return alias

    base = text if text.startswith("aten::") else f"aten::{text}"
    dispatcher_part = base.removeprefix("aten::")
    overloadless = f"aten::{dispatcher_part.split('.', 1)[0]}" if "." in dispatcher_part else ""
    candidates = (
        base,
        overloadless,
        f"{base}.Tensor",
        f"{base}.out",
        f"{base}.int",
        f"{base}.dim",
        f"{base}.Scalar",
        f"aten::nn.functional.{text}" if not text.startswith("aten::") else "",
    )
    for candidate in candidates:
        if candidate and _contract_surface_has_executable_evidence(candidate):
            return (candidate,)
    for candidate in candidates:
        if candidate and _contract_surface_exists(candidate):
            return (candidate,)
    return (base,)


def _extend_contract_surfaces(target, surface):
    for candidate in _candidate_contract_surfaces(surface):
        if candidate and candidate not in target:
            target.append(candidate)


def _contract_surfaces_for_item(item):
    surfaces = []
    function_name = getattr(item, "originalname", None) or item.name.split("[", 1)[0]
    for surface in _HANDWRITTEN_FUNCTION_CONTRACT_ALIASES.get(function_name, ()):
        _extend_contract_surfaces(surfaces, surface)

    if hasattr(item, "callspec"):
        params = item.callspec.params
        for name, value in params.items():
            for surface in _HANDWRITTEN_PARAM_CONTRACT_ALIASES.get((name, str(value)), ()):
                _extend_contract_surfaces(surfaces, surface)
        if "op" in params:
            op_param = params["op"]
            _extend_contract_surfaces(surfaces, getattr(op_param, "name", str(op_param)))
        elif "op_name" in params:
            _extend_contract_surfaces(surfaces, params["op_name"])
        elif "inplace_op" in params:
            _extend_contract_surfaces(surfaces, params["inplace_op"])
        elif isinstance(params.get("entry"), dict):
            _extend_contract_surfaces(surfaces, params["entry"].get("name"))

    for surface in _marker_string_args(item, "cpu_contract"):
        _extend_contract_surfaces(surfaces, surface)
    for surface in _marker_string_args(item, "covers"):
        _extend_contract_surfaces(surfaces, surface)
    return surfaces


def _fixed_dtype_contract_gates_for_item(item):
    gates = []
    for marker in item.iter_markers(name="cpu_contract_dtype"):
        raw_args = list(marker.args)
        surface = marker.kwargs.get("surface") or marker.kwargs.get("dispatcher_name")
        if raw_args and isinstance(raw_args[0], str):
            surface = raw_args.pop(0)
        raw_dtypes = marker.kwargs.get("dtypes", marker.kwargs.get("dtype", None))
        if raw_dtypes is None:
            raw_dtypes = raw_args
        elif isinstance(raw_dtypes, (str, torch.dtype)):
            raw_dtypes = [raw_dtypes]
        for dtype_value in raw_dtypes or ():
            dtype = dtype_value if isinstance(dtype_value, torch.dtype) else str_to_dtype(str(dtype_value))
            if dtype is None and not str(dtype_value).startswith("torch."):
                dtype = str_to_dtype(f"torch.{dtype_value}")
            if surface and dtype is not None:
                gates.append((surface, dtype))
    return gates


def _fixed_dtype_contract_skip_for_item(item, supported_dtypes, op_name):
    for surface, dtype in _fixed_dtype_contract_gates_for_item(item):
        dtype_str = dtype_to_str(dtype)
        disposition = dtype_manifest_disposition(
            dtype,
            dtype_str,
            supported_dtypes,
            op_name or surface,
            dtype_label=f"{dtype_str} (fixed dtype)",
        )
        extra = {
            "dtype": dtype_str,
            "cpu_contract_fixed_dtype": True,
            "cpu_contract_fixed_surface": surface,
        }
        if not disposition.allowed:
            return disposition.skip_reason, disposition.detail, extra
        contract = contract_disposition(surface, dtype)
        extra.update(
            {
                "cpu_contract_status": contract.status,
                "source_expected": list(contract.source_expected),
                "source_probe_mismatches": list(contract.mismatches),
            }
        )
        if not contract.allowed and contract.status != "not_recorded":
            return contract.skip_reason or "cpu_contract_unknown", contract.detail, extra
    return None, "", {}


def _dtype_contract_skip_for_surfaces(surfaces, dtype, *, input_condition="clean"):
    for surface in surfaces:
        contract = contract_disposition(surface, dtype, input_condition=input_condition)
        if contract.status == NOT_RECORDED:
            continue
        if not contract.allowed:
            return (
                contract.skip_reason or "cpu_contract_unknown",
                contract.detail,
            )
    return None, ""


def _requires_dtype_contract(item):
    if not hasattr(item, "callspec"):
        return False
    return any(name in item.callspec.params for name in _DTYPE_PARAM_NAMES)


def _semantic_level_for_item(item):
    if hasattr(item, "callspec") and isinstance(item.callspec.params.get("entry"), dict):
        entry = item.callspec.params["entry"]
        level = entry.get("semantic_level")
        if level is not None:
            level = validate_semantic_level(level)
            return {
                "semantic_level": level,
                "level_reason": entry.get("level_reason", "Generated coverage case semantic level."),
                "level_source": entry.get("level_source", "generated_case"),
            }
        if entry.get("generated") or entry.get("surface_kind"):
            default = generated_level_for_entry(entry)
            return {
                "semantic_level": default.level,
                "level_reason": default.reason,
                "level_source": default.source,
            }
    if hasattr(item, "callspec") and "semantic_level" in item.callspec.params:
        level = validate_semantic_level(item.callspec.params["semantic_level"])
        reason = item.callspec.params.get("level_reason")
        source = item.callspec.params.get("level_source")
        return {
            "semantic_level": level,
            "level_reason": str(reason) if reason else "Declared by parametrized semantic_level case.",
            "level_source": str(source) if source else "param_case",
        }

    markers = list(item.iter_markers(name="semantic_level"))
    if markers:
        marker = markers[0]
        level = marker_value_to_level(marker.args, marker.kwargs)
        reason = marker.kwargs.get("reason")
        return {
            "semantic_level": level,
            "level_reason": str(reason) if reason else "Declared by pytest semantic_level marker.",
            "level_source": "test_marker",
        }

    suite = _canonical_suite_for_item(item)
    default = suite_default_level(suite)
    return {
        "semantic_level": default.level,
        "level_reason": default.reason,
        "level_source": default.source,
    }

def _apply_resource_limit_overrides(manifest, cli_max_mem=None, cli_max_tensor=None):
    limits = manifest.setdefault("resource_limits", {})
    if cli_max_mem is not None:
        limits["max_device_memory_mb"] = cli_max_mem
    if cli_max_tensor is not None:
        limits["max_tensor_size_mb"] = cli_max_tensor
    return limits


def _normalize_cli_dtype_filter(cli_dtypes):
    effective = {}
    labels = []
    for raw_name in cli_dtypes or []:
        dtype_name = str(raw_name).strip()
        dtype = str_to_dtype(dtype_name)
        if dtype is None and not dtype_name.startswith("torch."):
            dtype = str_to_dtype(f"torch.{dtype_name}")
        if dtype is None:
            raise pytest.UsageError(f"Unknown --dtype value: {raw_name!r}")
        if dtype in effective:
            continue
        effective[dtype] = True
        labels.append(dtype_to_str(dtype))
    return effective, labels


def _serialize_supported_dtype_declarations(supported_dtypes):
    declarations = []
    for dtype_key, value in (supported_dtypes or {}).items():
        if isinstance(dtype_key, torch.dtype):
            dtype_label = dtype_to_str(dtype_key)
        else:
            dtype_label = str(dtype_key)
        declarations.append({"dtype": dtype_label, "value": value})
    return declarations


def _apply_cli_dtype_filter(manifest, cli_dtypes):
    if not cli_dtypes:
        return []
    effective, labels = _normalize_cli_dtype_filter(cli_dtypes)
    manifest["_declared_supported_dtypes"] = _serialize_supported_dtype_declarations(
        manifest.get("supported_dtypes", {})
    )
    manifest["supported_dtypes"] = effective
    manifest["dtype_filter"] = labels
    return labels


def pytest_configure(config):
    global _MANIFEST, _DEVICE_NAME, _HARDWARE_KEY, _RESULTS_DIR, _START_TIME, _SHOW_SKIPS, _REPORT_SKIPS
    global _SUBPROCESS_MODE, _MEMORY_MODE, _CLEANUP_THRESHOLD, _MAX_DEVICE_MEM, _MAX_TENSOR_SIZE, _BASELINE_RESULTS
    global _COLLECT_ONLY, _ARTIFACT_WRITES_ENABLED, _ACTUAL_DEVICE_COUNT, _REQUESTED_SEMANTIC_LEVEL, _SEMANTIC_LEVEL_SELECTION
    global _KNOWN_SEGFAULT_POLICY, _KNOWN_SEGFAULTS_ACTIVE, _KNOWN_SEGFAULT_WARNINGS
    global _KNOWN_SEGFAULT_AUDIT
    global _ADAPTIVE_ISOLATION_MODE, _ADAPTIVE_ISOLATION_LOAD, _ADAPTIVE_ISOLATION_ACTIVE
    global _ADAPTIVE_ISOLATION_REJECTED, _ADAPTIVE_ISOLATION_WARNINGS, _SESSION_COMPLETED
    global _SESSION_PROBE_FAILURES, _SESSION_PROBE_FAILURE_KEYS

    # Register custom markers
    config.addinivalue_line("markers", "gate: backend registration gate tests — run first")
    config.addinivalue_line("markers", "smoke: smoke tests only")
    config.addinivalue_line("markers", "medium: medium tests")
    config.addinivalue_line("markers", "opinfo: OpInfo breadth tests")
    config.addinivalue_line("markers", "workload: real-world workloads")
    config.addinivalue_line("markers", "stress: stress tests")
    config.addinivalue_line("markers", "requires(capability): required capabilities")
    config.addinivalue_line("markers", "adversarial: adversarial test suite")
    config.addinivalue_line("markers", "covers(dispatcher_name, surface=None): dispatcher overload covered by this test")
    config.addinivalue_line("markers", "covers_category(category): coverage category covered by this test")
    config.addinivalue_line("markers", "cpu_contract(dispatcher_name): dispatcher overload whose CPU dtype contract gates this handwritten test")
    config.addinivalue_line("markers", "cpu_contract_exempt(reason=None): handwritten dtype-parametrized test has no single dispatcher dtype contract")
    config.addinivalue_line("markers", "cpu_contract_dtype(dispatcher_name, dtype): hardcoded backend dtype contract gate for handwritten tests")
    config.addinivalue_line("markers", "generated: generated coverage tests")
    config.addinivalue_line("markers", "semantic_level(level, reason=None): semantic priority level from 1 to 8")

    # 1. Load manifest
    _MANIFEST = load_manifest()
    try:
        _SEMANTIC_LEVEL_SELECTION = normalize_level_selection(
            _MANIFEST,
            cli_level=config.getoption("--level"),
            cli_level_exact=config.getoption("--level-exact"),
            cli_level_range=config.getoption("--level-range"),
        )
        _REQUESTED_SEMANTIC_LEVEL = _SEMANTIC_LEVEL_SELECTION.max_level
    except SemanticLevelError as exc:
        pytest.exit(str(exc), returncode=1)
    _MANIFEST["semantic_level"] = _REQUESTED_SEMANTIC_LEVEL
    _MANIFEST["semantic_level_selection"] = _SEMANTIC_LEVEL_SELECTION.to_metadata()
    
    # 2. Command line overrides
    cli_device = config.getoption("--device")
    if cli_device != "auto":
        _MANIFEST["device_name"] = cli_device
        
    non_interactive = config.getoption("--non-interactive")
    backend_import = _MANIFEST.get("backend_import")
    is_validation = config.getoption("--validation")
    _COLLECT_ONLY = bool(getattr(config.option, "collectonly", False))
    
    if is_validation:
        _DEVICE_NAME = "cpu"
        _MANIFEST["capabilities"] = {
            "inference": True,
            "training": True,
            "serialization": True,
            "rng": True,
            "device_generator": True,
            "rng_distributions": True,
            "double_backward": True,
            "gradcheck": True,
            "gradient_checkpointing": True,
            "autocast": True,
            "fused_optimizer": True,
            "dataloader": True,
            "module_hooks": True,
            "channels_last": True,
            "sparse": True,
            "nested": False,  # nested SDPA requires accelerator (CPU fallback has different semantics)
            "named_tensor": True,
            "foreach": True,
            "fp8": True,
            "quantized_container_plumbing": True,
            "custom_quantized_decode": False,
            "compile": True,
            "pinned_memory": True,
            "deterministic": True,
            "native_quantization": True,
            "device_api": False,
            "guard_alloc": False,
            "streams": False,
            "events": False,
            "multi_device": False,
            "ieee754": True,
        }
        _MANIFEST["supported_dtypes"] = {
            torch.float32: True,
            torch.float64: True,
            torch.float16: True,
            torch.bfloat16: True,
            torch.int64: True,
            torch.int32: True,
            torch.int16: True,
            torch.int8: True,
            torch.uint8: True,
            torch.bool: True,
            torch.complex64: True,
            torch.complex128: True,
        }
        _MANIFEST["skip_ops"] = []
        from torchcts.core.quantized_decoders import KNOWN_CONTAINER_FORMATS
        _MANIFEST["supported_container_formats"] = {
            name: True for name in sorted(KNOWN_CONTAINER_FORMATS)
        }
        _MANIFEST["custom_container_decoders"] = {}
        _MANIFEST["device_count"] = 1
        _MANIFEST.setdefault("ieee754_seed", 67)
        _MANIFEST.setdefault("max_samples", 10)
        _MANIFEST.setdefault("max_samples_ieee754", 3)
    else:
        if _COLLECT_ONLY or config.getoption("--show-skips"):
            configured_name = _MANIFEST.get("device_name", "auto")
            if configured_name == "auto":
                _DEVICE_NAME = "cpu"
            else:
                _DEVICE_NAME = configured_name
        else:
            try:
                _DEVICE_NAME = get_device_backend(
                    _MANIFEST.get("device_name", "auto"),
                    backend_import,
                    non_interactive
                )
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                pytest.exit(str(e))

    _apply_cli_dtype_filter(_MANIFEST, config.getoption("--dtype"))

    declared_device_count = _MANIFEST.get("device_count", 1)
    runtime_device_count = None if _COLLECT_ONLY else _get_runtime_device_count(_DEVICE_NAME)
    if runtime_device_count is not None:
        _ACTUAL_DEVICE_COUNT = runtime_device_count
        _MANIFEST["effective_device_count"] = runtime_device_count
        _MANIFEST["_declared_device_count"] = declared_device_count
        if runtime_device_count < declared_device_count:
            print(
                f"Warning: manifest declares device_count={declared_device_count}, "
                f"but runtime exposes {runtime_device_count}; gating multi-device tests accordingly.",
                file=sys.stderr,
            )
    else:
        _ACTUAL_DEVICE_COUNT = declared_device_count
        _MANIFEST["effective_device_count"] = declared_device_count

    # Dynamic Hardware Resolution
    if not is_validation:
        hw_config = _MANIFEST.setdefault("hardware", {})
        
        # System memory auto-detection
        if hw_config.get("system_memory_gb") == "auto":
            try:
                import psutil
                hw_config["system_memory_gb"] = int(psutil.virtual_memory().total / (1024**3))
            except Exception:
                hw_config["system_memory_gb"] = 8  # fallback
                
        # Device memory auto-detection
        if hw_config.get("device_memory_gb") == "auto":
            from torchcts.core.device import get_device_total_memory
            # Detect for each available device
            detected_mems = []
            for dev_idx in range(_ACTUAL_DEVICE_COUNT):
                try:
                    mem_bytes = get_device_total_memory(_DEVICE_NAME, dev_idx)
                    if mem_bytes is not None:
                        detected_mems.append(int(mem_bytes / (1024**3)))
                    else:
                        detected_mems.append(4)  # fallback
                except Exception:
                    detected_mems.append(4)  # fallback
            hw_config["device_memory_gb"] = detected_mems

    _HARDWARE_KEY = get_hardware_key(_DEVICE_NAME, _MANIFEST)
    _RESULTS_DIR = config.getoption("--results-dir")
    os.environ["TORCHCTS_RESULTS_DIR"] = str(_RESULTS_DIR)
    os.environ["TORCHCTS_HARDWARE_KEY"] = str(_HARDWARE_KEY)
    os.environ["TORCHCTS_DEVICE_NAME"] = str(_DEVICE_NAME)
    os.environ["TORCHCTS_PYTORCH_VERSION"] = str(torch.__version__)

    _SHOW_SKIPS = config.getoption("--show-skips")
    _REPORT_SKIPS = config.getoption("--report-skips")
    _SUBPROCESS_MODE = config.getoption("--subprocess-per-test")
    _ADAPTIVE_ISOLATION_MODE = config.getoption("--adaptive-isolation")
    _ADAPTIVE_ISOLATION_LOAD = None
    _ADAPTIVE_ISOLATION_ACTIVE = {}
    _ADAPTIVE_ISOLATION_REJECTED = []
    _ADAPTIVE_ISOLATION_WARNINGS = []
    _SESSION_COMPLETED = False
    _SESSION_PROBE_FAILURES = []
    _SESSION_PROBE_FAILURE_KEYS = set()
    policy_option = config.getoption("--known-segfault-policy")
    _KNOWN_SEGFAULT_POLICY = policy_option or "isolate"
    _KNOWN_SEGFAULT_AUDIT = bool(config.getoption("--known-segfault-audit"))
    _KNOWN_SEGFAULTS_ACTIVE = []
    _KNOWN_SEGFAULT_WARNINGS = []

    # Diagnostic probes only. Probe failures never decide conformance outcomes.
    if not is_validation and not (_COLLECT_ONLY or _SHOW_SKIPS or _KNOWN_SEGFAULT_AUDIT):
        supported_dtypes = _MANIFEST.setdefault("supported_dtypes", {})
        _apply_declared_dtype_probes(
            supported_dtypes,
            _DEVICE_NAME,
        )

        # Skip in xdist workers: concurrent subprocess probes cause hangs.
        if not _IS_XDIST_WORKER:
            caps = _MANIFEST.setdefault("capabilities", {})
            _apply_declared_capability_probes(
                caps,
                _DEVICE_NAME,
                backend_import=backend_import,
            )

            # Hard prerequisite: inference must be True
            if not caps.get("inference", True):
                pytest.exit(
                    "FATAL: capability 'inference' is False. A backend that cannot "
                    "perform inference cannot be tested.",
                    returncode=1,
                )

    # 3. Hardware details
    # Build effective runtime unsupported-pattern classification based on device.
    global _RUNTIME_UNSUPPORTED_PATTERNS
    _RUNTIME_UNSUPPORTED_PATTERNS = list(_RUNTIME_UNSUPPORTED_PATTERNS_UNIVERSAL)
    if _DEVICE_NAME == "mps":
        _RUNTIME_UNSUPPORTED_PATTERNS.extend(_RUNTIME_UNSUPPORTED_PATTERNS_MPS)
    if _KNOWN_SEGFAULT_POLICY == "isolate":
        try:
            from torchcts.core.known_segfaults import (
                active_known_segfaults,
                expired_known_segfault_warnings,
                load_known_segfaults,
            )

            known_entries = load_known_segfaults(os.getcwd())
            _KNOWN_SEGFAULTS_ACTIVE = active_known_segfaults(
                known_entries,
                backend=_DEVICE_NAME,
                torch_version=torch.__version__,
                hardware_key=_HARDWARE_KEY,
            )
            _KNOWN_SEGFAULT_WARNINGS = expired_known_segfault_warnings(_KNOWN_SEGFAULTS_ACTIVE)
            if _KNOWN_SEGFAULT_WARNINGS and not _is_child_process():
                for warning in _KNOWN_SEGFAULT_WARNINGS:
                    print(f"Warning: {warning}", file=sys.stderr)
        except Exception as exc:
            pytest.exit(f"Invalid known segfault ledger: {exc}", returncode=1)
    _ARTIFACT_WRITES_ENABLED = not (_COLLECT_ONLY or _SHOW_SKIPS or _KNOWN_SEGFAULT_AUDIT)
    if _ARTIFACT_WRITES_ENABLED:
        os.makedirs(_RESULTS_DIR, exist_ok=True)
    
    # Memory configurations
    _MEMORY_MODE = config.getoption("--memory-mode")
    _CLEANUP_THRESHOLD = int(_MANIFEST.get("resource_limits", {}).get("cleanup_threshold_pct", 80))
    
    limits = _apply_resource_limit_overrides(
        _MANIFEST,
        cli_max_mem=config.getoption("--max-device-memory"),
        cli_max_tensor=config.getoption("--max-tensor-size"),
    )
    _MAX_DEVICE_MEM = limits.get("max_device_memory_mb")
    _MAX_TENSOR_SIZE = limits.get("max_tensor_size_mb")

    # Load baseline results for regression detection
    _BASELINE_RESULTS = {}
    if _ARTIFACT_WRITES_ENABLED:
        # Merge orphaned xdist worker files from a previously killed parallel run.
        # Workers flush after every test, so these contain all results up to the hang.
        if not _IS_XDIST_WORKER:
            orphan_pattern = os.path.join(_RESULTS_DIR, f"{_HARDWARE_KEY}_latest.gw*.json")
            if glob.glob(orphan_pattern):
                print("Recovering partial result file(s) from a previous parallel run...")
                count = _merge_xdist_worker_files(_RESULTS_DIR, _HARDWARE_KEY)
                print(f"  Merged {count} result(s) into {_HARDWARE_KEY}_latest.json")

        latest_json_path = os.path.join(_RESULTS_DIR, f"{_HARDWARE_KEY}_latest.json")
        if os.path.exists(latest_json_path):
            try:
                with open(latest_json_path, "r", encoding="utf-8") as f:
                    _BASELINE_RESULTS = json.load(f).get("results", {})
            except Exception:
                pass

        if _ADAPTIVE_ISOLATION_MODE == "auto" and not _is_child_process():
            try:
                from torchcts.core.adaptive_isolation import load_adaptive_isolation

                _ADAPTIVE_ISOLATION_LOAD = load_adaptive_isolation(
                    _RESULTS_DIR,
                    hardware_key=_HARDWARE_KEY,
                    device_name=_DEVICE_NAME,
                    torch_version=torch.__version__,
                )
                _ADAPTIVE_ISOLATION_WARNINGS = list(_ADAPTIVE_ISOLATION_LOAD.warnings)
                if _ADAPTIVE_ISOLATION_WARNINGS and not _IS_XDIST_WORKER:
                    for warning in _ADAPTIVE_ISOLATION_WARNINGS:
                        print(f"Warning: {warning}", file=sys.stderr)
            except Exception as exc:
                _ADAPTIVE_ISOLATION_LOAD = None
                _ADAPTIVE_ISOLATION_WARNINGS = [f"adaptive isolation disabled: {exc}"]
                if not _IS_XDIST_WORKER:
                    print(f"Warning: {_ADAPTIVE_ISOLATION_WARNINGS[0]}", file=sys.stderr)

    # Start timing
    _START_TIME = time.time()

    # Open per-test run log for hang diagnosis
    global _RUN_LOG_FH
    if _ARTIFACT_WRITES_ENABLED:
        log_suffix = f".{_XDIST_WORKER_ID}" if _IS_XDIST_WORKER else ""
        run_log_path = os.path.join(_RESULTS_DIR, f"{_HARDWARE_KEY}_runlog{log_suffix}.txt")
        _RUN_LOG_FH = open(run_log_path, "w", encoding="utf-8")
        print(f"  Run log: {run_log_path}")

    # Append custom_test_dirs from manifest to pytest collection paths
    custom_dirs = _MANIFEST.get("custom_test_dirs", [])
    for cdir in custom_dirs:
        abs_cdir = os.path.abspath(cdir)
        if os.path.isdir(abs_cdir):
            if abs_cdir not in config.args:
                config.args.append(abs_cdir)
        else:
            print(
                f"Warning: custom_test_dirs entry '{cdir}' is not a valid directory, skipping.",
                file=sys.stderr,
            )
    
    # 4. Prepare shared test data on device
    if not _SHOW_SKIPS and not _SUBPROCESS_MODE and not _COLLECT_ONLY:
        try:
            refresh_shared_data(_DEVICE_NAME)
        except Exception as e:
            # CPU target or device not ready
            pass

    # 5. Disable sparse tensor invariant checks to opt out of warnings and overhead
    try:
        torch.sparse.check_sparse_tensor_invariants.disable()
    except Exception:
        pass

def pytest_collection_modifyitems(session, config, items):
    global _MANIFEST, _DEVICE_NAME, _SESSION_SKIPS, _SHOW_SKIPS
    is_validation = config.getoption("--validation")
    site_stats_records = []
    site_stats_enabled = _site_stats_collection_enabled()
    
    # Optional CLI suite filter
    suite = config.getoption("--suite")
    if suite:
        filtered_items = []
        deselected_items = []
        for item in items:
            # Gate tests always run, regardless of suite filter
            if item.get_closest_marker("gate"):
                filtered_items.append(item)
                continue

            filepath = str(item.fspath).replace("\\", "/")
            is_match = False
            if suite == "opinfo":
                is_match = "opinfo/" in filepath
            elif suite == "operators":
                is_match = "operators/" in filepath
            elif suite == "training":
                is_match = "training/" in filepath
            elif suite == "compiler":
                is_match = "compiler/" in filepath
            elif suite == "device_api":
                is_match = "device_api/" in filepath
            elif suite == "autograd":
                is_match = "autograd/" in filepath
            elif suite == "memory":
                is_match = "memory/" in filepath
            elif suite == "dtypes":
                is_match = "dtypes/" in filepath
            elif suite == "strides":
                is_match = "strides/" in filepath
            elif suite == "workloads":
                is_match = "workloads/" in filepath
            elif suite == "rng":
                is_match = "rng/" in filepath
            elif suite == "serialization":
                is_match = "serialization/" in filepath
            elif suite == "errors":
                is_match = "errors/" in filepath
            elif suite == "stress":
                is_match = "stress/" in filepath
            elif suite == "multi_device":
                is_match = "multi_device/" in filepath
            elif suite == "adversarial":
                is_match = "test_adversarial.py" in filepath
            elif suite == "generated":
                is_match = "generated/" in filepath
            elif suite == "custom":
                standard_dirs = ["opinfo/", "operators/", "training/", "compiler/", "device_api/", "autograd/", "memory/", "dtypes/", "strides/", "workloads/", "rng/", "serialization/", "errors/", "stress/", "multi_device/", "generated/"]
                is_match = not any(d in filepath for d in standard_dirs)
            
            if is_match:
                filtered_items.append(item)
            else:
                deselected_items.append(item)
                if site_stats_enabled:
                    site_stats_records.append(_site_stats_collection_record(
                        item,
                        "structured_deselected",
                        skip_reason="suite_filter",
                        skip_detail=f"excluded by --suite {suite}",
                    ))
        
        config.hook.pytest_deselected(items=deselected_items)
        items[:] = filtered_items

    # Read capabilities and dtypes from _MANIFEST (already configured by pytest_configure)
    caps = _MANIFEST.get("capabilities", {})
    supported_dtypes = _MANIFEST.get("supported_dtypes", {})
    skip_ops = set(_MANIFEST.get("skip_ops", []))
    device_count = _MANIFEST.get("effective_device_count", _MANIFEST.get("device_count", 1))

    keep_items = []
    dtype_deselected_items = []
    selection_deselected_items = []
    
    for item in items:
        skip_reason = None
        detail = ""
        dtype_manifest_skip = False
        skip_record_extra = {}
        
        # Determine ATen op name
        op_name = None
        if hasattr(item, "callspec"):
            if "op" in item.callspec.params:
                op_param = item.callspec.params["op"]
                op_name = getattr(op_param, "name", str(op_param))
            elif "op_name" in item.callspec.params:
                op_name = item.callspec.params["op_name"]
            elif isinstance(item.callspec.params.get("entry"), dict):
                op_name = item.callspec.params["entry"].get("name")
        contract_surfaces = _contract_surfaces_for_item(item)
        if op_name is None and contract_surfaces:
            op_name = contract_surfaces[0]
        contract_exempt_reason = _cpu_contract_exempt_reason(item)

        if (
            is_validation
            and _requires_dtype_contract(item)
            and not contract_surfaces
            and contract_exempt_reason is None
            and _canonical_suite_for_item(item) != "selftest"
        ):
            pytest.exit(
                f"{item.nodeid} has dtype-parametrized handwritten coverage but no "
                "op/op_name/covers/cpu_contract surface or cpu_contract_exempt marker",
                returncode=1,
            )

        # 1. Dtype check. This is a run filter, so it must happen before
        # capability skips; --dtype should remove non-selected dtype items even
        # if another static skip would also apply.
        if hasattr(item, "callspec"):
            for dtype_param in ("dtype", "autocast_dtype"):
                if dtype_param not in item.callspec.params:
                    continue
                dt = item.callspec.params[dtype_param]
                dt_str = dtype_to_str(dt)
                dtype_label = dt_str if dtype_param == "dtype" else f"{dt_str} ({dtype_param})"
                disposition = dtype_manifest_disposition(
                    dt,
                    dt_str,
                    supported_dtypes,
                    op_name,
                    dtype_label=dtype_label,
                )
                if not disposition.allowed:
                    skip_reason = disposition.skip_reason
                    detail = disposition.detail
                    dtype_manifest_skip = skip_reason in _DTYPE_MANIFEST_SKIP_REASONS
                elif contract_surfaces:
                    input_condition = item.callspec.params.get("input_condition", "clean")
                    generated_entry = item.callspec.params.get("entry")
                    if isinstance(generated_entry, dict) and contract_surfaces == [generated_entry.get("name")]:
                        contract = contract_disposition(contract_surfaces[0], dt, input_condition=input_condition)
                        if not contract.allowed and contract.status == "not_recorded":
                            from torchcts.generated.coverage_helpers import probe_generated_clean_cpu_contract

                            probe = probe_generated_clean_cpu_contract(
                                generated_entry,
                                dt,
                                _MANIFEST,
                                enforce_recorded_contract=False,
                            )
                            if probe["status"] != "supported":
                                if probe["status"] == "unsupported":
                                    skip_reason = "cpu_contract_unsupported"
                                elif probe["status"] == "pending":
                                    skip_reason = "cpu_contract_pending"
                                else:
                                    skip_reason = "cpu_contract_unknown"
                                detail = probe["detail"]
                                dtype_manifest_skip = skip_reason in _DTYPE_CONTRACT_SKIP_REASONS
                        elif not contract.allowed:
                            skip_reason = contract.skip_reason
                            detail = contract.detail
                            dtype_manifest_skip = skip_reason in _DTYPE_CONTRACT_SKIP_REASONS
                    else:
                        skip_reason, detail = _dtype_contract_skip_for_surfaces(
                            contract_surfaces,
                            dt,
                            input_condition=input_condition,
                        )
                        dtype_manifest_skip = skip_reason in _DTYPE_CONTRACT_SKIP_REASONS
                break

        # 2. src_dtype / dst_dtype check (e.g. test_copy_cast)
        if not skip_reason and hasattr(item, "callspec"):
            for dtype_param in ("src_dtype", "dst_dtype"):
                if dtype_param in item.callspec.params:
                    dt = item.callspec.params[dtype_param]
                    dt_str = dtype_to_str(dt)
                    disposition = dtype_manifest_disposition(
                        dt,
                        dt_str,
                        supported_dtypes,
                        op_name,
                        dtype_label=f"{dt_str} ({dtype_param})",
                    )
                    if not disposition.allowed:
                        skip_reason = disposition.skip_reason
                        detail = disposition.detail
                        dtype_manifest_skip = skip_reason in _DTYPE_MANIFEST_SKIP_REASONS
                        break
                    elif contract_surfaces:
                        skip_reason, detail = _dtype_contract_skip_for_surfaces(contract_surfaces, dt)
                        if skip_reason:
                            dtype_manifest_skip = skip_reason in _DTYPE_CONTRACT_SKIP_REASONS
                            break

        # 3. Fixed hardcoded dtype contract gates
        if not skip_reason:
            skip_reason, detail, skip_record_extra = _fixed_dtype_contract_skip_for_item(
                item,
                supported_dtypes,
                op_name,
            )
            if skip_reason:
                dtype_manifest_skip = skip_reason in (_DTYPE_MANIFEST_SKIP_REASONS | _DTYPE_CONTRACT_SKIP_REASONS)

        # 4. Capability check
        req_caps = get_required_capabilities(item)
        missing_caps = [c for c in req_caps if not caps.get(c, False) and c != "multi_device"]
        if not skip_reason and missing_caps:
            skip_reason = "capability_not_declared"
            detail = f"requires capabilities: {', '.join(missing_caps)}"
        elif not skip_reason and "multi_device" in req_caps and device_count < 2:
            skip_reason = "device_count"
            declared = _MANIFEST.get("device_count", device_count)
            detail = (
                f"requires device_count>=2, runtime exposes {device_count}"
                if declared == device_count
                else f"requires device_count>=2, manifest declares {declared} but runtime exposes {device_count}"
            )

        # 5. Op exclusions
        if not skip_reason and op_name and op_name in skip_ops:
            skip_reason = "op_excluded"
            detail = f"{op_name} is in skip_ops list"

        # 6. CPU device cannot run cross-device or device-module tests
        if not skip_reason and _DEVICE_NAME == "cpu":
            filepath = str(item.fspath)
            test_name = item.name
            # Cross-device error tests make no sense on CPU
            if "test_error_handling_cross_device" in test_name:
                skip_reason = "cpu_not_applicable"
                detail = "cross-device error checks not applicable on CPU"
            # Device module method/memory tests need a real device
            elif "test_device_module_methods" in test_name or "test_device_memory_query" in test_name:
                skip_reason = "cpu_not_applicable"
                detail = "device module tests not applicable on CPU"

        # 7. Device module availability check
        if not skip_reason and hasattr(item, "callspec"):
            test_name = item.name
            if "test_device_module_methods" in test_name or "test_device_memory_query" in test_name:
                from torchcts.core.device import get_device_module
                if get_device_module(_DEVICE_NAME) is None:
                    skip_reason = "no_device_module"
                    detail = f"No custom device module found for torch.{_DEVICE_NAME}"

        # 8. set_device support check
        if not skip_reason:
            test_name = item.name
            if "test_set_device_context" in test_name:
                _mod = torch.cuda if _DEVICE_NAME == "cuda" else getattr(torch, _DEVICE_NAME, None)
                if _mod is None or not hasattr(_mod, "set_device"):
                    skip_reason = "set_device_not_supported"
                    detail = f"Device module for {_DEVICE_NAME} does not support set_device"

        # 9. OOM recovery manifest check
        if not skip_reason:
            test_name = item.name
            if "test_oom_recovery" in test_name:
                hw_config = _MANIFEST.get("hardware", {})
                if not hw_config.get("oom_recoverable", True):
                    skip_reason = "oom_not_recoverable"
                    detail = "OOM recovery not marked as recoverable in manifest"

        # 10. float64 required for gradcheck
        if not skip_reason:
            test_name = item.name
            if "test_gradcheck" in test_name:
                disposition = dtype_manifest_disposition(
                    torch.float64,
                    "torch.float64",
                    supported_dtypes,
                    op_name,
                    dtype_label="torch.float64 (required for gradcheck)",
                )
                if not disposition.allowed:
                    skip_reason = disposition.skip_reason
                    detail = disposition.detail
                    dtype_manifest_skip = skip_reason in _DTYPE_MANIFEST_SKIP_REASONS
                elif contract_surfaces:
                    skip_reason, detail = _dtype_contract_skip_for_surfaces(contract_surfaces, torch.float64)
                    if skip_reason:
                        dtype_manifest_skip = skip_reason in _DTYPE_CONTRACT_SKIP_REASONS

        # 11. MPS index_reduce NaN hang workaround
        if not skip_reason and _DEVICE_NAME == "mps" and op_name == "index_reduce":
            if hasattr(item, "callspec") and item.callspec.params.get("input_condition") == "has_nan":
                skip_reason = "framework_bug"
                detail = "index_reduce hangs infinitely on MPS when input/destination contains NaN (CAS loop GPU thread deadlock)"

        # 12. Semantic test level
        if not skip_reason:
            try:
                semantic = _semantic_level_for_item(item)
            except SemanticLevelError as exc:
                pytest.exit(f"Invalid semantic level metadata on {item.nodeid}: {exc}", returncode=1)
            if not _SEMANTIC_LEVEL_SELECTION.contains(semantic["semantic_level"]):
                if (
                    _SEMANTIC_LEVEL_SELECTION.mode == "cumulative"
                    and semantic["semantic_level"] > _SEMANTIC_LEVEL_SELECTION.max_level
                ):
                    skip_reason = "semantic_level_gt_requested"
                    detail = (
                        f"semantic_level={semantic['semantic_level']} exceeds requested "
                        f"level {_REQUESTED_SEMANTIC_LEVEL}"
                    )
                else:
                    skip_reason = "semantic_level_out_of_range"
                    detail = (
                        f"semantic_level={semantic['semantic_level']} is outside "
                        f"{_SEMANTIC_LEVEL_SELECTION.label}"
                    )

        if skip_reason:
            _SESSION_SKIPS[item.nodeid] = _skip_record_for_item(item, skip_reason, detail, skip_record_extra)
            if dtype_manifest_skip:
                dtype_deselected_items.append(item)
                if site_stats_enabled:
                    site_stats_records.append(_site_stats_collection_record(
                        item,
                        "structured_deselected",
                        skip_reason=skip_reason,
                        skip_detail=detail,
                    ))
            elif skip_reason in _SEMANTIC_SELECTION_SKIP_REASONS:
                selection_deselected_items.append(item)
                if site_stats_enabled:
                    site_stats_records.append(_site_stats_collection_record(
                        item,
                        "structured_deselected",
                        skip_reason=skip_reason,
                        skip_detail=detail,
                    ))
            else:
                item.add_marker(pytest.mark.skip(reason=detail))
                keep_items.append(item)
                if site_stats_enabled:
                    site_stats_records.append(_site_stats_collection_record(
                        item,
                        "pytest_skip_marked",
                        skip_reason=skip_reason,
                        skip_detail=detail,
                    ))
        else:
            keep_items.append(item)
            if site_stats_enabled:
                site_stats_records.append(_site_stats_collection_record(item, "executable"))

    _merge_pending_manifest_skips(include_opinfo=config.getoption("--suite") in (None, "opinfo"))
    deselected_items = dtype_deselected_items + selection_deselected_items
    if deselected_items:
        config.hook.pytest_deselected(items=deselected_items)
    if site_stats_enabled:
        _write_site_stats_collection_records(config, site_stats_records)

    known_segfault_descriptors = []
    if _KNOWN_SEGFAULT_POLICY == "isolate" and _KNOWN_SEGFAULTS_ACTIVE:
        known_segfault_descriptors = _validate_known_segfault_collection(
            config,
            _KNOWN_SEGFAULTS_ACTIVE,
            keep_items,
        )
        if _KNOWN_SEGFAULT_AUDIT:
            _print_known_segfault_audit(config, _KNOWN_SEGFAULTS_ACTIVE, known_segfault_descriptors)
            pytest.exit("Known segfault audit complete", returncode=0)
    elif _KNOWN_SEGFAULT_AUDIT:
        print("\nKnown segfault audit: 0 active rule(s)")
        pytest.exit("Known segfault audit complete", returncode=0)

    _finalize_adaptive_isolation_for_collection(config, keep_items)

    if not _SHOW_SKIPS:
        # Reorder: gate tests run first
        gate_items = [i for i in keep_items if i.get_closest_marker("gate")]
        non_gate_items = [i for i in keep_items if not i.get_closest_marker("gate")]
        items[:] = gate_items + non_gate_items

    if _SHOW_SKIPS:
        # Print audit report and exit
        print(f"\n  SKIP AUDIT ({len(_SESSION_SKIPS)} skipped)")
        print("  " + "─" * 25)
        
        # Group by reason
        reasons = {}
        for nid, r in _SESSION_SKIPS.items():
            reasons[r["skip_reason"]] = reasons.get(r["skip_reason"], 0) + 1
        for k, v in reasons.items():
            print(f"    {k:<25}: {v}")
            
        print("\n  Full skip list:")
        print("  " + "─" * 15)
        for nid, r in _SESSION_SKIPS.items():
            # Truncate long nodeids
            short_nid = nid.split("/")[-1]
            print(f"  {short_nid:<60} {r['skip_reason']:<25} {r['detail']}")
            
        # Empty items to stop test run
        items.clear()

def flush_results_to_disk():
    global _SESSION_RESULTS, _SESSION_SKIPS, _START_TIME, _DEVICE_NAME, _HARDWARE_KEY, _RESULTS_DIR
    global _ARTIFACT_WRITES_ENABLED, _SESSION_COMPLETED, _SESSION_PROBE_FAILURES

    if not _ARTIFACT_WRITES_ENABLED:
        return
    
    elapsed = time.time() - _START_TIME
    data = {
        "metadata": {
            "device_name": _DEVICE_NAME,
            "hardware_key": _HARDWARE_KEY,
            "pytorch_version": torch.__version__,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            "elapsed_sec": elapsed,
            "collect_only": _COLLECT_ONLY,
            "semantic_level": _REQUESTED_SEMANTIC_LEVEL,
            "semantic_level_selection": _SEMANTIC_LEVEL_SELECTION.to_metadata(),
            "dtype_filter": list(_MANIFEST.get("dtype_filter") or []),
            "skip_count": len(_SESSION_SKIPS),
            "session_completed": _SESSION_COMPLETED,
            "harness_probe_failure_count": len(_SESSION_PROBE_FAILURES),
            "harness_probe_failure_artifact": (
                _harness_probe_artifact_path() if _SESSION_PROBE_FAILURES else None
            ),
        },
        "results": _SESSION_RESULTS,
        "skips": _SESSION_SKIPS,
        "harness_probe_failures": _SESSION_PROBE_FAILURES,
    }
    
    
    # Under xdist, each worker writes to its own file to avoid clobbering
    if _IS_XDIST_WORKER:
        latest_path = os.path.join(_RESULTS_DIR, f"{_HARDWARE_KEY}_latest.{_XDIST_WORKER_ID}.json")
    else:
        latest_path = os.path.join(_RESULTS_DIR, f"{_HARDWARE_KEY}_latest.json")
    _atomic_json_dump(latest_path, data)

def _merge_xdist_worker_files(results_dir, hardware_key, latest_path=None):
    """Merge per-worker result files into a single latest.json.
    
    Used by both: (1) controller at session end, (2) startup recovery of
    orphaned files from a killed parallel run.
    Returns the number of merged results, or 0 if no worker files found.
    """
    pattern = os.path.join(results_dir, f"{hardware_key}_latest.gw*.json")
    worker_files = sorted(glob.glob(pattern))
    if not worker_files:
        return 0
    
    if latest_path is None:
        latest_path = os.path.join(results_dir, f"{hardware_key}_latest.json")
    
    # Load existing latest.json as base (if any)
    merged_results = {}
    merged_skips = {}
    merged_probe_failures = []
    merged_probe_failure_keys = set()
    merged_metadata = None
    total_elapsed = 0.0

    def add_probe_failures(records):
        for record in records or []:
            if not isinstance(record, dict):
                continue
            key = harness_probe_failure_key(record)
            if key in merged_probe_failure_keys:
                continue
            merged_probe_failure_keys.add(key)
            merged_probe_failures.append(record)

    if os.path.exists(latest_path):
        try:
            with open(latest_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            merged_results = existing.get("results", {})
            merged_skips = existing.get("skips", {})
            add_probe_failures(existing.get("harness_probe_failures", []))
            merged_metadata = existing.get("metadata")
        except Exception:
            pass
    
    # Merge worker files on top
    for wf in worker_files:
        try:
            with open(wf, "r", encoding="utf-8") as f:
                wdata = json.load(f)
            merged_results.update(wdata.get("results", {}))
            merged_skips.update(wdata.get("skips", {}))
            add_probe_failures(wdata.get("harness_probe_failures", []))
            wm = wdata.get("metadata", {})
            total_elapsed = max(total_elapsed, wm.get("elapsed_sec", 0.0))
            if merged_metadata is None:
                merged_metadata = wm
        except Exception:
            continue
    
    if merged_metadata and total_elapsed > 0:
        merged_metadata["elapsed_sec"] = total_elapsed
    if merged_metadata is not None:
        merged_metadata["harness_probe_failure_count"] = len(merged_probe_failures)
        if merged_probe_failures and not merged_metadata.get("harness_probe_failure_artifact"):
            merged_metadata["harness_probe_failure_artifact"] = None
    
    merged_data = {
        "metadata": merged_metadata or {},
        "results": merged_results,
        "skips": merged_skips,
        "harness_probe_failures": merged_probe_failures,
    }
    _atomic_json_dump(latest_path, merged_data)
    
    # Clean up worker files
    for wf in worker_files:
        try:
            os.remove(wf)
        except Exception:
            pass
    
    return len(merged_results)

@pytest.fixture(autouse=True)
def test_setup_teardown(request):
    # Setup
    clear_metrics()
    
    # Inject device and manifest as fixtures for hand-written tests
    # We can attach them to request if needed, or define standard fixtures below
    yield
    
    # Teardown / Memory management cleanup
    global _DEVICE_NAME, _MEMORY_MODE, _CLEANUP_THRESHOLD
    if _IS_XDIST_WORKER:
        # Under xdist: ALWAYS synchronize + empty cache after each test.
        # Multiple workers pile up uncommitted GPU work and cached allocations;
        # without aggressive cleanup the Metal driver deadlocks under contention.
        synchronize(_DEVICE_NAME)
        empty_cache(_DEVICE_NAME)
    elif _MEMORY_MODE == "conservative":
        synchronize(_DEVICE_NAME)
        empty_cache(_DEVICE_NAME)
    elif _MEMORY_MODE == "balanced":
        # Check memory threshold
        allocated = memory_allocated(_DEVICE_NAME)
        # device_memory_gb contains memory pool size
        dev_mem_list = _MANIFEST.get("hardware", {}).get("device_memory_gb", [24])
        # Skip memory threshold check if device memory is not resolved (e.g. "auto")
        if isinstance(dev_mem_list, (list, tuple)) and dev_mem_list and not isinstance(dev_mem_list[0], str):
            dev_mem_limit = float(dev_mem_list[0]) * (1024 ** 3)
            if allocated > dev_mem_limit * int(_CLEANUP_THRESHOLD) / 100:
                synchronize(_DEVICE_NAME)
                empty_cache(_DEVICE_NAME)

    if (
        _MANIFEST.get("capabilities", {}).get("compile", False)
        and ("compiler/" in str(request.node.fspath).replace("\\", "/") or "compile" in get_required_capabilities(request.node))
        and hasattr(torch, "_dynamo")
    ):
        try:
            torch._dynamo.reset()
        except Exception:
            pass

@pytest.fixture
def device():
    global _DEVICE_NAME
    return _DEVICE_NAME

@pytest.fixture
def manifest():
    global _MANIFEST
    return _MANIFEST

@pytest.fixture
def compare():
    from torchcts.core.comparer import compare_tensors
    global _MANIFEST

    def _compare(actual, expected, category, dtype, **kwargs):
        manifest_overrides = kwargs.pop(
            "manifest_overrides",
            _MANIFEST.get("tolerance_overrides", {}),
        )
        return compare_tensors(
            actual,
            expected,
            category,
            dtype,
            manifest_overrides=manifest_overrides,
            **kwargs,
        )

    return _compare

@pytest.fixture
def input_gen():
    from torchcts.core.input_gen import make_tensor
    return make_tensor

def pytest_runtest_makereport(item, call):
    global _SESSION_RESULTS

    # Run only at the end of the call phase (or setup if setup fails)
    if call.when == "call" or (call.when == "setup" and call.excinfo is not None):
        metrics = get_metrics()
        status = "PASS"
        err_msg = None
        err_type = None
        skip_reason = None
        skip_detail = None
        
        if call.excinfo is not None:
            if call.excinfo.typename == "Skipped":
                status = "SKIP"
                err_msg = str(call.excinfo.value)
                err_type = "Skipped"
                
                # Record in skip session audit
                previous_skip = _SESSION_SKIPS.get(item.nodeid)
                skip_reason = _runtime_skip_reason(err_msg, previous_skip, item)
                skip_detail = previous_skip.get("detail") if previous_skip else err_msg
                _SESSION_SKIPS[item.nodeid] = _skip_record_for_item(item, skip_reason, skip_detail)
            else:
                status = "FAIL" if call.excinfo.typename == "AssertionError" else "ERROR"
                err_msg = str(call.excinfo.value)
                err_type = call.excinfo.typename
                # Attach traceback
                tb = "".join(traceback.format_tb(call.excinfo.tb))
                err_msg += "\n" + tb
                # Truncate message
                if len(err_msg) > 10000:
                    err_msg = err_msg[:9997] + "..."

        metadata = _extract_result_metadata(item)
        failure_stage = _failure_stage_for_exception(err_type, err_msg) if err_type else None
            
        # Register test record
        record = {
            "status": status,
            "phase": call.when,
            "failure_stage": failure_stage,
            "suite": metadata["suite"],
            "test_kind": metadata["test_kind"],
            "capability": metadata["capability"],
            "is_plumbing": metadata["is_plumbing"],
            "is_conformance": metadata["is_conformance"],
            "op": metadata["op"],
            "dispatcher_name": metadata["dispatcher_name"],
            "schema": metadata["schema"],
            "strategy": metadata["strategy"],
            "strategy_family": metadata["strategy_family"],
            "sample_descriptor": metadata["sample_descriptor"],
            "dtype": metadata["dtype"],
            "covers": metadata["covers"],
            "coverage_kind": metadata["coverage_kind"],
            "surface_kind": metadata["surface_kind"],
            "variant_kind": metadata["variant_kind"],
            "coverage_id": metadata["coverage_id"],
            "coverage_status": metadata["coverage_status"],
            "semantic_level": metadata["semantic_level"],
            "requested_level": metadata["requested_level"],
            "semantic_level_selection": metadata["semantic_level_selection"],
            "level_reason": metadata["level_reason"],
            "level_source": metadata["level_source"],
            "semantic_skip_reason": skip_reason if skip_reason and skip_reason.startswith("semantic_") else None,
            "maxerr": metrics["max_abs_err"] if status == "PASS" or metrics["max_abs_err"] > 0 else None,
            "cosim": metrics["cosim"] if status == "PASS" else None,
            "golden_pass": metrics.get("golden_pass", True) if status == "PASS" else None,
            "usable_pass": metrics.get("usable_pass", True) if status == "PASS" else None,
            "quality_warning": metrics.get("quality_warning"),
            "input_condition": metadata.get("input_condition"),
            "error_message": err_msg,
            "error_type": err_type,
            "shapes": metadata["shapes"],
            "duration_ms": call.duration * 1000,
            "last_tested": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
        }
        if status == "SKIP":
            record["skip_reason"] = skip_reason
            record["skip_detail"] = skip_detail
        runtime_unsupported = getattr(item, "_runtime_unsupported_error", None)
        if runtime_unsupported:
            record["runtime_unsupported_matched_pattern"] = runtime_unsupported["matched_pattern"]
            record["runtime_unsupported_error"] = runtime_unsupported["message"]
            if status in ("FAIL", "ERROR"):
                record["classification"] = "backend_runtime_unsupported"
            
        _SESSION_RESULTS[item.nodeid] = record
        
        # Attach diagnosis for failures/errors
        if status in ("FAIL", "ERROR") and err_msg and err_type:
            from torchcts.core.diagnose import diagnose
            diag = diagnose(err_type, err_msg)
            if diag:
                record["diagnosis"] = {
                    "likely_cause": diag.likely_cause,
                    "remediation": diag.remediation,
                    "confidence": diag.confidence,
                }
        
        # Flush result immediately for crash resilience
        flush_results_to_disk()

def pytest_runtest_logstart(nodeid, location):
    """Write each test's node ID to the run log before it executes.

    The file is flushed immediately so that if the process hangs,
    the last line in the log is the test that caused the freeze.
    """
    if _RUN_LOG_FH is not None:
        elapsed = time.time() - _START_TIME
        _RUN_LOG_FH.write(f"{elapsed:8.1f}s  {nodeid}\n")
        _RUN_LOG_FH.flush()


def _subprocess_child_command(item):
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        item.nodeid,
        "--device",
        _DEVICE_NAME,
        "--results-dir",
        _RESULTS_DIR,
        "--known-segfault-policy",
        "off",
        "--adaptive-isolation",
        "off",
    ]
    level_exact = item.config.getoption("--level-exact")
    level_range = item.config.getoption("--level-range")
    if level_exact is not None:
        cmd.extend(["--level-exact", str(level_exact)])
    elif level_range is not None:
        cmd.extend(["--level-range", str(level_range)])
    else:
        cmd.extend(["--level", str(_REQUESTED_SEMANTIC_LEVEL)])
    dtype_filters = item.config.getoption("--dtype") or []
    for dtype_name in dtype_filters:
        cmd.extend(["--dtype", dtype_name])
    memory_mode = item.config.getoption("--memory-mode")
    if memory_mode:
        cmd.extend(["--memory-mode", memory_mode])
    max_device_memory = item.config.getoption("--max-device-memory")
    if max_device_memory is not None:
        cmd.extend(["--max-device-memory", str(max_device_memory)])
    max_tensor_size = item.config.getoption("--max-tensor-size")
    if max_tensor_size is not None:
        cmd.extend(["--max-tensor-size", str(max_tensor_size)])
    if item.config.getoption("--validation"):
        cmd.append("--validation")
    return cmd


def _load_latest_result_for_item(item):
    latest_path = os.path.join(_RESULTS_DIR, f"{_HARDWARE_KEY}_latest.json")
    if not os.path.exists(latest_path):
        return None
    try:
        with open(latest_path, "r", encoding="utf-8") as f:
            latest_data = json.load(f)
    except Exception:
        return None
    return latest_data.get("results", {}).get(item.nodeid)


def _subprocess_error_record(item, status, error_type, error_message, duration_ms, command, returncode=None, stdout="", stderr="", timed_out=False):
    metadata = _extract_result_metadata(item)
    failure_stage = "subprocess_timeout" if timed_out else "process"
    record = {
        "nodeid": item.nodeid,
        "status": status,
        "phase": "subprocess",
        "failure_stage": failure_stage,
        "suite": metadata["suite"],
        "test_kind": metadata["test_kind"],
        "capability": metadata["capability"],
        "is_plumbing": metadata["is_plumbing"],
        "is_conformance": metadata["is_conformance"],
        "op": metadata["op"] or item.name,
        "dispatcher_name": metadata["dispatcher_name"],
        "schema": metadata["schema"],
        "strategy": metadata["strategy"],
        "strategy_family": metadata["strategy_family"],
        "sample_descriptor": metadata["sample_descriptor"],
        "dtype": metadata["dtype"],
        "covers": metadata["covers"],
        "coverage_kind": metadata["coverage_kind"],
        "surface_kind": metadata["surface_kind"],
        "variant_kind": metadata["variant_kind"],
        "coverage_id": metadata["coverage_id"],
        "coverage_status": metadata["coverage_status"],
        "semantic_level": metadata["semantic_level"],
        "requested_level": metadata["requested_level"],
        "semantic_level_selection": metadata["semantic_level_selection"],
        "level_reason": metadata["level_reason"],
        "level_source": metadata["level_source"],
        "semantic_skip_reason": None,
        "maxerr": None,
        "cosim": None,
        "error_message": error_message,
        "error_type": error_type,
        "subprocess": {
            "command": cmd_to_string(command),
            "command_args": list(command),
            "returncode": returncode,
            "signal": _signal_name(returncode),
            "timed_out": timed_out,
            "duration_seconds": duration_ms / 1000.0,
            "stdout_tail": _text_tail(stdout),
            "stderr_tail": _text_tail(stderr),
        },
        "shapes": metadata["shapes"],
        "duration_ms": duration_ms,
        "last_tested": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    from torchcts.core.diagnose import diagnose

    diag = diagnose(error_type, error_message, returncode or 0)
    if diag:
        record["diagnosis"] = {
            "likely_cause": diag.likely_cause,
            "remediation": diag.remediation,
            "confidence": diag.confidence,
        }
    return record


def _subprocess_pass_record(item, duration_ms, command, returncode=0, stdout="", stderr=""):
    metadata = _extract_result_metadata(item)
    return {
        "nodeid": item.nodeid,
        "status": "PASS",
        "phase": "subprocess_child",
        "failure_stage": None,
        "suite": metadata["suite"],
        "test_kind": metadata["test_kind"],
        "capability": metadata["capability"],
        "is_plumbing": metadata["is_plumbing"],
        "is_conformance": metadata["is_conformance"],
        "op": metadata["op"] or item.name,
        "dispatcher_name": metadata["dispatcher_name"],
        "schema": metadata["schema"],
        "strategy": metadata["strategy"],
        "strategy_family": metadata["strategy_family"],
        "sample_descriptor": metadata["sample_descriptor"],
        "dtype": metadata["dtype"],
        "covers": metadata["covers"],
        "coverage_kind": metadata["coverage_kind"],
        "surface_kind": metadata["surface_kind"],
        "variant_kind": metadata["variant_kind"],
        "coverage_id": metadata["coverage_id"],
        "coverage_status": metadata["coverage_status"],
        "semantic_level": metadata["semantic_level"],
        "requested_level": metadata["requested_level"],
        "semantic_level_selection": metadata["semantic_level_selection"],
        "level_reason": metadata["level_reason"],
        "level_source": metadata["level_source"],
        "semantic_skip_reason": None,
        "maxerr": None,
        "cosim": None,
        "golden_pass": None,
        "usable_pass": None,
        "quality_warning": None,
        "input_condition": metadata.get("input_condition"),
        "error_message": None,
        "error_type": None,
        "subprocess": {
            "command": cmd_to_string(command),
            "command_args": list(command),
            "returncode": returncode,
            "signal": _signal_name(returncode),
            "timed_out": False,
            "duration_seconds": duration_ms / 1000.0,
            "stdout_tail": _text_tail(stdout),
            "stderr_tail": _text_tail(stderr),
        },
        "shapes": metadata["shapes"],
        "duration_ms": duration_ms,
        "last_tested": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def cmd_to_string(command):
    return " ".join(str(part) for part in command)


def pytest_runtest_protocol(item, nextitem):
    global _SUBPROCESS_MODE, _SESSION_RESULTS
    
    # Subprocess execution wrapper
    # Checks if subprocess mode is enabled and we are in the parent process
    is_child = _is_child_process()
    known_segfault_match = _known_segfault_match_for_item(item)
    adaptive_isolation_match = _adaptive_isolation_match_for_item(item)
    needs_isolation = _SUBPROCESS_MODE or known_segfault_match is not None or adaptive_isolation_match is not None
    if needs_isolation and not is_child:
        # Run test node in child process
        cmd = _subprocess_child_command(item)
        env = os.environ.copy()
        env["_TORCHCTS_SUBPROCESS"] = "1"
        env["TORCHCTS_NON_INTERACTIVE"] = "1"
        
        # Start timing
        start_t = time.time()
        timeout = float(item.config.getoption("--subprocess-timeout") or 120.0)
        try:
            # Run test in child process with a standard timeout.
            res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
            duration = (time.time() - start_t) * 1000

            child_record = _load_latest_result_for_item(item)
            if child_record is not None and not _is_process_crash(res.returncode, res.stderr, res.stdout):
                child_record["nodeid"] = item.nodeid
                child_phase = child_record.get("phase")
                if child_phase and child_phase != "subprocess_child":
                    child_record["child_phase"] = child_phase
                child_record["phase"] = "subprocess_child"
                if known_segfault_match:
                    child_record.update(
                        _known_segfault_result_fields(
                            known_segfault_match,
                            resolved=child_record.get("status") == "PASS",
                            actual_signal=_signal_name(res.returncode),
                        )
                    )
                    if child_record.get("status") == "PASS":
                        print(
                            f"Warning: known segfault {known_segfault_match['id']} passed; "
                            "rule may be stale, intentionally broad, or order-dependent.",
                            file=sys.stderr,
                        )
                if adaptive_isolation_match:
                    child_record.update(
                        _adaptive_isolation_result_fields(
                            adaptive_isolation_match,
                            known_segfault_match=known_segfault_match,
                            resolved=child_record.get("status") == "PASS",
                        )
                    )
                child_record["subprocess"] = {
                    "command": cmd_to_string(cmd),
                    "command_args": list(cmd),
                    "returncode": res.returncode,
                    "signal": _signal_name(res.returncode),
                    "timed_out": False,
                    "duration_seconds": duration / 1000.0,
                    "stdout_tail": _text_tail(res.stdout),
                    "stderr_tail": _text_tail(res.stderr),
                }
                _SESSION_RESULTS[item.nodeid] = child_record
            elif res.returncode == 0 and not _is_process_crash(res.returncode, res.stderr, res.stdout):
                record = _subprocess_pass_record(
                    item,
                    duration,
                    cmd,
                    returncode=res.returncode,
                    stdout=res.stdout,
                    stderr=res.stderr,
                )
                if known_segfault_match:
                    record.update(
                        _known_segfault_result_fields(
                            known_segfault_match,
                            resolved=True,
                            actual_signal=_signal_name(res.returncode),
                        )
                    )
                    print(
                        f"Warning: known segfault {known_segfault_match['id']} passed; "
                        "rule may be stale, intentionally broad, or order-dependent.",
                        file=sys.stderr,
                    )
                if adaptive_isolation_match:
                    record.update(
                        _adaptive_isolation_result_fields(
                            adaptive_isolation_match,
                            known_segfault_match=known_segfault_match,
                            resolved=True,
                        )
                    )
                _SESSION_RESULTS[item.nodeid] = record
            else:
                stdout_tail = _text_tail(res.stdout)
                stderr_tail = _text_tail(res.stderr)
                signal_name = _signal_name(res.returncode)
                process_crash = _is_process_crash(res.returncode, res.stderr, res.stdout)
                err_msg = (
                    f"Subprocess failed before writing a result record. "
                    f"returncode={res.returncode} signal={signal_name}\n"
                    f"stderr_tail:\n{stderr_tail}\nstdout_tail:\n{stdout_tail}"
                )
                err_type = "ProcessCrash" if process_crash else "SubprocessFailure"
                status = "ERROR"
                record = _subprocess_error_record(
                    item,
                    status,
                    err_type,
                    _text_tail(err_msg, 10000),
                    duration,
                    cmd,
                    returncode=res.returncode,
                    stdout=res.stdout,
                    stderr=res.stderr,
                )
                if known_segfault_match:
                    record.update(
                        _known_segfault_result_fields(
                            known_segfault_match,
                            actual_signal=_signal_name(res.returncode),
                        )
                    )
                    if process_crash:
                        record["classification"] = _known_segfault_process_classification(
                            known_segfault_match,
                            stdout=res.stdout,
                            stderr=res.stderr,
                        )
                if adaptive_isolation_match:
                    record.update(
                        _adaptive_isolation_result_fields(
                            adaptive_isolation_match,
                            known_segfault_match=known_segfault_match,
                        )
                    )
                _SESSION_RESULTS[item.nodeid] = record
                flush_results_to_disk()
                
        except subprocess.TimeoutExpired as exc:
            duration = (time.time() - start_t) * 1000
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            record = _subprocess_error_record(
                item,
                "ERROR",
                "TimeoutError",
                f"TIMEOUT (exceeded {timeout:g} seconds)",
                duration,
                cmd,
                stdout=stdout,
                stderr=stderr,
                timed_out=True,
            )
            if known_segfault_match:
                record.update(_known_segfault_result_fields(known_segfault_match))
            if adaptive_isolation_match:
                record.update(
                    _adaptive_isolation_result_fields(
                        adaptive_isolation_match,
                        known_segfault_match=known_segfault_match,
                    )
                )
            _SESSION_RESULTS[item.nodeid] = record
            flush_results_to_disk()
            
        return True # Handled protocol, don't run test in parent process
        
    return None # Fallback to standard execution

def pytest_sessionfinish(session, exitstatus):
    global _SESSION_RESULTS, _RESULTS_DIR, _HARDWARE_KEY, _BASELINE_RESULTS
    global _ARTIFACT_WRITES_ENABLED, _RUN_LOG_FH, _SESSION_COMPLETED

    if not _is_child_process():
        if any(record.get("status") in ("FAIL", "ERROR") for record in _SESSION_RESULTS.values()):
            session.exitstatus = 1

    # Close run log
    if _RUN_LOG_FH is not None:
        try:
            _RUN_LOG_FH.close()
        except Exception:
            pass
        _RUN_LOG_FH = None

    if not _ARTIFACT_WRITES_ENABLED:
        return
    
    # Save the latest JSON file
    _SESSION_COMPLETED = True
    flush_results_to_disk()

    # Under xdist: workers just flush and return; controller merges
    if _IS_XDIST_WORKER:
        return  # Worker done — controller will merge our file

    # Check if we're the xdist controller (workers wrote per-worker files)
    _merge_xdist_worker_files(_RESULTS_DIR, _HARDWARE_KEY)

    # Load the completed latest.json
    latest_path = os.path.join(_RESULTS_DIR, f"{_HARDWARE_KEY}_latest.json")
    if os.path.exists(latest_path):
        with open(latest_path, "r", encoding="utf-8") as f:
            current_data = json.load(f)
            
        # Copy to history directory
        history_dir = os.path.join(_RESULTS_DIR, f"{_HARDWARE_KEY}_history")
        os.makedirs(history_dir, exist_ok=True)
        
        timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        history_path = os.path.join(history_dir, f"{timestamp_str}.json")
        _atomic_json_dump(history_path, current_data)
            
        # Build report
        from torchcts.core.report import build_report
        # If baseline results loaded, construct baseline object
        baseline_obj = None
        if _BASELINE_RESULTS:
            # We construct a mock baseline data containing metadata and results
            baseline_obj = {"results": _BASELINE_RESULTS}
            
        scorecard, markdown = build_report(current_data, baseline_obj, include_skips=_REPORT_SKIPS)
        
        # Save report
        report_path = os.path.join(_RESULTS_DIR, f"{_HARDWARE_KEY}_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(markdown)
            
        # If in parent process or normal execution, print scorecard to stdout
        is_child = _is_child_process()
        if not is_child:
            try:
                print(scorecard)
            except UnicodeEncodeError:
                try:
                    sys.stdout.buffer.write(scorecard.encode(sys.stdout.encoding or "utf-8", errors="replace"))
                    sys.stdout.flush()
                except Exception:
                    pass

def pytest_ignore_collect(collection_path, config):
    path_str = str(collection_path)
    if "test_transformer.py" in path_str or "test_e2e_models.py" in path_str:
        try:
            import transformers
        except ImportError:
            return True
    return False

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    outcome = yield
    if outcome.excinfo is not None:
        exc_type, exc_val, exc_tb = outcome.excinfo
        if exc_type.__name__ in ("Skipped", "Failed", "OutcomeException"):
            return
        msg = str(exc_val)
        matched_pattern = _runtime_unsupported_pattern_match(msg)
        if matched_pattern:
            item._runtime_unsupported_error = {
                "matched_pattern": matched_pattern,
                "message": msg,
            }
            return
