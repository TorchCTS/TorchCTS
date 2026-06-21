import os
import sys
import json
import time
import datetime
import subprocess
import warnings
import traceback
import faulthandler
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

from torchcts.core.device import (
    get_device_backend,
    synchronize,
    empty_cache,
    memory_allocated
)
from torchcts.core.report import get_hardware_key
from torchcts.core.opinfo_adapter import str_to_dtype, dtype_to_str
from torchcts.core.comparer import clear_metrics, get_metrics
from torchcts.core.input_gen import refresh_shared_data

# Global session variables
_MANIFEST = {}
_DEVICE_NAME = "cpu"
_HARDWARE_KEY = "unknown"
_RESULTS_DIR = "./results"
_START_TIME = 0
_SESSION_RESULTS = {}
_SESSION_SKIPS = {}
_BASELINE_RESULTS = {}
_SHOW_SKIPS = False
_REPORT_SKIPS = False
_SUBPROCESS_MODE = False
_MEMORY_MODE = "balanced"
_CLEANUP_THRESHOLD = 80

_HARDWARE_UNSUPPORTED_PATTERNS = [
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
]
_MAX_DEVICE_MEM = None
_MAX_TENSOR_SIZE = None
_COLLECT_ONLY = False
_ARTIFACT_WRITES_ENABLED = True
_ACTUAL_DEVICE_COUNT = 1


def _is_child_process():
    return (
        os.environ.get("_TORCHCTS_SUBPROCESS") == "1"
        or os.environ.get("_TORCH_CTS_SUBPROCESS") == "1"
        or os.environ.get("_BACKEND_VALIDATOR_SUBPROCESS") == "1"
    )


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
        "stress",
        "multi_device",
    ):
        token = f"/{suite_name}/"
        if token in filepath:
            return suite_name
    return "custom"


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
    }

    filepath = str(item.fspath)
    if "test_quantized.py" in filepath or "test_guard_alloc.py" in filepath:
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

    return metadata


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
    group.addoption("--suite", choices=["opinfo", "operators", "training", "compiler", "device_api", "autograd", "memory", "custom", "dtypes", "strides", "workloads", "rng", "serialization", "errors", "stress", "multi_device", "adversarial"], help="Limit test collection to a specific suite")
    group.addoption("--memory-mode", default="balanced", choices=["conservative", "balanced", "performance"], help="Memory cleanup cadence")
    group.addoption("--max-device-memory", type=int, help="Cap maximum device memory allowed (MB)")
    group.addoption("--max-tensor-size", type=int, help="Cap maximum single tensor size allowed (MB)")
    group.addoption("--show-skips", action="store_true", help="Dry-run: print skips and exit")
    group.addoption("--report-skips", action="store_true", help="Include skip audit in report")
    group.addoption("--results-dir", default="./results", help="Directory to save JSON/Markdown results")
    group.addoption("--non-interactive", action="store_true", help="Error instead of prompting in auto device selection")
    group.addoption("--subprocess-per-test", action="store_true", help="Run each test in a separate subprocess for crash isolation")
    group.addoption("--benchmark", action="store_true", help="Run tests in benchmarking mode")
    group.addoption("--ref-device", help="Optional reference device to compare against (e.g., cpu, mps)")
    group.addoption("--validation", action="store_true", help="CPU reference validation mode: run entire suite on CPU")

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
            
    filepath = str(item.fspath).replace("\\", "/")
    if "autograd/" in filepath or "test_opinfo_backward" in filepath:
        reqs.add("training")
    if "test_double_backward" in filepath:
        reqs.add("double_backward")
    if "test_gradcheck" in filepath:
        reqs.add("gradcheck")
    if "test_mixed_precision" in filepath:
        reqs.add("autocast")
    if "test_dataloader" in filepath:
        reqs.add("dataloader")
    if "test_module_hooks" in filepath:
        reqs.add("module_hooks")
    if "channels_last" in filepath:
        reqs.add("channels_last")
    if "sparse" in filepath:
        reqs.add("sparse")
    if "compiler/" in filepath:
        reqs.add("compile")
    if "serialization/" in filepath:
        reqs.add("serialization")
    if "rng/" in filepath:
        reqs.add("generator")
    if "device_api/" in filepath:
        reqs.add("device_api")
    if "multi_device/" in filepath:
        reqs.add("multi_device")
    if "guard_alloc" in filepath:
        reqs.add("guard_alloc")
        
    return reqs

def pytest_configure(config):
    global _MANIFEST, _DEVICE_NAME, _HARDWARE_KEY, _RESULTS_DIR, _START_TIME, _SHOW_SKIPS, _REPORT_SKIPS
    global _SUBPROCESS_MODE, _MEMORY_MODE, _CLEANUP_THRESHOLD, _MAX_DEVICE_MEM, _MAX_TENSOR_SIZE, _BASELINE_RESULTS
    global _COLLECT_ONLY, _ARTIFACT_WRITES_ENABLED, _ACTUAL_DEVICE_COUNT

    # Register custom markers
    config.addinivalue_line("markers", "gate: backend registration gate tests — run first, abort on failure")
    config.addinivalue_line("markers", "smoke: smoke tests only")
    config.addinivalue_line("markers", "medium: medium tests")
    config.addinivalue_line("markers", "opinfo: OpInfo breadth tests")
    config.addinivalue_line("markers", "workload: real-world workloads")
    config.addinivalue_line("markers", "stress: stress tests")
    config.addinivalue_line("markers", "requires(capability): required capabilities")
    config.addinivalue_line("markers", "adversarial: adversarial test suite")
    config.addinivalue_line("markers", "benchmarkable: safe to repeat many times in benchmark mode")

    # 1. Load manifest
    _MANIFEST = load_manifest()
    
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
            "generator": True,
            "double_backward": True,
            "gradcheck": True,
            "gradient_checkpointing": True,
            "autocast": True,
            "fused_optimizer": True,
            "dataloader": True,
            "module_hooks": True,
            "channels_last": True,
            "sparse": True,
            "nested": True,
            "foreach": True,
            "fp8": True,
            "quantized": True,
            "compile": True,
            "pinned_memory": True,
            "deterministic": True,
            "device_api": False,
            "guard_alloc": False,
            "streams": False,
            "events": False,
            "multi_device": False,
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
        _MANIFEST["device_count"] = 1
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

    # Probes and capability overrides (only when executing, not in validation/collectonly modes)
    if not is_validation and not _COLLECT_ONLY:
        # Dynamic Dtype Probing
        supported_dtypes = _MANIFEST.setdefault("supported_dtypes", {})
        probed_supported = {}
        for dt, val in list(supported_dtypes.items()):
            # Only probe if the dtype was mapped to True or a pattern
            if val:
                try:
                    # Try allocating a small tensor on target device
                    torch.zeros(1, dtype=dt, device=_DEVICE_NAME)
                    probed_supported[dt] = val
                except Exception:
                    # Device does not support this dtype, remove it
                    pass
        supported_dtypes.clear()
        supported_dtypes.update(probed_supported)

        # Dynamic Capability Probing (pinned_memory, sparse)
        from torchcts.core.device import probe_capability
        caps = _MANIFEST.setdefault("capabilities", {})
        for cap in ["pinned_memory", "sparse"]:
            if caps.get(cap, False):
                if not probe_capability(_DEVICE_NAME, cap):
                    caps[cap] = False
                    print(f"Probe: {cap} -> disabled (subprocess probe failed)")

    # 3. Hardware details
    _HARDWARE_KEY = get_hardware_key(_DEVICE_NAME, _MANIFEST)
    _RESULTS_DIR = config.getoption("--results-dir")
    
    _SHOW_SKIPS = config.getoption("--show-skips")
    _REPORT_SKIPS = config.getoption("--report-skips")
    _SUBPROCESS_MODE = config.getoption("--subprocess-per-test")
    _ARTIFACT_WRITES_ENABLED = not (_COLLECT_ONLY or _SHOW_SKIPS)
    if _ARTIFACT_WRITES_ENABLED:
        os.makedirs(_RESULTS_DIR, exist_ok=True)
    
    # Memory configurations
    _MEMORY_MODE = config.getoption("--memory-mode")
    _CLEANUP_THRESHOLD = _MANIFEST.get("resource_limits", {}).get("cleanup_threshold_pct", 80)
    
    cli_max_mem = config.getoption("--max-device-memory")
    _MAX_DEVICE_MEM = cli_max_mem if cli_max_mem is not None else _MANIFEST.get("resource_limits", {}).get("max_device_memory_mb")
    
    cli_max_tensor = config.getoption("--max-tensor-size")
    _MAX_TENSOR_SIZE = cli_max_tensor if cli_max_tensor is not None else _MANIFEST.get("resource_limits", {}).get("max_tensor_size_mb")

    # Load baseline results for regression detection
    _BASELINE_RESULTS = {}
    if _ARTIFACT_WRITES_ENABLED:
        latest_json_path = os.path.join(_RESULTS_DIR, f"{_HARDWARE_KEY}_latest.json")
        if os.path.exists(latest_json_path):
            try:
                with open(latest_json_path, "r", encoding="utf-8") as f:
                    _BASELINE_RESULTS = json.load(f).get("results", {})
            except Exception:
                pass

    # Start timing
    _START_TIME = time.time()
    
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
            elif suite == "custom":
                standard_dirs = ["opinfo/", "operators/", "training/", "compiler/", "device_api/", "autograd/", "memory/", "dtypes/", "strides/", "workloads/", "rng/", "serialization/", "errors/", "stress/", "multi_device/"]
                is_match = not any(d in filepath for d in standard_dirs)
            
            if is_match:
                filtered_items.append(item)
            else:
                deselected_items.append(item)
        
        config.hook.pytest_deselected(items=deselected_items)
        items[:] = filtered_items

    # Check capabilities and supported dtypes
    is_validation = config.getoption("--validation")
    if is_validation:
        caps = {
            "inference": True,
            "training": True,
            "serialization": True,
            "generator": True,
            "double_backward": True,
            "gradcheck": True,
            "gradient_checkpointing": True,
            "autocast": True,
            "fused_optimizer": True,
            "dataloader": True,
            "module_hooks": True,
            "channels_last": True,
            "sparse": True,
            "nested": True,
            "foreach": True,
            "fp8": True,
            "quantized": True,
            "compile": False,
            "pinned_memory": True,
            "deterministic": True,
            "device_api": False,
            "guard_alloc": False,
            "streams": False,
            "events": False,
            "multi_device": False,
        }
        supported_dtypes = {
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
        skip_ops = set()
        device_count = 1
    else:
        caps = _MANIFEST.get("capabilities", {})
        supported_dtypes = _MANIFEST.get("supported_dtypes", {})
        skip_ops = set(_MANIFEST.get("skip_ops", []))
        device_count = _MANIFEST.get("effective_device_count", _MANIFEST.get("device_count", 1))
    
    # Dynamic compiler functional check
    if caps.get("compile", False) and not (_COLLECT_ONLY or _SHOW_SKIPS):
        try:
            @torch.compile(fullgraph=True)
            def _dummy_fn(x):
                return x + 1.0
            _dummy_fn(torch.ones(1, device=_DEVICE_NAME))
        except Exception:
            caps["compile"] = False
            
    # Optional CLI dtype filter
    cli_dtypes = config.getoption("--dtype")
    if cli_dtypes:
        # Overwrite manifest dtypes with CLI filters
        supported_dtypes = {}
        for dt_name in cli_dtypes:
            dt = str_to_dtype(dt_name) or str_to_dtype(f"torch.{dt_name}")
            if dt:
                supported_dtypes[dt] = True

    keep_items = []
    
    for item in items:
        skip_reason = None
        detail = ""
        
        # Determine ATen op name
        op_name = None
        if hasattr(item, "callspec"):
            if "op" in item.callspec.params:
                op_param = item.callspec.params["op"]
                op_name = getattr(op_param, "name", str(op_param))
            elif "op_name" in item.callspec.params:
                op_name = item.callspec.params["op_name"]

        # Gate tests always run — skip all filtering
        if item.get_closest_marker("gate"):
            keep_items.append(item)
            continue

        # 1. Capability check
        req_caps = get_required_capabilities(item)
        missing_caps = [c for c in req_caps if not caps.get(c, False) and c != "multi_device"]
        if missing_caps:
            skip_reason = "capability_not_declared"
            detail = f"requires capabilities: {', '.join(missing_caps)}"
        elif "multi_device" in req_caps and device_count < 2:
            skip_reason = "device_count"
            declared = _MANIFEST.get("device_count", device_count)
            detail = (
                f"requires device_count>=2, runtime exposes {device_count}"
                if declared == device_count
                else f"requires device_count>=2, manifest declares {declared} but runtime exposes {device_count}"
            )
            
        # 2. Dtype check
        if not skip_reason and hasattr(item, "callspec") and "dtype" in item.callspec.params:
            dt = item.callspec.params["dtype"]
            dt_str = dtype_to_str(dt)
            
            dtype_allowed = False
            dtype_filter = None
            
            if dt in supported_dtypes:
                dtype_allowed = True
                dtype_filter = supported_dtypes[dt]
            elif dt_str in supported_dtypes:
                dtype_allowed = True
                dtype_filter = supported_dtypes[dt_str]
                
            if not dtype_allowed:
                skip_reason = "dtype_not_listed"
                detail = f"{dt_str} not in supported_dtypes"
            elif isinstance(dtype_filter, str) and op_name:
                import re
                if not re.search(dtype_filter, op_name):
                    skip_reason = "dtype_regex_filtered"
                    detail = f"op {op_name} filtered out by {dt_str} regex: {dtype_filter}"
                    
        # 3. Op exclusions
        if not skip_reason and op_name and op_name in skip_ops:
            skip_reason = "op_excluded"
            detail = f"{op_name} is in skip_ops list"

        # 4. src_dtype / dst_dtype check (e.g. test_copy_cast)
        if not skip_reason and hasattr(item, "callspec"):
            for dtype_param in ("src_dtype", "dst_dtype"):
                if dtype_param in item.callspec.params:
                    dt = item.callspec.params[dtype_param]
                    if dt not in supported_dtypes:
                        dt_str = dtype_to_str(dt)
                        skip_reason = "dtype_not_listed"
                        detail = f"{dt_str} ({dtype_param}) not in supported_dtypes"
                        break

        # 5. CPU device cannot run cross-device or device-module tests
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

        # 6. Device module availability check
        if not skip_reason and hasattr(item, "callspec"):
            test_name = item.name
            if "test_device_module_methods" in test_name or "test_device_memory_query" in test_name:
                from torchcts.core.device import get_device_module
                if get_device_module(_DEVICE_NAME) is None:
                    skip_reason = "no_device_module"
                    detail = f"No custom device module found for torch.{_DEVICE_NAME}"

        # 7. set_device support check
        if not skip_reason:
            test_name = item.name
            if "test_set_device_context" in test_name:
                _mod = torch.cuda if _DEVICE_NAME == "cuda" else getattr(torch, _DEVICE_NAME, None)
                if _mod is None or not hasattr(_mod, "set_device"):
                    skip_reason = "set_device_not_supported"
                    detail = f"Device module for {_DEVICE_NAME} does not support set_device"

        # 8. OOM recovery manifest check
        if not skip_reason:
            test_name = item.name
            if "test_oom_recovery" in test_name:
                hw_config = _MANIFEST.get("hardware", {})
                if not hw_config.get("oom_recoverable", True):
                    skip_reason = "oom_not_recoverable"
                    detail = "OOM recovery not marked as recoverable in manifest"

        # 9. float64 required for gradcheck
        if not skip_reason:
            test_name = item.name
            if "test_gradcheck" in test_name:
                if torch.float64 not in supported_dtypes:
                    skip_reason = "dtype_not_listed"
                    detail = "float64 not in supported_dtypes (required for gradcheck)"

        if skip_reason:
            metadata = _extract_result_metadata(item)
            _SESSION_SKIPS[item.nodeid] = {
                "suite": metadata["suite"],
                "capability": metadata["capability"],
                "is_plumbing": metadata["is_plumbing"],
                "is_conformance": metadata["is_conformance"],
                "op": metadata["op"] or item.name,
                "dtype": metadata["dtype"],
                "skip_reason": skip_reason,
                "detail": detail
            }
            if _SHOW_SKIPS:
                item.add_marker(pytest.mark.skip(reason=detail))
                keep_items.append(item)
        else:
            keep_items.append(item)

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
    global _ARTIFACT_WRITES_ENABLED

    if not _ARTIFACT_WRITES_ENABLED:
        return
    
    elapsed = time.time() - _START_TIME
    data = {
        "metadata": {
            "device_name": _DEVICE_NAME,
            "hardware_key": _HARDWARE_KEY,
            "pytorch_version": torch.__version__,
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z"),
            "elapsed_sec": elapsed,
            "collect_only": _COLLECT_ONLY,
        },
        "results": _SESSION_RESULTS,
        "skips": _SESSION_SKIPS if _REPORT_SKIPS else {}
    }
    
    latest_path = os.path.join(_RESULTS_DIR, f"{_HARDWARE_KEY}_latest.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

@pytest.fixture(autouse=True)
def test_setup_teardown(request):
    # Setup
    clear_metrics()
    
    # Inject device and manifest as fixtures for hand-written tests
    # We can attach them to request if needed, or define standard fixtures below
    yield
    
    # Teardown / Memory management cleanup
    global _DEVICE_NAME, _MEMORY_MODE, _CLEANUP_THRESHOLD
    if _MEMORY_MODE == "conservative":
        synchronize(_DEVICE_NAME)
        empty_cache(_DEVICE_NAME)
    elif _MEMORY_MODE == "balanced":
        # Check memory threshold
        allocated = memory_allocated(_DEVICE_NAME)
        # device_memory_gb contains memory pool size
        dev_mem_list = _MANIFEST.get("hardware", {}).get("device_memory_gb", [24])
        dev_mem_limit = dev_mem_list[0] * (1024 ** 3)
        if allocated > dev_mem_limit * _CLEANUP_THRESHOLD / 100:
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
    return compare_tensors

@pytest.fixture
def input_gen():
    from torchcts.core.input_gen import make_tensor
    return make_tensor

def pytest_runtest_makereport(item, call):
    global _SESSION_RESULTS
    
    # If a gate test fails, abort the entire session immediately
    if call.when == "call" and call.excinfo is not None and item.get_closest_marker("gate"):
        if call.excinfo.typename != "Skipped":
            pytest.exit(
                f"GATE FAILURE: {item.name} — backend is not functional.\n"
                f"  {call.excinfo.typename}: {call.excinfo.value}",
                returncode=1,
            )

    # Run only at the end of the call phase (or setup if setup fails)
    if call.when == "call" or (call.when == "setup" and call.excinfo is not None):
        metrics = get_metrics()
        status = "PASS"
        err_msg = None
        err_type = None
        
        if call.excinfo is not None:
            if call.excinfo.typename == "Skipped":
                status = "SKIP"
                err_msg = str(call.excinfo.value)
                err_type = "Skipped"
                
                # Record in skip session audit
                if hasattr(item, "_hardware_unsupported_reason"):
                    skip_reason = "hardware_unsupported"
                else:
                    skip_reason = "runtime_skip"
                
                metadata = _extract_result_metadata(item)
                _SESSION_SKIPS[item.nodeid] = {
                    "suite": metadata["suite"],
                    "capability": metadata["capability"],
                    "is_plumbing": metadata["is_plumbing"],
                    "is_conformance": metadata["is_conformance"],
                    "op": metadata["op"] or item.name,
                    "dtype": metadata["dtype"],
                    "skip_reason": skip_reason,
                    "detail": err_msg
                }
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
            
        # Register test record
        record = {
            "status": status,
            "suite": metadata["suite"],
            "test_kind": metadata["test_kind"],
            "capability": metadata["capability"],
            "is_plumbing": metadata["is_plumbing"],
            "is_conformance": metadata["is_conformance"],
            "op": metadata["op"],
            "dtype": metadata["dtype"],
            "maxerr": metrics["max_abs_err"] if status == "PASS" or metrics["max_abs_err"] > 0 else None,
            "cosim": metrics["cosim"] if status == "PASS" else None,
            "error_message": err_msg,
            "error_type": err_type,
            "shapes": metadata["shapes"],
            "duration_ms": call.duration * 1000,
            "last_tested": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")
        }
        
        if hasattr(item, "bench_stats"):
            record["bench_stats"] = item.bench_stats
            
        _SESSION_RESULTS[item.nodeid] = record
        
        # Flush result immediately for crash resilience
        flush_results_to_disk()

def pytest_runtest_protocol(item, nextitem):
    global _SUBPROCESS_MODE, _SESSION_RESULTS
    
    # Subprocess execution wrapper
    # Checks if subprocess mode is enabled and we are in the parent process
    is_child = _is_child_process()
    if _SUBPROCESS_MODE and not is_child:
        # Run test node in child process
        cmd = [sys.executable, "-m", "pytest", item.nodeid]
        
        # Forward CLI arguments except wrapper/subprocess options
        for arg in sys.argv[1:]:
            if arg not in ("-m", "torchcts", "run", "--subprocess-per-test", item.nodeid):
                cmd.append(arg)
                
        env = os.environ.copy()
        env["_TORCHCTS_SUBPROCESS"] = "1"
        
        # Start timing
        start_t = time.time()
        try:
            # Run test in child process with a standard timeout (e.g. 30s)
            res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30.0)
            duration = (time.time() - start_t) * 1000
            
            # Check exit code
            if res.returncode == 0:
                # The child ran successfully and updated results JSON
                # Parent doesn't need to overwrite it, but let's reload it
                # to sync with session results
                latest_path = os.path.join(_RESULTS_DIR, f"{_HARDWARE_KEY}_latest.json")
                if os.path.exists(latest_path):
                    with open(latest_path, "r", encoding="utf-8") as f:
                        latest_data = json.load(f)
                        if item.nodeid in latest_data.get("results", {}):
                            _SESSION_RESULTS[item.nodeid] = latest_data["results"][item.nodeid]
            else:
                # Hard crash / segfault
                err_msg = res.stderr or res.stdout
                err_type = "ProcessCrash"
                status = "ERROR"
                
                # Check for standard segfault signals
                if res.returncode == -11 or res.returncode == 139:
                    err_msg = "SEGFAULT (exit code -11)"
                elif res.returncode == -9 or res.returncode == 137:
                    err_msg = "OOM KILLED (exit code -9)"
                    
                metadata = _extract_result_metadata(item)
                _SESSION_RESULTS[item.nodeid] = {
                    "status": status,
                    "suite": metadata["suite"],
                    "test_kind": metadata["test_kind"],
                    "capability": metadata["capability"],
                    "is_plumbing": metadata["is_plumbing"],
                    "is_conformance": metadata["is_conformance"],
                    "op": metadata["op"] or item.name,
                    "dtype": metadata["dtype"],
                    "maxerr": None,
                    "cosim": None,
                    "error_message": err_msg[:500],
                    "error_type": err_type,
                    "shapes": metadata["shapes"],
                    "duration_ms": duration,
                    "last_tested": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")
                }
                flush_results_to_disk()
                
        except subprocess.TimeoutExpired:
            duration = (time.time() - start_t) * 1000
            metadata = _extract_result_metadata(item)
            _SESSION_RESULTS[item.nodeid] = {
                "status": "ERROR",
                "suite": metadata["suite"],
                "test_kind": metadata["test_kind"],
                "capability": metadata["capability"],
                "is_plumbing": metadata["is_plumbing"],
                "is_conformance": metadata["is_conformance"],
                "op": metadata["op"] or item.name,
                "dtype": metadata["dtype"],
                "maxerr": None,
                "cosim": None,
                "error_message": "TIMEOUT (exceeded 30 seconds)",
                "error_type": "TimeoutError",
                "shapes": metadata["shapes"],
                "duration_ms": duration,
                "last_tested": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")
            }
            flush_results_to_disk()
            
        return True # Handled protocol, don't run test in parent process
        
    return None # Fallback to standard execution

def pytest_sessionfinish(session, exitstatus):
    global _SESSION_RESULTS, _RESULTS_DIR, _HARDWARE_KEY, _BASELINE_RESULTS
    global _ARTIFACT_WRITES_ENABLED

    if not _ARTIFACT_WRITES_ENABLED:
        return
    
    # Save the latest JSON file
    flush_results_to_disk()
    
    # Load the completed latest.json
    latest_path = os.path.join(_RESULTS_DIR, f"{_HARDWARE_KEY}_latest.json")
    if os.path.exists(latest_path):
        with open(latest_path, "r", encoding="utf-8") as f:
            current_data = json.load(f)
            
        # Copy to history directory
        history_dir = os.path.join(_RESULTS_DIR, f"{_HARDWARE_KEY}_history")
        os.makedirs(history_dir, exist_ok=True)
        
        timestamp_str = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
        history_path = os.path.join(history_dir, f"{timestamp_str}.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(current_data, f, indent=2)
            
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

def pytest_pyfunc_call(pyfuncitem):
    if pyfuncitem.config.getoption("--benchmark"):
        global _DEVICE_NAME
        if pyfuncitem.get_closest_marker("benchmarkable") is None:
            pytest.skip("Benchmark mode only runs tests marked benchmarkable.")
        # Resolve argument names needed by the test function
        args_to_pass = {k: v for k, v in pyfuncitem.funcargs.items() if k in pyfuncitem._fixtureinfo.argnames}
        
        # Warmup (10 iterations)
        for _ in range(10):
            pyfuncitem.obj(**args_to_pass)
            
        # Repetitions (100 iterations)
        import time
        import numpy as np
        latencies = []
        for _ in range(100):
            clear_metrics()
            start = time.perf_counter()
            pyfuncitem.obj(**args_to_pass)
            synchronize(_DEVICE_NAME)
            latencies.append(time.perf_counter() - start)
            
        # Compute stats
        latencies_ms = np.array(latencies) * 1000
        median = float(np.median(latencies_ms))
        min_val = float(np.min(latencies_ms))
        max_val = float(np.max(latencies_ms))
        std = float(np.std(latencies_ms))
        
        pyfuncitem.bench_stats = {
            "median_ms": median,
            "min_ms": min_val,
            "max_ms": max_val,
            "std_ms": std,
            "repetitions": 100
        }
        
        print(f"\nBENCHMARK [{pyfuncitem.name}]: Median: {median:.3f} ms, Min: {min_val:.3f} ms, Max: {max_val:.3f} ms, Std: {std:.3f} ms")
        return True # Handled execution
    return None

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
        import re
        msg = str(exc_val)
        matched = False
        for pattern in _HARDWARE_UNSUPPORTED_PATTERNS:
            if re.search(pattern, msg):
                matched = True
                break
        if matched:
            # Mark this item as a hardware unsupported skip
            item._hardware_unsupported_reason = f"Hardware/Framework unsupported: {msg}"
            # Clear the failed outcome before raising skip to avoid PluggyTeardownRaisedWarning
            outcome.force_result(None)
            pytest.skip(item._hardware_unsupported_reason)
