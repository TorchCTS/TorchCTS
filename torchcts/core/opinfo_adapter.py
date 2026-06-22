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
import re
import torch

try:
    import fcntl
except ImportError:
    fcntl = None

DTYPE_TO_STR = {
    torch.float32: "torch.float32",
    torch.float64: "torch.float64",
    torch.float16: "torch.float16",
    torch.bfloat16: "torch.bfloat16",
    torch.int64: "torch.int64",
    torch.int32: "torch.int32",
    torch.int16: "torch.int16",
    torch.int8: "torch.int8",
    torch.uint8: "torch.uint8",
    torch.bool: "torch.bool",
    torch.complex32: "torch.complex32",
    torch.complex64: "torch.complex64",
    torch.complex128: "torch.complex128",
    torch.float8_e4m3fn: "torch.float8_e4m3fn",
    torch.float8_e5m2: "torch.float8_e5m2",
    torch.float8_e4m3fnuz: "torch.float8_e4m3fnuz",
    torch.float8_e5m2fnuz: "torch.float8_e5m2fnuz",
}
STR_TO_DTYPE = {v: k for k, v in DTYPE_TO_STR.items()}

def dtype_to_str(dt):
    return DTYPE_TO_STR.get(dt, str(dt))

def str_to_dtype(s):
    if s.startswith("torch."):
        name = s.replace("torch.", "")
        if hasattr(torch, name):
            return getattr(torch, name)
    return STR_TO_DTYPE.get(s, None)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Dtypes that can carry gradients
DIFFERENTIABLE = frozenset({
    torch.float16, torch.float32, torch.float64, torch.bfloat16,
    torch.complex32, torch.complex64, torch.complex128,
})

# Ops that only work on CUDA — skip everywhere else
SKIP_OPS = frozenset({
    "jiterator_unary", "jiterator_binary",
    "jiterator_4inputs_with_extra_args",
    "jiterator_binary_return_by_ref",
    "jiterator_2inputs_2outputs",
})

# ---------------------------------------------------------------------------
# Adaptive known-failures cache
# ---------------------------------------------------------------------------

_KNOWN_FAILURES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "opinfo_cache", "known_failures.json"
)

_known_failures_mem = None  # in-memory cache, loaded once per process

def load_known_failures():
    """Load known CPU reference failures for the current PyTorch version."""
    global _known_failures_mem
    if _known_failures_mem is not None:
        return _known_failures_mem

    version = torch.__version__
    if os.path.exists(_KNOWN_FAILURES_PATH):
        try:
            with open(_KNOWN_FAILURES_PATH, "r", encoding="utf-8") as f:
                all_failures = json.load(f)
            _known_failures_mem = all_failures.get(version, {})
        except Exception:
            _known_failures_mem = {}
    else:
        _known_failures_mem = {}
    return _known_failures_mem


def record_known_failure(phase, op_name, dtype_str, error_msg):
    """Record a CPU reference failure. Written to disk immediately.
    
    Args:
        phase: "forward" or "backward"
        op_name: e.g. "svd_lowrank"
        dtype_str: e.g. "torch.complex128"
        error_msg: truncated error string
    """
    version = torch.__version__
    key = f"{op_name}@{dtype_str}"
    truncated = error_msg[:200]
    
    os.makedirs(os.path.dirname(_KNOWN_FAILURES_PATH), exist_ok=True)

    # Read-modify-write with file lock
    try:
        with open(_KNOWN_FAILURES_PATH, "r+", encoding="utf-8") as f:
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_EX)
            elif sys.platform == "win32":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            try:
                all_failures = json.load(f)
            except json.JSONDecodeError:
                all_failures = {}
            all_failures.setdefault(version, {}).setdefault(phase, {})
            all_failures[version][phase][key] = truncated
            f.seek(0)
            f.truncate()
            json.dump(all_failures, f, indent=2)
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_UN)
            elif sys.platform == "win32":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    except FileNotFoundError:
        all_failures = {version: {phase: {key: truncated}}}
        with open(_KNOWN_FAILURES_PATH, "w", encoding="utf-8") as f:
            json.dump(all_failures, f, indent=2)

    # Update in-memory cache
    global _known_failures_mem
    if _known_failures_mem is None:
        _known_failures_mem = {}
    _known_failures_mem.setdefault(phase, {})[key] = truncated


def is_cpu_reference_failure(exc):
    """Return True if the exception indicates CPU doesn't implement this op/dtype."""
    if isinstance(exc, NotImplementedError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "not implemented" in msg:
            return True
        if "svd_backward" in msg:
            return True
    return False

# ---------------------------------------------------------------------------
# Manifest dtype filter helper
# ---------------------------------------------------------------------------

def _dtype_matches_manifest(dt, dt_str, supported_dtypes, op_name):
    """Check if a dtype is allowed by the manifest's supported_dtypes config."""
    dtype_filter = supported_dtypes.get(dt, supported_dtypes.get(dt_str))
    if dtype_filter is None:
        return False
    if dtype_filter is True:
        return True
    if isinstance(dtype_filter, str):
        return bool(re.search(dtype_filter, op_name))
    return False

# ---------------------------------------------------------------------------
# Test list builders (replace get_filtered_op_tests + probing)
# ---------------------------------------------------------------------------

def get_forward_op_tests(manifest):
    """Build forward test list from op_db metadata + known failures. No probing."""
    import torch.testing._internal.common_methods_invocations as cmi

    supported_dtypes = manifest.get("supported_dtypes", {})
    skip_ops = set(manifest.get("skip_ops", [])) | SKIP_OPS
    capabilities = manifest.get("capabilities", {})
    known = load_known_failures().get("forward", {})
    tests = []

    seen = set()
    for op in cmi.op_db:
        if op.name in seen or op.name in skip_ops:
            continue
        seen.add(op.name)

        # Skip sparse ops if backend doesn't support sparse
        if (getattr(op, "supports_sparse", False) or
                getattr(op, "supports_sparse_csr", False)):
            if not capabilities.get("sparse", False):
                continue

        for d in op.dtypes:
            dt_str = dtype_to_str(d)

            # Rule 4: skip previously-discovered CPU failures
            if f"{op.name}@{dt_str}" in known:
                continue

            if _dtype_matches_manifest(d, dt_str, supported_dtypes, op.name):
                tests.append((op.name, dt_str))

    return tests


def get_backward_op_tests(manifest):
    """Build backward test list from op_db metadata + known failures.
    
    Static rules applied:
      1. Filter to differentiable dtypes (float/complex)
      2. Skip if supports_autograd == False
      3. Skip jiterator_* (in SKIP_OPS)
      4. Skip previously-discovered CPU failures
    """
    import torch.testing._internal.common_methods_invocations as cmi

    supported_dtypes = manifest.get("supported_dtypes", {})
    skip_ops = set(manifest.get("skip_ops", [])) | SKIP_OPS
    capabilities = manifest.get("capabilities", {})
    known = load_known_failures().get("backward", {})
    tests = []

    seen = set()
    for op in cmi.op_db:
        if op.name in seen or op.name in skip_ops:
            continue
        seen.add(op.name)

        # Rule 2: skip ops that don't support autograd
        if not getattr(op, "supports_autograd", True):
            continue

        if (getattr(op, "supports_sparse", False) or
                getattr(op, "supports_sparse_csr", False)):
            if not capabilities.get("sparse", False):
                continue

        for d in op.backward_dtypes:
            # Rule 1: only differentiable dtypes
            if d not in DIFFERENTIABLE:
                continue

            dt_str = dtype_to_str(d)

            # Rule 4: skip previously-discovered CPU failures
            if f"{op.name}@{dt_str}" in known:
                continue

            if _dtype_matches_manifest(d, dt_str, supported_dtypes, op.name):
                tests.append((op.name, dt_str))

    return tests


def get_error_op_tests(manifest):
    """Build error test list. Calls op.error_inputs at collection time (fast, no tensor ops)."""
    import torch.testing._internal.common_methods_invocations as cmi

    skip_ops = set(manifest.get("skip_ops", [])) | SKIP_OPS
    ops = []

    seen = set()
    for op in cmi.op_db:
        if op.name in seen or op.name in skip_ops:
            continue
        seen.add(op.name)

        if not hasattr(op, "error_inputs_func") or op.error_inputs_func is None:
            continue
        try:
            errs = list(op.error_inputs("cpu"))
            if errs:
                ops.append(op.name)
        except Exception:
            pass

    return ops

# ---------------------------------------------------------------------------
# Backward dtype helper
# ---------------------------------------------------------------------------

def get_backward_dtypes(op_name):
    """Get backward-compatible dtypes from OpInfo metadata. No probing."""
    op = get_live_opinfo(op_name)
    if op is None:
        return []
    if not getattr(op, "supports_autograd", True):
        return []
    return [d for d in op.backward_dtypes if d in DIFFERENTIABLE]

# ---------------------------------------------------------------------------
# Live OpInfo access
# ---------------------------------------------------------------------------

_live_opinfo_cache = {}

def get_live_opinfo(op_name):
    if op_name in _live_opinfo_cache:
        return _live_opinfo_cache[op_name]
    try:
        import torch.testing._internal.common_methods_invocations as cmi
        for op in cmi.op_db:
            if op.name == op_name:
                _live_opinfo_cache[op_name] = op
                return op
    except Exception as e:
        print(f"Error loading live generator for {op_name}: {e}", file=sys.stderr)
    _live_opinfo_cache[op_name] = None
    return None


def get_op_sample_inputs(op_name, device, dtype):
    op = get_live_opinfo(op_name)
    if op is None:
        return
        
    try:
        samples = op.sample_inputs(device, dtype, requires_grad=False)
        for sample in samples:
            yield sample
    except Exception as e:
        # Check if this is a hardware/framework unsupported exception
        import re
        import torchcts.conftest as conftest
        patterns = getattr(conftest, "_HARDWARE_UNSUPPORTED_PATTERNS", [])
        msg = str(e)
        if any(re.search(pat, msg) for pat in patterns):
            raise e
        # Log non-hardware sample generation failures so they're visible for debugging
        print(f"Warning: sample_inputs for {op_name}({dtype}) failed: {type(e).__name__}: {e}", file=sys.stderr)


def get_op_error_inputs(op_name, device):
    op = get_live_opinfo(op_name)
    if op is None:
        return
        
    try:
        # error_inputs accepts device as positional argument
        errors = op.error_inputs(device)
        for err in errors:
            yield err
    except Exception as e:
        pass
