# Copyright (c) 2026 Kris Bailey. MIT License.
# See LICENSE file in the project root for full license information.

"""Diagnostic engine for classifying test failures into actionable causes."""

from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class Diagnosis:
    likely_cause: str
    remediation: str
    confidence: str  # 'high', 'medium', 'low'


_RULES: list[tuple[Callable, Diagnosis]] = []


def _register(match_fn: Callable, cause: str, remediation: str, confidence: str = 'medium'):
    """Register a diagnostic rule."""
    _RULES.append((match_fn, Diagnosis(cause, remediation, confidence)))


def diagnose(error_type: str, error_message: str, exit_code: int = 0) -> Optional[Diagnosis]:
    """Match an error against registered rules and return the first diagnosis, or None."""
    for match_fn, diag in _RULES:
        if match_fn(error_type, error_message, exit_code):
            return diag
    return None


# ---------------------------------------------------------------------------
# Rule 1: Operator not registered
# ---------------------------------------------------------------------------
_register(
    lambda et, msg, ec: 'Could not run' in msg,
    cause='Operator not registered for the target backend device.',
    remediation='Register the operator in your backend dispatcher, or add it to a CPU fallback list.',
    confidence='high',
)

# ---------------------------------------------------------------------------
# Rule 2: Numerical precision
# ---------------------------------------------------------------------------
_register(
    lambda et, msg, ec: et == 'AssertionError' and any(k in msg for k in ['maxerr', 'max_abs_err', 'diff=']),
    cause='Numerical precision loss in a backend kernel.',
    remediation='Review accumulator precision (use float32 accumulators for fp16/bf16 matmul). Check tolerance overrides in the manifest.',
    confidence='medium',
)

# ---------------------------------------------------------------------------
# Rule 3: Segfault / abort
# ---------------------------------------------------------------------------
_register(
    lambda et, msg, ec: ec in (-11, 139, -6, 134),
    cause='Memory corruption, segmentation fault, or abort signal.',
    remediation='Run with memory validation enabled (Address Sanitizer, Metal validation, cuda-memcheck). Check kernel thread grid bounds and buffer size calculations.',
    confidence='high',
)

# ---------------------------------------------------------------------------
# Rule 4: OOM
# ---------------------------------------------------------------------------
_register(
    lambda et, msg, ec: ec in (-9, 137) or 'OOM' in msg.upper(),
    cause='Out of memory — excessive allocation or memory leak.',
    remediation='Check for memory leaks in the custom allocator. Reduce tensor sizes or increase resource_limits in the manifest.',
    confidence='high',
)

# ---------------------------------------------------------------------------
# Rule 5: Complex dtype unsupported
# ---------------------------------------------------------------------------
_register(
    lambda et, msg, ec: 'complex' in msg.lower() and 'not support' in msg.lower(),
    cause='Complex dtype not supported by the backend kernel.',
    remediation='Register a CPU fallback for complex-dtype ops, or disable complex tests in the manifest supported_dtypes.',
    confidence='high',
)

# ---------------------------------------------------------------------------
# Rule 6: Gradient mismatch
# ---------------------------------------------------------------------------
_register(
    lambda et, msg, ec: 'jacobian mismatch' in msg.lower() or 'gradcheck' in msg.lower(),
    cause='Incorrect backward derivative in a custom kernel.',
    remediation='Verify analytical gradients against finite differences. Check the autograd graph for detached tensors or missing gradient registration.',
    confidence='medium',
)

# ---------------------------------------------------------------------------
# Rule 7: Timeout / deadlock
# ---------------------------------------------------------------------------
_register(
    lambda et, msg, ec: et == 'TimeoutError' or 'TIMEOUT' in msg.upper(),
    cause='Test hung — likely a GPU synchronization deadlock or infinite loop.',
    remediation='Check for missing synchronization barriers, command buffer submission deadlocks, or blocking waits without timeouts.',
    confidence='medium',
)

# ---------------------------------------------------------------------------
# Rule 8: Non-contiguous tensor
# ---------------------------------------------------------------------------
_register(
    lambda et, msg, ec: 'is_contiguous' in msg or 'stride' in msg.lower(),
    cause='Kernel does not handle non-contiguous tensor inputs.',
    remediation='Add .contiguous() calls in the dispatcher wrapper, or update the kernel to compute with tensor strides.',
    confidence='medium',
)
