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

# Description: fast broad sweep: every op, float32 only, 1 sample, no NaN/Inf

import torch

manifest = {
    "manifest_version": 1,
    "device_name": "auto",
    "backend_import": None,
    "supported_dtypes": {
        torch.float32: True,
    },
    "device_count": 1,
    "ieee754_seed": 67,
    "max_samples": 1,             # 1 sample per op — just prove it runs
    "max_samples_ieee754": 1,     # Irrelevant (ieee754 disabled) but set for consistency
    "hardware": {
        "memory_model": "discrete",
        "device_memory_gb": "auto",
        "system_memory_gb": "auto",
        "oom_recoverable": True,
    },
    "resource_limits": {
        "max_device_memory_mb": None,
        "max_system_memory_mb": None,
        "max_tensor_size_mb": None,
        "cleanup_threshold_pct": 80,
    },
    "capabilities": {
        "inference": True,
        "training": False,
        "serialization": False,
        "generator": False,
        "double_backward": False,
        "gradcheck": False,
        "gradient_checkpointing": False,
        "autocast": False,
        "fused_optimizer": False,
        "dataloader": False,
        "module_hooks": False,
        "channels_last": False,
        "sparse": False,
        "nested": False,
        "foreach": False,
        "fp8": False,
        "quantized": False,
        "compile": False,
        "pinned_memory": False,
        "streams": False,
        "events": False,
        "deterministic": False,
        "guard_alloc": False,
        "ieee754": False,
    },
    "skip_ops": [],
    "tolerance_overrides": {},
    "supported_container_formats": {},
    "custom_container_decoders": {},
    "reference_device": None,
    "custom_test_dirs": [],
}
