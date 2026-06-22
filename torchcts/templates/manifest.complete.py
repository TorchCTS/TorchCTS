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

import torch

manifest = {
    "manifest_version": 1,
    "device_name": "auto",
    "backend_import": None,
    "supported_dtypes": {
        torch.float16: True,
        torch.float32: True,
        torch.float64: True,
        torch.bfloat16: r"^(?!aten\.linalg_|aten\.fft_)",
        torch.int8: True,
        torch.int16: True,
        torch.int32: True,
        torch.int64: True,
        torch.uint8: True,
        torch.uint16: True,
        torch.uint32: True,
        torch.uint64: True,
        torch.bool: True,
        torch.complex32: True,
        torch.complex64: True,
        torch.complex128: True,
    },
    "device_count": 1,  # Opt in to multi-device only after runtime validation on real hardware.
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
        "fp8": False,
        "quantized": False,
        "compile": True,
        "pinned_memory": True,
        "streams": False,
        "events": False,
        "deterministic": True,
        "guard_alloc": False,
    },
    "skip_ops": [
        "aten.grid_sampler",
        "aten._weight_norm_interface",
    ],
    "tolerance_overrides": {},
    "supported_container_formats": {
        "int4_symmetric": False,
        "int4_asymmetric": False,
        "nf4": False,
        "fp4_e2m1": False,
        "fp4_bnb": False,
        "fp6_e3m2": False,
        "fp6_e2m3": False,
        "int2_ternary": False,
        "e8m0fnu": False,
    },
    "custom_container_decoders": {},
    "reference_device": None,
    "custom_test_dirs": [],
}
