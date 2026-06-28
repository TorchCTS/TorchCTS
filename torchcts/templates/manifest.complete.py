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

# Description: broadest template, intended for explicit opt-in only

import torch

manifest = {
    "manifest_version": 1,
    "device_name": "auto",
    "backend_import": None,
    "supported_dtypes": {
        torch.float16: True,
        torch.float32: True,
        torch.float64: True,
        torch.bfloat16: True,
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
    "ieee754_seed": 67,
    "max_samples": 10,            # Max passing samples per test node (clean tier). 0 = no cap.
    "max_samples_ieee754": 3,     # Max passing samples per test node (NaN/Inf tiers). 0 = no cap.
    "semantic_level": 8,          # Full release-depth conformance.
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
        "nested": True,
        "named_tensor": True,
        "foreach": True,
        "fp8": True,
        "quantized_container_plumbing": True,
        "native_quantization": True,
        "custom_quantized_decode": False,
        "compile": True,
        "pinned_memory": True,
        "streams": True,
        "events": True,
        "deterministic": True,
        "guard_alloc": True,
        "device_api": True,
        "multi_device": False,
        "ieee754": True,
    },
    "skip_ops": [],
    "tolerance_overrides": {},
    "supported_container_formats": {
        # 2-bit
        "int2_ternary": True,
        # 4-bit
        "int4_symmetric": True,
        "int4_asymmetric": True,
        "uint4": True,
        "nf4": True,
        "fp4_e2m1": True,
        "fp4_bnb": True,
        "af4": True,
        "mxfp4": True,
        "nvfp4": True,
        # 6-bit
        "fp6_e3m2": True,
        "fp6_e2m3": True,
        "mxfp6_e3m2": True,
        "mxfp6_e2m3": True,
        # 8-bit
        "fp8_e4m3fn": True,
        "fp8_e5m2": True,
        "fp8_e4m3fnuz": True,
        "fp8_e5m2fnuz": True,
        "int8_symmetric": True,
        "int8_asymmetric": True,
        "uint8": True,
        "e8m0fnu": True,
        "mxfp8_e4m3": True,
        "mxfp8_e5m2": True,
        "mxint8": True,
    },
    # Optional semantic decode hooks for custom_quantized_decode.
    # Values are "module:function" callables:
    # decode(packed, scale, zero_point, shape, dtype, device) -> Tensor.
    "custom_container_decoders": {},
    "custom_test_dirs": [],
}
