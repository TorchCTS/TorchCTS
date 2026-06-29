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

from dataclasses import dataclass
import torch

@dataclass
class Tol:
    rtol: float
    atol: float

    def kw(self):
        return {"rtol": self.rtol, "atol": self.atol}

    def scaled(self, factor):
        return Tol(self.rtol * factor, self.atol * factor)

@dataclass
class TieredTol:
    """Two-tier tolerance: golden (quality target) and usable (practically correct)."""
    golden: Tol
    usable: Tol

    def get(self, tier="golden"):
        return self.golden if tier == "golden" else self.usable

    def kw(self, tier="golden"):
        return self.get(tier).kw()

# Default tolerances for each category and dtype
DEFAULT_TOLERANCES = {
    # ── exact ──
    ("exact", torch.float64): TieredTol(golden=Tol(0.0, 0.0), usable=Tol(0.0, 0.0)),
    ("exact", torch.float32): TieredTol(golden=Tol(0.0, 0.0), usable=Tol(0.0, 0.0)),
    ("exact", torch.float16): TieredTol(golden=Tol(0.0, 0.0), usable=Tol(0.0, 0.0)),
    ("exact", torch.bfloat16): TieredTol(golden=Tol(0.0, 0.0), usable=Tol(0.0, 0.0)),
    
    # ── elementwise ──  (usable = golden * 5)
    ("elementwise", torch.float64): TieredTol(golden=Tol(1e-7, 1e-7), usable=Tol(5e-7, 5e-7)),
    ("elementwise", torch.float32): TieredTol(golden=Tol(1e-5, 1e-5), usable=Tol(5e-5, 5e-5)),
    ("elementwise", torch.float16): TieredTol(golden=Tol(1e-3, 1e-3), usable=Tol(5e-3, 5e-3)),
    ("elementwise", torch.bfloat16): TieredTol(golden=Tol(1e-2, 1e-2), usable=Tol(5e-2, 5e-2)),
    
    # ── reduction ──  (usable = golden * 5)
    ("reduction", torch.float64): TieredTol(golden=Tol(1e-6, 1e-6), usable=Tol(5e-6, 5e-6)),
    ("reduction", torch.float32): TieredTol(golden=Tol(1e-4, 1e-4), usable=Tol(5e-4, 5e-4)),
    ("reduction", torch.float16): TieredTol(golden=Tol(1e-2, 1e-2), usable=Tol(5e-2, 5e-2)),
    ("reduction", torch.bfloat16): TieredTol(golden=Tol(2e-2, 5e-2), usable=Tol(1e-1, 2.5e-1)),
    
    # ── matmul ──  (usable = golden * 5)
    ("matmul", torch.float64): TieredTol(golden=Tol(1e-7, 1e-7), usable=Tol(5e-7, 5e-7)),
    ("matmul", torch.float32): TieredTol(golden=Tol(1e-4, 1e-4), usable=Tol(5e-4, 5e-4)),
    ("matmul", torch.float16): TieredTol(golden=Tol(1e-2, 1e-2), usable=Tol(5e-2, 5e-2)),
    ("matmul", torch.bfloat16): TieredTol(golden=Tol(2e-2, 5e-2), usable=Tol(1e-1, 2.5e-1)),
    
    # ── matmul_backward ──  (usable = golden * 5)
    ("matmul_backward", torch.float64): TieredTol(golden=Tol(1e-6, 1e-6), usable=Tol(5e-6, 5e-6)),
    ("matmul_backward", torch.float32): TieredTol(golden=Tol(2e-4, 2e-4), usable=Tol(1e-3, 1e-3)),
    ("matmul_backward", torch.float16): TieredTol(golden=Tol(2e-2, 2e-2), usable=Tol(1e-1, 1e-1)),
    ("matmul_backward", torch.bfloat16): TieredTol(golden=Tol(3e-2, 8e-2), usable=Tol(1.5e-1, 4e-1)),
    
    # ── conv ──  (usable = golden * 5)
    ("conv", torch.float64): TieredTol(golden=Tol(1e-6, 1e-6), usable=Tol(5e-6, 5e-6)),
    ("conv", torch.float32): TieredTol(golden=Tol(2e-4, 2e-4), usable=Tol(1e-3, 1e-3)),
    ("conv", torch.float16): TieredTol(golden=Tol(2e-2, 2e-2), usable=Tol(1e-1, 1e-1)),
    ("conv", torch.bfloat16): TieredTol(golden=Tol(3e-2, 8e-2), usable=Tol(1.5e-1, 4e-1)),
    
    # ── norm ──  (usable = golden * 5)
    ("norm", torch.float64): TieredTol(golden=Tol(1e-6, 1e-6), usable=Tol(5e-6, 5e-6)),
    ("norm", torch.float32): TieredTol(golden=Tol(1e-4, 1e-4), usable=Tol(5e-4, 5e-4)),
    ("norm", torch.float16): TieredTol(golden=Tol(1e-2, 1e-2), usable=Tol(5e-2, 5e-2)),
    ("norm", torch.bfloat16): TieredTol(golden=Tol(2e-2, 5e-2), usable=Tol(1e-1, 2.5e-1)),
    
    # ── sdpa ──  (usable = golden * 5)
    ("sdpa", torch.float32): TieredTol(golden=Tol(2e-4, 2e-4), usable=Tol(1e-3, 1e-3)),
    ("sdpa", torch.float16): TieredTol(golden=Tol(2e-2, 2e-2), usable=Tol(1e-1, 1e-1)),
    ("sdpa", torch.bfloat16): TieredTol(golden=Tol(3e-2, 8e-2), usable=Tol(1.5e-1, 4e-1)),
    
    # ── loss ──  (usable = golden * 5)
    ("loss", torch.float64): TieredTol(golden=Tol(1e-7, 1e-7), usable=Tol(5e-7, 5e-7)),
    ("loss", torch.float32): TieredTol(golden=Tol(1e-4, 1e-4), usable=Tol(5e-4, 5e-4)),
    ("loss", torch.float16): TieredTol(golden=Tol(1e-2, 1e-2), usable=Tol(5e-2, 5e-2)),
    ("loss", torch.bfloat16): TieredTol(golden=Tol(2e-2, 5e-2), usable=Tol(1e-1, 2.5e-1)),
    
    # ── loss_prod ──  (usable = golden * 5)
    ("loss_prod", torch.float32): TieredTol(golden=Tol(5e-4, 5e-4), usable=Tol(2.5e-3, 2.5e-3)),
    ("loss_prod", torch.float16): TieredTol(golden=Tol(3e-2, 3e-2), usable=Tol(1.5e-1, 1.5e-1)),
    ("loss_prod", torch.bfloat16): TieredTol(golden=Tol(5e-2, 1e-1), usable=Tol(2.5e-1, 5e-1)),
    
    # ── optimizer ──  (usable = golden * 5)
    ("optimizer", torch.float64): TieredTol(golden=Tol(1e-7, 1e-7), usable=Tol(5e-7, 5e-7)),
    ("optimizer", torch.float32): TieredTol(golden=Tol(1e-4, 1e-4), usable=Tol(5e-4, 5e-4)),
    ("optimizer", torch.float16): TieredTol(golden=Tol(1e-2, 1e-2), usable=Tol(5e-2, 5e-2)),
    ("optimizer", torch.bfloat16): TieredTol(golden=Tol(2e-2, 5e-2), usable=Tol(1e-1, 2.5e-1)),
    
    # ── linalg ──  (usable = golden * 5)
    ("linalg", torch.float64): TieredTol(golden=Tol(1e-6, 1e-6), usable=Tol(5e-6, 5e-6)),
    ("linalg", torch.float32): TieredTol(golden=Tol(1e-3, 1e-3), usable=Tol(5e-3, 5e-3)),
    ("linalg", torch.float16): TieredTol(golden=Tol(1e-2, 1e-2), usable=Tol(5e-2, 5e-2)),
    ("linalg", torch.bfloat16): TieredTol(golden=Tol(2e-2, 5e-2), usable=Tol(1e-1, 2.5e-1)),
    
    # ── fft ──  (usable = golden * 5)
    ("fft", torch.float64): TieredTol(golden=Tol(1e-7, 1e-7), usable=Tol(5e-7, 5e-7)),
    ("fft", torch.float32): TieredTol(golden=Tol(1e-4, 1e-4), usable=Tol(5e-4, 5e-4)),
    ("fft", torch.float16): TieredTol(golden=Tol(1e-2, 1e-2), usable=Tol(5e-2, 5e-2)),
    
    # ── quant_decode ──  (usable = golden * 5)
    ("quant_decode", torch.float32): TieredTol(golden=Tol(1e-4, 1e-4), usable=Tol(5e-4, 5e-4)),
    ("quant_decode", torch.float16): TieredTol(golden=Tol(1e-2, 1e-2), usable=Tol(5e-2, 5e-2)),
    ("quant_decode", torch.bfloat16): TieredTol(golden=Tol(2e-2, 5e-2), usable=Tol(1e-1, 2.5e-1)),
    
    # ── dequant_matmul ──  (usable = golden * 5)
    ("dequant_matmul", torch.float32): TieredTol(golden=Tol(2e-3, 2e-3), usable=Tol(1e-2, 1e-2)),
    ("dequant_matmul", torch.float16): TieredTol(golden=Tol(2e-2, 2e-2), usable=Tol(1e-1, 1e-1)),
    ("dequant_matmul", torch.bfloat16): TieredTol(golden=Tol(3e-2, 8e-2), usable=Tol(1.5e-1, 4e-1)),
    
    # ── backward ──  (usable = golden * 5)
    ("backward", torch.float64): TieredTol(golden=Tol(1e-6, 1e-6), usable=Tol(5e-6, 5e-6)),
    ("backward", torch.float32): TieredTol(golden=Tol(1e-4, 1e-4), usable=Tol(5e-4, 5e-4)),
    ("backward", torch.float16): TieredTol(golden=Tol(1e-2, 1e-2), usable=Tol(5e-2, 5e-2)),
    ("backward", torch.bfloat16): TieredTol(golden=Tol(2e-2, 5e-2), usable=Tol(1e-1, 2.5e-1)),
    
    # ── compile ──  (usable = golden * 3)
    ("compile", torch.float64): TieredTol(golden=Tol(1e-7, 1e-7), usable=Tol(3e-7, 3e-7)),
    ("compile", torch.float32): TieredTol(golden=Tol(1e-5, 1e-5), usable=Tol(3e-5, 3e-5)),
    ("compile", torch.float16): TieredTol(golden=Tol(1e-3, 1e-3), usable=Tol(3e-3, 3e-3)),
    ("compile", torch.bfloat16): TieredTol(golden=Tol(1e-2, 1e-2), usable=Tol(3e-2, 3e-2)),
    
    # ── copy ──  (no loosening for bit-exact ops)
    ("copy", torch.float64): TieredTol(golden=Tol(1e-7, 1e-7), usable=Tol(1e-7, 1e-7)),
    ("copy", torch.float32): TieredTol(golden=Tol(1e-5, 1e-5), usable=Tol(1e-5, 1e-5)),
    ("copy", torch.float16): TieredTol(golden=Tol(1e-3, 1e-3), usable=Tol(1e-3, 1e-3)),
    ("copy", torch.bfloat16): TieredTol(golden=Tol(1e-2, 1e-2), usable=Tol(1e-2, 1e-2)),
    
    # ── serialization ──  (no loosening for bit-exact ops)
    ("serialization", torch.float64): TieredTol(golden=Tol(0.0, 0.0), usable=Tol(0.0, 0.0)),
    ("serialization", torch.float32): TieredTol(golden=Tol(0.0, 0.0), usable=Tol(0.0, 0.0)),
    ("serialization", torch.float16): TieredTol(golden=Tol(0.0, 0.0), usable=Tol(0.0, 0.0)),
    ("serialization", torch.bfloat16): TieredTol(golden=Tol(0.0, 0.0), usable=Tol(0.0, 0.0)),
    
    # ── statistical ──  (usable = golden * 5)
    ("statistical", torch.float32): TieredTol(golden=Tol(5e-2, 5e-2), usable=Tol(2.5e-1, 2.5e-1)),
    
    # ── noncontiguous_mm ──  (usable = golden * 5)
    ("noncontiguous_mm", torch.float32): TieredTol(golden=Tol(2e-4, 2e-4), usable=Tol(1e-3, 1e-3)),
    ("noncontiguous_mm", torch.float16): TieredTol(golden=Tol(2e-2, 2e-2), usable=Tol(1e-1, 1e-1)),
    ("noncontiguous_mm", torch.bfloat16): TieredTol(golden=Tol(3e-2, 8e-2), usable=Tol(1.5e-1, 4e-1)),
    
    # ── strided_reduction ──  (usable = golden * 5)
    ("strided_reduction", torch.float32): TieredTol(golden=Tol(2e-4, 2e-4), usable=Tol(1e-3, 1e-3)),
    ("strided_reduction", torch.float16): TieredTol(golden=Tol(2e-2, 2e-2), usable=Tol(1e-1, 1e-1)),
    ("strided_reduction", torch.bfloat16): TieredTol(golden=Tol(3e-2, 8e-2), usable=Tol(1.5e-1, 4e-1)),
    
    # ── workload_e2e ──  (usable = golden * 5)
    ("workload_e2e", torch.float32): TieredTol(golden=Tol(1e-3, 1e-3), usable=Tol(5e-3, 5e-3)),
    ("workload_e2e", torch.float16): TieredTol(golden=Tol(1e-2, 1e-2), usable=Tol(5e-2, 5e-2)),
    ("workload_e2e", torch.bfloat16): TieredTol(golden=Tol(2e-2, 5e-2), usable=Tol(1e-1, 2.5e-1)),
    
    # ── nested_sdpa ──  (usable = golden * 5)
    ("nested_sdpa", torch.float32): TieredTol(golden=Tol(2e-4, 2e-4), usable=Tol(1e-3, 1e-3)),
    ("nested_sdpa", torch.float16): TieredTol(golden=Tol(2e-2, 2e-2), usable=Tol(1e-1, 1e-1)),
    ("nested_sdpa", torch.bfloat16): TieredTol(golden=Tol(3e-2, 1e-1), usable=Tol(1.5e-1, 5e-1)),
    
    # ── gqa_sdpa ──  (usable = golden * 5)
    ("gqa_sdpa", torch.float32): TieredTol(golden=Tol(2e-4, 2e-4), usable=Tol(1e-3, 1e-3)),
    ("gqa_sdpa", torch.float16): TieredTol(golden=Tol(2e-2, 2e-2), usable=Tol(1e-1, 1e-1)),
    ("gqa_sdpa", torch.bfloat16): TieredTol(golden=Tol(3e-2, 8e-2), usable=Tol(1.5e-1, 4e-1)),
    
    # ── native_quantization ──  (usable = golden * 5)
    ("native_quantization", torch.float32): TieredTol(golden=Tol(1e-4, 1e-4), usable=Tol(5e-4, 5e-4)),
    
    # ── grid_sample ──  (usable = golden * 5)
    ("grid_sample", torch.float64): TieredTol(golden=Tol(1e-6, 1e-6), usable=Tol(5e-6, 5e-6)),
    ("grid_sample", torch.float32): TieredTol(golden=Tol(1e-4, 1e-4), usable=Tol(5e-4, 5e-4)),
    ("grid_sample", torch.float16): TieredTol(golden=Tol(1e-2, 1e-2), usable=Tol(5e-2, 5e-2)),
    ("grid_sample", torch.bfloat16): TieredTol(golden=Tol(2e-2, 5e-2), usable=Tol(1e-1, 2.5e-1)),
}

# FP8 default fallback values  (usable = golden * 3)
for dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz, torch.float8_e5m2fnuz):
    DEFAULT_TOLERANCES[("default", dtype)] = TieredTol(golden=Tol(1e-1, 1e-1), usable=Tol(3e-1, 3e-1))
    DEFAULT_TOLERANCES[("matmul", dtype)] = TieredTol(golden=Tol(1e-1, 1e-1), usable=Tol(3e-1, 3e-1))
    DEFAULT_TOLERANCES[("copy", dtype)] = TieredTol(golden=Tol(1e-2, 1e-2), usable=Tol(1e-2, 1e-2))

def _dtype_from_override_key(dtype_key):
    if isinstance(dtype_key, torch.dtype):
        return dtype_key
    if isinstance(dtype_key, str):
        name = dtype_key.strip()
        if name.startswith("torch."):
            name = name[len("torch."):]
        value = getattr(torch, name, None)
        if isinstance(value, torch.dtype):
            return value
    return None

def normalize_tolerance_overrides(manifest_overrides):
    if not manifest_overrides:
        return {}

    normalized = {}
    for override_key, val in manifest_overrides.items():
        if isinstance(override_key, (tuple, list)) and len(override_key) == 2:
            category, dtype_key = override_key
        elif isinstance(override_key, str):
            if ":" in override_key:
                category, dtype_key = override_key.split(":", 1)
            elif "/" in override_key:
                category, dtype_key = override_key.split("/", 1)
            else:
                raise ValueError(
                    "tolerance_overrides string keys must use 'category:dtype' "
                    f"or 'category/dtype', got {override_key!r}"
                )
            category = category.strip()
            dtype_key = dtype_key.strip()
        else:
            raise ValueError(f"Invalid tolerance override key: {override_key!r}")

        if not isinstance(category, str) or not category:
            raise ValueError(f"Invalid tolerance override category: {category!r}")

        dtype = _dtype_from_override_key(dtype_key)
        if dtype is None:
            raise ValueError(f"Invalid tolerance override dtype: {dtype_key!r}")

        normalized[(category, dtype)] = val
    return normalized

def _override_to_tol(val, default_tol, tier):
    if isinstance(val, TieredTol):
        return val.get(tier)
    if isinstance(val, Tol):
        return val
    if isinstance(val, dict):
        return Tol(val.get("rtol", default_tol.rtol), val.get("atol", default_tol.atol))
    if isinstance(val, (tuple, list)) and len(val) == 2:
        return Tol(val[0], val[1])
    raise ValueError(f"Invalid tolerance override value: {val!r}")

def get_tolerance(category, dtype, tier="golden", manifest_overrides=None):
    # Integer/bool default is exact
    if dtype in (torch.int64, torch.int32, torch.int16, torch.int8,
                 torch.uint8, torch.uint16, torch.uint32, torch.uint64, torch.bool):
        tol = Tol(0.0, 0.0)
    else:
        key = (category, dtype)
        if key in DEFAULT_TOLERANCES:
            entry = DEFAULT_TOLERANCES[key]
        else:
            default_key = ("default", dtype)
            if default_key in DEFAULT_TOLERANCES:
                entry = DEFAULT_TOLERANCES[default_key]
            else:
                # Absolute fallback
                entry = TieredTol(golden=Tol(1e-3, 1e-3), usable=Tol(5e-3, 5e-3))
        
        # Unwrap TieredTol if needed
        if isinstance(entry, TieredTol):
            tol = entry.get(tier)
        else:
            tol = entry

    # Apply manifest overrides if present
    if manifest_overrides:
        for (ok_cat, ok_dt), val in normalize_tolerance_overrides(manifest_overrides).items():
            if ok_cat == category and ok_dt == dtype:
                return _override_to_tol(val, tol, tier)
                
    return tol
