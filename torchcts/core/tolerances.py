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

# Default tolerances for each category and dtype
DEFAULT_TOLERANCES = {
    # ── exact ──
    ("exact", torch.float64): Tol(0.0, 0.0),
    ("exact", torch.float32): Tol(0.0, 0.0),
    ("exact", torch.float16): Tol(0.0, 0.0),
    ("exact", torch.bfloat16): Tol(0.0, 0.0),
    
    # ── elementwise ──
    ("elementwise", torch.float64): Tol(1e-7, 1e-7),
    ("elementwise", torch.float32): Tol(1e-5, 1e-5),
    ("elementwise", torch.float16): Tol(1e-3, 1e-3),
    ("elementwise", torch.bfloat16): Tol(1e-2, 1e-2),
    
    # ── reduction ──
    ("reduction", torch.float64): Tol(1e-6, 1e-6),
    ("reduction", torch.float32): Tol(1e-4, 1e-4),
    ("reduction", torch.float16): Tol(1e-2, 1e-2),
    ("reduction", torch.bfloat16): Tol(2e-2, 5e-2),
    
    # ── matmul ──
    ("matmul", torch.float64): Tol(1e-7, 1e-7),
    ("matmul", torch.float32): Tol(1e-4, 1e-4),
    ("matmul", torch.float16): Tol(1e-2, 1e-2),
    ("matmul", torch.bfloat16): Tol(2e-2, 5e-2),
    
    # ── matmul_backward ──
    ("matmul_backward", torch.float64): Tol(1e-6, 1e-6),
    ("matmul_backward", torch.float32): Tol(2e-4, 2e-4),
    ("matmul_backward", torch.float16): Tol(2e-2, 2e-2),
    ("matmul_backward", torch.bfloat16): Tol(3e-2, 8e-2),
    
    # ── conv ──
    ("conv", torch.float64): Tol(1e-6, 1e-6),
    ("conv", torch.float32): Tol(2e-4, 2e-4),
    ("conv", torch.float16): Tol(2e-2, 2e-2),
    ("conv", torch.bfloat16): Tol(3e-2, 8e-2),
    
    # ── norm ──
    ("norm", torch.float64): Tol(1e-6, 1e-6),
    ("norm", torch.float32): Tol(1e-4, 1e-4),
    ("norm", torch.float16): Tol(1e-2, 1e-2),
    ("norm", torch.bfloat16): Tol(2e-2, 5e-2),
    
    # ── sdpa ──
    ("sdpa", torch.float32): Tol(2e-4, 2e-4),
    ("sdpa", torch.float16): Tol(2e-2, 2e-2),
    ("sdpa", torch.bfloat16): Tol(3e-2, 8e-2),
    
    # ── loss ──
    ("loss", torch.float64): Tol(1e-7, 1e-7),
    ("loss", torch.float32): Tol(1e-4, 1e-4),
    ("loss", torch.float16): Tol(1e-2, 1e-2),
    ("loss", torch.bfloat16): Tol(2e-2, 5e-2),
    
    # ── loss_prod ──
    ("loss_prod", torch.float32): Tol(5e-4, 5e-4),
    ("loss_prod", torch.float16): Tol(3e-2, 3e-2),
    ("loss_prod", torch.bfloat16): Tol(5e-2, 1e-1),
    
    # ── optimizer ──
    ("optimizer", torch.float64): Tol(1e-7, 1e-7),
    ("optimizer", torch.float32): Tol(1e-4, 1e-4),
    ("optimizer", torch.float16): Tol(1e-2, 1e-2),
    ("optimizer", torch.bfloat16): Tol(2e-2, 5e-2),
    
    # ── linalg ──
    ("linalg", torch.float64): Tol(1e-6, 1e-6),
    ("linalg", torch.float32): Tol(1e-3, 1e-3),
    ("linalg", torch.float16): Tol(1e-2, 1e-2),
    ("linalg", torch.bfloat16): Tol(2e-2, 5e-2),
    
    # ── fft ──
    ("fft", torch.float64): Tol(1e-7, 1e-7),
    ("fft", torch.float32): Tol(1e-4, 1e-4),
    ("fft", torch.float16): Tol(1e-2, 1e-2),
    
    # ── quant_decode ──
    ("quant_decode", torch.float32): Tol(1e-4, 1e-4),
    ("quant_decode", torch.float16): Tol(1e-2, 1e-2),
    ("quant_decode", torch.bfloat16): Tol(2e-2, 5e-2),
    
    # ── dequant_matmul ──
    ("dequant_matmul", torch.float32): Tol(2e-3, 2e-3),
    ("dequant_matmul", torch.float16): Tol(2e-2, 2e-2),
    ("dequant_matmul", torch.bfloat16): Tol(3e-2, 8e-2),
    
    # ── backward ──
    ("backward", torch.float64): Tol(1e-6, 1e-6),
    ("backward", torch.float32): Tol(1e-4, 1e-4),
    ("backward", torch.float16): Tol(1e-2, 1e-2),
    ("backward", torch.bfloat16): Tol(2e-2, 5e-2),
    
    # ── compile ──
    ("compile", torch.float64): Tol(1e-7, 1e-7),
    ("compile", torch.float32): Tol(1e-5, 1e-5),
    ("compile", torch.float16): Tol(1e-3, 1e-3),
    ("compile", torch.bfloat16): Tol(1e-2, 1e-2),
    
    # ── copy ──
    ("copy", torch.float64): Tol(1e-7, 1e-7),
    ("copy", torch.float32): Tol(1e-5, 1e-5),
    ("copy", torch.float16): Tol(1e-3, 1e-3),
    ("copy", torch.bfloat16): Tol(1e-2, 1e-2),
    
    # ── serialization ──
    ("serialization", torch.float64): Tol(0.0, 0.0),
    ("serialization", torch.float32): Tol(0.0, 0.0),
    ("serialization", torch.float16): Tol(0.0, 0.0),
    ("serialization", torch.bfloat16): Tol(0.0, 0.0),
    
    # ── statistical ──
    ("statistical", torch.float32): Tol(5e-2, 5e-2),
    
    # ── noncontiguous_mm ──
    ("noncontiguous_mm", torch.float32): Tol(2e-4, 2e-4),
    ("noncontiguous_mm", torch.float16): Tol(2e-2, 2e-2),
    ("noncontiguous_mm", torch.bfloat16): Tol(3e-2, 8e-2),
    
    # ── strided_reduction ──
    ("strided_reduction", torch.float32): Tol(2e-4, 2e-4),
    ("strided_reduction", torch.float16): Tol(2e-2, 2e-2),
    ("strided_reduction", torch.bfloat16): Tol(3e-2, 8e-2),
    
    # ── workload_e2e ──
    ("workload_e2e", torch.float32): Tol(1e-3, 1e-3),
    ("workload_e2e", torch.float16): Tol(1e-2, 1e-2),
    ("workload_e2e", torch.bfloat16): Tol(2e-2, 5e-2),
}

# FP8 default fallback values
for dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz, torch.float8_e5m2fnuz):
    DEFAULT_TOLERANCES[("default", dtype)] = Tol(1e-1, 1e-1)
    DEFAULT_TOLERANCES[("matmul", dtype)] = Tol(1e-1, 1e-1)
    DEFAULT_TOLERANCES[("copy", dtype)] = Tol(1e-2, 1e-2)

def get_tolerance(category, dtype, manifest_overrides=None):
    # Integer/bool default is exact
    if dtype in (torch.int64, torch.int32, torch.int16, torch.int8,
                 torch.uint8, torch.uint16, torch.uint32, torch.uint64, torch.bool):
        tol = Tol(0.0, 0.0)
    else:
        key = (category, dtype)
        if key in DEFAULT_TOLERANCES:
            tol = DEFAULT_TOLERANCES[key]
        else:
            default_key = ("default", dtype)
            if default_key in DEFAULT_TOLERANCES:
                tol = DEFAULT_TOLERANCES[default_key]
            else:
                # Absolute fallback
                tol = Tol(1e-3, 1e-3)

    # Apply manifest overrides if present
    if manifest_overrides:
        # Check both direct key and stringified key
        for override_key, val in manifest_overrides.items():
            ok_cat, ok_dt = override_key
            if ok_cat == category and (ok_dt == dtype or str(ok_dt) == str(dtype)):
                if isinstance(val, Tol):
                    return val
                elif isinstance(val, dict):
                    return Tol(val.get("rtol", tol.rtol), val.get("atol", tol.atol))
                elif isinstance(val, (tuple, list)) and len(val) == 2:
                    return Tol(val[0], val[1])
                
    return tol
