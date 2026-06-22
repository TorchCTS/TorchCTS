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

import warnings
import torch
from unittest.mock import patch

class LayoutDispatchTracker:
    def __init__(self):
        self.copy_count = 0
        self.patcher = None
        self._orig_contiguous = torch.Tensor.contiguous

    def __enter__(self):
        self.copy_count = 0
        
        def _mock_contiguous(tensor_self, *args, **kwargs):
            # If the tensor is not contiguous, calling .contiguous() triggers a copy
            if not tensor_self.is_contiguous():
                self.copy_count += 1
                warnings.warn(
                    f"Performance Warning: Silent copy to contiguous detected. "
                    f"Tensor shape: {tensor_self.shape}, strides: {tensor_self.stride()}, dtype: {tensor_self.dtype}",
                    UserWarning,
                    stacklevel=2
                )
            return self._orig_contiguous(tensor_self, *args, **kwargs)
        
        self.patcher = patch.object(torch.Tensor, "contiguous", _mock_contiguous)
        self.patcher.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.patcher:
            self.patcher.stop()
