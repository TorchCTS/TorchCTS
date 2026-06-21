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
