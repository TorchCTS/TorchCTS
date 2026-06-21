import pytest
import torch
from torchcts.core.device import synchronize

@pytest.mark.medium
@pytest.mark.parametrize("dtype", [torch.float32])
def test_noncontiguous_tensors(dtype, device, manifest, compare, input_gen):
    # Sliced non-contiguous layout input
    # create shape (32, 32) from (64, 64) base using slicing [::2, ::2]
    x_dev = input_gen((32, 32), dtype, device, layout="sliced")
    y_dev = input_gen((32, 32), dtype, device, layout="sliced")
    
    assert not x_dev.is_contiguous()
    assert not y_dev.is_contiguous()
    
    expected = x_dev.cpu() + y_dev.cpu()
    actual = x_dev + y_dev
    synchronize(device)
    
    compare(actual, expected, category="strided_reduction", dtype=dtype)
