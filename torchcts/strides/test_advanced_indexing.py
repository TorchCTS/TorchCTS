import pytest
import torch
from torchcts.core.device import synchronize

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.medium
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", ["mixed", "boolean"])
def test_advanced_indexing_mixed(case, dtype, device, manifest, compare, input_gen):
    x_dev = input_gen((8, 8, 8), dtype, device)
    x_cpu = x_dev.cpu()
    
    if case == "mixed":
        # 1. Mixed slice and index tensor
        idx_cpu = torch.tensor([1, 3, 5], dtype=torch.int64)
        idx_dev = idx_cpu.to(device)
        
        expected = x_cpu[:, idx_cpu, 2:6]
        actual = x_dev[:, idx_dev, 2:6]
        synchronize(device)
        compare(actual, expected, category="exact", dtype=dtype)
        
    elif case == "boolean":
        # 2. Boolean mask indexing
        mask_cpu = x_cpu[0, :, 0] > 0.0
        mask_dev = mask_cpu.to(device)
        
        expected_mask = x_cpu[0, mask_cpu, :]
        actual_mask = x_dev[0, mask_dev, :]
        synchronize(device)
        compare(actual_mask, expected_mask, category="exact", dtype=dtype)
