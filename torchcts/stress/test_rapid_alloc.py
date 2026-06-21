import pytest
import torch
import random

@pytest.mark.stress
@pytest.mark.parametrize("num_iterations", [100, 500])
def test_rapid_alloc_free(num_iterations, device):
    # Stress the allocator with dynamic sizes in a loop
    tensors = []
    sizes = [1024, 2048, 4096, 8192, 16384, 32768]
    
    for _ in range(num_iterations):
        sz = random.choice(sizes)
        # Allocate
        t = torch.randn(sz, device=device)
        tensors.append(t)
        
        # Randomly release some
        if len(tensors) > 20:
            idx = random.randint(0, len(tensors) - 1)
            del tensors[idx]
            
    # Clean up remaining
    tensors.clear()
