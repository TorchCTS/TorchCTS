import pytest
import torch
from torchcts.core.device import synchronize

FOREACH_DTYPES = [torch.float32]

@pytest.mark.smoke
@pytest.mark.requires("foreach")
@pytest.mark.parametrize("dtype", FOREACH_DTYPES)
@pytest.mark.parametrize("op_name", ["add", "mul", "lerp"])
def test_foreach_op(dtype, op_name, device, compare, input_gen):
    tensors_dev = [input_gen((8, 8), dtype, device) for _ in range(3)]
    tensors_cpu = [t.cpu() for t in tensors_dev]
    
    if op_name == "add":
        expected = torch._foreach_add(tensors_cpu, 2.5)
        actual = torch._foreach_add(tensors_dev, 2.5)
    elif op_name == "mul":
        expected = torch._foreach_mul(tensors_cpu, 1.5)
        actual = torch._foreach_mul(tensors_dev, 1.5)
    elif op_name == "lerp":
        tensors_target_dev = [input_gen((8, 8), dtype, device) for _ in range(3)]
        tensors_target_cpu = [t.cpu() for t in tensors_target_dev]
        expected = torch._foreach_lerp(tensors_cpu, tensors_target_cpu, 0.5)
        actual = torch._foreach_lerp(tensors_dev, tensors_target_dev, 0.5)
        
    synchronize(device)
    for act, exp in zip(actual, expected):
        compare(act, exp, category="elementwise", dtype=dtype)
