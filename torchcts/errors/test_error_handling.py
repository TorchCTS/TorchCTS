import pytest
import torch

@pytest.mark.smoke
@pytest.mark.parametrize("shapes", [
    ((4, 3), (5, 2)),
    ((2, 3), (4, 3))
])
def test_error_handling_shapes(shapes, device):
    shape_x, shape_y = shapes
    x = torch.randn(shape_x, device=device)
    y = torch.randn(shape_y, device=device)
    
    with pytest.raises(RuntimeError):
        torch.mm(x, y)

@pytest.mark.smoke
@pytest.mark.parametrize("op_name", ["add", "sub", "mul"])
def test_error_handling_cross_device(op_name, device):
    # CPU deselection handled at collection time in conftest.
        
    # 2. Cross-device operation must raise RuntimeError
    x_cpu = torch.randn(5)
    x_dev = torch.randn(5, device=device)
    
    op_fn = getattr(torch, op_name) if hasattr(torch, op_name) else getattr(x_cpu, f"__{op_name}__")
    
    with pytest.raises(RuntimeError):
        op_fn(x_cpu, x_dev)
