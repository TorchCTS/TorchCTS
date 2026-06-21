import pytest
import torch
from torch.autograd import gradcheck

@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.requires("gradcheck")
@pytest.mark.parametrize("op_name", ["pow", "sin", "exp", "sigmoid"])
def test_gradcheck_ops(op_name, device):
    # float64 support is verified at collection time via conftest.
    # If we get here, the manifest declares float64 as supported.
    x = torch.randn(2, 2, dtype=torch.float64, device=device, requires_grad=True)
        
    if op_name == "pow":
        func = lambda inputs: inputs.pow(3).sum()
    elif op_name == "sin":
        func = lambda inputs: torch.sin(inputs).sum()
    elif op_name == "exp":
        func = lambda inputs: torch.exp(inputs).sum()
    elif op_name == "sigmoid":
        func = lambda inputs: torch.sigmoid(inputs).sum()
        
    assert gradcheck(func, (x,), eps=1e-6, atol=1e-4)
