import pytest
import torch
from torchcts.core.device import synchronize

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

class HookModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)
    def forward(self, x):
        return self.fc(x)

@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.requires("module_hooks")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", ["hooks", "checkpointing"])
def test_module_hooks_and_gradient_checkpointing(case, dtype, device, manifest):
    if case == "hooks":
        model = HookModel().to(device)
        if dtype != torch.float32:
            model = model.to(dtype)
        
        # 1. Forward and backward hook
        forward_called = False
        backward_called = False
        
        def fwd_hook(module, inputs, outputs):
            nonlocal forward_called
            forward_called = True
            
        def bwd_hook(module, grad_input, grad_output):
            nonlocal backward_called
            backward_called = True
            
        model.register_forward_hook(fwd_hook)
        model.register_full_backward_hook(bwd_hook)
        
        x = torch.randn(2, 4, dtype=dtype, device=device, requires_grad=True)
        out = model(x).sum()
        out.backward()
        synchronize(device)
        
        assert forward_called, "Forward hook not triggered."
        assert backward_called, "Backward hook not triggered."
        
    elif case == "checkpointing":
        # 2. Gradient checkpointing
        model = HookModel().to(device)
        if dtype != torch.float32:
            model = model.to(dtype)
        x = torch.randn(2, 4, dtype=dtype, device=device, requires_grad=True)
        
        # Checkpoint forward
        out_cp = torch.utils.checkpoint.checkpoint(model, x, use_reentrant=False).sum()
        out_cp.backward()
        synchronize(device)
        
        assert x.grad is not None
