import pytest
import torch

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scheduler_name", ["StepLR", "ExponentialLR"])
def test_lr_schedulers(scheduler_name, dtype, device):
    w = torch.randn(2, 2, dtype=dtype, device=device, requires_grad=True)
    opt = torch.optim.SGD([w], lr=0.1)
    
    if scheduler_name == "StepLR":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    elif scheduler_name == "ExponentialLR":
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.5)
    
    # Check step
    sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.05)
