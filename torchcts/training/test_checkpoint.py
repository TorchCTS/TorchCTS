import pytest
import torch
import os
from torchcts.core.device import synchronize

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)
    def forward(self, x):
        return self.fc(x)

@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_checkpoint_roundtrip(dtype, device, manifest, compare, tmp_path):
    model = SimpleModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    
    # 1. Save state
    state = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict()
    }
    
    checkpoint_path = os.path.join(tmp_path, "model.pt")
    torch.save(state, checkpoint_path)
    
    # 2. Load state
    loaded_state = torch.load(checkpoint_path, map_location=device)
    
    model_new = SimpleModel().to(device)
    opt_new = torch.optim.Adam(model_new.parameters(), lr=0.1)
    
    model_new.load_state_dict(loaded_state["model"])
    opt_new.load_state_dict(loaded_state["optimizer"])
    
    synchronize(device)
    
    # Verify weight equality
    compare(model_new.fc.weight, model.fc.weight, category="exact", dtype=dtype)
