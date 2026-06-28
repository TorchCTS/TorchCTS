# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest
import torch
import os
from torchcts.core.device import synchronize

pytestmark = pytest.mark.covers_category("gradient_checkpointing")

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
