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
from torchcts.core.device import synchronize

pytestmark = pytest.mark.covers_category("workload")

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

class ResBlock(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_c)
        self.conv2 = torch.nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_c)
        
    def forward(self, x):
        residual = x
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return torch.nn.functional.relu(out)

class ViTPatchEmbedding(torch.nn.Module):
    def __init__(self, in_c=3, embed_dim=64, patch_size=16):
        super().__init__()
        self.proj = torch.nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x) # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2) # (B, embed_dim, num_patches)
        x = x.transpose(1, 2) # (B, num_patches, embed_dim)
        return x

@pytest.mark.workload
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("component", ["cnn", "vit"])
def test_vision_components(component, dtype, device, manifest, compare, input_gen):

    if component == "cnn":
        # 1. CNN ResBlock
        model_cpu = ResBlock(8, 8)
        model_dev = ResBlock(8, 8).to(device)
        
        with torch.no_grad():
            for p_dev, p_cpu in zip(model_dev.parameters(), model_cpu.parameters()):
                p_dev.copy_(p_cpu)

        if dtype != torch.float32:
            model_cpu = model_cpu.to(dtype)
            model_dev = model_dev.to(dtype)
                
        x_cpu = torch.randn(2, 8, 16, 16, dtype=dtype)
        x_dev = x_cpu.to(device)
        
        expected = model_cpu(x_cpu)
        actual = model_dev(x_dev)
        synchronize(device)
        compare(actual, expected, category="workload_e2e", dtype=dtype)
        
    elif component == "vit":
        # 2. ViT Patch Embedding
        vit_cpu = ViTPatchEmbedding(3, 16, 4)
        vit_dev = ViTPatchEmbedding(3, 16, 4).to(device)
        
        with torch.no_grad():
            for p_dev, p_cpu in zip(vit_dev.parameters(), vit_cpu.parameters()):
                p_dev.copy_(p_cpu)

        if dtype != torch.float32:
            vit_cpu = vit_cpu.to(dtype)
            vit_dev = vit_dev.to(dtype)
                
        x_vit_cpu = torch.randn(2, 3, 16, 16, dtype=dtype)
        x_vit_dev = x_vit_cpu.to(device)
        
        expected_vit = vit_cpu(x_vit_cpu)
        actual_vit = vit_dev(x_vit_dev)
        synchronize(device)
        compare(actual_vit, expected_vit, category="workload_e2e", dtype=dtype)
