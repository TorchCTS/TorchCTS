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

# Skip if transformers is not installed
pytest.importorskip("transformers")

from transformers import AutoModelForCausalLM, AutoModel
from torchcts.workloads.model_configs import get_gpt2_config, get_bert_config, get_qwen2_config

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.workload
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("model_name", ["gpt2"])
def test_gpt2_e2e_logits(model_name, dtype, device, manifest, compare):
        
    config = get_gpt2_config()
    if config is None:
        pytest.skip("GPT-2 config not available")
        
    # Lazy load modeling classes to prevent crashes on missing dependencies
    from transformers import GPT2LMHeadModel
    
    if device == "cpu":
        torch.manual_seed(42)
        model_dev = GPT2LMHeadModel(config).eval().to(device)
        if dtype != torch.float32:
            model_dev = model_dev.to(dtype)
        input_ids_dev = torch.randint(0, 1000, (2, 32), dtype=torch.int64).to(device)
        with torch.no_grad():
            actual = model_dev(input_ids_dev).logits
        synchronize(device)
    else:
        # Initialize CPU reference
        torch.manual_seed(42)
        model_cpu = GPT2LMHeadModel(config).eval()
        
        # Initialize Device model
        torch.manual_seed(42)
        model_dev = GPT2LMHeadModel(config).eval().to(device)
        
        # Keep parameters and buffers aligned across devices.
        model_dev.load_state_dict(model_cpu.state_dict())

        if dtype != torch.float32:
            model_cpu = model_cpu.to(dtype)
            model_dev = model_dev.to(dtype)
                
        input_ids = torch.randint(0, 1000, (2, 32), dtype=torch.int64)
        input_ids_dev = input_ids.to(device)
        
        # Eager outputs
        with torch.no_grad():
            expected = model_cpu(input_ids).logits
            actual = model_dev(input_ids_dev).logits
            
        synchronize(device)
        compare(actual, expected, category="workload_e2e", dtype=dtype)

@pytest.mark.workload
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("model_name", ["qwen"])
def test_qwen_e2e_logits(model_name, dtype, device, manifest, compare):
        
    config = get_qwen2_config()
    if config is None:
        pytest.skip("Qwen2 config not available")
        
    from transformers import Qwen2ForCausalLM
    
    if device == "cpu":
        torch.manual_seed(42)
        model_dev = Qwen2ForCausalLM(config).eval().to(device)
        if dtype != torch.float32:
            model_dev = model_dev.to(dtype)
        input_ids_dev = torch.randint(0, 1000, (2, 32), dtype=torch.int64).to(device)
        with torch.no_grad():
            actual = model_dev(input_ids_dev).logits
        synchronize(device)
    else:
        torch.manual_seed(42)
        model_cpu = Qwen2ForCausalLM(config).eval()
        
        torch.manual_seed(42)
        model_dev = Qwen2ForCausalLM(config).eval().to(device)
        
        model_dev.load_state_dict(model_cpu.state_dict())

        if dtype != torch.float32:
            model_cpu = model_cpu.to(dtype)
            model_dev = model_dev.to(dtype)
                
        input_ids = torch.randint(0, 1000, (2, 32), dtype=torch.int64)
        input_ids_dev = input_ids.to(device)
        
        with torch.no_grad():
            expected = model_cpu(input_ids).logits
            actual = model_dev(input_ids_dev).logits
            
        synchronize(device)
        compare(actual, expected, category="workload_e2e", dtype=dtype)
