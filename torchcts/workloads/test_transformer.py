import pytest
import torch
from torchcts.core.device import synchronize

# Skip if transformers is not installed
pytest.importorskip("transformers")

from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from torchcts.workloads.model_configs import get_gpt2_config

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.workload
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("block_type", ["gpt2"])
def test_transformer_block(block_type, dtype, device, manifest, compare):
        
    config = get_gpt2_config()
    if config is None:
        pytest.skip("Transformers config not available")
        
    if device == "cpu":
        block_dev = GPT2Block(config).train().to(device)
        if dtype != torch.float32:
            block_dev = block_dev.to(dtype)
        x_dev = torch.randn(2, 16, 64, dtype=dtype, requires_grad=True).to(device)
        out_dev = block_dev(x_dev)[0]
        out_dev.sum().backward()
        synchronize(device)
    else:
        # Create block on CPU
        block_cpu = GPT2Block(config).train()
        block_dev = GPT2Block(config).train().to(device)
        
        # Keep parameters and buffers aligned across devices.
        block_dev.load_state_dict(block_cpu.state_dict())

        if dtype != torch.float32:
            block_cpu = block_cpu.to(dtype)
            block_dev = block_dev.to(dtype)
                
        # Inputs
        x_cpu = torch.randn(2, 16, 64, dtype=dtype, requires_grad=True)
        x_dev = x_cpu.clone().detach().to(device)
        x_dev.requires_grad = True
        
        # Forward
        out_cpu = block_cpu(x_cpu)[0]
        out_dev = block_dev(x_dev)[0]
        
        # Backward
        out_cpu.sum().backward()
        out_dev.sum().backward()
        
        synchronize(device)
        
        # Compare forward and backward gradients
        compare(out_dev, out_cpu, category="workload_e2e", dtype=dtype)
        compare(x_dev.grad, x_cpu.grad, category="workload_e2e", dtype=dtype)


@pytest.mark.workload
def test_mixed_precision_transformer_block(device, compare):
    """Self-contained mixed-precision transformer: int64 indices -> f32 embed -> bf16 attn -> f32 output."""
    B, S, D = 2, 16, 64
    H, HEAD_D = 2, 32

    # Shared weights — create once, copy to device
    torch.manual_seed(42)
    emb_weight = torch.randn(100, D) * 0.02
    ln_weight = torch.ones(D)
    ln_bias = torch.zeros(D)
    w_up = torch.randn(D * 4, D, dtype=torch.bfloat16) * 0.02
    w_down = torch.randn(D, D * 4, dtype=torch.bfloat16) * 0.02

    # CPU modules
    emb_cpu = torch.nn.Embedding(100, D)
    emb_cpu.weight.data.copy_(emb_weight)
    ln_cpu = torch.nn.LayerNorm(D)
    ln_cpu.weight.data.copy_(ln_weight)
    ln_cpu.bias.data.copy_(ln_bias)

    # Device modules — same weights
    emb_dev = torch.nn.Embedding(100, D).to(device)
    emb_dev.weight.data.copy_(emb_weight)
    ln_dev = torch.nn.LayerNorm(D).to(device)
    ln_dev.weight.data.copy_(ln_weight)
    ln_dev.bias.data.copy_(ln_bias)

    w_up_dev = w_up.to(device)
    w_down_dev = w_down.to(device)

    indices = torch.randint(0, 100, (B, S), dtype=torch.int64)
    indices_dev = indices.to(device)

    def forward(emb, ln, w_u, w_d, idx):
        x = emb(idx)
        res = x.clone()
        x = ln(x)
        x_bf16 = x.to(torch.bfloat16)
        Q = K = V = x_bf16.view(B, S, H, HEAD_D).transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, S, D)
        x = res + attn
        h = torch.nn.functional.linear(x.to(torch.bfloat16), w_u)
        h = torch.nn.functional.silu(h)
        h = torch.nn.functional.linear(h, w_d)
        return h.to(torch.float32)

    logits_cpu = forward(emb_cpu, ln_cpu, w_up, w_down, indices)
    logits_dev = forward(emb_dev, ln_dev, w_up_dev, w_down_dev, indices_dev)
    synchronize(device)
    compare(logits_dev, logits_cpu, category="workload_e2e", dtype=torch.bfloat16)


@pytest.mark.workload
def test_mixed_precision_backward(device, compare):
    """Mixed-precision forward+backward: verify gradients match between CPU and device."""
    B, S, D = 1, 8, 32

    # Shared initial values
    torch.manual_seed(42)
    x_init = torch.randn(B, S, D)
    ln_sd = torch.nn.LayerNorm(D).state_dict()
    w_init = torch.randn(D, D, dtype=torch.bfloat16)

    # CPU path
    x_cpu = x_init.clone().requires_grad_(True)
    ln_cpu = torch.nn.LayerNorm(D)
    ln_cpu.load_state_dict(ln_sd)
    w_cpu = w_init.clone().requires_grad_(True)

    h_cpu = ln_cpu(x_cpu)
    h_cpu = torch.nn.functional.linear(h_cpu.to(torch.bfloat16), w_cpu).to(torch.float32)
    h_cpu.sum().backward()

    # Device path
    x_dev = x_init.clone().to(device).requires_grad_(True)
    ln_dev = torch.nn.LayerNorm(D).to(device)
    ln_dev.load_state_dict(ln_sd)
    w_dev = w_init.clone().to(device).requires_grad_(True)

    h_dev = ln_dev(x_dev)
    h_dev = torch.nn.functional.linear(h_dev.to(torch.bfloat16), w_dev).to(torch.float32)
    h_dev.sum().backward()

    synchronize(device)
    compare(h_dev, h_cpu, category="workload_e2e", dtype=torch.float32)
    compare(x_dev.grad, x_cpu.grad, category="workload_e2e", dtype=torch.float32)
    compare(w_dev.grad, w_cpu.grad, category="workload_e2e", dtype=torch.bfloat16)

