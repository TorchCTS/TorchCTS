import pytest
import torch
from torchcts.core.device import synchronize

@pytest.mark.workload
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("is_causal", [True, False])
def test_sdpa_causal(dtype, is_causal, device, manifest, compare, input_gen):
    B, H, S, D = 2, 4, 32, 16
    
    q_dev = input_gen((B, H, S, D), dtype, device)
    k_dev = input_gen((B, H, S, D), dtype, device)
    v_dev = input_gen((B, H, S, D), dtype, device)
    
    if is_causal:
        expected = torch.nn.functional.scaled_dot_product_attention(
            q_dev.cpu(), k_dev.cpu(), v_dev.cpu(), is_causal=True
        )
        actual = torch.nn.functional.scaled_dot_product_attention(
            q_dev, k_dev, v_dev, is_causal=True
        )
        synchronize(device)
        compare(actual, expected, category="sdpa", dtype=dtype)
    else:
        # Non-causal with attention mask
        mask_cpu = torch.ones(B, 1, S, S, dtype=torch.bool)
        # create triangular causal mask manually
        mask_cpu = torch.tril(mask_cpu)
        mask_dev = mask_cpu.to(device)
        
        expected_masked = torch.nn.functional.scaled_dot_product_attention(
            q_dev.cpu(), k_dev.cpu(), v_dev.cpu(), attn_mask=mask_cpu
        )
        actual_masked = torch.nn.functional.scaled_dot_product_attention(
            q_dev, k_dev, v_dev, attn_mask=mask_dev
        )
        synchronize(device)
        compare(actual_masked, expected_masked, category="sdpa", dtype=dtype)


@pytest.mark.workload
@pytest.mark.parametrize("Sq,Sk", [
    (1, 1), (1, 7), (1, 32),   # single-query cross-attention
    (7, 7), (7, 32),            # short-query
    (32, 32),                   # matched
])
def test_sdpa_asymmetric_lengths(Sq, Sk, device, compare, input_gen):
    """SDPA with Q and K/V having different sequence lengths (non-causal)."""
    B, H, D = 1, 2, 16
    q = input_gen((B, H, Sq, D), torch.float32, device)
    k = input_gen((B, H, Sk, D), torch.float32, device)
    v = input_gen((B, H, Sk, D), torch.float32, device)

    # Non-causal — always valid regardless of Sq vs Sk
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    ref = torch.nn.functional.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu(), is_causal=False)
    synchronize(device)
    compare(out, ref, category="sdpa", dtype=torch.float32)


@pytest.mark.workload
@pytest.mark.parametrize("S", [3, 7, 15, 33, 65])
def test_sdpa_non_power_of_2(S, device, compare, input_gen):
    """Causal SDPA with non-power-of-2 sequence lengths."""
    B, H, D = 1, 2, 16
    q = input_gen((B, H, S, D), torch.float32, device)
    k = input_gen((B, H, S, D), torch.float32, device)
    v = input_gen((B, H, S, D), torch.float32, device)

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = torch.nn.functional.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu(), is_causal=True)
    synchronize(device)
    compare(out, ref, category="sdpa", dtype=torch.float32)

