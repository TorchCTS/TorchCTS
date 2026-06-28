# Copyright (c) 2026 Kris Bailey. MIT License.

import pytest
import torch

from torchcts.core.device import synchronize

pytestmark = pytest.mark.covers_category("sdpa")


def _compare_tuple(actual, expected, compare, *, category):
    assert len(actual) == len(expected)
    for actual_item, expected_item in zip(actual, expected):
        if isinstance(expected_item, torch.Tensor):
            compare(actual_item, expected_item, category=category, dtype=expected_item.dtype)
        else:
            assert actual_item == expected_item


def _assert_returned_outputs(returned, outputs):
    assert len(returned) == len(outputs)
    for returned_item, output_item in zip(returned, outputs):
        assert returned_item is output_item


@pytest.mark.workload
@pytest.mark.covers("aten::scaled_dot_product_attention")
@pytest.mark.covers("aten::_scaled_dot_product_attention_math")
def test_scaled_dot_product_attention_dispatcher_variants(device, compare):
    q_cpu = torch.linspace(-0.75, 0.75, steps=64, dtype=torch.float32).reshape(1, 2, 4, 8)
    k_cpu = torch.linspace(0.5, -0.5, steps=64, dtype=torch.float32).reshape(1, 2, 4, 8)
    v_cpu = torch.linspace(-1.0, 1.0, steps=64, dtype=torch.float32).reshape(1, 2, 4, 8)
    q_dev = q_cpu.to(device)
    k_dev = k_cpu.to(device)
    v_dev = v_cpu.to(device)

    expected = torch.ops.aten.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu, None, 0.0, False)
    actual = torch.ops.aten.scaled_dot_product_attention(q_dev, k_dev, v_dev, None, 0.0, False)
    synchronize(device)
    compare(actual, expected, category="sdpa", dtype=torch.float32)

    expected_math = torch.ops.aten._scaled_dot_product_attention_math(q_cpu, k_cpu, v_cpu, None, 0.0, False, None)
    actual_math = torch.ops.aten._scaled_dot_product_attention_math(q_dev, k_dev, v_dev, None, 0.0, False, None)
    synchronize(device)
    _compare_tuple(actual_math, expected_math, compare, category="sdpa")


@pytest.mark.workload
@pytest.mark.covers("aten::_native_multi_head_attention")
@pytest.mark.covers("aten::_native_multi_head_attention.out")
def test_native_multi_head_attention_dispatcher_variants(device, compare):
    batch, seq, embed_dim, num_heads = 2, 3, 8, 2
    query_cpu = torch.linspace(-0.5, 0.5, steps=batch * seq * embed_dim, dtype=torch.float32).reshape(batch, seq, embed_dim)
    key_cpu = torch.linspace(0.25, -0.25, steps=batch * seq * embed_dim, dtype=torch.float32).reshape(batch, seq, embed_dim)
    value_cpu = torch.linspace(-1.0, 1.0, steps=batch * seq * embed_dim, dtype=torch.float32).reshape(batch, seq, embed_dim)
    qkv_weight_cpu = torch.linspace(-0.2, 0.2, steps=3 * embed_dim * embed_dim, dtype=torch.float32).reshape(3 * embed_dim, embed_dim)
    qkv_bias_cpu = torch.linspace(-0.1, 0.1, steps=3 * embed_dim, dtype=torch.float32)
    proj_weight_cpu = torch.eye(embed_dim, dtype=torch.float32)
    proj_bias_cpu = torch.linspace(0.05, -0.05, steps=embed_dim, dtype=torch.float32)

    args_cpu = (
        query_cpu,
        key_cpu,
        value_cpu,
        embed_dim,
        num_heads,
        qkv_weight_cpu,
        qkv_bias_cpu,
        proj_weight_cpu,
        proj_bias_cpu,
        None,
        True,
        True,
        None,
    )
    args_dev = tuple(arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args_cpu)

    expected = torch.ops.aten._native_multi_head_attention(*args_cpu)
    actual = torch.ops.aten._native_multi_head_attention(*args_dev)
    synchronize(device)
    _compare_tuple(actual, expected, compare, category="sdpa")

    out_cpu = [torch.empty_like(expected[0]), torch.empty_like(expected[1])]
    out_dev = [torch.empty_like(actual[0]), torch.empty_like(actual[1])]
    expected_return = torch.ops.aten._native_multi_head_attention.out(*args_cpu, out0=out_cpu[0], out1=out_cpu[1])
    actual_return = torch.ops.aten._native_multi_head_attention.out(*args_dev, out0=out_dev[0], out1=out_dev[1])
    synchronize(device)
    _assert_returned_outputs(expected_return, out_cpu)
    _assert_returned_outputs(actual_return, out_dev)
    _compare_tuple(out_dev, out_cpu, compare, category="sdpa")


@pytest.mark.workload
@pytest.mark.covers("aten::_transformer_encoder_layer_fwd")
@pytest.mark.covers("aten::_transformer_encoder_layer_fwd.out")
def test_transformer_encoder_layer_fwd_dispatcher_variants(device, compare):
    batch, seq, embed_dim, num_heads, hidden_dim = 2, 3, 8, 2, 16
    src_cpu = torch.linspace(-1.0, 1.0, steps=batch * seq * embed_dim, dtype=torch.float32).reshape(batch, seq, embed_dim)
    qkv_weight_cpu = torch.linspace(-0.2, 0.2, steps=3 * embed_dim * embed_dim, dtype=torch.float32).reshape(3 * embed_dim, embed_dim)
    qkv_bias_cpu = torch.linspace(-0.1, 0.1, steps=3 * embed_dim, dtype=torch.float32)
    proj_weight_cpu = torch.eye(embed_dim, dtype=torch.float32)
    proj_bias_cpu = torch.linspace(0.05, -0.05, steps=embed_dim, dtype=torch.float32)
    norm_weight_1_cpu = torch.ones(embed_dim, dtype=torch.float32)
    norm_bias_1_cpu = torch.zeros(embed_dim, dtype=torch.float32)
    norm_weight_2_cpu = torch.ones(embed_dim, dtype=torch.float32)
    norm_bias_2_cpu = torch.zeros(embed_dim, dtype=torch.float32)
    ffn_weight_1_cpu = torch.linspace(-0.15, 0.15, steps=hidden_dim * embed_dim, dtype=torch.float32).reshape(hidden_dim, embed_dim)
    ffn_bias_1_cpu = torch.linspace(-0.05, 0.05, steps=hidden_dim, dtype=torch.float32)
    ffn_weight_2_cpu = torch.linspace(0.12, -0.12, steps=embed_dim * hidden_dim, dtype=torch.float32).reshape(embed_dim, hidden_dim)
    ffn_bias_2_cpu = torch.linspace(0.03, -0.03, steps=embed_dim, dtype=torch.float32)

    args_cpu = (
        src_cpu,
        embed_dim,
        num_heads,
        qkv_weight_cpu,
        qkv_bias_cpu,
        proj_weight_cpu,
        proj_bias_cpu,
        False,
        False,
        1e-5,
        norm_weight_1_cpu,
        norm_bias_1_cpu,
        norm_weight_2_cpu,
        norm_bias_2_cpu,
        ffn_weight_1_cpu,
        ffn_bias_1_cpu,
        ffn_weight_2_cpu,
        ffn_bias_2_cpu,
        None,
        None,
    )
    args_dev = tuple(arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args_cpu)

    expected = torch.ops.aten._transformer_encoder_layer_fwd(*args_cpu)
    actual = torch.ops.aten._transformer_encoder_layer_fwd(*args_dev)
    synchronize(device)
    compare(actual, expected, category="workload_e2e", dtype=torch.float32)

    out_cpu = torch.empty_like(expected)
    out_dev = torch.empty_like(actual)
    expected_return = torch.ops.aten._transformer_encoder_layer_fwd.out(*args_cpu, out=out_cpu)
    actual_return = torch.ops.aten._transformer_encoder_layer_fwd.out(*args_dev, out=out_dev)
    synchronize(device)
    assert expected_return is out_cpu
    assert actual_return is out_dev
    compare(out_dev, out_cpu, category="workload_e2e", dtype=torch.float32)
