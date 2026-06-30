# Copyright (c) 2026 Kris Bailey. MIT License.

import pytest
import torch
from torchcts.core.device import synchronize


@pytest.mark.requires("native_quantization")
@pytest.mark.medium
@pytest.mark.covers("aten::quantize_per_tensor")
@pytest.mark.covers("aten::dequantize.self")
@pytest.mark.covers("aten::q_scale")
@pytest.mark.covers("aten::q_zero_point")
def test_quantize_per_tensor_qint8_roundtrip(device, compare):
    torch.manual_seed(42)
    x = torch.randn(128, 768, device=device)
    x_cpu = x.cpu()

    qt = torch.quantize_per_tensor(x, scale=0.05, zero_point=0, dtype=torch.qint8)
    qt_cpu = torch.quantize_per_tensor(x_cpu, scale=0.05, zero_point=0, dtype=torch.qint8)

    assert qt.is_quantized
    assert abs(qt.q_scale() - 0.05) < 1e-6
    assert qt.q_zero_point() == 0

    deq = qt.dequantize()
    deq_cpu = qt_cpu.dequantize()
    synchronize(device)
    compare(deq, deq_cpu, category="native_quantization", dtype=torch.float32)


@pytest.mark.requires("native_quantization")
@pytest.mark.medium
@pytest.mark.covers("aten::quantize_per_tensor")
@pytest.mark.covers("aten::dequantize.self")
@pytest.mark.covers("aten::q_scale")
@pytest.mark.covers("aten::q_zero_point")
def test_quantize_per_tensor_quint8_roundtrip(device, compare):
    torch.manual_seed(42)
    x = torch.randn(128, 768, device=device)
    x_cpu = x.cpu()

    qt = torch.quantize_per_tensor(x, scale=0.05, zero_point=128, dtype=torch.quint8)
    qt_cpu = torch.quantize_per_tensor(x_cpu, scale=0.05, zero_point=128, dtype=torch.quint8)

    assert qt.is_quantized
    assert abs(qt.q_scale() - 0.05) < 1e-6
    assert qt.q_zero_point() == 128

    deq = qt.dequantize()
    deq_cpu = qt_cpu.dequantize()
    synchronize(device)
    compare(deq, deq_cpu, category="native_quantization", dtype=torch.float32)


@pytest.mark.requires("native_quantization")
@pytest.mark.medium
@pytest.mark.covers("aten::quantize_per_tensor.tensor_qparams")
@pytest.mark.covers("aten::dequantize.self")
def test_quantize_per_tensor_tensor_qparams(device, compare):
    torch.manual_seed(42)
    x = torch.randn(128, 768, device=device)
    x_cpu = x.cpu()

    scale = torch.tensor(0.05)
    zero_point = torch.tensor(0)

    qt = torch.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype=torch.qint8)
    qt_cpu = torch.quantize_per_tensor(x_cpu, scale=scale, zero_point=zero_point, dtype=torch.qint8)

    deq = qt.dequantize()
    deq_cpu = qt_cpu.dequantize()
    synchronize(device)
    compare(deq, deq_cpu, category="native_quantization", dtype=torch.float32)


@pytest.mark.requires("native_quantization")
@pytest.mark.medium
@pytest.mark.covers("aten::quantize_per_tensor_dynamic")
@pytest.mark.covers("aten::dequantize.self")
def test_quantize_per_tensor_dynamic(device, compare):
    torch.manual_seed(42)
    x = torch.randn(128, 768, device=device)
    x_cpu = x.cpu()

    qt = torch.quantize_per_tensor_dynamic(x, dtype=torch.qint8, reduce_range=False)
    qt_cpu = torch.quantize_per_tensor_dynamic(x_cpu, dtype=torch.qint8, reduce_range=False)

    deq = qt.dequantize()
    deq_cpu = qt_cpu.dequantize()
    synchronize(device)
    compare(deq, deq_cpu, category="native_quantization", dtype=torch.float32)


@pytest.mark.requires("native_quantization")
@pytest.mark.medium
@pytest.mark.covers("aten::quantize_per_channel")
@pytest.mark.covers("aten::dequantize.self")
@pytest.mark.covers("aten::qscheme")
@pytest.mark.covers("aten::q_per_channel_axis")
@pytest.mark.covers("aten::q_per_channel_scales")
@pytest.mark.covers("aten::q_per_channel_zero_points")
def test_quantize_per_channel(device, compare):
    torch.manual_seed(42)
    x = torch.randn(32, 64, device=device)
    x_cpu = x.cpu()

    scales = torch.rand(32, dtype=torch.float64) * 0.1 + 0.01
    zero_points = torch.zeros(32, dtype=torch.int64)

    qt = torch.quantize_per_channel(x, scales, zero_points, axis=0, dtype=torch.qint8)
    qt_cpu = torch.quantize_per_channel(x_cpu, scales, zero_points, axis=0, dtype=torch.qint8)

    assert qt.qscheme() == torch.per_channel_affine
    assert qt.q_per_channel_axis() == 0
    assert torch.equal(qt.q_per_channel_scales().cpu(), scales)
    assert torch.equal(qt.q_per_channel_zero_points().cpu(), zero_points)

    deq = qt.dequantize()
    deq_cpu = qt_cpu.dequantize()
    synchronize(device)
    compare(deq, deq_cpu, category="native_quantization", dtype=torch.float32)


@pytest.mark.requires("native_quantization")
@pytest.mark.medium
@pytest.mark.covers("aten::_choose_qparams_per_tensor")
def test_choose_qparams_per_tensor(device):
    torch.manual_seed(42)
    x = torch.randn(128, 768, device=device)

    scale, zp = torch.ops.aten._choose_qparams_per_tensor(x, reduce_range=False)
    assert scale > 0

    scale_rr, zp_rr = torch.ops.aten._choose_qparams_per_tensor(x, reduce_range=True)
    assert scale_rr > 0
    assert abs(scale - scale_rr) > 0


@pytest.mark.requires("native_quantization")
@pytest.mark.medium
@pytest.mark.covers("aten::_make_per_tensor_quantized_tensor")
@pytest.mark.covers("aten::int_repr")
@pytest.mark.covers("aten::q_scale")
@pytest.mark.covers("aten::q_zero_point")
def test_make_per_tensor_quantized_tensor(device):
    torch.manual_seed(42)
    int_data = torch.randint(-128, 127, (128, 768), dtype=torch.int8, device=device)

    qt = torch._make_per_tensor_quantized_tensor(int_data, 0.05, 10)

    assert qt.is_quantized
    assert abs(qt.q_scale() - 0.05) < 1e-6
    assert qt.q_zero_point() == 10
    assert qt.int_repr().shape == (128, 768)


@pytest.mark.requires("native_quantization")
@pytest.mark.medium
@pytest.mark.covers("aten::int_repr.out")
def test_quantized_int_repr_out(device):
    int_data = torch.randint(0, 255, (8, 16), dtype=torch.uint8, device=device)
    qt = torch._make_per_tensor_quantized_tensor(int_data, 0.05, 10)

    expected = qt.cpu().int_repr()
    out = torch.empty(0, dtype=expected.dtype, device=device)
    returned = torch.ops.aten.int_repr.out(qt, out=out)
    synchronize(device)

    assert returned is out
    assert out.dtype == expected.dtype
    assert out.shape == expected.shape
    assert torch.equal(out.cpu(), expected)


@pytest.mark.requires("native_quantization")
@pytest.mark.medium
@pytest.mark.covers("aten::_make_per_channel_quantized_tensor")
@pytest.mark.covers("aten::qscheme")
def test_make_per_channel_quantized_tensor(device):
    torch.manual_seed(42)
    int_data = torch.randint(-128, 127, (32, 64), dtype=torch.int8, device=device)
    scales = torch.rand(32, dtype=torch.float64) * 0.1 + 0.01
    zeros = torch.zeros(32, dtype=torch.int64)

    qt = torch._make_per_channel_quantized_tensor(int_data, scales, zeros, 0)

    assert qt.is_quantized
    assert qt.qscheme() == torch.per_channel_affine


@pytest.mark.requires("native_quantization")
@pytest.mark.medium
@pytest.mark.covers("aten::q_per_channel_scales.out")
@pytest.mark.covers("aten::q_per_channel_zero_points.out")
@pytest.mark.cpu_contract_dtype("aten::q_per_channel_scales.out", torch.float64)
@pytest.mark.cpu_contract_dtype("aten::q_per_channel_zero_points.out", torch.int64)
def test_quantized_per_channel_qparams_out(device):
    int_data = torch.randint(-128, 127, (4, 6), dtype=torch.int8, device=device)
    scales = torch.tensor([0.05, 0.10, 0.15, 0.20], dtype=torch.float64)
    zero_points = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    qt = torch._make_per_channel_quantized_tensor(int_data, scales, zero_points, 0)

    scales_out = torch.empty(0, dtype=torch.float64, device=device)
    zeros_out = torch.empty(0, dtype=torch.int64, device=device)

    returned_scales = torch.ops.aten.q_per_channel_scales.out(qt, out=scales_out)
    returned_zeros = torch.ops.aten.q_per_channel_zero_points.out(qt, out=zeros_out)
    synchronize(device)

    assert returned_scales is scales_out
    assert returned_zeros is zeros_out
    assert torch.equal(scales_out.cpu(), scales)
    assert torch.equal(zeros_out.cpu(), zero_points)


@pytest.mark.requires("native_quantization")
@pytest.mark.medium
@pytest.mark.covers("aten::quantize_per_tensor.out")
@pytest.mark.covers("aten::quantize_per_tensor.tensor_qparams_out")
@pytest.mark.covers("aten::quantize_per_tensor_dynamic.out")
@pytest.mark.covers("aten::quantize_per_channel.out")
@pytest.mark.cpu_contract_dtype("aten::quantize_per_channel.out", torch.float64)
@pytest.mark.cpu_contract_dtype("aten::quantize_per_channel.out", torch.int64)
def test_quantize_out_variants(device, compare):
    x = torch.tensor([[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]], dtype=torch.float32, device=device)
    x_cpu = x.cpu()

    expected_tensor = torch.quantize_per_tensor(x_cpu, 0.05, 10, torch.qint8)
    out_tensor = torch.empty_quantized(x.shape, torch.quantize_per_tensor(x, 0.05, 10, torch.qint8))
    returned_tensor = torch.ops.aten.quantize_per_tensor.out(x, 0.05, 10, torch.qint8, out=out_tensor)
    synchronize(device)
    assert returned_tensor is out_tensor
    assert torch.equal(out_tensor.cpu().int_repr(), expected_tensor.int_repr())
    compare(out_tensor.dequantize(), expected_tensor.dequantize(), category="native_quantization", dtype=torch.float32)

    expected_qparams = torch.quantize_per_tensor(
        x_cpu,
        torch.tensor(0.05),
        torch.tensor(10),
        torch.qint8,
    )
    out_qparams = torch.empty_quantized(x.shape, torch.quantize_per_tensor(x, 0.05, 10, torch.qint8))
    returned_qparams = torch.ops.aten.quantize_per_tensor.tensor_qparams_out(
        x,
        torch.tensor(0.05, device=device),
        torch.tensor(10, device=device),
        torch.qint8,
        out=out_qparams,
    )
    synchronize(device)
    assert returned_qparams is out_qparams
    assert torch.equal(out_qparams.cpu().int_repr(), expected_qparams.int_repr())

    expected_dynamic = torch.quantize_per_tensor_dynamic(x_cpu, torch.qint8, False)
    out_dynamic = torch.empty_quantized(x.shape, torch.quantize_per_tensor(x, 0.05, 10, torch.qint8))
    returned_dynamic = torch.ops.aten.quantize_per_tensor_dynamic.out(x, torch.qint8, False, out=out_dynamic)
    synchronize(device)
    assert returned_dynamic is out_dynamic
    compare(out_dynamic.dequantize(), expected_dynamic.dequantize(), category="native_quantization", dtype=torch.float32)

    scales = torch.tensor([0.05, 0.10], dtype=torch.float64)
    zero_points = torch.tensor([0, 1], dtype=torch.int64)
    expected_channel = torch.quantize_per_channel(x_cpu, scales, zero_points, 0, torch.qint8)
    out_channel = torch.empty_quantized(x.shape, torch.quantize_per_channel(x, scales, zero_points, 0, torch.qint8))
    returned_channel = torch.ops.aten.quantize_per_channel.out(
        x,
        scales.to(device),
        zero_points.to(device),
        0,
        torch.qint8,
        out=out_channel,
    )
    synchronize(device)
    assert returned_channel is out_channel
    assert torch.equal(out_channel.cpu().int_repr(), expected_channel.int_repr())
    assert torch.equal(out_channel.q_per_channel_scales().cpu(), scales)
    assert torch.equal(out_channel.q_per_channel_zero_points().cpu(), zero_points)


@pytest.mark.requires("native_quantization")
@pytest.mark.medium
@pytest.mark.covers("aten::quantize_per_tensor.tensors")
@pytest.mark.covers("aten::quantize_per_tensor.tensors_out")
@pytest.mark.covers("aten::dequantize.self_out")
@pytest.mark.covers("aten::dequantize.tensors")
@pytest.mark.covers("aten::dequantize.tensors_out")
@pytest.mark.cpu_contract_dtype("aten::quantize_per_tensor.tensors", torch.float32)
@pytest.mark.cpu_contract_dtype("aten::quantize_per_tensor.tensors", torch.int64)
def test_quantize_tensor_list_and_dequantize_out_variants(device, compare):
    tensors = [
        torch.tensor([[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]], dtype=torch.float32, device=device),
        torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=torch.float32, device=device),
    ]
    tensors_cpu = [tensor.cpu() for tensor in tensors]
    scales = torch.tensor([0.05, 0.10], dtype=torch.float32, device=device)
    zero_points = torch.tensor([10, 20], dtype=torch.int64, device=device)
    scales_cpu = scales.cpu()
    zero_points_cpu = zero_points.cpu()

    quantized = torch.ops.aten.quantize_per_tensor.tensors(tensors, scales, zero_points, torch.qint8)
    expected_quantized = torch.ops.aten.quantize_per_tensor.tensors(tensors_cpu, scales_cpu, zero_points_cpu, torch.qint8)
    assert len(quantized) == len(expected_quantized)
    for actual, expected in zip(quantized, expected_quantized):
        assert torch.equal(actual.cpu().int_repr(), expected.int_repr())

    out_quantized = [
        torch.empty_quantized(tensors[0].shape, quantized[0]),
        torch.empty_quantized(tensors[1].shape, quantized[1]),
    ]
    returned_quantized = torch.ops.aten.quantize_per_tensor.tensors_out(
        tensors,
        scales,
        zero_points,
        torch.qint8,
        out=out_quantized,
    )
    synchronize(device)
    assert returned_quantized is None
    for actual, expected in zip(out_quantized, expected_quantized):
        assert torch.equal(actual.cpu().int_repr(), expected.int_repr())

    dequantized = torch.ops.aten.dequantize.tensors(quantized)
    assert len(dequantized) == len(tensors)
    for actual, expected in zip(dequantized, expected_quantized):
        compare(actual, expected.dequantize(), category="native_quantization", dtype=torch.float32)

    out_self = torch.empty(0, dtype=torch.float32, device=device)
    returned_self = torch.ops.aten.dequantize.self_out(quantized[0], out=out_self)
    synchronize(device)
    assert returned_self is out_self
    compare(out_self, expected_quantized[0].dequantize(), category="native_quantization", dtype=torch.float32)

    out_dequantized = [torch.empty(0, dtype=torch.float32, device=device), torch.empty(0, dtype=torch.float32, device=device)]
    returned_dequantized = torch.ops.aten.dequantize.tensors_out(quantized, out=out_dequantized)
    synchronize(device)
    assert returned_dequantized is None
    for actual, expected in zip(out_dequantized, expected_quantized):
        compare(actual, expected.dequantize(), category="native_quantization", dtype=torch.float32)


@pytest.mark.requires("native_quantization")
@pytest.mark.medium
@pytest.mark.covers("aten::_make_per_tensor_quantized_tensor.out")
@pytest.mark.covers("aten::_make_per_channel_quantized_tensor.out")
@pytest.mark.covers("aten::empty_quantized")
@pytest.mark.covers("aten::empty_quantized.out")
@pytest.mark.cpu_contract_dtype("aten::_make_per_channel_quantized_tensor.out", torch.float64)
@pytest.mark.cpu_contract_dtype("aten::_make_per_channel_quantized_tensor.out", torch.int64)
def test_make_quantized_out_and_empty_quantized(device):
    int_data = torch.arange(-6, 6, dtype=torch.int8, device=device).reshape(2, 6)
    int_cpu = int_data.cpu()

    expected_tensor = torch._make_per_tensor_quantized_tensor(int_cpu, 0.05, 10)
    out_tensor = torch.empty_quantized(int_data.shape, torch._make_per_tensor_quantized_tensor(int_data, 0.05, 10))
    returned_tensor = torch.ops.aten._make_per_tensor_quantized_tensor.out(int_data, 0.05, 10, out=out_tensor)
    synchronize(device)
    assert returned_tensor is out_tensor
    assert torch.equal(out_tensor.cpu().int_repr(), expected_tensor.int_repr())
    assert abs(out_tensor.q_scale() - 0.05) < 1e-6
    assert out_tensor.q_zero_point() == 10

    scales = torch.tensor([0.05, 0.10], dtype=torch.float64)
    zero_points = torch.tensor([0, 1], dtype=torch.int64)
    expected_channel = torch._make_per_channel_quantized_tensor(int_cpu, scales, zero_points, 0)
    out_channel = torch.empty_quantized(
        int_data.shape,
        torch._make_per_channel_quantized_tensor(int_data, scales, zero_points, 0),
    )
    returned_channel = torch.ops.aten._make_per_channel_quantized_tensor.out(
        int_data,
        scales.to(device),
        zero_points.to(device),
        0,
        out=out_channel,
    )
    synchronize(device)
    assert returned_channel is out_channel
    assert torch.equal(out_channel.cpu().int_repr(), expected_channel.int_repr())
    assert torch.equal(out_channel.q_per_channel_scales().cpu(), scales)
    assert torch.equal(out_channel.q_per_channel_zero_points().cpu(), zero_points)

    template = torch._make_per_tensor_quantized_tensor(int_data, 0.05, 10)
    empty = torch.ops.aten.empty_quantized([2, 6], template)
    assert empty.is_quantized
    assert empty.shape == (2, 6)
    assert abs(empty.q_scale() - template.q_scale()) < 1e-6
    assert empty.q_zero_point() == template.q_zero_point()

    out_empty = torch.empty_quantized((0,), template)
    returned_empty = torch.ops.aten.empty_quantized.out([2, 6], template, out=out_empty)
    synchronize(device)
    assert returned_empty is out_empty
    assert out_empty.is_quantized
    assert out_empty.shape == (2, 6)
    assert abs(out_empty.q_scale() - template.q_scale()) < 1e-6
    assert out_empty.q_zero_point() == template.q_zero_point()


@pytest.mark.requires("native_quantization")
@pytest.mark.medium
@pytest.mark.covers("aten::quantized_batch_norm")
@pytest.mark.covers("aten::quantized_batch_norm.out")
def test_quantized_batch_norm_out(device, compare):
    x_cpu = torch.randn(1, 2, 3, 3)
    x = x_cpu.to(device)
    q_cpu = torch.quantize_per_tensor(x_cpu, 0.05, 10, torch.quint8)
    q = torch.quantize_per_tensor(x, 0.05, 10, torch.quint8)
    weight = torch.ones(2, device=device)
    bias = torch.zeros(2, device=device)
    mean = torch.zeros(2, device=device)
    var = torch.ones(2, device=device)
    weight_cpu = weight.cpu()
    bias_cpu = bias.cpu()
    mean_cpu = mean.cpu()
    var_cpu = var.cpu()

    expected = torch.ops.aten.quantized_batch_norm(q_cpu, weight_cpu, bias_cpu, mean_cpu, var_cpu, 1e-5, 0.05, 10)
    actual = torch.ops.aten.quantized_batch_norm(q, weight, bias, mean, var, 1e-5, 0.05, 10)
    synchronize(device)
    assert torch.equal(actual.cpu().int_repr(), expected.int_repr())
    compare(actual.dequantize(), expected.dequantize(), category="native_quantization", dtype=torch.float32)

    out = torch.empty_quantized((0,), q)
    returned = torch.ops.aten.quantized_batch_norm.out(q, weight, bias, mean, var, 1e-5, 0.05, 10, out=out)
    synchronize(device)
    assert returned is out
    assert torch.equal(out.cpu().int_repr(), expected.int_repr())


@pytest.mark.requires("native_quantization")
@pytest.mark.medium
@pytest.mark.covers("aten::quantized_max_pool1d")
@pytest.mark.covers("aten::quantized_max_pool1d.out")
@pytest.mark.covers("aten::quantized_max_pool2d")
@pytest.mark.covers("aten::quantized_max_pool2d.out")
@pytest.mark.covers("aten::quantized_max_pool3d")
@pytest.mark.covers("aten::quantized_max_pool3d.out")
def test_quantized_max_pool_variants(device):
    x1_cpu = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]])
    x2_cpu = torch.randn(1, 2, 4, 4)
    x3_cpu = torch.randn(1, 2, 3, 4, 4)
    cases = [
        (
            torch.quantize_per_tensor(x1_cpu.to(device), 0.1, 10, torch.quint8),
            torch.quantize_per_tensor(x1_cpu, 0.1, 10, torch.quint8),
            torch.ops.aten.quantized_max_pool1d,
            ([2], [1], [0], [1], False),
        ),
        (
            torch.quantize_per_tensor(x2_cpu.to(device), 0.05, 10, torch.quint8),
            torch.quantize_per_tensor(x2_cpu, 0.05, 10, torch.quint8),
            torch.ops.aten.quantized_max_pool2d,
            ([2, 2], [1, 1], [0, 0], [1, 1], False),
        ),
        (
            torch.quantize_per_tensor(x3_cpu.to(device), 0.05, 10, torch.quint8),
            torch.quantize_per_tensor(x3_cpu, 0.05, 10, torch.quint8),
            torch.ops.aten.quantized_max_pool3d,
            ([2, 2, 2], [1, 1, 1], [0, 0, 0], [1, 1, 1], False),
        ),
    ]

    for q, q_cpu, op, args in cases:
        expected = op(q_cpu, *args)
        actual = op(q, *args)
        synchronize(device)
        assert torch.equal(actual.cpu().int_repr(), expected.int_repr())

        out = torch.empty_quantized((0,), q)
        returned = op.out(q, *args, out=out)
        synchronize(device)
        assert returned is out
        assert torch.equal(out.cpu().int_repr(), expected.int_repr())


@pytest.mark.requires("native_quantization")
@pytest.mark.medium
@pytest.mark.covers("aten::fake_quantize_per_tensor_affine")
@pytest.mark.covers("aten::fake_quantize_per_tensor_affine_cachemask_backward")
def test_fake_quantize_per_tensor_affine(device, compare):
    torch.manual_seed(42)
    x = torch.randn(32, 64, device=device, requires_grad=True)
    x_cpu = x.detach().cpu().requires_grad_(True)

    out_dev = torch.fake_quantize_per_tensor_affine(
        x, scale=0.1, zero_point=0, quant_min=-128, quant_max=127
    )
    out_cpu = torch.fake_quantize_per_tensor_affine(
        x_cpu, scale=0.1, zero_point=0, quant_min=-128, quant_max=127
    )
    synchronize(device)
    compare(out_dev, out_cpu, category="native_quantization", dtype=torch.float32)

    out_dev.sum().backward()
    out_cpu.sum().backward()
    synchronize(device)

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    compare(x.grad, x_cpu.grad, category="native_quantization", dtype=torch.float32)


@pytest.mark.requires("native_quantization")
@pytest.mark.medium
@pytest.mark.covers("aten::fake_quantize_per_channel_affine")
@pytest.mark.covers("aten::fake_quantize_per_channel_affine_cachemask_backward")
def test_fake_quantize_per_channel_affine(device, compare):
    torch.manual_seed(42)
    x = torch.randn(32, 64, device=device, requires_grad=True)
    x_cpu = x.detach().cpu().requires_grad_(True)

    scale = torch.rand(32, device=device) * 0.1 + 0.01
    scale_cpu = scale.cpu()
    zero_point = torch.zeros(32, dtype=torch.int32, device=device)
    zp_cpu = zero_point.cpu()

    out_dev = torch.fake_quantize_per_channel_affine(
        x, scale, zero_point, axis=0, quant_min=-128, quant_max=127
    )
    out_cpu = torch.fake_quantize_per_channel_affine(
        x_cpu, scale_cpu, zp_cpu, axis=0, quant_min=-128, quant_max=127
    )
    synchronize(device)
    compare(out_dev, out_cpu, category="native_quantization", dtype=torch.float32)

    out_dev.sum().backward()
    out_cpu.sum().backward()
    synchronize(device)

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    compare(x.grad, x_cpu.grad, category="native_quantization", dtype=torch.float32)


@pytest.mark.requires("native_quantization")
@pytest.mark.medium
@pytest.mark.covers("aten::_empty_affine_quantized.out")
@pytest.mark.covers("aten::_empty_per_channel_affine_quantized")
@pytest.mark.covers("aten::_empty_per_channel_affine_quantized.out")
@pytest.mark.cpu_contract_dtype("aten::_empty_per_channel_affine_quantized", torch.float64)
@pytest.mark.cpu_contract_dtype("aten::_empty_per_channel_affine_quantized", torch.int64)
def test_empty_affine_quantized_dispatcher_variants(device):
    template_cpu = torch.quantize_per_tensor(torch.zeros(2, 3), 1.0, 0, torch.quint8)
    template = torch.quantize_per_tensor(torch.zeros(2, 3, device=device), 1.0, 0, torch.quint8)

    out_tensor = torch.empty_quantized((2, 3), template)
    returned_tensor = torch.ops.aten._empty_affine_quantized.out(
        [2, 3],
        scale=0.1,
        zero_point=2,
        out=out_tensor,
    )
    synchronize(device)
    assert returned_tensor is out_tensor
    assert out_tensor.shape == (2, 3)
    assert out_tensor.dtype == torch.quint8
    assert out_tensor.qscheme() == torch.per_tensor_affine
    assert abs(out_tensor.q_scale() - 0.1) < 1e-6
    assert out_tensor.q_zero_point() == 2

    scales_cpu = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    zero_points_cpu = torch.tensor([0, 1, 2], dtype=torch.int64)
    scales = scales_cpu.to(device)
    zero_points = zero_points_cpu.to(device)

    per_channel = torch.ops.aten._empty_per_channel_affine_quantized(
        [2, 3],
        scales=scales,
        zero_points=zero_points,
        axis=1,
        dtype=torch.quint8,
        layout=torch.strided,
        device=torch.device(device),
        pin_memory=False,
    )
    synchronize(device)
    assert per_channel.shape == (2, 3)
    assert per_channel.dtype == torch.quint8
    assert per_channel.qscheme() == torch.per_channel_affine
    assert per_channel.q_per_channel_axis() == 1
    assert torch.equal(per_channel.q_per_channel_scales().cpu(), scales_cpu)
    assert torch.equal(per_channel.q_per_channel_zero_points().cpu(), zero_points_cpu)

    per_channel_template = torch.quantize_per_channel(
        torch.zeros(2, 3, device=device),
        scales,
        zero_points,
        1,
        torch.quint8,
    )
    out_channel = torch.empty_quantized((2, 3), per_channel_template)
    returned_channel = torch.ops.aten._empty_per_channel_affine_quantized.out(
        [2, 3],
        scales=scales,
        zero_points=zero_points,
        axis=1,
        out=out_channel,
    )
    synchronize(device)
    assert returned_channel is out_channel
    assert out_channel.shape == (2, 3)
    assert out_channel.dtype == torch.quint8
    assert out_channel.qscheme() == torch.per_channel_affine
    assert out_channel.q_per_channel_axis() == 1
    assert torch.equal(out_channel.q_per_channel_scales().cpu(), scales_cpu)
    assert torch.equal(out_channel.q_per_channel_zero_points().cpu(), zero_points_cpu)

    assert template_cpu.q_scale() == 1.0


def _run_fused_moving_avg_obs_fake_quant(target_device, *, variant):
    x = torch.tensor([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]], dtype=torch.float32, device=target_device)
    observer_on = torch.tensor([1], dtype=torch.long, device=target_device)
    fake_quant_on = torch.tensor([1], dtype=torch.long, device=target_device)
    running_min = torch.tensor([0.0], dtype=torch.float32, device=target_device)
    running_max = torch.tensor([0.0], dtype=torch.float32, device=target_device)
    scale = torch.tensor([1.0], dtype=torch.float32, device=target_device)
    zero_point = torch.tensor([0], dtype=torch.int32, device=target_device)
    args = (
        x,
        observer_on,
        fake_quant_on,
        running_min,
        running_max,
        scale,
        zero_point,
        0.1,
        0,
        255,
        -1,
        False,
        False,
    )
    if variant == "mutating":
        output, mask = torch.ops.aten._fused_moving_avg_obs_fq_helper(*args)
        return output, mask, running_min, running_max, scale, zero_point
    if variant == "out":
        out0 = torch.empty_like(x)
        out1 = torch.empty_like(x, dtype=torch.bool)
        returned = torch.ops.aten._fused_moving_avg_obs_fq_helper.out(*args, out0=out0, out1=out1)
        return returned, out0, out1, running_min, running_max, scale, zero_point
    if variant == "functional":
        return torch.ops.aten._fused_moving_avg_obs_fq_helper_functional(*args)
    if variant == "public":
        output = torch.ops.aten.fused_moving_avg_obs_fake_quant(*args)
        return output, running_min, running_max, scale, zero_point
    raise AssertionError(f"unknown fused moving average fake quant variant: {variant}")


@pytest.mark.requires("native_quantization")
@pytest.mark.medium
@pytest.mark.covers("aten::_fused_moving_avg_obs_fq_helper")
@pytest.mark.covers("aten::_fused_moving_avg_obs_fq_helper.out")
@pytest.mark.covers("aten::_fused_moving_avg_obs_fq_helper_functional")
@pytest.mark.covers("aten::fused_moving_avg_obs_fake_quant")
def test_fused_moving_avg_obs_fake_quant_dispatcher_variants(device, compare):
    expected = _run_fused_moving_avg_obs_fake_quant("cpu", variant="mutating")
    actual = _run_fused_moving_avg_obs_fake_quant(device, variant="mutating")
    synchronize(device)
    compare(actual[0], expected[0], category="native_quantization", dtype=torch.float32)
    compare(actual[1], expected[1], category="exact", dtype=torch.bool)
    for actual_item, expected_item in zip(actual[2:], expected[2:]):
        compare(actual_item, expected_item, category="native_quantization", dtype=expected_item.dtype)

    expected = _run_fused_moving_avg_obs_fake_quant("cpu", variant="out")
    actual = _run_fused_moving_avg_obs_fake_quant(device, variant="out")
    synchronize(device)
    assert expected[0][0] is expected[1]
    assert expected[0][1] is expected[2]
    assert actual[0][0] is actual[1]
    assert actual[0][1] is actual[2]
    compare(actual[1], expected[1], category="native_quantization", dtype=torch.float32)
    compare(actual[2], expected[2], category="exact", dtype=torch.bool)
    for actual_item, expected_item in zip(actual[3:], expected[3:]):
        compare(actual_item, expected_item, category="native_quantization", dtype=expected_item.dtype)

    expected = _run_fused_moving_avg_obs_fake_quant("cpu", variant="functional")
    actual = _run_fused_moving_avg_obs_fake_quant(device, variant="functional")
    synchronize(device)
    assert len(actual) == len(expected)
    for actual_item, expected_item in zip(actual, expected):
        category = "exact" if expected_item.dtype == torch.bool else "native_quantization"
        compare(actual_item, expected_item, category=category, dtype=expected_item.dtype)

    expected = _run_fused_moving_avg_obs_fake_quant("cpu", variant="public")
    actual = _run_fused_moving_avg_obs_fake_quant(device, variant="public")
    synchronize(device)
    compare(actual[0], expected[0], category="native_quantization", dtype=torch.float32)
    for actual_item, expected_item in zip(actual[1:], expected[1:]):
        compare(actual_item, expected_item, category="native_quantization", dtype=expected_item.dtype)
