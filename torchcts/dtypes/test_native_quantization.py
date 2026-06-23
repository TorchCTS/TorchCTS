# Copyright (c) 2026 Kris Bailey. MIT License.

import pytest
import torch
from torchcts.core.device import synchronize


@pytest.mark.requires("native_quantization")
@pytest.mark.medium
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
def test_quantize_per_channel(device, compare):
    torch.manual_seed(42)
    x = torch.randn(32, 64, device=device)
    x_cpu = x.cpu()

    scales = torch.rand(32, dtype=torch.float64) * 0.1 + 0.01
    zero_points = torch.zeros(32, dtype=torch.int64)

    qt = torch.quantize_per_channel(x, scales, zero_points, axis=0, dtype=torch.qint8)
    qt_cpu = torch.quantize_per_channel(x_cpu, scales, zero_points, axis=0, dtype=torch.qint8)

    assert qt.qscheme() == torch.per_channel_affine

    deq = qt.dequantize()
    deq_cpu = qt_cpu.dequantize()
    synchronize(device)
    compare(deq, deq_cpu, category="native_quantization", dtype=torch.float32)


@pytest.mark.requires("native_quantization")
@pytest.mark.medium
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
