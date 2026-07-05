"""FFT-family path-shape runners."""

from __future__ import annotations

import torch

from torchcts.path_shapes.runners.common import compare_tensor, float_data, run_device_op, torch_dtype


def _real_input(case: dict, dtype: torch.dtype):
    s = case["shape"]
    dims = (s.get("batch", 1), s["length"])
    if case.get("stride_mode") == "every_other":
        base = float_data((dims[0], dims[1] * 2), dtype)
        return base[:, ::2][:, : dims[1]]
    return float_data(dims, dtype)


def run_rfft(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    x_cpu = _real_input(case, dtype)
    actual = run_device_op(case, device, lambda: torch.fft.rfft(x_cpu.to(device)))
    expected = torch.fft.rfft(x_cpu)
    compare_tensor(actual, expected, compare, category="fft", dtype=dtype, device=device)


def run_irfft(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    x_cpu = torch.fft.rfft(_real_input(case, dtype))
    n = case["shape"]["length"]
    actual = run_device_op(case, device, lambda: torch.fft.irfft(x_cpu.to(device), n=n))
    expected = torch.fft.irfft(x_cpu, n=n)
    compare_tensor(actual, expected, compare, category="fft", dtype=dtype, device=device)


def run_fft(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    real_cpu = _real_input(case, dtype)
    imag_cpu = float_data(real_cpu.shape, dtype, scale=0.004)
    x_cpu = torch.complex(real_cpu.to(torch.float32), imag_cpu.to(torch.float32))
    actual = run_device_op(case, device, lambda: torch.fft.fft(x_cpu.to(device)))
    expected = torch.fft.fft(x_cpu)
    compare_tensor(actual, expected, compare, category="fft", dtype=torch.float32, device=device)


def run_fft2(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    real_cpu = float_data((s.get("batch", 1), s["h"], s["w"]), dtype)
    imag_cpu = float_data(real_cpu.shape, dtype, scale=0.004)
    x_cpu = torch.complex(real_cpu.to(torch.float32), imag_cpu.to(torch.float32))
    actual = run_device_op(case, device, lambda: torch.fft.fft2(x_cpu.to(device)))
    expected = torch.fft.fft2(x_cpu)
    compare_tensor(actual, expected, compare, category="fft", dtype=torch.float32, device=device)
