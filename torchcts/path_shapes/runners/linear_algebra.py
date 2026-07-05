"""Linear-algebra-family path-shape runners."""

from __future__ import annotations

import torch

from torchcts.path_shapes.runners.common import compare_tensor, float_data, run_device_op, torch_dtype


def _spd(batch_shape, n, dtype):
    base = float_data((*batch_shape, n, n), dtype, scale=0.002)
    eye = torch.eye(n, dtype=dtype).expand(*batch_shape, n, n)
    return base.transpose(-1, -2).matmul(base) + eye * 1.5


def run_solve(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    batch = tuple(s.get("batch_shape", []))
    n, rhs = s["n"], s["rhs"]
    a_cpu = _spd(batch, n, dtype)
    b_cpu = float_data((*batch, n, rhs), dtype, scale=0.01)
    actual = run_device_op(case, device, lambda: torch.linalg.solve(a_cpu.to(device), b_cpu.to(device)))
    expected = torch.linalg.solve(a_cpu, b_cpu)
    compare_tensor(actual, expected, compare, category="linalg", dtype=dtype, device=device)


def run_cholesky(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    a_cpu = _spd(tuple(s.get("batch_shape", [])), s["n"], dtype)
    actual = run_device_op(case, device, lambda: torch.linalg.cholesky(a_cpu.to(device)))
    expected = torch.linalg.cholesky(a_cpu)
    compare_tensor(actual, expected, compare, category="linalg", dtype=dtype, device=device)


def run_qr(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    a_cpu = float_data((*tuple(s.get("batch_shape", [])), s["m"], s["n"]), dtype, scale=0.01)
    actual_q, actual_r = run_device_op(case, device, lambda: torch.linalg.qr(a_cpu.to(device), mode=s.get("mode", "reduced")))
    expected_q, expected_r = torch.linalg.qr(a_cpu, mode=s.get("mode", "reduced"))
    compare_tensor(actual_q.abs(), expected_q.abs(), compare, category="linalg", dtype=dtype, device=device)
    compare_tensor(actual_r.abs(), expected_r.abs(), compare, category="linalg", dtype=dtype, device=device)


def run_eigh(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    a_cpu = _spd(tuple(s.get("batch_shape", [])), s["n"], dtype)
    actual_vals, _actual_vecs = run_device_op(case, device, lambda: torch.linalg.eigh(a_cpu.to(device)))
    expected_vals, _expected_vecs = torch.linalg.eigh(a_cpu)
    compare_tensor(actual_vals, expected_vals, compare, category="linalg", dtype=dtype, device=device)


def run_svdvals(case: dict, device: str, compare) -> None:
    dtype = torch_dtype(case["dtype"])
    s = case["shape"]
    a_cpu = float_data((*tuple(s.get("batch_shape", [])), s["m"], s["n"]), dtype, scale=0.01)
    actual = run_device_op(case, device, lambda: torch.linalg.svdvals(a_cpu.to(device)))
    expected = torch.linalg.svdvals(a_cpu)
    compare_tensor(actual, expected, compare, category="linalg", dtype=dtype, device=device)
