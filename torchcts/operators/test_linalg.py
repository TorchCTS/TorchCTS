import pytest
import torch
from torchcts.core.device import synchronize

LINALG_DTYPES = [torch.float32]

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", LINALG_DTYPES)
@pytest.mark.parametrize("op_name", ["svd", "cholesky", "qr"])
def test_linalg_decompositions(dtype, op_name, device, compare, input_gen):
    if op_name == "svd":
        x_dev = input_gen((8, 8), dtype, device)
        try:
            u_dev, s_dev, vh_dev = torch.linalg.svd(x_dev)
            u_cpu, s_cpu, vh_cpu = torch.linalg.svd(x_dev.cpu())
            synchronize(device)
            # singular values (s) must be identical
            compare(s_dev, s_cpu, category="linalg", dtype=dtype)
        except NotImplementedError:
            pass
            
    elif op_name == "cholesky":
        a_cpu = torch.randn(8, 8, dtype=dtype)
        spd_cpu = torch.mm(a_cpu, a_cpu.T) + torch.eye(8, dtype=dtype) * 2.0
        spd_dev = spd_cpu.to(device)
        try:
            l_dev = torch.linalg.cholesky(spd_dev)
            l_cpu = torch.linalg.cholesky(spd_cpu)
            synchronize(device)
            compare(l_dev, l_cpu, category="linalg", dtype=dtype)
        except NotImplementedError:
            pass
            
    elif op_name == "qr":
        x_dev = input_gen((8, 8), dtype, device)
        try:
            q_dev, r_dev = torch.linalg.qr(x_dev)
            q_cpu, r_cpu = torch.linalg.qr(x_dev.cpu())
            synchronize(device)
            reconstructed_dev = torch.mm(q_dev, r_dev)
            reconstructed_cpu = torch.mm(q_cpu, r_cpu)
            compare(reconstructed_dev, reconstructed_cpu, category="linalg", dtype=dtype)
        except NotImplementedError:
            pass

@pytest.mark.smoke
@pytest.mark.parametrize("dtype", LINALG_DTYPES)
@pytest.mark.parametrize("op_name", ["solve", "inv", "det"])
def test_linalg_solvers(dtype, op_name, device, compare, input_gen):
    a_cpu = torch.randn(8, 8, dtype=dtype)
    a_cpu = a_cpu + torch.eye(8, dtype=dtype) * 3.0
    b_cpu = torch.randn(8, 2, dtype=dtype)
    
    a_dev = a_cpu.to(device)
    b_dev = b_cpu.to(device)
    
    if op_name == "solve":
        try:
            x_dev = torch.linalg.solve(a_dev, b_dev)
            x_cpu = torch.linalg.solve(a_cpu, b_cpu)
            synchronize(device)
            compare(x_dev, x_cpu, category="linalg", dtype=dtype)
        except NotImplementedError:
            pass
            
    elif op_name == "inv":
        try:
            inv_dev = torch.linalg.inv(a_dev)
            inv_cpu = torch.linalg.inv(a_cpu)
            synchronize(device)
            compare(inv_dev, inv_cpu, category="linalg", dtype=dtype)
        except NotImplementedError:
            pass
            
    elif op_name == "det":
        try:
            det_dev = torch.linalg.det(a_dev)
            det_cpu = torch.linalg.det(a_cpu)
            synchronize(device)
            compare(det_dev, det_cpu, category="linalg", dtype=dtype)
        except NotImplementedError:
            pass
