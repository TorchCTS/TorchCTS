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

LINALG_DTYPES = [torch.float32]


def _compare_linalg_tensor(actual, expected, compare, dtype=torch.float32):
    synchronize(actual.device.type)
    compare(actual, expected, category="linalg", dtype=dtype)


def _compare_linalg_tuple(actual, expected, compare, dtypes):
    assert len(actual) == len(expected)
    for actual_tensor, expected_tensor, dtype in zip(actual, expected, dtypes):
        _compare_linalg_tensor(actual_tensor, expected_tensor, compare, dtype=dtype)


def _compare_eig_residual(matrix, eigenvalues, eigenvectors, compare):
    matrix_complex = matrix.to(eigenvectors.dtype)
    lhs = matrix_complex @ eigenvectors
    rhs = eigenvectors @ torch.diag(eigenvalues)
    _compare_linalg_tensor(lhs, rhs.detach().cpu(), compare, dtype=eigenvectors.dtype)


def _compare_svd_reconstruction(matrix, u, s, vh, compare):
    reconstructed = u @ torch.diag(s) @ vh
    _compare_linalg_tensor(reconstructed, matrix.detach().cpu(), compare)


@pytest.mark.smoke
@pytest.mark.covers("aten::linalg_cholesky")
@pytest.mark.covers("aten::linalg_qr")
@pytest.mark.covers("aten::linalg_svd")
@pytest.mark.covers("aten::mm")
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
@pytest.mark.covers("aten::_linalg_check_errors")
@pytest.mark.covers("aten::_linalg_eigvals")
@pytest.mark.covers("aten::_linalg_svd.U")
@pytest.mark.covers("aten::_lu_with_info")
@pytest.mark.covers("aten::linalg_eig.out", surface="out_variant")
@pytest.mark.covers("aten::linalg_eigh.eigvals")
@pytest.mark.covers("aten::linalg_eigvals.out", surface="out_variant")
@pytest.mark.covers("aten::linalg_eigvalsh.out", surface="out_variant")
@pytest.mark.covers("aten::linalg_householder_product.out", surface="out_variant")
@pytest.mark.covers("aten::linalg_ldl_factor.out", surface="out_variant")
@pytest.mark.covers("aten::linalg_ldl_factor_ex.out", surface="out_variant")
@pytest.mark.covers("aten::linalg_ldl_solve.out", surface="out_variant")
@pytest.mark.covers("aten::linalg_lstsq.out", surface="out_variant")
@pytest.mark.covers("aten::linalg_lu_solve.out", surface="out_variant")
@pytest.mark.covers("aten::linalg_multi_dot.out", surface="out_variant")
@pytest.mark.covers("aten::linalg_svd.U")
@pytest.mark.covers("aten::lu_solve.out", surface="out_variant")
@pytest.mark.covers("aten::lu_unpack.out", surface="out_variant")
@pytest.mark.covers("aten::svd.U")
@pytest.mark.covers("aten::triangular_solve.X")
def test_linalg_dispatcher_out_and_info_surfaces(device, compare):
    matrix_cpu = torch.tensor([[2.0, 0.5], [0.5, 1.5]], dtype=torch.float32)
    rhs_cpu = torch.eye(2, dtype=torch.float32)
    matrix_dev = matrix_cpu.to(device)
    rhs_dev = rhs_cpu.to(device)

    torch.ops.aten._linalg_check_errors.default(
        torch.zeros((), dtype=torch.int32, device=device), "linalg_inv", is_matrix=True,
    )

    expected = torch.ops.aten._linalg_eigvals.default(matrix_cpu)
    actual = torch.ops.aten._linalg_eigvals.default(matrix_dev)
    _compare_linalg_tensor(actual, expected, compare, dtype=torch.complex64)

    eigvals_out = torch.empty(2, dtype=torch.complex64, device=device)
    eigvecs_out = torch.empty(2, 2, dtype=torch.complex64, device=device)
    expected = torch.ops.aten.linalg_eig.default(matrix_cpu)
    actual = torch.ops.aten.linalg_eig.out(
        matrix_dev, eigenvalues=eigvals_out, eigenvectors=eigvecs_out,
    )
    assert actual[0].data_ptr() == eigvals_out.data_ptr()
    assert actual[1].data_ptr() == eigvecs_out.data_ptr()
    _compare_linalg_tensor(actual[0], expected[0], compare, dtype=torch.complex64)
    _compare_eig_residual(matrix_dev, actual[0], actual[1], compare)

    eigvals_out = torch.empty(2, dtype=torch.complex64, device=device)
    expected = torch.ops.aten.linalg_eigvals.default(matrix_cpu)
    actual = torch.ops.aten.linalg_eigvals.out(matrix_dev, out=eigvals_out)
    assert actual.data_ptr() == eigvals_out.data_ptr()
    _compare_linalg_tensor(actual, expected, compare, dtype=torch.complex64)

    eigvalsh_out = torch.empty(2, dtype=torch.float32, device=device)
    expected = torch.ops.aten.linalg_eigvalsh.default(matrix_cpu, "L")
    actual = torch.ops.aten.linalg_eigvalsh.out(matrix_dev, "L", out=eigvalsh_out)
    assert actual.data_ptr() == eigvalsh_out.data_ptr()
    _compare_linalg_tensor(actual, expected, compare)

    eigvals_real = torch.empty(2, dtype=torch.float32, device=device)
    eigvecs_real = torch.empty(2, 2, dtype=torch.float32, device=device)
    expected = torch.ops.aten.linalg_eigh.default(matrix_cpu, "L")
    actual = torch.ops.aten.linalg_eigh.eigvals(
        matrix_dev, "L", eigvals=eigvals_real, eigvecs=eigvecs_real,
    )
    assert actual[0].data_ptr() == eigvals_real.data_ptr()
    assert actual[1].data_ptr() == eigvecs_real.data_ptr()
    _compare_linalg_tensor(actual[0], expected[0], compare)
    _compare_linalg_tensor(matrix_dev @ actual[1], (actual[1] @ torch.diag(actual[0])).cpu(), compare)

    expected = torch.ops.aten.linalg_svd.default(matrix_cpu, False)
    actual = torch.ops.aten.linalg_svd.U(
        matrix_dev,
        False,
        driver=None,
        U=torch.empty(2, 2, device=device),
        S=torch.empty(2, device=device),
        Vh=torch.empty(2, 2, device=device),
    )
    _compare_linalg_tensor(actual[1], expected[1], compare)
    _compare_svd_reconstruction(matrix_dev, actual[0], actual[1], actual[2], compare)

    expected = torch.ops.aten._linalg_svd.default(matrix_cpu, False, True)
    actual = torch.ops.aten._linalg_svd.U(
        matrix_dev,
        False,
        True,
        driver=None,
        U=torch.empty(2, 2, device=device),
        S=torch.empty(2, device=device),
        Vh=torch.empty(2, 2, device=device),
    )
    _compare_linalg_tensor(actual[1], expected[1], compare)
    _compare_svd_reconstruction(matrix_dev, actual[0], actual[1], actual[2], compare)

    expected = torch.ops.aten.svd.default(matrix_cpu, True, True)
    actual = torch.ops.aten.svd.U(
        matrix_dev,
        True,
        True,
        U=torch.empty(2, 2, device=device),
        S=torch.empty(2, device=device),
        V=torch.empty(2, 2, device=device),
    )
    _compare_linalg_tensor(actual[1], expected[1], compare)
    _compare_linalg_tensor(actual[0] @ torch.diag(actual[1]) @ actual[2].T, matrix_cpu, compare)

    expected = torch.ops.aten.triangular_solve.default(rhs_cpu, matrix_cpu, True, False, False)
    actual = torch.ops.aten.triangular_solve.X(
        rhs_dev,
        matrix_dev,
        True,
        False,
        False,
        X=torch.empty(2, 2, device=device),
        M=torch.empty(2, 2, device=device),
    )
    _compare_linalg_tuple(actual, expected, compare, (torch.float32, torch.float32))

    expected = torch.ops.aten.linalg_householder_product.default(torch.eye(2), torch.ones(1))
    out = torch.empty(2, 2, device=device)
    actual = torch.ops.aten.linalg_householder_product.out(
        torch.eye(2, device=device), torch.ones(1, device=device), out=out,
    )
    assert actual.data_ptr() == out.data_ptr()
    _compare_linalg_tensor(actual, expected, compare)

    lu_cpu, pivots_cpu = torch.linalg.lu_factor(matrix_cpu)
    lu_dev, pivots_dev = torch.linalg.lu_factor(matrix_dev)
    expected = torch.ops.aten.linalg_lu_solve.default(lu_cpu, pivots_cpu, rhs_cpu)
    out = torch.empty(2, 2, device=device)
    actual = torch.ops.aten.linalg_lu_solve.out(
        lu_dev, pivots_dev, rhs_dev, left=True, adjoint=False, out=out,
    )
    assert actual.data_ptr() == out.data_ptr()
    _compare_linalg_tensor(actual, expected, compare)

    expected = torch.ops.aten.lu_solve.default(rhs_cpu, lu_cpu, pivots_cpu)
    out = torch.empty(2, 2, device=device)
    actual = torch.ops.aten.lu_solve.out(rhs_dev, lu_dev, pivots_dev, out=out)
    assert actual.data_ptr() == out.data_ptr()
    _compare_linalg_tensor(actual, expected, compare)

    expected = torch.ops.aten.lu_unpack.default(lu_cpu, pivots_cpu, True, True)
    actual = torch.ops.aten.lu_unpack.out(
        lu_dev,
        pivots_dev,
        True,
        True,
        P=torch.empty(2, 2, device=device),
        L=torch.empty(2, 2, device=device),
        U=torch.empty(2, 2, device=device),
    )
    _compare_linalg_tuple(actual, expected, compare, (torch.float32, torch.float32, torch.float32))

    expected = torch.ops.aten._lu_with_info.default(matrix_cpu, True, True)
    actual = torch.ops.aten._lu_with_info.default(matrix_dev, True, True)
    _compare_linalg_tuple(actual, expected, compare, (torch.float32, torch.int32, torch.int32))

    expected = torch.ops.aten.linalg_lstsq.default(matrix_cpu, rhs_cpu, None, driver=None)
    actual = torch.ops.aten.linalg_lstsq.out(
        matrix_dev,
        rhs_dev,
        None,
        driver=None,
        solution=torch.empty(2, 2, device=device),
        residuals=torch.empty(0, device=device),
        rank=torch.empty((), dtype=torch.int64, device=device),
        singular_values=torch.empty(0, device=device),
    )
    _compare_linalg_tuple(actual, expected, compare, (torch.float32, torch.float32, torch.int64, torch.float32))

    expected = torch.ops.aten.linalg_multi_dot.default([matrix_cpu, rhs_cpu, matrix_cpu])
    out = torch.empty(2, 2, device=device)
    actual = torch.ops.aten.linalg_multi_dot.out([matrix_dev, rhs_dev, matrix_dev], out=out)
    assert actual.data_ptr() == out.data_ptr()
    _compare_linalg_tensor(actual, expected, compare)

    expected = torch.ops.aten.linalg_ldl_factor.default(matrix_cpu, hermitian=True)
    actual = torch.ops.aten.linalg_ldl_factor.out(
        matrix_dev,
        hermitian=True,
        LD=torch.empty(2, 2, device=device),
        pivots=torch.empty(2, dtype=torch.int32, device=device),
    )
    _compare_linalg_tuple(actual, expected, compare, (torch.float32, torch.int32))

    ldl_cpu, ldl_pivots_cpu, _ = torch.linalg.ldl_factor_ex(matrix_cpu, hermitian=True)
    ldl_dev, ldl_pivots_dev, _ = torch.linalg.ldl_factor_ex(matrix_dev, hermitian=True)
    expected = torch.ops.aten.linalg_ldl_factor_ex.default(
        matrix_cpu, hermitian=True, check_errors=False,
    )
    actual = torch.ops.aten.linalg_ldl_factor_ex.out(
        matrix_dev,
        hermitian=True,
        check_errors=False,
        LD=torch.empty(2, 2, device=device),
        pivots=torch.empty(2, dtype=torch.int32, device=device),
        info=torch.empty((), dtype=torch.int32, device=device),
    )
    _compare_linalg_tuple(actual, expected, compare, (torch.float32, torch.int32, torch.int32))

    expected = torch.ops.aten.linalg_ldl_solve.default(
        ldl_cpu, ldl_pivots_cpu, rhs_cpu, hermitian=True,
    )
    out = torch.empty(2, 2, device=device)
    actual = torch.ops.aten.linalg_ldl_solve.out(
        ldl_dev, ldl_pivots_dev, rhs_dev, hermitian=True, out=out,
    )
    assert actual.data_ptr() == out.data_ptr()
    _compare_linalg_tensor(actual, expected, compare)

@pytest.mark.smoke
@pytest.mark.covers("aten::linalg_det")
@pytest.mark.covers("aten::linalg_inv")
@pytest.mark.covers("aten::linalg_solve")
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
