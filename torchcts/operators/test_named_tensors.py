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

from __future__ import annotations

import math

import pytest
import torch

from torchcts.core.device import synchronize


pytestmark = [
    pytest.mark.smoke,
    pytest.mark.requires("named_tensor"),
    pytest.mark.covers_category("named_tensor"),
    pytest.mark.filterwarnings("ignore:Named tensors and all their associated APIs are an experimental feature:UserWarning"),
]


def _named_arange(shape: tuple[int, ...], names: tuple[str, ...], *, device: str) -> torch.Tensor:
    values = torch.arange(math.prod(shape), dtype=torch.float32, device=device).reshape(shape)
    return values.refine_names(*names)


def _assert_named_tensor_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.names == expected.names
    assert actual.dtype == expected.dtype
    assert tuple(actual.shape) == tuple(expected.shape)
    torch.testing.assert_close(actual.rename(None).detach().cpu(), expected.rename(None).detach().cpu())


def _assert_named_sequence_equal(actual, expected) -> None:
    assert len(actual) == len(expected)
    for actual_item, expected_item in zip(actual, expected):
        _assert_named_tensor_equal(actual_item, expected_item)


@pytest.mark.covers("aten::align_as")
@pytest.mark.covers("aten::align_to")
@pytest.mark.covers("aten::align_to.ellipsis_idx")
@pytest.mark.covers("aten::diagonal.Dimname")
@pytest.mark.covers("aten::flatten.DimnameList")
@pytest.mark.covers("aten::flatten.named_out_dim")
@pytest.mark.covers("aten::flatten.using_names")
@pytest.mark.covers("aten::refine_names")
@pytest.mark.covers("aten::rename")
@pytest.mark.covers("aten::rename_")
@pytest.mark.covers("aten::select.Dimname")
@pytest.mark.covers("aten::size.Dimname")
@pytest.mark.covers("aten::squeeze.dimname")
@pytest.mark.covers("aten::stride.Dimname")
@pytest.mark.covers("aten::transpose.Dimname")
@pytest.mark.covers("aten::unbind.Dimname")
@pytest.mark.covers("aten::unflatten.Dimname")
def test_named_tensor_view_and_metadata_public_api(device):
    x = _named_arange((2, 3, 4), ("batch", "rows", "cols"), device=device)
    x_cpu = x.detach().cpu()
    matrix = _named_arange((3, 4), ("rows", "cols"), device=device)
    matrix_cpu = matrix.detach().cpu()

    _assert_named_tensor_equal(x.align_to("cols", "batch", "rows"), x_cpu.align_to("cols", "batch", "rows"))
    _assert_named_tensor_equal(x.align_to("cols", ..., "rows"), x_cpu.align_to("cols", ..., "rows"))
    _assert_named_tensor_equal(matrix.align_as(x), matrix_cpu.align_as(x_cpu))
    _assert_named_tensor_equal(matrix.diagonal(outdim="diag", dim1="rows", dim2="cols"), matrix_cpu.diagonal(outdim="diag", dim1="rows", dim2="cols"))
    _assert_named_tensor_equal(x.flatten(["rows", "cols"], "features"), x_cpu.flatten(["rows", "cols"], "features"))
    _assert_named_tensor_equal(x.flatten(1, 2, "features"), x_cpu.flatten(1, 2, "features"))
    _assert_named_tensor_equal(x.flatten("rows", "cols", "features"), x_cpu.flatten("rows", "cols", "features"))
    _assert_named_tensor_equal(matrix.rename(None).refine_names("rows", "cols"), matrix_cpu.rename(None).refine_names("rows", "cols"))
    _assert_named_tensor_equal(matrix.rename("height", "width"), matrix_cpu.rename("height", "width"))
    renamed = matrix.clone()
    renamed_cpu = matrix_cpu.clone()
    assert renamed.rename_("height", "width") is renamed
    assert renamed_cpu.rename_("height", "width") is renamed_cpu
    _assert_named_tensor_equal(renamed, renamed_cpu)
    _assert_named_tensor_equal(x.select("rows", 1), x_cpu.select("rows", 1))
    _assert_named_tensor_equal(x[:, 0:1, :].squeeze("rows"), x_cpu[:, 0:1, :].squeeze("rows"))
    _assert_named_tensor_equal(matrix.transpose("rows", "cols"), matrix_cpu.transpose("rows", "cols"))
    _assert_named_sequence_equal(x.unbind("rows"), x_cpu.unbind("rows"))
    flattened = x.flatten(["rows", "cols"], "features")
    flattened_cpu = x_cpu.flatten(["rows", "cols"], "features")
    _assert_named_tensor_equal(
        flattened.unflatten("features", (("rows", 3), ("cols", 4))),
        flattened_cpu.unflatten("features", (("rows", 3), ("cols", 4))),
    )
    assert x.size("cols") == x_cpu.size("cols") == 4
    assert x.stride("cols") == x_cpu.stride("cols")
    synchronize(device)


@pytest.mark.covers("aten::cummax.dimname_out")
@pytest.mark.covers("aten::cummin.dimname_out")
@pytest.mark.covers("aten::cumprod.dimname_out")
@pytest.mark.covers("aten::cumsum.dimname_out")
@pytest.mark.covers("aten::logcumsumexp.dimname_out")
@pytest.mark.covers("aten::logsumexp.names_out")
@pytest.mark.covers("aten::mean.names_out")
@pytest.mark.covers("aten::prod.Dimname_out")
@pytest.mark.covers("aten::std.correction_names_out")
@pytest.mark.covers("aten::std.names_out")
@pytest.mark.covers("aten::sum.DimnameList_out")
@pytest.mark.covers("aten::var.correction_names_out")
@pytest.mark.covers("aten::var.names_out")
def test_named_tensor_reduction_out_public_api(device):
    x = _named_arange((3, 4), ("rows", "cols"), device=device)
    x_cpu = x.detach().cpu()

    def check_out(call, expected_call, out_shape=(3,), out_dtype=torch.float32):
        out = torch.empty(out_shape, dtype=out_dtype, device=device)
        expected_out = torch.empty(out_shape, dtype=out_dtype)
        actual = call(out)
        expected = expected_call(expected_out)
        assert actual is out
        _assert_named_tensor_equal(actual, expected)

    check_out(lambda out: torch.sum(x, dim=["cols"], out=out), lambda out: torch.sum(x_cpu, dim=["cols"], out=out))
    check_out(lambda out: torch.mean(x, dim=["cols"], out=out), lambda out: torch.mean(x_cpu, dim=["cols"], out=out))
    check_out(lambda out: torch.prod(x + 1, dim="cols", out=out), lambda out: torch.prod(x_cpu + 1, dim="cols", out=out))
    check_out(lambda out: torch.logsumexp(x, dim=["cols"], out=out), lambda out: torch.logsumexp(x_cpu, dim=["cols"], out=out))
    check_out(lambda out: torch.std(x, dim=["cols"], unbiased=False, out=out), lambda out: torch.std(x_cpu, dim=["cols"], unbiased=False, out=out))
    check_out(lambda out: torch.std(x, dim=["cols"], correction=0, out=out), lambda out: torch.std(x_cpu, dim=["cols"], correction=0, out=out))
    check_out(lambda out: torch.var(x, dim=["cols"], unbiased=False, out=out), lambda out: torch.var(x_cpu, dim=["cols"], unbiased=False, out=out))
    check_out(lambda out: torch.var(x, dim=["cols"], correction=0, out=out), lambda out: torch.var(x_cpu, dim=["cols"], correction=0, out=out))
    check_out(lambda out: torch.cumsum(x, dim="cols", out=out), lambda out: torch.cumsum(x_cpu, dim="cols", out=out), out_shape=(3, 4))
    check_out(lambda out: torch.cumprod(x + 1, dim="cols", out=out), lambda out: torch.cumprod(x_cpu + 1, dim="cols", out=out), out_shape=(3, 4))
    check_out(lambda out: torch.logcumsumexp(x, dim="cols", out=out), lambda out: torch.logcumsumexp(x_cpu, dim="cols", out=out), out_shape=(3, 4))

    for op in (torch.cummax, torch.cummin):
        values = torch.empty((3, 4), dtype=torch.float32, device=device)
        indices = torch.empty((3, 4), dtype=torch.long, device=device)
        expected_values = torch.empty((3, 4), dtype=torch.float32)
        expected_indices = torch.empty((3, 4), dtype=torch.long)
        actual = op(x, dim="cols", out=(values, indices))
        expected = op(x_cpu, dim="cols", out=(expected_values, expected_indices))
        assert actual[0] is values
        assert actual[1] is indices
        _assert_named_sequence_equal(actual, expected)
    synchronize(device)


@pytest.mark.covers("aten::cumprod_.dimname")
@pytest.mark.covers("aten::cumsum_.dimname")
@pytest.mark.covers("aten::index_fill_.Dimname_Scalar")
@pytest.mark.covers("aten::index_fill_.Dimname_Tensor")
def test_named_tensor_inplace_public_api(device):
    x = _named_arange((3, 4), ("rows", "cols"), device=device)
    x_cpu = x.detach().cpu()
    index = torch.tensor([0, 2], dtype=torch.long, device=device)
    index_cpu = index.cpu()

    actual = x.clone()
    expected = x_cpu.clone()
    assert actual.cumsum_("cols") is actual
    assert expected.cumsum_("cols") is expected
    _assert_named_tensor_equal(actual, expected)

    actual = (x + 1).clone()
    expected = (x_cpu + 1).clone()
    assert actual.cumprod_("cols") is actual
    assert expected.cumprod_("cols") is expected
    _assert_named_tensor_equal(actual, expected)

    actual = x.clone()
    expected = x_cpu.clone()
    assert actual.index_fill_("rows", index, 9.0) is actual
    assert expected.index_fill_("rows", index_cpu, 9.0) is expected
    _assert_named_tensor_equal(actual, expected)

    actual = x.clone()
    expected = x_cpu.clone()
    assert actual.index_fill_("rows", index, torch.tensor(5.0, device=device)) is actual
    assert expected.index_fill_("rows", index_cpu, torch.tensor(5.0)) is expected
    _assert_named_tensor_equal(actual, expected)
    synchronize(device)


@pytest.mark.covers("aten::cat.names_out")
@pytest.mark.covers("aten::concat.names_out")
@pytest.mark.covers("aten::concatenate.names_out")
def test_named_tensor_cat_out_public_api(device):
    a = torch.ones((2, 4), dtype=torch.float32, device=device).refine_names("rows", "cols")
    b = torch.full((3, 4), 2.0, dtype=torch.float32, device=device).refine_names("rows", "cols")
    a_cpu = a.detach().cpu()
    b_cpu = b.detach().cpu()

    for op in (torch.cat, torch.concat, torch.concatenate):
        out = torch.empty((5, 4), dtype=torch.float32, device=device)
        expected_out = torch.empty((5, 4), dtype=torch.float32)
        actual = op([a, b], dim="rows", out=out)
        expected = op([a_cpu, b_cpu], dim="rows", out=expected_out)
        assert actual is out
        _assert_named_tensor_equal(actual, expected)
    synchronize(device)
