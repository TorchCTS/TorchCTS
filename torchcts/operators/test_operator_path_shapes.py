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

from torchcts.path_shapes import pytest_params_for_runner
from torchcts.path_shapes.runners import run_path_shape_case


_OPERATOR_PATH_SHAPE_RUNNERS = (
    "matmul.mm",
    "matmul.bmm",
    "matmul.matmul",
    "matmul.addmm",
    "matmul.linear",
    "convolution.conv1d",
    "convolution.conv2d",
    "convolution.conv3d",
    "convolution.conv_transpose2d",
    "reduction.sum",
    "reduction.mean",
    "reduction.amax",
    "reduction.argmax",
    "reduction.prod",
    "indexing.index_select",
    "indexing.gather",
    "indexing.scatter_add",
    "indexing.scatter_reduce",
    "indexing.take",
    "indexing.masked_select",
    "sorting.sort",
    "sorting.topk",
    "sorting.kthvalue",
    "fft.rfft",
    "fft.irfft",
    "fft.fft",
    "fft.fft2",
    "normalization.layer_norm",
    "normalization.group_norm",
    "normalization.batch_norm",
    "spatial.avg_pool2d",
    "spatial.max_pool2d",
    "spatial.adaptive_avg_pool2d",
    "spatial.interpolate",
    "linear_algebra.solve",
    "linear_algebra.cholesky",
    "linear_algebra.qr",
    "linear_algebra.eigh",
    "linear_algebra.svdvals",
    "broadcasting.where",
    "broadcasting.add",
    "broadcasting.mul",
    "broadcasting.masked_fill",
)


def pytest_generate_tests(metafunc):
    if "path_shape_case" in metafunc.fixturenames:
        metafunc.parametrize(
            "path_shape_case",
            pytest_params_for_runner(_OPERATOR_PATH_SHAPE_RUNNERS, metafunc.config),
        )


def test_operator_path_shape_case(path_shape_case, device, compare):
    run_path_shape_case(path_shape_case, device, compare)
