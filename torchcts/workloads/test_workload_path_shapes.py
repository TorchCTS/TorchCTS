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


_WORKLOAD_PATH_SHAPE_RUNNERS = (
    "attention.sdpa",
    "model_patterns.vision_block",
    "model_patterns.transformer_block_fragment",
    "model_patterns.patch_embedding",
    "model_patterns.depthwise_separable",
    "model_patterns.residual_norm",
)


def pytest_generate_tests(metafunc):
    if "path_shape_case" in metafunc.fixturenames:
        metafunc.parametrize(
            "path_shape_case",
            pytest_params_for_runner(_WORKLOAD_PATH_SHAPE_RUNNERS, metafunc.config),
        )


def test_workload_path_shape_case(path_shape_case, device, compare):
    run_path_shape_case(path_shape_case, device, compare)
