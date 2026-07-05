"""Deterministic path-shape corpus specs."""

from torchcts.path_shapes.specs import (
    attention,
    broadcasting,
    convolution,
    fft,
    indexing,
    linear_algebra,
    matmul,
    model_patterns,
    normalization,
    reduction,
    sorting,
    spatial,
)


SPEC_MODULES = (
    matmul,
    convolution,
    attention,
    reduction,
    indexing,
    sorting,
    fft,
    normalization,
    spatial,
    linear_algebra,
    model_patterns,
    broadcasting,
)
