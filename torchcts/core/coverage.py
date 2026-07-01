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

import ast
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import datetime as _datetime
import json
import os
from pprint import pformat
import re
import runpy
from pathlib import Path

import torch

from torchcts.core.semantic_levels import (
    SemanticLevelInfo,
    generated_level_for_entry,
    marker_value_to_level,
    semantic_level_description,
    suite_default_level,
    suite_for_path,
    validate_semantic_level,
)
from torchcts.core.dtype_contracts import mismatch_counts as dtype_contract_mismatch_counts
from torchcts.core.oracles import oracle_spec_for
from torchcts.op_metadata import runtime_unavailable_op_entries


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_EXCLUSIONS = Path("coverage_exclusions.json")
DEFAULT_OUTPUT_DIR = Path("results") / "coverage"
DEFAULT_INVENTORY_PATH = DEFAULT_OUTPUT_DIR / "inventory.json"
DEFAULT_AUDIT_PATH = DEFAULT_OUTPUT_DIR / "audit.json"
DEFAULT_GENERATED_CASES_PATH = DEFAULT_OUTPUT_DIR / "generated_cases.json"
DEFAULT_UNKNOWNS_PATH = DEFAULT_OUTPUT_DIR / "unknowns.md"
DEFAULT_UNMAPPED_TESTS_PATH = DEFAULT_OUTPUT_DIR / "unmapped_tests.md"
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_DIR / "summary.md"
DEFAULT_SEMANTIC_LEVELS_PATH = DEFAULT_OUTPUT_DIR / "semantic_levels.md"
DEFAULT_PENDING_REVIEW_PATH = DEFAULT_OUTPUT_DIR / "pending_review.json"
DEFAULT_PENDING_REVIEW_MD_PATH = DEFAULT_OUTPUT_DIR / "pending_review.md"
PACKAGE_EXCLUSIONS_PATH = PACKAGE_ROOT / "coverage_exclusions.json"
DEFAULT_GENERATED_CASES_MODULE_PATH = PACKAGE_ROOT / "generated" / "generated_cases.py"

DISPATCH_KEYS = (
    "CPU",
    "MPS",
    "CUDA",
    "Meta",
    "CompositeImplicitAutograd",
    "CompositeExplicitAutograd",
    "PrivateUse1",
)

EXCLUSION_CATEGORIES = frozenset({
    "backend_specific_internal",
    "dispatcher_plumbing",
    "distributed_or_c10d",
    "cpu_reference_invalid",
    "unsafe_direct_invocation",
    "covered_by_public_surface",
    "deprecated_or_removed",
    "manual_future_scope",
})

EXCLUSION_MATCH_TYPES = frozenset({"exact", "base", "regex"})
FINAL_STATUSES = frozenset({
    "covered_opinfo",
    "covered_handwritten",
    "covered_generated",
    "covered_oracle",
    "covered_backend_pack",
    "covered_property",
    "pending_oracle",
    "pending_backend_pack",
    "pending_property",
    "excluded",
    "excluded_framework_plumbing",
    "excluded_deprecated_or_removed",
    "excluded_unsupported_public_api",
    "excluded_distributed_scope",
    "excluded_host_storage",
    "unknown",
    "not_backend_relevant",
    "unavailable_in_pytorch_runtime",
})

COVERED_STATUSES = frozenset({
    "covered_opinfo",
    "covered_handwritten",
    "covered_generated",
    "covered_oracle",
    "covered_backend_pack",
    "covered_property",
})

PENDING_STATUSES = frozenset({
    "pending_oracle",
    "pending_backend_pack",
    "pending_property",
})

RUNTIME_UNAVAILABLE_STATUSES = frozenset({
    "unavailable_in_pytorch_runtime",
})

EXCLUDED_STATUSES = frozenset({
    "excluded",
    "excluded_framework_plumbing",
    "excluded_deprecated_or_removed",
    "excluded_unsupported_public_api",
    "excluded_distributed_scope",
    "excluded_host_storage",
})

PENDING_BLOCKER_TYPES = frozenset({
    "needs_cpu_oracle",
    "needs_backend_pack",
    "needs_property_runner",
    "needs_public_proxy_proof",
    "needs_valid_internal_inputs",
    "kernel_unavailable_in_host_build",
    "python_binding_uninvokable",
    "out_of_backend_conformance_scope",
})

SURFACE_KINDS = frozenset({
    "autograd_backward",
    "factory",
    "functional_data",
    "layout_storage",
    "metadata_device",
    "mutating_or_inplace",
    "not_backend_relevant",
    "out_variant",
    "rng",
    "view_or_alias",
})

RNG_BASE_PATTERNS = (
    "rand",
    "randn",
    "randint",
    "random",
    "normal",
    "uniform",
    "bernoulli",
    "multinomial",
    "poisson",
    "exponential",
    "geometric",
    "cauchy",
    "log_normal",
)

LAYOUT_BASE_PATTERNS = (
    "sparse",
    "nested",
    "storage",
    "stride",
    "strided",
    "indices",
    "values",
    "coalesce",
    "compressed",
    "crow_indices",
    "col_indices",
    "ccol_indices",
    "row_indices",
)

VIEW_BASE_PATTERNS = (
    "view",
    "reshape",
    "alias",
    "detach",
    "transpose",
    "permute",
    "select",
    "slice",
    "split",
    "squeeze",
    "unsqueeze",
    "flatten",
    "expand",
    "as_strided",
    "diagonal",
    "unfold",
)

CURATED_METADATA_BASES = frozenset({
    "can_cast",
    "promote_types",
    "result_type",
})

GENERATED_METADATA_BASES = frozenset({
    "_assert_tensor_metadata",
    "_assert_async",
    "_debug_has_internal_overlap",
    "_dim_arange",
    "_dimI",
    "_dimV",
    "_has_compatible_shallow_copy_type",
    "_is_all_true",
    "_is_any_true",
    "_is_zerotensor",
    "_local_scalar_dense",
    "_shape_as_tensor",
    "_version",
    "dense_dim",
    "dim",
    "is_complex",
    "is_conj",
    "is_contiguous",
    "is_floating_point",
    "is_inference",
    "is_leaf",
    "is_neg",
    "is_nonzero",
    "is_pinned",
    "is_same_size",
    "is_set_to",
    "is_signed",
    "numel",
    "output_nr",
    "result_type",
    "retains_grad",
    "size",
    "sym_is_contiguous",
    "sym_numel",
    "sym_size",
})

OPINFO_ALIAS_OVERRIDES = {
    "bitwise_and": ("__and__",),
    "bitwise_left_shift": ("__lshift__",),
    "bitwise_or": ("__or__",),
    "bitwise_right_shift": ("__rshift__",),
    "bitwise_xor": ("__xor__",),
}

GENERATED_OPINFO_OUT_ALLOWLIST = frozenset({
    "abs",
    "acos",
    "acosh",
    "add",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "ceil",
    "cos",
    "cosh",
    "div",
    "erf",
    "erfc",
    "exp",
    "exp2",
    "expm1",
    "floor",
    "frac",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "log2",
    "mul",
    "neg",
    "reciprocal",
    "round",
    "rsqrt",
    "sigmoid",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "sub",
    "tan",
    "tanh",
    "trunc",
})

GENERATED_OPINFO_UNARY_INPLACE_ALLOWLIST = frozenset({
    "abs",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "ceil",
    "cos",
    "cosh",
    "erf",
    "erfc",
    "exp",
    "exp2",
    "expm1",
    "floor",
    "frac",
    "log",
    "log10",
    "log1p",
    "log2",
    "neg",
    "reciprocal",
    "round",
    "rsqrt",
    "sigmoid",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "trunc",
})

GENERATED_OPINFO_VIEW_STRATEGIES = {
    "aten::broadcast_to": "broadcast_to",
    "aten::conj": "conj",
    "aten::diagonal": "diagonal",
    "aten::expand": "expand",
    "aten::expand_as": "expand_as",
    "aten::flatten.using_ints": "flatten",
    "aten::imag": "imag",
    "aten::mH": "mH",
    "aten::mT": "mT",
    "aten::narrow": "narrow",
    "aten::permute": "permute",
    "aten::positive": "positive",
    "aten::ravel": "ravel",
    "aten::real": "real",
    "aten::reshape": "reshape",
    "aten::reshape_as": "reshape_as",
    "aten::select.int": "select",
    "aten::slice.Tensor": "slice",
    "aten::t": "t",
    "aten::transpose.int": "transpose",
    "aten::unfold": "unfold",
    "aten::unsqueeze": "unsqueeze",
    "aten::view": "view",
    "aten::view_as": "view_as",
}

GENERATED_MANUAL_SHAPE_VIEW_SURFACES = frozenset({
    "aten::_cast_Byte",
    "aten::_cast_Char",
    "aten::_cast_Double",
    "aten::_cast_Float",
    "aten::_cast_Half",
    "aten::_cast_Int",
    "aten::_cast_Long",
    "aten::_cast_Short",
    "aten::_conj_copy",
    "aten::_copy_from",
    "aten::_copy_from_and_resize",
    "aten::_neg_view",
    "aten::_neg_view_copy",
    "aten::_new_zeros_with_same_feature_meta",
    "aten::_reshape_alias",
    "aten::_reshape_alias_copy",
    "aten::_reshape_from_tensor",
    "aten::_reshape_copy",
    "aten::_stack",
    "aten::_to_copy",
    "aten::adjoint",
    "aten::alias",
    "aten::alias_copy",
    "aten::channel_shuffle",
    "aten::chunk",
    "aten::contiguous",
    "aten::copy",
    "aten::data",
    "aten::detach",
    "aten::detach_copy",
    "aten::diagonal_copy",
    "aten::dsplit.array",
    "aten::dsplit.int",
    "aten::expand_copy",
    "aten::fft_fftshift",
    "aten::fft_ifftshift",
    "aten::flatten_dense_tensors",
    "aten::hsplit.array",
    "aten::hsplit.int",
    "aten::lift_fresh_copy",
    "aten::lift",
    "aten::linalg_diagonal",
    "aten::matrix_H",
    "aten::moveaxis.int",
    "aten::moveaxis.intlist",
    "aten::movedim.int",
    "aten::movedim.intlist",
    "aten::narrow.Tensor",
    "aten::native_channel_shuffle",
    "aten::nonzero_numpy",
    "aten::numpy_T",
    "aten::pad_sequence",
    "aten::permute_copy",
    "aten::pixel_shuffle",
    "aten::pixel_unshuffle",
    "aten::resolve_conj",
    "aten::resolve_neg",
    "aten::select_copy.int",
    "aten::set",
    "aten::set.source_Storage",
    "aten::set.source_Storage_storage_offset",
    "aten::set.source_Tensor",
    "aten::slice_inverse",
    "aten::slice_copy.Tensor",
    "aten::split.sizes",
    "aten::split_copy.Tensor",
    "aten::split_with_sizes",
    "aten::split_with_sizes_copy",
    "aten::squeeze",
    "aten::squeeze.dim",
    "aten::squeeze.dims",
    "aten::squeeze_copy",
    "aten::squeeze_copy.dim",
    "aten::squeeze_copy.dims",
    "aten::swapaxes",
    "aten::swapdims",
    "aten::to.device",
    "aten::to.dtype",
    "aten::to.dtype_layout",
    "aten::to.other",
    "aten::to_dense",
    "aten::transpose_copy.int",
    "aten::tensor_split.indices",
    "aten::tensor_split.sections",
    "aten::tensor_split.tensor_indices_or_sections",
    "aten::unfold_copy",
    "aten::unflatten.int",
    "aten::unflatten_dense_tensors",
    "aten::unsafe_split.Tensor",
    "aten::unsafe_split_with_sizes",
    "aten::unsqueeze_copy",
    "aten::view_as_complex",
    "aten::view_as_complex_copy",
    "aten::view_as_real_copy",
    "aten::view.dtype",
    "aten::view_copy.dtype",
    "aten::view_copy",
    "aten::vsplit.array",
    "aten::vsplit.int",
    "aten::type_as",
    "aten::zero",
})

GENERATED_MANUAL_SHAPE_OUT_SURFACES = frozenset({
    "aten::_chunk_cat.out",
    "aten::_conj_copy.out",
    "aten::_copy_from.out",
    "aten::_copy_from_and_resize.out",
    "aten::_neg_view_copy.out",
    "aten::_new_zeros_with_same_feature_meta.out",
    "aten::_reshape_alias_copy.out",
    "aten::_stack.out",
    "aten::_to_copy.out",
    "aten::alias_copy.out",
    "aten::as_strided_scatter.out",
    "aten::argsort.stable_out",
    "aten::as_strided_copy.out",
    "aten::cat.out",
    "aten::channel_shuffle.out",
    "aten::clone.out",
    "aten::block_diag.out",
    "aten::column_stack.out",
    "aten::concat.out",
    "aten::concatenate.out",
    "aten::copy.out",
    "aten::detach_copy.out",
    "aten::diag.out",
    "aten::diag_embed.out",
    "aten::diagonal_copy.out",
    "aten::diff.out",
    "aten::dstack.out",
    "aten::expand_copy.out",
    "aten::flip.out",
    "aten::hstack.out",
    "aten::glu.out",
    "aten::msort.out",
    "aten::narrow_copy.out",
    "aten::nonzero_static.out",
    "aten::permute_copy.out",
    "aten::repeat.out",
    "aten::repeat_interleave.Tensor_out",
    "aten::roll.out",
    "aten::rot90.out",
    "aten::row_stack.out",
    "aten::select_copy.int_out",
    "aten::set.out",
    "aten::set.source_Storage_out",
    "aten::set.source_Storage_storage_offset_out",
    "aten::set.source_Tensor_out",
    "aten::slice_copy.Tensor_out",
    "aten::split_copy.Tensor_out",
    "aten::split_with_sizes_copy.out",
    "aten::squeeze_copy.dim_out",
    "aten::squeeze_copy.dims_out",
    "aten::squeeze_copy.out",
    "aten::stack.out",
    "aten::t_copy.out",
    "aten::transpose_copy.int_out",
    "aten::tril_indices.out",
    "aten::triu_indices.out",
    "aten::unbind_copy.int_out",
    "aten::unfold_copy.out",
    "aten::unsqueeze_copy.out",
    "aten::lift_fresh_copy.out",
    "aten::lift.out",
    "aten::pixel_shuffle.out",
    "aten::pixel_unshuffle.out",
    "aten::view_as_complex_copy.out",
    "aten::view_as_real_copy.out",
    "aten::view_copy.out",
    "aten::view_copy.dtype_out",
    "aten::vstack.out",
    "aten::zero.out",
})

GENERATED_MANUAL_SHAPE_INPLACE_SURFACES = frozenset({
    "aten::as_strided_",
    "aten::fill_diagonal_",
    "aten::set_",
    "aten::set_.source_Storage",
    "aten::set_.source_Storage_storage_offset",
    "aten::set_.source_Tensor",
    "aten::set_.source_Tensor_storage_offset",
    "aten::set_data",
    "aten::squeeze_",
    "aten::squeeze_.dim",
    "aten::squeeze_.dims",
    "aten::swapaxes_",
    "aten::swapdims_",
    "aten::t_",
    "aten::transpose_",
    "aten::tril_",
    "aten::triu_",
    "aten::unsqueeze_",
    "aten::zero_",
})

GENERATED_FACTORY_STRATEGIES = {
    "aten::_efficientzerotensor": "zero_tensor",
    "aten::bartlett_window": "window",
    "aten::bartlett_window.periodic": "window",
    "aten::blackman_window": "window",
    "aten::blackman_window.periodic": "window",
    "aten::fft_fftfreq": "frequency",
    "aten::fft_rfftfreq": "frequency",
    "aten::hamming_window": "window",
    "aten::hamming_window.periodic": "window",
    "aten::hamming_window.periodic_alpha": "window",
    "aten::hamming_window.periodic_alpha_beta": "window",
    "aten::hann_window": "window",
    "aten::hann_window.periodic": "window",
    "aten::kaiser_window": "window",
    "aten::kaiser_window.beta": "window",
    "aten::kaiser_window.periodic": "window",
    "aten::empty_permuted": "empty",
    "aten::range": "range",
    "aten::range.step": "range",
}

GENERATED_SAFE_INTERNAL_FACTORIES = frozenset({
    "aten::_efficientzerotensor",
})

GENERATED_FOREACH_UNARY_ALLOWLIST = frozenset({
    "abs",
    "acos",
    "asin",
    "atan",
    "ceil",
    "clone",
    "cos",
    "cosh",
    "erf",
    "erfc",
    "exp",
    "expm1",
    "floor",
    "frac",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "log2",
    "max",
    "neg",
    "reciprocal",
    "round",
    "rsqrt",
    "sigmoid",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "trunc",
    "zero",
})

GENERATED_FOREACH_BASIC_BINARY_ALLOWLIST = frozenset({"add", "div", "mul", "sub"})
GENERATED_FOREACH_BASIC_BINARY_OVERLOADS = frozenset({"List", "Scalar", "ScalarList", "Tensor"})
GENERATED_FOREACH_EXTREMA_ALLOWLIST = frozenset({"clamp_max", "clamp_min", "maximum", "minimum"})
GENERATED_FOREACH_EXTREMA_OVERLOADS = frozenset({"List", "Scalar", "ScalarList"})
GENERATED_FOREACH_POW_OVERLOADS = frozenset({"List", "Scalar", "ScalarList", "ScalarAndTensor"})
GENERATED_FOREACH_TERNARY_ALLOWLIST = frozenset({"addcdiv", "addcmul"})
GENERATED_FOREACH_TERNARY_OVERLOADS = frozenset({"Scalar", "ScalarList", "Tensor"})
GENERATED_FOREACH_LERP_OVERLOADS = frozenset({"List", "Scalar", "ScalarList"})
GENERATED_FOREACH_NORM_OVERLOADS = frozenset({"Scalar"})

GENERATED_BITWISE_BASES = frozenset({
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_not",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
})

BITWISE_DUNDER_BASES = {
    "__iand__": "bitwise_and",
    "__ilshift__": "bitwise_left_shift",
    "__ior__": "bitwise_or",
    "__irshift__": "bitwise_right_shift",
    "__ixor__": "bitwise_xor",
    "__lshift__": "bitwise_left_shift",
    "__rshift__": "bitwise_right_shift",
}

GENERATED_ELEMENTWISE_BASES = frozenset({
    "_add_relu",
    "_conj_physical",
    "absolute",
    "add",
    "addcdiv",
    "addcmul",
    "angle",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "atan2",
    "celu",
    "clamp",
    "clamp_max",
    "clamp_min",
    "clip",
    "complex",
    "conj_physical",
    "copysign",
    "deg2rad",
    "digamma",
    "div",
    "divide",
    "eq",
    "elu",
    "erfinv",
    "fmax",
    "fmin",
    "float_power",
    "floor_divide",
    "fill",
    "fmod",
    "fix",
    "gcd",
    "ge",
    "gelu",
    "greater",
    "greater_equal",
    "gt",
    "hardshrink",
    "hardsigmoid",
    "hardswish",
    "hardtanh",
    "heaviside",
    "hypot",
    "i0",
    "isinf",
    "isnan",
    "isneginf",
    "isposinf",
    "lcm",
    "le",
    "leaky_relu",
    "less",
    "less_equal",
    "lerp",
    "ldexp",
    "lgamma",
    "log_sigmoid",
    "logaddexp",
    "logaddexp2",
    "logit",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "lt",
    "maximum",
    "minimum",
    "mish",
    "mul",
    "multiply",
    "mvlgamma",
    "nan_to_num",
    "ne",
    "negative",
    "nextafter",
    "not_equal",
    "polar",
    "polygamma",
    "pow",
    "prelu",
    "remainder",
    "relu",
    "relu6",
    "rad2deg",
    "round",
    "rsub",
    "selu",
    "sgn",
    "sign",
    "signbit",
    "silu",
    "sinc",
    "softplus",
    "softshrink",
    "square",
    "sub",
    "subtract",
    "threshold",
    "true_divide",
    "where",
})

GENERATED_REDUCTION_BASES = frozenset({
    "_log_softmax_backward_data",
    "_log_softmax",
    "_logcumsumexp",
    "_segment_reduce_backward",
    "_safe_softmax",
    "_softmax_backward_data",
    "_softmax",
    "all",
    "amax",
    "amin",
    "aminmax",
    "any",
    "argmax",
    "argmin",
    "count_nonzero",
    "cumprod",
    "cumsum",
    "frobenius_norm",
    "histc",
    "logcumsumexp",
    "logsumexp",
    "mean",
    "nanmean",
    "nanquantile",
    "nansum",
    "norm",
    "prod",
    "quantile",
    "renorm",
    "segment_reduce",
    "log_softmax",
    "softmax",
    "std",
    "sum",
    "trace",
    "var",
})

GENERATED_INDEXING_BASES = frozenset({
    "bincount",
    "bucketize",
    "diagonal_scatter",
    "embedding",
    "embedding_renorm",
    "gather",
    "_index_put_impl",
    "_masked_scale",
    "_masked_softmax",
    "index",
    "index_add",
    "index_copy",
    "index_fill",
    "index_put",
    "index_reduce",
    "index_select",
    "isin",
    "masked_fill",
    "masked_scatter",
    "masked_select",
    "nonzero",
    "one_hot",
    "put",
    "scatter",
    "scatter_add",
    "scatter_reduce",
    "select_scatter",
    "slice_scatter",
    "searchsorted",
    "take",
    "take_along_dim",
})

GENERATED_RNG_BASES = frozenset({
    "_sample_dirichlet",
    "_standard_gamma",
    "bernoulli",
    "binomial",
    "cauchy",
    "exponential",
    "geometric",
    "log_normal",
    "multinomial",
    "normal",
    "poisson",
    "rand",
    "rand_like",
    "randint",
    "randint_like",
    "randn",
    "randn_like",
    "randperm",
    "random",
    "random_",
    "uniform",
})

GENERATED_SPECIAL_MATH_BASES = frozenset({
    "_dirichlet_grad",
    "_standard_gamma_grad",
    "igamma",
    "igammac",
    "xlogy",
})

GENERATED_MULTI_OUTPUT_REDUCTION_BASES = frozenset({
    "_aminmax",
    "_fake_quantize_per_tensor_affine_cachemask_tensor_qparams",
    "_fake_quantize_learnable_per_channel_affine",
    "_fake_quantize_learnable_per_tensor_affine",
    "_embedding_bag",
    "_embedding_bag_forward_only",
    "_unique",
    "_unique2",
    "aminmax",
    "_batch_norm_impl_index",
    "_batch_norm_impl_index_backward",
    "_batch_norm_no_update",
    "_batch_norm_with_update",
    "_batch_norm_with_update_functional",
    "_ctc_loss",
    "_ctc_loss_backward",
    "_native_batch_norm_legit_no_training",
    "_native_batch_norm_legit",
    "_native_batch_norm_legit_functional",
    "batch_norm_backward",
    "batch_norm_update_stats",
    "cummax",
    "cummin",
    "embedding_bag",
    "fake_quantize_per_channel_affine_cachemask",
    "fake_quantize_per_tensor_affine",
    "fake_quantize_per_tensor_affine_cachemask",
    "frexp",
    "geqrf",
    "kthvalue",
    "_histogramdd_from_bin_cts",
    "_histogramdd_from_bin_tensors",
    "_histogramdd_bin_edges",
    "_linalg_det",
    "_linalg_eigh",
    "_linalg_slogdet",
    "_linalg_solve_ex",
    "histogram",
    "linalg_cholesky_ex",
    "linalg_inv_ex",
    "linalg_lu",
    "linalg_lu_factor",
    "linalg_lu_factor_ex",
    "linalg_qr",
    "linalg_slogdet",
    "linalg_solve_ex",
    "log_sigmoid_forward",
    "max",
    "median",
    "min",
    "mode",
    "multilabel_margin_loss_forward",
    "nanmedian",
    "native_batch_norm",
    "native_batch_norm_backward",
    "nll_loss_forward",
    "nll_loss2d_forward",
    "qr",
    "slogdet",
    "sort",
    "std_mean",
    "topk",
    "unique_dim",
    "unique_dim_consecutive",
    "unique_consecutive",
    "var_mean",
})

GENERATED_UPSAMPLE_BASE_MARKER = "upsample"

GENERATED_POOLING_BASES = frozenset({
    "_adaptive_avg_pool2d",
    "_adaptive_avg_pool3d",
    "adaptive_avg_pool1d",
    "adaptive_avg_pool2d",
    "adaptive_avg_pool3d",
    "adaptive_max_pool1d",
    "adaptive_max_pool2d",
    "adaptive_max_pool3d",
    "avg_pool1d",
    "avg_pool2d",
    "avg_pool3d",
    "fractional_max_pool2d",
    "fractional_max_pool3d",
    "max_pool1d_with_indices",
    "max_pool2d_with_indices",
    "max_pool3d_with_indices",
})

GENERATED_CONVOLUTION_BASES = frozenset({
    "_conv_depthwise2d",
    "_convolution",
    "_slow_conv2d_forward",
    "conv_depthwise3d",
    "col2im",
    "convolution",
    "convolution_overrideable",
    "im2col",
    "slow_conv3d",
    "slow_conv3d_forward",
    "slow_conv_dilated2d",
    "slow_conv_dilated3d",
    "thnn_conv2d",
})

GENERATED_GRID_BASES = frozenset({
    "_grid_sampler_2d_cpu_fallback",
    "affine_grid_generator",
    "grid_sampler",
    "grid_sampler_2d",
    "grid_sampler_3d",
})

GENERATED_GRID_BACKWARD_BASES = frozenset({
    "_grid_sampler_2d_cpu_fallback_backward",
    "grid_sampler_2d_backward",
    "grid_sampler_3d_backward",
})

GENERATED_FACTORY_NAMES_OUT_BASES = frozenset({
    "empty",
    "full",
    "ones",
    "zeros",
})

GENERATED_RNN_CELL_BASES = frozenset({
    "gru_cell",
    "lstm_cell",
    "rnn_relu_cell",
    "rnn_tanh_cell",
})

GENERATED_FFT_BASES = frozenset({
    "_fft_c2c",
    "_fft_c2r",
    "_fft_r2c",
    "fft_fft",
    "fft_fft2",
    "fft_fftn",
    "fft_hfft",
    "fft_hfft2",
    "fft_hfftn",
    "fft_ifft",
    "fft_ifft2",
    "fft_ifftn",
    "fft_ihfft",
    "fft_ihfft2",
    "fft_ihfftn",
    "fft_irfft",
    "fft_irfft2",
    "fft_irfftn",
    "fft_rfft",
    "fft_rfft2",
    "fft_rfftn",
})

GENERATED_LOSS_BASES = frozenset({
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "cosine_embedding_loss",
    "cross_entropy_loss",
    "ctc_loss",
    "hinge_embedding_loss",
    "huber_loss",
    "l1_loss",
    "margin_ranking_loss",
    "mse_loss",
    "multi_margin_loss",
    "multilabel_margin_loss",
    "nll_loss",
    "nll_loss2d",
    "nll_loss_nd",
    "kl_div",
    "smooth_l1_loss",
    "soft_margin_loss",
    "triplet_margin_loss",
})

GENERATED_PADDING_BASES = frozenset({
    "_pad_circular",
    "_pad_enum",
    "constant_pad_nd",
    "pad",
    "reflection_pad1d",
    "reflection_pad2d",
    "reflection_pad3d",
    "replication_pad1d",
    "replication_pad2d",
    "replication_pad3d",
})

GENERATED_LINALG_BASES = frozenset({
    "_cdist_forward",
    "_cholesky_solve_helper",
    "_addmm_activation",
    "_euclidean_dist",
    "_pdist_forward",
    "addmv",
    "addr",
    "cholesky",
    "cholesky_inverse",
    "cholesky_solve",
    "cross",
    "dist",
    "dot",
    "ger",
    "inner",
    "inverse",
    "kron",
    "linalg_cholesky",
    "linalg_cond",
    "linalg_cross",
    "linalg_det",
    "linalg_inv",
    "linalg_matmul",
    "linalg_matrix_exp",
    "linalg_matrix_norm",
    "linalg_matrix_power",
    "linalg_matrix_rank",
    "linalg_norm",
    "linalg_pinv",
    "linalg_solve",
    "linalg_solve_triangular",
    "linalg_svdvals",
    "linalg_tensorinv",
    "linalg_tensorsolve",
    "linalg_vecdot",
    "linalg_vector_norm",
    "linalg__powsum",
    "matrix_power",
    "mv",
    "native_norm",
    "nuclear_norm",
    "orgqr",
    "ormqr",
    "norm_except_dim",
    "outer",
    "pairwise_distance",
    "pdist",
    "tensordot",
    "tril",
    "triu",
    "vander",
    "vdot",
})

GENERATED_FACTORY_OUT_BASES = frozenset({
    "arange",
    "bartlett_window",
    "blackman_window",
    "empty",
    "empty_like",
    "empty_permuted",
    "empty_strided",
    "eye",
    "fft_fftfreq",
    "fft_rfftfreq",
    "full",
    "full_like",
    "hamming_window",
    "hann_window",
    "kaiser_window",
    "linspace",
    "logspace",
    "new_empty",
    "new_empty_strided",
    "new_full",
    "new_ones",
    "new_zeros",
    "ones",
    "ones_like",
    "range",
    "scalar_tensor",
    "zeros",
    "zeros_like",
})

GENERATED_MATMUL_SURFACES = frozenset({
    "aten::addbmm.out",
    "aten::addbmm_",
    "aten::addmm.out",
    "aten::addmm_",
    "aten::baddbmm.out",
    "aten::baddbmm_",
    "aten::bmm.out",
    "aten::chain_matmul",
    "aten::chain_matmul.out",
    "aten::linear.out",
    "aten::matmul.out",
    "aten::mm.out",
})


@dataclass
class CoverageMarker:
    nodeid: str
    path: str
    name: str
    covers: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    surfaces: dict[str, str] = field(default_factory=dict)
    semantic_level: int | None = None
    level_reason: str | None = None
    level_source: str | None = None
    generated: bool = False
    source: str = "marker"


HANDWRITTEN_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("torchcts/autograd/", ("autograd_behavior",)),
    ("torchcts/compiler/", ("compile_behavior",)),
    ("torchcts/device_api/", ("metadata_device", "device_api_behavior")),
    ("torchcts/dtypes/test_complex.py", ("dtype_complex",)),
    ("torchcts/dtypes/test_copy_cast.py", ("dtype_copy_cast",)),
    ("torchcts/dtypes/test_fp8.py", ("dtype_fp8",)),
    ("torchcts/dtypes/test_native_quantization.py", ("quantized_native",)),
    ("torchcts/dtypes/test_quantized.py", ("quantized_container",)),
    ("torchcts/errors/", ("error_behavior",)),
    ("torchcts/memory/test_allocator.py", ("allocator",)),
    ("torchcts/memory/test_determinism.py", ("deterministic_memory",)),
    ("torchcts/memory/test_guard_alloc.py", ("guard_alloc",)),
    ("torchcts/multi_device/", ("metadata_device", "multi_device_behavior")),
    ("torchcts/operators/test_creation.py", ("factory",)),
    ("torchcts/operators/test_foreach.py", ("foreach",)),
    ("torchcts/operators/test_nested.py", ("layout_storage", "nested")),
    ("torchcts/operators/test_sparse.py", ("layout_storage", "sparse")),
    ("torchcts/operators/test_view_shape.py", ("view_or_alias",)),
    ("torchcts/operators/", ("handwritten_operator_suite",)),
    ("torchcts/rng/", ("rng",)),
    ("torchcts/serialization/", ("serialization",)),
    ("torchcts/stress/", ("stress",)),
    ("torchcts/strides/", ("layout_storage", "stride_behavior")),
    ("torchcts/training/", ("training_workflow",)),
    ("torchcts/workloads/", ("workload",)),
)


def coverage_categories_for_path(path: str | os.PathLike) -> list[str]:
    normalized = str(path).replace("\\", "/")
    if normalized.startswith("/"):
        try:
            normalized = Path(normalized).resolve().relative_to(PACKAGE_ROOT.parent).as_posix()
        except Exception:
            pass
    for prefix, categories in HANDWRITTEN_CATEGORY_RULES:
        if normalized.startswith(prefix):
            return list(categories)
    return []


def _split_dispatcher_name(name: str) -> tuple[str, str]:
    if not name.startswith("aten::"):
        raise ValueError(f"Expected aten dispatcher name, got {name!r}")
    suffix = name.split("::", 1)[1]
    parts = suffix.split(".")
    base = parts[0]
    overload = ".".join(parts[1:]) if len(parts) > 1 else ""
    return base, overload


def _schema_for(name: str):
    base, overload = _split_dispatcher_name(name)
    return torch._C._dispatch_find_schema_or_throw(f"aten::{base}", overload).schema()


def _is_tensorish_type(value) -> bool:
    return "Tensor" in str(value)


def _alias_info_dict(alias_info) -> dict | None:
    if alias_info is None:
        return None
    return {
        "is_write": bool(getattr(alias_info, "is_write", False)),
        "repr": repr(alias_info),
    }


def _schema_arg_record(arg) -> dict:
    return {
        "name": arg.name,
        "type": str(arg.type),
        "tensor": _is_tensorish_type(arg.type),
        "alias": _alias_info_dict(getattr(arg, "alias_info", None)),
        "kwarg_only": bool(getattr(arg, "kwarg_only", False)),
        "is_out": bool(getattr(arg, "is_out", False)),
    }


def _schema_return_record(ret) -> dict:
    return {
        "name": ret.name,
        "type": str(ret.type),
        "tensor": _is_tensorish_type(ret.type),
        "alias": _alias_info_dict(getattr(ret, "alias_info", None)),
    }


def _has_write_alias(arg_records: list[dict]) -> bool:
    for arg in arg_records:
        alias = arg.get("alias")
        if alias and alias.get("is_write"):
            return True
    return False


def _has_return_alias(return_records: list[dict]) -> bool:
    return any(ret.get("alias") for ret in return_records)


def _dispatch_registration_map(name: str) -> dict[str, bool]:
    registrations = {}
    for key in DISPATCH_KEYS:
        try:
            registrations[key] = bool(torch._C._dispatch_has_kernel_for_dispatch_key(name, key))
        except Exception:
            registrations[key] = False
    return registrations


def classify_surface(name: str, schema=None) -> tuple[str, str]:
    if schema is None:
        schema = _schema_for(name)
    base, overload = _split_dispatcher_name(name)
    suffix = name.split("::", 1)[1]
    arg_records = [_schema_arg_record(arg) for arg in schema.arguments]
    return_records = [_schema_return_record(ret) for ret in schema.returns]
    tensor_args = [arg for arg in arg_records if arg["tensor"]]
    tensor_returns = [ret for ret in return_records if ret["tensor"]]
    out_variant = suffix.endswith(".out") or any(arg["name"] == "out" for arg in arg_records)
    mutating = suffix.endswith("_") or _has_write_alias(arg_records)
    view_alias = bool(tensor_returns and _has_return_alias(return_records))
    factory = bool(tensor_returns and not tensor_args)

    if not tensor_args and not tensor_returns:
        if base in CURATED_METADATA_BASES:
            return "metadata_device", "metadata"
        return "not_backend_relevant", "metadata"

    variant_kind = "functional"
    if out_variant:
        variant_kind = "out"
    elif mutating:
        variant_kind = "inplace"
    elif view_alias:
        variant_kind = "view"
    elif factory:
        variant_kind = "factory"

    if out_variant:
        return "out_variant", variant_kind
    if mutating:
        return "mutating_or_inplace", variant_kind
    if base.startswith(RNG_BASE_PATTERNS) or any(part in base for part in ("dropout", "bernoulli")):
        return "rng", variant_kind
    if "backward" in base or base.endswith("_backward_data") or base == "_backward":
        return "autograd_backward", variant_kind
    if factory and base in {"empty_permuted"}:
        return "factory", variant_kind
    if any(part in base for part in LAYOUT_BASE_PATTERNS):
        return "layout_storage", variant_kind
    if view_alias or any(part in base for part in VIEW_BASE_PATTERNS):
        return "view_or_alias", variant_kind
    if factory:
        return "factory", variant_kind
    return "functional_data", variant_kind


def build_dispatcher_inventory() -> dict:
    entries = []
    for name in sorted(op for op in torch._C._dispatch_get_all_op_names() if op.startswith("aten::")):
        schema = _schema_for(name)
        base, overload = _split_dispatcher_name(name)
        args = [_schema_arg_record(arg) for arg in schema.arguments]
        returns = [_schema_return_record(ret) for ret in schema.returns]
        tensor_args = [arg for arg in args if arg["tensor"]]
        tensor_returns = [ret for ret in returns if ret["tensor"]]
        surface_kind, variant_kind = classify_surface(name, schema)
        entries.append({
            "name": name,
            "base_name": base,
            "overload": overload,
            "schema": str(schema),
            "args": args,
            "returns": returns,
            "tensor_args": tensor_args,
            "tensor_returns": tensor_returns,
            "has_tensor_args": bool(tensor_args),
            "has_tensor_returns": bool(tensor_returns),
            "surface_kind": surface_kind,
            "variant_kind": variant_kind,
            "dispatch": _dispatch_registration_map(name),
        })

    counts = Counter(entry["surface_kind"] for entry in entries)
    return {
        "metadata": {
            "pytorch_version": torch.__version__,
            "generated_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            "total_aten_overloads": len(entries),
            "surface_counts": dict(sorted(counts.items())),
        },
        "entries": entries,
    }


def _normalize_op_name(value) -> str | None:
    if not value:
        return None
    text = str(value)
    if text.startswith("aten::"):
        text = text.split("::", 1)[1]
    return text


def _alias_names_for_opinfo(op) -> set[str]:
    names = set()
    for alias in getattr(op, "aliases", ()) or ():
        for attr in ("name", "aten_name", "op"):
            value = getattr(alias, attr, None)
            normalized = _normalize_op_name(value)
            if normalized:
                names.add(normalized)
    return names


def _alias_override_names_for_opinfo(op) -> set[str]:
    return set(OPINFO_ALIAS_OVERRIDES.get(getattr(op, "name", None), ()))


def build_opinfo_map() -> dict:
    import torch.testing._internal.common_methods_invocations as cmi

    bases = defaultdict(list)
    exact = defaultdict(list)
    supports_out = {}
    for op in cmi.op_db:
        op_supports_out = bool(getattr(op, "supports_out", False))
        supports_out[op.name] = op_supports_out
        raw_names = {
            op.name,
            getattr(op, "aten_name", None),
            getattr(op, "decomp_aten_name", None),
            getattr(op, "aten_backward_name", None),
        }
        raw_names.update(_alias_names_for_opinfo(op))
        raw_names.update(_alias_override_names_for_opinfo(op))
        for raw_name in raw_names:
            normalized = _normalize_op_name(raw_name)
            if not normalized:
                continue
            base = normalized.split(".", 1)[0]
            bases[base].append(op.name)
            exact[normalized].append(op.name)
            supports_out.setdefault(base, op_supports_out)
    return {
        "bases": {key: sorted(set(value)) for key, value in bases.items()},
        "exact": {key: sorted(set(value)) for key, value in exact.items()},
        "supports_out": supports_out,
    }


def _attribute_path(node) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attribute_path(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    return None


def _literal_string(node) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _literal_int(node) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    return None


def _marker_from_decorator(decorator) -> tuple[str, list, dict[str, str], dict[str, object]] | None:
    if not isinstance(decorator, ast.Call):
        return None
    path = _attribute_path(decorator.func)
    if path not in (
        "pytest.mark.covers",
        "pytest.mark.covers_category",
        "pytest.mark.requires",
        "pytest.mark.semantic_level",
    ):
        return None
    values = []
    surfaces = {}
    metadata: dict[str, object] = {}
    for arg in decorator.args:
        if path == "pytest.mark.semantic_level":
            value = _literal_int(arg)
        else:
            value = _literal_string(arg)
        if value is not None:
            values.append(value)
    for keyword in decorator.keywords:
        if keyword.arg == "surface":
            value = _literal_string(keyword.value)
            if value:
                for covered in values:
                    surfaces[covered] = value
        elif path == "pytest.mark.semantic_level" and keyword.arg in {"level", "reason"}:
            if keyword.arg == "level":
                value = _literal_int(keyword.value)
            else:
                value = _literal_string(keyword.value)
            if value is not None:
                metadata[keyword.arg] = value
    return path.rsplit(".", 1)[-1], values, surfaces, metadata


def _markers_from_pytestmark(node) -> list[tuple[str, list, dict[str, str], dict[str, object]]]:
    if isinstance(node, ast.Call):
        marker = _marker_from_decorator(node)
        return [marker] if marker else []
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        markers = []
        for item in node.elts:
            markers.extend(_markers_from_pytestmark(item))
        return markers
    return []


def _module_pytestmark_markers(tree) -> list[tuple[str, list, dict[str, str], dict[str, object]]]:
    markers = []
    for node in tree.body:
        value = None
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets):
                value = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "pytestmark":
                value = node.value
        if value is not None:
            markers.extend(_markers_from_pytestmark(value))
    return markers


def _merge_markers(markers) -> tuple[list[str], list[str], list[str], dict[str, str], dict[str, object] | None, list[str]]:
    covers = []
    categories = []
    capabilities = []
    surfaces = {}
    semantic = None
    errors = []
    for marker in markers:
        if not marker:
            continue
        marker_name, values, marker_surfaces, marker_metadata = marker
        if marker_name == "covers":
            covers.extend(values)
            surfaces.update(marker_surfaces)
        elif marker_name == "covers_category":
            categories.extend(values)
        elif marker_name == "requires":
            capabilities.extend(values)
        elif marker_name == "semantic_level":
            try:
                level = marker_value_to_level(tuple(values), marker_metadata)
            except Exception as exc:
                errors.append(f"invalid semantic_level marker: {exc}")
                continue
            semantic = {
                "semantic_level": level,
                "level_reason": marker_metadata.get("reason") or "Declared by pytest semantic_level marker.",
                "level_source": "test_marker",
            }
    return covers, categories, capabilities, surfaces, semantic, errors


def _iter_test_functions_with_markers(tree, inherited_markers=None):
    inherited_markers = list(inherited_markers or [])
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                markers = inherited_markers + [
                    marker for marker in (_marker_from_decorator(decorator) for decorator in node.decorator_list)
                    if marker
                ]
                yield node, markers
        elif isinstance(node, ast.ClassDef):
            class_markers = inherited_markers + [
                marker for marker in (_marker_from_decorator(decorator) for decorator in node.decorator_list)
                if marker
            ]
            yield from _iter_test_functions_with_markers(node, class_markers)


def collect_coverage_markers(root: str | os.PathLike | None = None) -> dict:
    root_path = Path(root) if root is not None else PACKAGE_ROOT
    markers: list[CoverageMarker] = []
    unmapped_tests: list[dict] = []
    errors: list[str] = []

    for path in sorted(root_path.glob("**/test_*.py")):
        rel = path.relative_to(root_path.parent).as_posix() if path.is_absolute() else path.as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception as exc:
            unmapped_tests.append({"nodeid": rel, "reason": f"parse_error: {type(exc).__name__}: {exc}"})
            continue

        module_markers = _module_pytestmark_markers(tree)
        for node, node_markers in _iter_test_functions_with_markers(tree, module_markers):
            covers, categories, capabilities, surfaces, semantic, marker_errors = _merge_markers(node_markers)
            path_categories = coverage_categories_for_path(rel)

            nodeid = f"{rel}::{node.name}"
            generated = "/generated/" in f"/{rel}"
            marker_source = "marker"
            explicit_marker = bool(covers or categories)
            if marker_errors:
                errors.extend(f"{nodeid}: {error}" for error in marker_errors)
            if not explicit_marker and path_categories:
                categories.extend(path_categories)
                marker_source = "path_rule"
            elif path_categories:
                categories.extend(path_categories)
                marker_source = "marker+path_rule"

            if semantic is None:
                default = suite_default_level(suite_for_path(rel))
                semantic = {
                    "semantic_level": default.level,
                    "level_reason": default.reason,
                    "level_source": default.source,
                }

            if covers or categories:
                markers.append(CoverageMarker(
                    nodeid=nodeid,
                    path=rel,
                    name=node.name,
                    covers=sorted(set(covers)),
                    categories=sorted(set(categories)),
                    capabilities=sorted(set(capabilities)),
                    surfaces=surfaces,
                    semantic_level=semantic["semantic_level"],
                    level_reason=semantic["level_reason"],
                    level_source=semantic["level_source"],
                    generated=generated,
                    source=marker_source,
                ))
            elif "/opinfo/" not in f"/{rel}" and "/selftest/" not in f"/{rel}":
                unmapped_tests.append({
                    "nodeid": nodeid,
                    "reason": "missing @pytest.mark.covers marker",
                    "semantic_level": semantic["semantic_level"],
                    "level_reason": semantic["level_reason"],
                    "level_source": semantic["level_source"],
                })

    return {
        "markers": [marker.__dict__ for marker in markers],
        "unmapped_tests": unmapped_tests,
        "errors": errors,
    }


def _load_json_file(path: Path) -> tuple[dict | None, str | None]:
    if not path.exists():
        return None, None
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"{path}: failed to parse JSON: {type(exc).__name__}: {exc}"


def load_exclusions(
    inventory: dict,
    package_path: Path | None = None,
    project_path: Path | None = None,
) -> dict:
    package_path = package_path or PACKAGE_EXCLUSIONS_PATH
    project_path = project_path or DEFAULT_PROJECT_EXCLUSIONS
    live_names = {entry["name"] for entry in inventory["entries"]}
    runtime_unavailable_entries = runtime_unavailable_op_entries(
        runtime_version=torch.__version__,
        live_names=live_names,
    )
    names = live_names | {entry["name"] for entry in runtime_unavailable_entries}
    bases = (
        {entry["base_name"] for entry in inventory["entries"]}
        | {entry["base_name"] for entry in runtime_unavailable_entries}
    )
    exclusions = []
    errors = []
    warnings = []

    for source, path in (("package", package_path), ("project", project_path)):
        data, error = _load_json_file(path)
        if error:
            errors.append(error)
            continue
        if data is None:
            continue
        if data.get("version") != 1:
            errors.append(f"{path}: coverage exclusion file version must be 1")
            continue
        raw_exclusions = data.get("exclusions")
        if not isinstance(raw_exclusions, list):
            errors.append(f"{path}: 'exclusions' must be a list")
            continue
        for index, exclusion in enumerate(raw_exclusions):
            exclusion = dict(exclusion)
            exclusion["_source"] = source
            exclusion["_source_path"] = str(path)
            exclusion["_index"] = index
            exclusions.append(exclusion)

    today = _datetime.date.today()
    required = {"name", "match", "surface", "category", "reason", "owner", "review_after"}
    for exclusion in exclusions:
        label = f"{exclusion.get('_source_path')}[{exclusion.get('_index')}]"
        missing = sorted(required - set(exclusion))
        if missing:
            errors.append(f"{label}: missing required exclusion field(s): {', '.join(missing)}")
            continue
        if exclusion["match"] not in EXCLUSION_MATCH_TYPES:
            errors.append(f"{label}: invalid match type {exclusion['match']!r}")
        if exclusion["surface"] not in SURFACE_KINDS:
            errors.append(f"{label}: invalid surface {exclusion['surface']!r}")
        if exclusion["category"] not in EXCLUSION_CATEGORIES:
            errors.append(f"{label}: invalid category {exclusion['category']!r}")
        if not isinstance(exclusion["reason"], str) or not exclusion["reason"].strip():
            errors.append(f"{label}: reason must be a non-empty string")
        try:
            review_after = _datetime.date.fromisoformat(exclusion["review_after"])
            if review_after < today:
                warnings.append(f"{label}: exclusion review date has expired: {exclusion['review_after']}")
        except Exception:
            errors.append(f"{label}: review_after must be YYYY-MM-DD")

        match_type = exclusion.get("match")
        name = exclusion.get("name")
        if match_type == "exact" and name not in names:
            errors.append(f"{label}: exact exclusion name {name!r} is not a known dispatcher overload")
        elif match_type == "base" and name.replace("aten::", "") not in bases:
            errors.append(f"{label}: base exclusion name {name!r} is not a known dispatcher base name")
        elif match_type == "regex":
            try:
                pattern = re.compile(name)
            except re.error as exc:
                errors.append(f"{label}: invalid exclusion regex {name!r}: {exc}")
                continue
            if not any(pattern.search(candidate) for candidate in names):
                errors.append(f"{label}: exclusion regex {name!r} did not match any dispatcher overload")
            if "regex" not in exclusion.get("reason", "").lower():
                errors.append(f"{label}: regex exclusion reason should explain why exact names are impractical")

    return {
        "exclusions": exclusions,
        "errors": errors,
        "warnings": warnings,
    }


def _matches_exclusion(entry: dict, exclusion: dict) -> bool:
    match_type = exclusion.get("match")
    name = exclusion.get("name")
    if match_type == "exact":
        return entry["name"] == name
    if match_type == "base":
        return entry["base_name"] == name.replace("aten::", "")
    if match_type == "regex":
        return bool(re.search(name, entry["name"]))
    return False


def _is_named_tensor_surface(name: str) -> bool:
    return bool(re.search(r"(?:dimname|names|named|align_tensors)", name, re.IGNORECASE))


def _is_vendor_or_backend_pack_surface(name: str) -> bool:
    return bool(re.search(
        r"(cudnn|miopen|mkldnn|mps|_triton|_cslt|nnpack|fbgemm|_propagate_xla_data)",
        name,
        re.IGNORECASE,
    ))


def _is_property_surface(name: str) -> bool:
    return bool(re.search(
        r"(autocast|pin_memory|record_stream|flash_attention|efficient_attention|"
        r"scaled_dot_product|fused_sdp|unsafe|_fw_primal|_make_dual|dropout)",
        name,
        re.IGNORECASE,
    ))


def _is_host_storage_surface(name: str) -> bool:
    return name in {"aten::from_file", "aten::from_file.out"}


def _status_from_exclusion(entry: dict, exclusion: dict | None) -> tuple[str, dict | None]:
    spec = oracle_spec_for(entry["name"])
    if spec is not None:
        return spec.coverage_status, spec.metadata()

    if exclusion is None:
        return "unknown", None

    name = entry["name"]
    category = exclusion.get("category")

    if _is_host_storage_surface(name):
        return "excluded_host_storage", None
    if _is_named_tensor_surface(name):
        return "excluded_unsupported_public_api", None
    if category == "dispatcher_plumbing":
        return "excluded_framework_plumbing", None
    if category == "distributed_or_c10d":
        return "excluded_distributed_scope", None
    if category == "deprecated_or_removed":
        return "excluded_deprecated_or_removed", None
    if category == "manual_future_scope":
        return "excluded_unsupported_public_api", None
    if category == "covered_by_public_surface":
        return "pending_property", None
    if _is_vendor_or_backend_pack_surface(name) or category == "backend_specific_internal":
        return "pending_backend_pack", None
    if _is_property_surface(name) or category == "unsafe_direct_invocation":
        return "pending_property", None
    if category == "cpu_reference_invalid":
        dispatch = entry.get("dispatch") or {}
        if any(dispatch.get(key) for key in ("MPS", "CUDA", "PrivateUse1")):
            return "pending_backend_pack", None
        if not dispatch.get("CPU"):
            return "pending_property", None
        return "pending_oracle", None
    return "excluded", None


def _coverage_kind_for_status(status: str) -> str:
    if status in {"covered_oracle", "pending_oracle"}:
        return "oracle"
    if status in {"covered_backend_pack", "pending_backend_pack"}:
        return "backend_pack"
    if status in {"covered_property", "pending_property"}:
        return "property"
    if status in EXCLUDED_STATUSES:
        return "excluded"
    if status == "covered_generated":
        return "generated"
    if status == "covered_handwritten":
        return "handwritten"
    if status == "covered_opinfo":
        return "opinfo"
    if status == "not_backend_relevant":
        return "not_backend_relevant"
    if status in RUNTIME_UNAVAILABLE_STATUSES:
        return "runtime_unavailable"
    return "unknown"


def _backend_gate_for_entry(entry: dict, exclusion: dict | None, oracle_payload: dict | None) -> str:
    if oracle_payload and oracle_payload.get("backend_gate"):
        return str(oracle_payload["backend_gate"])

    name = entry.get("name", "").lower()
    category = (exclusion or {}).get("category")
    if any(token in name for token in ("cudnn", "cusparse", "cusolver", "_cslt", "_triton")):
        return "cuda"
    if "miopen" in name:
        return "rocm"
    if "mps" in name:
        return "mps"
    if "xla" in name:
        return "xla"
    if "fbgemm" in name:
        return "fbgemm"
    if "mkldnn" in name or "nnpack" in name:
        return "cpu_build"
    if "quantized" in name and category == "unsafe_direct_invocation":
        return "quantized"

    dispatch = entry.get("dispatch") or {}
    for key, gate in (
        ("CUDA", "cuda"),
        ("MPS", "mps"),
        ("PrivateUse1", "privateuse1"),
        ("CPU", "cpu"),
    ):
        if dispatch.get(key):
            return gate
    return "any"


def _has_runtime_backend_registration(entry: dict) -> bool:
    dispatch = entry.get("dispatch") or {}
    return any(dispatch.get(key) for key in ("CPU", "MPS", "CUDA", "PrivateUse1"))


def _pending_review_for_entry(
    entry: dict,
    status: str,
    exclusion: dict | None,
    oracle_payload: dict | None,
) -> dict | None:
    if status not in PENDING_STATUSES | EXCLUDED_STATUSES:
        return None

    category = (exclusion or {}).get("category")
    backend_gate = _backend_gate_for_entry(entry, exclusion, oracle_payload)
    blocker_type = "out_of_backend_conformance_scope"
    required_closure = "none_excluded_from_backend_conformance"

    if status == "pending_oracle":
        blocker_type = "needs_cpu_oracle"
        required_closure = "implement_cpu_reference_oracle"
    elif status == "pending_backend_pack":
        blocker_type = "needs_backend_pack"
        required_closure = "implement_backend_gated_runner"
        if category == "cpu_reference_invalid" and backend_gate in {"any", "cpu"} and not _has_runtime_backend_registration(entry):
            blocker_type = "kernel_unavailable_in_host_build"
    elif status == "pending_property":
        blocker_type = "needs_property_runner"
        required_closure = "implement_property_runner"
        if category == "covered_by_public_surface":
            blocker_type = "needs_public_proxy_proof"
            required_closure = "prove_public_proxy_or_add_direct_runner"
        elif category == "unsafe_direct_invocation":
            blocker_type = "needs_valid_internal_inputs"
            required_closure = "construct_valid_internal_inputs_and_property_runner"
        elif category == "cpu_reference_invalid" and not _has_runtime_backend_registration(entry):
            blocker_type = "kernel_unavailable_in_host_build"
            required_closure = "validate_on_backend_build_or_keep_pending"
    elif status == "excluded_framework_plumbing":
        blocker_type = "out_of_backend_conformance_scope"
        required_closure = "none_dispatcher_plumbing"
    elif status == "excluded_deprecated_or_removed":
        blocker_type = "out_of_backend_conformance_scope"
        required_closure = "none_deprecated_or_removed"
    elif status == "excluded_distributed_scope":
        blocker_type = "out_of_backend_conformance_scope"
        required_closure = "none_distributed_scope"
    elif status == "excluded_host_storage":
        blocker_type = "out_of_backend_conformance_scope"
        required_closure = "none_host_storage_only"
    elif status == "excluded_unsupported_public_api":
        blocker_type = "out_of_backend_conformance_scope"
        required_closure = "none_unsupported_public_api"

    return {
        "blocker_type": blocker_type,
        "required_closure": required_closure,
        "backend_gate": backend_gate,
        "next_family": (oracle_payload or {}).get("oracle_id") or category or entry.get("surface_kind"),
        "reason": (oracle_payload or {}).get("reason") or (exclusion or {}).get("reason") or "",
        "source_category": category,
        "review_after": (exclusion or {}).get("review_after"),
        "owner": (exclusion or {}).get("owner"),
    }


def _opinfo_matches_for_entry(entry: dict, opinfo_map: dict) -> list[str]:
    base_matches = opinfo_map["bases"].get(entry["base_name"], [])
    exact_name = entry["base_name"] if not entry["overload"] else f"{entry['base_name']}.{entry['overload']}"
    exact_matches = opinfo_map["exact"].get(exact_name, [])
    inplace_base_matches = []
    if entry["base_name"].endswith("_"):
        inplace_base_matches = opinfo_map["bases"].get(entry["base_name"].rstrip("_"), [])
    return sorted(set(base_matches) | set(exact_matches) | set(inplace_base_matches))


def _opinfo_covers(entry: dict, opinfo_map: dict) -> tuple[bool, list[str]]:
    matches = _opinfo_matches_for_entry(entry, opinfo_map)
    if entry["surface_kind"] in {"out_variant", "mutating_or_inplace", "view_or_alias"}:
        return False, matches
    if entry["variant_kind"] in {"out", "inplace", "view"}:
        return False, matches
    return bool(matches), matches


def _manual_foreach_strategy(foreach_name: str, overload: str, entry: dict, *, require_tensor_return: bool) -> dict | None:
    if foreach_name in GENERATED_FOREACH_UNARY_ALLOWLIST and not overload:
        if require_tensor_return and len(entry.get("tensor_returns", [])) != 1:
            return None
        return {
            "strategy": "manual_foreach",
            "family": "unary",
            "foreach_name": foreach_name,
        }

    if foreach_name == "copy" and not overload:
        if require_tensor_return and len(entry.get("tensor_returns", [])) != 1:
            return None
        return {
            "strategy": "manual_foreach",
            "family": "copy",
            "foreach_name": foreach_name,
        }

    if foreach_name in {"norm", "powsum"} and overload in GENERATED_FOREACH_NORM_OVERLOADS:
        if require_tensor_return and len(entry.get("tensor_returns", [])) != 1:
            return None
        return {
            "strategy": "manual_foreach",
            "family": "norm",
            "foreach_name": foreach_name,
            "overload": overload,
        }

    if foreach_name in GENERATED_FOREACH_BASIC_BINARY_ALLOWLIST and overload in GENERATED_FOREACH_BASIC_BINARY_OVERLOADS:
        if require_tensor_return and len(entry.get("tensor_returns", [])) != 1:
            return None
        return {
            "strategy": "manual_foreach",
            "family": "binary",
            "foreach_name": foreach_name,
            "overload": overload,
        }

    if foreach_name in GENERATED_FOREACH_EXTREMA_ALLOWLIST and overload in GENERATED_FOREACH_EXTREMA_OVERLOADS:
        if require_tensor_return and len(entry.get("tensor_returns", [])) != 1:
            return None
        return {
            "strategy": "manual_foreach",
            "family": "extrema",
            "foreach_name": foreach_name,
            "overload": overload,
        }

    if foreach_name == "pow" and overload in GENERATED_FOREACH_POW_OVERLOADS:
        if require_tensor_return and len(entry.get("tensor_returns", [])) != 1:
            return None
        return {
            "strategy": "manual_foreach",
            "family": "pow",
            "foreach_name": foreach_name,
            "overload": overload,
        }

    if foreach_name in GENERATED_FOREACH_TERNARY_ALLOWLIST and overload in GENERATED_FOREACH_TERNARY_OVERLOADS:
        if require_tensor_return and len(entry.get("tensor_returns", [])) != 1:
            return None
        return {
            "strategy": "manual_foreach",
            "family": "ternary",
            "foreach_name": foreach_name,
            "overload": overload,
        }

    if foreach_name == "lerp" and overload in GENERATED_FOREACH_LERP_OVERLOADS:
        if require_tensor_return and len(entry.get("tensor_returns", [])) != 1:
            return None
        return {
            "strategy": "manual_foreach",
            "family": "lerp",
            "foreach_name": foreach_name,
            "overload": overload,
        }

    return None


def _is_out_arg(arg: dict) -> bool:
    return bool(arg.get("is_out")) or arg.get("name") == "out"


def _manual_bitwise_strategy(entry: dict) -> dict | None:
    base_name = entry["base_name"]
    surface_kind = entry["surface_kind"]
    overload = entry["overload"]

    if surface_kind == "out_variant":
        canonical = BITWISE_DUNDER_BASES.get(base_name, base_name)
        if canonical not in GENERATED_BITWISE_BASES:
            return None
        if overload != "out" and not overload.endswith("_out"):
            return None
        if len([arg for arg in entry.get("args", []) if arg.get("name") == "out"]) != 1:
            return None
        if len(entry.get("tensor_returns", [])) != 1:
            return None
        return {
            "strategy": "manual_bitwise",
            "family": canonical,
        }

    if surface_kind == "mutating_or_inplace":
        if base_name in BITWISE_DUNDER_BASES:
            canonical = BITWISE_DUNDER_BASES[base_name]
        elif base_name.endswith("_"):
            canonical = base_name.rstrip("_")
        else:
            return None
        if canonical not in GENERATED_BITWISE_BASES:
            return None
        if len(entry.get("tensor_returns", [])) != 1:
            return None
        return {
            "strategy": "manual_bitwise",
            "family": canonical,
        }

    return None


def _manual_special_math_strategy(entry: dict) -> dict | None:
    surface_kind = entry["surface_kind"]
    if surface_kind not in {"functional_data", "out_variant", "mutating_or_inplace"}:
        return None

    base_name = entry["base_name"]
    canonical = base_name.rstrip("_")
    if not (canonical.startswith("special_") or canonical in GENERATED_SPECIAL_MATH_BASES):
        return None
    if len(entry.get("tensor_returns", [])) != 1:
        return None

    if surface_kind == "out_variant":
        out_args = [arg for arg in entry.get("args", []) if arg.get("name") == "out"]
        if len(out_args) != 1:
            return None
    elif surface_kind == "mutating_or_inplace":
        if not base_name.endswith("_"):
            return None

    return {
        "strategy": "manual_special_math",
        "family": canonical,
    }


def _manual_elementwise_strategy(entry: dict) -> dict | None:
    surface_kind = entry["surface_kind"]
    if surface_kind not in {"functional_data", "out_variant", "mutating_or_inplace"}:
        return None

    base_name = entry["base_name"]
    canonical = base_name.rstrip("_")
    if canonical not in GENERATED_ELEMENTWISE_BASES:
        return None
    if len(entry.get("tensor_returns", [])) != 1:
        return None

    args = entry.get("args", [])
    if surface_kind == "out_variant":
        out_args = [arg for arg in args if _is_out_arg(arg)]
        if len(out_args) != 1:
            return None
    elif surface_kind == "mutating_or_inplace":
        if not base_name.endswith("_"):
            return None

    safe_types = {
        "Tensor",
        "Tensor?",
        "number",
        "Optional[number]",
        "Optional[float]",
        "int",
        "float",
        "bool",
        "str",
        "str?",
        "Optional[str]",
    }
    for arg in args:
        if arg.get("name") == "out":
            continue
        if arg.get("tensor"):
            continue
        if arg.get("type") not in safe_types:
            return None

    return {
        "strategy": "manual_elementwise",
        "family": canonical,
    }


def _manual_reduction_strategy(entry: dict) -> dict | None:
    surface_kind = entry["surface_kind"]
    if surface_kind not in {"functional_data", "out_variant", "mutating_or_inplace", "autograd_backward"}:
        return None
    canonical = entry["base_name"].rstrip("_")
    if canonical not in GENERATED_REDUCTION_BASES:
        return None
    if len(entry.get("tensor_returns", [])) != 1:
        return None
    overload = entry.get("overload", "")
    if "Dimname" in overload or "dimname" in overload:
        return None

    args = entry.get("args", [])
    if surface_kind == "out_variant":
        out_args = [arg for arg in args if _is_out_arg(arg)]
        if len(out_args) != 1:
            return None
    elif surface_kind == "mutating_or_inplace":
        if not entry["base_name"].endswith("_"):
            return None

    safe_types = {
        "Tensor",
        "bool",
        "float",
        "int",
        "List[int]",
        "number",
        "Optional[int]",
        "Optional[List[int]]",
        "Optional[number]",
        "Optional[str]",
        "ScalarType",
        "str",
    }
    for arg in args:
        if _is_out_arg(arg):
            continue
        if arg.get("tensor"):
            continue
        if arg.get("name") == "dtype":
            continue
        if arg.get("type") not in safe_types:
            return None

    return {
        "strategy": "manual_reduction",
        "family": canonical,
    }


def _manual_indexing_strategy(entry: dict) -> dict | None:
    surface_kind = entry["surface_kind"]
    if surface_kind not in {"functional_data", "out_variant", "mutating_or_inplace", "view_or_alias"}:
        return None

    canonical = entry["base_name"].rstrip("_")
    if canonical not in GENERATED_INDEXING_BASES:
        return None
    if len(entry.get("tensor_returns", [])) != 1:
        return None

    # Named-tensor overloads need named sample inputs and backend-specific
    # named-tensor support checks. Keep them unknown until that strategy exists.
    overload = entry.get("overload", "")
    if "Dimname" in overload or "dimname" in overload:
        return None

    args = entry.get("args", [])
    if surface_kind == "out_variant":
        out_args = [arg for arg in args if arg.get("name") == "out"]
        if len(out_args) != 1:
            return None
    elif surface_kind == "mutating_or_inplace":
        if not entry["base_name"].endswith("_"):
            return None

    safe_types = {
        "Tensor",
        "bool",
        "float",
        "int",
        "List[int]",
        "number",
        "Optional[int]",
        "Optional[str]",
        "str",
        "SymInt",
        "SymInt?",
    }
    for arg in args:
        if arg.get("name") == "out":
            continue
        if arg.get("tensor"):
            continue
        if arg.get("name") == "dim" and arg.get("type") == "str":
            return None
        if arg.get("type") not in safe_types:
            return None

    return {
        "strategy": "manual_indexing",
        "family": canonical,
    }


def _manual_rng_strategy(entry: dict) -> dict | None:
    surface_kind = entry["surface_kind"]
    if surface_kind not in {"functional_data", "rng", "out_variant", "mutating_or_inplace"}:
        return None

    canonical = entry["base_name"].rstrip("_")
    if canonical not in GENERATED_RNG_BASES:
        return None
    if len(entry.get("tensor_returns", [])) != 1:
        return None

    args = entry.get("args", [])
    if any(arg.get("name") == "names" for arg in args):
        return None
    if surface_kind == "out_variant":
        out_args = [arg for arg in args if arg.get("name") == "out"]
        if len(out_args) != 1:
            return None
    elif surface_kind == "mutating_or_inplace":
        if not entry["base_name"].endswith("_"):
            return None

    safe_types = {
        "Tensor",
        "bool",
        "float",
        "int",
        "List[int]",
        "number",
        "Optional[Generator]",
        "Optional[int]",
        "Optional[List[str]]",
    }
    for arg in args:
        if arg.get("name") == "out":
            continue
        if arg.get("tensor"):
            continue
        if arg.get("type") not in safe_types:
            return None

    return {
        "strategy": "manual_rng",
        "family": canonical,
    }


def _manual_multi_output_reduction_strategy(entry: dict) -> dict | None:
    surface_kind = entry["surface_kind"]
    if surface_kind not in {"functional_data", "out_variant", "mutating_or_inplace", "autograd_backward"}:
        return None

    canonical = entry["base_name"].rstrip("_")
    if canonical not in GENERATED_MULTI_OUTPUT_REDUCTION_BASES:
        return None
    if canonical == "fake_quantize_per_tensor_affine" and entry.get("overload") != "tensor_qparams":
        return None
    if canonical == "histogram" and entry["surface_kind"] != "mutating_or_inplace":
        return None
    if canonical in {"cummax", "cummin"} and entry.get("overload", "").startswith("dimname"):
        return None
    if canonical == "_histogramdd_bin_edges" and entry["surface_kind"] == "out_variant":
        return None
    if len(entry.get("tensor_returns", [])) not in {1, 2, 3, 4, 5, 6}:
        return None

    args = entry.get("args", [])
    if any(arg.get("type") == "List[Tensor]" for arg in args) and "histogram" not in canonical:
        return None

    safe_types = {
        "Tensor",
        "bool",
        "float",
        "int",
        "List[int]",
        "List[bool]",
        "Optional[bool]",
        "Optional[int]",
        "Optional[List[float]]",
        "Optional[List[int]]",
        "Optional[number]",
        "Optional[Tensor]",
        "str",
    }
    for arg in args:
        if arg.get("is_out"):
            continue
        if arg.get("tensor"):
            continue
        if arg.get("type") not in safe_types:
            return None

    return {
        "strategy": "manual_multi_output_reduction",
        "family": canonical,
    }


def _manual_upsample_strategy(entry: dict) -> dict | None:
    surface_kind = entry["surface_kind"]
    if surface_kind not in {"functional_data", "out_variant"}:
        return None

    base_name = entry["base_name"]
    if GENERATED_UPSAMPLE_BASE_MARKER not in base_name or "backward" in base_name:
        return None
    if len(entry.get("tensor_returns", [])) != 1:
        return None

    args = entry.get("args", [])
    if surface_kind == "out_variant":
        out_args = [arg for arg in args if arg.get("is_out")]
        if len(out_args) != 1:
            return None

    safe_types = {
        "Tensor",
        "bool",
        "List[int]",
        "Optional[float]",
        "Optional[List[float]]",
        "Optional[List[int]]",
    }
    for arg in args:
        if arg.get("is_out"):
            continue
        if arg.get("tensor"):
            continue
        if arg.get("type") not in safe_types:
            return None

    return {
        "strategy": "manual_upsample",
        "family": base_name,
    }


def _manual_pooling_strategy(entry: dict) -> dict | None:
    surface_kind = entry["surface_kind"]
    if surface_kind not in {"functional_data", "out_variant", "layout_storage", "mutating_or_inplace"}:
        return None

    base_name = entry["base_name"]
    if base_name not in GENERATED_POOLING_BASES:
        return None
    if "backward" in base_name or "quantized" in base_name or "mkldnn" in base_name:
        return None
    if len(entry.get("tensor_returns", [])) not in {1, 2}:
        return None

    args = entry.get("args", [])
    safe_types = {
        "Tensor",
        "bool",
        "List[int]",
        "Optional[int]",
    }
    for arg in args:
        if arg.get("is_out"):
            continue
        if arg.get("tensor"):
            continue
        if arg.get("type") not in safe_types:
            return None

    return {
        "strategy": "manual_pooling",
        "family": base_name,
    }


def _manual_convolution_strategy(entry: dict) -> dict | None:
    surface_kind = entry["surface_kind"]
    if surface_kind not in {"functional_data", "out_variant", "mutating_or_inplace"}:
        return None

    base_name = entry["base_name"]
    if base_name not in GENERATED_CONVOLUTION_BASES:
        return None
    if "backward" in base_name or "transpose" in base_name:
        return None
    if len(entry.get("tensor_returns", [])) != 1:
        return None

    safe_types = {
        "Tensor",
        "bool",
        "int",
        "List[int]",
    }
    for arg in entry.get("args", []):
        if arg.get("is_out"):
            continue
        if arg.get("tensor"):
            continue
        if arg.get("type") not in safe_types:
            return None

    return {
        "strategy": "manual_convolution",
        "family": base_name,
    }


def _manual_grid_strategy(entry: dict) -> dict | None:
    surface_kind = entry["surface_kind"]
    if surface_kind not in {"functional_data", "out_variant"}:
        return None

    base_name = entry["base_name"]
    if base_name not in GENERATED_GRID_BASES:
        return None
    if len(entry.get("tensor_returns", [])) != 1:
        return None

    safe_types = {
        "Tensor",
        "bool",
        "int",
        "List[int]",
    }
    for arg in entry.get("args", []):
        if arg.get("is_out"):
            continue
        if arg.get("tensor"):
            continue
        if arg.get("type") not in safe_types:
            return None

    return {
        "strategy": "manual_grid",
        "family": base_name,
    }


def _manual_grid_backward_strategy(entry: dict) -> dict | None:
    surface_kind = entry["surface_kind"]
    if surface_kind not in {"autograd_backward", "out_variant"}:
        return None

    base_name = entry["base_name"]
    if base_name not in GENERATED_GRID_BACKWARD_BASES:
        return None
    if len(entry.get("tensor_returns", [])) != 2:
        return None
    if surface_kind == "out_variant":
        out_args = [arg for arg in entry.get("args", []) if _is_out_arg(arg)]
        if len(out_args) != 2:
            return None

    safe_types = {
        "Tensor",
        "bool",
        "int",
        "List[bool]",
    }
    for arg in entry.get("args", []):
        if _is_out_arg(arg):
            continue
        if arg.get("tensor"):
            continue
        if arg.get("type") not in safe_types:
            return None

    return {
        "strategy": "manual_grid_backward",
        "family": base_name,
    }


def _manual_rnn_cell_strategy(entry: dict) -> dict | None:
    if entry["surface_kind"] != "functional_data":
        return None
    base_name = entry["base_name"]
    if base_name not in GENERATED_RNN_CELL_BASES:
        return None
    if len(entry.get("tensor_returns", [])) not in {1, 2}:
        return None

    safe_types = {
        "Tensor",
        "Tensor?",
        "List[Tensor]",
    }
    for arg in entry.get("args", []):
        if arg.get("tensor"):
            continue
        if arg.get("type") not in safe_types:
            return None
    return {
        "strategy": "manual_rnn_cell",
        "family": base_name,
    }


def _manual_fft_strategy(entry: dict) -> dict | None:
    if entry["surface_kind"] not in {"functional_data", "out_variant"}:
        return None
    base_name = entry["base_name"]
    if base_name not in GENERATED_FFT_BASES:
        return None
    if len(entry.get("tensor_returns", [])) != 1:
        return None

    safe_types = {
        "Tensor",
        "bool",
        "int",
        "List[int]",
        "Optional[int]",
        "Optional[List[int]]",
        "Optional[str]",
    }
    for arg in entry.get("args", []):
        if arg.get("is_out"):
            continue
        if arg.get("tensor"):
            continue
        if arg.get("type") not in safe_types:
            return None
    return {
        "strategy": "manual_fft",
        "family": base_name,
    }


def _manual_loss_strategy(entry: dict) -> dict | None:
    surface_kind = entry["surface_kind"]
    if surface_kind not in {"functional_data", "out_variant"}:
        return None

    base_name = entry["base_name"]
    if base_name not in GENERATED_LOSS_BASES:
        return None
    if "backward" in base_name:
        return None
    if len(entry.get("tensor_returns", [])) != 1:
        return None

    safe_types = {
        "Tensor",
        "bool",
        "float",
        "int",
        "List[int]",
        "number",
    }
    for arg in entry.get("args", []):
        if arg.get("is_out"):
            continue
        if arg.get("tensor"):
            continue
        if arg.get("type") not in safe_types:
            return None

    return {
        "strategy": "manual_loss",
        "family": base_name,
    }


def _manual_padding_strategy(entry: dict) -> dict | None:
    if entry["surface_kind"] not in {"functional_data", "out_variant"}:
        return None
    base_name = entry["base_name"]
    if base_name not in GENERATED_PADDING_BASES:
        return None
    if len(entry.get("tensor_returns", [])) > 1:
        return None

    safe_types = {
        "Tensor",
        "float",
        "int",
        "List[int]",
        "number",
        "Optional[float]",
        "str",
    }
    for arg in entry.get("args", []):
        if arg.get("is_out"):
            continue
        if arg.get("tensor"):
            continue
        if arg.get("type") not in safe_types:
            return None
    return {
        "strategy": "manual_padding",
        "family": base_name,
    }


def _manual_linalg_strategy(entry: dict) -> dict | None:
    surface_kind = entry["surface_kind"]
    if surface_kind not in {"functional_data", "out_variant", "mutating_or_inplace"}:
        return None

    base_name = entry["base_name"].rstrip("_")
    if base_name not in GENERATED_LINALG_BASES:
        return None
    if len(entry.get("tensor_returns", [])) != 1:
        return None

    safe_types = {
        "Tensor",
        "bool",
        "float",
        "int",
        "List[int]",
        "number",
        "Optional[float]",
        "Optional[int]",
        "Optional[List[int]]",
        "Optional[number]",
        "Optional[str]",
        "str",
        "str?",
    }
    for arg in entry.get("args", []):
        if arg.get("is_out"):
            continue
        if arg.get("tensor"):
            continue
        if arg.get("type") not in safe_types:
            return None

    return {
        "strategy": "manual_linalg",
        "family": base_name,
    }


def _manual_metadata_strategy(entry: dict) -> dict | None:
    if entry["surface_kind"] not in {"functional_data", "metadata_device"}:
        return None
    base_name = entry["base_name"]
    if base_name not in GENERATED_METADATA_BASES:
        return None
    if "Dimname" in entry.get("overload", ""):
        return None

    safe_types = {
        "Tensor",
        "bool",
        "Device?",
        "int",
        "Layout?",
        "List[int]",
        "MemoryFormat",
        "number",
        "Optional[Device]",
        "Optional[int]",
        "Optional[List[int]]",
        "ScalarType",
        "ScalarType?",
        "str",
    }
    for arg in entry.get("args", []):
        if arg.get("tensor"):
            continue
        if arg.get("type") not in safe_types:
            return None
    return {
        "strategy": "manual_metadata",
        "family": base_name,
    }


def _manual_factory_out_strategy(entry: dict) -> dict | None:
    if entry["surface_kind"] != "out_variant":
        return None
    if entry["base_name"] not in GENERATED_FACTORY_OUT_BASES:
        return None
    if len(entry.get("tensor_returns", [])) != 1:
        return None

    args = entry.get("args", [])
    has_names_arg = any(arg.get("name") == "names" for arg in args)
    if has_names_arg and entry["base_name"] not in GENERATED_FACTORY_NAMES_OUT_BASES:
        return None
    out_args = [arg for arg in args if arg.get("name") == "out"]
    if len(out_args) != 1:
        return None

    safe_types = {
        "Tensor",
        "bool",
        "float",
        "int",
        "List[int]",
        "number",
        "Optional[int]",
        "Optional[List[str]]",
    }
    for arg in args:
        if arg.get("name") == "out":
            continue
        if arg.get("tensor"):
            continue
        if arg.get("type") not in safe_types:
            return None

    return {
        "strategy": "manual_factory_out",
        "family": entry["base_name"],
    }


def _manual_matmul_strategy(entry: dict) -> dict | None:
    if entry["name"] not in GENERATED_MATMUL_SURFACES:
        return None
    if entry["surface_kind"] not in {"functional_data", "out_variant", "mutating_or_inplace"}:
        return None
    if entry["base_name"].startswith("_"):
        return None
    return {
        "strategy": "manual_matmul",
        "family": entry["base_name"],
    }


def _manual_shape_strategy(entry: dict) -> dict | None:
    name = entry["name"]
    if name in GENERATED_MANUAL_SHAPE_VIEW_SURFACES:
        if entry["surface_kind"] not in {"functional_data", "view_or_alias"}:
            return None
    elif name in GENERATED_MANUAL_SHAPE_OUT_SURFACES:
        if entry["surface_kind"] != "out_variant":
            return None
        out_args = [arg for arg in entry.get("args", []) if arg.get("name") == "out"]
        if len(out_args) != 1:
            return None
        if out_args[0].get("type") != "List[Tensor]" and len(entry.get("tensor_returns", [])) != 1:
            return None
    elif name in GENERATED_MANUAL_SHAPE_INPLACE_SURFACES:
        if entry["surface_kind"] != "mutating_or_inplace":
            return None
        expected_returns = 0 if name == "aten::set_data" else 1
        if len(entry.get("tensor_returns", [])) != expected_returns:
            return None
    else:
        return None
    return {
        "strategy": "manual_shape",
        "family": entry["base_name"].rstrip("_"),
    }


def _generated_strategy_for_entry(entry: dict, opinfo_map: dict) -> dict | None:
    if entry["surface_kind"] == "out_variant":
        rng_strategy = _manual_rng_strategy(entry)
        if rng_strategy is not None:
            return rng_strategy
        multi_reduction_strategy = _manual_multi_output_reduction_strategy(entry)
        if multi_reduction_strategy is not None:
            return multi_reduction_strategy
        upsample_strategy = _manual_upsample_strategy(entry)
        if upsample_strategy is not None:
            return upsample_strategy
        pooling_strategy = _manual_pooling_strategy(entry)
        if pooling_strategy is not None:
            return pooling_strategy
        convolution_strategy = _manual_convolution_strategy(entry)
        if convolution_strategy is not None:
            return convolution_strategy
        grid_strategy = _manual_grid_strategy(entry)
        if grid_strategy is not None:
            return grid_strategy
        grid_backward_strategy = _manual_grid_backward_strategy(entry)
        if grid_backward_strategy is not None:
            return grid_backward_strategy
        rnn_cell_strategy = _manual_rnn_cell_strategy(entry)
        if rnn_cell_strategy is not None:
            return rnn_cell_strategy
        fft_strategy = _manual_fft_strategy(entry)
        if fft_strategy is not None:
            return fft_strategy
        loss_strategy = _manual_loss_strategy(entry)
        if loss_strategy is not None:
            return loss_strategy
        padding_strategy = _manual_padding_strategy(entry)
        if padding_strategy is not None:
            return padding_strategy
        linalg_strategy = _manual_linalg_strategy(entry)
        if linalg_strategy is not None:
            return linalg_strategy
        matmul_strategy = _manual_matmul_strategy(entry)
        if matmul_strategy is not None:
            return matmul_strategy
        shape_strategy = _manual_shape_strategy(entry)
        if shape_strategy is not None:
            return shape_strategy
        matmul_strategy = _manual_matmul_strategy(entry)
        if matmul_strategy is not None:
            return matmul_strategy
        bitwise_strategy = _manual_bitwise_strategy(entry)
        if bitwise_strategy is not None:
            return bitwise_strategy
        special_strategy = _manual_special_math_strategy(entry)
        if special_strategy is not None:
            return special_strategy
        elementwise_strategy = _manual_elementwise_strategy(entry)
        if elementwise_strategy is not None:
            return elementwise_strategy
        reduction_strategy = _manual_reduction_strategy(entry)
        if reduction_strategy is not None:
            return reduction_strategy
        indexing_strategy = _manual_indexing_strategy(entry)
        if indexing_strategy is not None:
            return indexing_strategy
        factory_out_strategy = _manual_factory_out_strategy(entry)
        if factory_out_strategy is not None:
            return factory_out_strategy

        if entry["base_name"].startswith("_foreach_"):
            overload = entry["overload"]
            if overload == "out":
                logical_overload = ""
            elif overload.endswith("_out"):
                logical_overload = overload.removesuffix("_out")
            else:
                return None
            foreach_name = entry["base_name"].removeprefix("_foreach_")
            out_args = [arg for arg in entry.get("args", []) if arg.get("name") == "out"]
            if len(out_args) != 1:
                return None
            return _manual_foreach_strategy(foreach_name, logical_overload, entry, require_tensor_return=False)

        if entry["variant_kind"] != "out":
            return None
        if entry["overload"] != "out":
            return None
        if entry["base_name"] not in GENERATED_OPINFO_OUT_ALLOWLIST:
            return None
        if entry["base_name"].startswith("_"):
            return None
        out_args = [arg for arg in entry.get("args", []) if arg.get("name") == "out"]
        if len(out_args) != 1:
            return None
        if len(entry.get("tensor_returns", [])) != 1:
            return None
        matches = _opinfo_matches_for_entry(entry, opinfo_map)
        if matches != [entry["base_name"]]:
            return None
        if not opinfo_map.get("supports_out", {}).get(entry["base_name"], False):
            return None
        return {
            "strategy": "opinfo_out",
            "opinfo_name": entry["base_name"],
        }

    if entry["surface_kind"] == "mutating_or_inplace":
        rng_strategy = _manual_rng_strategy(entry)
        if rng_strategy is not None:
            return rng_strategy
        multi_reduction_strategy = _manual_multi_output_reduction_strategy(entry)
        if multi_reduction_strategy is not None:
            return multi_reduction_strategy
        pooling_strategy = _manual_pooling_strategy(entry)
        if pooling_strategy is not None:
            return pooling_strategy
        convolution_strategy = _manual_convolution_strategy(entry)
        if convolution_strategy is not None:
            return convolution_strategy
        linalg_strategy = _manual_linalg_strategy(entry)
        if linalg_strategy is not None:
            return linalg_strategy
        shape_strategy = _manual_shape_strategy(entry)
        if shape_strategy is not None:
            return shape_strategy
        matmul_strategy = _manual_matmul_strategy(entry)
        if matmul_strategy is not None:
            return matmul_strategy
        bitwise_strategy = _manual_bitwise_strategy(entry)
        if bitwise_strategy is not None:
            return bitwise_strategy
        special_strategy = _manual_special_math_strategy(entry)
        if special_strategy is not None:
            return special_strategy
        elementwise_strategy = _manual_elementwise_strategy(entry)
        if elementwise_strategy is not None:
            return elementwise_strategy
        reduction_strategy = _manual_reduction_strategy(entry)
        if reduction_strategy is not None:
            return reduction_strategy
        indexing_strategy = _manual_indexing_strategy(entry)
        if indexing_strategy is not None:
            return indexing_strategy

        if entry["base_name"].startswith("_foreach_") and entry["base_name"].endswith("_"):
            foreach_name = entry["base_name"].removeprefix("_foreach_").removesuffix("_")
            return _manual_foreach_strategy(
                foreach_name,
                entry["overload"],
                entry,
                require_tensor_return=False,
            )

        if entry["variant_kind"] != "inplace":
            return None
        if entry["overload"]:
            return None
        if not entry["base_name"].endswith("_"):
            return None
        opinfo_name = entry["base_name"].rstrip("_")
        if opinfo_name not in GENERATED_OPINFO_UNARY_INPLACE_ALLOWLIST:
            return None
        if entry["base_name"].startswith("_"):
            return None
        args = entry.get("args", [])
        if len(args) != 1 or args[0].get("name") != "self" or not args[0].get("tensor"):
            return None
        if len(entry.get("tensor_returns", [])) != 1:
            return None
        matches = _opinfo_matches_for_entry(entry, opinfo_map)
        if matches != [opinfo_name]:
            return None
        return {
            "strategy": "opinfo_inplace_unary",
            "opinfo_name": opinfo_name,
        }

    if entry["surface_kind"] == "view_or_alias":
        shape_strategy = _manual_shape_strategy(entry)
        if shape_strategy is not None:
            return shape_strategy
        indexing_strategy = _manual_indexing_strategy(entry)
        if indexing_strategy is not None:
            return indexing_strategy
        opinfo_name = GENERATED_OPINFO_VIEW_STRATEGIES.get(entry["name"])
        if not opinfo_name:
            return None
        if entry["base_name"].startswith("_"):
            return None
        if len(entry.get("tensor_returns", [])) != 1:
            return None
        if not any(ret.get("alias") for ret in entry.get("returns", [])):
            return None
        matches = _opinfo_matches_for_entry(entry, opinfo_map)
        if matches != [opinfo_name]:
            return None
        return {
            "strategy": "opinfo_view_alias",
            "opinfo_name": opinfo_name,
        }

    if entry["surface_kind"] == "factory":
        family = GENERATED_FACTORY_STRATEGIES.get(entry["name"])
        if not family:
            return None
        if entry["base_name"].startswith("_") and entry["name"] not in GENERATED_SAFE_INTERNAL_FACTORIES:
            return None
        if len(entry.get("tensor_args", [])) != 0:
            return None
        if len(entry.get("tensor_returns", [])) != 1:
            return None
        return {
            "strategy": "manual_factory",
            "family": family,
        }

    if entry["surface_kind"] == "functional_data" and entry["base_name"].startswith("_foreach_"):
        foreach_name = entry["base_name"].removeprefix("_foreach_")
        return _manual_foreach_strategy(
            foreach_name,
            entry["overload"],
            entry,
            require_tensor_return=True,
        )

    if entry["surface_kind"] == "rng":
        rng_strategy = _manual_rng_strategy(entry)
        if rng_strategy is not None:
            return rng_strategy

    if entry["surface_kind"] == "metadata_device":
        metadata_strategy = _manual_metadata_strategy(entry)
        if metadata_strategy is not None:
            return metadata_strategy

    if entry["surface_kind"] == "autograd_backward":
        multi_reduction_strategy = _manual_multi_output_reduction_strategy(entry)
        if multi_reduction_strategy is not None:
            return multi_reduction_strategy
        grid_backward_strategy = _manual_grid_backward_strategy(entry)
        if grid_backward_strategy is not None:
            return grid_backward_strategy
        reduction_strategy = _manual_reduction_strategy(entry)
        if reduction_strategy is not None:
            return reduction_strategy
        return None

    if entry["surface_kind"] == "functional_data":
        rng_strategy = _manual_rng_strategy(entry)
        if rng_strategy is not None:
            return rng_strategy
        metadata_strategy = _manual_metadata_strategy(entry)
        if metadata_strategy is not None:
            return metadata_strategy
        multi_reduction_strategy = _manual_multi_output_reduction_strategy(entry)
        if multi_reduction_strategy is not None:
            return multi_reduction_strategy
        upsample_strategy = _manual_upsample_strategy(entry)
        if upsample_strategy is not None:
            return upsample_strategy
        pooling_strategy = _manual_pooling_strategy(entry)
        if pooling_strategy is not None:
            return pooling_strategy
        convolution_strategy = _manual_convolution_strategy(entry)
        if convolution_strategy is not None:
            return convolution_strategy
        grid_strategy = _manual_grid_strategy(entry)
        if grid_strategy is not None:
            return grid_strategy
        rnn_cell_strategy = _manual_rnn_cell_strategy(entry)
        if rnn_cell_strategy is not None:
            return rnn_cell_strategy
        fft_strategy = _manual_fft_strategy(entry)
        if fft_strategy is not None:
            return fft_strategy
        loss_strategy = _manual_loss_strategy(entry)
        if loss_strategy is not None:
            return loss_strategy
        padding_strategy = _manual_padding_strategy(entry)
        if padding_strategy is not None:
            return padding_strategy
        linalg_strategy = _manual_linalg_strategy(entry)
        if linalg_strategy is not None:
            return linalg_strategy
        matmul_strategy = _manual_matmul_strategy(entry)
        if matmul_strategy is not None:
            return matmul_strategy
        shape_strategy = _manual_shape_strategy(entry)
        if shape_strategy is not None:
            return shape_strategy
        special_strategy = _manual_special_math_strategy(entry)
        if special_strategy is not None:
            return special_strategy
        elementwise_strategy = _manual_elementwise_strategy(entry)
        if elementwise_strategy is not None:
            return elementwise_strategy
        reduction_strategy = _manual_reduction_strategy(entry)
        if reduction_strategy is not None:
            return reduction_strategy
        indexing_strategy = _manual_indexing_strategy(entry)
        if indexing_strategy is not None:
            return indexing_strategy

    if entry["surface_kind"] == "layout_storage":
        pooling_strategy = _manual_pooling_strategy(entry)
        if pooling_strategy is not None:
            return pooling_strategy

    return None


def _generated_case_depth_for_entry(entry: dict, strategy: dict | None) -> dict:
    if not strategy:
        return {
            "planned_count": 0,
            "required_count": 0,
            "optional_count": 0,
            "case_ids": [],
            "required_case_ids": [],
            "optional_case_ids": [],
            "tags": [],
            "cases": [],
        }
    try:
        from torchcts.sample_generation import sample_case_depth_for_entry

        planned_entry = dict(entry)
        planned_entry["generated"] = {"strategy": strategy}
        return sample_case_depth_for_entry(planned_entry)
    except Exception as exc:
        return {
            "planned_count": 0,
            "required_count": 0,
            "optional_count": 0,
            "case_ids": [],
            "required_case_ids": [],
            "optional_case_ids": [],
            "tags": [],
            "cases": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _marker_maps(marker_data: dict) -> tuple[dict[str, list[dict]], dict[str, list[dict]], list[dict]]:
    handwritten = defaultdict(list)
    generated = defaultdict(list)
    categories = []
    for marker in marker_data["markers"]:
        for covered in marker.get("covers", []):
            if marker.get("generated"):
                generated[covered].append(marker)
            else:
                handwritten[covered].append(marker)
        for category in marker.get("categories", []):
            categories.append({
                "category": category,
                "nodeid": marker["nodeid"],
                "capabilities": marker.get("capabilities", []),
                "semantic_level": marker.get("semantic_level"),
                "level_reason": marker.get("level_reason"),
                "level_source": marker.get("level_source"),
                "generated": marker.get("generated", False),
            })
    return dict(handwritten), dict(generated), categories


def _level_payload(level_info: SemanticLevelInfo) -> dict:
    return {
        "semantic_level": level_info.level,
        "semantic_levels": [level_info.level],
        "min_semantic_level": level_info.level,
        "max_semantic_level": level_info.level,
        "level_reason": level_info.reason,
        "level_source": level_info.source,
    }


def _marker_level_payload(markers: list[dict], *, fallback: SemanticLevelInfo) -> dict:
    levels = [
        validate_semantic_level(marker["semantic_level"])
        for marker in markers
        if marker.get("semantic_level") is not None
    ]
    if not levels:
        return _level_payload(fallback)
    minimum = min(levels)
    selected = next((marker for marker in markers if marker.get("semantic_level") == minimum), None)
    return {
        "semantic_level": minimum,
        "semantic_levels": sorted(set(levels)),
        "min_semantic_level": minimum,
        "max_semantic_level": max(levels),
        "level_reason": (
            selected.get("level_reason")
            if selected
            else f"Covered by hand-authored marker at level {minimum}."
        ),
        "level_source": selected.get("level_source") if selected else "marker",
    }


def _generated_depth_level_payload(entry: dict, depth: dict, *, fallback: SemanticLevelInfo) -> dict:
    levels = [
        validate_semantic_level(level)
        for level in depth.get("semantic_levels", [])
        if level is not None
    ]
    if not levels:
        return _level_payload(fallback)
    minimum = min(levels)
    selected_case = next(
        (
            case for case in depth.get("cases", [])
            if case.get("semantic_level") == minimum
        ),
        None,
    )
    return {
        "semantic_level": minimum,
        "semantic_levels": sorted(set(levels)),
        "min_semantic_level": minimum,
        "max_semantic_level": max(levels),
        "level_reason": (
            selected_case.get("level_reason")
            if selected_case
            else f"Generated cases for {entry['name']} start at level {minimum}."
        ),
        "level_source": selected_case.get("level_source") if selected_case else "generated_case_depth",
    }


def _audit_entry_level_payload(
    entry: dict,
    *,
    status: str,
    hand_matches: list[dict],
    generated_matches: list[dict],
    generated_strategy: dict | None,
    generated_case_depth: dict,
) -> dict:
    if status == "covered_handwritten":
        return _marker_level_payload(
            hand_matches,
            fallback=suite_default_level(suite_for_path(hand_matches[0]["path"] if hand_matches else "")),
        )
    if status == "covered_generated":
        fallback = generated_level_for_entry({**entry, "generated": {"strategy": generated_strategy}})
        if generated_strategy:
            return _generated_depth_level_payload(entry, generated_case_depth, fallback=fallback)
        return _marker_level_payload(generated_matches, fallback=fallback)
    if status == "covered_opinfo":
        return _level_payload(suite_default_level("opinfo"))
    if status in {
        "covered_oracle",
        "covered_backend_pack",
        "covered_property",
        "pending_oracle",
        "pending_backend_pack",
        "pending_property",
        "excluded",
        "excluded_framework_plumbing",
        "excluded_deprecated_or_removed",
        "excluded_unsupported_public_api",
        "excluded_distributed_scope",
        "excluded_host_storage",
        "unknown",
        "unavailable_in_pytorch_runtime",
    }:
        return _level_payload(generated_level_for_entry(entry))
    return {
        "semantic_level": None,
        "semantic_levels": [],
        "min_semantic_level": None,
        "max_semantic_level": None,
        "level_reason": "Not backend relevant.",
        "level_source": "not_backend_relevant",
    }


def _summarize_generated_case_depth(entries: list[dict]) -> dict:
    by_strategy = Counter()
    by_tag = Counter()
    by_level = Counter()
    surfaces_with_case_plan = 0
    planned = 0
    required = 0
    optional = 0
    for entry in entries:
        if entry.get("status") != "covered_generated":
            continue
        generated = entry.get("generated") or {}
        depth = generated.get("case_depth") or {}
        planned_count = int(depth.get("planned_count", 0) or 0)
        if planned_count <= 0:
            continue
        surfaces_with_case_plan += 1
        planned += planned_count
        required += int(depth.get("required_count", 0) or 0)
        optional += int(depth.get("optional_count", 0) or 0)
        strategy = (generated.get("strategy") or {}).get("strategy") or "marker_only"
        by_strategy[strategy] += planned_count
        for tag in depth.get("tags", []):
            by_tag[str(tag)] += planned_count
        for case in depth.get("cases", []):
            if case.get("semantic_level") is not None:
                by_level[str(case["semantic_level"])] += 1
    return {
        "generated_surfaces_with_case_plan": surfaces_with_case_plan,
        "generated_semantic_cases": planned,
        "required_generated_semantic_cases": required,
        "optional_generated_semantic_cases": optional,
        "by_strategy": dict(sorted(by_strategy.items())),
        "by_tag": dict(sorted(by_tag.items())),
        "by_semantic_level": dict(sorted(by_level.items(), key=lambda item: int(item[0]))),
        "note": "Counts semantic case families before dtype, device, and IEEE754 input-condition expansion.",
    }


def build_audit(root: str | os.PathLike | None = None) -> dict:
    inventory = build_dispatcher_inventory()
    live_names = {entry["name"] for entry in inventory["entries"]}
    runtime_unavailable_entries = runtime_unavailable_op_entries(
        runtime_version=torch.__version__,
        live_names=live_names,
    )
    opinfo_map = build_opinfo_map()
    marker_data = collect_coverage_markers(root)
    handwritten_markers, generated_markers, category_markers = _marker_maps(marker_data)
    exclusion_data = load_exclusions(inventory)

    errors = list(exclusion_data["errors"])
    errors.extend(marker_data.get("errors", []))
    warnings = list(exclusion_data["warnings"])
    valid_names = live_names | {entry["name"] for entry in runtime_unavailable_entries}
    for marker in marker_data["markers"]:
        for covered in marker.get("covers", []):
            if covered not in valid_names:
                errors.append(f"{marker['nodeid']} marks unknown dispatcher surface {covered!r}")
    audited_entries = []

    for entry in inventory["entries"]:
        entry = dict(entry)
        opinfo_covered, opinfo_matches = _opinfo_covers(entry, opinfo_map)
        hand_matches = handwritten_markers.get(entry["name"], [])
        generated_matches = generated_markers.get(entry["name"], [])
        generated_strategy = _generated_strategy_for_entry(entry, opinfo_map)
        generated_case_depth = _generated_case_depth_for_entry(entry, generated_strategy)
        if generated_strategy and generated_case_depth.get("error"):
            errors.append(
                f"{entry['name']} generated strategy {generated_strategy.get('strategy')!r} "
                f"has invalid sample case metadata: {generated_case_depth['error']}"
            )
        exclusion_matches = [
            exclusion for exclusion in exclusion_data["exclusions"]
            if _matches_exclusion(entry, exclusion)
        ]
        exclusion_match = exclusion_matches[0] if exclusion_matches else None
        custom_status, oracle_payload = _status_from_exclusion(entry, exclusion_match)

        if entry["surface_kind"] == "not_backend_relevant":
            status = "not_backend_relevant"
        elif exclusion_match:
            status = custom_status
        elif hand_matches:
            status = "covered_handwritten"
        elif generated_matches or generated_strategy:
            status = "covered_generated"
        elif opinfo_covered:
            status = "covered_opinfo"
        else:
            status = "unknown"
        pending_review = _pending_review_for_entry(entry, status, exclusion_match, oracle_payload)

        level_payload = _audit_entry_level_payload(
            entry,
            status=status,
            hand_matches=hand_matches,
            generated_matches=generated_matches,
            generated_strategy=generated_strategy,
            generated_case_depth=generated_case_depth,
        )

        entry["opinfo"] = {"covered": opinfo_covered, "matches": opinfo_matches}
        entry["handwritten"] = {
            "covered": bool(hand_matches),
            "markers": [
                {
                    "nodeid": marker["nodeid"],
                    "path": marker["path"],
                    "semantic_level": marker.get("semantic_level"),
                    "level_reason": marker.get("level_reason"),
                    "level_source": marker.get("level_source"),
                }
                for marker in hand_matches
            ],
        }
        entry["generated"] = {
            "covered": bool(generated_matches or generated_strategy),
            "strategy": generated_strategy,
            "case_depth": generated_case_depth,
            "markers": [
                {
                    "nodeid": marker["nodeid"],
                    "path": marker["path"],
                    "semantic_level": marker.get("semantic_level"),
                    "level_reason": marker.get("level_reason"),
                    "level_source": marker.get("level_source"),
                }
                for marker in generated_matches
            ],
        }
        entry["exclusion"] = exclusion_match
        entry["oracle"] = oracle_payload
        entry["coverage_kind"] = _coverage_kind_for_status(status)
        entry["pending_review"] = pending_review
        entry["status"] = status
        entry.update(level_payload)
        audited_entries.append(entry)

    for entry in runtime_unavailable_entries:
        entry = dict(entry)
        status = "unavailable_in_pytorch_runtime"
        level_payload = _audit_entry_level_payload(
            entry,
            status=status,
            hand_matches=[],
            generated_matches=[],
            generated_strategy=None,
            generated_case_depth={},
        )
        entry["opinfo"] = {"covered": False, "matches": []}
        entry["handwritten"] = {"covered": False, "markers": []}
        entry["generated"] = {"covered": False, "strategy": None, "case_depth": {}, "markers": []}
        entry["exclusion"] = None
        entry["oracle"] = None
        entry["coverage_kind"] = _coverage_kind_for_status(status)
        entry["pending_review"] = None
        entry["status"] = status
        entry.update(level_payload)
        audited_entries.append(entry)

    status_counts = Counter(entry["status"] for entry in audited_entries)
    surface_counts = Counter(entry["surface_kind"] for entry in audited_entries)
    semantic_level_counts = Counter(
        str(entry["semantic_level"])
        for entry in audited_entries
        if entry.get("semantic_level") is not None and entry["status"] != "not_backend_relevant"
    )
    semantic_level_status_counts = defaultdict(Counter)
    semantic_level_surface_counts = defaultdict(Counter)
    for entry in audited_entries:
        if entry.get("semantic_level") is None or entry["status"] == "not_backend_relevant":
            continue
        level_key = str(entry["semantic_level"])
        semantic_level_status_counts[level_key][entry["status"]] += 1
        semantic_level_surface_counts[level_key][entry["surface_kind"]] += 1
    generated_case_depth = _summarize_generated_case_depth(audited_entries)
    unknown_entries = [entry for entry in audited_entries if entry["status"] == "unknown"]
    pending_blocker_counts = Counter(
        entry["pending_review"]["blocker_type"]
        for entry in audited_entries
        if entry.get("pending_review")
    )
    pending_backend_gate_counts = Counter(
        entry["pending_review"]["backend_gate"]
        for entry in audited_entries
        if entry.get("pending_review")
    )
    coverage_kind_counts = Counter(
        entry.get("coverage_kind", "unknown")
        for entry in audited_entries
        if entry["status"] != "not_backend_relevant"
    )

    if unknown_entries:
        warnings.append(
            f"TorchCTS coverage audit found {len(unknown_entries)} unknown tensor-touching ATen surfaces."
        )

    return {
        "metadata": {
            "pytorch_version": torch.__version__,
            "generated_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            "total_aten_overloads": len(audited_entries),
            "unknown_count": len(unknown_entries),
            "status_counts": dict(sorted(status_counts.items())),
            "surface_counts": dict(sorted(surface_counts.items())),
            "semantic_level_counts": dict(sorted(semantic_level_counts.items(), key=lambda item: int(item[0]))),
            "semantic_level_status_counts": {
                level: dict(sorted(counts.items()))
                for level, counts in sorted(semantic_level_status_counts.items(), key=lambda item: int(item[0]))
            },
            "semantic_level_surface_counts": {
                level: dict(sorted(counts.items()))
                for level, counts in sorted(semantic_level_surface_counts.items(), key=lambda item: int(item[0]))
            },
            "semantic_level_descriptions": {
                str(level): semantic_level_description(level)
                for level in range(1, 9)
            },
            "generated_case_depth": generated_case_depth,
            "pending_blocker_counts": dict(sorted(pending_blocker_counts.items())),
            "pending_backend_gate_counts": dict(sorted(pending_backend_gate_counts.items())),
            "coverage_kind_counts": dict(sorted(coverage_kind_counts.items())),
            "dtype_contract_mismatch_counts": dtype_contract_mismatch_counts(),
            "default_output_dir": str(DEFAULT_OUTPUT_DIR),
        },
        "entries": audited_entries,
        "coverage_markers": marker_data["markers"],
        "category_markers": category_markers,
        "unmapped_tests": marker_data["unmapped_tests"],
        "warnings": warnings,
        "errors": errors,
    }


def _ensure_output_dir() -> None:
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def write_inventory() -> dict:
    _ensure_output_dir()
    inventory = build_dispatcher_inventory()
    DEFAULT_INVENTORY_PATH.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    return inventory


def _group_unknowns(entries: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for entry in entries:
        if entry["status"] == "unknown":
            grouped[entry["surface_kind"]].append(entry)
    return dict(grouped)


def render_unknowns_markdown(audit: dict) -> str:
    lines = ["# TorchCTS Unknown Coverage Surfaces", ""]
    unknowns = [entry for entry in audit["entries"] if entry["status"] == "unknown"]
    lines.append(f"Unknown tensor-touching ATen surfaces: {len(unknowns)}")
    lines.append("")
    for surface, entries in sorted(_group_unknowns(audit["entries"]).items()):
        lines.append(f"## {surface} ({len(entries)})")
        lines.append("")
        for entry in entries[:500]:
            lines.append(f"- `{entry['name']}` - `{entry['schema']}`")
        if len(entries) > 500:
            lines.append(f"- ... {len(entries) - 500} more")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_unmapped_tests_markdown(audit: dict) -> str:
    lines = ["# TorchCTS Unmapped Hand-Authored Tests", ""]
    unmapped = audit.get("unmapped_tests", [])
    lines.append(f"Unmapped tests: {len(unmapped)}")
    lines.append("")
    for item in unmapped:
        lines.append(f"- `{item['nodeid']}` - {item['reason']}")
    return "\n".join(lines).rstrip() + "\n"


def pending_review_records(audit: dict) -> list[dict]:
    records = []
    for entry in audit.get("entries", []):
        review = entry.get("pending_review")
        if not review:
            continue
        records.append({
            "name": entry["name"],
            "schema": entry["schema"],
            "status": entry["status"],
            "surface_kind": entry["surface_kind"],
            "coverage_kind": entry.get("coverage_kind"),
            "blocker_type": review.get("blocker_type"),
            "required_closure": review.get("required_closure"),
            "backend_gate": review.get("backend_gate"),
            "next_family": review.get("next_family"),
            "reason": review.get("reason"),
            "source_category": review.get("source_category"),
            "owner": review.get("owner"),
            "review_after": review.get("review_after"),
            "dispatch": entry.get("dispatch") or {},
        })
    return records


def build_pending_review_artifact(audit: dict) -> dict:
    records = pending_review_records(audit)
    blocker_counts = Counter(record["blocker_type"] for record in records)
    backend_gate_counts = Counter(record["backend_gate"] for record in records)
    status_counts = Counter(record["status"] for record in records)
    family_counts = Counter(record["next_family"] for record in records)
    return {
        "metadata": {
            "pytorch_version": audit.get("metadata", {}).get("pytorch_version"),
            "generated_at": audit.get("metadata", {}).get("generated_at"),
            "pending_or_excluded_count": len(records),
            "blocker_counts": dict(sorted(blocker_counts.items())),
            "backend_gate_counts": dict(sorted(backend_gate_counts.items())),
            "status_counts": dict(sorted(status_counts.items())),
            "family_counts": dict(sorted(family_counts.items())),
        },
        "records": records,
    }


def render_pending_review_markdown(audit: dict) -> str:
    artifact = build_pending_review_artifact(audit)
    metadata = artifact["metadata"]
    records = artifact["records"]
    lines = [
        "# TorchCTS Pending Coverage Review",
        "",
        f"Pending or excluded surfaces: {metadata['pending_or_excluded_count']}",
        "",
        "## By Blocker Type",
        "",
    ]
    for blocker, count in metadata["blocker_counts"].items():
        lines.append(f"- `{blocker}`: {count}")
    if not metadata["blocker_counts"]:
        lines.append("- none")
    lines.extend(["", "## By Backend Gate", ""])
    for gate, count in metadata["backend_gate_counts"].items():
        lines.append(f"- `{gate}`: {count}")
    if not metadata["backend_gate_counts"]:
        lines.append("- none")
    lines.extend(["", "## Records", ""])
    for record in records:
        lines.append(
            f"- `{record['name']}` - `{record['status']}` - "
            f"`{record['blocker_type']}` - `{record['required_closure']}` - "
            f"gate `{record['backend_gate']}`"
        )
        if record.get("reason"):
            lines.append(f"  Reason: {record['reason']}")
    return "\n".join(lines).rstrip() + "\n"


def render_summary_markdown(audit: dict) -> str:
    metadata = audit["metadata"]
    status_counts = metadata.get("status_counts", {})
    runtime_unavailable = sum(status_counts.get(status, 0) for status in RUNTIME_UNAVAILABLE_STATUSES)
    relevant_total = (
        metadata["total_aten_overloads"]
        - status_counts.get("not_backend_relevant", 0)
        - runtime_unavailable
    )
    covered_total = sum(status_counts.get(status, 0) for status in COVERED_STATUSES)
    covered_percent = (covered_total / relevant_total * 100.0) if relevant_total else 100.0
    lines = [
        "# TorchCTS Coverage Summary",
        "",
        f"PyTorch: `{metadata['pytorch_version']}`",
        f"Total ATen overloads: {metadata['total_aten_overloads']}",
        f"Backend-relevant overloads: {relevant_total}",
        f"Runtime-unavailable overloads: {runtime_unavailable}",
        f"Covered relevant overloads: {covered_total} ({covered_percent:.1f}%)",
        f"Unknown surfaces: {metadata['unknown_count']}",
        "",
        "## Unknown Warning",
        "",
    ]
    if metadata["unknown_count"]:
        lines.extend([
            f"WARNING: TorchCTS coverage audit found {metadata['unknown_count']} unknown tensor-touching ATen surfaces.",
            "These surfaces are not counted as covered and generated tests will skip them.",
        ])
    else:
        lines.append("No unknown tensor-touching ATen surfaces remain.")
    lines.extend([
        "",
        "## Status Counts",
        "",
    ])
    for status in sorted(FINAL_STATUSES):
        lines.append(f"- `{status}`: {metadata.get('status_counts', {}).get(status, 0)}")
    lines.extend(["", "## Surface Counts", ""])
    for surface, count in metadata.get("surface_counts", {}).items():
        lines.append(f"- `{surface}`: {count}")

    lines.extend(["", "## Semantic Level Counts", ""])
    level_descriptions = metadata.get("semantic_level_descriptions", {})
    semantic_level_counts = metadata.get("semantic_level_counts", {})
    if semantic_level_counts:
        for level, count in semantic_level_counts.items():
            description = level_descriptions.get(str(level), "")
            suffix = f" - {description}" if description else ""
            lines.append(f"- level `{level}`: {count}{suffix}")
    else:
        lines.append("- none")

    semantic_status = metadata.get("semantic_level_status_counts", {})
    if semantic_status:
        lines.extend(["", "By status:", ""])
        for level, counts in semantic_status.items():
            summary = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
            lines.append(f"- level `{level}`: {summary}")

    excluded_by_category = Counter()
    terminal_by_status = Counter()
    for entry in audit["entries"]:
        if entry["status"] in EXCLUDED_STATUSES | PENDING_STATUSES:
            terminal_by_status[entry["status"]] += 1
        if entry["status"] in EXCLUDED_STATUSES | PENDING_STATUSES and entry.get("exclusion"):
            excluded_by_category[entry["exclusion"]["category"]] += 1
    lines.extend(["", "## Pending And Excluded Surfaces", ""])
    if terminal_by_status:
        for status, count in sorted(terminal_by_status.items()):
            lines.append(f"- `{status}`: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "By original exclusion category:", ""])
    if excluded_by_category:
        for category, count in sorted(excluded_by_category.items()):
            lines.append(f"- `{category}`: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "By blocker type:", ""])
    pending_blockers = metadata.get("pending_blocker_counts", {})
    if pending_blockers:
        for blocker, count in pending_blockers.items():
            lines.append(f"- `{blocker}`: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "By backend gate:", ""])
    pending_gates = metadata.get("pending_backend_gate_counts", {})
    if pending_gates:
        for gate, count in pending_gates.items():
            lines.append(f"- `{gate}`: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "By dtype contract source/probe mismatch:", ""])
    mismatch_counts = metadata.get("dtype_contract_mismatch_counts", {})
    if mismatch_counts:
        for mismatch, count in mismatch_counts.items():
            lines.append(f"- `{mismatch}`: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Coverage Kind Counts", ""])
    coverage_kind_counts = metadata.get("coverage_kind_counts", {})
    if coverage_kind_counts:
        for kind, count in coverage_kind_counts.items():
            lines.append(f"- `{kind}`: {count}")
    else:
        lines.append("- none")
    generated_by_family = Counter()
    for entry in audit["entries"]:
        if entry["status"] != "covered_generated":
            continue
        strategy = (entry.get("generated") or {}).get("strategy") or {}
        family = strategy.get("strategy") or entry["surface_kind"]
        generated_by_family[f"{entry['surface_kind']}:{family}"] += 1
    lines.extend(["", "## Generated Coverage Count By Family", ""])
    if generated_by_family:
        for family, count in sorted(generated_by_family.items()):
            lines.append(f"- `{family}`: {count}")
    else:
        lines.append("- none")

    case_depth = metadata.get("generated_case_depth", {})
    lines.extend(["", "## Generated Sample Case Depth", ""])
    lines.append(case_depth.get(
        "note",
        "Counts semantic case families before dtype, device, and IEEE754 input-condition expansion.",
    ))
    lines.append("")
    lines.append(f"- generated surfaces with case plans: {case_depth.get('generated_surfaces_with_case_plan', 0)}")
    lines.append(f"- generated semantic cases: {case_depth.get('generated_semantic_cases', 0)}")
    lines.append(f"- required generated semantic cases: {case_depth.get('required_generated_semantic_cases', 0)}")
    lines.append(f"- optional generated semantic cases: {case_depth.get('optional_generated_semantic_cases', 0)}")
    by_strategy = case_depth.get("by_strategy", {})
    if by_strategy:
        lines.append("")
        lines.append("By strategy:")
        for strategy, count in sorted(by_strategy.items()):
            lines.append(f"- `{strategy}`: {count}")
    by_level = case_depth.get("by_semantic_level", {})
    if by_level:
        lines.append("")
        lines.append("By semantic level:")
        for level, count in sorted(by_level.items(), key=lambda item: int(item[0])):
            lines.append(f"- level `{level}`: {count}")

    unknown_by_surface = Counter(
        entry["surface_kind"] for entry in audit["entries"] if entry["status"] == "unknown"
    )
    unknown_by_base = Counter(
        entry["base_name"] for entry in audit["entries"] if entry["status"] == "unknown"
    )
    lines.extend(["", "## Top Unknown Families", ""])
    if unknown_by_surface:
        lines.append("By surface kind:")
        for surface, count in unknown_by_surface.most_common(20):
            lines.append(f"- `{surface}`: {count}")
        lines.append("")
        lines.append("By dispatcher base name:")
        for base_name, count in unknown_by_base.most_common(20):
            lines.append(f"- `{base_name}`: {count}")
    else:
        lines.append("- none")

    capability_counts = Counter()
    for marker in audit.get("coverage_markers", []):
        capabilities = marker.get("capabilities") or ["uncategorized"]
        for capability in capabilities:
            capability_counts[capability] += 1
    lines.extend(["", "## Per-Capability Marker Summary", ""])
    if capability_counts:
        for capability, count in sorted(capability_counts.items()):
            lines.append(f"- `{capability}`: {count}")
    else:
        lines.append("- no coverage markers with capability metadata")

    unmapped = audit.get("unmapped_tests", [])
    lines.extend(["", "## Unmapped Hand-Authored Tests", ""])
    lines.append(f"Unmapped tests: {len(unmapped)}")
    for item in unmapped[:20]:
        lines.append(f"- `{item['nodeid']}` - {item['reason']}")
    if len(unmapped) > 20:
        lines.append(f"- ... {len(unmapped) - 20} more")

    lines.extend(["", "## Warnings", ""])
    if audit.get("warnings"):
        for warning in audit["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")
    return "\n".join(lines).rstrip() + "\n"


def render_semantic_levels_markdown(audit: dict) -> str:
    metadata = audit["metadata"]
    descriptions = metadata.get("semantic_level_descriptions", {})
    lines = [
        "# TorchCTS Semantic Levels",
        "",
        f"PyTorch: `{metadata.get('pytorch_version', 'unknown')}`",
        "",
        "## Level Policy",
        "",
    ]
    for level in range(1, 9):
        lines.append(f"- level `{level}`: {descriptions.get(str(level), semantic_level_description(level))}")

    lines.extend(["", "## Counts By Status", ""])
    status_counts = metadata.get("semantic_level_status_counts", {})
    if status_counts:
        for level, counts in status_counts.items():
            summary = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
            lines.append(f"- level `{level}`: {summary}")
    else:
        lines.append("- none")

    lines.extend(["", "## Counts By Surface", ""])
    surface_counts = metadata.get("semantic_level_surface_counts", {})
    if surface_counts:
        for level, counts in surface_counts.items():
            summary = ", ".join(f"{surface}={count}" for surface, count in sorted(counts.items()))
            lines.append(f"- level `{level}`: {summary}")
    else:
        lines.append("- none")

    lines.extend(["", "## Surface Assignments", ""])
    relevant_entries = [
        entry for entry in audit.get("entries", [])
        if entry.get("status") != "not_backend_relevant"
    ]
    if not relevant_entries:
        lines.append("- none")
        return "\n".join(lines).rstrip() + "\n"

    lines.extend([
        "| Level | Status | Surface | Dispatcher | Source | Reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for entry in sorted(
        relevant_entries,
        key=lambda item: (
            item.get("semantic_level") if item.get("semantic_level") is not None else 99,
            item.get("status", ""),
            item.get("surface_kind", ""),
            item.get("name", ""),
        ),
    ):
        reason = str(entry.get("level_reason") or "").replace("|", "\\|")
        lines.append(
            "| "
            f"{entry.get('semantic_level')} | "
            f"`{entry.get('status')}` | "
            f"`{entry.get('surface_kind')}` | "
            f"`{entry.get('name')}` | "
            f"`{entry.get('level_source')}` | "
            f"{reason} |"
        )

    return "\n".join(lines).rstrip() + "\n"


def _generated_case_entry(entry: dict) -> dict:
    generated = entry.get("generated", {})
    case = {
        "name": entry["name"],
        "base_name": entry["base_name"],
        "overload": entry.get("overload", ""),
        "schema": entry.get("schema", ""),
        "args": entry.get("args", []),
        "returns": entry.get("returns", []),
        "surface_kind": entry.get("surface_kind"),
        "variant_kind": entry.get("variant_kind"),
        "status": entry.get("status"),
        "semantic_level": entry.get("semantic_level"),
        "semantic_levels": entry.get("semantic_levels", []),
        "min_semantic_level": entry.get("min_semantic_level"),
        "max_semantic_level": entry.get("max_semantic_level"),
        "level_reason": entry.get("level_reason"),
        "level_source": entry.get("level_source"),
        "coverage_kind": entry.get("coverage_kind"),
    }
    if entry.get("oracle"):
        case["oracle"] = dict(entry["oracle"])
    if generated.get("strategy"):
        case["generated"] = {"strategy": generated["strategy"]}
        if generated.get("case_depth"):
            case["generated"]["case_depth"] = generated["case_depth"]
    if entry.get("exclusion"):
        case["exclusion"] = {
            "category": entry["exclusion"].get("category"),
            "reason": entry["exclusion"].get("reason"),
        }
    return case


def build_generated_cases_manifest(audit: dict) -> dict:
    cases_by_surface = defaultdict(list)
    included_statuses = {
        "unknown",
        "excluded",
        "unavailable_in_pytorch_runtime",
        "covered_generated",
        "covered_oracle",
        "covered_backend_pack",
        "covered_property",
        "pending_oracle",
        "pending_backend_pack",
        "pending_property",
    }
    for entry in audit.get("entries", []):
        if entry.get("status") not in included_statuses:
            continue
        surface_kind = entry.get("surface_kind")
        if surface_kind == "not_backend_relevant":
            continue
        cases_by_surface[surface_kind].append(_generated_case_entry(entry))

    sorted_cases = {
        surface: sorted(cases, key=lambda item: item["name"])
        for surface, cases in sorted(cases_by_surface.items())
    }
    return {
        "metadata": {
            "pytorch_version": audit["metadata"]["pytorch_version"],
            "generated_at": audit["metadata"]["generated_at"],
            "source_audit": str(DEFAULT_AUDIT_PATH),
            "case_count": sum(len(cases) for cases in sorted_cases.values()),
            "semantic_level_counts": audit["metadata"].get("semantic_level_counts", {}),
            "semantic_level_descriptions": audit["metadata"].get("semantic_level_descriptions", {}),
        },
        "cases_by_surface": sorted_cases,
    }


def render_generated_cases_module(manifest: dict) -> str:
    rendered = pformat(manifest, width=120, sort_dicts=True)
    return (
        "# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>\n"
        "#\n"
        "# Auto-generated by `python -m torchcts coverage materialize`.\n"
        "# Do not edit this file by hand; update coverage markers/strategies and rematerialize.\n"
        "\n"
        "from __future__ import annotations\n"
        "\n"
        f"GENERATED_CASES = {rendered}\n"
    )


def write_generated_cases_artifacts(audit: dict, *, write_module: bool = False) -> dict:
    _ensure_output_dir()
    manifest = build_generated_cases_manifest(audit)
    DEFAULT_GENERATED_CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_GENERATED_CASES_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if write_module:
        DEFAULT_GENERATED_CASES_MODULE_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_GENERATED_CASES_MODULE_PATH.write_text(render_generated_cases_module(manifest), encoding="utf-8")
    return manifest


def write_audit_artifacts(audit: dict) -> None:
    _ensure_output_dir()
    DEFAULT_AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    inventory = {
        "metadata": {
            "pytorch_version": audit["metadata"]["pytorch_version"],
            "generated_at": audit["metadata"]["generated_at"],
            "total_aten_overloads": audit["metadata"]["total_aten_overloads"],
            "surface_counts": audit["metadata"]["surface_counts"],
        },
        "entries": [
            {key: value for key, value in entry.items() if key not in ("opinfo", "handwritten", "generated", "exclusion", "status")}
            for entry in audit["entries"]
        ],
    }
    DEFAULT_INVENTORY_PATH.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    write_generated_cases_artifacts(audit)
    DEFAULT_UNKNOWNS_PATH.write_text(render_unknowns_markdown(audit), encoding="utf-8")
    DEFAULT_UNMAPPED_TESTS_PATH.write_text(render_unmapped_tests_markdown(audit), encoding="utf-8")
    DEFAULT_SUMMARY_PATH.write_text(render_summary_markdown(audit), encoding="utf-8")
    DEFAULT_SEMANTIC_LEVELS_PATH.write_text(render_semantic_levels_markdown(audit), encoding="utf-8")
    DEFAULT_PENDING_REVIEW_PATH.write_text(
        json.dumps(build_pending_review_artifact(audit), indent=2),
        encoding="utf-8",
    )
    DEFAULT_PENDING_REVIEW_MD_PATH.write_text(render_pending_review_markdown(audit), encoding="utf-8")


def run_inventory_command() -> int:
    inventory = write_inventory()
    print(f"Wrote coverage inventory: {DEFAULT_INVENTORY_PATH}")
    print(f"ATen overloads inventoried: {inventory['metadata']['total_aten_overloads']}")
    return 0


def _print_audit_warning(audit: dict) -> None:
    unknown_count = audit["metadata"]["unknown_count"]
    if unknown_count:
        print(
            "\n"
            f"WARNING: TorchCTS coverage audit found {unknown_count} unknown tensor-touching ATen surfaces.\n"
            "These surfaces are not counted as covered and generated tests will skip them.\n"
            f"See {DEFAULT_UNKNOWNS_PATH}.\n"
        )


def run_audit_command() -> int:
    audit = build_audit()
    write_audit_artifacts(audit)
    print(f"Wrote coverage audit: {DEFAULT_AUDIT_PATH}")
    print(f"Wrote coverage summary: {DEFAULT_SUMMARY_PATH}")
    print(f"Wrote semantic-level review: {DEFAULT_SEMANTIC_LEVELS_PATH}")
    print(f"Wrote pending coverage review: {DEFAULT_PENDING_REVIEW_PATH}")
    _print_audit_warning(audit)
    if audit["errors"]:
        for error in audit["errors"]:
            print(f"Error: {error}")
        return 1
    for warning in audit["warnings"]:
        if "unknown tensor-touching" not in warning:
            print(f"Warning: {warning}")
    return 0


def _load_or_build_default_audit() -> dict:
    if DEFAULT_AUDIT_PATH.exists():
        try:
            audit = json.loads(DEFAULT_AUDIT_PATH.read_text(encoding="utf-8"))
            metadata = audit.get("metadata", {})
            if "generated_case_depth" in metadata and "semantic_level_counts" in metadata:
                return audit
        except Exception:
            pass
    audit = build_audit()
    write_audit_artifacts(audit)
    return audit


def _build_and_write_default_audit() -> dict:
    audit = build_audit()
    write_audit_artifacts(audit)
    return audit


def run_report_command() -> int:
    audit = _build_and_write_default_audit()
    summary = render_summary_markdown(audit)
    _ensure_output_dir()
    DEFAULT_SUMMARY_PATH.write_text(summary, encoding="utf-8")
    DEFAULT_SEMANTIC_LEVELS_PATH.write_text(render_semantic_levels_markdown(audit), encoding="utf-8")
    DEFAULT_PENDING_REVIEW_PATH.write_text(
        json.dumps(build_pending_review_artifact(audit), indent=2),
        encoding="utf-8",
    )
    DEFAULT_PENDING_REVIEW_MD_PATH.write_text(render_pending_review_markdown(audit), encoding="utf-8")
    print(summary)
    return 0


def run_materialize_command() -> int:
    audit = _build_and_write_default_audit()
    if audit.get("errors"):
        for error in audit["errors"]:
            print(f"Error: {error}")
        return 1
    manifest = write_generated_cases_artifacts(audit, write_module=True)
    print(f"Wrote generated cases JSON: {DEFAULT_GENERATED_CASES_PATH}")
    print(f"Wrote generated cases module: {DEFAULT_GENERATED_CASES_MODULE_PATH}")
    print(f"Generated case entries: {manifest['metadata']['case_count']}")
    for surface, cases in manifest["cases_by_surface"].items():
        print(f"  {surface}: {len(cases)}")
    return 0


def _validate_audit_consistency(audit: dict) -> list[str]:
    errors = []
    seen_names = set()
    for index, entry in enumerate(audit.get("entries", [])):
        name = entry.get("name")
        if not name:
            errors.append(f"entries[{index}] is missing dispatcher name")
        elif name in seen_names:
            errors.append(f"entries[{index}] duplicates dispatcher name {name}")
        else:
            seen_names.add(name)

        status = entry.get("status")
        if status not in FINAL_STATUSES:
            errors.append(f"entries[{index}] has invalid status {status!r}")
        if entry.get("surface_kind") != "not_backend_relevant" and status == "not_backend_relevant":
            errors.append(f"entries[{index}] {entry.get('name')} is marked not_backend_relevant inconsistently")
        if (entry.get("has_tensor_args") or entry.get("has_tensor_returns")) and status == "not_backend_relevant":
            errors.append(f"entries[{index}] {entry.get('name')} touches tensors but is marked not_backend_relevant")
        if status in EXCLUDED_STATUSES | PENDING_STATUSES and not entry.get("exclusion"):
            errors.append(f"entries[{index}] {entry.get('name')} is {status} without exclusion metadata")
        if status in EXCLUDED_STATUSES | PENDING_STATUSES:
            review = entry.get("pending_review")
            if not isinstance(review, dict):
                errors.append(f"entries[{index}] {entry.get('name')} is {status} without pending_review metadata")
            else:
                blocker = review.get("blocker_type")
                if blocker not in PENDING_BLOCKER_TYPES:
                    errors.append(
                        f"entries[{index}] {entry.get('name')} has invalid blocker_type {blocker!r}"
                    )
                if not review.get("required_closure"):
                    errors.append(f"entries[{index}] {entry.get('name')} pending_review is missing required_closure")
                if not review.get("backend_gate"):
                    errors.append(f"entries[{index}] {entry.get('name')} pending_review is missing backend_gate")
                if not review.get("next_family"):
                    errors.append(f"entries[{index}] {entry.get('name')} pending_review is missing next_family")
        if status == "covered_handwritten" and not entry.get("handwritten", {}).get("covered"):
            errors.append(f"entries[{index}] {entry.get('name')} is covered_handwritten without marker metadata")
        if status == "covered_generated" and not entry.get("generated", {}).get("covered"):
            errors.append(f"entries[{index}] {entry.get('name')} is covered_generated without marker metadata")
        if status in {"covered_oracle", "covered_backend_pack", "covered_property"} and not entry.get("oracle"):
            errors.append(f"entries[{index}] {entry.get('name')} is {status} without oracle metadata")
        if status != "not_backend_relevant":
            level = entry.get("semantic_level")
            try:
                validate_semantic_level(level)
            except Exception as exc:
                errors.append(f"entries[{index}] {entry.get('name')} has invalid semantic_level {level!r}: {exc}")
            levels = entry.get("semantic_levels")
            if not isinstance(levels, list) or not levels:
                errors.append(f"entries[{index}] {entry.get('name')} is missing semantic_levels")
            else:
                for level_value in levels:
                    try:
                        validate_semantic_level(level_value)
                    except Exception as exc:
                        errors.append(
                            f"entries[{index}] {entry.get('name')} has invalid semantic_levels value "
                            f"{level_value!r}: {exc}"
                        )
                if level is not None and level not in levels:
                    errors.append(f"entries[{index}] {entry.get('name')} semantic_level is not included in semantic_levels")
            if not entry.get("level_reason"):
                errors.append(f"entries[{index}] {entry.get('name')} is missing level_reason")
            if not entry.get("level_source"):
                errors.append(f"entries[{index}] {entry.get('name')} is missing level_source")
        generated = entry.get("generated", {})
        generated_strategy = (generated or {}).get("strategy")
        generated_depth = (generated or {}).get("case_depth") or {}
        if generated_strategy and generated_depth.get("error"):
            errors.append(f"entries[{index}] {entry.get('name')} has invalid generated case depth: {generated_depth['error']}")
        if generated_strategy and int(generated_depth.get("planned_count", 0) or 0) <= 0:
            errors.append(f"entries[{index}] {entry.get('name')} has a generated strategy but no planned sample cases")
        if generated_strategy:
            for case_index, case in enumerate(generated_depth.get("cases", [])):
                try:
                    validate_semantic_level(case.get("semantic_level"))
                except Exception as exc:
                    errors.append(
                        f"entries[{index}] {entry.get('name')} generated case {case_index} "
                        f"has invalid semantic_level {case.get('semantic_level')!r}: {exc}"
                    )
                if not case.get("level_reason"):
                    errors.append(
                        f"entries[{index}] {entry.get('name')} generated case {case_index} is missing level_reason"
                    )
                if not case.get("level_source"):
                    errors.append(
                        f"entries[{index}] {entry.get('name')} generated case {case_index} is missing level_source"
                    )
        if status == "covered_opinfo" and not entry.get("opinfo", {}).get("covered"):
            errors.append(f"entries[{index}] {entry.get('name')} is covered_opinfo without OpInfo metadata")
    return errors


def run_check_command(strict_unknowns: bool = False) -> int:
    audit = _build_and_write_default_audit()
    errors = list(audit.get("errors", []))
    errors.extend(_validate_audit_consistency(audit))
    if strict_unknowns and audit["metadata"]["unknown_count"]:
        errors.append(f"strict_unknowns enabled with {audit['metadata']['unknown_count']} unknown surfaces")
    if errors:
        for error in errors:
            print(f"Error: {error}")
        return 1
    _print_audit_warning(audit)
    print("Coverage audit is internally consistent.")
    return 0


def _load_generated_cases_manifest() -> dict | None:
    if DEFAULT_GENERATED_CASES_MODULE_PATH.exists():
        try:
            namespace = runpy.run_path(str(DEFAULT_GENERATED_CASES_MODULE_PATH))
            manifest = namespace.get("GENERATED_CASES")
            if isinstance(manifest, dict):
                return manifest
        except Exception:
            pass
    if DEFAULT_GENERATED_CASES_PATH.exists():
        try:
            manifest = json.loads(DEFAULT_GENERATED_CASES_PATH.read_text(encoding="utf-8"))
            if isinstance(manifest, dict):
                return manifest
        except Exception:
            pass
    return None


def generated_entries_for(
    surface_kind: str,
    audit: dict | None = None,
    build_if_missing: bool = False,
) -> list[dict]:
    if audit is None:
        manifest = _load_generated_cases_manifest()
        if manifest is not None:
            return list(manifest.get("cases_by_surface", {}).get(surface_kind, []))
    if audit is None:
        if DEFAULT_AUDIT_PATH.exists():
            try:
                audit = json.loads(DEFAULT_AUDIT_PATH.read_text(encoding="utf-8"))
            except Exception:
                audit = None
        if audit is None and build_if_missing:
            audit = _load_or_build_default_audit()
        if audit is None:
            return []
    return [
        entry for entry in audit.get("entries", [])
        if entry.get("surface_kind") == surface_kind
        and entry.get("status") in {"unknown", "excluded", "covered_generated", "unavailable_in_pytorch_runtime"}
    ]
