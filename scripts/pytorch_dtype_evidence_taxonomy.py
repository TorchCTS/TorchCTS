#!/usr/bin/env python3
"""Semantic storage taxonomy for PyTorch dtype-contract evidence.

This taxonomy is deliberately independent from ``torchcts/op_metadata.json``'s
runtime-facing categories.  Existing metadata is an input hint, not the source
of truth for the physical evidence layout.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OP_METADATA_PATH = REPO_ROOT / "torchcts" / "op_metadata.json"
TAXONOMY_VERSION = 1

CATEGORIES = frozenset({
    "autograd/backward",
    "autograd/internal",
    "framework/backend",
    "framework/control-flow",
    "framework/dispatcher",
    "framework/distributed",
    "framework/metadata",
    "language/containers",
    "language/operator-protocol",
    "language/scalars",
    "linear-algebra/decomposition",
    "linear-algebra/matmul",
    "math/comparison",
    "math/elementwise-binary",
    "math/elementwise-unary",
    "math/reduction",
    "math/special",
    "neural-network/attention",
    "neural-network/convolution",
    "neural-network/dropout",
    "neural-network/interpolation",
    "neural-network/loss",
    "neural-network/normalization",
    "neural-network/padding",
    "neural-network/pooling",
    "neural-network/rnn",
    "neural-network/softmax",
    "optimization/foreach",
    "quantization/core",
    "random/philox",
    "random/sampling",
    "sparse/core",
    "spectral/fft",
    "spectral/windows",
    "tensor/conversion",
    "tensor/copy",
    "tensor/creation",
    "tensor/indexing",
    "tensor/masked",
    "tensor/shape",
    "tensor/sorting",
    "tensor/utility",
})

METADATA_CATEGORY_MAP = {
    "attention": "neural-network/attention",
    "autograd_internal": "autograd/internal",
    "comparison": "math/comparison",
    "convolution": "neural-network/convolution",
    "copy": "tensor/copy",
    "creation": "tensor/creation",
    "dropout": "neural-network/dropout",
    "elementwise_binary": "math/elementwise-binary",
    "elementwise_unary": "math/elementwise-unary",
    "fft": "spectral/fft",
    "foreach_optimizer": "optimization/foreach",
    "indexing": "tensor/indexing",
    "interpolation": "neural-network/interpolation",
    "linalg": "linear-algebra/decomposition",
    "loss": "neural-network/loss",
    "matmul": "linear-algebra/matmul",
    "normalization": "neural-network/normalization",
    "padding": "neural-network/padding",
    "pooling": "neural-network/pooling",
    "quantization": "quantization/core",
    "random": "random/sampling",
    "reduction": "math/reduction",
    "rnn": "neural-network/rnn",
    "shape": "tensor/shape",
    "softmax": "neural-network/softmax",
    "sorting": "tensor/sorting",
    "sparse": "sparse/core",
}

# Exact exceptions are intentionally small and documented.  Keys are complete
# dispatcher/operator keys so adding an overload cannot silently inherit an
# exception intended for a different surface.
EXACT_OVERRIDES: dict[str, tuple[str, str]] = {
    "aten::_dimI": ("framework/metadata", "internal tensor dimension query"),
    "aten::_dimV": ("framework/metadata", "internal tensor dimension query"),
    "aten::_version": ("framework/metadata", "internal tensor version counter"),
    "aten::data": ("framework/metadata", "Tensor.data protocol surface"),
    "aten::torch": ("framework/dispatcher", "TorchScript builtin namespace value"),
}

NN_FUNCTIONAL_CATEGORIES = {
    "adaptive_avg_pool1d": "neural-network/pooling",
    "adaptive_avg_pool2d": "neural-network/pooling",
    "adaptive_avg_pool3d": "neural-network/pooling",
    "adaptive_max_pool1d": "neural-network/pooling",
    "adaptive_max_pool2d": "neural-network/pooling",
    "adaptive_max_pool3d": "neural-network/pooling",
    "alpha_dropout": "neural-network/dropout",
    "avg_pool1d": "neural-network/pooling",
    "avg_pool2d": "neural-network/pooling",
    "avg_pool3d": "neural-network/pooling",
    "batch_norm": "neural-network/normalization",
    "bilinear": "linear-algebra/matmul",
    "binary_cross_entropy": "neural-network/loss",
    "binary_cross_entropy_with_logits": "neural-network/loss",
    "channel_shuffle": "tensor/shape",
    "conv1d": "neural-network/convolution",
    "conv2d": "neural-network/convolution",
    "conv3d": "neural-network/convolution",
    "conv_transpose1d": "neural-network/convolution",
    "conv_transpose2d": "neural-network/convolution",
    "conv_transpose3d": "neural-network/convolution",
    "cosine_embedding_loss": "neural-network/loss",
    "cross_entropy": "neural-network/loss",
    "ctc_loss": "neural-network/loss",
    "dropout": "neural-network/dropout",
    "dropout2d": "neural-network/dropout",
    "dropout3d": "neural-network/dropout",
    "embedding": "tensor/indexing",
    "embedding_bag": "tensor/indexing",
    "feature_alpha_dropout": "neural-network/dropout",
    "fractional_max_pool2d": "neural-network/pooling",
    "fractional_max_pool3d": "neural-network/pooling",
    "gaussian_nll_loss": "neural-network/loss",
    "grid_sample": "neural-network/interpolation",
    "group_norm": "neural-network/normalization",
    "hinge_embedding_loss": "neural-network/loss",
    "huber_loss": "neural-network/loss",
    "instance_norm": "neural-network/normalization",
    "interpolate": "neural-network/interpolation",
    "kl_div": "neural-network/loss",
    "l1_loss": "neural-network/loss",
    "layer_norm": "neural-network/normalization",
    "linear": "linear-algebra/matmul",
    "local_response_norm": "neural-network/normalization",
    "margin_ranking_loss": "neural-network/loss",
    "max_pool1d": "neural-network/pooling",
    "max_pool2d": "neural-network/pooling",
    "max_pool3d": "neural-network/pooling",
    "max_unpool1d": "neural-network/pooling",
    "max_unpool2d": "neural-network/pooling",
    "max_unpool3d": "neural-network/pooling",
    "mse_loss": "neural-network/loss",
    "multi_head_attention_forward": "neural-network/attention",
    "multi_margin_loss": "neural-network/loss",
    "multilabel_margin_loss": "neural-network/loss",
    "multilabel_soft_margin_loss": "neural-network/loss",
    "nll_loss": "neural-network/loss",
    "normalize": "neural-network/normalization",
    "one_hot": "tensor/conversion",
    "pad": "neural-network/padding",
    "pixel_shuffle": "tensor/shape",
    "pixel_unshuffle": "tensor/shape",
    "poisson_nll_loss": "neural-network/loss",
    "rms_norm": "neural-network/normalization",
    "scaled_dot_product_attention": "neural-network/attention",
    "smooth_l1_loss": "neural-network/loss",
    "soft_margin_loss": "neural-network/loss",
    "softmin": "neural-network/softmax",
    "triplet_margin_loss": "neural-network/loss",
    "triplet_margin_with_distance_loss": "neural-network/loss",
    "unfold": "tensor/shape",
    "upsample_bilinear": "neural-network/interpolation",
    "upsample_nearest": "neural-network/interpolation",
}

NN_FUNCTIONAL_ACTIVATIONS = frozenset({
    "celu", "elu", "gelu", "glu", "hardshrink", "hardsigmoid",
    "hardswish", "hardtanh", "leaky_relu", "logsigmoid", "mish",
    "prelu", "relu", "relu6", "rrelu", "selu", "silu", "softplus",
    "softshrink", "softsign", "tanhshrink", "threshold",
})

BACKWARD_MARKERS = (
    "backward", "_grad", "_jvp", "jvp", "vjp", "grad_input",
    "grad_weight", "grad_bias",
)

FRAMEWORK_METADATA_FAMILIES = frozenset({
    "_debug_has_internal_overlap", "_has_compatible_shallow_copy_type",
    "_is_zerotensor", "_local_scalar_dense", "dense_dim", "dim",
    "is_complex", "is_conj", "is_contiguous", "is_floating_point",
    "is_inference", "is_leaf", "is_neg", "is_pinned", "is_same_size",
    "is_set_to", "is_signed", "item", "numel", "output_nr", "retains_grad",
    "size", "sym_is_contiguous", "sym_numel", "sym_size",
})

CONVERSION_FAMILIES = frozenset({
    "Complex", "_cast_Byte", "_cast_Char", "_cast_Double", "_cast_Float",
    "_cast_Half", "_cast_Int", "_cast_Long", "_cast_Short", "bfloat16",
    "bool", "byte", "cdouble", "cfloat", "chalf", "char", "complex",
    "double", "float", "half", "int", "long", "one_hot", "short", "to",
    "to_dense", "type_as", "view_as_complex", "view_as_real",
})

COPY_FAMILY_MARKERS = (
    "copy", "clone", "detach", "lift", "resolve_conj", "resolve_neg",
)

CREATION_FAMILIES = frozenset({
    "_dim_arange", "_efficientzerotensor", "_new_zeros_with_same_feature_meta",
    "arange", "empty", "eye", "full", "linspace", "logspace",
    "new_empty", "new_full", "new_ones", "new_zeros", "ones", "range",
    "scalar_tensor", "signal_window", "tensor", "tril_indices", "triu_indices",
    "zeros",
})

WINDOW_FAMILIES = frozenset({
    "bartlett_window", "blackman_window", "hamming_window", "hann_window",
    "kaiser_window",
})

INDEX_MARKERS = (
    "index", "scatter", "gather", "select", "slice", "take",
    "bucketize", "searchsorted", "embedding", "masked", "unravel",
)

SHAPE_MARKERS = (
    "reshape", "view", "flatten", "unflatten", "squeeze", "unsqueeze",
    "transpose", "permute", "movedim", "swap", "split", "chunk", "stack",
    "cat", "concat", "broadcast", "repeat", "ravel", "atleast",
    "meshgrid", "diag_embed", "diagflat", "diagonal_scatter", "pixel_shuffle",
    "pixel_unshuffle", "channel_shuffle", "unbind", "resize", "as_strided",
)

SORT_MARKERS = (
    "sort", "topk", "kth", "unique", "argsort", "argwhere", "bincount",
    "histc", "histogram", "quantile", "median", "mode",
)

MATMUL_FAMILIES = frozenset({
    "_addmm_activation", "addbmm", "addmm", "addmv", "addr", "baddbmm",
    "bilinear", "block_diag", "chain_matmul", "dot", "ger", "inner", "kron",
    "linear", "linalg_matmul", "matmul", "mm", "mv", "outer", "vdot",
})

LINALG_FAMILIES = frozenset({
    "cholesky", "cholesky_inverse", "cholesky_solve", "corrcoef", "cov",
    "cross", "einsum", "frobenius_norm", "householder_product", "lu",
    "matrix_exp", "matrix_power", "nuclear_norm", "pca_lowrank", "pinv", "qr",
    "solve", "svd", "svd_lowrank", "vander",
})

REDUCTION_FAMILIES = frozenset({
    "_aminmax", "_is_all_true", "_is_any_true", "_segment_reduce", "all",
    "amax", "amin", "aminmax", "any", "argmax", "argmin", "count_nonzero",
    "cummax", "cummin", "cumprod", "cumsum", "dist", "logsumexp", "max",
    "mean", "min", "nanmean", "nanmedian", "nansum", "norm", "prod",
    "segment_reduce", "std", "sum", "var",
})

COMPARISON_MARKERS = (
    "allclose", "equal", "isclose", "isin", "greater", "less", "logical",
)

BINARY_FAMILIES = frozenset({
    "__iand__", "__ilshift__", "__ior__", "__irshift__", "__ixor__",
    "__lshift__", "__radd__", "__rand__", "__rdiv__", "__rmod__", "__rmul__",
    "__ror__", "__rpow__", "__rshift__", "__rsub__", "__rxor__", "add",
    "addcdiv", "addcmul", "arctan2", "bitwise_and", "bitwise_left_shift",
    "bitwise_or", "bitwise_right_shift", "bitwise_xor", "clamp_max", "clamp_min",
    "divide", "float_power", "igamma", "igammac", "ldexp", "lerp", "logaddexp",
    "logaddexp2", "maximum", "minimum", "multiply", "nextafter", "pow", "rsub",
    "subtract", "true_divide", "xlogy",
})

UNARY_FAMILIES = frozenset({
    "_add_relu", "_add_relu_", "_conj_physical", "absolute", "absolute_", "angle",
    "arccos", "arccos_", "arccosh", "arccosh_", "arcsin", "arcsin_", "arcsinh",
    "arcsinh_", "arctan", "arctan_", "arctanh", "arctanh_", "conj", "deg2rad",
    "deg2rad_", "fix", "fix_", "frexp", "glu", "imag", "nan_to_num", "nan_to_num_",
    "negative", "negative_", "polar", "positive", "rad2deg", "rad2deg_", "real",
    "relu6", "relu6_", "rrelu_with_noise", "sinc", "sinc_", "zero", "zero_",
})

RANDOM_MARKERS = (
    "random", "rand", "normal", "uniform", "cauchy", "exponential", "geometric",
    "binomial", "dirichlet", "sobol", "multinomial", "bernoulli", "poisson",
)

RANDOM_FAMILIES = frozenset({"_standard_gamma"})

SPECIAL_FAMILIES = frozenset({"gamma", "igamma", "igammac"})

UTILITY_FAMILIES = frozenset({
    "_compute_linear_combination",
    "alias", "cartesian_prod",
    "combinations", "cumulative_trapezoid", "diff", "fill", "fill_", "gradient",
    "hash_tensor", "result_type", "set", "set_", "set_data", "trapezoid", "trapz",
})


@dataclass(frozen=True)
class Classification:
    op: str
    canonical_family: str
    metadata_category: str | None
    rule_id: str
    category: str
    exact_override: bool = False


def load_metadata_categories(path: Path = OP_METADATA_PATH) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    ops = data.get("ops") if isinstance(data, dict) else None
    if not isinstance(ops, dict):
        raise ValueError(f"{path} does not contain an ops object")
    return {
        str(op): str(entry["category"])
        for op, entry in ops.items()
        if isinstance(entry, dict) and isinstance(entry.get("category"), str)
    }


def canonical_family(op: str) -> str:
    if "::" not in op:
        raise ValueError(f"Expected dispatcher-style operator key, got {op!r}")
    name = op.split("::", 1)[1]
    if name.startswith("torch.ops.aten."):
        name = name.removeprefix("torch.ops.aten.")
    for namespace in ("nn.functional.", "signal.windows.", "linalg.", "fft.", "special.", "masked."):
        if name.startswith(namespace):
            return name
    return name.split(".", 1)[0]


def category_path(category: str) -> str:
    if category not in CATEGORIES:
        raise ValueError(f"Unknown dtype evidence category {category!r}")
    return f"{category}.jsonl"


def _contains_any(value: str, markers: tuple[str, ...]) -> bool:
    return any(marker in value for marker in markers)


def _namespace_category(family: str) -> tuple[str, str] | None:
    if family.startswith("signal.windows."):
        return "spectral/windows", "namespace:signal.windows"
    if family in WINDOW_FAMILIES:
        return "spectral/windows", "namespace:signal.windows-alias"
    if family.startswith("fft.") or family.startswith("fft_"):
        return "spectral/fft", "namespace:fft"
    if family.startswith("special.") or family.startswith("special_"):
        return "math/special", "namespace:special"
    if family.startswith("masked.") or family.startswith("masked_") or family.startswith("_masked_"):
        return "tensor/masked", "namespace:masked"
    if family.startswith("linalg.") or family.startswith("linalg_"):
        leaf = family.rsplit(".", 1)[-1]
        if leaf in {"matmul", "multi_dot", "vecdot"} or family == "linalg_matmul":
            return "linear-algebra/matmul", "namespace:linalg-matmul"
        return "linear-algebra/decomposition", "namespace:linalg"
    if family.startswith("_foreach_") or "_foreach_" in family:
        return "optimization/foreach", "namespace:foreach"
    if "sparse" in family:
        return "sparse/core", "namespace:sparse"
    if family == "coalesce":
        return "sparse/core", "namespace:sparse"
    if "quantiz" in family or family.startswith("q_"):
        return "quantization/core", "namespace:quantization"
    if family.startswith("nn.functional."):
        function = family.split(".")[-1]
        category = NN_FUNCTIONAL_CATEGORIES.get(function)
        if category:
            return category, f"namespace:nn.functional.{function}"
        if function in NN_FUNCTIONAL_ACTIVATIONS:
            return "math/elementwise-unary", f"namespace:nn.functional.{function}"
        if function in {"cosine_similarity", "pairwise_distance", "pdist"}:
            return "math/reduction", f"namespace:nn.functional.{function}"
        raise ValueError(f"Unclassified nn.functional family {family!r}")
    return None


def _framework_category(family: str) -> tuple[str, str] | None:
    lowered = family.lower()
    if lowered.startswith("_assert") or lowered.startswith("assert_") or lowered.startswith("sym_constrain") or lowered == "is_nonzero":
        return "framework/control-flow", "framework:control-flow"
    if _contains_any(lowered, ("all_gather", "all_reduce", "reduce_scatter", "distributed")):
        return "framework/distributed", "framework:distributed"
    if _contains_any(lowered, ("cudnn", "mkldnn", "mps", "xpu", "miopen", "onednn")):
        return "framework/backend", "framework:backend"
    if family in FRAMEWORK_METADATA_FAMILIES:
        return "framework/metadata", "framework:metadata"
    return None


def _semantic_category(family: str) -> tuple[str, str] | None:
    lowered = family.lower()
    comparable = family[:-1] if family.endswith("_") and not family.endswith("__") else family
    comparable_lowered = comparable.lower()
    if comparable in CONVERSION_FAMILIES or lowered.startswith("_cast_"):
        return "tensor/conversion", "semantic:conversion"
    if _contains_any(lowered, COPY_FAMILY_MARKERS):
        return "tensor/copy", "semantic:copy"
    if comparable in CREATION_FAMILIES or comparable_lowered.endswith("_like"):
        return "tensor/creation", "semantic:creation"
    if family == "__getitem__":
        return "language/containers", "semantic:container"
    if family.startswith("__"):
        return "language/operator-protocol", "semantic:operator-protocol"
    if family in {"Complex", "Scalar", "item"}:
        return "language/scalars", "semantic:scalar"
    if family in {"H", "T"}:
        return "tensor/shape", "semantic:shape"
    if _contains_any(lowered, ("attention", "flash_sdp", "efficient_attention")):
        return "neural-network/attention", "semantic:attention"
    if "conv" in lowered or lowered in {"thnn_conv2d"}:
        return "neural-network/convolution", "semantic:convolution"
    if "pool" in lowered:
        return "neural-network/pooling", "semantic:pooling"
    if "dropout" in lowered:
        return "neural-network/dropout", "semantic:dropout"
    if _contains_any(lowered, ("upsample", "interpolate", "grid_sample")):
        return "neural-network/interpolation", "semantic:interpolation"
    if "loss" in lowered:
        return "neural-network/loss", "semantic:loss"
    if _contains_any(lowered, ("batch_norm", "layer_norm", "group_norm", "instance_norm", "rms_norm", "native_norm")):
        return "neural-network/normalization", "semantic:normalization"
    if lowered.startswith("pad") or lowered.endswith("_pad"):
        return "neural-network/padding", "semantic:padding"
    if _contains_any(lowered, ("softmax", "log_sigmoid", "logsigmoid")):
        return "neural-network/softmax", "semantic:softmax"
    if _contains_any(lowered, ("rnn", "lstm", "gru")):
        return "neural-network/rnn", "semantic:rnn"
    if comparable in MATMUL_FAMILIES or family == "__rmatmul__":
        return "linear-algebra/matmul", "semantic:matmul"
    if comparable in LINALG_FAMILIES:
        return "linear-algebra/decomposition", "semantic:linalg"
    if family in {"stft", "istft"}:
        return "spectral/fft", "semantic:fft"
    if comparable in SPECIAL_FAMILIES:
        return "math/special", "semantic:special"
    if comparable in RANDOM_FAMILIES or _contains_any(lowered, RANDOM_MARKERS):
        category = "random/philox" if "philox" in lowered else "random/sampling"
        return category, "semantic:random"
    index_family = (
        comparable in {"bucketize", "diagonal_scatter", "index_fill", "put", "scatter", "scatter_reduce", "searchsorted", "select", "slice", "slice_inverse", "slice_scatter", "take_along_dim", "unravel_index"}
        or comparable.startswith(("_index_", "_unsafe_masked_index", "index_"))
        or _contains_any(lowered, tuple(marker for marker in INDEX_MARKERS if len(marker) >= 5))
    )
    if index_family:
        return "tensor/indexing", "semantic:indexing"
    if _contains_any(lowered, SHAPE_MARKERS) or family in {"_shape_as_tensor", "numpy_T"}:
        return "tensor/shape", "semantic:shape"
    if _contains_any(lowered, SORT_MARKERS):
        return "tensor/sorting", "semantic:sorting"
    if comparable in REDUCTION_FAMILIES or family in {"_cdist_forward", "_euclidean_dist", "_pdist_forward", "cdist", "pairwise_distance", "pdist"}:
        return "math/reduction", "semantic:reduction"
    if _contains_any(lowered, COMPARISON_MARKERS) or lowered.startswith("is_"):
        return "math/comparison", "semantic:comparison"
    if comparable in BINARY_FAMILIES:
        return "math/elementwise-binary", "semantic:elementwise-binary"
    if comparable in UNARY_FAMILIES:
        return "math/elementwise-unary", "semantic:elementwise-unary"
    if family in UTILITY_FAMILIES:
        return "tensor/utility", "exact-family:utility"
    return None


def classify_operator(op: str, metadata_categories: dict[str, str]) -> Classification:
    family = canonical_family(op)
    metadata_category = metadata_categories.get(op)

    override = EXACT_OVERRIDES.get(op)
    if override:
        category, _reason = override
        return Classification(op, family, metadata_category, "exact-override", category, True)

    lowered = family.lower()
    if _contains_any(lowered, BACKWARD_MARKERS):
        return Classification(op, family, metadata_category, "autograd:backward", "autograd/backward")

    framework = _framework_category(family)
    if framework:
        category, rule_id = framework
        return Classification(op, family, metadata_category, rule_id, category)

    namespace = _namespace_category(family)
    if namespace:
        category, rule_id = namespace
        return Classification(op, family, metadata_category, rule_id, category)

    mapped_metadata = METADATA_CATEGORY_MAP.get(metadata_category or "")
    if mapped_metadata:
        return Classification(op, family, metadata_category, f"metadata:{metadata_category}", mapped_metadata)

    semantic = _semantic_category(family)
    if semantic:
        category, rule_id = semantic
        return Classification(op, family, metadata_category, rule_id, category)

    raise ValueError(
        f"No dtype evidence taxonomy rule for {op!r} "
        f"(family={family!r}, metadata_category={metadata_category!r})"
    )


def classify_operators(
    ops: list[str] | tuple[str, ...] | set[str],
    metadata_categories: dict[str, str] | None = None,
) -> dict[str, Classification]:
    categories = metadata_categories if metadata_categories is not None else load_metadata_categories()
    results: dict[str, Classification] = {}
    for op in sorted(ops):
        classification = classify_operator(op, categories)
        if classification.category not in CATEGORIES:
            raise ValueError(f"{op}: classifier returned unknown category {classification.category!r}")
        results[op] = classification
    return results


def write_audit_report(path: Path, classifications: dict[str, Classification]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "canonical_family": item.canonical_family,
            "category": item.category,
            "exact_override": item.exact_override,
            "metadata_category": item.metadata_category,
            "op": item.op,
            "rule_id": item.rule_id,
        }
        for item in classifications.values()
    ]
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
