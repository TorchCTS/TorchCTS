"""Runner registry for curated path-shape cases."""

from torchcts.path_shapes.runners.attention import run_sdpa
from torchcts.path_shapes.runners.broadcasting import run_add, run_masked_fill, run_mul, run_where
from torchcts.path_shapes.runners.convolution import run_conv1d, run_conv2d, run_conv3d, run_conv_transpose2d
from torchcts.path_shapes.runners.fft import run_fft, run_fft2, run_irfft, run_rfft
from torchcts.path_shapes.runners.indexing import (
    run_gather,
    run_index_select,
    run_masked_select,
    run_scatter_add,
    run_scatter_reduce,
    run_take,
)
from torchcts.path_shapes.runners.linear_algebra import run_cholesky, run_eigh, run_qr, run_solve, run_svdvals
from torchcts.path_shapes.runners.matmul import run_addmm, run_bmm, run_linear, run_matmul, run_mm
from torchcts.path_shapes.runners.model_patterns import (
    run_depthwise_separable,
    run_patch_embedding,
    run_residual_norm,
    run_transformer_block_fragment,
    run_vision_block,
)
from torchcts.path_shapes.runners.normalization import run_batch_norm, run_group_norm, run_layer_norm
from torchcts.path_shapes.runners.reduction import run_amax, run_argmax, run_mean, run_prod, run_sum
from torchcts.path_shapes.runners.sorting import run_kthvalue, run_sort, run_topk
from torchcts.path_shapes.runners.spatial import run_adaptive_avg_pool2d, run_avg_pool2d, run_interpolate, run_max_pool2d


RUNNERS = {
    "matmul.mm": run_mm,
    "matmul.bmm": run_bmm,
    "matmul.matmul": run_matmul,
    "matmul.addmm": run_addmm,
    "matmul.linear": run_linear,
    "convolution.conv1d": run_conv1d,
    "convolution.conv2d": run_conv2d,
    "convolution.conv3d": run_conv3d,
    "convolution.conv_transpose2d": run_conv_transpose2d,
    "attention.sdpa": run_sdpa,
    "reduction.sum": run_sum,
    "reduction.mean": run_mean,
    "reduction.amax": run_amax,
    "reduction.argmax": run_argmax,
    "reduction.prod": run_prod,
    "indexing.index_select": run_index_select,
    "indexing.gather": run_gather,
    "indexing.scatter_add": run_scatter_add,
    "indexing.scatter_reduce": run_scatter_reduce,
    "indexing.take": run_take,
    "indexing.masked_select": run_masked_select,
    "sorting.sort": run_sort,
    "sorting.topk": run_topk,
    "sorting.kthvalue": run_kthvalue,
    "fft.rfft": run_rfft,
    "fft.irfft": run_irfft,
    "fft.fft": run_fft,
    "fft.fft2": run_fft2,
    "normalization.layer_norm": run_layer_norm,
    "normalization.group_norm": run_group_norm,
    "normalization.batch_norm": run_batch_norm,
    "spatial.avg_pool2d": run_avg_pool2d,
    "spatial.max_pool2d": run_max_pool2d,
    "spatial.adaptive_avg_pool2d": run_adaptive_avg_pool2d,
    "spatial.interpolate": run_interpolate,
    "linear_algebra.solve": run_solve,
    "linear_algebra.cholesky": run_cholesky,
    "linear_algebra.qr": run_qr,
    "linear_algebra.eigh": run_eigh,
    "linear_algebra.svdvals": run_svdvals,
    "model_patterns.vision_block": run_vision_block,
    "model_patterns.transformer_block_fragment": run_transformer_block_fragment,
    "model_patterns.patch_embedding": run_patch_embedding,
    "model_patterns.depthwise_separable": run_depthwise_separable,
    "model_patterns.residual_norm": run_residual_norm,
    "broadcasting.where": run_where,
    "broadcasting.add": run_add,
    "broadcasting.mul": run_mul,
    "broadcasting.masked_fill": run_masked_fill,
}


def run_path_shape_case(case: dict, device: str, compare) -> None:
    runner_name = case["runner"]
    try:
        runner = RUNNERS[runner_name]
    except KeyError as exc:
        raise AssertionError(f"unknown path-shape runner {runner_name!r}") from exc
    runner(case, device, compare)
