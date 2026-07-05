# Path-Shape Catalog

This catalog explains the runner-level intent behind
`torchcts/path_shapes/corpus.json`. Generated `source_note` fields point at the
anchors in this file.

The corpus is intentionally pre-expanded. A row is a concrete operation shape,
dtype, layout/stride mode, cost class, and resource tier. The family name is the
human domain. The runner name is the exact execution path.

## Matmul

### matmul-mm

Exercises `torch.mm` with skinny decode matrices, tile tails, transposed
operands, padded leading dimensions, and K sizes that cross common tile
boundaries.

### matmul-bmm

Exercises `torch.bmm` with small and odd batches, tile-tail matrix dimensions,
and zero-stride batch expansion.

### matmul-matmul

Exercises rank-3 `torch.matmul` broadcast paths that are distinct from direct
`mm` and `bmm` calls.

### matmul-addmm

Exercises fused bias/add matrix multiply with non-default `alpha` and `beta`
scalars.

### matmul-linear

Exercises the public `linear` path with model-shaped input, weight, and bias
dimensions.

## Convolution

### convolution-conv1d

Exercises temporal convolution with odd lengths and stride changes.

### convolution-conv2d

Exercises 2D convolution with odd image sizes, channels-last layout, grouped and
depthwise cases, kernel-size changes, and stride changes.

### convolution-conv3d

Exercises small volumetric convolution paths with odd depth/height/width.

### convolution-conv_transpose2d

Exercises decoder-style transposed convolution with stride and output-padding
paths.

## Attention

### attention-sdpa

Exercises scaled dot-product attention across decode, square, masked,
non-contiguous-mask, causal, and head-dimension boundary cases.

## Reduction

### reduction-sum

Exercises dimensional sum reductions with and without `keepdim`.

### reduction-mean

Exercises dimensional mean reductions with and without `keepdim`.

### reduction-amax

Exercises max-value reductions over different dimensions and rank patterns.

### reduction-argmax

Exercises index-producing reductions over different dimensions and rank
patterns.

### reduction-prod

Exercises product reductions, including cases that are more numerically fragile
than sum-like reductions.

## Indexing

### indexing-index_select

Exercises repeated and boundary index selection over both matrix dimensions.

### indexing-gather

Exercises duplicate gather indices and dimension-dependent gather shapes.

### indexing-scatter_add

Exercises duplicate destination accumulation.

### indexing-scatter_reduce

Exercises duplicate destination reduction through the public scatter-reduce
contract.

### indexing-take

Exercises flattened indexing and boundary indices.

### indexing-masked_select

Exercises boolean mask selection on contiguous and transposed inputs.

## Sorting

### sorting-sort

Exercises full ordering with ties across dimensions and ascending/descending
directions.

### sorting-topk

Exercises partial ordering with ties, sorted output, and dimension variation.

### sorting-kthvalue

Exercises rank selection with tied values across dimensions.

## FFT

### fft-rfft

Exercises real FFT plan paths for power-of-two, odd, tail, batched, and
strided lengths.

### fft-irfft

Exercises inverse real FFT plan paths matching the real FFT shape catalog.

### fft-fft

Exercises complex FFT plan paths over the same length and batch families.

### fft-fft2

Exercises small 2D FFT plans with odd rectangular shapes.

## Normalization

### normalization-layer_norm

Exercises single- and multi-dimension normalized shapes with multiple epsilon
values.

### normalization-group_norm

Exercises group counts, channel divisibility, and channels-last layout.

### normalization-batch_norm

Exercises training/eval behavior and channels-last layout.

## Spatial

### spatial-avg_pool2d

Exercises ceil-mode and count-include-padding pooling on odd image sizes.

### spatial-max_pool2d

Exercises max-pooling output-boundary behavior with ceil mode.

### spatial-adaptive_avg_pool2d

Exercises adaptive pooling with odd target output sizes.

### spatial-interpolate

Exercises nearest and bilinear resize paths with odd output sizes.

## Linear Algebra

### linear_algebra-solve

Exercises small and batched linear solves.

### linear_algebra-cholesky

Exercises small and batched symmetric positive-definite factorizations.

### linear_algebra-qr

Exercises tall, wide, and square QR factorization paths.

### linear_algebra-eigh

Exercises small and batched symmetric eigenvalue paths.

### linear_algebra-svdvals

Exercises singular-value paths for rectangular and square inputs.

## Model Patterns

### model_patterns-vision_block

Exercises a composed convolution-bias-ReLU block.

### model_patterns-transformer_block_fragment

Exercises transformer attention fragments through the SDPA runner with model-like
batch/head/sequence/head-dimension combinations.

### model_patterns-patch_embedding

Exercises vision-transformer patch embedding with convolution, flatten, and
transpose behavior.

### model_patterns-depthwise_separable

Exercises depthwise convolution, activation, and pointwise convolution as a
mobile-vision pattern.

### model_patterns-residual_norm

Exercises residual add followed by layer normalization.

## Broadcasting

### broadcasting-where

Exercises mask and value broadcasting in `torch.where`.

### broadcasting-add

Exercises elementwise add with implicit expansion and zero-stride inputs.

### broadcasting-mul

Exercises elementwise multiply with implicit expansion and zero-stride inputs.

### broadcasting-masked_fill

Exercises scalar masked fill with broadcast-shaped masks.
