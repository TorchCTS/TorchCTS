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
import torchcts.conftest as conftest
from torchcts.core.opinfo_adapter import (
    get_forward_op_tests,
    get_live_opinfo,
    get_op_sample_inputs,
    str_to_dtype,
    record_known_failure,
    is_cpu_reference_failure,
    InputCondition,
    prepare_sample,
)
from torchcts.core.comparer import compare_nan_propagation, compare_inf_propagation
from torchcts.core.device import synchronize


# ---------------------------------------------------------------------------
# Op categorization — maps op names to tolerance categories
# ---------------------------------------------------------------------------

def get_op_category(op_name):
    op_lower = op_name.lower()

    # Attention / SDPA
    if any(k in op_lower for k in (
        "scaled_dot_product_attention", "multi_head_attention",
    )):
        return "sdpa"

    # Matmul-family (check before "reduction" since addmm has "mm")
    if any(k in op_lower for k in (
        "_mm", "matmul", "dot", "mv", "bmm", "addmm", "addbmm", "addmv",
        "baddbmm", "addr", "linear",
    )):
        return "matmul"

    # Loss functions
    if any(k in op_lower for k in (
        "cross_entropy", "nll_loss", "binary_cross_entropy", "mse_loss",
        "l1_loss", "smooth_l1_loss", "huber_loss", "kl_div",
        "poisson_nll_loss", "margin_ranking_loss", "hinge_embedding_loss",
        "multi_margin_loss", "multilabel_margin_loss",
        "multilabel_soft_margin_loss", "soft_margin_loss",
        "triplet_margin_loss", "cosine_embedding_loss", "ctc_loss",
    )):
        return "loss"

    # Grid sampling
    if any(k in op_lower for k in ("grid_sample", "grid_sampler")):
        return "grid_sample"

    # Convolution
    if "conv" in op_lower:
        return "conv"

    # Reductions (check after conv since conv_transpose shouldn't match)
    if any(k in op_lower for k in (
        "sum", "mean", "std", "var", "norm", "amax", "amin", "prod",
        "cumsum", "cumprod", "logsumexp", "logcumsumexp",
        "softmax", "_softmax_backward_data",
        "index_reduce", "_unsafe_masked_index",
        "cov",
    )):
        return "reduction"

    # Normalization layers
    if any(k in op_lower for k in (
        "batch_norm", "group_norm", "layer_norm", "instance_norm",
        "normalize",
    )):
        return "norm"

    # Linear algebra
    if "linalg" in op_lower or any(k in op_lower for k in (
        "cholesky", "svd", "eig", "qr", "lu", "solve", "householder",
        "det", "slogdet", "pinverse", "matrix_power",
    )):
        return "linalg"

    # FFT
    if "fft" in op_lower:
        return "fft"

    # Copy/clone
    if any(k in op_lower for k in ("copy", "clone", "to_copy")):
        return "copy"

    # Interpolation / upsampling (uses accumulation)
    if any(k in op_lower for k in ("upsample", "interpolate")):
        return "reduction"

    return "elementwise"


# ---------------------------------------------------------------------------
# Build test list from op_db metadata + known failures (no probing)
# ---------------------------------------------------------------------------

try:
    op_tests = get_forward_op_tests(conftest._MANIFEST)
except Exception:
    op_tests = []

if not op_tests:
    # Dummy parameter to avoid pytest collection errors
    op_tests = [("dummy", "dummy", "clean")]


# ---------------------------------------------------------------------------
# Op classification sets
# ---------------------------------------------------------------------------

# Ops whose outputs are inherently nondeterministic — value comparison is invalid
_NONDETERMINISTIC_OPS = frozenset({
    # Uninitialized memory
    "empty", "empty_like", "empty_permuted", "empty_strided",
    "new_empty", "new_empty_strided",
    # Random sampling
    "bernoulli", "geometric", "multinomial",
    "rand_like", "randint", "randint_like", "randn", "randn_like",
    "nn.functional.dropout", "nn.functional.dropout2d", "nn.functional.dropout3d",
    "nn.functional.alpha_dropout", "nn.functional.feature_alpha_dropout",
    "nn.functional.fractional_max_pool2d", "nn.functional.fractional_max_pool3d",
    # Decompositions with inherent sign/order ambiguity
    "svd_lowrank", "pca_lowrank",
    # Random distributions
    "normal", "uniform", "log_normal", "cauchy", "exponential",
})

_UNINITIALIZED_OPS = frozenset({
    "empty", "empty_like", "empty_permuted", "empty_strided",
    "new_empty", "new_empty_strided",
})

# Ops that return (values, indices) where index ordering depends on sort stability
_SORT_OPS = frozenset({
    "sort", "argsort", "topk", "kthvalue",
})

# Ops that return indices where tie-breaking is unspecified
_ARGMAX_OPS = frozenset({
    "masked.argmax", "masked.argmin", "argmax", "argmin",
})

# Ops where we override dropout_p=0 for numerical comparison
_DROPOUT_OVERRIDE_OPS = frozenset({
    "nn.functional.scaled_dot_product_attention",
    "nn.functional.multi_head_attention_forward",
})




# ---------------------------------------------------------------------------
# Sort output comparison helpers
# ---------------------------------------------------------------------------

def _compare_sort_output(actual, expected, op_name, sample, category, dtype, compare):
    """Compare sort/topk outputs handling unstable tie-breaking.

    For stable=False (default):
      - Values must match within tolerance
      - Indices are validated by reconstruction: input[indices] == values

    For stable=True:
      - Both values AND indices must match exactly
    """
    __tracebackhide__ = True
    stable = sample.kwargs.get("stable", False)

    # Extract values and indices from the output
    if isinstance(actual, (tuple, torch.return_types.sort)):
        act_values, act_indices = actual[0], actual[1]
        exp_values, exp_indices = expected[0], expected[1]
    elif op_name == "argsort":
        # argsort returns only indices
        act_indices = actual
        exp_indices = expected
        act_values = None
        exp_values = None
    else:
        # Fallback: single tensor output
        compare(actual, expected, category=category, dtype=dtype)
        return

    # Values must always match within tolerance
    if act_values is not None:
        compare(act_values, exp_values, category=category, dtype=dtype)

    if stable:
        # Stable sort: indices must match exactly
        if act_indices is not None and exp_indices is not None:
            if not torch.equal(act_indices.cpu(), exp_indices.cpu()):
                raise AssertionError(
                    f"Stable sort indices mismatch for {op_name}. "
                    f"Mismatched elements: {(act_indices.cpu() != exp_indices.cpu()).sum().item()}"
                )
    else:
        # Unstable sort: verify indices produce valid sorted values.
        # For sort-family ops, dim can be in different positional arg slots:
        #   sort(input, dim=-1, descending=False, stable=False)  -> dim is args[0]
        #   topk(input, k, dim=-1, largest=True, sorted=True)    -> dim is args[1]
        #   kthvalue(input, k, dim=-1, keepdim=False)            -> dim is args[1]
        #   argsort(input, dim=-1, descending=False, stable=False) -> dim is args[0]

        def _get_sort_dim(op_name, sample):
            if "dim" in sample.kwargs:
                return sample.kwargs["dim"]
            if op_name in ("topk", "kthvalue"):
                # args = (k, [dim, ...])
                return sample.args[1] if len(sample.args) > 1 else -1
            else:
                # sort, argsort: args = ([dim, ...])
                return sample.args[0] if sample.args else -1

        if op_name in ("topk", "kthvalue"):
            # topk/kthvalue: output may have different shape from input (reduced dim
            # for kthvalue, reduced size for topk). Validate by comparing the sorted
            # values, which are deterministic (no tie-breaking ambiguity in values).
            # The values comparison already happened above. For indices, we verify
            # that input[actual_indices] matches actual_values by advanced indexing.
            if act_values is not None and act_indices is not None:
                dim = _get_sort_dim(op_name, sample)
                dev = act_values.device
                original_input = sample.input
                if isinstance(original_input, torch.Tensor):
                    original_input = original_input.to(dev)
                try:
                    # Handle keepdim=False: indices may have fewer dims than input.
                    # Unsqueeze at the sort dim to make gather work, then squeeze back.
                    indices_for_gather = act_indices
                    needs_squeeze = act_indices.ndim < original_input.ndim
                    if needs_squeeze:
                        indices_for_gather = act_indices.unsqueeze(dim)
                    reconstructed = torch.gather(original_input, dim, indices_for_gather)
                    if needs_squeeze:
                        reconstructed = reconstructed.squeeze(dim)
                    compare(reconstructed, act_values, category="exact", dtype=dtype)
                except Exception as e:
                    raise AssertionError(
                        f"Unstable {op_name} index validation failed: "
                        f"input[indices] != values. {e}"
                    ) from e
        elif op_name == "argsort":
            # argsort: compare gathered values from device vs CPU indices
            dim = _get_sort_dim(op_name, sample)
            dev = act_indices.device
            original_input = sample.input.to(dev)
            try:
                gathered_act = torch.gather(original_input, dim, act_indices)
                gathered_exp = torch.gather(sample.input.to(exp_indices.device), dim, exp_indices)
                compare(gathered_act, gathered_exp, category=category, dtype=dtype)
            except AssertionError:
                raise
            except Exception as e:
                raise AssertionError(
                    f"argsort index validation failed: {e}"
                ) from e
        elif act_values is not None and act_indices is not None:
            # sort/msort: standard gather reconstruction
            dim = _get_sort_dim(op_name, sample)
            dev = act_values.device
            original_input = sample.input
            if isinstance(original_input, torch.Tensor):
                original_input = original_input.to(dev)
            try:
                reconstructed = torch.gather(original_input, dim, act_indices)
                compare(reconstructed, act_values, category="exact", dtype=dtype)
            except Exception as e:
                raise AssertionError(
                    f"Unstable sort index validation failed for {op_name}: "
                    f"input[indices] != sorted values. {e}"
                ) from e


def _override_dropout(sample, op_name):
    """Override dropout_p to 0.0 for deterministic numerical comparison.

    Returns a new SampleInput with dropout_p=0.0 if applicable, else the original.
    """
    from torch.testing._internal.opinfo.core import SampleInput

    if "dropout_p" in sample.kwargs:
        new_kwargs = {**sample.kwargs, "dropout_p": 0.0}
        return SampleInput(
            sample.input.clone() if isinstance(sample.input, torch.Tensor) else sample.input,
            args=sample.args,
            kwargs=new_kwargs,
        )
    elif (op_name == "nn.functional.multi_head_attention_forward"
          and len(sample.args) > 9):
        # MHA passes dropout as positional arg[9]
        args_list = list(sample.args)
        args_list[9] = 0.0
        return SampleInput(
            sample.input.clone() if isinstance(sample.input, torch.Tensor) else sample.input,
            args=tuple(args_list),
            kwargs=sample.kwargs,
        )
    return sample


# ---------------------------------------------------------------------------
# NaN/Inf tier comparison helpers
# ---------------------------------------------------------------------------

def _compare_special_tier(actual, expected, condition):
    """Structural + NaN/Inf propagation comparison for non-clean tiers."""
    __tracebackhide__ = True

    def _compare_item(act, exp):
        __tracebackhide__ = True
        if isinstance(act, torch.Tensor) and isinstance(exp, torch.Tensor):
            assert act.shape == exp.shape, (
                f"Shape mismatch: got {act.shape}, expected {exp.shape}"
            )
            assert act.dtype == exp.dtype, (
                f"Dtype mismatch: got {act.dtype}, expected {exp.dtype}"
            )
            if act.is_floating_point() or act.is_complex():
                if condition == InputCondition.HAS_NAN:
                    compare_nan_propagation(act, exp)
                if condition == InputCondition.HAS_INF:
                    compare_inf_propagation(act, exp)
        elif isinstance(act, (list, tuple)) and isinstance(exp, (list, tuple)):
            assert len(act) == len(exp), (
                f"Output sequence lengths differ: got {len(act)}, expected {len(exp)}"
            )
            for a, e in zip(act, exp):
                _compare_item(a, e)
        elif isinstance(act, dict) and isinstance(exp, dict):
            assert len(act) == len(exp), (
                f"Output dict sizes differ: got {len(act)}, expected {len(exp)}"
            )
            for k in act:
                assert k in exp, f"Key {k} not in CPU reference output keys"
                _compare_item(act[k], exp[k])

    _compare_item(actual, expected)


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

@pytest.mark.opinfo
@pytest.mark.parametrize("op_name, dtype_str, input_condition", op_tests,
                         ids=[f"{c}-{op}-{dt}" for op, dt, c in op_tests])
def test_op_forward(op_name, dtype_str, input_condition, device, compare):
    if op_name == "dummy":
        pytest.skip("No OpInfo tests matched the manifest filters.")

    dtype = str_to_dtype(dtype_str)

    # Determine category
    category = get_op_category(op_name)

    # Get live generator function from OpInfo
    op_info = get_live_opinfo(op_name)
    assert op_info is not None, f"Live OpInfo for {op_name} disappeared after collection"
    op_fn = op_info.op

    # Sample caps from manifest
    max_samples = conftest._MANIFEST.get("max_samples", 10)
    max_samples_ieee = conftest._MANIFEST.get("max_samples_ieee754", 3)
    sample_cap = max_samples_ieee if input_condition != InputCondition.CLEAN else max_samples

    # IEEE 754 seed from manifest
    ieee754_seed = conftest._MANIFEST.get("ieee754_seed", 67)

    tested_any = False
    passed_count = 0
    stable_sort_samples = []  # Collect passing samples for stable sort retest
    has_any_samples = False

    for i, raw_sample in enumerate(get_op_sample_inputs(op_name, device, dtype)):
        has_any_samples = True
        if sample_cap and passed_count >= sample_cap:
            break

        # Transform this sample for the target condition (lazy — one at a time)
        sample = prepare_sample(raw_sample, input_condition,
                                ieee754_seed=ieee754_seed, sample_index=i)

        # Apply dropout override for attention ops
        if op_name in _DROPOUT_OVERRIDE_OPS:
            sample = _override_dropout(sample, op_name)

        # Both CPU and device get copies of the SAME transformed sample
        cpu_input = sample.input.cpu() if isinstance(sample.input, torch.Tensor) else sample.input
        cpu_args = [a.cpu() if isinstance(a, torch.Tensor) else a for a in sample.args]
        cpu_kwargs = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in sample.kwargs.items()}

        if device == "cpu":
            # CPU validation mode: execute target op once, skip comparison
            try:
                dev_input = sample.input.to(device) if isinstance(sample.input, torch.Tensor) else sample.input
                dev_args = [a.to(device) if isinstance(a, torch.Tensor) else a for a in sample.args]
                dev_kwargs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sample.kwargs.items()}

                actual = op_fn(dev_input, *dev_args, **dev_kwargs)
                synchronize(device)
            except Exception:
                continue
            tested_any = True
            passed_count += 1
            continue

        # Run reference CPU op
        cpu_error = None
        expected = None
        try:
            expected = op_fn(cpu_input, *cpu_args, **cpu_kwargs)
        except Exception as e:
            if is_cpu_reference_failure(e):
                record_known_failure("forward", op_name, dtype_str, f"{type(e).__name__}: {e}")
                continue
            if input_condition != InputCondition.CLEAN:
                cpu_error = e  # For NaN/Inf tiers, capture error for comparison
            else:
                continue  # Clean tier: skip sample on CPU error

        # Run target device op
        dev_error = None
        actual = None
        try:
            dev_input = sample.input.to(device) if isinstance(sample.input, torch.Tensor) else sample.input
            dev_args = [a.to(device) if isinstance(a, torch.Tensor) else a for a in sample.args]
            dev_kwargs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sample.kwargs.items()}

            actual = op_fn(dev_input, *dev_args, **dev_kwargs)
            synchronize(device)
        except Exception as e:
            if input_condition != InputCondition.CLEAN:
                dev_error = e  # For NaN/Inf tiers, capture error for comparison
            else:
                raise RuntimeError(f"Execution failed on device {device}: {e}") from e

        # --- Error matching for NaN/Inf tiers ---
        if input_condition != InputCondition.CLEAN and (cpu_error is not None or dev_error is not None):
            if cpu_error is not None and dev_error is not None:
                # Both raised — consistent rejection = PASS
                tested_any = True
                passed_count += 1
                continue
            elif cpu_error is not None:
                raise AssertionError(
                    f"CPU raised {type(cpu_error).__name__} but device succeeded "
                    f"for {op_name} ({input_condition}): {cpu_error}"
                )
            else:
                raise AssertionError(
                    f"Device raised {type(dev_error).__name__} but CPU succeeded "
                    f"for {op_name} ({input_condition}): {dev_error}"
                )

        # --- Comparison logic ---
        if input_condition != InputCondition.CLEAN:
            # Non-clean tiers: structural + NaN/Inf propagation check only
            _compare_special_tier(actual, expected, input_condition)
        elif op_name in _NONDETERMINISTIC_OPS:
            _compare_nondeterministic(actual, expected, op_name)
        elif op_name in _SORT_OPS:
            _compare_sort_output(actual, expected, op_name, sample, category, dtype, compare)
        elif op_name in _ARGMAX_OPS:
            # Argmax/argmin: just verify shape/dtype match (tie-breaking is unspecified)
            _compare_nondeterministic(actual, expected, op_name)
        else:
            _compare_recursive(actual, expected, category, dtype, compare)
        tested_any = True
        passed_count += 1

        # Collect passing samples for stable sort retest
        if op_name == "sort" and input_condition == InputCondition.CLEAN:
            stable_sort_samples.append(sample)

    if not has_any_samples:
        pytest.skip(f"No sample inputs generated for {op_name} with {dtype_str}")

    if not tested_any:
        pytest.skip(f"All sample inputs for {op_name} were skipped or failed on CPU reference")

    # For sort ops on clean tier, also run stable=True variant
    if (input_condition == InputCondition.CLEAN
            and op_name == "sort"
            and device != "cpu"
            and stable_sort_samples):
        _run_stable_sort_tests(stable_sort_samples, op_fn, device, category, dtype, compare)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _compare_recursive(act, exp, category, dtype, compare):
    """Value comparison for deterministic ops."""
    __tracebackhide__ = True
    if isinstance(act, torch.Tensor) and isinstance(exp, torch.Tensor):
        compare(act, exp, category=category, dtype=dtype)
    elif isinstance(act, (list, tuple)) and isinstance(exp, (list, tuple)):
        assert len(act) == len(exp), (
            f"Output sequence lengths differ: got {len(act)}, expected {len(exp)}"
        )
        for a, e in zip(act, exp):
            _compare_recursive(a, e, category, dtype, compare)
    elif isinstance(act, dict) and isinstance(exp, dict):
        assert len(act) == len(exp), (
            f"Output dict sizes differ: got {len(act)}, expected {len(exp)}"
        )
        for k in act:
            assert k in exp, f"Key {k} not in CPU reference output keys"
            _compare_recursive(act[k], exp[k], category, dtype, compare)


def _compare_nondeterministic(act, exp, op_name):
    """Structural comparison for nondeterministic/random/uninitialized ops."""
    __tracebackhide__ = True
    if isinstance(act, torch.Tensor) and isinstance(exp, torch.Tensor):
        assert act.shape == exp.shape, (
            f"Shape mismatch: got {act.shape}, expected {exp.shape}"
        )
        assert act.dtype == exp.dtype, (
            f"Dtype mismatch: got {act.dtype}, expected {exp.dtype}"
        )
        if op_name not in _UNINITIALIZED_OPS:
            if act.is_floating_point() or act.is_complex():
                if torch.isfinite(exp).all():
                    assert torch.isfinite(act).all(), (
                        f"Output tensor contains non-finite values (NaN/Inf) "
                        f"but CPU reference was finite."
                    )
    elif isinstance(act, (list, tuple)) and isinstance(exp, (list, tuple)):
        assert len(act) == len(exp), (
            f"Output sequence lengths differ: got {len(act)}, expected {len(exp)}"
        )
        for a, e in zip(act, exp):
            _compare_nondeterministic(a, e, op_name)
    elif isinstance(act, dict) and isinstance(exp, dict):
        assert len(act) == len(exp), (
            f"Output dict sizes differ: got {len(act)}, expected {len(exp)}"
        )
        for k in act:
            assert k in exp, f"Key {k} not in CPU reference output keys"
            _compare_nondeterministic(act[k], exp[k], op_name)


def _run_stable_sort_tests(samples, op_fn, device, category, dtype, compare):
    """Run additional stable=True sort tests for sort op.

    Clones existing samples and adds stable=True to kwargs, then compares
    both values and indices exactly.
    """
    __tracebackhide__ = True
    for sample in samples:
        stable_kwargs = {**sample.kwargs, "stable": True}

        cpu_input = sample.input.cpu() if isinstance(sample.input, torch.Tensor) else sample.input
        cpu_args = [a.cpu() if isinstance(a, torch.Tensor) else a for a in sample.args]
        cpu_kwargs = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in stable_kwargs.items()}

        try:
            expected = op_fn(cpu_input, *cpu_args, **cpu_kwargs)
        except Exception:
            continue

        try:
            dev_input = sample.input.to(device) if isinstance(sample.input, torch.Tensor) else sample.input
            dev_args = [a.to(device) if isinstance(a, torch.Tensor) else a for a in sample.args]
            dev_kwargs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in stable_kwargs.items()}

            actual = op_fn(dev_input, *dev_args, **dev_kwargs)
            synchronize(device)
        except Exception as e:
            raise RuntimeError(
                f"Execution failed on device {device} (stable=True): {e}"
            ) from e

        # For stable sort, both values and indices must match
        if isinstance(actual, (tuple,)):
            act_values, act_indices = actual[0], actual[1]
            exp_values, exp_indices = expected[0], expected[1]
            compare(act_values, exp_values, category=category, dtype=dtype)
            if not torch.equal(act_indices.cpu(), exp_indices.cpu()):
                raise AssertionError(
                    f"Stable sort indices mismatch. "
                    f"Mismatched: {(act_indices.cpu() != exp_indices.cpu()).sum().item()}"
                )
