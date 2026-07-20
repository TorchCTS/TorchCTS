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

import math

import pytest
import torch
import torchcts.conftest as conftest
from torchcts.core.backward_references import resolve_opinfo_backward_reference
from torchcts.core.opinfo_adapter import (
    get_backward_op_tests,
    get_live_opinfo,
    get_op_sample_inputs,
    str_to_dtype,
    is_cpu_reference_failure,
    classify_sample,
    InputCondition,
    stabilize_sample_randomness,
)
from torchcts.core.device import synchronize
from torchcts.core.runtime_evidence import record_opinfo_oracle_failure

pytestmark = pytest.mark.covers_category("opinfo_backward")

_DROPOUT_OVERRIDE_OPS = frozenset({
    "nn.functional.scaled_dot_product_attention",
    "nn.functional.multi_head_attention_forward",
})

_F32_OPMATH_BACKWARD_OPS = frozenset({
    "addbmm",
    "grid_sampler_2d",
    "grid_sampler_3d",
    "nn.functional.conv_transpose1d",
    "nn.functional.conv_transpose2d",
    "nn.functional.conv_transpose3d",
    "nn.functional.soft_margin_loss",
})

_RANDOMIZED_LINALG_BACKWARD_OPS = frozenset({"pca_lowrank", "svd_lowrank"})


def _gradient_comparison_dtype(gradient: torch.Tensor, op_dtype: torch.dtype) -> torch.dtype:
    """Select tolerances from the tensor actually being compared."""

    return gradient.dtype if isinstance(gradient, torch.Tensor) else op_dtype


def _project_output_for_backward(sample, output):
    """Apply the OpInfo's representation-invariant backward projection."""

    projection = getattr(sample, "output_process_fn_grad", None)
    if callable(projection):
        projected = projection(output)
        if projected is None:
            raise AssertionError("OpInfo output_process_fn_grad returned None")
        return projected
    return output


def _compare_rrelu_saved_noise_gradient(
    input_tensor: torch.Tensor,
    output: torch.Tensor,
    kwargs: dict,
    compare,
) -> None:
    if input_tensor.grad is None:
        raise AssertionError("RReLU input gradient was not populated")
    lower = float(kwargs.get("lower", 1.0 / 8.0))
    upper = float(kwargs.get("upper", 1.0 / 3.0))
    training = bool(kwargs.get("training", False))
    positive = input_tensor > 0
    if bool(positive.any()):
        compare(
            input_tensor.grad[positive],
            torch.ones_like(input_tensor.grad[positive]),
            category="exact",
            dtype=input_tensor.grad.dtype,
        )
    negative = input_tensor < 0
    finite_negative = negative & torch.isfinite(input_tensor)
    if bool(finite_negative.any()):
        if training:
            expected_gradient = (
                output[finite_negative] / input_tensor[finite_negative]
            ).detach()
        else:
            expected_gradient = torch.full_like(
                input_tensor.grad[finite_negative], (lower + upper) / 2
            )
        compare(
            input_tensor.grad[finite_negative],
            expected_gradient,
            category="backward",
            dtype=input_tensor.grad.dtype,
        )
    negative_infinite = negative & torch.isneginf(input_tensor)
    if bool(negative_infinite.any()):
        gradients = input_tensor.grad[negative_infinite].detach().cpu()
        outputs = output[negative_infinite].detach().cpu()
        if training:
            epsilon = 32 * torch.finfo(gradients.dtype).eps
            if not bool(
                torch.isfinite(gradients).all()
                and ((gradients >= lower - epsilon) & (gradients <= upper + epsilon)).all()
            ):
                raise AssertionError(
                    f"RReLU -inf-input gradients must use saved slopes in [{lower}, {upper}]"
                )
        else:
            torch.testing.assert_close(
                gradients,
                torch.full_like(gradients, (lower + upper) / 2),
            )
        valid_output = (
            ((gradients > 0) & torch.isneginf(outputs))
            | ((gradients < 0) & torch.isposinf(outputs))
            | ((gradients == 0) & torch.isnan(outputs))
        )
        if not bool(valid_output.all()):
            raise AssertionError("RReLU -inf output class does not match its saved slope")
    zero = input_tensor == 0
    if bool(zero.any()):
        zero_gradient = input_tensor.grad[zero].detach().cpu()
        if training:
            epsilon = 32 * torch.finfo(zero_gradient.dtype).eps
            if not bool(
                ((zero_gradient >= lower - epsilon) & (zero_gradient <= upper + epsilon)).all()
            ):
                raise AssertionError(
                    f"RReLU zero-input gradients must use saved slopes in [{lower}, {upper}]"
                )
        else:
            torch.testing.assert_close(
                zero_gradient,
                torch.full_like(zero_gradient, (lower + upper) / 2),
            )


def _randomized_linalg_direction(tensor: torch.Tensor, salt: int) -> torch.Tensor:
    phase = torch.arange(tensor.numel(), dtype=torch.float64).reshape(tensor.shape)
    real = torch.sin((phase + salt) * 0.731)
    if tensor.is_complex():
        imaginary = torch.cos((phase + salt) * 1.127)
        direction = torch.complex(real, imaginary).to(tensor.dtype)
    else:
        direction = real.to(tensor.dtype)
    norm = torch.linalg.vector_norm(direction)
    if float(norm) == 0:
        raise AssertionError("randomized-linalg finite-difference direction is zero")
    return (direction / norm).to(tensor.device)


def _randomized_linalg_backward_contract(
    op_name,
    op_fn,
    sample,
    dev_input,
    dev_args,
    dev_kwargs,
    dtype,
    device,
) -> None:
    """Check randomized-linalg gradients against same-device finite differences."""

    directions: dict[int, torch.Tensor] = {}
    differentiable: list[torch.Tensor] = []

    def collect(value):
        if isinstance(value, torch.Tensor):
            if value.requires_grad:
                differentiable.append(value)
                directions[id(value)] = _randomized_linalg_direction(
                    value, len(differentiable) * 17
                )
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)

    collect(dev_input)
    collect(dev_args)
    collect(dev_kwargs)
    if not differentiable:
        raise AssertionError(f"{op_name} randomized backward has no differentiable operands")

    predicted = 0.0
    for tensor in differentiable:
        if tensor.grad is None:
            raise AssertionError(f"{op_name} did not populate a randomized-linalg gradient")
        if not bool(torch.isfinite(torch.view_as_real(tensor.grad) if tensor.grad.is_complex() else tensor.grad).all()):
            raise AssertionError(f"{op_name} produced a nonfinite randomized-linalg gradient")
        direction = directions[id(tensor)]
        predicted += float(torch.real((tensor.grad.conj() * direction).sum()).detach().cpu())

    def perturb(value, sign, epsilon):
        if isinstance(value, torch.Tensor):
            base = value.detach()
            direction = directions.get(id(value))
            return base if direction is None else base + sign * epsilon * direction
        if isinstance(value, list):
            return [perturb(item, sign, epsilon) for item in value]
        if isinstance(value, tuple):
            return tuple(perturb(item, sign, epsilon) for item in value)
        if isinstance(value, dict):
            return {key: perturb(item, sign, epsilon) for key, item in value.items()}
        return value

    def projected_real_loss(output) -> torch.Tensor:
        projected = _project_output_for_backward(sample, output)
        terms = []

        def collect_terms(value):
            if isinstance(value, torch.Tensor):
                terms.append(value.real.sum() if value.is_complex() else value.sum())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect_terms(item)

        collect_terms(projected)
        if not terms:
            raise AssertionError(f"{op_name} randomized projection has no tensor outputs")
        return sum(terms[1:], terms[0])

    epsilon = 1e-5 if dtype in {torch.float64, torch.complex128} else 2e-3
    losses = []
    for sign in (1.0, -1.0):
        output = op_fn(
            perturb(dev_input, sign, epsilon),
            *perturb(dev_args, sign, epsilon),
            **perturb(dev_kwargs, sign, epsilon),
        )
        synchronize(device)
        losses.append(float(projected_real_loss(output).detach().cpu()))
    measured = (losses[0] - losses[1]) / (2 * epsilon)
    if not math.isfinite(predicted) or not math.isfinite(measured):
        raise AssertionError(
            f"{op_name} randomized finite difference is nonfinite: "
            f"backward={predicted}, finite_difference={measured}"
        )
    if dtype in {torch.float64, torch.complex128}:
        rtol, atol = 5e-4, 5e-5
    else:
        rtol, atol = 5e-2, 2e-2
    torch.testing.assert_close(
        torch.tensor(predicted, dtype=torch.float64),
        torch.tensor(measured, dtype=torch.float64),
        rtol=rtol,
        atol=atol,
        msg=(
            f"{op_name} randomized backward disagrees with a same-device "
            "fixed-seed directional derivative"
        ),
    )


def _override_dropout(sample, op_name):
    from torch.testing._internal.opinfo.core import SampleInput

    if op_name == "nn.functional.scaled_dot_product_attention" and "dropout_p" in sample.kwargs:
        return SampleInput(
            sample.input.clone() if isinstance(sample.input, torch.Tensor) else sample.input,
            args=sample.args,
            kwargs={**sample.kwargs, "dropout_p": 0.0},
        )
    if (op_name == "nn.functional.multi_head_attention_forward"
            and len(sample.args) > 9):
        args_list = list(sample.args)
        args_list[9] = 0.0
        return SampleInput(
            sample.input.clone() if isinstance(sample.input, torch.Tensor) else sample.input,
            args=tuple(args_list),
            kwargs=sample.kwargs,
        )
    return sample


# Build backward test list from op_db metadata (no probing)
op_tests = get_backward_op_tests(conftest._MANIFEST)

if not op_tests:
    op_tests = [("dummy", "dummy")]

@pytest.mark.opinfo
@pytest.mark.parametrize("op_name, dtype_str", op_tests)
@pytest.mark.requires("training")
def test_op_backward(op_name, dtype_str, device, compare, request):
    if op_name == "dummy":
        pytest.fail("Empty OpInfo autograd selection placeholder was not deselected at collection time.")
        
    dtype = str_to_dtype(dtype_str)
    op_info = get_live_opinfo(op_name)

    # Generate sample inputs on the reference device, then move exact clones to
    # the backend under test. Backend failures during sample construction are
    # not the same thing as failures of the op being tested.
    all_samples = [
        stabilize_sample_randomness(sample, op_name)
        for sample in get_op_sample_inputs(op_name, device, dtype, requires_grad=True)
    ]
    assert all_samples, f"No trainable sample inputs generated for {op_name} with {dtype_str}"

    # Filter to clean samples only — backward through NaN/Inf is not well-specified
    if op_name in _DROPOUT_OVERRIDE_OPS:
        all_samples = [_override_dropout(s, op_name) for s in all_samples]

    samples = [s for s in all_samples if classify_sample(s) == InputCondition.CLEAN]
    if not samples:
        pytest.fail(f"No clean (NaN/Inf-free) samples for backward test of {op_name} with {dtype_str}")

    op_fn = op_info.op
    category = "matmul_backward" if "mm" in op_name or "matmul" in op_name else "backward"

    tested_any = False
    cpu_failures = []

    def failure_summary():
        if not cpu_failures:
            return "no CPU failure details recorded"
        shown = "; ".join(cpu_failures[:3])
        if len(cpu_failures) > 3:
            shown += f"; ... {len(cpu_failures) - 3} more"
        return shown

    def check_requires_grad(out):
        if isinstance(out, torch.Tensor):
            return out.requires_grad
        elif isinstance(out, (list, tuple)):
            return any(check_requires_grad(o) for o in out)
        return False

    def run_backward(out):
        tensors_to_backward = []
        def collect_tensors(o):
            if isinstance(o, torch.Tensor):
                if (o.dtype.is_floating_point or o.dtype.is_complex) and o.requires_grad:
                    tensors_to_backward.append(o)
            elif isinstance(o, (list, tuple)):
                for item in o:
                    collect_tensors(item)
        collect_tensors(out)
        
        for idx, t in enumerate(tensors_to_backward):
            retain = (idx < len(tensors_to_backward) - 1)
            if t.layout in (torch.sparse_coo, torch.sparse_csr):
                t_sum = t.sum()
                t_sum.backward(torch.ones_like(t_sum), retain_graph=retain)
            else:
                t.backward(torch.ones_like(t), retain_graph=retain)

    def get_differentiable_tensors(obj):
        tensors = []
        if isinstance(obj, torch.Tensor):
            if obj.requires_grad:
                tensors.append(obj)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                tensors.extend(get_differentiable_tensors(item))
        elif isinstance(obj, dict):
            for item in obj.values():
                tensors.extend(get_differentiable_tensors(item))
        return tensors

    def force_requires_grad(obj):
        modified = False
        if isinstance(obj, torch.Tensor):
            if obj.dtype.is_floating_point or obj.dtype.is_complex:
                obj.requires_grad = True
                modified = True
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                if force_requires_grad(item):
                    modified = True
        elif isinstance(obj, dict):
            for item in obj.values():
                if force_requires_grad(item):
                    modified = True
        return modified

    def clone_to_device(obj, target_device, detach=False):
        if isinstance(obj, torch.Tensor):
            t = obj.to(target_device)
            if detach:
                t = t.detach()
            if obj.requires_grad:
                t.requires_grad = True
            return t
        elif isinstance(obj, list):
            return [clone_to_device(item, target_device, detach) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(clone_to_device(item, target_device, detach) for item in obj)
        elif isinstance(obj, dict):
            return {k: clone_to_device(v, target_device, detach) for k, v in obj.items()}
        return obj

    def clone_to_reference_dtype(obj, reference_dtype):
        if isinstance(obj, torch.Tensor):
            target_dtype = reference_dtype if (obj.is_floating_point() or obj.is_complex()) else obj.dtype
            result = obj.detach().to(device="cpu", dtype=target_dtype)
            if obj.requires_grad and (result.is_floating_point() or result.is_complex()):
                result.requires_grad_(True)
            return result
        if isinstance(obj, list):
            return [clone_to_reference_dtype(item, reference_dtype) for item in obj]
        if isinstance(obj, tuple):
            return tuple(clone_to_reference_dtype(item, reference_dtype) for item in obj)
        if isinstance(obj, dict):
            return {key: clone_to_reference_dtype(value, reference_dtype) for key, value in obj.items()}
        return obj

    def copy_reference_gradients(reference_obj, public_obj):
        if isinstance(reference_obj, torch.Tensor) and isinstance(public_obj, torch.Tensor):
            if reference_obj.grad is not None:
                public_obj.grad = reference_obj.grad.detach().to(public_obj.dtype)
            return
        if isinstance(reference_obj, (list, tuple)) and isinstance(public_obj, (list, tuple)):
            for reference_item, public_item in zip(reference_obj, public_obj):
                copy_reference_gradients(reference_item, public_item)
        elif isinstance(reference_obj, dict) and isinstance(public_obj, dict):
            for key in reference_obj:
                copy_reference_gradients(reference_obj[key], public_obj[key])

    def compare_gradients(dev_obj, cpu_obj, name, *, category_override=None):
        __tracebackhide__ = True
        nonlocal tested_any
        if isinstance(dev_obj, torch.Tensor):
            if dev_obj.requires_grad:
                if dev_obj.grad is not None and cpu_obj.grad is not None:
                    if cpu_obj.layout == torch.sparse_csr and cpu_obj.grad.layout == torch.sparse_csr:
                        crow = cpu_obj.crow_indices()
                        columns = cpu_obj.col_indices()
                        rows = torch.repeat_interleave(
                            torch.arange(cpu_obj.shape[0], device=columns.device),
                            crow[1:] - crow[:-1],
                        )
                        if dev_obj.grad.layout == torch.sparse_csr:
                            if not torch.equal(dev_obj.grad.crow_indices().cpu(), crow.cpu()):
                                raise AssertionError(f"Gradient CSR row pattern mismatch for {name}")
                            if not torch.equal(dev_obj.grad.col_indices().cpu(), columns.cpu()):
                                raise AssertionError(f"Gradient CSR column pattern mismatch for {name}")
                            actual_gradient = dev_obj.grad.values()
                        else:
                            actual_gradient = dev_obj.grad[rows.to(dev_obj.grad.device), columns.to(dev_obj.grad.device)]
                        compare(
                            actual_gradient,
                            cpu_obj.grad.values(),
                            category=category_override or category,
                            dtype=_gradient_comparison_dtype(actual_gradient, dtype),
                        )
                    else:
                        compare(
                            dev_obj.grad,
                            cpu_obj.grad,
                            category=category_override or category,
                            dtype=_gradient_comparison_dtype(dev_obj.grad, dtype),
                        )
                    tested_any = True
                elif dev_obj.grad is None and cpu_obj.grad is None:
                    pass
                else:
                    raise AssertionError(f"Gradient mismatch for {name}: device grad is {type(dev_obj.grad)}, CPU grad is {type(cpu_obj.grad)}")
        elif isinstance(dev_obj, (list, tuple)):
            for idx, (d_item, c_item) in enumerate(zip(dev_obj, cpu_obj)):
                compare_gradients(
                    d_item,
                    c_item,
                    f"{name}[{idx}]",
                    category_override=category_override,
                )
        elif isinstance(dev_obj, dict):
            for k in dev_obj:
                compare_gradients(
                    dev_obj[k],
                    cpu_obj[k],
                    f"{name}['{k}']",
                    category_override=category_override,
                )

    for sample in samples:
        backward_contract = resolve_opinfo_backward_reference(op_name, sample, dtype)
        # Check if sample contains any differentiable tensors
        diff_tensors = get_differentiable_tensors(sample.input)
        diff_tensors.extend(get_differentiable_tensors(sample.args))
        diff_tensors.extend(get_differentiable_tensors(sample.kwargs))

        if not diff_tensors:
            # Force requires_grad=True
            force_requires_grad(sample.input)
            force_requires_grad(sample.args)
            force_requires_grad(sample.kwargs)
            
            diff_tensors = get_differentiable_tensors(sample.input)
            diff_tensors.extend(get_differentiable_tensors(sample.args))
            diff_tensors.extend(get_differentiable_tensors(sample.kwargs))

        if not diff_tensors:
            continue

        if device == "cpu":
            # CPU validation mode: execute CPU forward/backward once, skip reference run and comparison
            try:
                dev_input = clone_to_device(sample.input, device, detach=True)
                dev_args = clone_to_device(sample.args, device, detach=True)
                dev_kwargs = clone_to_device(sample.kwargs, device, detach=True)

                actual_out = op_fn(dev_input, *dev_args, **dev_kwargs)
                synchronize(device)
            except Exception as exc:
                cpu_failures.append(f"cpu validation forward: {type(exc).__name__}: {exc}")
                continue

            if not isinstance(actual_out, (torch.Tensor, list, tuple)):
                continue

            if backward_contract is not None:
                try:
                    if not backward_contract.populate_gradients(dev_input, dev_args, dev_kwargs):
                        backward_contract = None
                except Exception as exc:
                    cpu_failures.append(
                        f"backward contract {backward_contract.reference_id}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
            if backward_contract is None:
                actual_backward_out = _project_output_for_backward(sample, actual_out)
                if not check_requires_grad(actual_backward_out):
                    continue

                try:
                    run_backward(actual_backward_out)
                    synchronize(device)
                except Exception as exc:
                    cpu_failures.append(f"cpu validation backward: {type(exc).__name__}: {exc}")
                    continue

            tested_any = True
            continue

        # Prepare CPU clone of the sample input
        cpu_input = clone_to_device(sample.input, "cpu", detach=True)
        cpu_args = clone_to_device(sample.args, "cpu", detach=True)
        cpu_kwargs = clone_to_device(sample.kwargs, "cpu", detach=True)

        if backward_contract is not None:
            try:
                if not backward_contract.populate_gradients(cpu_input, cpu_args, cpu_kwargs):
                    backward_contract = None
            except Exception as e:
                cpu_failures.append(
                    f"backward contract {backward_contract.reference_id}: {type(e).__name__}: {e}"
                )
                continue

        reference_dtype = None
        if dtype in {torch.float16, torch.bfloat16} and op_name in _F32_OPMATH_BACKWARD_OPS:
            reference_dtype = torch.float32
        elif dtype == torch.float32 and op_name in {"linalg.matrix_exp", "matrix_exp"}:
            reference_dtype = torch.float64
        use_wide_reference = reference_dtype is not None and backward_contract is None
        expected_out = None
        if backward_contract is not None:
            pass
        elif use_wide_reference:
            try:
                reference_input = clone_to_reference_dtype(cpu_input, reference_dtype)
                reference_args = clone_to_reference_dtype(cpu_args, reference_dtype)
                reference_kwargs = clone_to_reference_dtype(cpu_kwargs, reference_dtype)
                expected_out = op_fn(reference_input, *reference_args, **reference_kwargs)
                run_backward(_project_output_for_backward(sample, expected_out))
                copy_reference_gradients(reference_input, cpu_input)
                copy_reference_gradients(reference_args, cpu_args)
                copy_reference_gradients(reference_kwargs, cpu_kwargs)
            except Exception as e:
                cpu_failures.append(f"higher-opmath backward reference: {type(e).__name__}: {e}")
                continue
        else:
            # Run CPU forward
            try:
                expected_out = op_fn(cpu_input, *cpu_args, **cpu_kwargs)
            except Exception as e:
                if is_cpu_reference_failure(e):
                    record_opinfo_oracle_failure(
                        "backward",
                        op_name,
                        dtype_str,
                        "cpu_forward",
                        e,
                        input_condition=InputCondition.CLEAN,
                        nodeid=request.node.nodeid,
                    )
                cpu_failures.append(f"cpu forward: {type(e).__name__}: {e}")
                continue

            # Backward test only applies if forward output is a single tensor or list of tensors
            if not isinstance(expected_out, (torch.Tensor, list, tuple)):
                continue

            if not check_requires_grad(expected_out):
                continue

        # Run Device forward
        try:
            dev_input = clone_to_device(sample.input, device, detach=True)
            dev_args = clone_to_device(sample.args, device, detach=True)
            dev_kwargs = clone_to_device(sample.kwargs, device, detach=True)

            actual_out = op_fn(dev_input, *dev_args, **dev_kwargs)
            synchronize(device)
        except Exception as e:
            raise RuntimeError(f"Device forward execution failed: {e}") from e

        # CPU backward, unless the f32 direct-backward reference already populated gradients.
        if backward_contract is None and not use_wide_reference:
            try:
                run_backward(_project_output_for_backward(sample, expected_out))
            except Exception as e:
                if is_cpu_reference_failure(e):
                    record_opinfo_oracle_failure(
                        "backward",
                        op_name,
                        dtype_str,
                        "cpu_backward",
                        e,
                        input_condition=InputCondition.CLEAN,
                        nodeid=request.node.nodeid,
                    )
                cpu_failures.append(f"cpu backward: {type(e).__name__}: {e}")
                continue

        # Device backward
        try:
            run_backward(
                actual_out
                if use_wide_reference
                else _project_output_for_backward(sample, actual_out)
            )
            synchronize(device)
        except Exception as e:
            raise RuntimeError(f"Device backward execution failed: {e}") from e

        if op_name == "nn.functional.rrelu":
            if not isinstance(actual_out, torch.Tensor) or not isinstance(dev_input, torch.Tensor):
                raise AssertionError("RReLU backward contract requires tensor input and output")
            _compare_rrelu_saved_noise_gradient(
                dev_input,
                actual_out,
                dev_kwargs,
                compare,
            )
            tested_any = True
            continue

        if op_name in _RANDOMIZED_LINALG_BACKWARD_OPS:
            _randomized_linalg_backward_contract(
                op_name,
                op_fn,
                sample,
                dev_input,
                dev_args,
                dev_kwargs,
                dtype,
                device,
            )
            tested_any = True
            continue

        # Compare gradients of all inputs
        # Match CPU inputs with device inputs recursively
        gradient_category = "backward" if use_wide_reference or backward_contract is not None else None
        compare_gradients(dev_input, cpu_input, "input", category_override=gradient_category)
        compare_gradients(dev_args, cpu_args, "args", category_override=gradient_category)
        compare_gradients(dev_kwargs, cpu_kwargs, "kwargs", category_override=gradient_category)

    if not tested_any:
        pytest.fail(
            f"No backward gradients could be computed or compared for {op_name}. "
            f"CPU/reference failures: {failure_summary()}"
        )
