import pytest
import torch
import torchcts.conftest as conftest
from torchcts.core.opinfo_adapter import (
    get_backward_op_tests,
    get_live_opinfo,
    str_to_dtype,
    record_known_failure,
    is_cpu_reference_failure,
)
from torchcts.core.device import synchronize

# Build backward test list from op_db metadata + known failures (no probing)
try:
    op_tests = get_backward_op_tests(conftest._MANIFEST)
except Exception:
    op_tests = []

if not op_tests:
    op_tests = [("dummy", "dummy")]

@pytest.mark.opinfo
@pytest.mark.parametrize("op_name, dtype_str", op_tests)
def test_op_backward(op_name, dtype_str, device, compare):
    if op_name == "dummy":
        pytest.skip("No OpInfo autograd tests matched the manifest filters.")
        
    dtype = str_to_dtype(dtype_str)
    op_info = get_live_opinfo(op_name)

    # Generate sample inputs with requires_grad=True
    samples = list(op_info.sample_inputs(device, dtype, requires_grad=True))
    assert samples, f"No trainable sample inputs generated for {op_name} with {dtype_str}"

    op_fn = op_info.op
    category = "matmul_backward" if "mm" in op_name or "matmul" in op_name else "backward"

    tested_any = False

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

    def compare_gradients(dev_obj, cpu_obj, name):
        __tracebackhide__ = True
        nonlocal tested_any
        if isinstance(dev_obj, torch.Tensor):
            if dev_obj.requires_grad:
                if dev_obj.grad is not None and cpu_obj.grad is not None:
                    compare(dev_obj.grad, cpu_obj.grad, category=category, dtype=dtype)
                    tested_any = True
                elif dev_obj.grad is None and cpu_obj.grad is None:
                    pass
                else:
                    raise AssertionError(f"Gradient mismatch for {name}: device grad is {type(dev_obj.grad)}, CPU grad is {type(cpu_obj.grad)}")
        elif isinstance(dev_obj, (list, tuple)):
            for idx, (d_item, c_item) in enumerate(zip(dev_obj, cpu_obj)):
                compare_gradients(d_item, c_item, f"{name}[{idx}]")
        elif isinstance(dev_obj, dict):
            for k in dev_obj:
                compare_gradients(dev_obj[k], cpu_obj[k], f"{name}['{k}']")

    for sample in samples:
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
            except Exception:
                continue

            if not isinstance(actual_out, (torch.Tensor, list, tuple)):
                continue

            if not check_requires_grad(actual_out):
                continue

            try:
                run_backward(actual_out)
                synchronize(device)
            except Exception:
                continue

            tested_any = True
            continue

        # Prepare CPU clone of the sample input
        cpu_input = clone_to_device(sample.input, "cpu", detach=True)
        cpu_args = clone_to_device(sample.args, "cpu", detach=True)
        cpu_kwargs = clone_to_device(sample.kwargs, "cpu", detach=True)

        # Run CPU forward
        try:
            expected_out = op_fn(cpu_input, *cpu_args, **cpu_kwargs)
        except Exception as e:
            if is_cpu_reference_failure(e):
                record_known_failure("backward", op_name, dtype_str, f"fwd {type(e).__name__}: {e}")
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

        # CPU backward
        try:
            run_backward(expected_out)
        except Exception as e:
            if is_cpu_reference_failure(e):
                record_known_failure("backward", op_name, dtype_str, f"bwd {type(e).__name__}: {e}")
            # CPU backward failed — skip this sample
            continue

        # Device backward
        try:
            run_backward(actual_out)
            synchronize(device)
        except Exception as e:
            raise RuntimeError(f"Device backward execution failed: {e}") from e

        # Compare gradients of all inputs
        # Match CPU inputs with device inputs recursively
        compare_gradients(dev_input, cpu_input, "input")
        compare_gradients(dev_args, cpu_args, "args")
        compare_gradients(dev_kwargs, cpu_kwargs, "kwargs")

    if not tested_any:
        pytest.skip(f"No backward gradients could be computed or compared for {op_name}")
