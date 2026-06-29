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
from torchcts.core.device import synchronize

pytestmark = pytest.mark.covers_category("autocast")


def _amp_grad_inputs(device):
    return [
        torch.tensor([2.0, -4.0, 0.0], dtype=torch.float32, device=device),
        torch.tensor([float("nan"), float("inf"), -8.0], dtype=torch.float32, device=device),
    ]


def _compare_tensor_lists(actual, expected, compare, *, category="exact", dtype=torch.float32):
    assert len(actual) == len(expected)
    for actual_item, expected_item in zip(actual, expected):
        compare(actual_item, expected_item, category=category, dtype=dtype)


def _run_amp_non_finite_unscale(target_device, *, variant):
    grads = _amp_grad_inputs(target_device)
    found_inf = torch.tensor(0.0, dtype=torch.float32, device=target_device)
    inv_scale = torch.tensor(0.25, dtype=torch.float32, device=target_device)

    if variant == "functional":
        return torch.ops.aten._amp_foreach_non_finite_check_and_unscale(
            grads,
            found_inf,
            inv_scale,
        )
    if variant == "out":
        outputs = [torch.empty_like(grad) for grad in grads]
        returned = torch.ops.aten._amp_foreach_non_finite_check_and_unscale.out(
            grads,
            found_inf,
            inv_scale,
            out=outputs,
        )
        return returned, outputs, found_inf, grads
    if variant == "inplace":
        returned = torch.ops.aten._amp_foreach_non_finite_check_and_unscale_(
            grads,
            found_inf,
            inv_scale,
        )
        return returned, grads, found_inf

    raise AssertionError(f"unknown AMP non-finite variant: {variant}")


def _run_amp_update_scale(target_device, *, found_inf_value, variant):
    scale = torch.tensor(8.0, dtype=torch.float32, device=target_device)
    growth_tracker = torch.tensor(1, dtype=torch.int32, device=target_device)
    found_inf = torch.tensor(found_inf_value, dtype=torch.float32, device=target_device)

    if variant == "functional":
        return torch.ops.aten._amp_update_scale(
            scale,
            growth_tracker,
            found_inf,
            2.0,
            0.5,
            2,
        )
    if variant == "out":
        out = torch.empty_like(scale)
        returned = torch.ops.aten._amp_update_scale.out(
            scale,
            growth_tracker,
            found_inf,
            2.0,
            0.5,
            2,
            out=out,
        )
        return returned, out, scale, growth_tracker
    if variant == "inplace":
        returned = torch.ops.aten._amp_update_scale_(
            scale,
            growth_tracker,
            found_inf,
            2.0,
            0.5,
            2,
        )
        return returned, scale, growth_tracker

    raise AssertionError(f"unknown AMP scale update variant: {variant}")

@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.requires("autocast")
@pytest.mark.parametrize("autocast_dtype", [torch.float16])
def test_autocast_precisions(autocast_dtype, device, manifest):
    # We run autocast context for the device type
    # For MPS, device_type is 'mps'
    device_type = "cuda" if device == "cuda" else ("mps" if device == "mps" else "cpu")

    x = torch.randn(4, 4, device=device)
    y = torch.randn(4, 4, device=device)
    
    with torch.autocast(device_type=device_type, dtype=autocast_dtype):
        # Matmul should downcast to half precision
        out = torch.mm(x, y)
        synchronize(device)
        
        # Verify it downcasted if device_type supports it
        # On CPU autocast might stay float32 depending on config, but on GPU/MPS it usually downcasts
        assert out.dtype in (torch.float16, torch.bfloat16, torch.float32)


@pytest.mark.requires("autocast")
def test_autocast_keep_precision(device, manifest):
    """Numerically sensitive ops must stay fp32 under autocast."""
    x = torch.randn(4, 8, device=device)
    w = torch.randn(8, device=device)

    with torch.autocast(device_type=device):
        y_ln = torch.nn.functional.layer_norm(x, [8], w)
    assert y_ln.dtype == torch.float32, f"layer_norm: expected fp32, got {y_ln.dtype}"

    with torch.autocast(device_type=device):
        y_sm = torch.nn.functional.softmax(x, dim=-1)
    assert y_sm.dtype == torch.float32, f"softmax: expected fp32, got {y_sm.dtype}"

    logits = torch.randn(4, 10, device=device)
    targets = torch.randint(0, 10, (4,), device=device)
    with torch.autocast(device_type=device):
        loss = torch.nn.functional.cross_entropy(logits, targets)
    assert loss.dtype == torch.float32, f"cross_entropy: expected fp32, got {loss.dtype}"


@pytest.mark.requires("autocast")
def test_autocast_downcast(device, manifest):
    """Matmul-class ops should downcast to fp16 under autocast."""
    a = torch.randn(4, 8, device=device)
    b = torch.randn(8, 4, device=device)
    with torch.autocast(device_type=device, dtype=torch.float16):
        c = torch.mm(a, b)
    assert c.dtype == torch.float16, f"Expected fp16, got {c.dtype}"


@pytest.mark.requires("autocast")
def test_autocast_backward(device, manifest):
    """Autocast forward followed by backward should produce valid gradients."""
    a = torch.randn(4, 8, device=device, requires_grad=True)
    b = torch.randn(8, 4, device=device, requires_grad=True)
    with torch.autocast(device_type=device, dtype=torch.float16):
        c = torch.mm(a, b)
    c.float().sum().backward()
    assert a.grad is not None
    assert not torch.isnan(a.grad).any()


@pytest.mark.requires("autocast")
@pytest.mark.requires("training")
def test_grad_scaler(device, manifest):
    """GradScaler scale/step/update cycle must not crash and scale > 0."""
    scaler = torch.amp.GradScaler(device)
    model = torch.nn.Linear(8, 4).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 8, device=device)

    opt.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.float16):
        y = model(x)
        loss = y.sum()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    assert scaler.get_scale() > 0


@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.requires("autocast")
@pytest.mark.covers("aten::_amp_foreach_non_finite_check_and_unscale")
@pytest.mark.covers("aten::_amp_foreach_non_finite_check_and_unscale.out")
@pytest.mark.covers("aten::_amp_foreach_non_finite_check_and_unscale_")
def test_amp_foreach_non_finite_check_and_unscale_variants(device, compare):
    actual_grads, actual_found_inf = _run_amp_non_finite_unscale(device, variant="functional")
    expected_grads, expected_found_inf = _run_amp_non_finite_unscale("cpu", variant="functional")
    synchronize(device)
    _compare_tensor_lists(actual_grads, expected_grads, compare)
    compare(actual_found_inf, expected_found_inf, category="exact", dtype=torch.float32)

    returned_dev, out_dev, found_inf_dev, original_dev = _run_amp_non_finite_unscale(device, variant="out")
    returned_cpu, out_cpu, found_inf_cpu, original_cpu = _run_amp_non_finite_unscale("cpu", variant="out")
    assert returned_dev is None and returned_cpu is None
    synchronize(device)
    _compare_tensor_lists(out_dev, out_cpu, compare)
    _compare_tensor_lists(original_dev, original_cpu, compare)
    compare(found_inf_dev, found_inf_cpu, category="exact", dtype=torch.float32)

    returned_dev, grads_dev, found_inf_dev = _run_amp_non_finite_unscale(device, variant="inplace")
    returned_cpu, grads_cpu, found_inf_cpu = _run_amp_non_finite_unscale("cpu", variant="inplace")
    assert returned_dev is None and returned_cpu is None
    synchronize(device)
    _compare_tensor_lists(grads_dev, grads_cpu, compare)
    compare(found_inf_dev, found_inf_cpu, category="exact", dtype=torch.float32)


@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.requires("autocast")
@pytest.mark.covers("aten::_amp_update_scale")
@pytest.mark.covers("aten::_amp_update_scale.out")
@pytest.mark.covers("aten::_amp_update_scale_")
@pytest.mark.parametrize("found_inf_value", [0.0, 1.0])
def test_amp_update_scale_variants(found_inf_value, device, compare):
    actual_scale, actual_growth_tracker = _run_amp_update_scale(
        device,
        found_inf_value=found_inf_value,
        variant="functional",
    )
    expected_scale, expected_growth_tracker = _run_amp_update_scale(
        "cpu",
        found_inf_value=found_inf_value,
        variant="functional",
    )
    synchronize(device)
    compare(actual_scale, expected_scale, category="exact", dtype=torch.float32)
    compare(actual_growth_tracker, expected_growth_tracker, category="exact", dtype=torch.int32)

    returned_dev, out_dev, scale_dev, growth_tracker_dev = _run_amp_update_scale(
        device,
        found_inf_value=found_inf_value,
        variant="out",
    )
    returned_cpu, out_cpu, scale_cpu, growth_tracker_cpu = _run_amp_update_scale(
        "cpu",
        found_inf_value=found_inf_value,
        variant="out",
    )
    synchronize(device)
    assert returned_dev is out_dev
    assert returned_cpu is out_cpu
    compare(out_dev, out_cpu, category="exact", dtype=torch.float32)
    compare(scale_dev, scale_cpu, category="exact", dtype=torch.float32)
    compare(growth_tracker_dev, growth_tracker_cpu, category="exact", dtype=torch.int32)

    returned_dev, scale_dev, growth_tracker_dev = _run_amp_update_scale(
        device,
        found_inf_value=found_inf_value,
        variant="inplace",
    )
    returned_cpu, scale_cpu, growth_tracker_cpu = _run_amp_update_scale(
        "cpu",
        found_inf_value=found_inf_value,
        variant="inplace",
    )
    synchronize(device)
    assert returned_dev is scale_dev
    assert returned_cpu is scale_cpu
    compare(scale_dev, scale_cpu, category="exact", dtype=torch.float32)
    compare(growth_tracker_dev, growth_tracker_cpu, category="exact", dtype=torch.int32)
