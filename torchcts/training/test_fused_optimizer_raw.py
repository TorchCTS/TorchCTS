# Copyright (c) 2026 Kris Bailey. MIT License.

import pytest
import torch

from torchcts.core.device import synchronize


def _params_and_grads(device):
    params = [
        torch.tensor([1.0, 2.0], dtype=torch.float32, device=device),
        torch.tensor([3.0, 4.0], dtype=torch.float32, device=device),
    ]
    grads = [
        torch.tensor([0.1, 0.2], dtype=torch.float32, device=device),
        torch.tensor([0.3, 0.4], dtype=torch.float32, device=device),
    ]
    return params, grads


def _zeros_like_params(params):
    return [torch.zeros_like(param) for param in params]


def _step_tensors(device):
    return [torch.tensor(1.0, dtype=torch.float32, device=device), torch.tensor(1.0, dtype=torch.float32, device=device)]


def _compare_tensor_lists(actual, expected, compare):
    assert len(actual) == len(expected)
    for actual_item, expected_item in zip(actual, expected):
        compare(actual_item, expected_item, category="optimizer", dtype=torch.float32)


def _compare_return_tuple(actual, expected, compare):
    assert len(actual) == len(expected)
    for actual_items, expected_items in zip(actual, expected):
        _compare_tensor_lists(actual_items, expected_items, compare)


@pytest.mark.medium
@pytest.mark.requires("fused_optimizer")
@pytest.mark.covers("aten::_fused_sgd")
@pytest.mark.covers("aten::_fused_sgd.tensor_lr")
@pytest.mark.covers("aten::_fused_sgd.out")
@pytest.mark.covers("aten::_fused_sgd.tensor_lr_out")
@pytest.mark.covers("aten::_fused_sgd_")
@pytest.mark.covers("aten::_fused_sgd_.tensor_lr")
def test_raw_fused_sgd_dispatcher_variants(device, compare):
    def functional(target_device, tensor_lr=False):
        params, grads = _params_and_grads(target_device)
        lr = torch.tensor(0.1, dtype=torch.float32, device=target_device) if tensor_lr else 0.1
        op = torch.ops.aten._fused_sgd.tensor_lr if tensor_lr else torch.ops.aten._fused_sgd
        return op(
            params,
            grads,
            [],
            weight_decay=0.0,
            momentum=0.0,
            lr=lr,
            dampening=0.0,
            nesterov=False,
            maximize=False,
            is_first_step=True,
        )

    _compare_return_tuple(functional(device), functional("cpu"), compare)
    _compare_return_tuple(functional(device, tensor_lr=True), functional("cpu", tensor_lr=True), compare)

    for tensor_lr in (False, True):
        params_dev, grads_dev = _params_and_grads(device)
        params_cpu, grads_cpu = _params_and_grads("cpu")
        out_dev = [torch.empty_like(param) for param in params_dev]
        out_cpu = [torch.empty_like(param) for param in params_cpu]
        lr_dev = torch.tensor(0.1, dtype=torch.float32, device=device) if tensor_lr else 0.1
        lr_cpu = torch.tensor(0.1, dtype=torch.float32) if tensor_lr else 0.1
        op = torch.ops.aten._fused_sgd.tensor_lr_out if tensor_lr else torch.ops.aten._fused_sgd.out
        assert op(
            params_dev,
            grads_dev,
            [],
            weight_decay=0.0,
            momentum=0.0,
            lr=lr_dev,
            dampening=0.0,
            nesterov=False,
            maximize=False,
            is_first_step=True,
            out=out_dev,
        ) is None
        assert op(
            params_cpu,
            grads_cpu,
            [],
            weight_decay=0.0,
            momentum=0.0,
            lr=lr_cpu,
            dampening=0.0,
            nesterov=False,
            maximize=False,
            is_first_step=True,
            out=out_cpu,
        ) is None
        synchronize(device)
        _compare_tensor_lists(out_dev, out_cpu, compare)

    for tensor_lr in (False, True):
        params_dev, grads_dev = _params_and_grads(device)
        params_cpu, grads_cpu = _params_and_grads("cpu")
        lr_dev = torch.tensor(0.1, dtype=torch.float32, device=device) if tensor_lr else 0.1
        lr_cpu = torch.tensor(0.1, dtype=torch.float32) if tensor_lr else 0.1
        op = torch.ops.aten._fused_sgd_.tensor_lr if tensor_lr else torch.ops.aten._fused_sgd_
        assert op(
            params_dev,
            grads_dev,
            [],
            weight_decay=0.0,
            momentum=0.0,
            lr=lr_dev,
            dampening=0.0,
            nesterov=False,
            maximize=False,
            is_first_step=True,
        ) is None
        assert op(
            params_cpu,
            grads_cpu,
            [],
            weight_decay=0.0,
            momentum=0.0,
            lr=lr_cpu,
            dampening=0.0,
            nesterov=False,
            maximize=False,
            is_first_step=True,
        ) is None
        synchronize(device)
        _compare_tensor_lists(params_dev, params_cpu, compare)


def _run_adagrad(target_device, tensor_lr=False, inplace=False, out=False):
    params, grads = _params_and_grads(target_device)
    state_sums = _zeros_like_params(params)
    state_steps = _step_tensors(target_device)
    lr = torch.tensor(0.1, dtype=torch.float32, device=target_device) if tensor_lr else 0.1
    kwargs = dict(lr_decay=0.0, weight_decay=0.0, eps=1e-10, maximize=False)
    if out:
        outputs = [torch.empty_like(param) for param in params]
        op = torch.ops.aten._fused_adagrad.tensor_lr_out if tensor_lr else torch.ops.aten._fused_adagrad.out
        returned = op(params, grads, state_sums, state_steps, lr=lr, out=outputs, **kwargs)
        return returned, outputs, state_sums
    if inplace:
        op = torch.ops.aten._fused_adagrad_.tensor_lr if tensor_lr else torch.ops.aten._fused_adagrad_
        returned = op(params, grads, state_sums, state_steps, lr=lr, **kwargs)
        return returned, params, state_sums
    op = torch.ops.aten._fused_adagrad.tensor_lr if tensor_lr else torch.ops.aten._fused_adagrad
    return op(params, grads, state_sums, state_steps, lr=lr, **kwargs)


@pytest.mark.medium
@pytest.mark.requires("fused_optimizer")
@pytest.mark.covers("aten::_fused_adagrad")
@pytest.mark.covers("aten::_fused_adagrad.tensor_lr")
@pytest.mark.covers("aten::_fused_adagrad.out")
@pytest.mark.covers("aten::_fused_adagrad.tensor_lr_out")
@pytest.mark.covers("aten::_fused_adagrad_")
@pytest.mark.covers("aten::_fused_adagrad_.tensor_lr")
def test_raw_fused_adagrad_dispatcher_variants(device, compare):
    _compare_return_tuple(_run_adagrad(device), _run_adagrad("cpu"), compare)
    _compare_return_tuple(_run_adagrad(device, tensor_lr=True), _run_adagrad("cpu", tensor_lr=True), compare)

    for tensor_lr in (False, True):
        returned_dev, out_dev, state_dev = _run_adagrad(device, tensor_lr=tensor_lr, out=True)
        returned_cpu, out_cpu, state_cpu = _run_adagrad("cpu", tensor_lr=tensor_lr, out=True)
        assert returned_dev is None and returned_cpu is None
        synchronize(device)
        _compare_tensor_lists(out_dev, out_cpu, compare)
        _compare_tensor_lists(state_dev, state_cpu, compare)

    for tensor_lr in (False, True):
        returned_dev, params_dev, state_dev = _run_adagrad(device, tensor_lr=tensor_lr, inplace=True)
        returned_cpu, params_cpu, state_cpu = _run_adagrad("cpu", tensor_lr=tensor_lr, inplace=True)
        assert returned_dev is None and returned_cpu is None
        synchronize(device)
        _compare_tensor_lists(params_dev, params_cpu, compare)
        _compare_tensor_lists(state_dev, state_cpu, compare)


def _run_adam_family(target_device, *, adamw: bool, tensor_lr=False, inplace=False, out=False):
    params, grads = _params_and_grads(target_device)
    exp_avgs = _zeros_like_params(params)
    exp_avg_sqs = _zeros_like_params(params)
    max_exp_avg_sqs = []
    state_steps = _step_tensors(target_device)
    lr = torch.tensor(0.1, dtype=torch.float32, device=target_device) if tensor_lr else 0.1
    kwargs = dict(beta1=0.9, beta2=0.999, weight_decay=0.0, eps=1e-8, amsgrad=False, maximize=False)
    base = torch.ops.aten._fused_adamw if adamw else torch.ops.aten._fused_adam
    base_inplace = torch.ops.aten._fused_adamw_ if adamw else torch.ops.aten._fused_adam_
    if out:
        outputs = [torch.empty_like(param) for param in params]
        op = base.tensor_lr_out if tensor_lr else base.out
        returned = op(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr=lr, out=outputs, **kwargs)
        return returned, outputs, exp_avgs, exp_avg_sqs
    if inplace:
        op = base_inplace.tensor_lr if tensor_lr else base_inplace
        returned = op(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr=lr, **kwargs)
        return returned, params, exp_avgs, exp_avg_sqs
    op = base.tensor_lr if tensor_lr else base
    return op(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr=lr, **kwargs)


@pytest.mark.medium
@pytest.mark.requires("fused_optimizer")
@pytest.mark.covers("aten::_fused_adam")
@pytest.mark.covers("aten::_fused_adam.tensor_lr")
@pytest.mark.covers("aten::_fused_adam.out")
@pytest.mark.covers("aten::_fused_adam.tensor_lr_out")
@pytest.mark.covers("aten::_fused_adam_")
@pytest.mark.covers("aten::_fused_adam_.tensor_lr")
def test_raw_fused_adam_dispatcher_variants(device, compare):
    _compare_return_tuple(_run_adam_family(device, adamw=False), _run_adam_family("cpu", adamw=False), compare)
    _compare_return_tuple(
        _run_adam_family(device, adamw=False, tensor_lr=True),
        _run_adam_family("cpu", adamw=False, tensor_lr=True),
        compare,
    )
    for tensor_lr in (False, True):
        returned_dev, out_dev, avg_dev, sq_dev = _run_adam_family(device, adamw=False, tensor_lr=tensor_lr, out=True)
        returned_cpu, out_cpu, avg_cpu, sq_cpu = _run_adam_family("cpu", adamw=False, tensor_lr=tensor_lr, out=True)
        assert returned_dev is None and returned_cpu is None
        synchronize(device)
        _compare_tensor_lists(out_dev, out_cpu, compare)
        _compare_tensor_lists(avg_dev, avg_cpu, compare)
        _compare_tensor_lists(sq_dev, sq_cpu, compare)
    for tensor_lr in (False, True):
        returned_dev, params_dev, avg_dev, sq_dev = _run_adam_family(device, adamw=False, tensor_lr=tensor_lr, inplace=True)
        returned_cpu, params_cpu, avg_cpu, sq_cpu = _run_adam_family("cpu", adamw=False, tensor_lr=tensor_lr, inplace=True)
        assert returned_dev is None and returned_cpu is None
        synchronize(device)
        _compare_tensor_lists(params_dev, params_cpu, compare)
        _compare_tensor_lists(avg_dev, avg_cpu, compare)
        _compare_tensor_lists(sq_dev, sq_cpu, compare)


@pytest.mark.medium
@pytest.mark.requires("fused_optimizer")
@pytest.mark.covers("aten::_fused_adamw")
@pytest.mark.covers("aten::_fused_adamw.tensor_lr")
@pytest.mark.covers("aten::_fused_adamw.out")
@pytest.mark.covers("aten::_fused_adamw.tensor_lr_out")
@pytest.mark.covers("aten::_fused_adamw_")
@pytest.mark.covers("aten::_fused_adamw_.tensor_lr")
def test_raw_fused_adamw_dispatcher_variants(device, compare):
    _compare_return_tuple(_run_adam_family(device, adamw=True), _run_adam_family("cpu", adamw=True), compare)
    _compare_return_tuple(
        _run_adam_family(device, adamw=True, tensor_lr=True),
        _run_adam_family("cpu", adamw=True, tensor_lr=True),
        compare,
    )
    for tensor_lr in (False, True):
        returned_dev, out_dev, avg_dev, sq_dev = _run_adam_family(device, adamw=True, tensor_lr=tensor_lr, out=True)
        returned_cpu, out_cpu, avg_cpu, sq_cpu = _run_adam_family("cpu", adamw=True, tensor_lr=tensor_lr, out=True)
        assert returned_dev is None and returned_cpu is None
        synchronize(device)
        _compare_tensor_lists(out_dev, out_cpu, compare)
        _compare_tensor_lists(avg_dev, avg_cpu, compare)
        _compare_tensor_lists(sq_dev, sq_cpu, compare)
    for tensor_lr in (False, True):
        returned_dev, params_dev, avg_dev, sq_dev = _run_adam_family(device, adamw=True, tensor_lr=tensor_lr, inplace=True)
        returned_cpu, params_cpu, avg_cpu, sq_cpu = _run_adam_family("cpu", adamw=True, tensor_lr=tensor_lr, inplace=True)
        assert returned_dev is None and returned_cpu is None
        synchronize(device)
        _compare_tensor_lists(params_dev, params_cpu, compare)
        _compare_tensor_lists(avg_dev, avg_cpu, compare)
        _compare_tensor_lists(sq_dev, sq_cpu, compare)
