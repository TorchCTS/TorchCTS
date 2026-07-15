# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies or substantial portions of the Software.
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
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import torchcts.conftest as conftest
import torchcts.core.contract_references as contract_references
import torchcts.generated.coverage_helpers as coverage_helpers
import torchcts.opinfo.test_opinfo_forward as opinfo_forward
from torchcts.core.coverage import generated_entries_for
from torchcts.core.contract_references import (
    ContractReferenceError,
    resolve_generated_forward_reference,
    resolve_opinfo_forward_reference,
)
from torchcts.core.reference_oracles import (
    complex_convolution_reference,
    complex_l1_loss_reference,
    complex_log2_reference,
    complex_tensor_integer_power_reference,
    complex_unit_alpha_add_sub_reference,
    conv_transpose3d_f32_reference,
    embedding_bag_scale_grad_by_freq_reference,
    grid_sampler_3d_backward_f32_reference,
    slow_complex_convolution_reference,
)
from torchcts.core.opinfo_adapter import InputCondition, get_op_sample_inputs, prepare_sample
from torchcts.sample_generation import (
    elementwise_sample,
    foreach_sample,
    loss_sample,
)


def _sample(input_value, *args, **kwargs):
    return SimpleNamespace(input=input_value, args=args, kwargs=kwargs)


def _compare(actual, expected, **_kwargs):
    torch.testing.assert_close(actual.cpu(), expected.cpu(), equal_nan=True)


def _assert_complex_special_equal(actual, expected, *, rtol=2e-4, atol=2e-4):
    actual_lanes = torch.view_as_real(actual)
    expected_lanes = torch.view_as_real(expected)
    assert torch.equal(torch.isnan(actual_lanes), torch.isnan(expected_lanes))
    assert torch.equal(torch.isinf(actual_lanes), torch.isinf(expected_lanes))
    inf_mask = torch.isinf(actual_lanes)
    assert torch.equal(
        torch.signbit(actual_lanes[inf_mask]),
        torch.signbit(expected_lanes[inf_mask]),
    )
    finite_mask = torch.isfinite(actual_lanes) & torch.isfinite(expected_lanes)
    torch.testing.assert_close(
        actual_lanes[finite_mask],
        expected_lanes[finite_mask],
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_complex_unit_alpha_reference_keeps_lanes_independent(dtype):
    left = torch.complex(
        torch.tensor([0.0, 1.0]),
        torch.tensor([0.5, float("inf")]),
    ).to(dtype)
    right = torch.complex(
        torch.tensor([-float("inf"), 2.0]),
        torch.tensor([0.25, 3.0]),
    ).to(dtype)
    added = complex_unit_alpha_add_sub_reference("add", left, right)
    subtracted = complex_unit_alpha_add_sub_reference("sub", left, right)
    reversed_subtraction = complex_unit_alpha_add_sub_reference("rsub", left, right)
    assert torch.isneginf(added.real[0]) and added.imag[0] == 0.75
    assert torch.isposinf(subtracted.real[0]) and subtracted.imag[0] == 0.25
    assert torch.isneginf(reversed_subtraction.real[0]) and reversed_subtraction.imag[0] == -0.25


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("operation", ["add", "sub", "rsub"])
def test_complex_unit_alpha_reference_broadcasts_and_preserves_special_lanes(dtype, operation):
    left = torch.tensor(
        [[complex(-float("inf"), 0.5)], [complex(3.0, float("inf"))]],
        dtype=dtype,
    )
    right = torch.tensor(
        [[complex(2.0, -0.25), complex(float("nan"), 4.0)]],
        dtype=dtype,
    )
    result = complex_unit_alpha_add_sub_reference(operation, left, right)
    left_real, right_real = torch.broadcast_tensors(left.real, right.real)
    left_imag, right_imag = torch.broadcast_tensors(left.imag, right.imag)
    if operation == "add":
        expected = torch.complex(left_real + right_real, left_imag + right_imag)
    elif operation == "sub":
        expected = torch.complex(left_real - right_real, left_imag - right_imag)
    else:
        expected = torch.complex(right_real - left_real, right_imag - left_imag)
    _assert_complex_special_equal(result, expected, rtol=0, atol=0)
    assert result.shape == (2, 2)


def test_unit_alpha_router_is_closed_and_semantic():
    dtype = torch.complex64
    left = torch.tensor([1 + 2j], dtype=dtype)
    right = torch.tensor([3 + 4j], dtype=dtype)
    assert resolve_opinfo_forward_reference(
        "add", _sample(left, right, alpha=2), dtype, "has_inf"
    ) is None
    assert resolve_opinfo_forward_reference(
        "add", _sample(left, right, alpha=torch.tensor(1)), dtype, "has_inf"
    ) is None
    assert resolve_opinfo_forward_reference(
        "add", _sample(left, right), dtype, "clean"
    ) is None
    assert resolve_opinfo_forward_reference(
        "multiply", _sample(left, right), dtype, "has_inf"
    ) is None
    matched = resolve_generated_forward_reference(
        "aten::add_.Tensor", _sample(left, right), dtype, "has_inf"
    )
    assert matched is not None
    assert matched.reference_id == "complex_unit_alpha_add_sub"


@pytest.mark.parametrize(
    "dispatcher_name,surface_kind",
    [
        ("aten::rsub.Tensor", "functional_data"),
        ("aten::subtract_.Tensor", "mutating_or_inplace"),
        ("aten::subtract.out", "out_variant"),
    ],
)
def test_manual_elementwise_runner_uses_contract_reference_and_preserves_identity(
    monkeypatch,
    dispatcher_name,
    surface_kind,
):
    dtype = torch.complex64
    sample = _sample(
        torch.tensor([complex(float("inf"), 0.5)], dtype=dtype),
        torch.tensor([complex(2.0, -0.25)], dtype=dtype),
        alpha=1,
    )
    monkeypatch.setattr(coverage_helpers, "_elementwise_sample", lambda *_args, **_kwargs: sample)

    def reject_mask_only_comparison(*_args, **_kwargs):
        raise AssertionError("special-tier mask comparison must not run")

    monkeypatch.setattr(coverage_helpers, "_compare_special_tier", reject_mask_only_comparison)
    qualified_name = dispatcher_name.split("::", 1)[1]
    packet_name, overload_name = qualified_name.split(".", 1)
    callable_op = getattr(getattr(torch.ops.aten, packet_name), overload_name)
    entry = {
        "name": dispatcher_name,
        "surface_kind": surface_kind,
        "schema": f"{dispatcher_name}(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
    }
    assert coverage_helpers._run_manual_elementwise_case(
        entry,
        callable_op,
        dtype,
        InputCondition.HAS_INF,
        "cpu",
        _compare,
        {"ieee754_seed": 67},
    )


def test_matched_runner_execution_failure_names_reference_id(monkeypatch):
    dtype = torch.complex64
    sample = _sample(
        torch.tensor([complex(float("inf"), 0.5)], dtype=dtype),
        torch.tensor([complex(2.0, -0.25)], dtype=dtype),
        alpha=1,
    )
    monkeypatch.setattr(coverage_helpers, "_elementwise_sample", lambda *_args, **_kwargs: sample)

    def fail(*_args, **_kwargs):
        raise RuntimeError("target failed")

    with pytest.raises(RuntimeError, match="reference_id=complex_unit_alpha_add_sub"):
        coverage_helpers._run_manual_elementwise_case(
            {
                "name": "aten::rsub.Tensor",
                "surface_kind": "functional_data",
                "schema": "rsub",
            },
            fail,
            dtype,
            InputCondition.HAS_INF,
            "cpu",
            _compare,
            {"ieee754_seed": 67},
        )


@pytest.mark.parametrize(
    "failure_kind,match",
    [
        ("device", "device mismatch"),
        ("dtype", "dtype mismatch"),
        ("structure", "type mismatch"),
    ],
)
def test_opinfo_contract_runner_rejects_metadata_before_value_comparison(
    monkeypatch,
    failure_kind,
    match,
):
    dtype = torch.complex64
    sample = _sample(torch.tensor([complex(float("inf"), 1.0)], dtype=dtype))

    if failure_kind == "device":
        actual = torch.empty((1,), dtype=dtype, device="cpu")
    elif failure_kind == "dtype":
        actual = torch.empty((1,), dtype=torch.complex128, device="meta")
    else:
        actual = [torch.empty((1,), dtype=dtype, device="meta")]

    monkeypatch.setattr(opinfo_forward, "get_live_opinfo", lambda _name: SimpleNamespace(op=lambda *_a, **_k: actual))
    monkeypatch.setattr(opinfo_forward, "get_op_sample_inputs", lambda *_a, **_k: [sample])
    monkeypatch.setattr(opinfo_forward, "prepare_sample", lambda raw, *_a, **_k: raw)
    monkeypatch.setattr(opinfo_forward, "synchronize", lambda _device: None)
    monkeypatch.setattr(
        conftest,
        "_MANIFEST",
        {"max_samples": 1, "max_samples_ieee754": 1, "ieee754_seed": 67},
    )

    def reject_value_comparison(*_args, **_kwargs):
        raise AssertionError("value comparer must not run before metadata validation")

    request = SimpleNamespace(node=SimpleNamespace(nodeid="selftest-contract-metadata"))
    with pytest.raises(AssertionError, match=rf"complex_log2.*{match}"):
        opinfo_forward.test_op_forward(
            "log2",
            "torch.complex64",
            "has_inf",
            "meta",
            reject_value_comparison,
            request,
        )


@pytest.mark.parametrize(
    "failure_kind,match",
    [
        ("device", "complex_unit_alpha_add_sub.*device mismatch"),
        ("dtype", "complex_unit_alpha_add_sub.*dtype mismatch"),
        ("structure", "expected Tensor.*complex_unit_alpha_add_sub"),
    ],
)
def test_generated_contract_runner_rejects_metadata_before_value_comparison(
    monkeypatch,
    failure_kind,
    match,
):
    dtype = torch.complex64
    sample = _sample(
        torch.tensor([complex(float("inf"), 0.5)], dtype=dtype),
        torch.tensor([complex(2.0, -0.25)], dtype=dtype),
        alpha=1,
    )
    monkeypatch.setattr(coverage_helpers, "_elementwise_sample", lambda *_a, **_k: sample)
    monkeypatch.setattr(coverage_helpers, "synchronize", lambda _device: None)

    if failure_kind == "device":
        actual = torch.empty((1,), dtype=dtype, device="cpu")
    elif failure_kind == "dtype":
        actual = torch.empty((1,), dtype=torch.complex128, device="meta")
    else:
        actual = (torch.empty((1,), dtype=dtype, device="meta"),)

    def reject_value_comparison(*_args, **_kwargs):
        raise AssertionError("value comparer must not run before metadata validation")

    with pytest.raises(AssertionError, match=match):
        coverage_helpers._run_manual_elementwise_case(
            {
                "name": "aten::rsub.Tensor",
                "surface_kind": "functional_data",
                "schema": "rsub",
            },
            lambda *_a, **_k: actual,
            dtype,
            InputCondition.HAS_INF,
            "meta",
            reject_value_comparison,
            {"ieee754_seed": 67},
        )


def test_foreach_contract_runner_rejects_wrong_element_metadata(monkeypatch):
    dtype = torch.complex64
    sample = _sample([torch.tensor([complex(float("inf"), 1.0)], dtype=dtype)])
    entry = {
        "name": "aten::_foreach_log2",
        "base_name": "_foreach_log2",
        "surface_kind": "functional_data",
        "generated": {"strategy": {"strategy": "manual_foreach"}},
    }
    monkeypatch.setattr(coverage_helpers, "_manual_foreach_sample", lambda *_a, **_k: sample)
    monkeypatch.setattr(
        coverage_helpers,
        "_manual_input_conditions",
        lambda *_a, **_k: [InputCondition.HAS_INF],
    )
    monkeypatch.setattr(
        coverage_helpers,
        "_dispatcher_callable",
        lambda _entry: lambda *_a, **_k: [
            torch.empty((1,), dtype=torch.complex128, device="meta")
        ],
    )
    monkeypatch.setattr(coverage_helpers, "synchronize", lambda _device: None)

    def reject_value_comparison(*_args, **_kwargs):
        raise AssertionError("value comparer must not run before metadata validation")

    with pytest.raises(AssertionError, match="complex_log2.*dtype mismatch"):
        coverage_helpers.run_manual_foreach_strategy(
            entry,
            "meta",
            reject_value_comparison,
            {"capabilities": {"ieee754": True}, "ieee754_seed": 67},
            dtype=dtype,
        )


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_every_named_generated_contract_route_matches_materialized_sample(dtype):
    routed_names = {
        "aten::subtract.Tensor",
        "aten::add_.Tensor",
        "aten::add.out",
        "aten::sub_.Tensor",
        "aten::subtract_.Tensor",
        "aten::sub.out",
        "aten::subtract.out",
        "aten::rsub.Tensor",
        "aten::rsub.Tensor_out",
        "aten::float_power.Tensor_Tensor",
        "aten::float_power.Tensor_Tensor_out",
        "aten::float_power_.Tensor",
        "aten::pow_.Tensor",
        "aten::pow.Tensor_Tensor_out",
        "aten::l1_loss",
        "aten::log2_",
        "aten::log2.out",
        "aten::_foreach_log2",
        "aten::_foreach_log2_",
        "aten::_foreach_log2.out",
    }
    entries = {}
    for surface in ("functional_data", "mutating_or_inplace", "out_variant"):
        for entry in generated_entries_for(surface):
            if entry["name"] in routed_names:
                entries[entry["name"]] = entry
    assert set(entries) == routed_names

    raw_log2_sample = next(iter(get_op_sample_inputs("log2", "cpu", dtype)))
    log2_sample = prepare_sample(
        raw_log2_sample,
        InputCondition.HAS_INF,
        ieee754_seed=67,
        sample_index=0,
        op_name="log2",
    )
    for name, entry in entries.items():
        strategy = (entry.get("generated", {}).get("strategy") or {}).get("strategy")
        if strategy == "manual_elementwise":
            sample = elementwise_sample(entry, dtype, input_condition=InputCondition.HAS_INF)
        elif strategy == "manual_loss":
            sample = loss_sample(entry, dtype, input_condition=InputCondition.HAS_INF)
        elif strategy == "manual_foreach":
            sample = foreach_sample(entry, dtype, input_condition=InputCondition.HAS_INF)
        else:
            assert strategy in {"opinfo_out", "opinfo_inplace_unary"}
            sample = log2_sample
        resolved = resolve_generated_forward_reference(
            name,
            sample,
            dtype,
            InputCondition.HAS_INF,
        )
        if name == "aten::float_power_.Tensor" and dtype == torch.complex64:
            assert resolved is None
        else:
            assert resolved is not None, name


@pytest.mark.parametrize(
    "dispatcher_name,surface_kind",
    [
        ("aten::_foreach_log2", "functional_data"),
        ("aten::_foreach_log2_", "mutating_or_inplace"),
        ("aten::_foreach_log2.out", "out_variant"),
    ],
)
def test_foreach_contract_runner_preserves_surface_identity(dispatcher_name, surface_kind):
    entry = next(
        entry
        for entry in generated_entries_for(surface_kind)
        if entry["name"] == dispatcher_name
    )
    coverage_helpers.run_manual_foreach_strategy(
        entry,
        "cpu",
        _compare,
        {"capabilities": {"ieee754": True}, "ieee754_seed": 67},
        dtype=torch.complex64,
    )


@pytest.mark.parametrize("exponent", [0, 1, 2, 3])
def test_complex_integer_power_reference(exponent):
    base = torch.complex(
        torch.tensor([-float("inf"), 1.0682]),
        torch.tensor([0.0625, float("inf")]),
    ).to(torch.complex64)
    exponent_tensor = torch.full(base.shape, complex(exponent, 0), dtype=torch.complex64)
    native = torch.pow(base, exponent_tensor)
    result = complex_tensor_integer_power_reference(base, exponent_tensor, native)
    expected = torch.pow(base, exponent)
    torch.testing.assert_close(result, expected, equal_nan=True)


def test_complex_integer_power_leaves_general_exponents_native():
    base = torch.tensor([2 + 1j] * 8, dtype=torch.complex64)
    exponent = torch.tensor(
        [
            -1 + 0j,
            1.5 + 0j,
            2 + 1j,
            complex(float("nan"), 0),
            complex(float("inf"), 0),
            complex(2, float("nan")),
            complex(2, float("inf")),
            complex(-float("inf"), 0),
        ],
        dtype=torch.complex64,
    )
    native = torch.pow(base, exponent)
    result = complex_tensor_integer_power_reference(base, exponent, native)
    torch.testing.assert_close(result, native, equal_nan=True)


def test_float_power_router_preserves_promotion_and_inplace_legality():
    base = torch.tensor([-float("inf") + 0.25j], dtype=torch.complex64)
    exponent = torch.tensor([2 + 0j], dtype=torch.complex64)
    functional = resolve_generated_forward_reference(
        "aten::float_power.Tensor_Tensor", _sample(base, exponent), torch.complex64, "has_inf"
    )
    assert functional is not None and functional.value.dtype == torch.complex128
    assert resolve_generated_forward_reference(
        "aten::float_power_.Tensor", _sample(base, exponent), torch.complex64, "has_inf"
    ) is None

    finite_base = torch.tensor([1.0001 + 0.3333j], dtype=torch.complex64)
    cubic = torch.tensor([3 + 0j], dtype=torch.complex64)
    native = torch.float_power(finite_base, cubic)
    corrected = complex_tensor_integer_power_reference(finite_base, cubic, native)
    base_c128 = finite_base.to(torch.complex128)
    expected = torch.mul(base_c128, torch.mul(base_c128, base_c128))
    assert corrected.dtype == torch.complex128
    torch.testing.assert_close(corrected, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_complex_l1_and_log2_references(dtype):
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    left = torch.complex(
        torch.tensor([float("inf"), 3.0], dtype=real_dtype),
        torch.tensor([-float("inf"), 4.0], dtype=real_dtype),
    )
    right = torch.complex(
        torch.tensor([-float("inf"), 0.0], dtype=real_dtype),
        torch.tensor([float("inf"), 0.0], dtype=real_dtype),
    )
    values = complex_l1_loss_reference(left, right, "none")
    assert torch.isinf(values[0])
    assert values[1] == 5

    finite = torch.tensor([3 + 4j, 6 + 8j], dtype=dtype)
    zeros = torch.zeros(2, dtype=real_dtype)
    torch.testing.assert_close(
        complex_l1_loss_reference(finite, zeros, "none"),
        torch.tensor([5.0, 10.0], dtype=real_dtype),
    )
    assert complex_l1_loss_reference(finite, zeros, "sum") == 15
    assert complex_l1_loss_reference(finite, zeros, "mean") == 7.5
    torch.testing.assert_close(
        complex_l1_loss_reference(finite, zeros, 0),
        torch.tensor([5.0, 10.0], dtype=real_dtype),
    )
    assert complex_l1_loss_reference(finite, zeros, 2) == 15
    assert complex_l1_loss_reference(finite, zeros, 1) == 7.5

    mixed = complex_l1_loss_reference(
        torch.tensor([3 + 4j], dtype=dtype),
        torch.tensor([0.0], dtype=real_dtype),
        "none",
    )
    assert mixed.dtype == real_dtype and mixed.item() == 5

    value = torch.complex(
        torch.tensor([float("inf"), -float("inf")], dtype=real_dtype),
        torch.tensor([1.0, 1.0], dtype=real_dtype),
    )
    result = complex_log2_reference(value)
    assert torch.isposinf(result.real).all()
    assert torch.isfinite(result.imag).all()
    torch.testing.assert_close(
        result.imag,
        torch.tensor([0.0, math.pi / math.log(2.0)], dtype=real_dtype),
    )


def test_matched_reference_failure_never_falls_back(monkeypatch):
    value = torch.tensor([complex(float("inf"), 1.0)], dtype=torch.complex64)

    def fail(_value):
        raise RuntimeError("intentional oracle failure")

    monkeypatch.setattr(contract_references, "complex_log2_reference", fail)
    with pytest.raises(ContractReferenceError, match="complex_log2"):
        resolve_opinfo_forward_reference("log2", _sample(value), torch.complex64, "has_inf")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("padding_mode", [0, 1, 2])
@pytest.mark.parametrize("align_corners", [False, True])
def test_grid_sampler_f32_reference_matches_f32_autograd_matrix(
    dtype,
    padding_mode,
    align_corners,
):
    input_tensor = torch.linspace(-1.0, 1.0, 24, dtype=torch.float32).reshape(1, 1, 2, 3, 4).to(dtype)
    grid = torch.tensor(
        [[[[[-1.25, -0.5, 0.25], [0.75, 1.2, -0.8]]]]],
        dtype=dtype,
    )
    grad_output = torch.tensor([[[[[0.75, -1.25]]]]], dtype=dtype)
    actual = grid_sampler_3d_backward_f32_reference(
        grad_output,
        input_tensor,
        grid,
        0,
        padding_mode,
        align_corners,
    )

    input_f32 = input_tensor.float().requires_grad_(True)
    grid_f32 = grid.float().requires_grad_(True)
    output_f32 = torch.ops.aten.grid_sampler_3d.default(
        input_f32,
        grid_f32,
        0,
        padding_mode,
        align_corners,
    )
    expected_f32 = torch.autograd.grad(
        output_f32,
        (input_f32, grid_f32),
        grad_outputs=grad_output.float(),
    )
    expected = tuple(item.to(dtype) for item in expected_f32)
    for actual_item, expected_item in zip(actual, expected):
        torch.testing.assert_close(actual_item, expected_item, rtol=0, atol=0)


@pytest.mark.parametrize(
    "input_shape,weight_shape,use_bias,kwargs",
    [
        (
            (1, 2, 2, 2, 2),
            (2, 3, 2, 2, 2),
            True,
            {"stride": 2, "padding": 1, "output_padding": 1},
        ),
        (
            (2, 2, 2, 2),
            (2, 1, 2, 2, 2),
            False,
            {"groups": 2},
        ),
        (
            (1, 4, 2, 2, 2),
            (4, 2, 2, 2, 2),
            True,
            {
                "stride": (2, 2, 2),
                "padding": (1, 1, 1),
                "output_padding": (1, 1, 1),
                "groups": 2,
                "dilation": (2, 2, 2),
            },
        ),
    ],
)
def test_bf16_conv_transpose3d_reference_matrix(input_shape, weight_shape, use_bias, kwargs):
    input_tensor = torch.linspace(-1, 1, torch.tensor(input_shape).prod().item()).reshape(input_shape).to(torch.bfloat16)
    weight = torch.linspace(-0.5, 0.75, torch.tensor(weight_shape).prod().item()).reshape(weight_shape).to(torch.bfloat16)
    out_channels = weight_shape[1] * kwargs.get("groups", 1)
    bias = torch.linspace(-0.25, 0.25, out_channels).to(torch.bfloat16) if use_bias else None
    actual = conv_transpose3d_f32_reference(input_tensor, weight, bias, **kwargs)
    expected = F.conv_transpose3d(
        input_tensor.float(),
        weight.float(),
        None if bias is None else bias.float(),
        **kwargs,
    ).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_low_precision_reference_routing_is_exact_and_closed():
    input_tensor = torch.zeros((1, 1, 2, 2, 2), dtype=torch.bfloat16)
    weight = torch.zeros((1, 1, 2, 2, 2), dtype=torch.bfloat16)
    sample = _sample(input_tensor, weight, None)
    assert resolve_opinfo_forward_reference(
        "nn.functional.conv_transpose3d",
        sample,
        torch.bfloat16,
        "clean",
    ) is not None
    assert resolve_opinfo_forward_reference(
        "nn.functional.conv_transpose3d",
        _sample(input_tensor.float(), weight.float(), None),
        torch.float32,
        "clean",
    ) is None
    assert resolve_opinfo_forward_reference(
        "nn.functional.conv3d",
        sample,
        torch.bfloat16,
        "clean",
    ) is None

    for name in (
        "aten::grid_sampler_3d_backward",
        "aten::grid_sampler_3d_backward.out",
    ):
        assert coverage_helpers._uses_f32_grid_backward_reference(
            {"name": name},
            torch.float16,
        )
        assert coverage_helpers._uses_f32_grid_backward_reference(
            {"name": name},
            torch.bfloat16,
        )
    assert not coverage_helpers._uses_f32_grid_backward_reference(
        {"name": "aten::grid_sampler_2d_backward"},
        torch.float16,
    )
    assert not coverage_helpers._uses_f32_grid_backward_reference(
        {"name": "aten::grid_sampler_3d_backward"},
        torch.float32,
    )


def test_matched_grid_backward_reference_failure_cannot_become_a_skip(monkeypatch):
    sample = SimpleNamespace(
        call_args=lambda: (
            torch.empty((1, 1, 1, 1, 1), dtype=torch.float16),
            torch.empty((1, 1, 2, 2, 2), dtype=torch.float16),
            torch.empty((1, 1, 1, 1, 3), dtype=torch.float16),
            0,
            0,
            False,
        )
    )
    monkeypatch.setattr(coverage_helpers, "sample_grid_backward", lambda *_a, **_k: sample)

    def fail_reference(*_args, **_kwargs):
        raise RuntimeError("intentional f32 oracle failure")

    monkeypatch.setattr(
        coverage_helpers,
        "grid_sampler_3d_backward_f32_reference",
        fail_reference,
    )

    def reject_target_execution(*_args, **_kwargs):
        raise AssertionError("target execution must not replace a failed reference")

    with pytest.raises(ContractReferenceError, match="grid_sampler_3d_backward_f32"):
        coverage_helpers._run_manual_grid_backward_case(
            {
                "name": "aten::grid_sampler_3d_backward",
                "surface_kind": "functional_data",
                "schema": "grid_sampler_3d_backward",
            },
            reject_target_execution,
            torch.float16,
            "cpu",
            _compare,
            {"ieee754_seed": 67},
        )


def test_embedding_bag_frequency_reference_uses_each_rows_count():
    grad = torch.tensor([[1.0, -0.5], [-0.25, 2.0]])
    indices = torch.tensor([1, 1, 2, 1, 2, 4])
    offset2bag = torch.tensor([0, 0, 0, 1, 1, 1])
    result = embedding_bag_scale_grad_by_freq_reference(grad, indices, offset2bag, 5)
    expected = torch.tensor([
        [0.0, 0.0],
        [1.75 / 3.0, 1.0 / 3.0],
        [0.375, 0.75],
        [0.0, 0.0],
        [-0.25, 2.0],
    ])
    torch.testing.assert_close(result, expected)
    assert torch.equal(result[[0, 3]], torch.zeros((2, 2), dtype=result.dtype))


@pytest.mark.parametrize(
    "op_name,positional_options,keyword_options",
    [
        (
            "nn.functional.conv1d",
            ((1,), (1,), (1,), 1),
            {"stride": (1,), "padding": (1,), "dilation": (1,), "groups": 1},
        ),
        (
            "nn.functional.conv_transpose1d",
            ((2,), (1,), (1,), 1, (1,)),
            {
                "stride": (2,),
                "padding": (1,),
                "output_padding": (1,),
                "groups": 1,
                "dilation": (1,),
            },
        ),
    ],
)
def test_complex_convolution_router_normalizes_public_argument_forms(
    op_name,
    positional_options,
    keyword_options,
):
    dtype = torch.complex64
    input_tensor = torch.tensor(
        [[[complex(float("inf"), 0.5), 1 + 2j, 3 + 4j]]],
        dtype=dtype,
    )
    weight = torch.tensor([[[1 - 0.5j, 2 + 0.25j]]], dtype=dtype)
    bias = torch.tensor([0.5 - 0.25j], dtype=dtype)
    positional = _sample(input_tensor, weight, bias, *positional_options)
    keyword = _sample(
        input_tensor,
        weight=weight,
        bias=bias,
        **keyword_options,
    )
    positional_reference = resolve_opinfo_forward_reference(
        op_name,
        positional,
        dtype,
        "has_inf",
    )
    keyword_reference = resolve_opinfo_forward_reference(
        op_name,
        keyword,
        dtype,
        "has_inf",
    )
    assert positional_reference is not None
    assert keyword_reference is not None
    _assert_complex_special_equal(positional_reference.value, keyword_reference.value)

    assert resolve_opinfo_forward_reference(
        op_name,
        _sample(input_tensor, weight, weight=weight),
        dtype,
        "has_inf",
    ) is None
    assert resolve_opinfo_forward_reference(
        op_name,
        _sample(input_tensor, weight=weight, unsupported=True),
        dtype,
        "has_inf",
    ) is None


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("case_name", ["batched_bias_int", "unbatched_no_bias_tuple"])
@pytest.mark.parametrize(
    "op_name,rank,transposed",
    [
        ("nn.functional.conv1d", 1, False),
        ("nn.functional.conv2d", 2, False),
        ("nn.functional.conv3d", 3, False),
        ("nn.functional.conv_transpose1d", 1, True),
        ("nn.functional.conv_transpose2d", 2, True),
        ("nn.functional.conv_transpose3d", 3, True),
    ],
)
def test_complex_convolution_fast_reference_is_proved_by_termwise_oracle(
    dtype,
    case_name,
    op_name,
    rank,
    transposed,
):
    generator = torch.Generator().manual_seed(0)
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    kernel = (2,) * rank
    if case_name == "batched_bias_int":
        input_shape = (1, 2, *((3,) * rank))
        weight_shape = (2, 2, *kernel)
        kwargs = {"groups": 1, "padding": 1, "stride": 1, "dilation": 1}
        if transposed:
            kwargs.update(stride=2, output_padding=1)
        use_bias = True
    else:
        spatial = (2,) * rank if transposed else (4,) * rank
        input_shape = (2, *spatial)
        weight_shape = (2, 1, *kernel)
        kwargs = {
            "groups": 2,
            "padding": (1,) * rank,
            "stride": ((2,) * rank if transposed else (1,) * rank),
            "dilation": (2,) * rank,
        }
        if transposed:
            kwargs["output_padding"] = (1,) * rank
        use_bias = False

    input_tensor = torch.complex(
        torch.randn(input_shape, generator=generator, dtype=real_dtype),
        torch.randn(input_shape, generator=generator, dtype=real_dtype),
    ).to(dtype)
    weight = torch.complex(
        torch.randn(weight_shape, generator=generator, dtype=real_dtype),
        torch.randn(weight_shape, generator=generator, dtype=real_dtype),
    ).to(dtype)
    bias = None
    if use_bias:
        bias = torch.complex(
            torch.randn(2, generator=generator, dtype=real_dtype),
            torch.randn(2, generator=generator, dtype=real_dtype),
        ).to(dtype)

    function = getattr(F, op_name.rsplit(".", 1)[-1])
    finite_fast = complex_convolution_reference(op_name, input_tensor, weight, bias, kwargs)
    finite_native = function(input_tensor, weight, bias, **kwargs)
    torch.testing.assert_close(finite_fast, finite_native, rtol=2e-4, atol=2e-4)

    for special_value in (complex(float("inf"), 0.25), complex(float("nan"), -0.5)):
        special_input = input_tensor.clone()
        special_input.reshape(-1)[0] = special_value
        fast = complex_convolution_reference(op_name, special_input, weight, bias, kwargs)
        slow = slow_complex_convolution_reference(op_name, special_input, weight, bias, kwargs)
        _assert_complex_special_equal(fast, slow)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize(
    "op_name",
    [
        "nn.functional.conv1d",
        "nn.functional.conv2d",
        "nn.functional.conv3d",
        "nn.functional.conv_transpose1d",
        "nn.functional.conv_transpose2d",
        "nn.functional.conv_transpose3d",
    ],
)
def test_complex_convolution_first_three_live_opinfo_samples_prove_runner_reference(dtype, op_name):
    function = getattr(F, op_name.rsplit(".", 1)[-1])
    samples = list(get_op_sample_inputs(op_name, "cpu", dtype))[:3]
    assert len(samples) == 3
    for sample_index, raw_sample in enumerate(samples):
        weight = raw_sample.args[0]
        bias = raw_sample.args[1] if len(raw_sample.args) > 1 else None
        finite_reference = complex_convolution_reference(
            op_name,
            raw_sample.input,
            weight,
            bias,
            dict(raw_sample.kwargs),
        )
        finite_native = function(raw_sample.input, *raw_sample.args, **raw_sample.kwargs)
        torch.testing.assert_close(finite_reference, finite_native, rtol=2e-4, atol=2e-4)

        for condition in (InputCondition.HAS_INF, InputCondition.HAS_NAN):
            sample = prepare_sample(
                raw_sample,
                condition,
                ieee754_seed=67,
                sample_index=sample_index,
                op_name=op_name,
            )
            resolved = resolve_opinfo_forward_reference(
                op_name,
                sample,
                dtype,
                condition,
            )
            assert resolved is not None
            assert resolved.reference_id == "complex_convolution_four_real"
            fast = resolved.value
            slow = slow_complex_convolution_reference(
                op_name,
                sample.input,
                sample.args[0],
                sample.args[1] if len(sample.args) > 1 else None,
                dict(sample.kwargs),
            )
            _assert_complex_special_equal(fast, slow)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize(
    "op_name,rank",
    [
        ("nn.functional.conv1d", 1),
        ("nn.functional.conv2d", 2),
        ("nn.functional.conv3d", 3),
    ],
)
@pytest.mark.parametrize("padding", ["same", "valid"])
def test_complex_convolution_string_padding_is_proved_by_termwise_oracle(
    dtype,
    op_name,
    rank,
    padding,
):
    generator = torch.Generator().manual_seed(1)
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    input_shape = (1, *((4,) * rank))
    weight_shape = (1, 1, *((2,) * rank))
    input_tensor = torch.complex(
        torch.randn(input_shape, generator=generator, dtype=real_dtype),
        torch.randn(input_shape, generator=generator, dtype=real_dtype),
    ).to(dtype)
    weight = torch.complex(
        torch.randn(weight_shape, generator=generator, dtype=real_dtype),
        torch.randn(weight_shape, generator=generator, dtype=real_dtype),
    ).to(dtype)
    kwargs = {"padding": padding}
    function = getattr(F, op_name.rsplit(".", 1)[-1])
    finite = complex_convolution_reference(op_name, input_tensor, weight, None, kwargs)
    native = function(input_tensor, weight, None, **kwargs)
    torch.testing.assert_close(finite, native, rtol=2e-4, atol=2e-4)

    for special_value in (complex(float("inf"), 0.25), complex(float("nan"), -0.5)):
        special_input = input_tensor.clone()
        special_input.reshape(-1)[0] = special_value
        fast = complex_convolution_reference(op_name, special_input, weight, None, kwargs)
        slow = slow_complex_convolution_reference(op_name, special_input, weight, None, kwargs)
        _assert_complex_special_equal(fast, slow)
