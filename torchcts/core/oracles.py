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

"""TorchCTS-owned coverage oracles and backend-pack metadata.

The coverage ledger uses this module for exact dispatcher surfaces that are not
well represented by OpInfo or the generic generated strategies.  A registered
surface is allowed to move out of the old broad exclusion bucket only when this
module publishes an executable disposition for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from torchcts.core.oracle_assertions import (
    assert_close_tensor as _assert_close_tensor,
    assert_out_identity,
    assert_same_tensor as _assert_same_tensor,
)
from torchcts.core.reference_oracles import (
    dynamic_int4_matmul_reference,
    linear_backward_reference,
    max_pool2d_backward_reference,
    pack_int4_values,
    tinygemm_int4_matmul_reference,
)


class OracleUnavailable(RuntimeError):
    """Raised when an oracle exists but cannot run on the current host/device."""


@dataclass(frozen=True)
class OracleSpec:
    """Coverage disposition for a dispatcher surface with custom handling."""

    surface: str
    oracle_id: str
    coverage_status: str
    coverage_kind: str
    runner: str
    backend_gate: str = "any"
    semantic_level: int = 5
    reason: str = ""

    def metadata(self) -> dict:
        return {
            "oracle_id": self.oracle_id,
            "coverage_kind": self.coverage_kind,
            "backend_gate": self.backend_gate,
            "reason": self.reason,
            "runner": self.runner,
        }


def _supports_quantized_engine() -> bool:
    return any(engine != "none" for engine in torch.backends.quantized.supported_engines)


def _select_quantized_engine() -> str:
    for preferred in ("fbgemm", "qnnpack"):
        if preferred in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = preferred
            return preferred
    for engine in torch.backends.quantized.supported_engines:
        if engine != "none":
            torch.backends.quantized.engine = engine
            return engine
    raise OracleUnavailable("backend_not_available: no quantized engine is available")


def _privateuse1_backend_name() -> str:
    try:
        return torch._C._get_privateuse1_backend_name()
    except Exception:
        return "privateuseone"


def _is_privateuse1_device_type(device_type: str) -> bool:
    return device_type in {"privateuseone", _privateuse1_backend_name()}


def _check_backend_gate(spec: OracleSpec, device: str) -> None:
    device_type = torch.device(device).type
    gate = spec.backend_gate
    if gate == "any":
        return
    if gate == "cpu":
        if device_type != "cpu":
            raise OracleUnavailable(f"backend_not_available: {spec.surface} requires CPU")
        return
    if gate == "mps":
        if device_type != "mps" or not torch.backends.mps.is_available():
            raise OracleUnavailable(f"backend_not_available: {spec.surface} requires MPS")
        return
    if gate == "cuda":
        if device_type != "cuda" or not torch.cuda.is_available():
            raise OracleUnavailable(f"backend_not_available: {spec.surface} requires CUDA")
        return
    if gate == "privateuse1":
        if not _is_privateuse1_device_type(device_type):
            raise OracleUnavailable(f"backend_not_available: {spec.surface} requires a PrivateUse1 backend")
        return
    if gate == "quantized":
        if not _supports_quantized_engine():
            raise OracleUnavailable(f"backend_not_available: {spec.surface} requires a quantized engine")
        return
    if gate == "fbgemm":
        if "fbgemm" not in torch.backends.quantized.supported_engines:
            raise OracleUnavailable(f"backend_not_available: {spec.surface} requires FBGEMM")
        return
    raise OracleUnavailable(f"backend_not_available: unsupported oracle backend gate {gate!r}")


def _raise_backend_unavailable_if_applicable(spec: OracleSpec, exc: Exception) -> None:
    message = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
    unavailable_fragments = (
        "Could not run",
        "not currently supported",
        "not implemented",
        "not supported on",
        "only available for these backends",
        "requires MPS",
        "requires CUDA",
        "not enabled for build",
        "only enabled with aotriton",
        "was not enabled for build",
    )
    if isinstance(exc, (NotImplementedError, RuntimeError)) and any(
        fragment in message for fragment in unavailable_fragments
    ):
        raise OracleUnavailable(f"backend_not_available: {spec.surface}: {message}") from exc
    raise exc


def _run_sobol(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    from torch.quasirandom import SobolEngine

    if spec.surface == "aten::_sobol_engine_initialize_state_":
        state = torch.zeros(3, SobolEngine.MAXBIT, dtype=torch.long)
        actual = torch.ops.aten._sobol_engine_initialize_state_(state, 3)
        expected = SobolEngine(3, scramble=False, seed=0).sobolstate
        if actual is not state:
            raise AssertionError(f"{spec.surface} did not return the mutated state tensor")
        _assert_same_tensor(state, expected, spec.surface)
        return

    if spec.surface == "aten::_sobol_engine_draw":
        engine = SobolEngine(3, scramble=False, seed=0)
        result, quasi = torch.ops.aten._sobol_engine_draw(
            engine.quasi.clone(),
            4,
            engine.sobolstate,
            engine.dimension,
            engine.num_generated,
            torch.float32,
        )
        expected = torch.tensor(
            [
                [0.5, 0.5, 0.5],
                [0.75, 0.25, 0.25],
                [0.25, 0.75, 0.75],
                [0.375, 0.375, 0.625],
            ],
            dtype=torch.float32,
        )
        if not torch.equal(result, expected):
            raise AssertionError(f"{spec.surface} produced an unexpected non-scrambled sequence")
        if quasi.shape != engine.quasi.shape or quasi.dtype != engine.quasi.dtype:
            raise AssertionError(f"{spec.surface} returned malformed quasi state")
        return

    if spec.surface == "aten::_sobol_engine_ff_":
        direct = SobolEngine(3, scramble=False, seed=0)
        public = SobolEngine(3, scramble=False, seed=0)
        torch.ops.aten._sobol_engine_ff_(
            direct.quasi,
            3 - 1,
            direct.sobolstate,
            direct.dimension,
            direct.num_generated,
        )
        public.fast_forward(3)
        _assert_same_tensor(direct.quasi, public.quasi, spec.surface)
        return

    if spec.surface == "aten::_sobol_engine_scramble_":
        seed = 123
        generator = torch.Generator()
        generator.manual_seed(seed)
        dimension = 3
        state = torch.zeros(dimension, SobolEngine.MAXBIT, dtype=torch.long)
        torch.ops.aten._sobol_engine_initialize_state_(state, dimension)
        ltm = torch.randint(
            2,
            (dimension, SobolEngine.MAXBIT, SobolEngine.MAXBIT),
            generator=generator,
        ).tril()
        actual = state.clone()
        returned = torch.ops.aten._sobol_engine_scramble_(actual, ltm, dimension)
        if returned is not actual:
            raise AssertionError(f"{spec.surface} did not return the mutated state tensor")
        if torch.equal(actual, state):
            raise AssertionError(f"{spec.surface} did not mutate Sobol state")
        return

    raise AssertionError(f"No Sobol oracle implementation for {spec.surface}")


def _make_dynamic_rnn(kind: str):
    import torch.nn as nn
    from torch.ao.nn.quantized.dynamic import GRU, LSTM
    from torch.ao.quantization import default_dynamic_qconfig

    _select_quantized_engine()
    torch.manual_seed(1729)
    if kind == "lstm":
        module = nn.LSTM(3, 4, 1, batch_first=True)
        module.qconfig = default_dynamic_qconfig
        qmodule = LSTM.from_float(module)
    elif kind == "gru":
        module = nn.GRU(3, 4, 1, batch_first=True)
        module.qconfig = default_dynamic_qconfig
        qmodule = GRU.from_float(module)
    else:
        raise AssertionError(f"Unknown dynamic RNN kind {kind!r}")
    return qmodule


def _run_quantized_rnn(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    kind = "lstm" if "lstm" in spec.surface else "gru"
    qmodule = _make_dynamic_rnn(kind)
    params = [weight_value.param for weight_value in qmodule._all_weight_values]
    input_tensor = torch.randn(2, 5, 3)

    if kind == "lstm":
        hx = [
            torch.zeros(qmodule.num_layers, 2, qmodule.hidden_size),
            torch.zeros(qmodule.num_layers, 2, qmodule.hidden_size),
        ]
        if spec.surface.endswith(".input"):
            direct = torch.ops.aten.quantized_lstm.input(
                input_tensor,
                hx,
                params,
                qmodule.bias,
                qmodule.num_layers,
                0.0,
                False,
                qmodule.bidirectional,
                qmodule.batch_first,
            )
            public_output, public_hx = qmodule(input_tensor, tuple(hx))
            expected = (public_output, public_hx[0], public_hx[1])
        elif spec.surface.endswith(".data"):
            lengths = torch.tensor([5, 3], dtype=torch.long)
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                input_tensor,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            direct = torch.ops.aten.quantized_lstm.data(
                packed.data,
                packed.batch_sizes,
                hx,
                params,
                qmodule.bias,
                qmodule.num_layers,
                0.0,
                False,
                qmodule.bidirectional,
            )
            public_packed, public_hx = qmodule(packed, tuple(hx))
            expected = (public_packed.data, public_hx[0], public_hx[1])
        else:
            raise AssertionError(f"No quantized LSTM oracle for {spec.surface}")
    else:
        hx = torch.zeros(qmodule.num_layers, 2, qmodule.hidden_size)
        if spec.surface.endswith(".input"):
            direct = torch.ops.aten.quantized_gru.input(
                input_tensor,
                hx,
                params,
                qmodule.bias,
                qmodule.num_layers,
                0.0,
                False,
                qmodule.bidirectional,
                qmodule.batch_first,
            )
            expected = qmodule(input_tensor, hx)
        elif spec.surface.endswith(".data"):
            lengths = torch.tensor([5, 3], dtype=torch.long)
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                input_tensor,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            direct = torch.ops.aten.quantized_gru.data(
                packed.data,
                packed.batch_sizes,
                hx,
                params,
                qmodule.bias,
                qmodule.num_layers,
                0.0,
                False,
                qmodule.bidirectional,
            )
            public_packed, public_hx = qmodule(packed, hx)
            expected = (public_packed.data, public_hx)
        else:
            raise AssertionError(f"No quantized GRU oracle for {spec.surface}")

    for index, (actual, expected_value) in enumerate(zip(direct, expected)):
        if not torch.allclose(actual, expected_value, rtol=0, atol=0):
            raise AssertionError(f"{spec.surface} output {index} disagrees with public dynamic quantized wrapper")


def _run_int4(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    if spec.surface == "aten::_convert_weight_to_int4pack_for_cpu":
        weight = (torch.arange(16 * 64, dtype=torch.int32).reshape(16, 64) % 16).contiguous()
        packed = torch.ops.aten._convert_weight_to_int4pack_for_cpu(weight, 4)
        if packed.dtype != torch.uint8 or tuple(packed.shape) != (16, 32):
            raise AssertionError(f"{spec.surface} returned malformed int4 pack {packed.shape} {packed.dtype}")
        activation = torch.linspace(-1.5, 1.5, steps=3 * 64, dtype=torch.float32).reshape(3, 64)
        qparams = torch.ones((2, 16, 2), dtype=torch.float32)
        qparams[..., 1] = 0
        actual = torch.ops.aten._weight_int4pack_mm_for_cpu(activation, packed, 32, qparams)
        expected = activation @ (weight.to(torch.float32) - 8.0).T
        _assert_close_tensor(actual, expected, spec.surface, rtol=1e-5, atol=1e-5)
        return

    if spec.surface == "aten::_weight_int4pack_mm_for_cpu":
        weight = (torch.arange(16 * 64, dtype=torch.int32).reshape(16, 64) % 16).contiguous()
        packed = torch.ops.aten._convert_weight_to_int4pack_for_cpu(weight, 4)
        activation = torch.linspace(-1.5, 1.5, steps=3 * 64, dtype=torch.float32).reshape(3, 64)
        qparams = torch.ones((2, 16, 2), dtype=torch.float32)
        qparams[..., 1] = 0
        actual = torch.ops.aten._weight_int4pack_mm_for_cpu(activation, packed, 32, qparams)
        expected = activation @ (weight.to(torch.float32) - 8.0).T
        _assert_close_tensor(actual, expected, spec.surface, rtol=1e-5, atol=1e-5)
        return

    raise OracleUnavailable(f"backend_not_available: {spec.surface} is gated to a backend not active in this run")


def _dynamic_int4_cases() -> tuple[dict, ...]:
    return (
        {
            "id": "even_k_per_tensor_scale_bias",
            "in_features": 32,
            "out_features": 4,
            "block_size": 32,
            "bytes": [0x77, 0x88, 0x99, 0xFF],
            "scales": torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float32),
            "bias": torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float32),
            "input": torch.cat(
                [torch.arange(0, 32), torch.arange(224, 256)],
            ).to(torch.float32).reshape(2, 32),
        },
        {
            "id": "odd_k_low_nibble_tail",
            "in_features": 33,
            "out_features": 3,
            "block_size": 33,
            "bytes": [0x89, 0x98, 0x8F],
            "scales": torch.tensor([1.0, 0.75, 1.25], dtype=torch.float32),
            "bias": None,
            "input": torch.stack(
                [
                    torch.cat([torch.arange(0, 32, dtype=torch.float32), torch.tensor([255.0])]),
                    torch.cat([torch.tensor([255.0]), torch.arange(31, -1, -1, dtype=torch.float32)]),
                ]
            ),
        },
        {
            "id": "grouped_scales",
            "in_features": 64,
            "out_features": 2,
            "block_size": 32,
            "bytes": [0x99, 0xFF],
            "scales": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            "bias": torch.tensor([0.25, -0.75], dtype=torch.float32),
            "input": torch.stack(
                [
                    torch.cat([torch.arange(0, 63, dtype=torch.float32), torch.tensor([255.0])]),
                    torch.cat([torch.tensor([255.0]), torch.arange(62, -1, -1, dtype=torch.float32)]),
                ]
            ),
        },
    )


def _dynamic_int4_weight_bytes(case: dict) -> torch.Tensor:
    in_features = case["in_features"]
    out_features = case["out_features"]
    bytes_per_row = (in_features + 1) // 2
    weights = torch.empty((out_features, bytes_per_row), dtype=torch.uint8)
    for row_index, byte_value in enumerate(case["bytes"]):
        weights[row_index].fill_(byte_value)
    return weights


def _run_dynamic_int4(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    for case in _dynamic_int4_cases():
        in_features = case["in_features"]
        out_features = case["out_features"]
        block_size = case["block_size"]
        weights = _dynamic_int4_weight_bytes(case)
        scales = case["scales"]
        bias = case["bias"]
        input_tensor = case["input"].to(device)

        try:
            packed = torch.ops.aten._dyn_quant_pack_4bit_weight(
                weights,
                scales,
                bias,
                block_size,
                in_features,
                out_features,
            )
            if packed.device.type != "cpu":
                raise AssertionError(f"{spec.surface} returned packed weights on {packed.device}")
            if packed.dtype not in {torch.uint8, torch.float32}:
                raise AssertionError(f"{spec.surface} returned packed weights with dtype {packed.dtype}")
            if packed.numel() == 0:
                raise AssertionError(f"{spec.surface} returned empty packed weights")

            actual = torch.ops.aten._dyn_quant_matmul_4bit(
                input_tensor,
                packed,
                block_size,
                in_features,
                out_features,
            )
        except Exception as exc:
            _raise_backend_unavailable_if_applicable(spec, exc)

        expected = dynamic_int4_matmul_reference(
            case["input"],
            weights,
            scales,
            bias,
            block_size=block_size,
            in_features=in_features,
            out_features=out_features,
        ).to(actual.dtype)
        if tuple(actual.shape) != (case["input"].shape[0], out_features):
            raise AssertionError(f"{spec.surface}.{case['id']} returned wrong shape {tuple(actual.shape)}")
        if actual.dtype != input_tensor.dtype:
            raise AssertionError(f"{spec.surface}.{case['id']} returned wrong dtype {actual.dtype}")
        _assert_close_tensor(
            actual.detach().cpu(),
            expected,
            f"{spec.surface}.{case['id']}",
            rtol=2e-4,
            atol=0.25,
        )


def _validate_tinygemm_int4_dimensions(
    *,
    out_features: int,
    in_features: int,
    group_size: int,
    inner_k_tiles: int,
) -> None:
    if inner_k_tiles not in {2, 4, 8}:
        raise ValueError(f"inner_k_tiles must be 2, 4, or 8, got {inner_k_tiles}")
    if out_features % 8:
        raise ValueError(f"out_features must be divisible by 8, got {out_features}")
    if in_features % group_size:
        raise ValueError(f"in_features must be divisible by group_size, got {in_features} and {group_size}")
    if in_features % (inner_k_tiles * 16):
        raise ValueError(
            "in_features must be divisible by inner_k_tiles * 16, got "
            f"in_features={in_features} inner_k_tiles={inner_k_tiles}"
        )


def _run_mps_int4_pack(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    torch.manual_seed(2404)
    out_features = 16
    in_features = 128
    group_size = 32
    values = (
        torch.arange(out_features * in_features, dtype=torch.int64).reshape(out_features, in_features) * 3
        + torch.arange(out_features, dtype=torch.int64).reshape(out_features, 1) * 5
        + torch.arange(in_features, dtype=torch.int64).reshape(1, in_features) * 7
    ).remainder(16).to(torch.uint8)
    w_int4x8 = pack_int4_values(values, even_k_in_high_bits=True).to(device)
    input_tensor = torch.randn(3, in_features, device=device, dtype=torch.float32)

    scale_values = torch.linspace(0.5, 1.75, steps=(in_features // group_size) * out_features)
    zero_values = torch.linspace(-1.25, 1.25, steps=(in_features // group_size) * out_features)
    scales_and_zeros = torch.empty(in_features // group_size, out_features, 2, dtype=torch.float32)
    scales_and_zeros[..., 0] = scale_values.reshape(in_features // group_size, out_features)
    scales_and_zeros[..., 1] = zero_values.reshape(in_features // group_size, out_features)
    expected = tinygemm_int4_matmul_reference(input_tensor, values, scales_and_zeros, group_size)

    for inner_k_tiles in (2, 4, 8):
        _validate_tinygemm_int4_dimensions(
            out_features=out_features,
            in_features=in_features,
            group_size=group_size,
            inner_k_tiles=inner_k_tiles,
        )
        try:
            packed = torch.ops.aten._convert_weight_to_int4pack(w_int4x8, inner_k_tiles)
            if packed.device.type != torch.device(device).type:
                raise AssertionError(f"{spec.surface} returned packed weights on {packed.device}")
            if packed.dtype != torch.int32:
                raise AssertionError(f"{spec.surface} returned packed weights with dtype {packed.dtype}")
            if packed.numel() == 0:
                raise AssertionError(f"{spec.surface} returned an empty packed weight tensor")

            actual = torch.ops.aten._weight_int4pack_mm(
                input_tensor,
                packed,
                group_size,
                scales_and_zeros.to(device),
            )
        except Exception as exc:
            _raise_backend_unavailable_if_applicable(spec, exc)

        _assert_close_tensor(
            actual,
            expected,
            f"{spec.surface}.inner_k_tiles_{inner_k_tiles}",
            rtol=2e-5,
            atol=2e-5,
        )


def _run_quantized_allocation(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    actual = torch.ops.aten._empty_affine_quantized(
        [2, 3],
        dtype=torch.quint8,
        device=torch.device("cpu"),
        scale=0.25,
        zero_point=7,
    )
    if tuple(actual.shape) != (2, 3):
        raise AssertionError(f"{spec.surface} returned wrong shape {tuple(actual.shape)}")
    if actual.dtype != torch.quint8:
        raise AssertionError(f"{spec.surface} returned wrong dtype {actual.dtype}")
    if actual.qscheme() != torch.per_tensor_affine:
        raise AssertionError(f"{spec.surface} returned wrong qscheme {actual.qscheme()}")
    if actual.q_scale() != 0.25 or actual.q_zero_point() != 7:
        raise AssertionError(f"{spec.surface} returned wrong quantization parameters")


def _run_linear_backward(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    torch.manual_seed(401)
    self = torch.randn(4, 3, dtype=torch.float32)
    weight = torch.randn(5, 3, dtype=torch.float32)
    bias = torch.randn(5, dtype=torch.float32)
    grad_output = torch.randn(4, 5, dtype=torch.float32)

    mps_self = self.to(device)
    mps_weight = weight.to(device)
    mps_grad_output = grad_output.to(device)
    expected = linear_backward_reference(self, grad_output, weight, bias)

    if spec.surface == "aten::linear_backward.out":
        out0 = torch.empty_like(mps_self)
        out1 = torch.empty_like(mps_weight)
        out2 = torch.empty_like(bias, device=device)
        actual = torch.ops.aten.linear_backward.out(
            mps_self,
            mps_grad_output,
            mps_weight,
            [True, True, True],
            out0=out0,
            out1=out1,
            out2=out2,
        )
        assert_out_identity(actual[0], out0, f"{spec.surface}.out0")
        assert_out_identity(actual[1], out1, f"{spec.surface}.out1")
        assert_out_identity(actual[2], out2, f"{spec.surface}.out2")
        for index, (actual_grad, expected_grad) in enumerate(zip(actual, expected)):
            _assert_close_tensor(actual_grad, expected_grad, f"{spec.surface}[{index}]")
        return

    actual = torch.ops.aten.linear_backward(mps_self, mps_grad_output, mps_weight, [True, True, True])
    for index, (actual_grad, expected_grad) in enumerate(zip(actual, expected)):
        _assert_close_tensor(actual_grad, expected_grad, f"{spec.surface}[{index}]")

    masked = torch.ops.aten.linear_backward(mps_self, mps_grad_output, mps_weight, [True, False, False])
    if masked[0] is None:
        raise AssertionError(f"{spec.surface} failed to return requested input gradient")
    if masked[1] is not None or masked[2] is not None:
        raise AssertionError(f"{spec.surface} returned gradients disabled by output_mask")
    _assert_close_tensor(masked[0], expected[0], f"{spec.surface}.masked_grad_input")


def _run_max_pool2d_backward(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    torch.manual_seed(402)
    self = torch.randn(2, 3, 6, 5, dtype=torch.float32)
    kernel_size = [2, 3]
    stride = [2, 1]
    padding = [0, 1]
    dilation = [1, 1]
    ceil_mode = False
    output = torch.nn.functional.max_pool2d(
        self,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    grad_output = torch.randn_like(output)
    actual = torch.ops.aten.max_pool2d_backward(
        grad_output.to(device),
        self.to(device),
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    )

    expected_grad_input = max_pool2d_backward_reference(
        self,
        grad_output,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    if spec.surface == "aten::max_pool2d_backward.out":
        out = torch.empty_like(self, device=device)
        actual = torch.ops.aten.max_pool2d_backward.out(
            grad_output.to(device),
            self.to(device),
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            out=out,
        )
        assert_out_identity(actual, out, spec.surface)
        _assert_close_tensor(actual, expected_grad_input, spec.surface)
        return

    _assert_close_tensor(actual, expected_grad_input, spec.surface)


def _run_unsafe_property(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    if spec.surface == "aten::_unsafe_view":
        actual = torch.ops.aten._unsafe_view(base, [2, 6])
        expected = base.reshape(2, 6)
        _assert_same_tensor(actual, expected, spec.surface)
        if actual.untyped_storage().data_ptr() != base.untyped_storage().data_ptr():
            raise AssertionError(f"{spec.surface} did not return a storage alias")
        return

    if spec.surface == "aten::_unsafe_view.out":
        expected = base.reshape(2, 6)
        out = torch.empty_like(expected)
        actual = torch.ops.aten._unsafe_view.out(base, [2, 6], out=out)
        if actual is not out:
            raise AssertionError(f"{spec.surface} did not return the provided out tensor")
        _assert_same_tensor(out, expected, spec.surface)
        return

    if spec.surface == "aten::_unsafe_index.Tensor":
        index = torch.tensor([0, 2])
        actual = torch.ops.aten._unsafe_index.Tensor(base, [index, None])
        expected = base[[0, 2]]
        _assert_same_tensor(actual, expected, spec.surface)
        return

    if spec.surface == "aten::_unsafe_index_put":
        index = torch.tensor([0, 2])
        values = torch.ones(2, 4, dtype=base.dtype)
        actual = torch.ops.aten._unsafe_index_put(base.clone(), [index, None], values, False)
        expected = base.clone()
        expected[[0, 2]] = values
        _assert_same_tensor(actual, expected, spec.surface)
        return

    if spec.surface == "aten::unsafe_split.Tensor_out":
        out = [torch.empty(1, 4), torch.empty(1, 4), torch.empty(1, 4)]
        actual = torch.ops.aten.unsafe_split.Tensor_out(base, 1, 0, out=out)
        if actual is not None:
            raise AssertionError(f"{spec.surface} should return None for Tensor[] out overload")
        expected = list(base.split(1, dim=0))
        for index, (actual_item, expected_item) in enumerate(zip(out, expected)):
            _assert_same_tensor(actual_item, expected_item, f"{spec.surface}[{index}]")
        return

    if spec.surface == "aten::unsafe_split_with_sizes.out":
        out = [torch.empty(1, 4), torch.empty(2, 4)]
        actual = torch.ops.aten.unsafe_split_with_sizes.out(base, [1, 2], 0, out=out)
        if actual is not None:
            raise AssertionError(f"{spec.surface} should return None for Tensor[] out overload")
        expected = list(base.split([1, 2], dim=0))
        for index, (actual_item, expected_item) in enumerate(zip(out, expected)):
            _assert_same_tensor(actual_item, expected_item, f"{spec.surface}[{index}]")
        return

    raise AssertionError(f"No unsafe-property oracle implementation for {spec.surface}")


def _run_autocast_property(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    if spec.surface == "aten::_autocast_to_full_precision":
        half = torch.ones(2, dtype=torch.float16)
        bf16 = torch.ones(2, dtype=torch.bfloat16)
        float_value = torch.ones(2, dtype=torch.float32)
        if torch.ops.aten._autocast_to_full_precision(half, False, True).dtype != torch.float32:
            raise AssertionError(f"{spec.surface} did not promote CPU float16 to float32")
        if torch.ops.aten._autocast_to_full_precision(bf16, False, True).dtype != torch.float32:
            raise AssertionError(f"{spec.surface} did not promote CPU bfloat16 to float32")
        if torch.ops.aten._autocast_to_full_precision(float_value, False, True).dtype != torch.float32:
            raise AssertionError(f"{spec.surface} changed CPU float32 dtype")
        if torch.ops.aten._autocast_to_full_precision(half, False, False) is not half:
            raise AssertionError(f"{spec.surface} should return self when CPU autocast is disabled")
        return

    if spec.surface == "aten::_autocast_to_reduced_precision":
        float_value = torch.ones(2, dtype=torch.float32)
        half = torch.ones(2, dtype=torch.float16)
        reduced = torch.ops.aten._autocast_to_reduced_precision(
            float_value,
            False,
            True,
            torch.float16,
            torch.bfloat16,
        )
        if reduced.dtype != torch.bfloat16:
            raise AssertionError(f"{spec.surface} did not reduce CPU float32 to requested CPU dtype")
        if torch.ops.aten._autocast_to_reduced_precision(
            half,
            False,
            True,
            torch.float16,
            torch.bfloat16,
        ) is not half:
            raise AssertionError(f"{spec.surface} should leave already-reduced CPU tensors unchanged")
        if torch.ops.aten._autocast_to_reduced_precision(
            float_value,
            False,
            False,
            torch.float16,
            torch.bfloat16,
        ) is not float_value:
            raise AssertionError(f"{spec.surface} should return self when CPU autocast is disabled")
        return

    raise AssertionError(f"No autocast-property oracle implementation for {spec.surface}")


def _run_native_batch_norm_no_stats(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    torch.manual_seed(215)
    eps = 1e-5
    momentum = 0.1
    cpu_input = torch.randn(2, 3, 4, 4, dtype=torch.float32)
    cpu_weight = torch.randn(3, dtype=torch.float32)
    cpu_bias = torch.randn(3, dtype=torch.float32)
    dev_input = cpu_input.to(device)
    dev_weight = cpu_weight.to(device)
    dev_bias = cpu_bias.to(device)
    out = torch.empty_like(dev_input)
    save_mean = torch.empty(3, dtype=torch.float32, device=device)
    save_invstd = torch.empty(3, dtype=torch.float32, device=device)

    actual = torch.ops.aten._native_batch_norm_legit.no_stats_out(
        dev_input,
        dev_weight,
        dev_bias,
        True,
        momentum,
        eps,
        out=out,
        save_mean=save_mean,
        save_invstd=save_invstd,
    )
    assert_out_identity(actual[0], out, f"{spec.surface}.out")
    assert_out_identity(actual[1], save_mean, f"{spec.surface}.save_mean")
    assert_out_identity(actual[2], save_invstd, f"{spec.surface}.save_invstd")

    expected = torch.nn.functional.batch_norm(
        cpu_input,
        running_mean=None,
        running_var=None,
        weight=cpu_weight,
        bias=cpu_bias,
        training=True,
        momentum=momentum,
        eps=eps,
    )
    expected_mean = cpu_input.mean(dim=(0, 2, 3))
    expected_var = cpu_input.var(dim=(0, 2, 3), unbiased=False)
    expected_invstd = torch.rsqrt(expected_var + eps)
    _assert_close_tensor(actual[0], expected, f"{spec.surface}.out")
    _assert_close_tensor(actual[1], expected_mean, f"{spec.surface}.save_mean")
    _assert_close_tensor(actual[2], expected_invstd, f"{spec.surface}.save_invstd")


def _assert_copy_result(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    _assert_same_tensor(actual, expected, label)
    if actual.device != expected.device:
        raise AssertionError(f"{label} device mismatch: {actual.device} vs {expected.device}")
    if actual.data_ptr() == expected.data_ptr():
        raise AssertionError(f"{label} returned a storage alias instead of a copy")
    if not actual.is_inference():
        raise AssertionError(f"{label} should return an inference tensor in its valid direct-call path")


def _run_forward_ad_inference_copy(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    with torch.inference_mode():
        primal = torch.randn(2, 3, device=device)
        tangent = torch.full_like(primal, 3.0)

        if spec.surface == "aten::_fw_primal_copy":
            actual = torch.ops.aten._fw_primal_copy.default(primal, 0)
            _assert_copy_result(actual, primal, spec.surface)
            return

        if spec.surface == "aten::_fw_primal_copy.out":
            out = torch.empty_like(primal)
            actual = torch.ops.aten._fw_primal_copy.out(primal, 0, out=out)
            assert_out_identity(actual, out, spec.surface)
            _assert_copy_result(out, primal, spec.surface)
            return

        if spec.surface == "aten::_make_dual_copy":
            actual = torch.ops.aten._make_dual_copy.default(primal, tangent, 0)
            _assert_copy_result(actual, primal, spec.surface)
            return

        if spec.surface == "aten::_make_dual_copy.out":
            out = torch.empty_like(primal)
            actual = torch.ops.aten._make_dual_copy.out(primal, tangent, 0, out=out)
            assert_out_identity(actual, out, spec.surface)
            _assert_copy_result(out, primal, spec.surface)
            return

    raise AssertionError(f"No forward-AD inference-copy implementation for {spec.surface}")


def _run_nested_select_backward(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    nested = torch.nested.nested_tensor([
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
        torch.arange(12, dtype=torch.float32).reshape(4, 3),
    ])
    grad_output = torch.ones(2, 3, dtype=torch.float32)

    actual = torch.ops.aten._nested_select_backward.default(grad_output, nested, 0, 0)
    if not actual.is_nested:
        raise AssertionError(f"{spec.surface} did not return a nested tensor")

    expected = torch.nested.nested_tensor([
        grad_output,
        torch.zeros(4, 3, dtype=torch.float32),
    ])
    _assert_same_tensor(
        torch.nested.to_padded_tensor(actual, 0.0),
        torch.nested.to_padded_tensor(expected, 0.0),
        spec.surface,
    )


def _assert_layout_shape_dtype(tensor: torch.Tensor, layout: torch.layout, shape: tuple[int, ...], label: str) -> None:
    if tensor.layout != layout:
        raise AssertionError(f"{label} layout mismatch: {tensor.layout} vs {layout}")
    if tuple(tensor.shape) != shape:
        raise AssertionError(f"{label} shape mismatch: {tuple(tensor.shape)} vs {shape}")
    if tensor.dtype != torch.float32:
        raise AssertionError(f"{label} dtype mismatch: {tensor.dtype}")


def _run_sparse_constructor_property(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    indices = torch.tensor([[0, 1, 1], [2, 0, 2]], dtype=torch.long)
    values = torch.tensor([3.0, 4.0, 5.0])
    crow = torch.tensor([0, 2, 3], dtype=torch.int64)
    col = torch.tensor([0, 2, 1], dtype=torch.int64)
    ccol = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    row = torch.tensor([0, 1, 0], dtype=torch.int64)
    block_crow = torch.tensor([0, 1, 2], dtype=torch.int64)
    block_col = torch.tensor([0, 1], dtype=torch.int64)
    block_values = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)

    if spec.surface == "aten::_sparse_coo_tensor_unsafe":
        actual = torch.ops.aten._sparse_coo_tensor_unsafe(
            indices,
            values,
            [2, 3],
            dtype=torch.float32,
            layout=torch.sparse_coo,
            device=torch.device("cpu"),
            pin_memory=False,
            is_coalesced=True,
        )
        _assert_layout_shape_dtype(actual, torch.sparse_coo, (2, 3), spec.surface)
        _assert_same_tensor(actual._indices(), indices, spec.surface)
        _assert_same_tensor(actual._values(), values, spec.surface)
        _assert_same_tensor(actual.to_dense(), torch.sparse_coo_tensor(indices, values, (2, 3)).to_dense(), spec.surface)
        return

    if spec.surface == "aten::_sparse_csr_tensor_unsafe":
        actual = torch.ops.aten._sparse_csr_tensor_unsafe(
            crow,
            col,
            values,
            [2, 3],
            dtype=torch.float32,
            layout=torch.sparse_csr,
            device=torch.device("cpu"),
            pin_memory=False,
        )
        _assert_layout_shape_dtype(actual, torch.sparse_csr, (2, 3), spec.surface)
        _assert_same_tensor(actual.crow_indices(), crow, spec.surface)
        _assert_same_tensor(actual.col_indices(), col, spec.surface)
        _assert_same_tensor(actual.values(), values, spec.surface)
        return

    if spec.surface == "aten::_sparse_csc_tensor_unsafe":
        actual = torch.ops.aten._sparse_csc_tensor_unsafe(
            ccol,
            row,
            values,
            [2, 3],
            dtype=torch.float32,
            layout=torch.sparse_csc,
            device=torch.device("cpu"),
            pin_memory=False,
        )
        _assert_layout_shape_dtype(actual, torch.sparse_csc, (2, 3), spec.surface)
        _assert_same_tensor(actual.ccol_indices(), ccol, spec.surface)
        _assert_same_tensor(actual.row_indices(), row, spec.surface)
        _assert_same_tensor(actual.values(), values, spec.surface)
        return

    if spec.surface == "aten::_sparse_bsr_tensor_unsafe":
        actual = torch.ops.aten._sparse_bsr_tensor_unsafe(
            block_crow,
            block_col,
            block_values,
            [4, 4],
            dtype=torch.float32,
            layout=torch.sparse_bsr,
            device=torch.device("cpu"),
            pin_memory=False,
        )
        _assert_layout_shape_dtype(actual, torch.sparse_bsr, (4, 4), spec.surface)
        _assert_same_tensor(actual.crow_indices(), block_crow, spec.surface)
        _assert_same_tensor(actual.col_indices(), block_col, spec.surface)
        _assert_same_tensor(actual.values(), block_values, spec.surface)
        return

    if spec.surface == "aten::_sparse_bsc_tensor_unsafe":
        actual = torch.ops.aten._sparse_bsc_tensor_unsafe(
            block_crow,
            block_col,
            block_values,
            [4, 4],
            dtype=torch.float32,
            layout=torch.sparse_bsc,
            device=torch.device("cpu"),
            pin_memory=False,
        )
        _assert_layout_shape_dtype(actual, torch.sparse_bsc, (4, 4), spec.surface)
        _assert_same_tensor(actual.ccol_indices(), block_crow, spec.surface)
        _assert_same_tensor(actual.row_indices(), block_col, spec.surface)
        _assert_same_tensor(actual.values(), block_values, spec.surface)
        return

    if spec.surface == "aten::_sparse_compressed_tensor_unsafe":
        actual = torch.ops.aten._sparse_compressed_tensor_unsafe(
            crow,
            col,
            values,
            [2, 3],
            dtype=torch.float32,
            layout=torch.sparse_csr,
            device=torch.device("cpu"),
            pin_memory=False,
        )
        _assert_layout_shape_dtype(actual, torch.sparse_csr, (2, 3), spec.surface)
        _assert_same_tensor(actual.crow_indices(), crow, spec.surface)
        _assert_same_tensor(actual.col_indices(), col, spec.surface)
        _assert_same_tensor(actual.values(), values, spec.surface)
        return

    raise AssertionError(f"No sparse-constructor property implementation for {spec.surface}")


def _run_cpu_flash_attention(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    import torch.nn.functional as F

    torch.manual_seed(2026)
    query = torch.randn(1, 2, 4, 8)
    key = torch.randn(1, 2, 4, 8)
    value = torch.randn(1, 2, 4, 8)

    if spec.surface == "aten::_scaled_dot_product_flash_attention_for_cpu":
        actual, logsumexp = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
            query,
            key,
            value,
            0.0,
            False,
        )
        expected = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        if not torch.allclose(actual, expected, rtol=0.0, atol=0.0):
            raise AssertionError(f"{spec.surface} output does not match public CPU SDPA")
        if tuple(logsumexp.shape) != (1, 2, 4) or logsumexp.dtype != torch.float32:
            raise AssertionError(f"{spec.surface} returned malformed logsumexp")
        if not torch.isfinite(logsumexp).all():
            raise AssertionError(f"{spec.surface} returned non-finite logsumexp for finite inputs")
        return

    if spec.surface == "aten::_scaled_dot_product_flash_attention_for_cpu_backward":
        direct_query = query.detach().clone().requires_grad_(True)
        direct_key = key.detach().clone().requires_grad_(True)
        direct_value = value.detach().clone().requires_grad_(True)
        out, logsumexp = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
            direct_query,
            direct_key,
            direct_value,
            0.0,
            False,
        )
        grad_out = torch.randn_like(out)
        actual = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu_backward(
            grad_out,
            direct_query,
            direct_key,
            direct_value,
            out,
            logsumexp,
            0.0,
            False,
        )

        ref_query = query.detach().clone().requires_grad_(True)
        ref_key = key.detach().clone().requires_grad_(True)
        ref_value = value.detach().clone().requires_grad_(True)
        expected_out = F.scaled_dot_product_attention(
            ref_query,
            ref_key,
            ref_value,
            dropout_p=0.0,
            is_causal=False,
        )
        expected_out.backward(grad_out)
        expected = (ref_query.grad, ref_key.grad, ref_value.grad)
        for index, (actual_grad, expected_grad) in enumerate(zip(actual, expected)):
            if not torch.allclose(actual_grad, expected_grad, rtol=0.0, atol=0.0):
                raise AssertionError(f"{spec.surface} gradient {index} does not match public CPU SDPA")
        return

    raise AssertionError(f"No CPU flash-attention implementation for {spec.surface}")


def _privateuse1_attention_sample(device: str):
    import torch.nn.functional as F

    torch.manual_seed(2026)
    query = torch.randn(1, 2, 8, 64)
    key = torch.randn(1, 2, 8, 64)
    value = torch.randn(1, 2, 8, 64)
    grad_out = torch.randn_like(query)
    expected = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

    ref_query = query.detach().clone().requires_grad_(True)
    ref_key = key.detach().clone().requires_grad_(True)
    ref_value = value.detach().clone().requires_grad_(True)
    ref_out = F.scaled_dot_product_attention(ref_query, ref_key, ref_value, dropout_p=0.0, is_causal=False)
    ref_out.backward(grad_out)

    return {
        "query": query,
        "key": key,
        "value": value,
        "grad_out": grad_out,
        "expected": expected,
        "expected_grads": (ref_query.grad, ref_key.grad, ref_value.grad),
        "device_query": query.to(device),
        "device_key": key.to(device),
        "device_value": value.to(device),
        "device_grad_out": grad_out.to(device),
    }


def _quantized_flash_attention_samples(device: str):
    import torch.nn.functional as F

    samples = []
    for dtype in (torch.float16, torch.bfloat16):
        torch.manual_seed(2026)
        sdpa_query = torch.randn(1, 2, 8, 64, dtype=dtype)
        sdpa_key = torch.randn(1, 2, 8, 64, dtype=dtype)
        sdpa_value = torch.randn(1, 2, 8, 64, dtype=dtype)
        flash_query = sdpa_query.transpose(1, 2).contiguous()
        flash_key = sdpa_key.transpose(1, 2).contiguous()
        flash_value = sdpa_value.transpose(1, 2).contiguous()

        sdpa_expected = F.scaled_dot_product_attention(
            sdpa_query,
            sdpa_key,
            sdpa_value,
            dropout_p=0.0,
            is_causal=False,
        )
        flash_expected = sdpa_expected.transpose(1, 2).contiguous()

        samples.append({
            "dtype": dtype,
            "sdpa_query": sdpa_query.to(device),
            "sdpa_key": sdpa_key.to(device),
            "sdpa_value": sdpa_value.to(device),
            "sdpa_expected": sdpa_expected,
            "flash_query": flash_query.to(device),
            "flash_key": flash_key.to(device),
            "flash_value": flash_value.to(device),
            "flash_expected": flash_expected,
            "descale": torch.ones((1,), device=device, dtype=torch.float32),
        })
    return samples


def _run_quantized_flash_attention(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)

    def _check_quantized_forward(result, expected, label: str, expected_len: int) -> None:
        if len(result) != expected_len:
            raise AssertionError(f"{label} returned {len(result)} values, expected {expected_len}")
        _assert_close_tensor(result[0], expected, label, rtol=2e-2, atol=2e-2)
        logsumexp = result[1]
        if tuple(logsumexp.shape) != (1, 2, 8):
            raise AssertionError(f"{label} returned malformed logsumexp shape {tuple(logsumexp.shape)}")
        if logsumexp.dtype != torch.float32:
            raise AssertionError(f"{label} returned malformed logsumexp dtype {logsumexp.dtype}")
        if not torch.isfinite(logsumexp.detach().cpu()).all().item():
            raise AssertionError(f"{label} returned non-finite logsumexp for finite inputs")

    try:
        for sample in _quantized_flash_attention_samples(device):
            dtype_label = str(sample["dtype"])

            if spec.surface == "aten::_flash_attention_forward.quantized":
                for descale_label, q_descale, k_descale, v_descale in (
                    ("none", None, None, None),
                    ("ones", sample["descale"], sample["descale"], sample["descale"]),
                ):
                    result = torch.ops.aten._flash_attention_forward.quantized(
                        sample["flash_query"],
                        sample["flash_key"],
                        sample["flash_value"],
                        None,
                        None,
                        8,
                        8,
                        0.0,
                        False,
                        False,
                        q_descale,
                        k_descale,
                        v_descale,
                    )
                    label = f"{spec.surface}.{dtype_label}.{descale_label}"
                    _check_quantized_forward(result, sample["flash_expected"], label, 5)
                continue

            if spec.surface == "aten::_scaled_dot_product_flash_attention.quantized":
                for descale_label, q_descale, k_descale, v_descale in (
                    ("none", None, None, None),
                    ("ones", sample["descale"], sample["descale"], sample["descale"]),
                ):
                    result = torch.ops.aten._scaled_dot_product_flash_attention.quantized(
                        sample["sdpa_query"],
                        sample["sdpa_key"],
                        sample["sdpa_value"],
                        q_descale,
                        k_descale,
                        v_descale,
                        0.0,
                        False,
                        False,
                    )
                    label = f"{spec.surface}.{dtype_label}.{descale_label}"
                    _check_quantized_forward(result, sample["sdpa_expected"], label, 9)
                continue

            raise AssertionError(f"No quantized flash-attention implementation for {spec.surface}")
    except Exception as exc:
        message = str(exc)
        if isinstance(exc, (NotImplementedError, RuntimeError)) and (
            "Could not run" in message or "only available for these backends" in message
        ):
            raise OracleUnavailable(f"backend_not_available: {spec.surface}: {message.splitlines()[0]}") from exc
        raise


def _run_privateuse1_attention(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    sample = _privateuse1_attention_sample(device)
    q = sample["device_query"]
    k = sample["device_key"]
    v = sample["device_value"]
    grad_out = sample["device_grad_out"]
    expected = sample["expected"]
    expected_grads = sample["expected_grads"]

    def _check_forward(result, label: str) -> None:
        _assert_close_tensor(result[0], expected, label, rtol=1e-4, atol=1e-4)
        logsumexp = result[1]
        if tuple(logsumexp.shape) not in {(1, 2, 8), (2, 8)}:
            raise AssertionError(f"{label} returned malformed logsumexp shape {tuple(logsumexp.shape)}")
        if logsumexp.dtype != torch.float32:
            raise AssertionError(f"{label} returned malformed logsumexp dtype {logsumexp.dtype}")
        if not torch.isfinite(logsumexp.detach().cpu()).all().item():
            raise AssertionError(f"{label} returned non-finite logsumexp for finite inputs")

    def _check_backward(result, label: str) -> None:
        for index, (actual_grad, expected_grad) in enumerate(zip(result[:3], expected_grads)):
            _assert_close_tensor(actual_grad, expected_grad, f"{label}.grad{index}", rtol=1e-4, atol=1e-4)

    if spec.surface == "aten::_scaled_dot_product_fused_attention_overrideable":
        result = torch.ops.aten._scaled_dot_product_fused_attention_overrideable(q, k, v, None, 0.0, False, False)
        _check_forward(result, spec.surface)
        return

    if spec.surface == "aten::_scaled_dot_product_flash_attention":
        result = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, False, False)
        _check_forward(result, spec.surface)
        return

    if spec.surface == "aten::_scaled_dot_product_efficient_attention":
        result = torch.ops.aten._scaled_dot_product_efficient_attention(q, k, v, None, True, 0.0, False)
        _check_forward(result, spec.surface)
        return

    if spec.surface == "aten::_flash_attention_forward":
        result = torch.ops.aten._flash_attention_forward(q, k, v, None, None, 8, 8, 0.0, False, False)
        _check_forward(result, spec.surface)
        return

    if spec.surface == "aten::_efficient_attention_forward":
        result = torch.ops.aten._efficient_attention_forward(q, k, v, None, None, None, 8, 8, 0.0, 0, True)
        _check_forward(result, spec.surface)
        return

    if spec.surface == "aten::_scaled_dot_product_fused_attention_overrideable_backward":
        forward = torch.ops.aten._scaled_dot_product_fused_attention_overrideable(q, k, v, None, 0.0, False, False)
        result = torch.ops.aten._scaled_dot_product_fused_attention_overrideable_backward(
            grad_out,
            q,
            k,
            v,
            torch.empty(0, device=device),
            [True, True, True, False],
            forward[0],
            forward[1],
            forward[2],
            forward[3],
            forward[4],
            forward[5],
            0.0,
            False,
            forward[6],
            forward[7],
        )
        _check_backward(result, spec.surface)
        return

    if spec.surface == "aten::_scaled_dot_product_flash_attention_backward":
        forward = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, False, False)
        result = torch.ops.aten._scaled_dot_product_flash_attention_backward(
            grad_out,
            q,
            k,
            v,
            forward[0],
            forward[1],
            forward[2],
            forward[3],
            forward[4],
            forward[5],
            0.0,
            False,
            forward[6],
            forward[7],
        )
        _check_backward(result, spec.surface)
        return

    if spec.surface == "aten::_scaled_dot_product_efficient_attention_backward":
        forward = torch.ops.aten._scaled_dot_product_efficient_attention(q, k, v, None, True, 0.0, False)
        result = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
            grad_out,
            q,
            k,
            v,
            torch.empty(0, device=device),
            forward[0],
            forward[1],
            forward[2],
            forward[3],
            0.0,
            [True, True, True, False],
            False,
        )
        _check_backward(result, spec.surface)
        return

    if spec.surface == "aten::_flash_attention_backward":
        forward = torch.ops.aten._flash_attention_forward(q, k, v, None, None, 8, 8, 0.0, False, False)
        result = torch.ops.aten._flash_attention_backward(
            grad_out,
            q,
            k,
            v,
            forward[0],
            forward[1],
            torch.empty(0, device=device, dtype=torch.int64),
            torch.empty(0, device=device, dtype=torch.int64),
            8,
            8,
            0.0,
            False,
            forward[2],
            forward[3],
        )
        _check_backward(result, spec.surface)
        return

    if spec.surface == "aten::_efficient_attention_backward":
        forward = torch.ops.aten._efficient_attention_forward(q, k, v, None, None, None, 8, 8, 0.0, 0, True)
        result = torch.ops.aten._efficient_attention_backward(
            grad_out,
            q,
            k,
            v,
            None,
            forward[0],
            None,
            None,
            8,
            8,
            forward[1],
            0.0,
            forward[2],
            forward[3],
            0,
            False,
        )
        _check_backward(result, spec.surface)
        return

    raise AssertionError(f"No PrivateUse1 attention implementation for {spec.surface}")


def _run_privateuse1_matmul_backward(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    torch.manual_seed(4011)
    grad = torch.randn(2, 4)
    left = torch.randn(2, 3)
    right = torch.randn(3, 4)
    expected = (grad @ right.t(), left.t() @ grad)

    device_grad = grad.to(device)
    device_left = left.to(device)
    device_right = right.to(device)
    masks = ([True, True],) if spec.surface.endswith(".out") else (
        [True, True],
        [True, False],
        [False, True],
        [False, False],
    )
    for mask in masks:
        if spec.surface == "aten::matmul_backward.out":
            out0 = torch.empty_like(device_left)
            out1 = torch.empty_like(device_right)
            actual = torch.ops.aten.matmul_backward.out(
                device_grad,
                device_left,
                device_right,
                mask,
                out0=out0,
                out1=out1,
            )
            assert_out_identity(actual[0], out0, f"{spec.surface}.out0.mask{mask}")
            assert_out_identity(actual[1], out1, f"{spec.surface}.out1.mask{mask}")
        else:
            actual = torch.ops.aten.matmul_backward(
                device_grad,
                device_left,
                device_right,
                mask,
            )
        for index, enabled in enumerate(mask):
            if enabled:
                _assert_close_tensor(actual[index], expected[index], f"{spec.surface}.mask{mask}.{index}")


def _run_privateuse1_resize_output(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    source = torch.empty(2, 3, device=device)
    target_size = [4, 5]

    if spec.surface == "aten::_resize_output":
        actual = torch.ops.aten._resize_output(source, target_size, torch.device(device))
        if tuple(actual.shape) != tuple(target_size):
            raise AssertionError(f"{spec.surface} returned shape {tuple(actual.shape)}, expected {tuple(target_size)}")
        if actual.device.type != torch.device(device).type:
            raise AssertionError(f"{spec.surface} returned tensor on {actual.device}, expected {device}")
        return

    if spec.surface == "aten::_resize_output.out":
        out = torch.empty(0, device=device)
        actual = torch.ops.aten._resize_output.out(source, target_size, torch.device(device), out=out)
        assert_out_identity(actual, out, spec.surface)
        if tuple(out.shape) != tuple(target_size):
            raise AssertionError(f"{spec.surface} resized to shape {tuple(out.shape)}, expected {tuple(target_size)}")
        if out.device.type != torch.device(device).type:
            raise AssertionError(f"{spec.surface} returned tensor on {out.device}, expected {device}")
        return

    if spec.surface == "aten::_resize_output_":
        actual = torch.ops.aten._resize_output_(source, target_size, torch.device(device))
        if actual is not source:
            raise AssertionError(f"{spec.surface} did not return the resized input tensor")
        if tuple(source.shape) != tuple(target_size):
            raise AssertionError(f"{spec.surface} resized to shape {tuple(source.shape)}, expected {tuple(target_size)}")
        return

    raise AssertionError(f"No PrivateUse1 resize-output implementation for {spec.surface}")


def _run_privateuse1_batch_norm_forward(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    torch.manual_seed(4012)
    input_tensor = torch.randn(4, 3, 5, 5)
    weight = torch.randn(3)
    bias = torch.randn(3)
    eps = 1e-5
    mean = input_tensor.mean(dim=(0, 2, 3))
    variance = input_tensor.var(dim=(0, 2, 3), unbiased=False)
    invstd = torch.rsqrt(variance + eps)
    expected = (
        (input_tensor - mean[None, :, None, None])
        * invstd[None, :, None, None]
        * weight[None, :, None, None]
        + bias[None, :, None, None]
    )

    device_input = input_tensor.to(device)
    device_mean, device_invstd = torch.ops.aten.batch_norm_stats(device_input, eps)

    if spec.surface in {"aten::batch_norm_stats", "aten::batch_norm_stats.out"}:
        if spec.surface.endswith(".out"):
            out0 = torch.empty_like(device_mean)
            out1 = torch.empty_like(device_invstd)
            actual_mean, actual_invstd = torch.ops.aten.batch_norm_stats.out(
                device_input,
                eps,
                out0=out0,
                out1=out1,
            )
            assert_out_identity(actual_mean, out0, f"{spec.surface}.out0")
            assert_out_identity(actual_invstd, out1, f"{spec.surface}.out1")
        else:
            actual_mean = device_mean
            actual_invstd = device_invstd
        _assert_close_tensor(actual_mean, mean, f"{spec.surface}.mean")
        _assert_close_tensor(actual_invstd, invstd, f"{spec.surface}.invstd")
        return

    if spec.surface in {"aten::batch_norm_elemt", "aten::batch_norm_elemt.out"}:
        if spec.surface.endswith(".out"):
            out = torch.empty_like(device_input)
            actual = torch.ops.aten.batch_norm_elemt.out(
                device_input,
                weight.to(device),
                bias.to(device),
                device_mean,
                device_invstd,
                eps,
                out=out,
            )
            assert_out_identity(actual, out, spec.surface)
        else:
            actual = torch.ops.aten.batch_norm_elemt(
                device_input,
                weight.to(device),
                bias.to(device),
                device_mean,
                device_invstd,
                eps,
            )
        _assert_close_tensor(actual, expected, spec.surface, rtol=1e-4, atol=1e-4)
        return

    raise AssertionError(f"No PrivateUse1 batch-norm forward implementation for {spec.surface}")


def _run_privateuse1_thnn_cell(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    torch.manual_seed(4013)

    if spec.surface in {"aten::_thnn_fused_gru_cell", "aten::_thnn_fused_gru_cell.out"}:
        input_gates = torch.randn(2, 12)
        hidden_gates = torch.randn(2, 12)
        hx = torch.randn(2, 4)
        input_bias = torch.randn(12)
        hidden_bias = torch.randn(12)
        i_r, i_z, i_n = (input_gates + input_bias).chunk(3, 1)
        h_r, h_z, h_n = (hidden_gates + hidden_bias).chunk(3, 1)
        reset = torch.sigmoid(i_r + h_r)
        update = torch.sigmoid(i_z + h_z)
        new = torch.tanh(i_n + reset * h_n)
        expected_hy = new + update * (hx - new)
        device_input_gates = input_gates.to(device)
        device_hidden_gates = hidden_gates.to(device)
        device_hx = hx.to(device)
        device_input_bias = input_bias.to(device)
        device_hidden_bias = hidden_bias.to(device)
        if spec.surface.endswith(".out"):
            out0 = torch.empty_like(device_hx)
            out1 = torch.empty(2, 24, device=device)
            actual_hy, workspace = torch.ops.aten._thnn_fused_gru_cell.out(
                device_input_gates,
                device_hidden_gates,
                device_hx,
                device_input_bias,
                device_hidden_bias,
                out0=out0,
                out1=out1,
            )
            assert_out_identity(actual_hy, out0, f"{spec.surface}.out0")
            assert_out_identity(workspace, out1, f"{spec.surface}.out1")
        else:
            actual_hy, workspace = torch.ops.aten._thnn_fused_gru_cell(
                device_input_gates,
                device_hidden_gates,
                device_hx,
                device_input_bias,
                device_hidden_bias,
            )
        _assert_close_tensor(actual_hy, expected_hy, f"{spec.surface}.hy", rtol=1e-4, atol=1e-4)
        if tuple(workspace.shape) != (2, 24):
            raise AssertionError(f"{spec.surface} returned malformed workspace shape {tuple(workspace.shape)}")
        return

    if spec.surface in {"aten::_thnn_fused_lstm_cell", "aten::_thnn_fused_lstm_cell.out"}:
        input_gates = torch.randn(2, 16)
        hidden_gates = torch.randn(2, 16)
        cx = torch.randn(2, 4)
        input_bias = torch.randn(16)
        hidden_bias = torch.randn(16)
        in_gate, forget_gate, cell_gate, out_gate = (input_gates + hidden_gates + input_bias + hidden_bias).chunk(4, 1)
        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)
        expected_cy = forget_gate * cx + in_gate * cell_gate
        expected_hy = out_gate * torch.tanh(expected_cy)
        device_input_gates = input_gates.to(device)
        device_hidden_gates = hidden_gates.to(device)
        device_cx = cx.to(device)
        device_input_bias = input_bias.to(device)
        device_hidden_bias = hidden_bias.to(device)
        if spec.surface.endswith(".out"):
            out0 = torch.empty_like(device_cx)
            out1 = torch.empty_like(device_cx)
            out2 = torch.empty(2, 16, device=device)
            actual_hy, actual_cy, workspace = torch.ops.aten._thnn_fused_lstm_cell.out(
                device_input_gates,
                device_hidden_gates,
                device_cx,
                device_input_bias,
                device_hidden_bias,
                out0=out0,
                out1=out1,
                out2=out2,
            )
            assert_out_identity(actual_hy, out0, f"{spec.surface}.out0")
            assert_out_identity(actual_cy, out1, f"{spec.surface}.out1")
            assert_out_identity(workspace, out2, f"{spec.surface}.out2")
        else:
            actual_hy, actual_cy, workspace = torch.ops.aten._thnn_fused_lstm_cell(
                device_input_gates,
                device_hidden_gates,
                device_cx,
                device_input_bias,
                device_hidden_bias,
            )
        _assert_close_tensor(actual_hy, expected_hy, f"{spec.surface}.hy", rtol=1e-4, atol=1e-4)
        _assert_close_tensor(actual_cy, expected_cy, f"{spec.surface}.cy", rtol=1e-4, atol=1e-4)
        if tuple(workspace.shape) != (2, 16):
            raise AssertionError(f"{spec.surface} returned malformed workspace shape {tuple(workspace.shape)}")
        return

    raise AssertionError(f"No PrivateUse1 THNN cell implementation for {spec.surface}")


def _run_privateuse1_pin_memory(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    torch.manual_seed(4014)
    source = torch.randn(4, device=device)

    if spec.surface == "aten::_pin_memory":
        actual = torch.ops.aten._pin_memory(source, None)
    elif spec.surface == "aten::_pin_memory.out":
        out = torch.empty_like(source)
        actual = torch.ops.aten._pin_memory.out(source, None, out=out)
        assert_out_identity(actual, out, spec.surface)
    elif spec.surface == "aten::pin_memory":
        actual = torch.ops.aten.pin_memory(source, None)
    else:
        raise AssertionError(f"No PrivateUse1 pin-memory implementation for {spec.surface}")

    if tuple(actual.shape) != tuple(source.shape):
        raise AssertionError(f"{spec.surface} returned malformed shape {tuple(actual.shape)}")
    if actual.dtype != source.dtype:
        raise AssertionError(f"{spec.surface} returned malformed dtype {actual.dtype}")
    if actual.device.type != source.device.type:
        raise AssertionError(f"{spec.surface} moved tensor to {actual.device}; expected device-preserving no-op semantics")
    _assert_close_tensor(actual, source, spec.surface)


def _run_mps_convolution(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    torch.manual_seed(31415)

    if spec.surface in {
        "aten::_mps_convolution",
        "aten::_mps_convolution.out",
        "aten::mps_convolution_backward",
    }:
        cpu_input = torch.randn(1, 2, 5, 5)
        cpu_weight = torch.randn(3, 2, 3, 3)
        cpu_bias = torch.randn(3)
        padding = [1, 1]
        stride = [1, 1]
        dilation = [1, 1]
        groups = 1
        expected = torch.nn.functional.conv2d(cpu_input, cpu_weight, cpu_bias, padding=1)
        mps_input = cpu_input.to("mps")
        mps_weight = cpu_weight.to("mps")
        mps_bias = cpu_bias.to("mps")

        if spec.surface == "aten::_mps_convolution":
            actual = torch.ops.aten._mps_convolution(
                mps_input,
                mps_weight,
                mps_bias,
                padding,
                stride,
                dilation,
                groups,
            )
            _assert_close_tensor(actual, expected, spec.surface)
            return

        if spec.surface == "aten::_mps_convolution.out":
            out = torch.empty_like(expected, device="mps")
            actual = torch.ops.aten._mps_convolution.out(
                mps_input,
                mps_weight,
                mps_bias,
                padding,
                stride,
                dilation,
                groups,
                out=out,
            )
            if actual is not out:
                raise AssertionError(f"{spec.surface} did not return the provided out tensor")
            _assert_close_tensor(out, expected, spec.surface)
            return

        grad_output = torch.randn_like(expected)
        actual = torch.ops.aten.mps_convolution_backward(
            mps_input,
            grad_output.to("mps"),
            mps_weight,
            padding,
            stride,
            dilation,
            groups,
            [True, True, False],
        )
        ref_input = cpu_input.detach().clone().requires_grad_(True)
        ref_weight = cpu_weight.detach().clone().requires_grad_(True)
        ref_bias = cpu_bias.detach().clone().requires_grad_(True)
        torch.nn.functional.conv2d(ref_input, ref_weight, ref_bias, padding=1).backward(grad_output)
        _assert_close_tensor(actual[0], ref_input.grad, f"{spec.surface}.grad_input")
        _assert_close_tensor(actual[1], ref_weight.grad, f"{spec.surface}.grad_weight")
        if actual[2] is not None:
            raise AssertionError(f"{spec.surface} unexpectedly returned a bias gradient")
        return

    if spec.surface in {
        "aten::_mps_convolution_transpose",
        "aten::_mps_convolution_transpose.out",
        "aten::mps_convolution_transpose_backward",
        "aten::mps_convolution_transpose_backward.out",
    }:
        cpu_input = torch.randn(1, 2, 5, 5)
        cpu_weight = torch.randn(2, 3, 3, 3)
        padding = [1, 1]
        output_padding = [0, 0]
        stride = [1, 1]
        dilation = [1, 1]
        groups = 1
        expected = torch.nn.functional.conv_transpose2d(cpu_input, cpu_weight, padding=1)
        mps_input = cpu_input.to("mps")
        mps_weight = cpu_weight.to("mps")

        if spec.surface == "aten::_mps_convolution_transpose":
            actual = torch.ops.aten._mps_convolution_transpose(
                mps_input,
                mps_weight,
                padding,
                output_padding,
                stride,
                dilation,
                groups,
            )
            _assert_close_tensor(actual, expected, spec.surface)
            return

        if spec.surface == "aten::_mps_convolution_transpose.out":
            out = torch.empty_like(expected, device="mps")
            actual = torch.ops.aten._mps_convolution_transpose.out(
                mps_input,
                mps_weight,
                padding,
                output_padding,
                stride,
                dilation,
                groups,
                out=out,
            )
            if actual is not out:
                raise AssertionError(f"{spec.surface} did not return the provided out tensor")
            _assert_close_tensor(out, expected, spec.surface)
            return

        grad_output = torch.randn_like(expected)
        if spec.surface == "aten::mps_convolution_transpose_backward":
            actual = torch.ops.aten.mps_convolution_transpose_backward(
                mps_input,
                grad_output.to("mps"),
                mps_weight,
                padding,
                output_padding,
                stride,
                dilation,
                groups,
                [True, True],
            )
        else:
            out0 = torch.empty_like(cpu_input, device="mps")
            out1 = torch.empty_like(cpu_weight, device="mps")
            actual = torch.ops.aten.mps_convolution_transpose_backward.out(
                mps_input,
                grad_output.to("mps"),
                mps_weight,
                padding,
                output_padding,
                stride,
                dilation,
                groups,
                [True, True],
                out0=out0,
                out1=out1,
            )
            if actual[0] is not out0 or actual[1] is not out1:
                raise AssertionError(f"{spec.surface} did not return the provided out tensors")
        ref_input = cpu_input.detach().clone().requires_grad_(True)
        ref_weight = cpu_weight.detach().clone().requires_grad_(True)
        torch.nn.functional.conv_transpose2d(ref_input, ref_weight, padding=1).backward(grad_output)
        _assert_close_tensor(actual[0], ref_input.grad, f"{spec.surface}.grad_input")
        _assert_close_tensor(actual[1], ref_weight.grad, f"{spec.surface}.grad_weight")
        return

    raise AssertionError(f"No MPS convolution implementation for {spec.surface}")


def _run_mps_sdpa_math(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    import torch.nn.functional as F

    torch.manual_seed(2718)
    query = torch.randn(1, 2, 4, 8)
    key = torch.randn(1, 2, 4, 8)
    value = torch.randn(1, 2, 4, 8)
    actual, attention = torch.ops.aten._scaled_dot_product_attention_math_for_mps(
        query.to("mps"),
        key.to("mps"),
        value.to("mps"),
        None,
        0.0,
        False,
        None,
    )
    expected = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    _assert_close_tensor(actual, expected, spec.surface)
    if tuple(attention.shape) != (1, 2, 4, 4):
        raise AssertionError(f"{spec.surface} returned malformed attention tensor shape {tuple(attention.shape)}")
    if attention.dtype != torch.float32:
        raise AssertionError(f"{spec.surface} returned malformed attention dtype {attention.dtype}")
    if not torch.isfinite(attention).all().item():
        raise AssertionError(f"{spec.surface} returned non-finite attention values for finite inputs")


def _mps_lstm_sample():
    import torch.nn as nn

    torch.manual_seed(1618)
    module = nn.LSTM(3, 4, 1, batch_first=True)
    input_tensor = torch.randn(2, 5, 3)
    h0 = torch.randn(1, 2, 4)
    c0 = torch.randn(1, 2, 4)
    params = [
        module.weight_ih_l0.detach(),
        module.weight_hh_l0.detach(),
        module.bias_ih_l0.detach(),
        module.bias_hh_l0.detach(),
    ]
    return input_tensor, h0, c0, params


def _run_mps_lstm(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    input_tensor, h0, c0, params = _mps_lstm_sample()
    mps_input = input_tensor.to("mps")
    mps_hx = [h0.to("mps"), c0.to("mps")]
    mps_params = [param.to("mps") for param in params]

    forward = torch.ops.aten._lstm_mps(
        mps_input,
        mps_hx,
        mps_params,
        True,
        1,
        0.0,
        False,
        False,
        True,
    )
    expected = torch.ops.aten.lstm.input(
        input_tensor,
        [h0, c0],
        params,
        True,
        1,
        0.0,
        False,
        False,
        True,
    )

    def _assert_forward_outputs(outputs, label: str) -> None:
        for index in range(3):
            _assert_close_tensor(outputs[index], expected[index], f"{label}[{index}]")

    def _assert_reserve_metadata(outputs, label: str) -> None:
        for index in range(3, 6):
            item = outputs[index]
            reference = forward[index]
            if item.device.type != "mps":
                raise AssertionError(f"{label} returned reserve tensor {index} on {item.device}")
            if item.dtype != reference.dtype:
                raise AssertionError(f"{label} reserve tensor {index} dtype mismatch: {item.dtype} vs {reference.dtype}")
            if tuple(item.shape) != tuple(reference.shape):
                raise AssertionError(
                    f"{label} reserve tensor {index} shape mismatch: {tuple(item.shape)} vs {tuple(reference.shape)}"
                )

    def _assert_lstm_backward_tuple(backward_tuple, label: str, ref_input, ref_h0, ref_c0, ref_params) -> None:
        _assert_close_tensor(backward_tuple[0], ref_input.grad, f"{label}.grad_input")
        _assert_close_tensor(backward_tuple[1][0], ref_h0.grad, f"{label}.grad_h")
        _assert_close_tensor(backward_tuple[1][1], ref_c0.grad, f"{label}.grad_c")
        for index, (actual_grad, ref_param) in enumerate(zip(backward_tuple[2], ref_params)):
            _assert_close_tensor(actual_grad, ref_param.grad, f"{label}.grad_param_{index}")

    def _assert_backward_matches(outputs, label: str, *, use_out: bool = False) -> None:
        grad_y = torch.randn_like(outputs[0])
        grad_hy = torch.randn_like(outputs[1])
        grad_cy = torch.randn_like(outputs[2])
        actual = torch.ops.aten.lstm_mps_backward(
            grad_y,
            grad_hy,
            grad_cy,
            outputs[3],
            outputs[4],
            mps_input,
            outputs[5],
            mps_hx,
            mps_params,
            True,
            1,
            0.0,
            False,
            False,
            True,
        )

        ref_input = input_tensor.detach().clone().requires_grad_(True)
        ref_h0 = h0.detach().clone().requires_grad_(True)
        ref_c0 = c0.detach().clone().requires_grad_(True)
        ref_params = [param.detach().clone().requires_grad_(True) for param in params]
        ref_out = torch.ops.aten.lstm.input(
            ref_input,
            [ref_h0, ref_c0],
            ref_params,
            True,
            1,
            0.0,
            False,
            False,
            True,
        )
        torch.autograd.backward(
            ref_out,
            (grad_y.cpu(), grad_hy.cpu(), grad_cy.cpu()),
        )
        _assert_lstm_backward_tuple(actual, label, ref_input, ref_h0, ref_c0, ref_params)

        if not use_out:
            return

        out0 = torch.empty_like(actual[0])
        out1 = [torch.empty_like(item) for item in actual[1]]
        out2 = [torch.empty_like(item) for item in actual[2]]
        returned = torch.ops.aten.lstm_mps_backward.out(
            grad_y,
            grad_hy,
            grad_cy,
            outputs[3],
            outputs[4],
            mps_input,
            outputs[5],
            mps_hx,
            mps_params,
            True,
            1,
            0.0,
            False,
            False,
            True,
            out0=out0,
            out1=out1,
            out2=out2,
        )
        if returned is not None:
            raise AssertionError(f"{label} should return None")
        _assert_lstm_backward_tuple((out0, out1, out2), label, ref_input, ref_h0, ref_c0, ref_params)

    if spec.surface == "aten::_lstm_mps":
        _assert_forward_outputs(forward, spec.surface)
        _assert_reserve_metadata(forward, spec.surface)
        return

    if spec.surface == "aten::_lstm_mps.out":
        outs = [torch.empty_like(item) for item in forward]
        actual = torch.ops.aten._lstm_mps.out(
            mps_input,
            mps_hx,
            mps_params,
            True,
            1,
            0.0,
            False,
            False,
            True,
            out0=outs[0],
            out1=outs[1],
            out2=outs[2],
            out3=outs[3],
            out4=outs[4],
            out5=outs[5],
        )
        if any(actual_item is not out_item for actual_item, out_item in zip(actual, outs)):
            raise AssertionError(f"{spec.surface} did not return the provided out tensors")
        _assert_forward_outputs(outs, spec.surface)
        _assert_reserve_metadata(outs, spec.surface)
        _assert_backward_matches(outs, f"{spec.surface}.reserve_backward")
        return

    if spec.surface == "aten::lstm_mps_backward":
        _assert_backward_matches(forward, spec.surface)
        return

    if spec.surface == "aten::lstm_mps_backward.out":
        _assert_backward_matches(forward, spec.surface, use_out=True)
        return

    raise AssertionError(f"No MPS LSTM implementation for {spec.surface}")


def _assert_fused_dropout_mask(mask: torch.Tensor, input_tensor: torch.Tensor, label: str) -> None:
    if tuple(mask.shape) != tuple(input_tensor.shape):
        raise AssertionError(f"{label} mask shape mismatch: {tuple(mask.shape)} vs {tuple(input_tensor.shape)}")
    if mask.device != input_tensor.device:
        raise AssertionError(f"{label} mask device mismatch: {mask.device} vs {input_tensor.device}")
    if mask.dtype not in (torch.bool, torch.uint8):
        raise AssertionError(f"{label} mask dtype mismatch: {mask.dtype}")
    mask_cpu = mask.detach().cpu()
    if mask.dtype == torch.bool:
        return
    if not torch.all((mask_cpu == 0) | (mask_cpu == 1)):
        raise AssertionError(f"{label} mask contains values other than 0 and 1")


def _assert_fused_dropout_result(
    input_tensor: torch.Tensor,
    output: torch.Tensor,
    mask: torch.Tensor,
    keep_probability: float,
    label: str,
) -> None:
    if tuple(output.shape) != tuple(input_tensor.shape):
        raise AssertionError(f"{label} output shape mismatch: {tuple(output.shape)} vs {tuple(input_tensor.shape)}")
    if output.dtype != input_tensor.dtype:
        raise AssertionError(f"{label} output dtype mismatch: {output.dtype} vs {input_tensor.dtype}")
    if output.device != input_tensor.device:
        raise AssertionError(f"{label} output device mismatch: {output.device} vs {input_tensor.device}")
    _assert_fused_dropout_mask(mask, input_tensor, label)
    expected = input_tensor * mask.to(dtype=input_tensor.dtype) * (1.0 / keep_probability)
    _assert_close_tensor(output, expected, label, rtol=1e-6, atol=1e-6)


def _cuda_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _assert_mem_eff_dropout_mask_fill(tensor: torch.Tensor, label: str) -> None:
    if tensor.dtype != torch.float32:
        raise AssertionError(f"{label} mask dtype mismatch: {tensor.dtype}")
    if not tensor.is_contiguous():
        raise AssertionError(f"{label} mask tensor is not contiguous")
    values = tensor.detach().cpu()
    if not torch.isfinite(values).all():
        raise AssertionError(f"{label} mask contains non-finite values")
    if not torch.all((values >= 0.0) & (values <= 1.0)):
        raise AssertionError(f"{label} mask contains values outside [0, 1]")


def _run_cuda_fused_dropout(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    device_obj = torch.device(device)
    input_tensor = torch.linspace(-3.0, 3.0, steps=64, device=device_obj, dtype=torch.float32).reshape(8, 8)
    keep_probability = 0.25

    try:
        if spec.surface == "aten::_fill_mem_eff_dropout_mask_":
            first = torch.empty((1, 2, 4, 8), device=device_obj, dtype=torch.float32)
            returned = torch.ops.aten._fill_mem_eff_dropout_mask_(first, keep_probability, 12345, 0)
            assert_out_identity(returned, first, spec.surface)
            _assert_mem_eff_dropout_mask_fill(first, spec.surface)

            second = torch.empty_like(first)
            torch.ops.aten._fill_mem_eff_dropout_mask_(second, keep_probability, 12345, 0)
            _assert_close_tensor(first, second, f"{spec.surface}.deterministic_seed", rtol=0.0, atol=0.0)
            return

        if spec.surface == "aten::_fused_dropout":
            generator = _cuda_generator(device_obj, 1729)
            output, mask = torch.ops.aten._fused_dropout(input_tensor, keep_probability, generator)
            _assert_fused_dropout_result(input_tensor, output, mask, keep_probability, spec.surface)
            return

        if spec.surface == "aten::_fused_dropout.out":
            last_exc: Exception | None = None
            for mask_dtype in (torch.bool, torch.uint8):
                out0 = torch.empty_like(input_tensor)
                out1 = torch.empty_like(input_tensor, dtype=mask_dtype)
                generator = _cuda_generator(device_obj, 1729)
                try:
                    output, mask = torch.ops.aten._fused_dropout.out(
                        input_tensor,
                        keep_probability,
                        generator,
                        out0=out0,
                        out1=out1,
                    )
                except Exception as exc:
                    last_exc = exc
                    continue
                assert_out_identity(output, out0, f"{spec.surface}.out0")
                assert_out_identity(mask, out1, f"{spec.surface}.out1")
                _assert_fused_dropout_result(input_tensor, output, mask, keep_probability, spec.surface)
                return
            if last_exc is not None:
                raise last_exc
            raise AssertionError(f"{spec.surface} did not execute with any supported mask dtype")
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)

    raise AssertionError(f"No CUDA fused-dropout oracle implementation for {spec.surface}")


def _run_backend_property(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    raise OracleUnavailable(f"coverage_strategy_pending: {spec.surface} backend-pack runner is not implemented yet")


_RUNNERS: dict[str, Callable[[OracleSpec, str], None]] = {
    "sobol": _run_sobol,
    "quantized_rnn": _run_quantized_rnn,
    "int4": _run_int4,
    "dynamic_int4": _run_dynamic_int4,
    "mps_int4_pack": _run_mps_int4_pack,
    "quantized_allocation": _run_quantized_allocation,
    "linear_backward": _run_linear_backward,
    "max_pool2d_backward": _run_max_pool2d_backward,
    "unsafe_property": _run_unsafe_property,
    "autocast_property": _run_autocast_property,
    "native_batch_norm_no_stats": _run_native_batch_norm_no_stats,
    "forward_ad_inference_copy": _run_forward_ad_inference_copy,
    "nested_select_backward": _run_nested_select_backward,
    "sparse_constructor_property": _run_sparse_constructor_property,
    "cpu_flash_attention": _run_cpu_flash_attention,
    "quantized_flash_attention": _run_quantized_flash_attention,
    "privateuse1_attention": _run_privateuse1_attention,
    "privateuse1_matmul_backward": _run_privateuse1_matmul_backward,
    "privateuse1_resize_output": _run_privateuse1_resize_output,
    "privateuse1_batch_norm_forward": _run_privateuse1_batch_norm_forward,
    "privateuse1_thnn_cell": _run_privateuse1_thnn_cell,
    "privateuse1_pin_memory": _run_privateuse1_pin_memory,
    "mps_convolution": _run_mps_convolution,
    "mps_sdpa_math": _run_mps_sdpa_math,
    "mps_lstm": _run_mps_lstm,
    "cuda_fused_dropout": _run_cuda_fused_dropout,
    "backend_property": _run_backend_property,
}


_SPECS: dict[str, OracleSpec] = {}


def _register(spec: OracleSpec) -> None:
    _SPECS[spec.surface] = spec


for _surface in (
    "aten::_sobol_engine_draw",
    "aten::_sobol_engine_ff_",
    "aten::_sobol_engine_initialize_state_",
    "aten::_sobol_engine_scramble_",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="sobol_engine_state",
        coverage_status="covered_oracle",
        coverage_kind="oracle",
        runner="sobol",
        backend_gate="cpu",
        semantic_level=5,
        reason="Sobol dispatcher helpers are validated against SobolEngine state protocol and exact low-dimensional sequences.",
    ))

_register(OracleSpec(
    surface="aten::_empty_affine_quantized",
    oracle_id="quantized_affine_allocation",
    coverage_status="covered_oracle",
    coverage_kind="oracle",
    runner="quantized_allocation",
    backend_gate="cpu",
    semantic_level=5,
    reason="Internal affine quantized allocation is validated for shape, dtype, qscheme, scale, and zero point.",
))

for _surface, _runner, _reason in (
    (
        "aten::linear_backward",
        "linear_backward",
        "MPS linear backward is validated against public CPU linear autograd gradients and output_mask behavior.",
    ),
    (
        "aten::linear_backward.out",
        "linear_backward",
        "MPS linear backward out variant is validated for out identity and CPU linear autograd gradients.",
    ),
    (
        "aten::max_pool2d_backward",
        "max_pool2d_backward",
        "MPS max-pool backward is validated against public CPU max_pool2d autograd gradients.",
    ),
    (
        "aten::max_pool2d_backward.out",
        "max_pool2d_backward",
        "MPS max-pool backward out variant is validated for out identity and CPU max_pool2d autograd gradients.",
    ),
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id=_runner,
        coverage_status="covered_backend_pack",
        coverage_kind="backend_pack",
        runner=_runner,
        backend_gate="mps",
        semantic_level=4,
        reason=_reason,
    ))

for _surface in (
    "aten::_unsafe_view",
    "aten::_unsafe_view.out",
    "aten::_unsafe_index.Tensor",
    "aten::_unsafe_index_put",
    "aten::unsafe_split.Tensor_out",
    "aten::unsafe_split_with_sizes.out",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="unsafe_valid_input_semantics",
        coverage_status="covered_property",
        coverage_kind="property",
        runner="unsafe_property",
        backend_gate="cpu",
        semantic_level=5,
        reason="Unsafe helpers are validated only for valid inputs against public-equivalent view/index/split semantics.",
    ))

for _surface in (
    "aten::_autocast_to_full_precision",
    "aten::_autocast_to_reduced_precision",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="autocast_cast_policy",
        coverage_status="covered_property",
        coverage_kind="property",
        runner="autocast_property",
        backend_gate="cpu",
        semantic_level=4,
        reason="Raw autocast helpers are validated for CPU promotion/reduction policy and disabled-autocast identity behavior.",
    ))

_register(OracleSpec(
    surface="aten::_native_batch_norm_legit.no_stats_out",
    oracle_id="native_batch_norm_no_stats_out",
    coverage_status="covered_property",
    coverage_kind="property",
    runner="native_batch_norm_no_stats",
    backend_gate="any",
    semantic_level=4,
    reason="Native batch-norm no-stats out helper is validated for out identity, save_mean/save_invstd, and public batch_norm output semantics.",
))

for _surface in (
    "aten::_fw_primal_copy",
    "aten::_fw_primal_copy.out",
    "aten::_make_dual_copy",
    "aten::_make_dual_copy.out",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="forward_ad_inference_copy",
        coverage_status="covered_property",
        coverage_kind="property",
        runner="forward_ad_inference_copy",
        backend_gate="any",
        semantic_level=5,
        reason="Forward-AD copy helpers are validated through their required inference-mode direct-call path for value copy, non-aliasing, inference tensor output, and out identity.",
    ))

_register(OracleSpec(
    surface="aten::_nested_select_backward",
    oracle_id="nested_select_backward",
    coverage_status="covered_property",
    coverage_kind="property",
    runner="nested_select_backward",
    backend_gate="cpu",
    semantic_level=5,
    reason="Nested select backward is validated with real nested tensors against the public select-backward scatter semantics.",
))

for _surface in (
    "aten::_sparse_bsc_tensor_unsafe",
    "aten::_sparse_bsr_tensor_unsafe",
    "aten::_sparse_compressed_tensor_unsafe",
    "aten::_sparse_coo_tensor_unsafe",
    "aten::_sparse_csc_tensor_unsafe",
    "aten::_sparse_csr_tensor_unsafe",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="sparse_unsafe_valid_constructor",
        coverage_status="covered_property",
        coverage_kind="property",
        runner="sparse_constructor_property",
        backend_gate="cpu",
        semantic_level=5,
        reason="Unsafe sparse constructors are validated only with invariant-preserving indices, values, layouts, and sizes.",
    ))

for _surface in (
    "aten::_fill_mem_eff_dropout_mask_",
    "aten::_fused_dropout",
    "aten::_fused_dropout.out",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="fused_dropout_backend_pack",
        coverage_status="covered_backend_pack",
        coverage_kind="backend_pack",
        runner="cuda_fused_dropout",
        backend_gate="cuda",
        semantic_level=5,
        reason="CUDA fused-dropout internals are validated with direct dispatcher calls for mask/output contracts, out identity, and memory-efficient mask-fill identity/determinism.",
    ))

for _surface in (
    "aten::_sparse_semi_structured_addmm",
    "aten::_sparse_semi_structured_apply",
    "aten::_sparse_semi_structured_apply_dense",
    "aten::_sparse_semi_structured_linear",
    "aten::_sparse_semi_structured_mm",
    "aten::_sparse_semi_structured_tile",
    "aten::_to_sparse_semi_structured",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="semi_structured_sparse_backend_pack",
        coverage_status="pending_backend_pack",
        coverage_kind="backend_pack",
        runner="backend_property",
        backend_gate="cuda",
        semantic_level=5,
        reason="Semi-structured sparse internals require encoded metadata and accelerator-specific kernels, so they need a CUDA backend-pack oracle instead of generic CPU direct invocation.",
    ))

for _surface in (
    "aten::_wrapped_linear_prepack",
    "aten::_wrapped_quantized_linear_prepacked",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="wrapped_quantized_linear_backend_pack",
        coverage_status="pending_backend_pack",
        coverage_kind="backend_pack",
        runner="backend_property",
        backend_gate="fbgemm",
        semantic_level=5,
        reason="Wrapped quantized linear internals depend on FBGEMM packed-weight state and need an FBGEMM backend-pack oracle.",
    ))

for _surface in (
    "aten::_scaled_dot_product_flash_attention_for_cpu",
    "aten::_scaled_dot_product_flash_attention_for_cpu_backward",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="cpu_flash_attention_public_sdpa",
        coverage_status="covered_backend_pack",
        coverage_kind="backend_pack",
        runner="cpu_flash_attention",
        backend_gate="cpu",
        semantic_level=5,
        reason="CPU flash-attention helper is validated against public CPU scaled_dot_product_attention forward/backward.",
    ))

for _surface in (
    "aten::_efficient_attention_forward",
    "aten::_flash_attention_forward",
    "aten::_scaled_dot_product_efficient_attention",
    "aten::_scaled_dot_product_flash_attention",
    "aten::_scaled_dot_product_fused_attention_overrideable",
    "aten::_efficient_attention_backward",
    "aten::_flash_attention_backward",
    "aten::_scaled_dot_product_efficient_attention_backward",
    "aten::_scaled_dot_product_flash_attention_backward",
    "aten::_scaled_dot_product_fused_attention_overrideable_backward",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="privateuse1_attention_public_sdpa",
        coverage_status="covered_property",
        coverage_kind="property",
        runner="privateuse1_attention",
        backend_gate="privateuse1",
        semantic_level=5,
        reason="PrivateUse1 attention internals are directly validated against public CPU scaled_dot_product_attention forward values and autograd gradients.",
    ))

for _surface in (
    "aten::_flash_attention_forward.quantized",
    "aten::_scaled_dot_product_flash_attention.quantized",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="quantized_flash_attention_public_sdpa",
        coverage_status="covered_property",
        coverage_kind="property",
        runner="quantized_flash_attention",
        backend_gate="any",
        semantic_level=2,
        reason="Quantized flash-attention dispatcher surfaces are validated against public CPU scaled_dot_product_attention for fp16 and bf16 identity-descale cases on any backend that implements the kernel.",
    ))

_register(OracleSpec(
    surface="aten::matmul_backward",
    oracle_id="privateuse1_matmul_backward_formula",
    coverage_status="covered_property",
    coverage_kind="property",
    runner="privateuse1_matmul_backward",
    backend_gate="privateuse1",
    semantic_level=5,
    reason="PrivateUse1 matmul_backward is validated against explicit matrix-gradient formulas and output_mask behavior.",
))

_register(OracleSpec(
    surface="aten::matmul_backward.out",
    oracle_id="privateuse1_matmul_backward_formula",
    coverage_status="covered_property",
    coverage_kind="property",
    runner="privateuse1_matmul_backward",
    backend_gate="privateuse1",
    semantic_level=5,
    reason="PrivateUse1 matmul_backward.out is validated against explicit matrix-gradient formulas for enabled outputs and out identity.",
))

for _surface in (
    "aten::_resize_output",
    "aten::_resize_output.out",
    "aten::_resize_output_",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="privateuse1_resize_output_property",
        coverage_status="covered_property",
        coverage_kind="property",
        runner="privateuse1_resize_output",
        backend_gate="privateuse1",
        semantic_level=5,
        reason="PrivateUse1 resize-output helpers are validated for device-preserving shape mutation and in-place return identity.",
    ))

for _surface in (
    "aten::batch_norm_stats",
    "aten::batch_norm_stats.out",
    "aten::batch_norm_elemt",
    "aten::batch_norm_elemt.out",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="privateuse1_batch_norm_forward_formula",
        coverage_status="covered_property",
        coverage_kind="property",
        runner="privateuse1_batch_norm_forward",
        backend_gate="privateuse1",
        semantic_level=5,
        reason="PrivateUse1 batch-norm forward helpers are validated against explicit per-channel mean, invstd, and normalization formulas.",
    ))

for _surface in (
    "aten::_thnn_fused_gru_cell",
    "aten::_thnn_fused_gru_cell.out",
    "aten::_thnn_fused_lstm_cell",
    "aten::_thnn_fused_lstm_cell.out",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="privateuse1_thnn_cell_formula",
        coverage_status="covered_property",
        coverage_kind="property",
        runner="privateuse1_thnn_cell",
        backend_gate="privateuse1",
        semantic_level=5,
        reason="PrivateUse1 fused THNN GRU/LSTM forward cells are validated against explicit gate formulas and workspace shape contracts.",
    ))

for _surface in (
    "aten::_pin_memory",
    "aten::_pin_memory.out",
    "aten::pin_memory",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="privateuse1_pin_memory_noop",
        coverage_status="covered_property",
        coverage_kind="property",
        runner="privateuse1_pin_memory",
        backend_gate="privateuse1",
        semantic_level=5,
        reason="PrivateUse1 pinned-memory surfaces are validated as device-preserving value-copy no-ops; no host-pinned allocator semantics are claimed.",
    ))

for _surface in (
    "aten::_mps_convolution",
    "aten::_mps_convolution.out",
    "aten::_mps_convolution_transpose",
    "aten::_mps_convolution_transpose.out",
    "aten::mps_convolution_backward",
    "aten::mps_convolution_transpose_backward",
    "aten::mps_convolution_transpose_backward.out",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="mps_convolution_cpu_reference",
        coverage_status="covered_backend_pack",
        coverage_kind="backend_pack",
        runner="mps_convolution",
        backend_gate="mps",
        semantic_level=5,
        reason="MPS convolution helpers are validated against CPU conv2d/conv_transpose2d forward and gradient references.",
    ))

_register(OracleSpec(
    surface="aten::_scaled_dot_product_attention_math_for_mps",
    oracle_id="mps_sdpa_math_public_reference",
    coverage_status="covered_backend_pack",
    coverage_kind="backend_pack",
    runner="mps_sdpa_math",
    backend_gate="mps",
    semantic_level=5,
    reason="MPS SDPA math helper is validated against public scaled_dot_product_attention output.",
))

for _surface in (
    "aten::_lstm_mps",
    "aten::_lstm_mps.out",
    "aten::lstm_mps_backward",
    "aten::lstm_mps_backward.out",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="mps_lstm_cpu_reference",
        coverage_status="covered_backend_pack",
        coverage_kind="backend_pack",
        runner="mps_lstm",
        backend_gate="mps",
        semantic_level=5,
        reason="MPS LSTM helpers are validated against public LSTM forward and autograd backward references.",
    ))

for _surface in (
    "aten::quantized_lstm.input",
    "aten::quantized_lstm.data",
    "aten::quantized_gru.input",
    "aten::quantized_gru.data",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="quantized_dynamic_rnn",
        coverage_status="covered_oracle",
        coverage_kind="oracle",
        runner="quantized_rnn",
        backend_gate="quantized",
        semantic_level=5,
        reason="Modern quantized RNN dispatcher surfaces are driven with PyTorch-created CellParamsBase objects.",
    ))

for _surface in (
    "aten::quantized_lstm.input_legacy",
    "aten::quantized_lstm.data_legacy",
    "aten::quantized_gru.input_legacy",
    "aten::quantized_gru.data_legacy",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="quantized_legacy_rnn_removed",
        coverage_status="excluded_deprecated_or_removed",
        coverage_kind="excluded",
        runner="backend_property",
        backend_gate="any",
        semantic_level=5,
        reason="PyTorch reports tensor-list legacy quantized RNN parameter overloads are no longer supported.",
    ))

for _surface in (
    "aten::quantized_lstm_cell",
    "aten::quantized_gru_cell",
    "aten::quantized_rnn_relu_cell",
    "aten::quantized_rnn_tanh_cell",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="quantized_static_rnn_cell",
        coverage_status="pending_backend_pack",
        coverage_kind="backend_pack",
        runner="backend_property",
        backend_gate="fbgemm",
        semantic_level=5,
        reason="Static quantized cell schemas require legacy tensor-packed FBGEMM invariants.",
    ))

for _surface in (
    "aten::_convert_weight_to_int4pack_for_cpu",
    "aten::_weight_int4pack_mm_for_cpu",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="int4_cpu_pack_value_oracle",
        coverage_status="covered_oracle",
        coverage_kind="oracle",
        runner="int4",
        backend_gate="cpu",
        semantic_level=5,
        reason="CPU int4 packed-weight helpers are validated against a value oracle for scale-1 zero-0 dequantized matmul semantics.",
    ))

for _surface in (
    "aten::_dyn_quant_pack_4bit_weight",
    "aten::_dyn_quant_matmul_4bit",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="dynamic_int4_pack_matmul_value_oracle",
        coverage_status="covered_oracle",
        coverage_kind="oracle",
        runner="dynamic_int4",
        backend_gate="cpu",
        semantic_level=5,
        reason="CPU dynamic 4-bit pack/matmul helpers are validated by round-tripping opaque packed weights through a nibble-unpack, grouped-scale, bias-aware value oracle.",
    ))

for _surface in (
    "aten::_convert_weight_to_int4pack",
    "aten::_weight_int4pack_mm",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="int4_mps_pack",
        coverage_status="covered_backend_pack",
        coverage_kind="backend_pack",
        runner="mps_int4_pack",
        backend_gate="mps",
        semantic_level=5,
        reason="Generic MPS int4 packed-weight helpers are validated with TinyGEMM byte packing, per-group scale/zero dequantization, and CPU matmul reference values.",
    ))

_register(OracleSpec(
    surface="aten::_weight_int4pack_mm_with_scales_and_zeros",
    oracle_id="int4_scales_zeros_meta",
    coverage_status="pending_property",
    coverage_kind="property",
    runner="backend_property",
    backend_gate="any",
    semantic_level=5,
    reason="This PyTorch build exposes only metadata coverage for the explicit scale/zero int4 matmul surface.",
))

for _surface in (
    "aten::_philox_key_fold_in",
    "aten::_philox_key_split",
    "aten::_philox_normal",
    "aten::_philox_normal.out",
    "aten::_philox_normal_",
    "aten::_philox_uniform",
    "aten::_philox_uniform.out",
    "aten::_philox_uniform_",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="philox_mps_rng",
        coverage_status="pending_backend_pack",
        coverage_kind="backend_pack",
        runner="backend_property",
        backend_gate="mps",
        semantic_level=5,
        reason="Philox helpers are MPS/Meta-only in this PyTorch build and need an MPS backend-pack oracle.",
    ))


def oracle_spec_for(surface: str) -> OracleSpec | None:
    return _SPECS.get(surface)


def all_oracle_specs() -> tuple[OracleSpec, ...]:
    return tuple(_SPECS.values())


def run_oracle_for_surface(surface: str, device: str) -> None:
    spec = oracle_spec_for(surface)
    if spec is None:
        raise OracleUnavailable(f"coverage_strategy_pending: no oracle spec for {surface}")
    runner = _RUNNERS.get(spec.runner)
    if runner is None:
        raise OracleUnavailable(f"coverage_strategy_pending: no oracle runner {spec.runner!r} for {surface}")
    runner(spec, device)
