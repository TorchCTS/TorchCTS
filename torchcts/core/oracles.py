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
import math
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
from torchcts.core.version_rules import parse_torch_version


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
    contract_status: str | None = None
    contract_ref: str = ""
    promotion_evidence: str = ""
    promotion_backend: str | None = None

    def metadata(self) -> dict:
        contract_status = self.contract_status
        if contract_status is None:
            if self.coverage_status.startswith("covered_"):
                contract_status = "accepted"
            elif self.runner == "backend_property":
                contract_status = "blocked"
            else:
                contract_status = "candidate"
        return {
            "oracle_id": self.oracle_id,
            "coverage_kind": self.coverage_kind,
            "backend_gate": self.backend_gate,
            "reason": self.reason,
            "runner": self.runner,
            "contract_status": contract_status,
            "contract_ref": self.contract_ref,
            "promotion_evidence": self.promotion_evidence,
            "promotion_backend": self.promotion_backend or self.backend_gate,
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
    if gate == "cpu_build":
        if device_type != "cpu":
            raise OracleUnavailable(f"backend_not_available: {spec.surface} requires a CPU build feature")
        return
    if gate == "mps":
        if device_type != "mps" or not torch.backends.mps.is_available():
            raise OracleUnavailable(f"backend_not_available: {spec.surface} requires MPS")
        return
    if gate == "cuda":
        if device_type != "cuda" or not torch.cuda.is_available():
            raise OracleUnavailable(f"backend_not_available: {spec.surface} requires CUDA")
        return
    if gate == "rocm":
        if device_type != "cuda" or not torch.cuda.is_available() or torch.version.hip is None:
            raise OracleUnavailable(f"backend_not_available: {spec.surface} requires ROCm/HIP")
        return
    if gate == "xla":
        raise OracleUnavailable(f"backend_not_available: {spec.surface} requires a PyTorch/XLA runtime")
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
        "Supported only on",
        "supported only on",
        "architecture mismatch",
        "should be overridden in python",
        "only supported on CUDA devices with compute capability",
        "tried to directly modify sizes for customized tensor",
        "in-place mkldnn operations are not supported",
        "ATen not compiled with MKLDNN support",
        "NNPACK SpatialConvolution_updateOutput failed",
        "could not execute a primitive",
        "Your CPU doesn't support FBGEMM",
        "not built with FBGEMM",
        "FBGEMM operators",
        "Unknown qengine",
        "unknown architecure",
        "unknown architecture",
    )
    if isinstance(exc, (NotImplementedError, RuntimeError, ValueError)) and any(
        fragment in message for fragment in unavailable_fragments
    ):
        raise OracleUnavailable(f"backend_not_available: {spec.surface}: {message}") from exc
    raise exc


def _runtime_torch_at_least(version: str) -> bool:
    current = parse_torch_version(torch.__version__)
    minimum = parse_torch_version(version)
    return current is not None and minimum is not None and current >= minimum


def _require_torch_at_least(spec: OracleSpec, version: str, reason: str) -> None:
    if not _runtime_torch_at_least(version):
        raise OracleUnavailable(
            f"backend_runtime_blocked: {spec.surface} requires PyTorch >= {version}: {reason}"
        )


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
        "aten::mps_convolution_backward.out",
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
        if spec.surface == "aten::mps_convolution_backward.out":
            out0 = torch.empty_like(cpu_input, device="mps")
            out1 = torch.empty_like(cpu_weight, device="mps")
            out2 = torch.empty_like(cpu_bias, device="mps")
            actual = torch.ops.aten.mps_convolution_backward.out(
                mps_input,
                grad_output.to("mps"),
                mps_weight,
                padding,
                stride,
                dilation,
                groups,
                [True, True, True],
                out0=out0,
                out1=out1,
                out2=out2,
            )
            if actual[0] is not out0 or actual[1] is not out1 or actual[2] is not out2:
                raise AssertionError(f"{spec.surface} did not return the provided out tensors")
        else:
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
        if spec.surface == "aten::mps_convolution_backward.out":
            _assert_close_tensor(actual[2], ref_bias.grad, f"{spec.surface}.grad_bias")
        elif actual[2] is not None:
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


def _run_mps_philox(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    device_obj = torch.device(device)
    key = torch.tensor([12345, 67890], dtype=torch.int64, device=device_obj)

    try:
        if spec.surface == "aten::_philox_key_split":
            actual = torch.ops.aten._philox_key_split(key, 3)
            repeated = torch.ops.aten._philox_key_split(key, 3)
            if actual.device.type != device_obj.type or actual.dtype != key.dtype:
                raise AssertionError(f"{spec.surface} returned malformed key tensor")
            if actual.shape[0] != 3:
                raise AssertionError(f"{spec.surface} returned wrong split count {tuple(actual.shape)}")
            _assert_close_tensor(actual, repeated, spec.surface, rtol=0.0, atol=0.0)
            return

        if spec.surface == "aten::_philox_key_fold_in":
            actual = torch.ops.aten._philox_key_fold_in(key, 7)
            repeated = torch.ops.aten._philox_key_fold_in(key, 7)
            if actual.device.type != device_obj.type or actual.dtype != key.dtype:
                raise AssertionError(f"{spec.surface} returned malformed key tensor")
            if tuple(actual.shape) != tuple(key.shape):
                raise AssertionError(f"{spec.surface} returned wrong key shape {tuple(actual.shape)}")
            _assert_close_tensor(actual, repeated, spec.surface, rtol=0.0, atol=0.0)
            return

        seed = torch.empty(16, dtype=torch.float32, device=device_obj)
        if spec.surface == "aten::_philox_uniform":
            actual = torch.ops.aten._philox_uniform(seed, key, -0.25, 0.75)
            repeated = torch.ops.aten._philox_uniform(seed, key, -0.25, 0.75)
            if actual.device.type != device_obj.type or tuple(actual.shape) != tuple(seed.shape):
                raise AssertionError(f"{spec.surface} returned malformed random tensor")
            if not bool(((actual >= -0.25) & (actual < 0.75)).all().item()):
                raise AssertionError(f"{spec.surface} returned values outside requested range")
            _assert_close_tensor(actual, repeated, spec.surface, rtol=0.0, atol=0.0)
            return

        if spec.surface == "aten::_philox_uniform.out":
            out = torch.empty_like(seed)
            actual = torch.ops.aten._philox_uniform.out(seed, key, -0.25, 0.75, out=out)
            assert_out_identity(actual, out, spec.surface)
            if not bool(((out >= -0.25) & (out < 0.75)).all().item()):
                raise AssertionError(f"{spec.surface} returned values outside requested range")
            return

        if spec.surface == "aten::_philox_uniform_":
            actual = torch.ops.aten._philox_uniform_(seed, key, -0.25, 0.75)
            assert_out_identity(actual, seed, spec.surface)
            if not bool(((seed >= -0.25) & (seed < 0.75)).all().item()):
                raise AssertionError(f"{spec.surface} returned values outside requested range")
            return

        if spec.surface == "aten::_philox_normal":
            actual = torch.ops.aten._philox_normal(seed, key, 0.5, 1.25)
            repeated = torch.ops.aten._philox_normal(seed, key, 0.5, 1.25)
            if actual.device.type != device_obj.type or tuple(actual.shape) != tuple(seed.shape):
                raise AssertionError(f"{spec.surface} returned malformed random tensor")
            if not torch.isfinite(actual).all().item():
                raise AssertionError(f"{spec.surface} returned non-finite values")
            _assert_close_tensor(actual, repeated, spec.surface, rtol=0.0, atol=0.0)
            return

        if spec.surface == "aten::_philox_normal.out":
            out = torch.empty_like(seed)
            actual = torch.ops.aten._philox_normal.out(seed, key, 0.5, 1.25, out=out)
            assert_out_identity(actual, out, spec.surface)
            if not torch.isfinite(out).all().item():
                raise AssertionError(f"{spec.surface} returned non-finite values")
            return

        if spec.surface == "aten::_philox_normal_":
            actual = torch.ops.aten._philox_normal_(seed, key, 0.5, 1.25)
            assert_out_identity(actual, seed, spec.surface)
            if not torch.isfinite(seed).all().item():
                raise AssertionError(f"{spec.surface} returned non-finite values")
            return
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)

    raise AssertionError(f"No MPS Philox oracle implementation for {spec.surface}")


def _mkldnn_is_available() -> bool:
    try:
        return bool(torch.backends.mkldnn.is_available()) and bool(torch.backends.mkldnn.enabled)
    except Exception:
        return False


def _require_mkldnn(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    if not _mkldnn_is_available():
        raise OracleUnavailable(f"backend_not_available: {spec.surface} requires an MKLDNN-enabled PyTorch build")


def _to_dense_if_mkldnn(tensor: torch.Tensor) -> torch.Tensor:
    if getattr(tensor, "is_mkldnn", False):
        return tensor.to_dense()
    return tensor


def _assert_close_mkldnn(actual: torch.Tensor, expected: torch.Tensor, label: str, *, rtol=1e-4, atol=1e-4) -> None:
    _assert_close_tensor(_to_dense_if_mkldnn(actual), expected, label, rtol=rtol, atol=atol)


def _assert_mkldnn_tensor(tensor: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if not getattr(tensor, "is_mkldnn", False):
        raise AssertionError(f"{label} did not return an MKLDNN tensor")
    if tuple(tensor.shape) != tuple(expected.shape):
        raise AssertionError(f"{label} shape mismatch: {tuple(tensor.shape)} vs {tuple(expected.shape)}")
    if tensor.dtype != expected.dtype:
        raise AssertionError(f"{label} dtype mismatch: {tensor.dtype} vs {expected.dtype}")


def _assert_mkldnn_out_identity(actual: torch.Tensor, out: torch.Tensor, label: str) -> None:
    if actual is not out:
        raise AssertionError(f"{label} did not return the provided out tensor")
    if not getattr(out, "is_mkldnn", False):
        raise AssertionError(f"{label} out tensor is not MKLDNN")


def _run_mkldnn_shape(spec: OracleSpec, device: str) -> None:
    _require_mkldnn(spec, device)
    source = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    mkldnn_source = source.to_mkldnn()

    try:
        if spec.surface == "aten::_mkldnn_reshape":
            expected = source.reshape(1, 2, 2, 6)
            actual = torch.ops.aten._mkldnn_reshape(mkldnn_source, list(expected.shape))
            _assert_mkldnn_tensor(actual, expected, spec.surface)
            _assert_close_mkldnn(actual, expected, spec.surface)
            return

        if spec.surface == "aten::_mkldnn_reshape.out":
            expected = source.reshape(1, 2, 2, 6)
            out = torch.empty_like(expected).to_mkldnn()
            actual = torch.ops.aten._mkldnn_reshape.out(mkldnn_source, list(expected.shape), out=out)
            _assert_mkldnn_out_identity(actual, out, spec.surface)
            _assert_close_mkldnn(out, expected, spec.surface)
            return

        if spec.surface == "aten::_mkldnn_transpose":
            expected = source.transpose(1, 2)
            actual = torch.ops.aten._mkldnn_transpose(mkldnn_source, 1, 2)
            _assert_mkldnn_tensor(actual, expected, spec.surface)
            _assert_close_mkldnn(actual, expected, spec.surface)
            return

        if spec.surface == "aten::_mkldnn_transpose.out":
            expected = source.transpose(1, 2)
            out = torch.empty_like(expected).to_mkldnn()
            actual = torch.ops.aten._mkldnn_transpose.out(mkldnn_source, 1, 2, out=out)
            _assert_mkldnn_out_identity(actual, out, spec.surface)
            _assert_close_mkldnn(out, expected, spec.surface)
            return

        if spec.surface == "aten::_mkldnn_transpose_":
            torch.ops.aten._mkldnn_transpose_(mkldnn_source, 1, 2)
            raise AssertionError(f"{spec.surface} unexpectedly succeeded despite PyTorch source marking it unsupported")

        if spec.surface == "aten::to_mkldnn":
            actual = torch.ops.aten.to_mkldnn(source, None)
            _assert_mkldnn_tensor(actual, source, spec.surface)
            _assert_close_mkldnn(actual, source, spec.surface)
            return

        if spec.surface == "aten::to_mkldnn.out":
            out = torch.empty_like(source).to_mkldnn()
            actual = torch.ops.aten.to_mkldnn.out(source, None, out=out)
            _assert_mkldnn_out_identity(actual, out, spec.surface)
            _assert_close_mkldnn(out, source, spec.surface)
            return

        if spec.surface == "aten::to_mkldnn_backward":
            grad_dense = torch.arange(24, dtype=torch.float32).reshape_as(source)
            actual = torch.ops.aten.to_mkldnn_backward(grad_dense.to_mkldnn(), source)
            _assert_close_tensor(actual, grad_dense, spec.surface)
            return
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)

    raise AssertionError(f"No MKLDNN shape oracle implementation for {spec.surface}")


def _mkldnn_linear_sample() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(501)
    input_tensor = torch.randn(3, 5, dtype=torch.float32)
    weight = torch.randn(4, 5, dtype=torch.float32)
    bias = torch.randn(4, dtype=torch.float32)
    grad_output = torch.randn(3, 4, dtype=torch.float32)
    return input_tensor, weight, bias, grad_output


def _mkldnn_linear_reference(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ref_input = input_tensor.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)
    ref_bias = bias.detach().clone().requires_grad_(True)
    output = torch.nn.functional.linear(ref_input, ref_weight, ref_bias)
    output.backward(grad_output)
    return output.detach(), ref_input.grad, ref_weight.grad, ref_bias.grad


def _run_mkldnn_linear(spec: OracleSpec, device: str) -> None:
    _require_mkldnn(spec, device)
    input_tensor, weight, bias, grad_output = _mkldnn_linear_sample()
    expected, grad_input, grad_weight, grad_bias = _mkldnn_linear_reference(input_tensor, weight, bias, grad_output)
    mkldnn_input = input_tensor.to_mkldnn()
    mkldnn_grad_output = grad_output.to_mkldnn()

    try:
        if spec.surface == "aten::mkldnn_linear":
            actual = torch.ops.aten.mkldnn_linear(mkldnn_input, weight, bias)
            _assert_mkldnn_tensor(actual, expected, spec.surface)
            _assert_close_mkldnn(actual, expected, spec.surface)
            return

        if spec.surface == "aten::mkldnn_linear.out":
            out = torch.empty_like(expected).to_mkldnn()
            actual = torch.ops.aten.mkldnn_linear.out(mkldnn_input, weight, bias, out=out)
            _assert_mkldnn_out_identity(actual, out, spec.surface)
            _assert_close_mkldnn(out, expected, spec.surface)
            return

        if spec.surface == "aten::mkldnn_linear_backward":
            actual = torch.ops.aten.mkldnn_linear_backward(mkldnn_input, mkldnn_grad_output, weight, [True, True, True])
            _assert_close_mkldnn(actual[0], grad_input, f"{spec.surface}.grad_input")
            _assert_close_tensor(actual[1], grad_weight, f"{spec.surface}.grad_weight")
            _assert_close_tensor(actual[2], grad_bias, f"{spec.surface}.grad_bias")
            return

        if spec.surface == "aten::mkldnn_linear_backward.out":
            out0 = torch.empty_like(input_tensor).to_mkldnn()
            out1 = torch.empty_like(weight)
            out2 = torch.empty_like(bias)
            actual = torch.ops.aten.mkldnn_linear_backward.out(
                mkldnn_input,
                mkldnn_grad_output,
                weight,
                [True, True, True],
                out0=out0,
                out1=out1,
                out2=out2,
            )
            _assert_mkldnn_out_identity(actual[0], out0, f"{spec.surface}.out0")
            assert_out_identity(actual[1], out1, f"{spec.surface}.out1")
            assert_out_identity(actual[2], out2, f"{spec.surface}.out2")
            _assert_close_mkldnn(out0, grad_input, f"{spec.surface}.grad_input")
            _assert_close_tensor(out1, grad_weight, f"{spec.surface}.grad_weight")
            _assert_close_tensor(out2, grad_bias, f"{spec.surface}.grad_bias")
            return

        if spec.surface == "aten::mkldnn_linear_backward_input":
            actual = torch.ops.aten.mkldnn_linear_backward_input(list(input_tensor.shape), mkldnn_grad_output, weight)
            _assert_close_mkldnn(actual, grad_input, spec.surface)
            return

        if spec.surface == "aten::mkldnn_linear_backward_input.out":
            out = torch.empty_like(input_tensor).to_mkldnn()
            actual = torch.ops.aten.mkldnn_linear_backward_input.out(
                list(input_tensor.shape),
                mkldnn_grad_output,
                weight,
                out=out,
            )
            _assert_mkldnn_out_identity(actual, out, spec.surface)
            _assert_close_mkldnn(out, grad_input, spec.surface)
            return

        if spec.surface == "aten::mkldnn_linear_backward_weights":
            actual = torch.ops.aten.mkldnn_linear_backward_weights(mkldnn_grad_output, mkldnn_input, weight, True)
            _assert_close_tensor(actual[0], grad_weight, f"{spec.surface}.grad_weight")
            _assert_close_tensor(actual[1], grad_bias, f"{spec.surface}.grad_bias")
            return

        if spec.surface == "aten::mkldnn_linear_backward_weights.out":
            out0 = torch.empty_like(weight)
            out1 = torch.empty_like(bias)
            actual = torch.ops.aten.mkldnn_linear_backward_weights.out(
                mkldnn_grad_output,
                mkldnn_input,
                weight,
                True,
                out0=out0,
                out1=out1,
            )
            assert_out_identity(actual[0], out0, f"{spec.surface}.out0")
            assert_out_identity(actual[1], out1, f"{spec.surface}.out1")
            _assert_close_tensor(out0, grad_weight, f"{spec.surface}.grad_weight")
            _assert_close_tensor(out1, grad_bias, f"{spec.surface}.grad_bias")
            return
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)

    raise AssertionError(f"No MKLDNN linear oracle implementation for {spec.surface}")


def _run_mkldnn_convolution(spec: OracleSpec, device: str) -> None:
    _require_mkldnn(spec, device)
    torch.manual_seed(502)

    if "conv3d" in spec.surface:
        input_tensor = torch.randn(1, 2, 4, 5, 5, dtype=torch.float32)
        weight = torch.randn(3, 2, 3, 3, 3, dtype=torch.float32)
        bias = torch.randn(3, dtype=torch.float32)
        padding = [1, 1, 1]
        stride = [1, 1, 1]
        dilation = [1, 1, 1]
        expected = torch.nn.functional.conv3d(input_tensor, weight, bias, padding=1)
    else:
        input_tensor = torch.randn(1, 2, 5, 5, dtype=torch.float32)
        weight = torch.randn(3, 2, 3, 3, dtype=torch.float32)
        bias = torch.randn(3, dtype=torch.float32)
        padding = [1, 1]
        stride = [1, 1]
        dilation = [1, 1]
        expected = torch.nn.functional.conv2d(input_tensor, weight, bias, padding=1)
    groups = 1
    mkldnn_input = input_tensor.to_mkldnn()

    try:
        if spec.surface == "aten::mkldnn_convolution":
            actual = torch.ops.aten.mkldnn_convolution(mkldnn_input, weight, bias, padding, stride, dilation, groups)
            _assert_mkldnn_tensor(actual, expected, spec.surface)
            _assert_close_mkldnn(actual, expected, spec.surface)
            return

        if spec.surface == "aten::mkldnn_convolution.out":
            out = torch.empty_like(expected).to_mkldnn()
            actual = torch.ops.aten.mkldnn_convolution.out(
                mkldnn_input,
                weight,
                bias,
                padding,
                stride,
                dilation,
                groups,
                out=out,
            )
            _assert_mkldnn_out_identity(actual, out, spec.surface)
            _assert_close_mkldnn(out, expected, spec.surface)
            return

        if spec.surface == "aten::mkldnn_reorder_conv2d_weight":
            reordered = torch.ops.aten.mkldnn_reorder_conv2d_weight(weight, padding, stride, dilation, groups, list(input_tensor.shape))
            _assert_mkldnn_tensor(reordered, weight, spec.surface)
            actual = torch.ops.aten.mkldnn_convolution(mkldnn_input, reordered, bias, padding, stride, dilation, groups)
            _assert_close_mkldnn(actual, expected, f"{spec.surface}.conv")
            return

        if spec.surface == "aten::mkldnn_reorder_conv2d_weight.out":
            out = torch.empty_like(weight).to_mkldnn()
            actual = torch.ops.aten.mkldnn_reorder_conv2d_weight.out(
                weight,
                padding,
                stride,
                dilation,
                groups,
                list(input_tensor.shape),
                out=out,
            )
            _assert_mkldnn_out_identity(actual, out, spec.surface)
            conv = torch.ops.aten.mkldnn_convolution(mkldnn_input, out, bias, padding, stride, dilation, groups)
            _assert_close_mkldnn(conv, expected, f"{spec.surface}.conv")
            return

        if spec.surface == "aten::mkldnn_reorder_conv3d_weight":
            reordered = torch.ops.aten.mkldnn_reorder_conv3d_weight(weight, padding, stride, dilation, groups, list(input_tensor.shape))
            _assert_mkldnn_tensor(reordered, weight, spec.surface)
            actual = torch.ops.aten.mkldnn_convolution(mkldnn_input, reordered, bias, padding, stride, dilation, groups)
            _assert_close_mkldnn(actual, expected, f"{spec.surface}.conv")
            return

        if spec.surface == "aten::mkldnn_reorder_conv3d_weight.out":
            out = torch.empty_like(weight).to_mkldnn()
            actual = torch.ops.aten.mkldnn_reorder_conv3d_weight.out(
                weight,
                padding,
                stride,
                dilation,
                groups,
                list(input_tensor.shape),
                out=out,
            )
            _assert_mkldnn_out_identity(actual, out, spec.surface)
            conv = torch.ops.aten.mkldnn_convolution(mkldnn_input, out, bias, padding, stride, dilation, groups)
            _assert_close_mkldnn(conv, expected, f"{spec.surface}.conv")
            return
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)

    raise AssertionError(f"No MKLDNN convolution oracle implementation for {spec.surface}")


def _run_mkldnn_pooling(spec: OracleSpec, device: str) -> None:
    _require_mkldnn(spec, device)
    torch.manual_seed(503)

    try:
        if "adaptive_avg_pool2d" in spec.surface:
            input_tensor = torch.randn(1, 2, 4, 4, dtype=torch.float32)
            mkldnn_input = input_tensor.to_mkldnn()
            output_size = [2, 2]
            expected = torch.nn.functional.adaptive_avg_pool2d(input_tensor, output_size)
            if spec.surface == "aten::mkldnn_adaptive_avg_pool2d":
                actual = torch.ops.aten.mkldnn_adaptive_avg_pool2d(mkldnn_input, output_size)
                _assert_close_mkldnn(actual, expected, spec.surface)
                return
            if spec.surface == "aten::mkldnn_adaptive_avg_pool2d.out":
                out = torch.empty_like(expected).to_mkldnn()
                actual = torch.ops.aten.mkldnn_adaptive_avg_pool2d.out(mkldnn_input, output_size, out=out)
                _assert_mkldnn_out_identity(actual, out, spec.surface)
                _assert_close_mkldnn(out, expected, spec.surface)
                return
            grad_output = torch.randn_like(expected)
            ref_input = input_tensor.detach().clone().requires_grad_(True)
            torch.nn.functional.adaptive_avg_pool2d(ref_input, output_size).backward(grad_output)
            if spec.surface == "aten::mkldnn_adaptive_avg_pool2d_backward":
                actual = torch.ops.aten.mkldnn_adaptive_avg_pool2d_backward(grad_output.to_mkldnn(), mkldnn_input)
                _assert_close_mkldnn(actual, ref_input.grad, spec.surface)
                return
            if spec.surface == "aten::mkldnn_adaptive_avg_pool2d_backward.out":
                out = torch.empty_like(input_tensor).to_mkldnn()
                actual = torch.ops.aten.mkldnn_adaptive_avg_pool2d_backward.out(
                    grad_output.to_mkldnn(),
                    mkldnn_input,
                    out=out,
                )
                _assert_mkldnn_out_identity(actual, out, spec.surface)
                _assert_close_mkldnn(out, ref_input.grad, spec.surface)
                return

        if "max_pool3d" in spec.surface:
            input_tensor = torch.randn(1, 2, 4, 4, 4, dtype=torch.float32)
            kernel_size = [2, 2, 2]
            stride = [2, 2, 2]
            padding = [0, 0, 0]
            dilation = [1, 1, 1]
            pool = torch.nn.functional.max_pool3d
        else:
            input_tensor = torch.randn(1, 2, 4, 4, dtype=torch.float32)
            kernel_size = [2, 2]
            stride = [2, 2]
            padding = [0, 0]
            dilation = [1, 1]
            pool = torch.nn.functional.max_pool2d

        mkldnn_input = input_tensor.to_mkldnn()
        expected = pool(input_tensor, kernel_size, stride, padding, dilation, False)
        op = torch.ops.aten.mkldnn_max_pool3d if "max_pool3d" in spec.surface else torch.ops.aten.mkldnn_max_pool2d
        backward_op = (
            torch.ops.aten.mkldnn_max_pool3d_backward
            if "max_pool3d" in spec.surface
            else torch.ops.aten.mkldnn_max_pool2d_backward
        )

        if spec.surface in {"aten::mkldnn_max_pool2d", "aten::mkldnn_max_pool3d"}:
            actual = op(mkldnn_input, kernel_size, stride, padding, dilation, False)
            _assert_close_mkldnn(actual, expected, spec.surface)
            return

        if spec.surface in {"aten::mkldnn_max_pool2d.out", "aten::mkldnn_max_pool3d.out"}:
            out = torch.empty_like(expected).to_mkldnn()
            actual = op.out(mkldnn_input, kernel_size, stride, padding, dilation, False, out=out)
            _assert_mkldnn_out_identity(actual, out, spec.surface)
            _assert_close_mkldnn(out, expected, spec.surface)
            return

        grad_output = torch.randn_like(expected)
        ref_input = input_tensor.detach().clone().requires_grad_(True)
        pool(ref_input, kernel_size, stride, padding, dilation, False).backward(grad_output)
        mkldnn_output = op(mkldnn_input, kernel_size, stride, padding, dilation, False)

        if spec.surface in {"aten::mkldnn_max_pool2d_backward", "aten::mkldnn_max_pool3d_backward"}:
            actual = backward_op(grad_output.to_mkldnn(), mkldnn_output, mkldnn_input, kernel_size, stride, padding, dilation, False)
            _assert_close_mkldnn(actual, ref_input.grad, spec.surface)
            return

        if spec.surface in {"aten::mkldnn_max_pool2d_backward.out", "aten::mkldnn_max_pool3d_backward.out"}:
            out = torch.empty_like(input_tensor).to_mkldnn()
            actual = backward_op.out(
                grad_output.to_mkldnn(),
                mkldnn_output,
                mkldnn_input,
                kernel_size,
                stride,
                padding,
                dilation,
                False,
                out=out,
            )
            _assert_mkldnn_out_identity(actual, out, spec.surface)
            _assert_close_mkldnn(out, ref_input.grad, spec.surface)
            return
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)

    raise AssertionError(f"No MKLDNN pooling oracle implementation for {spec.surface}")


def _run_nnpack_convolution(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    if not torch.backends.nnpack.is_available():
        raise OracleUnavailable(f"backend_not_available: {spec.surface} requires NNPACK")
    torch.manual_seed(504)
    input_tensor = torch.randn(1, 2, 8, 8, dtype=torch.float32)
    weight = torch.randn(3, 2, 3, 3, dtype=torch.float32)
    bias = torch.randn(3, dtype=torch.float32)
    padding = [1, 1]
    stride = [1, 1]
    expected = torch.nn.functional.conv2d(input_tensor, weight, bias, padding=padding, stride=stride)

    try:
        if spec.surface == "aten::_nnpack_spatial_convolution":
            actual = torch.ops.aten._nnpack_spatial_convolution(input_tensor, weight, bias, padding, stride)
            _assert_close_tensor(actual, expected, spec.surface, rtol=1e-4, atol=1e-4)
            return
        if spec.surface == "aten::_nnpack_spatial_convolution.out":
            out = torch.empty_like(expected)
            actual = torch.ops.aten._nnpack_spatial_convolution.out(
                input_tensor,
                weight,
                bias,
                padding,
                stride,
                out=out,
            )
            assert_out_identity(actual, out, spec.surface)
            _assert_close_tensor(out, expected, spec.surface, rtol=1e-4, atol=1e-4)
            return
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)

    raise AssertionError(f"No NNPACK convolution oracle implementation for {spec.surface}")


def _fbgemm_weight_sample() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_tensor = torch.linspace(-0.75, 0.75, steps=8, dtype=torch.float32).reshape(2, 4)
    weight = torch.tensor(
        [
            [-0.50, -0.20, 0.10, 0.40],
            [0.35, -0.15, 0.25, -0.45],
            [0.20, 0.30, -0.35, 0.15],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor([0.10, -0.20, 0.05], dtype=torch.float32)
    return input_tensor, weight, bias


def _fbgemm_quantize_pack(weight: torch.Tensor):
    qweight, col_offsets, scale, zero_point = torch.ops.aten.fbgemm_linear_quantize_weight(weight)
    packed = torch.ops.aten.fbgemm_pack_quantized_matrix(qweight)
    return qweight, col_offsets, float(scale), int(zero_point), packed


def _fbgemm_dequantize_weight(qweight: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    return (qweight.to(torch.float32) - float(zero_point)) * float(scale)


def _run_fbgemm_linear(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    input_tensor, weight, bias = _fbgemm_weight_sample()

    try:
        if spec.surface == "aten::fbgemm_linear_quantize_weight":
            qweight, col_offsets, scale, zero_point = torch.ops.aten.fbgemm_linear_quantize_weight(weight)
            if qweight.dtype != torch.int8 or tuple(qweight.shape) != tuple(weight.shape):
                raise AssertionError(f"{spec.surface} returned malformed quantized weight")
            if col_offsets.dtype != torch.int32 or tuple(col_offsets.shape) != (weight.shape[0],):
                raise AssertionError(f"{spec.surface} returned malformed column offsets")
            if not math.isfinite(float(scale)) or float(scale) <= 0:
                raise AssertionError(f"{spec.surface} returned invalid weight scale {scale!r}")
            if not isinstance(int(zero_point), int):
                raise AssertionError(f"{spec.surface} returned invalid zero point {zero_point!r}")
            return

        if spec.surface in {
            "aten::fbgemm_pack_quantized_matrix",
            "aten::fbgemm_pack_quantized_matrix.KN",
            "aten::fbgemm_linear_int8_weight",
            "aten::fbgemm_linear_int8_weight_fp32_activation",
        }:
            qweight, col_offsets, scale, zero_point, packed = _fbgemm_quantize_pack(weight)
            if qweight.dtype != torch.int8 or tuple(qweight.shape) != tuple(weight.shape):
                raise AssertionError(f"{spec.surface} returned malformed quantized weight")
            if col_offsets.dtype != torch.int32 or tuple(col_offsets.shape) != (weight.shape[0],):
                raise AssertionError(f"{spec.surface} returned malformed column offsets")
            if not math.isfinite(scale) or scale <= 0:
                raise AssertionError(f"{spec.surface} returned invalid weight scale {scale!r}")

            if spec.surface == "aten::fbgemm_pack_quantized_matrix.KN":
                packed = torch.ops.aten.fbgemm_pack_quantized_matrix.KN(qweight, qweight.shape[1], qweight.shape[0])
            elif spec.surface == "aten::fbgemm_pack_quantized_matrix":
                packed = torch.ops.aten.fbgemm_pack_quantized_matrix(qweight)

            if spec.surface in {"aten::fbgemm_pack_quantized_matrix", "aten::fbgemm_pack_quantized_matrix.KN"}:
                actual = torch.ops.aten.fbgemm_linear_int8_weight(
                    input_tensor,
                    qweight,
                    packed,
                    col_offsets,
                    scale,
                    zero_point,
                    bias,
                )
                expected = input_tensor @ _fbgemm_dequantize_weight(qweight, scale, zero_point).T + bias
                _assert_close_tensor(actual, expected, spec.surface, rtol=0.0, atol=0.08)
                return

            actual = torch.ops.aten.fbgemm_linear_int8_weight_fp32_activation(
                input_tensor,
                qweight,
                packed,
                col_offsets,
                scale,
                zero_point,
                bias,
            )
            alias = torch.ops.aten.fbgemm_linear_int8_weight(
                input_tensor,
                qweight,
                packed,
                col_offsets,
                scale,
                zero_point,
                bias,
            )
            expected = input_tensor @ _fbgemm_dequantize_weight(qweight, scale, zero_point).T + bias
            _assert_close_tensor(actual, alias, f"{spec.surface}.alias", rtol=0.0, atol=0.0)
            _assert_close_tensor(actual, expected, spec.surface, rtol=0.0, atol=0.08)
            return

        if spec.surface in {
            "aten::fbgemm_pack_gemm_matrix_fp16",
            "aten::fbgemm_linear_fp16_weight",
            "aten::fbgemm_linear_fp16_weight.out",
            "aten::fbgemm_linear_fp16_weight_fp32_activation",
            "aten::fbgemm_linear_fp16_weight_fp32_activation.out",
        }:
            packed_fp16 = torch.ops.aten.fbgemm_pack_gemm_matrix_fp16(weight)
            expected = input_tensor @ weight.T + bias

            if spec.surface == "aten::fbgemm_pack_gemm_matrix_fp16":
                actual = torch.ops.aten.fbgemm_linear_fp16_weight(input_tensor, packed_fp16, bias)
                _assert_close_tensor(actual, expected, spec.surface, rtol=2e-2, atol=2e-2)
                return

            if spec.surface == "aten::fbgemm_linear_fp16_weight":
                actual = torch.ops.aten.fbgemm_linear_fp16_weight(input_tensor, packed_fp16, bias)
                _assert_close_tensor(actual, expected, spec.surface, rtol=2e-2, atol=2e-2)
                return

            if spec.surface == "aten::fbgemm_linear_fp16_weight.out":
                out = torch.empty(0, dtype=torch.float32)
                actual = torch.ops.aten.fbgemm_linear_fp16_weight.out(input_tensor, packed_fp16, bias, out)
                assert_out_identity(actual, out, spec.surface)
                _assert_close_tensor(out, expected, spec.surface, rtol=2e-2, atol=2e-2)
                return

            if spec.surface == "aten::fbgemm_linear_fp16_weight_fp32_activation":
                actual = torch.ops.aten.fbgemm_linear_fp16_weight_fp32_activation(input_tensor, packed_fp16, bias)
                _assert_close_tensor(actual, expected, spec.surface, rtol=2e-2, atol=2e-2)
                return

            if spec.surface == "aten::fbgemm_linear_fp16_weight_fp32_activation.out":
                out = torch.empty(0, dtype=torch.float32)
                actual = torch.ops.aten.fbgemm_linear_fp16_weight_fp32_activation.out(input_tensor, packed_fp16, bias, out)
                assert_out_identity(actual, out, spec.surface)
                _assert_close_tensor(out, expected, spec.surface, rtol=2e-2, atol=2e-2)
                return
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)

    raise AssertionError(f"No FBGEMM linear oracle implementation for {spec.surface}")


def _run_fbgemm_wrapped_linear(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    input_tensor, weight, bias = _fbgemm_weight_sample()
    input_scale = torch.tensor(0.05, dtype=torch.float32)
    input_zero_point = torch.tensor(2, dtype=torch.int64)
    weight_scale = torch.tensor(0.04, dtype=torch.float32)
    weight_zero_point = torch.tensor(0, dtype=torch.int64)
    output_scale = torch.tensor(0.03, dtype=torch.float32)
    output_zero_point = torch.tensor(1, dtype=torch.int64)

    try:
        packed = torch.ops.aten._wrapped_linear_prepack(weight, weight_scale, weight_zero_point, bias)
        if spec.surface == "aten::_wrapped_linear_prepack":
            actual = torch.ops.aten._wrapped_quantized_linear_prepacked(
                input_tensor,
                input_scale,
                input_zero_point,
                packed,
                output_scale,
                output_zero_point,
                weight.shape[0],
            )
            qinput = torch.quantize_per_tensor(input_tensor, float(input_scale), int(input_zero_point), torch.quint8)
            qweight = torch.quantize_per_tensor(weight, float(weight_scale), int(weight_zero_point), torch.qint8)
            public_packed = torch.ops.quantized.linear_prepack(qweight, bias)
            expected = torch.ops.quantized.linear(
                qinput,
                public_packed,
                float(output_scale),
                int(output_zero_point),
            ).dequantize()
            _assert_close_tensor(actual, expected, spec.surface, rtol=0.0, atol=0.0)
            return

        if spec.surface == "aten::_wrapped_quantized_linear_prepacked":
            actual = torch.ops.aten._wrapped_quantized_linear_prepacked(
                input_tensor,
                input_scale,
                input_zero_point,
                packed,
                output_scale,
                output_zero_point,
                weight.shape[0],
            )
            qinput = torch.quantize_per_tensor(input_tensor, float(input_scale), int(input_zero_point), torch.quint8)
            qweight = torch.quantize_per_tensor(weight, float(weight_scale), int(weight_zero_point), torch.qint8)
            public_packed = torch.ops.quantized.linear_prepack(qweight, bias)
            expected = torch.ops.quantized.linear(
                qinput,
                public_packed,
                float(output_scale),
                int(output_zero_point),
            ).dequantize()
            _assert_close_tensor(actual, expected, spec.surface, rtol=0.0, atol=0.0)
            return
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)

    raise AssertionError(f"No wrapped FBGEMM linear oracle implementation for {spec.surface}")


def _fbgemm_linear_int8_reference(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    qweight, col_offsets, scale, zero_point, packed = _fbgemm_quantize_pack(weight)
    return torch.ops.aten.fbgemm_linear_int8_weight(
        input_tensor,
        qweight,
        packed,
        col_offsets,
        scale,
        zero_point,
        bias,
    )


def _fbgemm_cell_params(weight_ih: torch.Tensor, weight_hh: torch.Tensor):
    qweight_ih, col_offsets_ih, scale_ih, zero_point_ih, packed_ih = _fbgemm_quantize_pack(weight_ih)
    qweight_hh, col_offsets_hh, scale_hh, zero_point_hh, packed_hh = _fbgemm_quantize_pack(weight_hh)
    return (
        qweight_ih,
        qweight_hh,
        packed_ih,
        packed_hh,
        col_offsets_ih,
        col_offsets_hh,
        scale_ih,
        scale_hh,
        zero_point_ih,
        zero_point_hh,
    )


def _run_fbgemm_quantized_cell(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    torch.manual_seed(506)
    batch = 2
    input_size = 3
    hidden_size = 4
    input_tensor = torch.randn(batch, input_size, dtype=torch.float32) * 0.25
    hx = torch.randn(batch, hidden_size, dtype=torch.float32) * 0.25

    try:
        if spec.surface == "aten::quantized_lstm_cell":
            gates = 4 * hidden_size
            cx = torch.randn(batch, hidden_size, dtype=torch.float32) * 0.25
            w_ih = torch.randn(gates, input_size, dtype=torch.float32) * 0.25
            w_hh = torch.randn(gates, hidden_size, dtype=torch.float32) * 0.25
            b_ih = torch.randn(gates, dtype=torch.float32) * 0.05
            b_hh = torch.randn(gates, dtype=torch.float32) * 0.05
            (
                qweight_ih,
                qweight_hh,
                packed_ih,
                packed_hh,
                col_offsets_ih,
                col_offsets_hh,
                scale_ih,
                scale_hh,
                zero_point_ih,
                zero_point_hh,
            ) = _fbgemm_cell_params(w_ih, w_hh)
            actual_hy, actual_cy = torch.ops.aten.quantized_lstm_cell(
                input_tensor,
                [hx, cx],
                qweight_ih,
                qweight_hh,
                b_ih,
                b_hh,
                packed_ih,
                packed_hh,
                col_offsets_ih,
                col_offsets_hh,
                scale_ih,
                scale_hh,
                zero_point_ih,
                zero_point_hh,
            )
            gates_tensor = _fbgemm_linear_int8_reference(input_tensor, w_ih, b_ih) + _fbgemm_linear_int8_reference(hx, w_hh, b_hh)
            ingate, forgetgate, cellgate, outgate = gates_tensor.chunk(4, dim=1)
            expected_cy = forgetgate.sigmoid() * cx + ingate.sigmoid() * cellgate.tanh()
            expected_hy = outgate.sigmoid() * expected_cy.tanh()
            _assert_close_tensor(actual_hy, expected_hy, f"{spec.surface}.hy", rtol=1e-5, atol=1e-5)
            _assert_close_tensor(actual_cy, expected_cy, f"{spec.surface}.cy", rtol=1e-5, atol=1e-5)
            return

        if spec.surface == "aten::quantized_gru_cell":
            gates = 3 * hidden_size
            w_ih = torch.randn(gates, input_size, dtype=torch.float32) * 0.25
            w_hh = torch.randn(gates, hidden_size, dtype=torch.float32) * 0.25
            b_ih = torch.randn(gates, dtype=torch.float32) * 0.05
            b_hh = torch.randn(gates, dtype=torch.float32) * 0.05
            (
                qweight_ih,
                qweight_hh,
                packed_ih,
                packed_hh,
                col_offsets_ih,
                col_offsets_hh,
                scale_ih,
                scale_hh,
                zero_point_ih,
                zero_point_hh,
            ) = _fbgemm_cell_params(w_ih, w_hh)
            actual = torch.ops.aten.quantized_gru_cell(
                input_tensor,
                hx,
                qweight_ih,
                qweight_hh,
                b_ih,
                b_hh,
                packed_ih,
                packed_hh,
                col_offsets_ih,
                col_offsets_hh,
                scale_ih,
                scale_hh,
                zero_point_ih,
                zero_point_hh,
            )
            input_gates = _fbgemm_linear_int8_reference(input_tensor, w_ih, b_ih)
            hidden_gates = _fbgemm_linear_int8_reference(hx, w_hh, b_hh)
            i_r, i_z, i_n = input_gates.chunk(3, dim=1)
            h_r, h_z, h_n = hidden_gates.chunk(3, dim=1)
            resetgate = (i_r + h_r).sigmoid()
            inputgate = (i_z + h_z).sigmoid()
            newgate = (i_n + resetgate * h_n).tanh()
            expected = newgate + inputgate * (hx - newgate)
            _assert_close_tensor(actual, expected, spec.surface, rtol=1e-5, atol=1e-5)
            return

        if spec.surface in {"aten::quantized_rnn_relu_cell", "aten::quantized_rnn_tanh_cell"}:
            w_ih = torch.randn(hidden_size, input_size, dtype=torch.float32) * 0.25
            w_hh = torch.randn(hidden_size, hidden_size, dtype=torch.float32) * 0.25
            b_ih = torch.randn(hidden_size, dtype=torch.float32) * 0.05
            b_hh = torch.randn(hidden_size, dtype=torch.float32) * 0.05
            (
                qweight_ih,
                qweight_hh,
                packed_ih,
                packed_hh,
                col_offsets_ih,
                col_offsets_hh,
                scale_ih,
                scale_hh,
                zero_point_ih,
                zero_point_hh,
            ) = _fbgemm_cell_params(w_ih, w_hh)
            op = (
                torch.ops.aten.quantized_rnn_relu_cell
                if spec.surface == "aten::quantized_rnn_relu_cell"
                else torch.ops.aten.quantized_rnn_tanh_cell
            )
            actual = op(
                input_tensor,
                hx,
                qweight_ih,
                qweight_hh,
                b_ih,
                b_hh,
                packed_ih,
                packed_hh,
                col_offsets_ih,
                col_offsets_hh,
                scale_ih,
                scale_hh,
                zero_point_ih,
                zero_point_hh,
            )
            preactivation = _fbgemm_linear_int8_reference(input_tensor, w_ih, b_ih) + _fbgemm_linear_int8_reference(hx, w_hh, b_hh)
            expected = preactivation.relu() if spec.surface == "aten::quantized_rnn_relu_cell" else preactivation.tanh()
            _assert_close_tensor(actual, expected, spec.surface, rtol=1e-5, atol=1e-5)
            return
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)

    raise AssertionError(f"No FBGEMM quantized-cell oracle implementation for {spec.surface}")


def _mkldnn_rnn_sample():
    import torch.nn as nn

    torch.manual_seed(505)
    module = nn.LSTM(3, 4, 1, batch_first=False)
    input_tensor = torch.randn(3, 2, 3)
    hx = torch.randn(1, 2, 4)
    cx = torch.randn(1, 2, 4)
    params = [
        module.weight_ih_l0.detach().clone(),
        module.weight_hh_l0.detach().clone(),
        module.bias_ih_l0.detach().clone(),
        module.bias_hh_l0.detach().clone(),
    ]
    return module, input_tensor, hx, cx, params


def _run_mkldnn_rnn(spec: OracleSpec, device: str) -> None:
    _require_mkldnn(spec, device)
    module, input_tensor, hx, cx, params = _mkldnn_rnn_sample()
    reverse = False
    batch_sizes: list[int] = []
    mode = 2
    hidden_size = 4
    num_layers = 1
    has_biases = True
    bidirectional = False
    batch_first = False

    def _forward(train: bool):
        return torch.ops.aten.mkldnn_rnn_layer(
            input_tensor,
            params[0],
            params[1],
            params[2],
            params[3],
            hx,
            cx,
            reverse,
            batch_sizes,
            mode,
            hidden_size,
            num_layers,
            has_biases,
            bidirectional,
            batch_first,
            train,
        )

    try:
        if spec.surface == "aten::mkldnn_rnn_layer":
            actual = _forward(False)
            expected_output, (expected_hy, expected_cy) = module(input_tensor, (hx, cx))
            _assert_close_tensor(actual[0], expected_output, f"{spec.surface}.output", rtol=1e-4, atol=1e-4)
            _assert_close_tensor(actual[1], expected_hy, f"{spec.surface}.hy", rtol=1e-4, atol=1e-4)
            _assert_close_tensor(actual[2], expected_cy, f"{spec.surface}.cy", rtol=1e-4, atol=1e-4)
            return

        if spec.surface == "aten::mkldnn_rnn_layer.out":
            expected_output, (expected_hy, expected_cy) = module(input_tensor, (hx, cx))
            out0 = torch.empty_like(expected_output)
            out1 = torch.empty_like(expected_hy)
            out2 = torch.empty_like(expected_cy)
            out3 = torch.empty(0, dtype=torch.uint8)
            actual = torch.ops.aten.mkldnn_rnn_layer.out(
                input_tensor,
                params[0],
                params[1],
                params[2],
                params[3],
                hx,
                cx,
                reverse,
                batch_sizes,
                mode,
                hidden_size,
                num_layers,
                has_biases,
                bidirectional,
                batch_first,
                False,
                out0=out0,
                out1=out1,
                out2=out2,
                out3=out3,
            )
            assert_out_identity(actual[0], out0, f"{spec.surface}.out0")
            assert_out_identity(actual[1], out1, f"{spec.surface}.out1")
            assert_out_identity(actual[2], out2, f"{spec.surface}.out2")
            assert_out_identity(actual[3], out3, f"{spec.surface}.out3")
            _assert_close_tensor(out0, expected_output, f"{spec.surface}.output", rtol=1e-4, atol=1e-4)
            _assert_close_tensor(out1, expected_hy, f"{spec.surface}.hy", rtol=1e-4, atol=1e-4)
            _assert_close_tensor(out2, expected_cy, f"{spec.surface}.cy", rtol=1e-4, atol=1e-4)
            return

        forward = _forward(True)
        grad_output = torch.randn_like(forward[0])
        grad_hy = torch.randn_like(forward[1])
        grad_cy = torch.randn_like(forward[2])
        ref_module, ref_input, ref_hx, ref_cx, ref_params = _mkldnn_rnn_sample()
        ref_input = input_tensor.detach().clone().requires_grad_(True)
        ref_hx = hx.detach().clone().requires_grad_(True)
        ref_cx = cx.detach().clone().requires_grad_(True)
        with torch.no_grad():
            ref_module.weight_ih_l0.copy_(params[0])
            ref_module.weight_hh_l0.copy_(params[1])
            ref_module.bias_ih_l0.copy_(params[2])
            ref_module.bias_hh_l0.copy_(params[3])
        ref_output, (ref_hy, ref_cy) = ref_module(ref_input, (ref_hx, ref_cx))
        torch.autograd.backward((ref_output, ref_hy, ref_cy), (grad_output, grad_hy, grad_cy))

        if spec.surface == "aten::mkldnn_rnn_layer_backward":
            actual = torch.ops.aten.mkldnn_rnn_layer_backward(
                input_tensor,
                params[0],
                params[1],
                params[2],
                params[3],
                hx,
                cx,
                forward[0],
                forward[1],
                forward[2],
                grad_output,
                grad_hy,
                grad_cy,
                reverse,
                mode,
                hidden_size,
                num_layers,
                has_biases,
                True,
                bidirectional,
                batch_sizes,
                batch_first,
                forward[3],
            )
            _assert_close_tensor(actual[0], ref_input.grad, f"{spec.surface}.grad_input", rtol=1e-4, atol=1e-4)
            _assert_close_tensor(actual[1], ref_module.weight_ih_l0.grad, f"{spec.surface}.grad_w_ih", rtol=1e-4, atol=1e-4)
            _assert_close_tensor(actual[2], ref_module.weight_hh_l0.grad, f"{spec.surface}.grad_w_hh", rtol=1e-4, atol=1e-4)
            _assert_close_tensor(actual[3], ref_module.bias_ih_l0.grad, f"{spec.surface}.grad_b_ih", rtol=1e-4, atol=1e-4)
            _assert_close_tensor(actual[4], ref_module.bias_hh_l0.grad, f"{spec.surface}.grad_b_hh", rtol=1e-4, atol=1e-4)
            _assert_close_tensor(actual[5], ref_hx.grad, f"{spec.surface}.grad_hx", rtol=1e-4, atol=1e-4)
            _assert_close_tensor(actual[6], ref_cx.grad, f"{spec.surface}.grad_cx", rtol=1e-4, atol=1e-4)
            return

        if spec.surface == "aten::mkldnn_rnn_layer_backward.out":
            out0 = torch.empty_like(input_tensor)
            out1 = torch.empty_like(params[0])
            out2 = torch.empty_like(params[1])
            out3 = torch.empty_like(params[2])
            out4 = torch.empty_like(params[3])
            out5 = torch.empty_like(hx)
            out6 = torch.empty_like(cx)
            actual = torch.ops.aten.mkldnn_rnn_layer_backward.out(
                input_tensor,
                params[0],
                params[1],
                params[2],
                params[3],
                hx,
                cx,
                forward[0],
                forward[1],
                forward[2],
                grad_output,
                grad_hy,
                grad_cy,
                reverse,
                mode,
                hidden_size,
                num_layers,
                has_biases,
                True,
                bidirectional,
                batch_sizes,
                batch_first,
                forward[3],
                out0=out0,
                out1=out1,
                out2=out2,
                out3=out3,
                out4=out4,
                out5=out5,
                out6=out6,
            )
            for index, out in enumerate((out0, out1, out2, out3, out4, out5, out6)):
                assert_out_identity(actual[index], out, f"{spec.surface}.out{index}")
            _assert_close_tensor(actual[0], ref_input.grad, f"{spec.surface}.grad_input", rtol=1e-4, atol=1e-4)
            _assert_close_tensor(actual[1], ref_module.weight_ih_l0.grad, f"{spec.surface}.grad_w_ih", rtol=1e-4, atol=1e-4)
            _assert_close_tensor(actual[2], ref_module.weight_hh_l0.grad, f"{spec.surface}.grad_w_hh", rtol=1e-4, atol=1e-4)
            _assert_close_tensor(actual[3], ref_module.bias_ih_l0.grad, f"{spec.surface}.grad_b_ih", rtol=1e-4, atol=1e-4)
            _assert_close_tensor(actual[4], ref_module.bias_hh_l0.grad, f"{spec.surface}.grad_b_hh", rtol=1e-4, atol=1e-4)
            _assert_close_tensor(actual[5], ref_hx.grad, f"{spec.surface}.grad_hx", rtol=1e-4, atol=1e-4)
            _assert_close_tensor(actual[6], ref_cx.grad, f"{spec.surface}.grad_cx", rtol=1e-4, atol=1e-4)
            return
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)

    raise AssertionError(f"No MKLDNN RNN oracle implementation for {spec.surface}")


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


def _semi_structured_dense_sample(device_obj: torch.device) -> torch.Tensor:
    rows = 32
    cols = 64
    base = torch.arange(rows * cols, device=device_obj, dtype=torch.float16).reshape(rows, cols)
    base = (base.remainder(17) + 1.0) / 17.0
    mask = torch.tensor([0.0, 0.0, 1.0, 1.0], device=device_obj, dtype=torch.float16).repeat(rows, cols // 4)
    return (base * mask).contiguous()


def _semi_structured_dense_unpruned_sample(device_obj: torch.device) -> torch.Tensor:
    rows = 32
    cols = 64
    base = torch.arange(rows * cols, device=device_obj, dtype=torch.float16).reshape(rows, cols)
    return ((base.remainder(97) - 48.0) / 17.0).contiguous()


def _semi_structured_pair(dense: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    packed, meta = torch.ops.aten._to_sparse_semi_structured(dense)
    if packed.device != dense.device or meta.device != dense.device:
        raise AssertionError("_to_sparse_semi_structured returned tensors on the wrong device")
    if packed.dtype != dense.dtype:
        raise AssertionError("_to_sparse_semi_structured packed dtype mismatch")
    if meta.dtype not in {torch.int16, torch.int32}:
        raise AssertionError(f"_to_sparse_semi_structured metadata dtype mismatch: {meta.dtype}")
    return packed, meta


def _run_cuda_semi_structured_sparse(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    device_obj = torch.device(device)
    dense = _semi_structured_dense_sample(device_obj)
    packed, meta = _semi_structured_pair(dense)
    rhs = torch.linspace(-0.75, 0.75, steps=dense.shape[1] * 8, device=device_obj, dtype=torch.float16).reshape(
        dense.shape[1],
        8,
    )
    input_for_linear = torch.linspace(
        -0.5,
        0.5,
        steps=32 * dense.shape[1],
        device=device_obj,
        dtype=torch.float16,
    ).reshape(32, dense.shape[1])

    try:
        if spec.surface == "aten::_to_sparse_semi_structured":
            actual = torch.ops.aten._sparse_semi_structured_mm(packed, meta, rhs)
            expected = dense @ rhs
            _assert_close_tensor(actual, expected, spec.surface, rtol=2e-2, atol=2e-2)
            return

        if spec.surface == "aten::_sparse_semi_structured_mm":
            actual = torch.ops.aten._sparse_semi_structured_mm(packed, meta, rhs)
            expected = dense @ rhs
            _assert_close_tensor(actual, expected, spec.surface, rtol=2e-2, atol=2e-2)
            return

        if spec.surface == "aten::_sparse_semi_structured_addmm":
            bias = torch.full((dense.shape[0],), 0.25, device=device_obj, dtype=torch.float16)
            actual = torch.ops.aten._sparse_semi_structured_addmm(bias, packed, meta, rhs, alpha=0.5, beta=2.0)
            expected = 2.0 * bias[:, None] + 0.5 * (dense @ rhs)
            _assert_close_tensor(actual, expected, spec.surface, rtol=2e-2, atol=2e-2)
            return

        if spec.surface == "aten::_sparse_semi_structured_linear":
            bias = torch.linspace(-0.2, 0.2, steps=dense.shape[0], device=device_obj, dtype=torch.float16)
            actual = torch.ops.aten._sparse_semi_structured_linear(input_for_linear, packed, meta, bias=bias)
            expected = input_for_linear @ dense.T + bias
            _assert_close_tensor(actual, expected, spec.surface, rtol=2e-2, atol=2e-2)
            return

        if spec.surface in {
            "aten::_sparse_semi_structured_tile",
            "aten::_sparse_semi_structured_apply",
            "aten::_sparse_semi_structured_apply_dense",
        }:
            unpruned = _semi_structured_dense_unpruned_sample(device_obj)
            tile_packed, tile_meta, tile_packed_t, tile_meta_t, thread_masks = torch.ops.aten._sparse_semi_structured_tile(
                unpruned,
                "",
                True,
            )
            if thread_masks.dtype != torch.uint8 or thread_masks.dim() != 3 or thread_masks.numel() == 0:
                raise AssertionError(f"{spec.surface} returned malformed thread masks")
            for label, tensor in (
                ("packed", tile_packed),
                ("meta", tile_meta),
                ("packed_t", tile_packed_t),
                ("meta_t", tile_meta_t),
            ):
                if tensor.device.type != device_obj.type or tensor.numel() == 0:
                    raise AssertionError(f"{spec.surface} returned malformed {label}")

            applied_packed, applied_packed_t = torch.ops.aten._sparse_semi_structured_apply(unpruned, thread_masks)
            _assert_close_tensor(applied_packed, tile_packed, f"{spec.surface}.apply_packed", rtol=0.0, atol=0.0)
            if applied_packed_t.numel() == 0 or applied_packed_t.device.type != device_obj.type:
                raise AssertionError(f"{spec.surface} returned malformed transposed packed tensor")

            if spec.surface == "aten::_sparse_semi_structured_tile":
                return

            if spec.surface == "aten::_sparse_semi_structured_apply":
                return

            pruned = torch.ops.aten._sparse_semi_structured_apply_dense(unpruned, thread_masks)
            if tuple(pruned.shape) != tuple(unpruned.shape) or pruned.dtype != unpruned.dtype:
                raise AssertionError(f"{spec.surface} returned malformed dense tensor")
            if not bool(((pruned == 0) | (pruned == unpruned)).all().item()):
                raise AssertionError(f"{spec.surface} produced values that are neither zero nor original input values")
            per_four = (pruned != 0).reshape(pruned.shape[0], pruned.shape[1] // 4, 4).sum(dim=2)
            if not bool((per_four <= 2).all().item()):
                raise AssertionError(f"{spec.surface} violated the 2:4 per-row sparsity bound")
            packed_pruned, meta_pruned = _semi_structured_pair(pruned)
            rhs_pruned = torch.linspace(
                -0.5,
                0.5,
                steps=pruned.shape[1] * 8,
                device=device_obj,
                dtype=torch.float16,
            ).reshape(pruned.shape[1], 8)
            actual = torch.ops.aten._sparse_semi_structured_mm(packed_pruned, meta_pruned, rhs_pruned)
            expected = pruned @ rhs_pruned
            _assert_close_tensor(actual, expected, spec.surface, rtol=2e-2, atol=2e-2)
            return
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)

    raise AssertionError(f"No CUDA semi-structured sparse oracle implementation for {spec.surface}")


def _cuda_cslt_sample(device_obj: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    dense = torch.zeros((64, 64), device=device_obj, dtype=torch.float16)
    dense[:, 0::2] = torch.linspace(
        -1.0,
        1.0,
        steps=dense.shape[0] * (dense.shape[1] // 2),
        device=device_obj,
        dtype=torch.float16,
    ).reshape(dense.shape[0], dense.shape[1] // 2)
    rhs = torch.linspace(
        -0.75,
        0.75,
        steps=dense.shape[1] * 16,
        device=device_obj,
        dtype=torch.float16,
    ).reshape(dense.shape[1], 16)
    return dense, rhs


def _run_cuda_cslt(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    _require_torch_at_least(
        spec,
        "2.12.1",
        "PyTorch 2.11.0+cu128 on Spark GB10 reports a cuSparseLt architecture mismatch",
    )
    device_obj = torch.device(device)
    dense, rhs = _cuda_cslt_sample(device_obj)
    try:
        compressed = torch.ops.aten._cslt_compress(dense)
        if compressed.device.type != device_obj.type or compressed.numel() == 0:
            raise AssertionError(f"{spec.surface} returned malformed compressed tensor")
        if compressed.dtype != dense.dtype:
            raise AssertionError(f"{spec.surface} compressed dtype mismatch: {compressed.dtype}")

        if spec.surface == "aten::_cslt_compress":
            actual = torch.ops.aten._cslt_sparse_mm(compressed, rhs, None, None, None, False, 0, 1, -1)
            expected = dense @ rhs
            _assert_close_tensor(actual, expected, spec.surface, rtol=2e-2, atol=2e-2)
            return

        if spec.surface == "aten::_cslt_sparse_mm_search":
            actual = torch.ops.aten._cslt_sparse_mm_search(compressed, rhs, None, None, None, False)
            if not isinstance(actual, int) or actual < 0:
                raise AssertionError(f"{spec.surface} returned invalid algorithm id {actual!r}")
            return

        if spec.surface == "aten::_cslt_sparse_mm":
            alg_id = torch.ops.aten._cslt_sparse_mm_search(compressed, rhs, None, None, None, False)
            actual = torch.ops.aten._cslt_sparse_mm(compressed, rhs, None, None, None, False, alg_id, 1, -1)
            expected = dense @ rhs
            _assert_close_tensor(actual, expected, spec.surface, rtol=2e-2, atol=2e-2)
            return
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)

    raise AssertionError(f"No cuSparseLt oracle implementation for {spec.surface}")


def _cuda_attention_sample(device_obj: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (1, 2, 8, 64)
    q = torch.linspace(-0.50, 0.50, steps=1 * 2 * 8 * 64, device=device_obj, dtype=torch.float16).reshape(shape)
    k = torch.linspace(0.45, -0.45, steps=1 * 2 * 8 * 64, device=device_obj, dtype=torch.float16).reshape(shape)
    v = torch.linspace(-0.25, 0.75, steps=1 * 2 * 8 * 64, device=device_obj, dtype=torch.float16).reshape(shape)
    return q, k, v


def _assert_cuda_attention_result(
    result,
    expected: torch.Tensor,
    spec: OracleSpec,
    device_obj: torch.device,
) -> None:
    if not isinstance(result, tuple) or len(result) != 9:
        raise AssertionError(f"{spec.surface} returned malformed cuDNN attention tuple")
    output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_mask = result
    _assert_close_tensor(output, expected, f"{spec.surface}.output", rtol=2e-2, atol=2e-2)
    if logsumexp.shape != expected.shape[:-1] + (1,) or logsumexp.device.type != device_obj.type:
        raise AssertionError(f"{spec.surface} returned malformed logsumexp tensor")
    if cum_seq_q is not None and cum_seq_q.device.type != device_obj.type:
        raise AssertionError(f"{spec.surface} returned cum_seq_q on the wrong device")
    if cum_seq_k is not None and cum_seq_k.device.type != device_obj.type:
        raise AssertionError(f"{spec.surface} returned cum_seq_k on the wrong device")
    if int(max_q) != expected.shape[-2] or int(max_k) != expected.shape[-2]:
        raise AssertionError(f"{spec.surface} returned malformed max sequence lengths")
    for name, value in (("philox_seed", philox_seed), ("philox_offset", philox_offset)):
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise AssertionError(f"{spec.surface} returned malformed {name}")
    if debug_mask is not None and debug_mask.device.type != device_obj.type:
        raise AssertionError(f"{spec.surface} returned debug mask on the wrong device")


def _run_cuda_cudnn_attention(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    _require_torch_at_least(
        spec,
        "2.12.1",
        "PyTorch 2.11.0+cu128 on Spark GB10 segfaulted during direct cuDNN attention probing",
    )
    import torch.nn.functional as F

    device_obj = torch.device(device)
    q, k, v = _cuda_attention_sample(device_obj)
    try:
        expected = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        if spec.surface == "aten::_scaled_dot_product_cudnn_attention":
            actual = torch.ops.aten._scaled_dot_product_cudnn_attention(q, k, v, None, True, 0.0, False, False)
            _assert_cuda_attention_result(actual, expected, spec, device_obj)
            return

        if spec.surface == "aten::_cudnn_attention_forward":
            actual = torch.ops.aten._cudnn_attention_forward(q, k, v, None, None, None, 8, 8, True, 0.0, False, False)
            _assert_cuda_attention_result(actual, expected, spec, device_obj)
            return
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)

    raise AssertionError(f"No cuDNN attention oracle implementation for {spec.surface}")


def _run_cuda_flash_attention_no_dropout_inplace(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    _require_torch_at_least(
        spec,
        "2.12.1",
        "Spark GB10 validation for this direct flash-attention inplace helper uses PyTorch 2.12.1+cu130",
    )
    import torch.nn.functional as F

    device_obj = torch.device(device)
    q = torch.linspace(-0.50, 0.50, steps=1 * 8 * 2 * 64, device=device_obj, dtype=torch.float16).reshape(
        1,
        8,
        2,
        64,
    )
    k = torch.linspace(0.40, -0.40, steps=1 * 8 * 2 * 64, device=device_obj, dtype=torch.float16).reshape(
        1,
        8,
        2,
        64,
    )
    v = torch.linspace(-0.20, 0.70, steps=1 * 8 * 2 * 64, device=device_obj, dtype=torch.float16).reshape(
        1,
        8,
        2,
        64,
    )
    out = torch.empty_like(q)
    try:
        logsumexp = torch.ops.aten._flash_attention_forward_no_dropout_inplace(
            out,
            q,
            k,
            v,
            None,
            None,
            8,
            8,
            0.0,
            False,
            False,
        )
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)

    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2)
    scores = (q.transpose(1, 2).float() @ k.transpose(1, 2).float().transpose(-2, -1)) / math.sqrt(q.shape[-1])
    expected_logsumexp = torch.logsumexp(scores, dim=-1)
    _assert_close_tensor(out, expected, f"{spec.surface}.out", rtol=2e-2, atol=2e-2)
    _assert_close_tensor(logsumexp, expected_logsumexp, f"{spec.surface}.logsumexp", rtol=1e-4, atol=1e-4)


def _run_cuda_fused_rms_norm_backward(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    _require_torch_at_least(
        spec,
        "2.12.1",
        "Spark GB10 validation for this fused RMSNorm backward helper uses PyTorch 2.12.1+cu130",
    )
    device_obj = torch.device(device)
    x = torch.linspace(-1.0, 1.0, steps=12, device=device_obj, dtype=torch.float32).reshape(3, 4)
    weight = torch.linspace(0.25, 1.0, steps=4, device=device_obj, dtype=torch.float32)
    grad_out = torch.linspace(0.5, -0.5, steps=12, device=device_obj, dtype=torch.float32).reshape(3, 4)
    normalized_shape = [4]
    eps = 1e-5

    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    rstd_ref = (x_ref.pow(2).mean(dim=-1, keepdim=True) + eps).rsqrt()
    y_ref = x_ref * rstd_ref * weight_ref
    y_ref.backward(grad_out)
    try:
        actual = torch.ops.aten._fused_rms_norm_backward(
            grad_out,
            x,
            normalized_shape,
            rstd_ref.detach(),
            weight,
            [True, True],
        )
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)
    _assert_close_tensor(actual[0], x_ref.grad, f"{spec.surface}.grad_input", rtol=1e-5, atol=1e-5)
    _assert_close_tensor(actual[1], weight_ref.grad, f"{spec.surface}.grad_weight", rtol=1e-5, atol=1e-5)


def _run_cuda_dtype_out_matmul(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    _require_torch_at_least(
        spec,
        "2.12.1",
        "Spark GB10 validation for dtype-out matmul overloads uses PyTorch 2.12.1+cu130",
    )
    device_obj = torch.device(device)
    try:
        if spec.surface == "aten::mm.dtype_out":
            lhs = torch.linspace(-1.0, 1.0, steps=12, device=device_obj, dtype=torch.float16).reshape(3, 4)
            rhs = torch.linspace(0.75, -0.75, steps=20, device=device_obj, dtype=torch.float16).reshape(4, 5)
            expected = lhs.float() @ rhs.float()
            out = torch.empty_like(expected)
            actual = torch.ops.aten.mm.dtype_out(lhs, rhs, torch.float32, out=out)
            assert_out_identity(actual, out, spec.surface)
            _assert_close_tensor(actual, expected, spec.surface, rtol=1e-3, atol=1e-3)
            return

        if spec.surface == "aten::bmm.dtype_out":
            lhs = torch.linspace(-1.0, 1.0, steps=2 * 3 * 4, device=device_obj, dtype=torch.float16).reshape(
                2,
                3,
                4,
            )
            rhs = torch.linspace(0.75, -0.75, steps=2 * 4 * 5, device=device_obj, dtype=torch.float16).reshape(
                2,
                4,
                5,
            )
            expected = lhs.float() @ rhs.float()
            out = torch.empty_like(expected)
            actual = torch.ops.aten.bmm.dtype_out(lhs, rhs, torch.float32, out=out)
            assert_out_identity(actual, out, spec.surface)
            _assert_close_tensor(actual, expected, spec.surface, rtol=1e-3, atol=1e-3)
            return

        if spec.surface == "aten::addmm.dtype_out":
            bias = torch.linspace(-0.2, 0.2, steps=15, device=device_obj, dtype=torch.float32).reshape(3, 5)
            lhs = torch.linspace(-1.0, 1.0, steps=12, device=device_obj, dtype=torch.float16).reshape(3, 4)
            rhs = torch.linspace(0.75, -0.75, steps=20, device=device_obj, dtype=torch.float16).reshape(4, 5)
            beta = 0.5
            alpha = 1.25
            expected = beta * bias + alpha * (lhs.float() @ rhs.float())
            out = torch.empty_like(expected)
            actual = torch.ops.aten.addmm.dtype_out(bias, lhs, rhs, torch.float32, beta=beta, alpha=alpha, out=out)
            assert_out_identity(actual, out, spec.surface)
            _assert_close_tensor(actual, expected, spec.surface, rtol=1e-3, atol=1e-3)
            return

        if spec.surface == "aten::baddbmm.dtype_out":
            bias = torch.linspace(-0.2, 0.2, steps=2 * 3 * 5, device=device_obj, dtype=torch.float32).reshape(2, 3, 5)
            lhs = torch.linspace(-1.0, 1.0, steps=2 * 3 * 4, device=device_obj, dtype=torch.float16).reshape(
                2,
                3,
                4,
            )
            rhs = torch.linspace(0.75, -0.75, steps=2 * 4 * 5, device=device_obj, dtype=torch.float16).reshape(
                2,
                4,
                5,
            )
            beta = 0.5
            alpha = 1.25
            expected = beta * bias + alpha * (lhs.float() @ rhs.float())
            out = torch.empty_like(expected)
            actual = torch.ops.aten.baddbmm.dtype_out(
                bias,
                lhs,
                rhs,
                torch.float32,
                beta=beta,
                alpha=alpha,
                out=out,
            )
            assert_out_identity(actual, out, spec.surface)
            _assert_close_tensor(actual, expected, spec.surface, rtol=1e-3, atol=1e-3)
            return
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)

    raise AssertionError(f"No dtype-out matmul oracle implementation for {spec.surface}")


def _cuda_batch_norm_stats_sample(device_obj: torch.device):
    eps = 1e-5
    x = torch.linspace(-1.0, 1.0, steps=2 * 3 * 4 * 4, device=device_obj, dtype=torch.float32).reshape(2, 3, 4, 4)
    grad_out = torch.linspace(0.75, -0.75, steps=x.numel(), device=device_obj, dtype=torch.float32).reshape_as(x)
    mean = x.mean(dim=(0, 2, 3))
    var = x.var(dim=(0, 2, 3), unbiased=False)
    invstd = (var + eps).rsqrt()
    weight = torch.linspace(0.5, 1.5, steps=3, device=device_obj, dtype=torch.float32)
    return x, grad_out, mean, invstd, weight, eps


def _run_cuda_batch_norm_internal(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    _require_torch_at_least(
        spec,
        "2.12.1",
        "Spark GB10 validation for CUDA batch-norm internal helpers uses PyTorch 2.12.1+cu130",
    )
    device_obj = torch.device(device)
    x, grad_out, mean, invstd, weight, eps = _cuda_batch_norm_stats_sample(device_obj)
    xmu = x - mean[None, :, None, None]
    count_value = x.numel() // x.shape[1]
    sum_dy_expected = grad_out.sum(dim=(0, 2, 3))
    sum_dy_xmu_expected = (grad_out * xmu).sum(dim=(0, 2, 3))
    grad_weight_expected = (grad_out * xmu * invstd[None, :, None, None]).sum(dim=(0, 2, 3))
    grad_bias_expected = sum_dy_expected
    try:
        if spec.surface == "aten::batch_norm_backward_reduce":
            actual = torch.ops.aten.batch_norm_backward_reduce(grad_out, x, mean, invstd, weight, True, True, True)
            expected = (sum_dy_expected, sum_dy_xmu_expected, grad_weight_expected, grad_bias_expected)
            for index, (actual_tensor, expected_tensor) in enumerate(zip(actual, expected)):
                _assert_close_tensor(actual_tensor, expected_tensor, f"{spec.surface}.out{index}", rtol=1e-5, atol=1e-5)
            return

        if spec.surface == "aten::batch_norm_backward_elemt":
            count = torch.tensor([count_value], device=device_obj, dtype=torch.int32)
            actual = torch.ops.aten.batch_norm_backward_elemt(
                grad_out,
                x,
                mean,
                invstd,
                weight,
                sum_dy_expected,
                sum_dy_xmu_expected,
                count,
            )
            expected = (
                grad_out
                - sum_dy_expected[None, :, None, None] / count_value
                - xmu * invstd[None, :, None, None].pow(2) * sum_dy_xmu_expected[None, :, None, None] / count_value
            ) * invstd[None, :, None, None] * weight[None, :, None, None]
            _assert_close_tensor(actual, expected, spec.surface, rtol=1e-5, atol=1e-5)
            return

        if spec.surface in {"aten::batch_norm_gather_stats", "aten::batch_norm_gather_stats_with_counts"}:
            chunks = torch.chunk(x, 2, dim=0)
            means = []
            invstds = []
            counts = []
            for chunk in chunks:
                means.append(chunk.mean(dim=(0, 2, 3)))
                invstds.append((chunk.var(dim=(0, 2, 3), unbiased=False) + eps).rsqrt())
                counts.append(chunk.numel() // chunk.shape[1])
            mean_all = torch.stack(means)
            invstd_all = torch.stack(invstds)
            running_mean = torch.full((3,), 0.25, device=device_obj, dtype=torch.float32)
            running_var = torch.full((3,), 1.5, device=device_obj, dtype=torch.float32)
            old_running_mean = running_mean.clone()
            old_running_var = running_var.clone()
            momentum = 0.1
            if spec.surface.endswith("with_counts"):
                count_arg = torch.tensor(counts, device=device_obj, dtype=torch.float32)
                actual_mean, actual_invstd = torch.ops.aten.batch_norm_gather_stats_with_counts(
                    x,
                    mean_all,
                    invstd_all,
                    running_mean,
                    running_var,
                    momentum,
                    eps,
                    count_arg,
                )
            else:
                actual_mean, actual_invstd = torch.ops.aten.batch_norm_gather_stats(
                    x,
                    mean_all,
                    invstd_all,
                    running_mean,
                    running_var,
                    momentum,
                    eps,
                    counts[0],
                )
            flat = x.permute(1, 0, 2, 3).reshape(3, -1)
            expected_mean = flat.mean(dim=1)
            expected_var = flat.var(dim=1, unbiased=False)
            expected_invstd = (expected_var + eps).rsqrt()
            expected_running_mean = (1.0 - momentum) * old_running_mean + momentum * expected_mean
            expected_running_var = (1.0 - momentum) * old_running_var + momentum * flat.var(dim=1, unbiased=True)
            _assert_close_tensor(actual_mean, expected_mean, f"{spec.surface}.mean", rtol=1e-5, atol=1e-5)
            _assert_close_tensor(actual_invstd, expected_invstd, f"{spec.surface}.invstd", rtol=1e-5, atol=1e-5)
            _assert_close_tensor(running_mean, expected_running_mean, f"{spec.surface}.running_mean", rtol=1e-5, atol=1e-5)
            _assert_close_tensor(running_var, expected_running_var, f"{spec.surface}.running_var", rtol=1e-5, atol=1e-5)
            return
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)

    raise AssertionError(f"No CUDA batch-norm internal oracle implementation for {spec.surface}")


def _run_cuda_thnn_cell_backward(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    _require_torch_at_least(
        spec,
        "2.12.1",
        "Spark GB10 validation for CUDA fused THNN cell backward helpers uses PyTorch 2.12.1+cu130",
    )
    device_obj = torch.device(device)
    try:
        if spec.surface == "aten::_thnn_fused_gru_cell_backward":
            torch.manual_seed(9301)
            input_gates = torch.randn(2, 12, device=device_obj, dtype=torch.float32, requires_grad=True)
            hidden_gates = torch.randn(2, 12, device=device_obj, dtype=torch.float32, requires_grad=True)
            hx = torch.randn(2, 4, device=device_obj, dtype=torch.float32, requires_grad=True)
            input_bias = torch.randn(12, device=device_obj, dtype=torch.float32, requires_grad=True)
            hidden_bias = torch.randn(12, device=device_obj, dtype=torch.float32, requires_grad=True)
            i_r, i_z, i_n = (input_gates + input_bias).chunk(3, 1)
            h_r, h_z, h_n = (hidden_gates + hidden_bias).chunk(3, 1)
            reset = torch.sigmoid(i_r + h_r)
            update = torch.sigmoid(i_z + h_z)
            new = torch.tanh(i_n + reset * h_n)
            expected_hy = new + update * (hx - new)
            grad_hy = torch.linspace(0.25, -0.25, steps=expected_hy.numel(), device=device_obj).reshape_as(expected_hy)
            expected_hy.backward(grad_hy)
            with torch.no_grad():
                _, workspace = torch.ops.aten._thnn_fused_gru_cell(
                    input_gates.detach(),
                    hidden_gates.detach(),
                    hx.detach(),
                    input_bias.detach(),
                    hidden_bias.detach(),
                )
                actual = torch.ops.aten._thnn_fused_gru_cell_backward(grad_hy, workspace, True)
            expected = (input_gates.grad, hidden_gates.grad, hx.grad, input_bias.grad, hidden_bias.grad)
            for index, (actual_tensor, expected_tensor) in enumerate(zip(actual, expected)):
                _assert_close_tensor(actual_tensor, expected_tensor, f"{spec.surface}.grad{index}", rtol=1e-5, atol=1e-5)
            return

        if spec.surface == "aten::_thnn_fused_lstm_cell_backward_impl":
            torch.manual_seed(9302)
            input_gates = torch.randn(2, 16, device=device_obj, dtype=torch.float32, requires_grad=True)
            hidden_gates = torch.randn(2, 16, device=device_obj, dtype=torch.float32, requires_grad=True)
            cx = torch.randn(2, 4, device=device_obj, dtype=torch.float32, requires_grad=True)
            input_bias = torch.randn(16, device=device_obj, dtype=torch.float32, requires_grad=True)
            hidden_bias = torch.randn(16, device=device_obj, dtype=torch.float32, requires_grad=True)
            gates = input_gates + hidden_gates + input_bias + hidden_bias
            in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)
            in_gate = torch.sigmoid(in_gate)
            forget_gate = torch.sigmoid(forget_gate)
            cell_gate = torch.tanh(cell_gate)
            out_gate = torch.sigmoid(out_gate)
            expected_cy = forget_gate * cx + in_gate * cell_gate
            expected_hy = out_gate * torch.tanh(expected_cy)
            grad_hy = torch.linspace(0.25, -0.25, steps=expected_hy.numel(), device=device_obj).reshape_as(expected_hy)
            grad_cy = torch.linspace(-0.15, 0.15, steps=expected_cy.numel(), device=device_obj).reshape_as(expected_cy)
            torch.autograd.backward((expected_hy, expected_cy), (grad_hy, grad_cy))
            with torch.no_grad():
                _, actual_cy, workspace = torch.ops.aten._thnn_fused_lstm_cell(
                    input_gates.detach(),
                    hidden_gates.detach(),
                    cx.detach(),
                    input_bias.detach(),
                    hidden_bias.detach(),
                )
                actual = torch.ops.aten._thnn_fused_lstm_cell_backward_impl(
                    grad_hy,
                    grad_cy,
                    cx.detach(),
                    actual_cy,
                    workspace,
                    True,
                )
            expected = (input_gates.grad, cx.grad, input_bias.grad)
            for index, (actual_tensor, expected_tensor) in enumerate(zip(actual, expected)):
                _assert_close_tensor(actual_tensor, expected_tensor, f"{spec.surface}.grad{index}", rtol=1e-5, atol=1e-5)
            return
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)

    raise AssertionError(f"No CUDA THNN cell backward oracle implementation for {spec.surface}")


def _run_cuda_mixed_dtypes_linear(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    _require_torch_at_least(
        spec,
        "2.12.1",
        "Spark GB10 validation for mixed-dtypes linear uses PyTorch 2.12.1+cu130",
    )
    try:
        from torch.quantization._quantized_conversions import quantized_weight_reorder_for_mixed_dtypes_linear_cutlass
    except Exception as exc:
        raise OracleUnavailable(f"backend_not_available: {spec.surface}: mixed-dtypes weight reorder helper unavailable") from exc

    device_obj = torch.device(device)
    logical_weight = torch.arange(-32, 32, device=device_obj, dtype=torch.int8).repeat(64, 1)
    reordered_weight = quantized_weight_reorder_for_mixed_dtypes_linear_cutlass(logical_weight, torch.int8)
    x = torch.linspace(-0.5, 0.5, steps=2 * 64, device=device_obj, dtype=torch.float16).reshape(2, 64)
    scale = torch.linspace(0.01, 0.02, steps=64, device=device_obj, dtype=torch.float16)
    bias = torch.linspace(-0.1, 0.1, steps=64, device=device_obj, dtype=torch.float16)
    expected = x.float() @ logical_weight.float().T * scale.float() + bias.float()
    try:
        actual = torch.ops.aten._mixed_dtypes_linear(x, reordered_weight, scale, bias=bias, activation=None)
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)
    _assert_close_tensor(actual.float(), expected, spec.surface, rtol=2e-2, atol=2e-2)


def _run_cuda_scaled_grouped_mm(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    _require_torch_at_least(
        spec,
        "2.12.1",
        "Spark GB10 validation for scaled grouped matmul uses PyTorch 2.12.1+cu130",
    )
    device_obj = torch.device(device)
    a_float = torch.linspace(-1.0, 1.0, steps=2 * 16 * 32, device=device_obj, dtype=torch.float16).reshape(2, 16, 32)
    b_float = torch.linspace(0.75, -0.75, steps=2 * 32 * 16, device=device_obj, dtype=torch.float16).reshape(
        2,
        16,
        32,
    )
    mat_a = a_float.to(torch.float8_e4m3fn)
    mat_b = b_float.to(torch.float8_e4m3fn).transpose(-1, -2)
    scale_a = torch.ones((2, 16), device=device_obj, dtype=torch.float32)
    scale_b = torch.ones((2, 16), device=device_obj, dtype=torch.float32)
    expected = torch.bmm(mat_a.float() * scale_a[:, :, None], mat_b.float() * scale_b[:, None, :]).to(torch.bfloat16)
    try:
        if spec.surface == "aten::_scaled_grouped_mm":
            actual = torch.ops.aten._scaled_grouped_mm(mat_a, mat_b, scale_a, scale_b, None, None, None, torch.bfloat16, False)
        elif spec.surface == "aten::_scaled_grouped_mm_v2":
            actual = torch.ops.aten._scaled_grouped_mm_v2(
                mat_a,
                mat_b,
                [scale_a],
                [1],
                [],
                [scale_b],
                [1],
                [],
                None,
                None,
                torch.bfloat16,
                [],
                False,
            )
        else:
            raise AssertionError(f"No scaled grouped matmul oracle implementation for {spec.surface}")
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)
    _assert_close_tensor(actual, expected, spec.surface, rtol=2e-2, atol=2e-2)


def _cuda_conv_sample(device_obj: torch.device):
    x = torch.linspace(-1.0, 1.0, steps=1 * 2 * 5 * 5, device=device_obj, dtype=torch.float32).reshape(1, 2, 5, 5)
    weight = torch.linspace(-0.75, 0.75, steps=3 * 2 * 3 * 3, device=device_obj, dtype=torch.float32).reshape(3, 2, 3, 3)
    bias = torch.linspace(-0.25, 0.25, steps=3, device=device_obj, dtype=torch.float32)
    return x, weight, bias, [1, 1], [1, 1], [1, 1], 1


def _run_cuda_cudnn_convolution(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    import torch.nn.functional as F

    device_obj = torch.device(device)
    x, weight, bias, padding, stride, dilation, groups = _cuda_conv_sample(device_obj)
    try:
        if spec.surface in {"aten::cudnn_convolution", "aten::cudnn_convolution.out"}:
            expected = F.conv2d(x, weight, None, stride=stride, padding=padding, dilation=dilation, groups=groups)
            if spec.surface.endswith(".out"):
                out = torch.empty_like(expected)
                actual = torch.ops.aten.cudnn_convolution.out(
                    x, weight, padding, stride, dilation, groups, False, True, False, out=out
                )
                assert_out_identity(actual, out, spec.surface)
            else:
                actual = torch.ops.aten.cudnn_convolution(x, weight, padding, stride, dilation, groups, False, True, False)
            _assert_close_tensor(actual, expected, spec.surface, rtol=1e-4, atol=1e-4)
            return

        if spec.surface in {"aten::cudnn_convolution_relu", "aten::cudnn_convolution_relu.out"}:
            expected = torch.relu(F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups))
            if spec.surface.endswith(".out"):
                out = torch.empty_like(expected)
                actual = torch.ops.aten.cudnn_convolution_relu.out(
                    x, weight, bias, stride, padding, dilation, groups, out=out
                )
                assert_out_identity(actual, out, spec.surface)
            else:
                actual = torch.ops.aten.cudnn_convolution_relu(x, weight, bias, stride, padding, dilation, groups)
            _assert_close_tensor(actual, expected, spec.surface, rtol=1e-4, atol=1e-4)
            return

        if spec.surface in {"aten::cudnn_convolution_add_relu", "aten::cudnn_convolution_add_relu.out"}:
            conv = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            z = torch.linspace(-0.2, 0.2, steps=conv.numel(), device=device_obj, dtype=torch.float32).reshape_as(conv)
            alpha = 0.75
            expected = torch.relu(conv + alpha * z)
            if spec.surface.endswith(".out"):
                out = torch.empty_like(expected)
                actual = torch.ops.aten.cudnn_convolution_add_relu.out(
                    x, weight, z, alpha, bias, stride, padding, dilation, groups, out=out
                )
                assert_out_identity(actual, out, spec.surface)
            else:
                actual = torch.ops.aten.cudnn_convolution_add_relu(x, weight, z, alpha, bias, stride, padding, dilation, groups)
            _assert_close_tensor(actual, expected, spec.surface, rtol=1e-4, atol=1e-4)
            return

        if spec.surface in {"aten::cudnn_convolution_transpose", "aten::cudnn_convolution_transpose.out"}:
            transposed_weight = torch.linspace(
                -0.5,
                0.5,
                steps=2 * 3 * 3 * 3,
                device=device_obj,
                dtype=torch.float32,
            ).reshape(2, 3, 3, 3)
            output_padding = [0, 0]
            expected = F.conv_transpose2d(
                x,
                transposed_weight,
                None,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                groups=groups,
                dilation=dilation,
            )
            if spec.surface.endswith(".out"):
                out = torch.empty_like(expected)
                actual = torch.ops.aten.cudnn_convolution_transpose.out(
                    x,
                    transposed_weight,
                    padding,
                    output_padding,
                    stride,
                    dilation,
                    groups,
                    False,
                    True,
                    False,
                    out=out,
                )
                assert_out_identity(actual, out, spec.surface)
            else:
                actual = torch.ops.aten.cudnn_convolution_transpose(
                    x,
                    transposed_weight,
                    padding,
                    output_padding,
                    stride,
                    dilation,
                    groups,
                    False,
                    True,
                    False,
                )
            _assert_close_tensor(actual, expected, spec.surface, rtol=1e-4, atol=1e-4)
            return
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)

    raise AssertionError(f"No cuDNN convolution oracle implementation for {spec.surface}")


def _cuda_grid_sample(device_obj: torch.device):
    theta = torch.tensor([[[0.8, 0.0, 0.1], [0.0, 0.7, -0.2]]], device=device_obj, dtype=torch.float32)
    size = (1, 1, 4, 5)
    x = torch.linspace(-1.0, 1.0, steps=size[0] * size[1] * size[2] * size[3], device=device_obj).reshape(size)
    grid = torch.nn.functional.affine_grid(theta, size, align_corners=True)
    return theta, size, x, grid


def _run_cuda_cudnn_grid(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    import torch.nn.functional as F

    device_obj = torch.device(device)
    theta, size, x, grid = _cuda_grid_sample(device_obj)
    n, c, h, w = size
    try:
        if spec.surface in {"aten::cudnn_affine_grid_generator", "aten::cudnn_affine_grid_generator.out"}:
            expected = F.affine_grid(theta, size, align_corners=True)
            if spec.surface.endswith(".out"):
                out = torch.empty_like(expected)
                actual = torch.ops.aten.cudnn_affine_grid_generator.out(theta, n, c, h, w, out=out)
                assert_out_identity(actual, out, spec.surface)
            else:
                actual = torch.ops.aten.cudnn_affine_grid_generator(theta, n, c, h, w)
            _assert_close_tensor(actual, expected, spec.surface, rtol=1e-5, atol=1e-5)
            return

        if spec.surface in {"aten::cudnn_affine_grid_generator_backward", "aten::cudnn_affine_grid_generator_backward.out"}:
            theta_ref = theta.detach().clone().requires_grad_(True)
            ref_grid = F.affine_grid(theta_ref, size, align_corners=True)
            grad = torch.linspace(-0.5, 0.5, steps=ref_grid.numel(), device=device_obj).reshape_as(ref_grid)
            ref_grid.backward(grad)
            expected = theta_ref.grad
            if spec.surface.endswith(".out"):
                out = torch.empty_like(expected)
                actual = torch.ops.aten.cudnn_affine_grid_generator_backward.out(grad, n, c, h, w, out=out)
                assert_out_identity(actual, out, spec.surface)
            else:
                actual = torch.ops.aten.cudnn_affine_grid_generator_backward(grad, n, c, h, w)
            _assert_close_tensor(actual, expected, spec.surface, rtol=1e-5, atol=1e-5)
            return

        if spec.surface in {"aten::cudnn_grid_sampler", "aten::cudnn_grid_sampler.out"}:
            expected = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            if spec.surface.endswith(".out"):
                out = torch.empty_like(expected)
                actual = torch.ops.aten.cudnn_grid_sampler.out(x, grid, out=out)
                assert_out_identity(actual, out, spec.surface)
            else:
                actual = torch.ops.aten.cudnn_grid_sampler(x, grid)
            _assert_close_tensor(actual, expected, spec.surface, rtol=1e-5, atol=1e-5)
            return

        if spec.surface in {"aten::cudnn_grid_sampler_backward", "aten::cudnn_grid_sampler_backward.out"}:
            x_ref = x.detach().clone().requires_grad_(True)
            grid_ref = grid.detach().clone().requires_grad_(True)
            ref = F.grid_sample(x_ref, grid_ref, mode="bilinear", padding_mode="zeros", align_corners=True)
            grad_output = torch.linspace(-0.5, 0.5, steps=ref.numel(), device=device_obj).reshape_as(ref)
            ref.backward(grad_output)
            expected = (x_ref.grad, grid_ref.grad)
            if spec.surface.endswith(".out"):
                out0 = torch.empty_like(expected[0])
                out1 = torch.empty_like(expected[1])
                actual = torch.ops.aten.cudnn_grid_sampler_backward.out(x, grid, grad_output, out0=out0, out1=out1)
                assert_out_identity(actual[0], out0, f"{spec.surface}.out0")
                assert_out_identity(actual[1], out1, f"{spec.surface}.out1")
            else:
                actual = torch.ops.aten.cudnn_grid_sampler_backward(x, grid, grad_output)
            _assert_close_tensor(actual[0], expected[0], f"{spec.surface}.grad_input", rtol=1e-5, atol=1e-5)
            _assert_close_tensor(actual[1], expected[1], f"{spec.surface}.grad_grid", rtol=1e-5, atol=1e-5)
            return
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)

    raise AssertionError(f"No cuDNN grid oracle implementation for {spec.surface}")


def _run_cuda_cudnn_batch_norm(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    import torch.nn.functional as F

    device_obj = torch.device(device)
    torch.manual_seed(7001)
    eps = 1e-5
    momentum = 0.1
    x = torch.randn(2, 3, 4, 4, device=device_obj, dtype=torch.float32)
    weight = torch.randn(3, device=device_obj, dtype=torch.float32)
    bias = torch.randn(3, device=device_obj, dtype=torch.float32)
    running_mean = torch.zeros(3, device=device_obj, dtype=torch.float32)
    running_var = torch.ones(3, device=device_obj, dtype=torch.float32)
    try:
        forward = torch.ops.aten.cudnn_batch_norm(
            x,
            weight,
            bias,
            running_mean.clone(),
            running_var.clone(),
            True,
            momentum,
            eps,
        )
        expected = F.batch_norm(
            x,
            running_mean.clone(),
            running_var.clone(),
            weight,
            bias,
            training=True,
            momentum=momentum,
            eps=eps,
        )
        if spec.surface in {"aten::cudnn_batch_norm", "aten::cudnn_batch_norm.out"}:
            if spec.surface.endswith(".out"):
                _require_torch_at_least(
                    spec,
                    "2.12.1",
                    "PyTorch 2.11.0+cu128 on Spark GB10 hits a pyobject_preservation internal assert",
                )
                outs = (
                    torch.empty_like(forward[0]),
                    torch.empty_like(forward[1]),
                    torch.empty_like(forward[2]),
                    torch.empty_like(forward[3]),
                )
                actual = torch.ops.aten.cudnn_batch_norm.out(
                    x,
                    weight,
                    bias,
                    running_mean.clone(),
                    running_var.clone(),
                    True,
                    momentum,
                    eps,
                    out0=outs[0],
                    out1=outs[1],
                    out2=outs[2],
                    out3=outs[3],
                )
                for index, out in enumerate(outs):
                    if index == 3 and actual[index].numel() == 0 and out.numel() == 0:
                        if (
                            actual[index].shape != out.shape
                            or actual[index].dtype != out.dtype
                            or actual[index].device.type != out.device.type
                        ):
                            raise AssertionError(f"{spec.surface}.out{index} returned malformed empty reserve tensor")
                    else:
                        assert_out_identity(actual[index], out, f"{spec.surface}.out{index}")
            else:
                actual = forward
            _assert_close_tensor(actual[0], expected, f"{spec.surface}.output", rtol=1e-4, atol=1e-4)
            if actual[1].shape != (3,) or actual[2].shape != (3,):
                raise AssertionError(f"{spec.surface} returned malformed cuDNN batch-norm save tensors")
            if actual[3].device.type != device_obj.type or actual[3].dtype != torch.uint8:
                raise AssertionError(f"{spec.surface} returned malformed cuDNN batch-norm reserve tensor")
            return

        if spec.surface in {"aten::cudnn_batch_norm_backward", "aten::cudnn_batch_norm_backward.out"}:
            grad_output = torch.randn_like(forward[0])
            x_ref = x.detach().clone().requires_grad_(True)
            weight_ref = weight.detach().clone().requires_grad_(True)
            bias_ref = bias.detach().clone().requires_grad_(True)
            ref = F.batch_norm(
                x_ref,
                running_mean.clone(),
                running_var.clone(),
                weight_ref,
                bias_ref,
                training=True,
                momentum=momentum,
                eps=eps,
            )
            ref.backward(grad_output)
            expected_grads = (x_ref.grad, weight_ref.grad, bias_ref.grad)
            if spec.surface.endswith(".out"):
                outs = tuple(torch.empty_like(item) for item in expected_grads)
                actual = torch.ops.aten.cudnn_batch_norm_backward.out(
                    x,
                    grad_output,
                    weight,
                    running_mean,
                    running_var,
                    forward[1],
                    forward[2],
                    eps,
                    forward[3],
                    out0=outs[0],
                    out1=outs[1],
                    out2=outs[2],
                )
                for index, out in enumerate(outs):
                    assert_out_identity(actual[index], out, f"{spec.surface}.out{index}")
            else:
                actual = torch.ops.aten.cudnn_batch_norm_backward(
                    x,
                    grad_output,
                    weight,
                    running_mean,
                    running_var,
                    forward[1],
                    forward[2],
                    eps,
                    forward[3],
                )
            for index, (actual_grad, expected_grad) in enumerate(zip(actual, expected_grads)):
                _assert_close_tensor(actual_grad, expected_grad, f"{spec.surface}.grad{index}", rtol=1e-4, atol=1e-4)
            return
    except Exception as exc:
        if spec.surface == "aten::cudnn_batch_norm.out" and "pyobject_preservation" in str(exc):
            raise OracleUnavailable(
                f"backend_runtime_bug: {spec.surface} hits a PyTorch pyobject_preservation internal assert"
            ) from exc
        _raise_backend_unavailable_if_applicable(spec, exc)

    raise AssertionError(f"No cuDNN batch-norm oracle implementation for {spec.surface}")


def _cuda_ctc_sample(device_obj: torch.device):
    logits = torch.tensor(
        [
            [[2.5, 0.1, -1.0]],
            [[0.2, 2.0, -0.5]],
            [[0.1, -0.2, 2.2]],
            [[2.0, -0.3, 0.4]],
        ],
        device=device_obj,
        dtype=torch.float32,
    )
    log_probs = logits.log_softmax(2)
    targets = torch.tensor([1, 2], device=device_obj, dtype=torch.int32)
    input_lengths = [4]
    target_lengths = [2]
    return log_probs, targets, input_lengths, target_lengths, 0


def _run_cuda_cudnn_ctc(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    import torch.nn.functional as F

    device_obj = torch.device(device)
    log_probs, targets, input_lengths, target_lengths, blank = _cuda_ctc_sample(device_obj)
    cpu_targets = targets.cpu()
    tensor_input_lengths = torch.tensor(input_lengths, device=device_obj, dtype=torch.int32)
    tensor_target_lengths = torch.tensor(target_lengths, device=device_obj, dtype=torch.int32)
    try:
        if spec.surface in {"aten::_use_cudnn_ctc_loss", "aten::_use_cudnn_ctc_loss.Tensor"}:
            if spec.surface.endswith(".Tensor"):
                actual = torch.ops.aten._use_cudnn_ctc_loss.Tensor(
                    log_probs,
                    targets,
                    tensor_input_lengths,
                    tensor_target_lengths,
                    blank,
                )
            else:
                actual = torch.ops.aten._use_cudnn_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank)
            if not isinstance(actual, bool):
                raise AssertionError(f"{spec.surface} returned non-bool {type(actual).__name__}")
            return

        expected = F.ctc_loss(
            log_probs,
            targets.to(torch.long),
            torch.tensor(input_lengths, device=device_obj, dtype=torch.long),
            torch.tensor(target_lengths, device=device_obj, dtype=torch.long),
            blank=blank,
            reduction="none",
            zero_infinity=False,
        )
        if spec.surface == "aten::_cudnn_ctc_loss.Tensor":
            actual = torch.ops.aten._cudnn_ctc_loss.Tensor(
                log_probs,
                targets,
                tensor_input_lengths,
                tensor_target_lengths,
                blank,
                True,
                False,
            )
        elif spec.surface == "aten::_cudnn_ctc_loss.out":
            probe = torch.ops.aten._cudnn_ctc_loss(log_probs, cpu_targets, input_lengths, target_lengths, blank, True, False)
            out0 = torch.empty_like(probe[0])
            out1 = torch.empty_like(probe[1])
            actual = torch.ops.aten._cudnn_ctc_loss.out(
                log_probs,
                cpu_targets,
                input_lengths,
                target_lengths,
                blank,
                True,
                False,
                out0=out0,
                out1=out1,
            )
            assert_out_identity(actual[0], out0, f"{spec.surface}.out0")
            assert_out_identity(actual[1], out1, f"{spec.surface}.out1")
        elif spec.surface == "aten::_cudnn_ctc_loss":
            actual = torch.ops.aten._cudnn_ctc_loss(log_probs, cpu_targets, input_lengths, target_lengths, blank, True, False)
        else:
            raise AssertionError(f"No cuDNN CTC oracle implementation for {spec.surface}")
        _assert_close_tensor(actual[0], expected, f"{spec.surface}.loss", rtol=1e-4, atol=1e-4)
        if actual[1].numel() == 0 or actual[1].device.type != device_obj.type:
            raise AssertionError(f"{spec.surface} returned malformed cuDNN CTC workspace tensor")
        return
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)


def _run_cuda_cudnn_dropout_state(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    device_obj = torch.device(device)
    try:
        if spec.surface == "aten::_cudnn_init_dropout_state":
            actual = torch.ops.aten._cudnn_init_dropout_state(
                0.25,
                True,
                1729,
                dtype=torch.uint8,
                device=device_obj,
                pin_memory=False,
            )
            if actual.device.type != device_obj.type or actual.dtype != torch.uint8 or actual.numel() == 0:
                raise AssertionError(f"{spec.surface} returned malformed dropout state")
            return
        if spec.surface == "aten::_cudnn_init_dropout_state.out":
            out = torch.empty(0, device=device_obj, dtype=torch.uint8)
            actual = torch.ops.aten._cudnn_init_dropout_state.out(0.25, True, 1729, out=out)
            assert_out_identity(actual, out, spec.surface)
            if actual.device.type != device_obj.type or actual.dtype != torch.uint8 or actual.numel() == 0:
                raise AssertionError(f"{spec.surface} returned malformed dropout state")
            return
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)
    raise AssertionError(f"No cuDNN dropout-state oracle implementation for {spec.surface}")


def _run_cuda_cudnn_is_acceptable(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    device_obj = torch.device(device)
    sample = torch.ones(2, 3, device=device_obj, dtype=torch.float32)
    try:
        actual = torch.ops.aten.cudnn_is_acceptable(sample)
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)
    expected = torch.backends.cudnn.is_acceptable(sample)
    if actual is not expected:
        raise AssertionError(f"{spec.surface} returned {actual}, expected {expected}")


def _run_cuda_triton_attention(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    import torch.nn.functional as F

    device_obj = torch.device(device)
    q = torch.randn(1, 2, 8, 32, device=device_obj, dtype=torch.float16)
    k = torch.randn(1, 2, 8, 32, device=device_obj, dtype=torch.float16)
    v = torch.randn(1, 2, 8, 32, device=device_obj, dtype=torch.float16)
    try:
        if spec.surface in {"aten::_triton_scaled_dot_attention", "aten::_triton_scaled_dot_attention.out"}:
            expected = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            if spec.surface.endswith(".out"):
                out = torch.empty_like(expected)
                actual = torch.ops.aten._triton_scaled_dot_attention.out(q, k, v, 0.0, out=out)
                assert_out_identity(actual, out, spec.surface)
            else:
                actual = torch.ops.aten._triton_scaled_dot_attention(q, k, v, 0.0)
            _assert_close_tensor(actual, expected, spec.surface, rtol=2e-2, atol=2e-2)
            return
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)
    raise AssertionError(f"No Triton attention oracle implementation for {spec.surface}")


def _cuda_lstm_sample(device_obj: torch.device):
    torch.manual_seed(8101)
    module = torch.nn.LSTM(3, 4, 1, batch_first=False).to(device_obj)
    module.train()
    module.flatten_parameters()
    x = torch.randn(5, 2, 3, device=device_obj)
    hx = torch.zeros(1, 2, 4, device=device_obj)
    cx = torch.zeros(1, 2, 4, device=device_obj)
    weights = [weight for weight in module._flat_weights]
    return module, x, hx, cx, weights


def _run_cuda_cudnn_rnn(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    device_obj = torch.device(device)
    module, x, hx, cx, weights = _cuda_lstm_sample(device_obj)
    weight_stride0 = 4
    mode = 2
    hidden_size = 4
    proj_size = 0
    num_layers = 1
    batch_first = False
    dropout = 0.0
    train = True
    bidirectional = False
    batch_sizes: list[int] = []
    dropout_state = None
    try:
        if spec.surface in {"aten::_cudnn_rnn_flatten_weight", "aten::_cudnn_rnn_flatten_weight.out"}:
            expected = torch.ops.aten._cudnn_rnn_flatten_weight(
                weights,
                weight_stride0,
                3,
                mode,
                hidden_size,
                proj_size,
                num_layers,
                batch_first,
                bidirectional,
            )
            if expected.numel() == 0 or expected.device.type != device_obj.type:
                raise AssertionError(f"{spec.surface} returned malformed flattened weight buffer")
            if spec.surface.endswith(".out"):
                out = torch.empty(0, device=device_obj, dtype=expected.dtype)
                actual = torch.ops.aten._cudnn_rnn_flatten_weight.out(
                    weights,
                    weight_stride0,
                    3,
                    mode,
                    hidden_size,
                    proj_size,
                    num_layers,
                    batch_first,
                    bidirectional,
                    out=out,
                )
                assert_out_identity(actual, out, spec.surface)
            else:
                actual = expected
            _assert_close_tensor(actual, expected, spec.surface, rtol=0, atol=0)
            return

        weight_buf = torch.ops.aten._cudnn_rnn_flatten_weight(
            weights,
            weight_stride0,
            3,
            mode,
            hidden_size,
            proj_size,
            num_layers,
            batch_first,
            bidirectional,
        ).detach()
        forward = torch.ops.aten._cudnn_rnn(
            x,
            weights,
            weight_stride0,
            weight_buf,
            hx,
            cx,
            mode,
            hidden_size,
            proj_size,
            num_layers,
            batch_first,
            dropout,
            train,
            bidirectional,
            batch_sizes,
            dropout_state,
        )
        public_out, (public_hy, public_cy) = module(x, (hx, cx))
        if spec.surface in {"aten::_cudnn_rnn", "aten::_cudnn_rnn.out"}:
            if spec.surface.endswith(".out"):
                outs = tuple(torch.empty_like(item) for item in forward)
                actual = torch.ops.aten._cudnn_rnn.out(
                    x,
                    weights,
                    weight_stride0,
                    weight_buf,
                    hx,
                    cx,
                    mode,
                    hidden_size,
                    proj_size,
                    num_layers,
                    batch_first,
                    dropout,
                    train,
                    bidirectional,
                    batch_sizes,
                    dropout_state,
                    out0=outs[0],
                    out1=outs[1],
                    out2=outs[2],
                    out3=outs[3],
                    out4=outs[4],
                )
                for index, out in enumerate(outs):
                    assert_out_identity(actual[index], out, f"{spec.surface}.out{index}")
            else:
                actual = forward
            _assert_close_tensor(actual[0], public_out, f"{spec.surface}.output", rtol=1e-5, atol=1e-5)
            _assert_close_tensor(actual[1], public_hy, f"{spec.surface}.hy", rtol=1e-5, atol=1e-5)
            _assert_close_tensor(actual[2], public_cy, f"{spec.surface}.cy", rtol=1e-5, atol=1e-5)
            if actual[3].device.type != device_obj.type or actual[4].device.type != device_obj.type:
                raise AssertionError(f"{spec.surface} returned malformed reserve or weight buffer tensors")
            return

        if spec.surface in {"aten::_cudnn_rnn_backward", "aten::_cudnn_rnn_backward.out"}:
            grad_output = torch.randn_like(forward[0])
            grad_hy = torch.randn_like(forward[1])
            grad_cy = torch.randn_like(forward[2])
            x_ref = x.detach().clone().requires_grad_(True)
            hx_ref = hx.detach().clone().requires_grad_(True)
            cx_ref = cx.detach().clone().requires_grad_(True)
            module.zero_grad(set_to_none=True)
            ref_out, (ref_hy, ref_cy) = module(x_ref, (hx_ref, cx_ref))
            torch.autograd.backward((ref_out, ref_hy, ref_cy), (grad_output, grad_hy, grad_cy))
            expected_weight_grads = [param.grad for param in module.parameters()]
            if spec.surface.endswith(".out"):
                out0 = torch.empty_like(x)
                out1 = torch.empty_like(hx)
                out2 = torch.empty_like(cx)
                out3 = [torch.empty_like(weight) for weight in weights]
                actual = torch.ops.aten._cudnn_rnn_backward.out(
                    x,
                    weights,
                    weight_stride0,
                    weight_buf,
                    hx,
                    cx,
                    forward[0],
                    grad_output,
                    grad_hy,
                    grad_cy,
                    mode,
                    hidden_size,
                    proj_size,
                    num_layers,
                    batch_first,
                    dropout,
                    train,
                    bidirectional,
                    batch_sizes,
                    dropout_state,
                    forward[3],
                    [True, True, True, True],
                    out0=out0,
                    out1=out1,
                    out2=out2,
                    out3=out3,
                )
                if actual is not None:
                    raise AssertionError(f"{spec.surface} should return None")
                actual = (out0, out1, out2, out3)
            else:
                actual = torch.ops.aten._cudnn_rnn_backward(
                    x,
                    weights,
                    weight_stride0,
                    weight_buf,
                    hx,
                    cx,
                    forward[0],
                    grad_output,
                    grad_hy,
                    grad_cy,
                    mode,
                    hidden_size,
                    proj_size,
                    num_layers,
                    batch_first,
                    dropout,
                    train,
                    bidirectional,
                    batch_sizes,
                    dropout_state,
                    forward[3],
                    [True, True, True, True],
                )
            _assert_close_tensor(actual[0], x_ref.grad, f"{spec.surface}.grad_input", rtol=1e-5, atol=1e-5)
            _assert_close_tensor(actual[1], hx_ref.grad, f"{spec.surface}.grad_hx", rtol=1e-5, atol=1e-5)
            _assert_close_tensor(actual[2], cx_ref.grad, f"{spec.surface}.grad_cx", rtol=1e-5, atol=1e-5)
            for index, (actual_grad, expected_grad) in enumerate(zip(actual[3], expected_weight_grads)):
                _assert_close_tensor(actual_grad, expected_grad, f"{spec.surface}.grad_weight{index}", rtol=1e-5, atol=1e-5)
            return
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)

    raise AssertionError(f"No cuDNN RNN oracle implementation for {spec.surface}")


def _run_rocm_miopen_convolution(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    import torch.nn.functional as F

    device_obj = torch.device(device)
    x, weight, bias, padding, stride, dilation, groups = _cuda_conv_sample(device_obj)
    try:
        if spec.surface in {"aten::miopen_convolution", "aten::miopen_convolution.out"}:
            expected = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            if spec.surface.endswith(".out"):
                out = torch.empty_like(expected)
                actual = torch.ops.aten.miopen_convolution.out(
                    x, weight, bias, padding, stride, dilation, groups, False, True, out=out
                )
                assert_out_identity(actual, out, spec.surface)
            else:
                actual = torch.ops.aten.miopen_convolution(x, weight, bias, padding, stride, dilation, groups, False, True)
            _assert_close_tensor(actual, expected, spec.surface, rtol=1e-4, atol=1e-4)
            return

        if spec.surface == "aten::miopen_convolution_relu":
            expected = torch.relu(F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups))
            actual = torch.ops.aten.miopen_convolution_relu(x, weight, bias, stride, padding, dilation, groups)
            _assert_close_tensor(actual, expected, spec.surface, rtol=1e-4, atol=1e-4)
            return

        if spec.surface == "aten::miopen_convolution_add_relu":
            conv = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            z = torch.linspace(-0.2, 0.2, steps=conv.numel(), device=device_obj, dtype=torch.float32).reshape_as(conv)
            alpha = 0.75
            actual = torch.ops.aten.miopen_convolution_add_relu(x, weight, z, alpha, bias, stride, padding, dilation, groups)
            expected = torch.relu(conv + alpha * z)
            _assert_close_tensor(actual, expected, spec.surface, rtol=1e-4, atol=1e-4)
            return

        if spec.surface in {"aten::miopen_convolution_transpose", "aten::miopen_convolution_transpose.out"}:
            transposed_weight = torch.linspace(
                -0.5,
                0.5,
                steps=2 * 3 * 3 * 3,
                device=device_obj,
                dtype=torch.float32,
            ).reshape(2, 3, 3, 3)
            output_padding = [0, 0]
            expected = F.conv_transpose2d(
                x,
                transposed_weight,
                bias=None,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                groups=groups,
                dilation=dilation,
            )
            if spec.surface.endswith(".out"):
                out = torch.empty_like(expected)
                actual = torch.ops.aten.miopen_convolution_transpose.out(
                    x,
                    transposed_weight,
                    None,
                    padding,
                    output_padding,
                    stride,
                    dilation,
                    groups,
                    False,
                    True,
                    out=out,
                )
                assert_out_identity(actual, out, spec.surface)
            else:
                actual = torch.ops.aten.miopen_convolution_transpose(
                    x,
                    transposed_weight,
                    None,
                    padding,
                    output_padding,
                    stride,
                    dilation,
                    groups,
                    False,
                    True,
                )
            _assert_close_tensor(actual, expected, spec.surface, rtol=1e-4, atol=1e-4)
            return

        if spec.surface in {"aten::miopen_depthwise_convolution", "aten::miopen_depthwise_convolution.out"}:
            depth_input = torch.randn(1, 3, 5, 5, device=device_obj, dtype=torch.float32)
            depth_weight = torch.randn(3, 1, 3, 3, device=device_obj, dtype=torch.float32)
            depth_bias = torch.randn(3, device=device_obj, dtype=torch.float32)
            expected = F.conv2d(depth_input, depth_weight, depth_bias, padding=1, groups=3)
            if spec.surface.endswith(".out"):
                out = torch.empty_like(expected)
                actual = torch.ops.aten.miopen_depthwise_convolution.out(
                    depth_input,
                    depth_weight,
                    depth_bias,
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    3,
                    False,
                    True,
                    out=out,
                )
                assert_out_identity(actual, out, spec.surface)
            else:
                actual = torch.ops.aten.miopen_depthwise_convolution(
                    depth_input,
                    depth_weight,
                    depth_bias,
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    3,
                    False,
                    True,
                )
            _assert_close_tensor(actual, expected, spec.surface, rtol=1e-4, atol=1e-4)
            return
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)

    raise AssertionError(f"No MIOpen convolution oracle implementation for {spec.surface}")


def _run_rocm_miopen_batch_norm(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    import torch.nn.functional as F

    device_obj = torch.device(device)
    torch.manual_seed(8001)
    eps = 1e-5
    momentum = 0.1
    x = torch.randn(2, 3, 4, 4, device=device_obj, dtype=torch.float32)
    weight = torch.randn(3, device=device_obj, dtype=torch.float32)
    bias = torch.randn(3, device=device_obj, dtype=torch.float32)
    running_mean = torch.zeros(3, device=device_obj, dtype=torch.float32)
    running_var = torch.ones(3, device=device_obj, dtype=torch.float32)
    try:
        forward = torch.ops.aten.miopen_batch_norm(
            x,
            weight,
            bias,
            running_mean.clone(),
            running_var.clone(),
            True,
            momentum,
            eps,
        )
        expected = F.batch_norm(
            x,
            running_mean.clone(),
            running_var.clone(),
            weight,
            bias,
            training=True,
            momentum=momentum,
            eps=eps,
        )
        if spec.surface in {"aten::miopen_batch_norm", "aten::miopen_batch_norm.out"}:
            if spec.surface.endswith(".out"):
                outs = tuple(torch.empty_like(item) for item in forward)
                actual = torch.ops.aten.miopen_batch_norm.out(
                    x,
                    weight,
                    bias,
                    running_mean.clone(),
                    running_var.clone(),
                    True,
                    momentum,
                    eps,
                    out0=outs[0],
                    out1=outs[1],
                    out2=outs[2],
                )
                for index, out in enumerate(outs):
                    assert_out_identity(actual[index], out, f"{spec.surface}.out{index}")
            else:
                actual = forward
            _assert_close_tensor(actual[0], expected, f"{spec.surface}.output", rtol=1e-4, atol=1e-4)
            if actual[1].shape != (3,) or actual[2].shape != (3,):
                raise AssertionError(f"{spec.surface} returned malformed save tensors")
            return

        grad_output = torch.randn_like(forward[0])
        x_ref = x.detach().clone().requires_grad_(True)
        weight_ref = weight.detach().clone().requires_grad_(True)
        bias_ref = bias.detach().clone().requires_grad_(True)
        ref = F.batch_norm(
            x_ref,
            running_mean.clone(),
            running_var.clone(),
            weight_ref,
            bias_ref,
            training=True,
            momentum=momentum,
            eps=eps,
        )
        ref.backward(grad_output)
        expected_grads = (x_ref.grad, weight_ref.grad, bias_ref.grad)
        if spec.surface == "aten::miopen_batch_norm_backward":
            actual = torch.ops.aten.miopen_batch_norm_backward(
                x,
                grad_output,
                weight,
                running_mean,
                running_var,
                forward[1],
                forward[2],
                eps,
            )
        elif spec.surface == "aten::miopen_batch_norm_backward.out":
            outs = tuple(torch.empty_like(item) for item in expected_grads)
            actual = torch.ops.aten.miopen_batch_norm_backward.out(
                x,
                grad_output,
                weight,
                running_mean,
                running_var,
                forward[1],
                forward[2],
                eps,
                out0=outs[0],
                out1=outs[1],
                out2=outs[2],
            )
            for index, out in enumerate(outs):
                assert_out_identity(actual[index], out, f"{spec.surface}.out{index}")
        else:
            raise AssertionError(f"No MIOpen batch-norm oracle implementation for {spec.surface}")
        for index, (actual_grad, expected_grad) in enumerate(zip(actual, expected_grads)):
            _assert_close_tensor(actual_grad, expected_grad, f"{spec.surface}.grad{index}", rtol=1e-4, atol=1e-4)
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)


def _run_rocm_miopen_ctc(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    import torch.nn.functional as F

    device_obj = torch.device(device)
    log_probs, targets, input_lengths, target_lengths, blank = _cuda_ctc_sample(device_obj)
    tensor_input_lengths = torch.tensor(input_lengths, device=device_obj, dtype=torch.int32)
    tensor_target_lengths = torch.tensor(target_lengths, device=device_obj, dtype=torch.int32)
    try:
        if spec.surface in {"aten::_use_miopen_ctc_loss", "aten::_use_miopen_ctc_loss.Tensor"}:
            if spec.surface.endswith(".Tensor"):
                actual = torch.ops.aten._use_miopen_ctc_loss.Tensor(
                    log_probs,
                    targets,
                    tensor_input_lengths,
                    tensor_target_lengths,
                    blank,
                )
            else:
                actual = torch.ops.aten._use_miopen_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank)
            if not isinstance(actual, bool):
                raise AssertionError(f"{spec.surface} returned non-bool {type(actual).__name__}")
            return

        expected = F.ctc_loss(
            log_probs,
            targets.to(torch.long),
            torch.tensor(input_lengths, device=device_obj, dtype=torch.long),
            torch.tensor(target_lengths, device=device_obj, dtype=torch.long),
            blank=blank,
            reduction="none",
            zero_infinity=False,
        )
        if spec.surface == "aten::miopen_ctc_loss.Tensor":
            actual = torch.ops.aten.miopen_ctc_loss.Tensor(
                log_probs,
                targets,
                tensor_input_lengths,
                tensor_target_lengths,
                blank,
                True,
                False,
            )
        elif spec.surface == "aten::miopen_ctc_loss.out":
            probe = torch.ops.aten.miopen_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, True, False)
            out0 = torch.empty_like(probe[0])
            out1 = torch.empty_like(probe[1])
            actual = torch.ops.aten.miopen_ctc_loss.out(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                blank,
                True,
                False,
                out0=out0,
                out1=out1,
            )
            assert_out_identity(actual[0], out0, f"{spec.surface}.out0")
            assert_out_identity(actual[1], out1, f"{spec.surface}.out1")
        else:
            actual = torch.ops.aten.miopen_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, True, False)
        _assert_close_tensor(actual[0], expected, f"{spec.surface}.loss", rtol=1e-4, atol=1e-4)
        if actual[1].device.type != device_obj.type or actual[1].numel() == 0:
            raise AssertionError(f"{spec.surface} returned malformed MIOpen CTC workspace")
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)


def _run_rocm_miopen_rnn(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    import torch.nn as nn

    device_obj = torch.device(device)
    torch.manual_seed(8002)
    module = nn.LSTM(3, 4, 1, batch_first=True).to(device_obj)
    x = torch.randn(2, 5, 3, device=device_obj, dtype=torch.float32)
    hx = torch.randn(1, 2, 4, device=device_obj, dtype=torch.float32)
    cx = torch.randn(1, 2, 4, device=device_obj, dtype=torch.float32)
    weights = [
        module.weight_ih_l0.detach(),
        module.weight_hh_l0.detach(),
        module.bias_ih_l0.detach(),
        module.bias_hh_l0.detach(),
    ]
    mode = 2
    hidden_size = 4
    weight_stride0 = 4
    num_layers = 1
    batch_first = True
    dropout = 0.0
    train = True
    bidirectional = False
    batch_sizes: list[int] = []
    try:
        forward = torch.ops.aten.miopen_rnn(
            x,
            weights,
            weight_stride0,
            hx,
            cx,
            mode,
            hidden_size,
            num_layers,
            batch_first,
            dropout,
            train,
            bidirectional,
            batch_sizes,
            None,
        )
        expected_output, (expected_hy, expected_cy) = module(x, (hx, cx))
        if spec.surface in {"aten::miopen_rnn", "aten::miopen_rnn.out"}:
            if spec.surface.endswith(".out"):
                outs = [torch.empty_like(item) for item in forward[:5]]
                actual = torch.ops.aten.miopen_rnn.out(
                    x,
                    weights,
                    weight_stride0,
                    hx,
                    cx,
                    mode,
                    hidden_size,
                    num_layers,
                    batch_first,
                    dropout,
                    train,
                    bidirectional,
                    batch_sizes,
                    None,
                    out0=outs[0],
                    out1=outs[1],
                    out2=outs[2],
                    out3=outs[3],
                    out4=outs[4],
                )
                for index, out in enumerate(outs):
                    assert_out_identity(actual[index], out, f"{spec.surface}.out{index}")
            else:
                actual = forward
            _assert_close_tensor(actual[0], expected_output, f"{spec.surface}.output", rtol=1e-4, atol=1e-4)
            _assert_close_tensor(actual[1], expected_hy, f"{spec.surface}.hy", rtol=1e-4, atol=1e-4)
            _assert_close_tensor(actual[2], expected_cy, f"{spec.surface}.cy", rtol=1e-4, atol=1e-4)
            if actual[4].device.type != device_obj.type or actual[4].numel() == 0:
                raise AssertionError(f"{spec.surface} returned malformed reserve tensor")
            return

        grad_output = torch.randn_like(forward[0])
        grad_hy = torch.randn_like(forward[1])
        grad_cy = torch.randn_like(forward[2])
        ref_x = x.detach().clone().requires_grad_(True)
        ref_hx = hx.detach().clone().requires_grad_(True)
        ref_cx = cx.detach().clone().requires_grad_(True)
        ref_module = nn.LSTM(3, 4, 1, batch_first=True).to(device_obj)
        with torch.no_grad():
            ref_module.weight_ih_l0.copy_(weights[0])
            ref_module.weight_hh_l0.copy_(weights[1])
            ref_module.bias_ih_l0.copy_(weights[2])
            ref_module.bias_hh_l0.copy_(weights[3])
        ref_output, (ref_hy, ref_cy) = ref_module(ref_x, (ref_hx, ref_cx))
        torch.autograd.backward((ref_output, ref_hy, ref_cy), (grad_output, grad_hy, grad_cy))
        if spec.surface == "aten::miopen_rnn_backward":
            actual = torch.ops.aten.miopen_rnn_backward(
                x,
                weights,
                weight_stride0,
                forward[3],
                hx,
                cx,
                forward[0],
                grad_output,
                grad_hy,
                grad_cy,
                mode,
                hidden_size,
                num_layers,
                batch_first,
                dropout,
                train,
                bidirectional,
                batch_sizes,
                None,
                forward[4],
                [True, True, True, True],
            )
        elif spec.surface == "aten::miopen_rnn_backward.out":
            out0 = torch.empty_like(x)
            out1 = torch.empty_like(hx)
            out2 = torch.empty_like(cx)
            out3 = [torch.empty_like(item) for item in weights]
            actual = torch.ops.aten.miopen_rnn_backward.out(
                x,
                weights,
                weight_stride0,
                forward[3],
                hx,
                cx,
                forward[0],
                grad_output,
                grad_hy,
                grad_cy,
                mode,
                hidden_size,
                num_layers,
                batch_first,
                dropout,
                train,
                bidirectional,
                batch_sizes,
                None,
                forward[4],
                [True, True, True, True],
                out0=out0,
                out1=out1,
                out2=out2,
                out3=out3,
            )
            if actual is not None:
                raise AssertionError(f"{spec.surface} should return None for the out overload")
            actual = (out0, out1, out2, out3)
        else:
            raise AssertionError(f"No MIOpen RNN oracle implementation for {spec.surface}")
        _assert_close_tensor(actual[0], ref_x.grad, f"{spec.surface}.grad_input", rtol=1e-4, atol=1e-4)
        _assert_close_tensor(actual[1], ref_hx.grad, f"{spec.surface}.grad_hx", rtol=1e-4, atol=1e-4)
        _assert_close_tensor(actual[2], ref_cx.grad, f"{spec.surface}.grad_cx", rtol=1e-4, atol=1e-4)
        for index, (actual_grad, expected_grad) in enumerate(zip(actual[3], (
            ref_module.weight_ih_l0.grad,
            ref_module.weight_hh_l0.grad,
            ref_module.bias_ih_l0.grad,
            ref_module.bias_hh_l0.grad,
        ))):
            _assert_close_tensor(actual_grad, expected_grad, f"{spec.surface}.grad_weight{index}", rtol=1e-4, atol=1e-4)
    except Exception as exc:
        _raise_backend_unavailable_if_applicable(spec, exc)


def _run_backend_property(spec: OracleSpec, device: str) -> None:
    _check_backend_gate(spec, device)
    if spec.contract_status == "blocked":
        reason = spec.reason or "backend-pack surface is blocked by the upstream runtime or direct dispatcher schema"
        lowered = reason.lower()
        if "schema" in lowered or "direct invocation" in lowered or "binding" in lowered:
            raise OracleUnavailable(f"backend_schema_blocked: {spec.surface}: {reason}")
        if (
            "runtime" in lowered
            or "segfault" in lowered
            or "unsupported" in lowered
            or "not currently supported" in lowered
            or "should be overridden" in lowered
            or "compute capability" in lowered
            or "hardware" in lowered
            or "xla" in lowered
            or "rocm" in lowered
        ):
            raise OracleUnavailable(f"backend_runtime_blocked: {spec.surface}: {reason}")
        raise OracleUnavailable(f"backend_contract_blocked: {spec.surface}: {reason}")
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
    "mps_philox": _run_mps_philox,
    "mkldnn_shape": _run_mkldnn_shape,
    "mkldnn_linear": _run_mkldnn_linear,
    "mkldnn_convolution": _run_mkldnn_convolution,
    "mkldnn_pooling": _run_mkldnn_pooling,
    "mkldnn_rnn": _run_mkldnn_rnn,
    "nnpack_convolution": _run_nnpack_convolution,
    "fbgemm_linear": _run_fbgemm_linear,
    "fbgemm_wrapped_linear": _run_fbgemm_wrapped_linear,
    "fbgemm_quantized_cell": _run_fbgemm_quantized_cell,
    "cuda_fused_dropout": _run_cuda_fused_dropout,
    "cuda_semi_structured_sparse": _run_cuda_semi_structured_sparse,
    "cuda_cslt": _run_cuda_cslt,
    "cuda_cudnn_attention": _run_cuda_cudnn_attention,
    "cuda_flash_attention_no_dropout_inplace": _run_cuda_flash_attention_no_dropout_inplace,
    "cuda_fused_rms_norm_backward": _run_cuda_fused_rms_norm_backward,
    "cuda_dtype_out_matmul": _run_cuda_dtype_out_matmul,
    "cuda_batch_norm_internal": _run_cuda_batch_norm_internal,
    "cuda_thnn_cell_backward": _run_cuda_thnn_cell_backward,
    "cuda_mixed_dtypes_linear": _run_cuda_mixed_dtypes_linear,
    "cuda_scaled_grouped_mm": _run_cuda_scaled_grouped_mm,
    "cuda_cudnn_convolution": _run_cuda_cudnn_convolution,
    "cuda_cudnn_grid": _run_cuda_cudnn_grid,
    "cuda_cudnn_batch_norm": _run_cuda_cudnn_batch_norm,
    "cuda_cudnn_ctc": _run_cuda_cudnn_ctc,
    "cuda_cudnn_dropout_state": _run_cuda_cudnn_dropout_state,
    "cuda_cudnn_is_acceptable": _run_cuda_cudnn_is_acceptable,
    "cuda_triton_attention": _run_cuda_triton_attention,
    "cuda_cudnn_rnn": _run_cuda_cudnn_rnn,
    "rocm_miopen_convolution": _run_rocm_miopen_convolution,
    "rocm_miopen_batch_norm": _run_rocm_miopen_batch_norm,
    "rocm_miopen_ctc": _run_rocm_miopen_ctc,
    "rocm_miopen_rnn": _run_rocm_miopen_rnn,
    "backend_property": _run_backend_property,
}


_SPECS: dict[str, OracleSpec] = {}


_CPU_BUILD_EVIDENCE = (
    "data/backend-pack-evidence/"
    "torchcts-evidence-thinkstationpgx-0f66-cpu-20260702T205910Z.tar.gz"
)

_CPU_BUILD_PROMOTED = frozenset({
    "aten::_mkldnn_reshape",
    "aten::_mkldnn_reshape.out",
    "aten::_mkldnn_transpose",
    "aten::_mkldnn_transpose.out",
    "aten::_nnpack_spatial_convolution",
    "aten::_nnpack_spatial_convolution.out",
    "aten::mkldnn_adaptive_avg_pool2d",
    "aten::mkldnn_adaptive_avg_pool2d.out",
    "aten::mkldnn_adaptive_avg_pool2d_backward",
    "aten::mkldnn_adaptive_avg_pool2d_backward.out",
    "aten::mkldnn_convolution",
    "aten::mkldnn_convolution.out",
    "aten::mkldnn_linear",
    "aten::mkldnn_linear.out",
    "aten::mkldnn_linear_backward",
    "aten::mkldnn_linear_backward.out",
    "aten::mkldnn_linear_backward_input",
    "aten::mkldnn_linear_backward_input.out",
    "aten::mkldnn_linear_backward_weights",
    "aten::mkldnn_linear_backward_weights.out",
    "aten::mkldnn_max_pool2d",
    "aten::mkldnn_max_pool2d.out",
    "aten::mkldnn_max_pool3d",
    "aten::mkldnn_max_pool3d.out",
    "aten::mkldnn_rnn_layer.out",
    "aten::mkldnn_rnn_layer_backward",
    "aten::mkldnn_rnn_layer_backward.out",
    "aten::to_mkldnn",
    "aten::to_mkldnn.out",
    "aten::to_mkldnn_backward",
})

_FBGEMM_EVIDENCE = _CPU_BUILD_EVIDENCE

_FBGEMM_PROMOTED = frozenset({
    "aten::fbgemm_linear_fp16_weight",
    "aten::fbgemm_linear_fp16_weight.out",
    "aten::fbgemm_linear_fp16_weight_fp32_activation",
    "aten::fbgemm_linear_fp16_weight_fp32_activation.out",
    "aten::fbgemm_linear_quantize_weight",
    "aten::fbgemm_pack_gemm_matrix_fp16",
})

_CUDA_SEMI_STRUCTURED_THREAD_MASK_EVIDENCE = (
    "data/backend-pack-evidence/"
    "torchcts-evidence-thinkstationpgx-0f66-cuda-20260702T205947Z.tar.gz"
)

_CUDA_SEMI_STRUCTURED_THREAD_MASK_PROMOTED = frozenset({
    "aten::_sparse_semi_structured_apply",
    "aten::_sparse_semi_structured_tile",
})


def _coverage_status_for(surface: str, promoted: frozenset[str]) -> str:
    return "covered_backend_pack" if surface in promoted else "pending_backend_pack"


def _contract_status_for(surface: str, promoted: frozenset[str]) -> str:
    return "accepted" if surface in promoted else "candidate"


def _promotion_evidence_for(surface: str, promoted: frozenset[str], evidence_path: str) -> str:
    return evidence_path if surface in promoted else ""


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
        contract_status="accepted",
        contract_ref="docs/coverage/contract-evidence.md#mps-autograd-backward-backend-pack",
        promotion_evidence="reviewed-local-mps-generated-autograd-backward-run-2026-06-29",
        promotion_backend="mps",
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
        contract_status="accepted",
        contract_ref="docs/coverage/contract-evidence.md#cuda-fused-dropout-backend-pack",
        promotion_evidence=(
            "data/backend-pack-evidence/"
            "torchcts-evidence-thinkstationpgx-0f66-cuda-20260701T195851Z.tar.gz"
        ),
        promotion_backend="cuda",
    ))

for _surface in (
    "aten::_sparse_semi_structured_addmm",
    "aten::_sparse_semi_structured_linear",
    "aten::_sparse_semi_structured_mm",
    "aten::_to_sparse_semi_structured",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="semi_structured_sparse_backend_pack",
        coverage_status="covered_backend_pack",
        coverage_kind="backend_pack",
        runner="cuda_semi_structured_sparse",
        backend_gate="cuda",
        semantic_level=5,
        reason=(
            "Semi-structured sparse conversion/mm/addmm/linear internals are "
            "covered by a CUDA direct-dispatch runner using PyTorch-created "
            "2:4 compressed values and dense references. Spark GB10 PyTorch "
            "2.11.0+cu128 promotion evidence used an LD_PRELOAD shim to bypass "
            "PyTorch's invalid GB10 capability guard by reporting SM 8.9."
        ),
        contract_status="accepted",
        contract_ref="docs/coverage/contract-evidence.md#cuda-semi-structured-sparse-backend-pack",
        promotion_evidence=(
            "data/backend-pack-evidence/"
            "torchcts-evidence-thinkstationpgx-0f66-cuda-20260702T165916Z.tar.gz"
        ),
        promotion_backend="cuda",
    ))

for _surface in (
    "aten::_sparse_semi_structured_apply",
    "aten::_sparse_semi_structured_apply_dense",
    "aten::_sparse_semi_structured_tile",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="semi_structured_sparse_thread_mask_backend_pack",
        coverage_status=_coverage_status_for(_surface, _CUDA_SEMI_STRUCTURED_THREAD_MASK_PROMOTED),
        coverage_kind="backend_pack",
        runner="cuda_semi_structured_sparse",
        backend_gate="cuda",
        semantic_level=5,
        reason=(
            "Semi-structured sparse tile/apply internals have a source-derived "
            "candidate runner that checks tile/apply packed equivalence, dense "
            "mask application, and resulting 2:4 sparse matmul behavior. Spark "
            "GB10 promotion evidence uses the same SM 8.9 guard-bypass caveat as "
            "the other semi-structured sparse promotions."
        ),
        contract_status=_contract_status_for(_surface, _CUDA_SEMI_STRUCTURED_THREAD_MASK_PROMOTED),
        contract_ref="docs/coverage/contract-evidence.md#cuda-semi-structured-sparse-thread-mask-helpers",
        promotion_evidence=_promotion_evidence_for(
            _surface,
            _CUDA_SEMI_STRUCTURED_THREAD_MASK_PROMOTED,
            _CUDA_SEMI_STRUCTURED_THREAD_MASK_EVIDENCE,
        ),
        promotion_backend="cuda",
    ))

for _surface in (
    "aten::_cslt_compress",
    "aten::_cslt_sparse_mm",
    "aten::_cslt_sparse_mm_search",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="cuda_cslt_backend_pack",
        coverage_status="covered_backend_pack",
        coverage_kind="backend_pack",
        runner="cuda_cslt",
        backend_gate="cuda",
        semantic_level=5,
        reason=(
            "cuSparseLt direct dispatcher helpers are validated by a CUDA "
            "runner that checks compressed matmul against dense references on "
            "a PyTorch build where cuSparseLt initializes for Spark GB10."
        ),
        contract_status="accepted",
        contract_ref="docs/coverage/contract-evidence.md#cuda-cusparselt-backend-pack",
        promotion_evidence=(
            "data/backend-pack-evidence/"
            "torchcts-evidence-thinkstationpgx-0f66-cuda-20260702T195008Z.tar.gz"
        ),
        promotion_backend="cuda",
    ))

for _surface in (
    "aten::_cudnn_attention_forward",
    "aten::_scaled_dot_product_cudnn_attention",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="cuda_cudnn_attention_backend_pack",
        coverage_status="covered_backend_pack",
        coverage_kind="backend_pack",
        runner="cuda_cudnn_attention",
        backend_gate="cuda",
        semantic_level=5,
        reason=(
            "cuDNN attention forward is validated by a CUDA runner that checks "
            "direct dispatcher output against public scaled-dot-product "
            "attention on a PyTorch build with a safe direct invocation path."
        ),
        contract_status="accepted",
        contract_ref="docs/coverage/contract-evidence.md#cuda-cudnn-attention-backend-pack",
        promotion_evidence=(
            "data/backend-pack-evidence/"
            "torchcts-evidence-thinkstationpgx-0f66-cuda-20260702T195008Z.tar.gz"
        ),
        promotion_backend="cuda",
    ))

for _surface in (
    "aten::_cudnn_attention_backward",
    "aten::_scaled_dot_product_cudnn_attention_backward",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="cuda_cudnn_attention_backend_pack",
        coverage_status="pending_backend_pack",
        coverage_kind="backend_pack",
        runner="backend_property",
        backend_gate="cuda",
        semantic_level=5,
        reason=(
            "cuDNN attention backward direct dispatcher probing remains blocked: "
            "PyTorch 2.12.1+cu130 on Spark GB10 exposes safe forward calls, but "
            "the backward overloads still fail with binding shape errors and "
            "internal assert failures."
        ),
        contract_status="blocked",
        contract_ref="docs/coverage/contract-evidence.md#cuda-cudnn-attention-backend-pack",
        promotion_backend="cuda",
    ))

for _surface in (
    "aten::cudnn_convolution",
    "aten::cudnn_convolution.out",
    "aten::cudnn_convolution_add_relu",
    "aten::cudnn_convolution_add_relu.out",
    "aten::cudnn_convolution_relu",
    "aten::cudnn_convolution_relu.out",
    "aten::cudnn_convolution_transpose",
    "aten::cudnn_convolution_transpose.out",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="cuda_cudnn_convolution_backend_pack",
        coverage_status="covered_backend_pack",
        coverage_kind="backend_pack",
        runner="cuda_cudnn_convolution",
        backend_gate="cuda",
        semantic_level=5,
        reason="cuDNN convolution internals are validated by direct CUDA dispatcher calls against public convolution formulas, including out identity.",
        contract_status="accepted",
        contract_ref="docs/coverage/contract-evidence.md#cuda-cudnn-convolution-backend-pack",
        promotion_evidence="data/backend-pack-evidence/torchcts-evidence-thinkstationpgx-0f66-cuda-20260702T191535Z.tar.gz",
        promotion_backend="cuda",
    ))

for _surface in (
    "aten::cudnn_affine_grid_generator",
    "aten::cudnn_affine_grid_generator.out",
    "aten::cudnn_affine_grid_generator_backward",
    "aten::cudnn_affine_grid_generator_backward.out",
    "aten::cudnn_grid_sampler",
    "aten::cudnn_grid_sampler.out",
    "aten::cudnn_grid_sampler_backward",
    "aten::cudnn_grid_sampler_backward.out",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="cuda_cudnn_grid_backend_pack",
        coverage_status="covered_backend_pack",
        coverage_kind="backend_pack",
        runner="cuda_cudnn_grid",
        backend_gate="cuda",
        semantic_level=5,
        reason="cuDNN grid internals are validated by direct CUDA dispatcher calls against public affine-grid/grid-sample values and gradients.",
        contract_status="accepted",
        contract_ref="docs/coverage/contract-evidence.md#cuda-cudnn-grid-backend-pack",
        promotion_evidence="data/backend-pack-evidence/torchcts-evidence-thinkstationpgx-0f66-cuda-20260702T191546Z.tar.gz",
        promotion_backend="cuda",
    ))

for _surface in (
    "aten::cudnn_batch_norm",
    "aten::cudnn_batch_norm_backward",
    "aten::cudnn_batch_norm_backward.out",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="cuda_cudnn_batch_norm_backend_pack",
        coverage_status="covered_backend_pack",
        coverage_kind="backend_pack",
        runner="cuda_cudnn_batch_norm",
        backend_gate="cuda",
        semantic_level=5,
        reason="cuDNN batch-norm internals are validated by direct CUDA dispatcher calls against public batch_norm/autograd references.",
        contract_status="accepted",
        contract_ref="docs/coverage/contract-evidence.md#cuda-cudnn-batch-norm-backend-pack",
        promotion_evidence="data/backend-pack-evidence/torchcts-evidence-thinkstationpgx-0f66-cuda-20260702T191752Z.tar.gz",
        promotion_backend="cuda",
    ))

_register(OracleSpec(
    surface="aten::cudnn_batch_norm.out",
    oracle_id="cuda_cudnn_batch_norm_backend_pack",
    coverage_status="covered_backend_pack",
    coverage_kind="backend_pack",
    runner="cuda_cudnn_batch_norm",
    backend_gate="cuda",
    semantic_level=5,
    reason=(
        "cuDNN batch-norm out is validated by a CUDA runner against public "
        "batch_norm output semantics on PyTorch 2.12.1+cu130; older PyTorch "
        "2.11.0+cu128 on Spark GB10 is runtime-gated due a binding internal assert."
    ),
    contract_status="accepted",
    contract_ref="docs/coverage/contract-evidence.md#cuda-cudnn-batch-norm-backend-pack",
    promotion_evidence=(
        "data/backend-pack-evidence/"
        "torchcts-evidence-thinkstationpgx-0f66-cuda-20260702T195008Z.tar.gz"
    ),
    promotion_backend="cuda",
))

for _surface in (
    "aten::_cudnn_ctc_loss",
    "aten::_cudnn_ctc_loss.Tensor",
    "aten::_cudnn_ctc_loss.out",
    "aten::_use_cudnn_ctc_loss",
    "aten::_use_cudnn_ctc_loss.Tensor",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="cuda_cudnn_ctc_backend_pack",
        coverage_status="covered_backend_pack",
        coverage_kind="backend_pack",
        runner="cuda_cudnn_ctc",
        backend_gate="cuda",
        semantic_level=5,
        reason="cuDNN CTC internals are validated by direct CUDA dispatcher calls against public ctc_loss and use-cudnn predicate contracts.",
        contract_status="accepted",
        contract_ref="docs/coverage/contract-evidence.md#cuda-cudnn-ctc-backend-pack",
        promotion_evidence="data/backend-pack-evidence/torchcts-evidence-thinkstationpgx-0f66-cuda-20260702T191837Z.tar.gz",
        promotion_backend="cuda",
    ))

for _surface in (
    "aten::_cudnn_init_dropout_state",
    "aten::_cudnn_init_dropout_state.out",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="cuda_cudnn_dropout_state_backend_pack",
        coverage_status="covered_backend_pack",
        coverage_kind="backend_pack",
        runner="cuda_cudnn_dropout_state",
        backend_gate="cuda",
        semantic_level=5,
        reason="cuDNN dropout-state internals are validated by direct CUDA dispatcher calls for state allocation device, dtype, non-empty state, and out identity.",
        contract_status="accepted",
        contract_ref="docs/coverage/contract-evidence.md#cuda-cudnn-dropout-state-backend-pack",
        promotion_evidence="data/backend-pack-evidence/torchcts-evidence-thinkstationpgx-0f66-cuda-20260702T191937Z.tar.gz",
        promotion_backend="cuda",
    ))

_register(OracleSpec(
    surface="aten::cudnn_is_acceptable",
    oracle_id="cuda_cudnn_is_acceptable_backend_pack",
    coverage_status="covered_backend_pack",
    coverage_kind="backend_pack",
    runner="cuda_cudnn_is_acceptable",
    backend_gate="cuda",
    semantic_level=5,
    reason="cuDNN acceptability predicate is validated by comparing direct dispatcher output to torch.backends.cudnn.is_acceptable.",
    contract_status="accepted",
    contract_ref="docs/coverage/contract-evidence.md#cuda-cudnn-acceptability-backend-pack",
    promotion_evidence="data/backend-pack-evidence/torchcts-evidence-thinkstationpgx-0f66-cuda-20260702T191937Z.tar.gz",
    promotion_backend="cuda",
))

for _surface in (
    "aten::_cudnn_rnn",
    "aten::_cudnn_rnn.out",
    "aten::_cudnn_rnn_backward",
    "aten::_cudnn_rnn_backward.out",
    "aten::_cudnn_rnn_flatten_weight",
    "aten::_cudnn_rnn_flatten_weight.out",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="cuda_cudnn_rnn_backend_pack",
        coverage_status="covered_backend_pack",
        coverage_kind="backend_pack",
        runner="cuda_cudnn_rnn",
        backend_gate="cuda",
        semantic_level=5,
        reason="cuDNN RNN internals are validated by direct CUDA dispatcher calls against public nn.LSTM forward/backward references and out identity.",
        contract_status="accepted",
        contract_ref="docs/coverage/contract-evidence.md#cuda-cudnn-rnn-backend-pack",
        promotion_evidence="data/backend-pack-evidence/torchcts-evidence-thinkstationpgx-0f66-cuda-20260702T192456Z.tar.gz",
        promotion_backend="cuda",
    ))

for _surface in (
    "aten::_triton_scaled_dot_attention",
    "aten::_triton_scaled_dot_attention.out",
    "aten::_triton_multi_head_attention",
    "aten::_triton_multi_head_attention.out",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="cuda_triton_attention_backend_pack",
        coverage_status="pending_backend_pack",
        coverage_kind="backend_pack",
        runner="backend_property",
        backend_gate="cuda",
        semantic_level=5,
        reason=(
            "PyTorch 2.11.0+cu128 and 2.12.1+cu130 report this Triton dispatcher "
            "surface should be overridden in Python before direct use."
        ),
        contract_status="blocked",
        contract_ref="docs/coverage/contract-evidence.md#cuda-triton-attention-backend-pack",
        promotion_backend="cuda",
    ))

_register(OracleSpec(
    surface="aten::_flash_attention_forward_no_dropout_inplace",
    oracle_id="cuda_flash_attention_no_dropout_inplace_backend_pack",
    coverage_status="covered_backend_pack",
    coverage_kind="backend_pack",
    runner="cuda_flash_attention_no_dropout_inplace",
    backend_gate="cuda",
    semantic_level=5,
    reason=(
        "Flash-attention no-dropout inplace is validated by a CUDA runner that "
        "checks the mutated output and logsumexp against public "
        "scaled_dot_product_attention formulas."
    ),
    contract_status="accepted",
    contract_ref="docs/coverage/contract-evidence.md#cuda-flash-attention-no-dropout-inplace-backend-pack",
    promotion_evidence=(
        "data/backend-pack-evidence/"
        "torchcts-evidence-thinkstationpgx-0f66-cuda-20260702T200741Z.tar.gz"
    ),
    promotion_backend="cuda",
))

_register(OracleSpec(
    surface="aten::_fused_rms_norm_backward",
    oracle_id="cuda_fused_rms_norm_backward_backend_pack",
    coverage_status="covered_backend_pack",
    coverage_kind="backend_pack",
    runner="cuda_fused_rms_norm_backward",
    backend_gate="cuda",
    semantic_level=5,
    reason=(
        "Fused RMSNorm backward is validated by a CUDA runner that checks direct "
        "gradients against an explicit RMSNorm autograd reference."
    ),
    contract_status="accepted",
    contract_ref="docs/coverage/contract-evidence.md#cuda-fused-rmsnorm-backward-backend-pack",
    promotion_evidence=(
        "data/backend-pack-evidence/"
        "torchcts-evidence-thinkstationpgx-0f66-cuda-20260702T200741Z.tar.gz"
    ),
    promotion_backend="cuda",
))

for _surface in (
    "aten::addmm.dtype_out",
    "aten::baddbmm.dtype_out",
    "aten::bmm.dtype_out",
    "aten::mm.dtype_out",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="cuda_dtype_out_matmul_backend_pack",
        coverage_status="covered_backend_pack",
        coverage_kind="backend_pack",
        runner="cuda_dtype_out_matmul",
        backend_gate="cuda",
        semantic_level=5,
        reason=(
            "CUDA dtype-out matmul overload is validated by a runner that checks "
            "out identity, output dtype, and dense fp32 matmul/addmm formulas."
        ),
        contract_status="accepted",
        contract_ref="docs/coverage/contract-evidence.md#cuda-dtype-out-matmul-backend-pack",
        promotion_evidence=(
            "data/backend-pack-evidence/"
            "torchcts-evidence-thinkstationpgx-0f66-cuda-20260702T200741Z.tar.gz"
        ),
        promotion_backend="cuda",
    ))

for _surface in (
    "aten::batch_norm_backward_elemt",
    "aten::batch_norm_backward_reduce",
    "aten::batch_norm_gather_stats",
    "aten::batch_norm_gather_stats_with_counts",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="cuda_batch_norm_internal_backend_pack",
        coverage_status="covered_backend_pack",
        coverage_kind="backend_pack",
        runner="cuda_batch_norm_internal",
        backend_gate="cuda",
        semantic_level=5,
        reason=(
            "CUDA batch-norm internal helper is validated by a runner that checks "
            "explicit per-channel backward/gather formulas and running-stat updates."
        ),
        contract_status="accepted",
        contract_ref="docs/coverage/contract-evidence.md#cuda-batch-norm-internal-backend-pack",
        promotion_evidence=(
            "data/backend-pack-evidence/"
            "torchcts-evidence-thinkstationpgx-0f66-cuda-20260702T200741Z.tar.gz"
        ),
        promotion_backend="cuda",
    ))

for _surface in (
    "aten::_thnn_fused_gru_cell_backward",
    "aten::_thnn_fused_lstm_cell_backward_impl",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="cuda_thnn_cell_backward_backend_pack",
        coverage_status="covered_backend_pack",
        coverage_kind="backend_pack",
        runner="cuda_thnn_cell_backward",
        backend_gate="cuda",
        semantic_level=5,
        reason=(
            "CUDA fused THNN cell backward helper is validated by a runner that "
            "checks direct gradients against explicit GRU/LSTM autograd formulas."
        ),
        contract_status="accepted",
        contract_ref="docs/coverage/contract-evidence.md#cuda-thnn-cell-backward-backend-pack",
        promotion_evidence=(
            "data/backend-pack-evidence/"
            "torchcts-evidence-thinkstationpgx-0f66-cuda-20260702T200741Z.tar.gz"
        ),
        promotion_backend="cuda",
    ))

_register(OracleSpec(
    surface="aten::_mixed_dtypes_linear",
    oracle_id="cuda_mixed_dtypes_linear_backend_pack",
    coverage_status="pending_backend_pack",
    coverage_kind="backend_pack",
    runner="cuda_mixed_dtypes_linear",
    backend_gate="cuda",
    semantic_level=5,
    reason=(
        "Mixed-dtypes linear has a candidate CUDA runner using PyTorch's "
        "CUTLASS weight reorder helper and a dense dequantized linear reference; "
        "Spark GB10 currently hits PyTorch's compute-capability guard."
    ),
    contract_status="candidate",
    contract_ref="docs/coverage/contract-evidence.md#cuda-mixed-dtypes-linear-backend-pack",
    promotion_backend="cuda",
))

for _surface in (
    "aten::_scaled_grouped_mm",
    "aten::_scaled_grouped_mm_v2",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="cuda_scaled_grouped_mm_backend_pack",
        coverage_status="pending_backend_pack",
        coverage_kind="backend_pack",
        runner="cuda_scaled_grouped_mm",
        backend_gate="cuda",
        semantic_level=5,
        reason=(
            "Scaled grouped matmul has a candidate CUDA runner using FP8 inputs, "
            "tensorwise scales, and dense grouped matmul references; Spark GB10 "
            "currently hits PyTorch's compute-capability guard."
        ),
        contract_status="candidate",
        contract_ref="docs/coverage/contract-evidence.md#cuda-scaled-grouped-matmul-backend-pack",
        promotion_backend="cuda",
    ))

for _surface in (
    "aten::_mkldnn_reshape",
    "aten::_mkldnn_reshape.out",
    "aten::_mkldnn_transpose",
    "aten::_mkldnn_transpose.out",
    "aten::_mkldnn_transpose_",
    "aten::to_mkldnn",
    "aten::to_mkldnn.out",
    "aten::to_mkldnn_backward",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="mkldnn_shape_backend_pack",
        coverage_status=_coverage_status_for(_surface, _CPU_BUILD_PROMOTED),
        coverage_kind="backend_pack",
        runner="mkldnn_shape",
        backend_gate="cpu_build",
        semantic_level=5,
        reason=(
            "MKLDNN shape/conversion helper has a source-derived candidate runner "
            "that validates dense logical values, MKLDNN layout, and out identity "
            "where PyTorch exposes a usable direct path."
        ),
        contract_status=_contract_status_for(_surface, _CPU_BUILD_PROMOTED),
        contract_ref="docs/coverage/contract-evidence.md#cpu-build-mkldnn-shape-and-conversion-backend-pack",
        promotion_evidence=_promotion_evidence_for(_surface, _CPU_BUILD_PROMOTED, _CPU_BUILD_EVIDENCE),
        promotion_backend="cpu_build",
    ))

for _surface in (
    "aten::mkldnn_linear",
    "aten::mkldnn_linear.out",
    "aten::mkldnn_linear_backward",
    "aten::mkldnn_linear_backward.out",
    "aten::mkldnn_linear_backward_input",
    "aten::mkldnn_linear_backward_input.out",
    "aten::mkldnn_linear_backward_weights",
    "aten::mkldnn_linear_backward_weights.out",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="mkldnn_linear_backend_pack",
        coverage_status=_coverage_status_for(_surface, _CPU_BUILD_PROMOTED),
        coverage_kind="backend_pack",
        runner="mkldnn_linear",
        backend_gate="cpu_build",
        semantic_level=5,
        reason=(
            "MKLDNN linear helper has a source-derived candidate runner that "
            "compares forward and backward values to public dense linear/autograd references."
        ),
        contract_status=_contract_status_for(_surface, _CPU_BUILD_PROMOTED),
        contract_ref="docs/coverage/contract-evidence.md#cpu-build-mkldnn-linear-backend-pack",
        promotion_evidence=_promotion_evidence_for(_surface, _CPU_BUILD_PROMOTED, _CPU_BUILD_EVIDENCE),
        promotion_backend="cpu_build",
    ))

for _surface in (
    "aten::mkldnn_convolution",
    "aten::mkldnn_convolution.out",
    "aten::mkldnn_reorder_conv2d_weight",
    "aten::mkldnn_reorder_conv2d_weight.out",
    "aten::mkldnn_reorder_conv3d_weight",
    "aten::mkldnn_reorder_conv3d_weight.out",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="mkldnn_convolution_backend_pack",
        coverage_status=_coverage_status_for(_surface, _CPU_BUILD_PROMOTED),
        coverage_kind="backend_pack",
        runner="mkldnn_convolution",
        backend_gate="cpu_build",
        semantic_level=5,
        reason=(
            "MKLDNN convolution/reorder helper has a source-derived candidate runner "
            "that checks direct convolution values against public dense conv references."
        ),
        contract_status=_contract_status_for(_surface, _CPU_BUILD_PROMOTED),
        contract_ref="docs/coverage/contract-evidence.md#cpu-build-mkldnn-convolution-backend-pack",
        promotion_evidence=_promotion_evidence_for(_surface, _CPU_BUILD_PROMOTED, _CPU_BUILD_EVIDENCE),
        promotion_backend="cpu_build",
    ))

for _surface in (
    "aten::mkldnn_adaptive_avg_pool2d",
    "aten::mkldnn_adaptive_avg_pool2d.out",
    "aten::mkldnn_adaptive_avg_pool2d_backward",
    "aten::mkldnn_adaptive_avg_pool2d_backward.out",
    "aten::mkldnn_max_pool2d",
    "aten::mkldnn_max_pool2d.out",
    "aten::mkldnn_max_pool2d_backward",
    "aten::mkldnn_max_pool2d_backward.out",
    "aten::mkldnn_max_pool3d",
    "aten::mkldnn_max_pool3d.out",
    "aten::mkldnn_max_pool3d_backward",
    "aten::mkldnn_max_pool3d_backward.out",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="mkldnn_pooling_backend_pack",
        coverage_status=_coverage_status_for(_surface, _CPU_BUILD_PROMOTED),
        coverage_kind="backend_pack",
        runner="mkldnn_pooling",
        backend_gate="cpu_build",
        semantic_level=5,
        reason=(
            "MKLDNN pooling helper has a source-derived candidate runner that "
            "compares forward and backward values to public dense pooling/autograd references."
        ),
        contract_status=_contract_status_for(_surface, _CPU_BUILD_PROMOTED),
        contract_ref="docs/coverage/contract-evidence.md#cpu-build-mkldnn-pooling-backend-pack",
        promotion_evidence=_promotion_evidence_for(_surface, _CPU_BUILD_PROMOTED, _CPU_BUILD_EVIDENCE),
        promotion_backend="cpu_build",
    ))

for _surface in (
    "aten::mkldnn_rnn_layer",
    "aten::mkldnn_rnn_layer.out",
    "aten::mkldnn_rnn_layer_backward",
    "aten::mkldnn_rnn_layer_backward.out",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="mkldnn_rnn_backend_pack",
        coverage_status=_coverage_status_for(_surface, _CPU_BUILD_PROMOTED),
        coverage_kind="backend_pack",
        runner="mkldnn_rnn",
        backend_gate="cpu_build",
        semantic_level=5,
        reason=(
            "MKLDNN RNN helper has a source-derived candidate runner that compares "
            "one-layer LSTM forward and backward values to public torch.nn.LSTM references."
        ),
        contract_status=_contract_status_for(_surface, _CPU_BUILD_PROMOTED),
        contract_ref="docs/coverage/contract-evidence.md#cpu-build-mkldnn-rnn-backend-pack",
        promotion_evidence=_promotion_evidence_for(_surface, _CPU_BUILD_PROMOTED, _CPU_BUILD_EVIDENCE),
        promotion_backend="cpu_build",
    ))

for _surface in (
    "aten::_nnpack_spatial_convolution",
    "aten::_nnpack_spatial_convolution.out",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="nnpack_convolution_backend_pack",
        coverage_status=_coverage_status_for(_surface, _CPU_BUILD_PROMOTED),
        coverage_kind="backend_pack",
        runner="nnpack_convolution",
        backend_gate="cpu_build",
        semantic_level=5,
        reason=(
            "NNPACK spatial convolution has a source-derived candidate runner that "
            "compares the direct helper to public dense conv2d when NNPACK executes."
        ),
        contract_status=_contract_status_for(_surface, _CPU_BUILD_PROMOTED),
        contract_ref="docs/coverage/contract-evidence.md#cpu-build-nnpack-convolution-backend-pack",
        promotion_evidence=_promotion_evidence_for(_surface, _CPU_BUILD_PROMOTED, _CPU_BUILD_EVIDENCE),
        promotion_backend="cpu_build",
    ))

for _surface in (
    "aten::miopen_convolution",
    "aten::miopen_convolution.out",
    "aten::miopen_convolution_add_relu",
    "aten::miopen_convolution_relu",
    "aten::miopen_convolution_transpose",
    "aten::miopen_convolution_transpose.out",
    "aten::miopen_depthwise_convolution",
    "aten::miopen_depthwise_convolution.out",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="rocm_miopen_convolution_backend_pack",
        coverage_status="pending_backend_pack",
        coverage_kind="backend_pack",
        runner="rocm_miopen_convolution",
        backend_gate="rocm",
        semantic_level=5,
        reason=(
            "MIOpen convolution helper has a source-derived candidate runner "
            "mirroring the cuDNN convolution value/out-identity contract; promotion "
            "requires execution on a ROCm/HIP PyTorch build."
        ),
        contract_status="candidate",
        contract_ref="docs/coverage/contract-evidence.md#rocm-miopen-convolution-backend-pack",
        promotion_backend="rocm",
    ))

for _surface in (
    "aten::miopen_batch_norm",
    "aten::miopen_batch_norm.out",
    "aten::miopen_batch_norm_backward",
    "aten::miopen_batch_norm_backward.out",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="rocm_miopen_batch_norm_backend_pack",
        coverage_status="pending_backend_pack",
        coverage_kind="backend_pack",
        runner="rocm_miopen_batch_norm",
        backend_gate="rocm",
        semantic_level=5,
        reason=(
            "MIOpen batch-norm helper has a source-derived candidate runner "
            "checking public batch_norm forward values and autograd gradients; "
            "promotion requires execution on a ROCm/HIP PyTorch build."
        ),
        contract_status="candidate",
        contract_ref="docs/coverage/contract-evidence.md#rocm-miopen-batch-norm-backend-pack",
        promotion_backend="rocm",
    ))

for _surface in (
    "aten::_use_miopen_ctc_loss",
    "aten::_use_miopen_ctc_loss.Tensor",
    "aten::miopen_ctc_loss",
    "aten::miopen_ctc_loss.Tensor",
    "aten::miopen_ctc_loss.out",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="rocm_miopen_ctc_backend_pack",
        coverage_status="pending_backend_pack",
        coverage_kind="backend_pack",
        runner="rocm_miopen_ctc",
        backend_gate="rocm",
        semantic_level=5,
        reason=(
            "MIOpen CTC helper has a source-derived candidate runner checking "
            "predicate return type, public CTC loss values, and workspace shape; "
            "promotion requires execution on a ROCm/HIP PyTorch build."
        ),
        contract_status="candidate",
        contract_ref="docs/coverage/contract-evidence.md#rocm-miopen-ctc-backend-pack",
        promotion_backend="rocm",
    ))

for _surface in (
    "aten::miopen_rnn",
    "aten::miopen_rnn.out",
    "aten::miopen_rnn_backward",
    "aten::miopen_rnn_backward.out",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="rocm_miopen_rnn_backend_pack",
        coverage_status="pending_backend_pack",
        coverage_kind="backend_pack",
        runner="rocm_miopen_rnn",
        backend_gate="rocm",
        semantic_level=5,
        reason=(
            "MIOpen RNN helper has a source-derived candidate runner checking "
            "one-layer LSTM forward values, gradients, reserve tensors, and out "
            "identity; promotion requires execution on a ROCm/HIP PyTorch build."
        ),
        contract_status="candidate",
        contract_ref="docs/coverage/contract-evidence.md#rocm-miopen-rnn-backend-pack",
        promotion_backend="rocm",
    ))

_register(OracleSpec(
    surface="aten::_propagate_xla_data",
    oracle_id="xla_data_propagation_bridge_blocked",
    coverage_status="pending_backend_pack",
    coverage_kind="backend_pack",
    runner="backend_property",
    backend_gate="xla",
    semantic_level=5,
    reason=(
        "XLA data propagation is owned by the PyTorch/XLA bridge and requires "
        "a real XLA runtime plus source-derived bridge contract; CPU/CUDA/MPS "
        "behavior cannot validate this surface."
    ),
    contract_status="blocked",
    contract_ref="docs/coverage/contract-evidence.md#xla-data-propagation-backend-pack",
    promotion_backend="xla",
))

for _surface in (
    "aten::fbgemm_linear_fp16_weight",
    "aten::fbgemm_linear_fp16_weight.out",
    "aten::fbgemm_linear_fp16_weight_fp32_activation",
    "aten::fbgemm_linear_fp16_weight_fp32_activation.out",
    "aten::fbgemm_linear_int8_weight",
    "aten::fbgemm_linear_int8_weight_fp32_activation",
    "aten::fbgemm_linear_quantize_weight",
    "aten::fbgemm_pack_gemm_matrix_fp16",
    "aten::fbgemm_pack_quantized_matrix",
    "aten::fbgemm_pack_quantized_matrix.KN",
):
    _register(OracleSpec(
        surface=_surface,
        oracle_id="fbgemm_linear_backend_pack",
        coverage_status=_coverage_status_for(_surface, _FBGEMM_PROMOTED),
        coverage_kind="backend_pack",
        runner="fbgemm_linear",
        backend_gate="fbgemm",
        semantic_level=5,
        reason=(
            "FBGEMM packed-linear helper has a source-derived candidate runner "
            "that uses PyTorch-created packed weights and compares values to "
            "dense/dequantized references."
        ),
        contract_status=_contract_status_for(_surface, _FBGEMM_PROMOTED),
        contract_ref="docs/coverage/contract-evidence.md#fbgemm-linear-backend-pack",
        promotion_evidence=_promotion_evidence_for(_surface, _FBGEMM_PROMOTED, _FBGEMM_EVIDENCE),
        promotion_backend="fbgemm",
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
        runner="fbgemm_wrapped_linear",
        backend_gate="fbgemm",
        semantic_level=5,
        reason=(
            "Wrapped quantized linear helper has a source-derived candidate runner "
            "that compares the AOTI wrapper path to public quantized linear with "
            "the same quantization parameters."
        ),
        contract_status="candidate",
        contract_ref="docs/coverage/contract-evidence.md#fbgemm-wrapped-quantized-linear-backend-pack",
        promotion_backend="fbgemm",
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
        contract_status="accepted",
        contract_ref="docs/coverage/contract-evidence.md#cpu-flash-attention-backend-pack",
        promotion_evidence="torchcts/selftest/test_harness_reporting.py::test_oracle_runner_executes_cpu_oracle_surfaces",
        promotion_backend="cpu_build",
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
        contract_status="accepted",
        contract_ref="docs/coverage/contract-evidence.md#mps-convolution-backend-pack",
        promotion_evidence="reviewed-local-mps-convolution-backend-pack-run",
        promotion_backend="mps",
    ))

_register(OracleSpec(
    surface="aten::mps_convolution_backward.out",
    oracle_id="mps_convolution_backward_out_blocked_schema",
    coverage_status="pending_backend_pack",
    coverage_kind="backend_pack",
    runner="backend_property",
    backend_gate="mps",
    semantic_level=5,
    reason=(
        "MPS convolution backward out overload has a CPU-reference contract, but "
        "PyTorch's direct Python binding expects an undefined third out tensor "
        "for the bias-gradient slot and rejects ordinary tensors, so TorchCTS "
        "does not have a safe direct invocation path."
    ),
    contract_status="blocked",
    contract_ref="docs/coverage/contract-evidence.md#mps-convolution-backend-pack",
    promotion_backend="mps",
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
    contract_status="accepted",
    contract_ref="docs/coverage/contract-evidence.md#mps-sdpa-math-backend-pack",
    promotion_evidence="reviewed-local-mps-sdpa-backend-pack-run",
    promotion_backend="mps",
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
        contract_status="accepted",
        contract_ref="docs/coverage/contract-evidence.md#mps-lstm-backend-pack",
        promotion_evidence="reviewed-local-mps-lstm-backend-pack-run",
        promotion_backend="mps",
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
        oracle_id="mps_philox_rng_backend_pack",
        coverage_status="pending_backend_pack",
        coverage_kind="backend_pack",
        runner="mps_philox",
        backend_gate="mps",
        semantic_level=5,
        reason=(
            "MPS Philox RNG helper has a source/schema-derived candidate runner "
            "checking key determinism, output identity, finite normal values, and "
            "uniform range bounds; current PyTorch MPS runtime reports the direct "
            "path is not currently supported."
        ),
        contract_status="candidate",
        contract_ref="docs/coverage/contract-evidence.md#mps-philox-rng-backend-pack",
        promotion_backend="mps",
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
        runner="fbgemm_quantized_cell",
        backend_gate="fbgemm",
        semantic_level=5,
        reason=(
            "Static quantized RNN cell has a source-derived candidate runner that "
            "builds legacy FBGEMM tensor-packed weights with PyTorch helpers and "
            "checks GRU/LSTM/RNN cell equations."
        ),
        contract_status="candidate",
        contract_ref="docs/coverage/contract-evidence.md#fbgemm-static-quantized-rnn-cell-backend-pack",
        promotion_backend="fbgemm",
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
        contract_status="accepted",
        contract_ref="docs/coverage/contract-evidence.md#mps-tinygemm-int4-pack-and-matmul-helpers",
        promotion_evidence="reviewed-local-mps-int4-backend-pack-run",
        promotion_backend="mps",
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
