#!/usr/bin/env python3
"""Generate independent frozen records for the former oracle self-tests.

The calculations use exact SymPy rationals, explicit IEEE lane formulas, and
scalar interpolation/convolution loops.  This script imports neither PyTorch
nor TorchCTS.  It writes canonical JSON to stdout and never edits accepted
fixtures.
"""

from __future__ import annotations

from fractions import Fraction
import json
import math
import struct

import sympy


def _f32(value) -> float:
    return struct.unpack(">f", struct.pack(">f", float(value)))[0]


def _f32_hex(value) -> str:
    return f"0x{struct.unpack('>I', struct.pack('>f', _f32(value)))[0]:08x}"


def _bf16_hex(value) -> str:
    bits = struct.unpack(">I", struct.pack(">f", _f32(value)))[0]
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return f"0x{(rounded >> 16) & 0xFFFF:04x}"


def _strides(shape: list[int]) -> list[int]:
    result = []
    stride = 1
    for size in reversed(shape):
        result.append(stride)
        stride *= size
    return list(reversed(result))


def _float_tensor(values, shape, *, dtype="torch.float32"):
    encoder = _f32_hex if dtype == "torch.float32" else _bf16_hex
    return {
        "dtype": dtype,
        "shape": shape,
        "strides": _strides(shape),
        "storage_offset": 0,
        "layout": "torch.strided",
        "encoding": "ieee754_bits",
        "values": [encoder(value) for value in values],
    }


def _complex_tensor(values, shape):
    return {
        "dtype": "torch.complex64",
        "shape": shape,
        "strides": _strides(shape),
        "storage_offset": 0,
        "layout": "torch.strided",
        "encoding": "complex_ieee754_bits",
        "real": [_f32_hex(real) for real, _imag in values],
        "imag": [_f32_hex(imag) for _real, imag in values],
    }


def _integer_tensor(values, shape, *, dtype="torch.int64"):
    bits = int(dtype.removeprefix("torch.int"))
    return {
        "dtype": dtype,
        "shape": shape,
        "strides": _strides(shape),
        "storage_offset": 0,
        "layout": "torch.strided",
        "encoding": "integer_decimal",
        "bit_width": bits,
        "signed": True,
        "values": [str(value) for value in values],
    }


def _lane_add(left, right, operation):
    if operation == "add":
        return (_f32(left[0] + right[0]), _f32(left[1] + right[1]))
    if operation == "sub":
        return (_f32(left[0] - right[0]), _f32(left[1] - right[1]))
    return (_f32(right[0] - left[0]), _f32(right[1] - left[1]))


def _complex_arithmetic():
    left = [(0.0, 0.5), (1.0, math.inf)]
    right = [(-math.inf, 0.25), (2.0, 3.0)]
    special_base = [(-math.inf, Fraction(1, 16)), (Fraction(5341, 5000), math.inf)]
    squared = []
    for real, imag in special_base:
        squared.append(
            (
                _f32(float(real) * float(real) - float(imag) * float(imag)),
                _f32(float(real) * float(imag) + float(imag) * float(real)),
            )
        )
    general_exponents = [
        (-1, 0),
        (1.5, 0),
        (2, 1),
        (math.nan, 0),
        (math.inf, 0),
        (2, math.nan),
        (2, math.inf),
        (-math.inf, 0),
    ]
    sentinel = [(11 + index, 1 + index) for index in range(8)]
    return {
        "unit_alpha": {
            "left": _complex_tensor(left, [2]),
            "right": _complex_tensor(right, [2]),
            "add": _complex_tensor([_lane_add(a, b, "add") for a, b in zip(left, right)], [2]),
            "sub": _complex_tensor([_lane_add(a, b, "sub") for a, b in zip(left, right)], [2]),
            "rsub": _complex_tensor([_lane_add(a, b, "rsub") for a, b in zip(left, right)], [2]),
        },
        "integer_power": {
            "base": _complex_tensor(special_base, [2]),
            "exponent": _complex_tensor([(2, 0), (2, 0)], [2]),
            "native_placeholder": _complex_tensor([(math.nan, math.nan), (math.nan, math.nan)], [2]),
            "expected": _complex_tensor(squared, [2]),
        },
        "general_power_passthrough": {
            "base": _complex_tensor([(2, 1)] * 8, [8]),
            "exponent": _complex_tensor(general_exponents, [8]),
            "native_sentinel": _complex_tensor(sentinel, [8]),
        },
    }


def _complex_unary():
    phase = sympy.N(sympy.pi / sympy.log(2), 80)
    return {
        "input": _complex_tensor([(math.inf, 1), (-math.inf, 1)], [2]),
        "expected": _complex_tensor([(math.inf, 0), (math.inf, float(phase))], [2]),
        "exact_expression": ["+infinity + 0*i", "+infinity + (pi/log(2))*i"],
        "phase_decimal_80": str(phase),
    }


def _complex_loss():
    return {
        "input": _complex_tensor([(3, 4), (math.inf, 1)], [2]),
        "target": _float_tensor([0, 2], [2]),
        "expected": _float_tensor([5, math.inf], [2]),
    }


def _embedding():
    grad = [(Fraction(1), Fraction(-1, 2)), (Fraction(-1, 4), Fraction(2))]
    indices = [1, 1, 2, 1, 2, 4]
    bags = [0, 0, 0, 1, 1, 1]
    counts = {row: indices.count(row) for row in set(indices)}
    result = [[Fraction(0), Fraction(0)] for _ in range(5)]
    for row, bag in zip(indices, bags):
        for lane in range(2):
            result[row][lane] += grad[bag][lane] / counts[row]
    return {
        "grad": _float_tensor([value for row in grad for value in row], [2, 2]),
        "indices": _integer_tensor(indices, [6]),
        "offset2bag": _integer_tensor(bags, [6]),
        "num_weights": 5,
        "row_frequencies": {str(row): count for row, count in sorted(counts.items())},
        "expected": _float_tensor([value for row in result for value in row], [5, 2]),
    }


def _reflect(pixel: Fraction, size: int) -> tuple[Fraction, int]:
    high = Fraction(size - 1)
    period = high * 2
    folded = pixel % period
    if folded <= high:
        return folded, 1
    return period - folded, -1


def _grid_backward():
    input_values = [Fraction(value) for value in range(8)]
    coordinates = [
        (Fraction(-5, 4), Fraction(-1, 2), Fraction(1, 4)),
        (Fraction(3, 4), Fraction(5, 4), Fraction(-3, 4)),
    ]
    grad_output = [Fraction(3, 4), Fraction(-5, 4)]
    grad_input = [Fraction(0) for _ in range(8)]
    grad_grid = []
    contributor_records = []
    for coordinate, upstream in zip(coordinates, grad_output):
        reflected = []
        normalized_derivatives = []
        for lane in coordinate:
            pixel = (lane + 1) / 2
            reflected_pixel, reflection_sign = _reflect(pixel, 2)
            reflected.append(reflected_pixel)
            normalized_derivatives.append(Fraction(reflection_sign, 2))
        px, py, pz = reflected
        point_contributors = []
        derivative_x = derivative_y = derivative_z = Fraction(0)
        for z in (0, 1):
            wz = pz if z else 1 - pz
            dwz = 1 if z else -1
            for y in (0, 1):
                wy = py if y else 1 - py
                dwy = 1 if y else -1
                for x in (0, 1):
                    wx = px if x else 1 - px
                    dwx = 1 if x else -1
                    weight = wx * wy * wz
                    flat_index = z * 4 + y * 2 + x
                    value = input_values[flat_index]
                    grad_input[flat_index] += upstream * weight
                    derivative_x += value * dwx * wy * wz
                    derivative_y += value * wx * dwy * wz
                    derivative_z += value * wx * wy * dwz
                    point_contributors.append(
                        {"index_zyx": [z, y, x], "weight": str(weight), "value": str(value)}
                    )
        grad_grid.extend(
            [
                upstream * derivative_x * normalized_derivatives[0],
                upstream * derivative_y * normalized_derivatives[1],
                upstream * derivative_z * normalized_derivatives[2],
            ]
        )
        contributor_records.append(point_contributors)
    return {
        "input": _float_tensor(input_values, [1, 1, 2, 2, 2], dtype="torch.bfloat16"),
        "grid": _float_tensor(
            [lane for coordinate in coordinates for lane in coordinate],
            [1, 1, 1, 2, 3],
            dtype="torch.bfloat16",
        ),
        "grad_output": _float_tensor(grad_output, [1, 1, 1, 1, 2], dtype="torch.bfloat16"),
        "interpolation_mode": 0,
        "padding_mode": 2,
        "align_corners": True,
        "contributors": contributor_records,
        "expected_grad_input": _float_tensor(
            grad_input, [1, 1, 2, 2, 2], dtype="torch.bfloat16"
        ),
        "expected_grad_grid": _float_tensor(
            grad_grid, [1, 1, 1, 2, 3], dtype="torch.bfloat16"
        ),
    }


def _complex_multiply(left, right):
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _complex_convolution():
    input_values = [(math.inf, Fraction(1, 4)), (1, -2)]
    weight_values = [(2, 3), (-1, Fraction(1, 2))]
    bias = (Fraction(1, 4), Fraction(-1, 2))
    output = [bias, bias, bias]
    for input_index, input_value in enumerate(input_values):
        for kernel_index, weight_value in enumerate(weight_values):
            product = _complex_multiply(input_value, weight_value)
            destination = input_index + kernel_index
            output[destination] = (
                output[destination][0] + product[0],
                output[destination][1] + product[1],
            )
    return {
        "op_name": "nn.functional.conv_transpose1d",
        "input": _complex_tensor(input_values, [1, 1, 2]),
        "weight": _complex_tensor(weight_values, [1, 1, 2]),
        "bias": _complex_tensor([bias], [1]),
        "kwargs": {
            "stride": [1],
            "padding": [0],
            "output_padding": [0],
            "groups": 1,
            "dilation": [1],
        },
        "expected": _complex_tensor(output, [1, 1, 3]),
        "term_map": [
            {"output": 0, "terms": [[0, 0]]},
            {"output": 1, "terms": [[0, 1], [1, 0]]},
            {"output": 2, "terms": [[1, 1]]},
        ],
    }


def main() -> None:
    payload = {
        "schema_version": 1,
        "generator": "scripts/oracle_fixtures/generate_migrated_selftest_records.py",
        "tools": {"python_stdlib": "struct/fractions/math", "sympy": sympy.__version__},
        "records": {
            "complex_arithmetic": _complex_arithmetic(),
            "complex_unary": _complex_unary(),
            "complex_loss": _complex_loss(),
            "embedding": _embedding(),
            "grid_backward": _grid_backward(),
            "complex_convolution": _complex_convolution(),
        },
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


if __name__ == "__main__":
    main()
