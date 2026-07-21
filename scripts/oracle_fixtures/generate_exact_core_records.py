#!/usr/bin/env python3
"""Generate exact/formula records for TorchCTS core references.

No PyTorch or TorchCTS code is imported.  Integer/layout cases use explicit
loops; real and complex finite cases use SymPy rationals or mpmath before one
declared public-dtype rounding step.
"""

from __future__ import annotations

from fractions import Fraction
import json
import math
import struct

import mpmath as mp
import sympy


def _f32(value):
    return struct.unpack(">f", struct.pack(">f", float(value)))[0]


def _f32_hex(value):
    return f"0x{struct.unpack('>I', struct.pack('>f', _f32(value)))[0]:08x}"


def _bf16_hex(value):
    bits = struct.unpack(">I", struct.pack(">f", _f32(value)))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return f"0x{(bits >> 16) & 0xffff:04x}"


def _strides(shape):
    result, stride = [], 1
    for size in reversed(shape):
        result.append(stride)
        stride *= size
    return list(reversed(result))


def _float(values, shape, dtype="torch.float32"):
    encode = _bf16_hex if dtype == "torch.bfloat16" else _f32_hex
    return {
        "dtype": dtype,
        "shape": list(shape),
        "strides": _strides(shape),
        "storage_offset": 0,
        "layout": "torch.strided",
        "encoding": "ieee754_bits",
        "values": [encode(value) for value in values],
    }


def _complex(values, shape):
    return {
        "dtype": "torch.complex64",
        "shape": list(shape),
        "strides": _strides(shape),
        "storage_offset": 0,
        "layout": "torch.strided",
        "encoding": "complex_ieee754_bits",
        "real": [_f32_hex(value[0]) for value in values],
        "imag": [_f32_hex(value[1]) for value in values],
    }


def _integer(values, shape, dtype="torch.int64"):
    unsigned = dtype.startswith("torch.uint")
    if dtype == "torch.bool":
        return {
            "dtype": dtype,
            "shape": list(shape),
            "strides": _strides(shape),
            "storage_offset": 0,
            "layout": "torch.strided",
            "encoding": "boolean",
            "values": [bool(value) for value in values],
        }
    bits = int(dtype.removeprefix("torch.uint" if unsigned else "torch.int"))
    return {
        "dtype": dtype,
        "shape": list(shape),
        "strides": _strides(shape),
        "storage_offset": 0,
        "layout": "torch.strided",
        "encoding": "integer_decimal",
        "bit_width": bits,
        "signed": not unsigned,
        "values": [str(int(value)) for value in values],
    }


def _cadd(left, right):
    return (left[0] + right[0], left[1] + right[1])


def _cmul(left, right):
    return (left[0] * right[0] - left[1] * right[1], left[0] * right[1] + left[1] * right[0])


def _matmul(left, right, m, k, n):
    output = []
    for row in range(m):
        for column in range(n):
            value = (Fraction(0), Fraction(0))
            for inner in range(k):
                value = _cadd(value, _cmul(left[row * k + inner], right[inner * n + column]))
            output.append(value)
    return output


def _cast_record():
    opmath_input = [Fraction(3, 2), Fraction(-9, 4), Fraction(257, 256)]
    public_input = [Fraction(257, 256), Fraction(-513, 256), Fraction(1, 3)]
    saturation = [-70000, -65504, 0, 65504, 70000]
    return {
        "opmath_input": _float(opmath_input, [3], "torch.bfloat16"),
        "opmath_expected": _float([Fraction(3, 2), Fraction(-9, 4), Fraction(1)], [3]),
        "public_input": _float(public_input, [3]),
        "public_bfloat16_expected": _float(public_input, [3], "torch.bfloat16"),
        "saturation_input": _float(saturation, [5]),
        "saturation_expected": _float([-65504, -65504, 0, 65504, 65504], [5]),
    }


def _matmul_record():
    left = [(1, 2), (-3, Fraction(1, 2)), (0, 0), (2, -1)]
    right = [(2, -1), (0, 3), (-1, 0), (4, 2)]
    output = _matmul(left, right, 2, 2, 2)
    return {
        "dispatcher": "aten::matmul",
        "left": _complex(left, [2, 2]),
        "right": _complex(right, [2, 2]),
        "expected": _complex(output, [2, 2]),
        "determinate_mask": _integer([True] * 8, [2, 2, 2], "torch.bool"),
        "exact_rational_output": [[str(real), str(imag)] for real, imag in output],
    }


def _softmargin_record():
    x = [-10, -1, 0, 1, 10]
    target = [1, -1, 1, -1, 1]
    grad = [1, -2, Fraction(1, 2), 1, 3]
    with mp.workdps(100):
        loss = [mp.log1p(mp.exp(-mp.mpf(a) * mp.mpf(b))) for a, b in zip(x, target)]
        derivative = [-mp.mpf(b) / (1 + mp.exp(mp.mpf(a) * mp.mpf(b))) for a, b in zip(x, target)]
        backward = [
            d * (mp.mpf(g.numerator) / g.denominator if isinstance(g, Fraction) else mp.mpf(g))
            for d, g in zip(derivative, grad)
        ]
    return {
        "input": _float(x, [5]),
        "target": _float(target, [5]),
        "grad": _float(grad, [5]),
        "loss_none": _float(loss, [5]),
        "loss_sum": _float([sum(loss)], []),
        "loss_mean": _float([sum(loss) / len(loss)], []),
        "backward_none": _float(backward, [5]),
        "backward_mean": _float([value / len(x) for value in backward], [5]),
        "decimal_loss": [mp.nstr(value, 80) for value in loss],
    }


def _segment_record():
    data = [2, 0, 3, 0, -4, 5]
    lengths = [3, 0, 3]
    initial = 2
    forward, backward = [], [Fraction(0) for _ in data]
    cursor = 0
    upstream = [1, 2, 3]
    for segment, length in enumerate(lengths):
        values = data[cursor : cursor + length]
        product = Fraction(initial)
        for value in values:
            product *= value
        forward.append(product)
        for index in range(length):
            derivative = Fraction(initial)
            for other, value in enumerate(values):
                if other != index:
                    derivative *= value
            backward[cursor + index] = derivative * upstream[segment]
        cursor += length
    forward[-1] = -0.0
    return {
        "data": _float(data, [6]),
        "lengths": _integer(lengths, [3]),
        "offsets": _integer([0, 3, 3, 6], [4]),
        "initial": initial,
        "grad": _float(upstream, [3]),
        "expected_forward": _float(forward, [3]),
        "expected_backward": _float(backward, [6]),
        "no_zero_data": _float([2, 3, 4], [3]),
        "no_zero_lengths": _integer([3], [1]),
        "no_zero_forward": _float([24], [1]),
        "no_zero_backward": _float([12, 8, 6], [3]),
    }


def _linear_record():
    x = [[1, -2, 3], [Fraction(1, 2), 0, -1]]
    weight = [[2, -1, Fraction(1, 2)], [-3, 4, 1]]
    bias = [Fraction(1, 4), Fraction(-1, 2)]
    grad = [[1, -2], [Fraction(1, 2), 3]]
    dx = [[sum(grad[row][out] * weight[out][inner] for out in range(2)) for inner in range(3)] for row in range(2)]
    dw = [[sum(grad[row][out] * x[row][inner] for row in range(2)) for inner in range(3)] for out in range(2)]
    db = [sum(grad[row][out] for row in range(2)) for out in range(2)]
    flat = lambda rows: [value for row in rows for value in row]
    return {
        "input": _float(flat(x), [2, 3]),
        "weight": _float(flat(weight), [2, 3]),
        "bias": _float(bias, [2]),
        "grad_output": _float(flat(grad), [2, 2]),
        "expected_input_grad": _float(flat(dx), [2, 3]),
        "expected_weight_grad": _float(flat(dw), [2, 3]),
        "expected_bias_grad": _float(db, [2]),
    }


def _pool_record():
    values = [2, 2, -1, 4, 0, 3, 3, -2, 5, 1, 5, 0]
    height, width = 3, 4
    grad_output = [1, 2, 3, 4, 5, 6]
    grad_input = [0] * len(values)
    winners = []
    output_index = 0
    for oy in range(2):
        for ox in range(3):
            candidates = [(values[(oy + ky) * width + ox + kx], oy + ky, ox + kx) for ky in range(2) for kx in range(2)]
            maximum = max(value for value, _y, _x in candidates)
            _value, winner_y, winner_x = next(item for item in candidates if item[0] == maximum)
            winners.append([winner_y, winner_x])
            grad_input[winner_y * width + winner_x] += grad_output[output_index]
            output_index += 1
    return {
        "input": _float(values, [1, 1, height, width]),
        "grad_output": _float(grad_output, [1, 1, 2, 3]),
        "parameters": {"kernel_size": [2, 2], "stride": [1, 1], "padding": [0, 0], "dilation": [1, 1], "ceil_mode": False},
        "winner_indices_yx": winners,
        "expected_input_grad": _float(grad_input, [1, 1, height, width]),
    }


def _int4_record():
    logical = [list(range(8)), list(range(8, 16))]
    packed_high, packed_low = [], []
    for row in logical:
        packed_high.extend([(row[index] << 4) | row[index + 1] for index in range(0, 8, 2)])
        packed_low.extend([row[index] | (row[index + 1] << 4) for index in range(0, 8, 2)])
    qparams = [
        (Fraction(1, 2), 1), (2, -1),
        (1, 0), (Fraction(-1, 2), 2),
    ]
    dequant = [[Fraction(0) for _ in range(8)] for _ in range(2)]
    for group in range(2):
        for out in range(2):
            scale, zero = qparams[group * 2 + out]
            for k in range(group * 4, group * 4 + 4):
                dequant[out][k] = (logical[out][k] - 8) * scale + zero
    input_values = [1, -2, Fraction(1, 2), 3, -1, 2, 0, Fraction(-1, 2)]
    mm = [sum(input_values[k] * dequant[out][k] for k in range(8)) for out in range(2)]
    dynamic_scales = [[Fraction(1, 2), 1], [2, Fraction(-1, 2)]]
    dynamic = [[(logical[out][k] - 8) * dynamic_scales[out][k // 4] for k in range(8)] for out in range(2)]
    bias = [Fraction(1, 4), -1]
    dynamic_mm = [sum(input_values[k] * dynamic[out][k] for k in range(8)) + bias[out] for out in range(2)]
    flat = lambda rows: [value for row in rows for value in row]
    return {
        "logical": _integer(flat(logical), [2, 8], "torch.int32"),
        "packed_even_high": _integer(packed_high, [2, 4], "torch.uint8"),
        "packed_even_low": _integer(packed_low, [2, 4], "torch.uint8"),
        "qparams": _float([value for pair in qparams for value in pair], [2, 2, 2]),
        "group_size": 4,
        "expected_dequant": _float(flat(dequant), [2, 8]),
        "input": _float(input_values, [1, 8]),
        "expected_matmul": _float(mm, [1, 2]),
        "dynamic_packed": _integer(packed_low, [2, 4], "torch.uint8"),
        "dynamic_unpacked": _float(flat(logical), [2, 8]),
        "dynamic_scales": _float(flat(dynamic_scales), [2, 2]),
        "dynamic_bias": _float(bias, [2]),
        "expected_dynamic_dequant": _float(flat(dynamic), [2, 8]),
        "expected_dynamic_matmul": _float(dynamic_mm, [1, 2]),
    }


def _quant_record():
    x = [[1, -2, Fraction(1, 2)], [0, 3, -1]]
    weight = [[2, -1, 4], [-3, 2, 1]]
    scales = [Fraction(1, 2), 2]
    result = [[sum(x[row][k] * weight[out][k] for k in range(3)) * scales[out] for out in range(2)] for row in range(2)]
    flat = lambda rows: [value for row in rows for value in row]
    return {"input": _float(flat(x), [2, 3]), "weight": _integer(flat(weight), [2, 3], "torch.int8"), "scales": _float(scales, [2]), "expected": _float(flat(result), [2, 2])}


def _histc_record():
    values = [-1, 0, 0, Fraction(1, 2), Fraction(1, 2), 1, 1, 2]
    def counts(bins, low, high):
        output = [0] * bins
        for value in values:
            if value < low or value > high:
                continue
            index = bins - 1 if value == high else int((value - low) / (high - low) * bins)
            output[index] += 1
        return output
    return {"input": _float(values, [8]), "minimum": 0.0, "maximum": 1.0, "bins_1": _float(counts(1, 0, 1), [1]), "bins_2": _float(counts(2, 0, 1), [2]), "bins_4": _float(counts(4, 0, 1), [4])}


def _im2col_record():
    image = list(range(12))
    columns = []
    for ky in range(2):
        for kx in range(2):
            for oy in range(2):
                for ox in range(3):
                    columns.append(image[(oy + ky) * 4 + ox + kx])
    reconstructed = [0] * 12
    cursor = 0
    for ky in range(2):
        for kx in range(2):
            for oy in range(2):
                for ox in range(3):
                    reconstructed[(oy + ky) * 4 + ox + kx] += columns[cursor]
                    cursor += 1
    params = {"kernel_size": [2, 2], "dilation": [1, 1], "padding": [0, 0], "stride": [1, 1]}
    return {"input": _float(image, [1, 1, 3, 4]), "parameters": params, "columns": _float(columns, [1, 4, 6]), "output_size": [3, 4], "reconstructed": _float(reconstructed, [1, 1, 3, 4])}


def _logit_record():
    inputs = [Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(-1, 10), Fraction(11, 10)]
    grad = [1, -1, 2, 3, 4]
    no_eps = [g / (x * (1 - x)) for x, g in zip(inputs, grad)]
    eps = [Fraction(0) if x < Fraction(1, 10) or x > Fraction(9, 10) else g / (x * (1 - x)) for x, g in zip(inputs, grad)]
    return {"input": _float(inputs, [5]), "grad": _float(grad, [5]), "expected_none": _float(no_eps, [5]), "epsilon": 0.1, "expected_eps": _float(eps, [5])}


def _polynomial_record():
    x_values = [Fraction(-1), Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(1)]
    degrees = [0, 1, 2, 3, 4]
    laguerre, families = [], {name: [] for name in "tuvw"}
    for x, degree in zip(x_values, degrees):
        if degree == 0:
            laguerre.append(Fraction(1))
        else:
            previous, current = Fraction(1), 1 - x
            for n in range(2, degree + 1):
                previous, current = current, ((2 * n - 1 - x) * current - (n - 1) * previous) / n
            laguerre.append(current)
        shifted = 2 * x - 1
        starts = {"t": shifted, "u": 2 * shifted, "v": 2 * shifted - 1, "w": 2 * shifted + 1}
        for family, first in starts.items():
            if degree == 0:
                value = Fraction(1)
            else:
                previous, value = Fraction(1), first
                for _ in range(2, degree + 1):
                    previous, value = value, 2 * shifted * value - previous
            families[family].append(value)
    families["t"][3] = -0.0
    families["u"][3] = -0.0
    return {"input": _float(x_values, [5]), "degrees": _integer(degrees, [5]), "laguerre": _float(laguerre, [5]), "shifted": {family: _float(values, [5]) for family, values in families.items()}}


def main():
    records = {
        "CP-CAST": _cast_record(),
        "CP-MATMUL": _matmul_record(),
        "CP-SOFTMARGIN": _softmargin_record(),
        "CP-SEGMENT": _segment_record(),
        "CP-LINEAR-BWD": _linear_record(),
        "CP-POOL": _pool_record(),
        "CP-INT4": _int4_record(),
        "CP-QUANT": _quant_record(),
        "CP-HISTC": _histc_record(),
        "CP-IM2COL": _im2col_record(),
        "CP-LOGIT": _logit_record(),
        "CP-POLYNOMIAL": _polynomial_record(),
    }
    payload = {"schema_version": 1, "generator": "scripts/oracle_fixtures/generate_exact_core_records.py", "tools": {"mpmath": mp.__version__, "sympy": sympy.__version__}, "records": records}
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


if __name__ == "__main__":
    main()
