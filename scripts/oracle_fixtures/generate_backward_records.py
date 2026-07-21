#!/usr/bin/env python3
"""Generate independent fixed records for backward references and routing."""

from __future__ import annotations

from fractions import Fraction
import json
import struct

import mpmath as mp


def _strides(shape):
    result, stride = [], 1
    for size in reversed(shape):
        result.append(stride)
        stride *= size
    return list(reversed(result))


def _f32_hex(value):
    return f"0x{struct.unpack('>I', struct.pack('>f', float(value)))[0]:08x}"


def _c64(values, shape):
    values = [complex(value) for value in values]
    return {
        "dtype": "torch.complex64", "shape": list(shape), "strides": _strides(shape),
        "storage_offset": 0, "layout": "torch.strided", "encoding": "complex_ieee754_bits",
        "real": [_f32_hex(value.real) for value in values],
        "imag": [_f32_hex(value.imag) for value in values],
    }


def _f32(values, shape):
    return {
        "dtype": "torch.float32", "shape": list(shape), "strides": _strides(shape),
        "storage_offset": 0, "layout": "torch.strided", "encoding": "ieee754_bits",
        "values": [_f32_hex(value) for value in values],
    }


def _i64(values, shape):
    return {
        "dtype": "torch.int64", "shape": list(shape), "strides": _strides(shape),
        "storage_offset": 0, "layout": "torch.strided", "encoding": "integer_decimal",
        "bit_width": 64, "signed": True, "values": [str(value) for value in values],
    }


def _linalg_backward():
    vander_input = [2, -1]
    vander_grad = [1] * 6
    vander_expected = [5, -1]
    exponent = [complex(1, 0.5), complex(-0.25, 1)]
    base = [complex(2, 1), complex(1.5, -0.5)]
    grad = [complex(1, 0), complex(0.5, -1)]
    with mp.workdps(100):
        exponent_grad, base_grad = [], []
        for e, b, g in zip(exponent, base, grad):
            em = mp.mpc(e.real, e.imag)
            bm = mp.mpc(b.real, b.imag)
            gm = mp.mpc(g.real, g.imag)
            output = mp.exp(em * mp.log(bm))
            exponent_grad.append(gm * mp.conj(output * mp.log(bm)))
            base_grad.append(gm * mp.conj(output * em / bm))
    return {
        "vander_input": _f32(vander_input, [2]),
        "vander_grad": _f32(vander_grad, [2, 3]),
        "vander_columns": 3,
        "vander_increasing": True,
        "vander_expected": _f32(vander_expected, [2]),
        "rpow_exponent": _c64(exponent, [2]),
        "rpow_base": _c64(base, [2]),
        "rpow_grad": _c64(grad, [2]),
        "rpow_exponent_expected": _c64(exponent_grad, [2]),
        "rpow_base_expected": _c64(base_grad, [2]),
        "cond_input": _f32([2, 0, 0, 1], [2, 2]),
        "cond_p": 2,
        "cond_grad": _f32([1], []),
        "cond_expected": _f32([1, 0, 0, -2], [2, 2]),
    }


def _group_norm_expected(values, grad, weight, groups, eps):
    channels = 2
    group_size = len(values) // groups
    input_grad = [mp.mpf(0)] * len(values)
    normalized = [mp.mpf(0)] * len(values)
    for group in range(groups):
        start, stop = group * group_size, (group + 1) * group_size
        row = list(map(mp.mpf, values[start:stop]))
        row_grad = [mp.mpf(grad[index]) * weight[index // 2] for index in range(start, stop)]
        mean = mp.fsum(row) / group_size
        variance = mp.fsum((value - mean) ** 2 for value in row) / group_size
        inv = 1 / mp.sqrt(variance + eps)
        norm = [(value - mean) * inv for value in row]
        for offset in range(group_size):
            normalized[start + offset] = norm[offset]
            input_grad[start + offset] = inv / group_size * (
                group_size * row_grad[offset]
                - mp.fsum(row_grad)
                - norm[offset] * mp.fsum(a * b for a, b in zip(row_grad, norm))
            )
    weight_grad = [
        mp.fsum(grad[index] * normalized[index] for index in range(channel * 2, channel * 2 + 2))
        for channel in range(channels)
    ]
    bias_grad = [sum(grad[channel * 2:channel * 2 + 2]) for channel in range(channels)]
    return input_grad, weight_grad, bias_grad


def _norm_sparse_backward():
    values = [1, 2, 3, 4]
    grad = [1, -1, 2, 3]
    weight = [2, 3]
    eps = mp.mpf("0.00001")
    with mp.workdps(100):
        input_grad, weight_grad, bias_grad = _group_norm_expected(values, grad, weight, 1, eps)
        denominator = mp.mpf(5) + mp.mpf("0.0000001")
        scale = mp.mpf(2) / denominator
        sensitivity = mp.mpf(4)
        contribution = -2 * sensitivity * mp.mpc(3, 4) / (denominator ** 2 * 5)
        renorm_expected = [scale + contribution, scale, 1, 1]
    matrix1 = [complex(1, 1), complex(2, 0), complex(0, 0), complex(1, -1)]
    matrix2 = [complex(2, 0), complex(1, 0), complex(0, 1), complex(-1, 0)]
    return {
        "group_input": _f32(values, [1, 2, 2]),
        "group_grad": _f32(grad, [1, 2, 2]),
        "group_weight": _f32(weight, [2]),
        "group_count": 1,
        "group_eps": float(eps),
        "group_input_expected": _f32(input_grad, [1, 2, 2]),
        "group_weight_expected": _f32(weight_grad, [2]),
        "group_bias_expected": _f32(bias_grad, [2]),
        "sampled_crow": _i64([0, 1, 2], [3]),
        "sampled_col": _i64([0, 1], [2]),
        "sampled_values": _c64([1, 1], [2]),
        "sampled_shape": [2, 2],
        "sampled_matrix1": _c64(matrix1, [2, 2]),
        "sampled_matrix2": _c64(matrix2, [2, 2]),
        "sampled_alpha": 2,
        "sampled_beta": 0.5,
        "sampled_input_expected": _c64([0.5, 0, 0, 0.5], [2, 2]),
        "sampled_left_expected": _c64([4, complex(0, -2), 2, -2], [2, 2]),
        "sampled_right_expected": _c64([complex(2, -2), 0, 4, complex(2, 2)], [2, 2]),
        "renorm_input": _c64([complex(3, 4), 1, 1, complex(0, 2)], [2, 2]),
        "renorm_grad": _c64([1, 1, 1, 1], [2, 2]),
        "renorm_dim": 0,
        "renorm_maxnorm": 2.0,
        "renorm_expected": _c64(renorm_expected, [2, 2]),
    }


def _routing():
    return {
        "has_reference": [
            {"op": "linalg.cond", "dtype": "torch.float32", "expected": True},
            {"op": "linalg.vander", "dtype": "torch.complex64", "expected": True},
            {"op": "__rpow__", "dtype": "torch.complex64", "expected": True},
            {"op": "__rpow__", "dtype": "torch.float32", "expected": False},
            {"op": "_segment_reduce", "dtype": "torch.bfloat16", "expected": True},
            {"op": "_segment_reduce", "dtype": "torch.float32", "expected": False},
            {"op": "unrelated", "dtype": "torch.float32", "expected": False},
        ]
    }


def main():
    records = {
        "CP-LINALG-BWD": _linalg_backward(),
        "CP-NORM-SPARSE-BWD": _norm_sparse_backward(),
        "CP-ROUTING": _routing(),
    }
    print(json.dumps({"schema_version": 1, "tools": {"mpmath": mp.__version__}, "records": records}, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


if __name__ == "__main__":
    main()
