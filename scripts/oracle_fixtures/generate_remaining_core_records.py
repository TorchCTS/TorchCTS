#!/usr/bin/env python3
"""Generate independent records for the remaining core numerical references.

This generator imports neither PyTorch nor TorchCTS.  It uses literal scalar
loops, exact rational arithmetic, and 100-digit mpmath formulas before an
explicit public-dtype rounding step.
"""

from __future__ import annotations

from fractions import Fraction
import cmath
import json
import math
import struct

import mpmath as mp


def _mpf(value):
    if isinstance(value, Fraction):
        return mp.mpf(value.numerator) / value.denominator
    return mp.mpf(value)


def _round(value, code):
    return struct.unpack(">" + code, struct.pack(">" + code, float(value)))[0]


def _hex(value, code, width):
    rounded = _round(value, code)
    integer_code = "I" if code == "f" else "Q"
    bits = struct.unpack(">" + integer_code, struct.pack(">" + code, rounded))[0]
    return f"0x{bits:0{width}x}"


def _strides(shape):
    result, stride = [], 1
    for size in reversed(shape):
        result.append(stride)
        stride *= size
    return list(reversed(result))


def _float(values, shape, dtype="torch.float32"):
    code, width = ("d", 16) if dtype == "torch.float64" else ("f", 8)
    return {
        "dtype": dtype,
        "shape": list(shape),
        "strides": _strides(shape),
        "storage_offset": 0,
        "layout": "torch.strided",
        "encoding": "ieee754_bits",
        "values": [_hex(value, code, width) for value in values],
    }


def _complex(values, shape, dtype="torch.complex64"):
    code, width = ("d", 16) if dtype == "torch.complex128" else ("f", 8)
    normalized = [complex(value) for value in values]
    return {
        "dtype": dtype,
        "shape": list(shape),
        "strides": _strides(shape),
        "storage_offset": 0,
        "layout": "torch.strided",
        "encoding": "complex_ieee754_bits",
        "real": [_hex(value.real, code, width) for value in normalized],
        "imag": [_hex(value.imag, code, width) for value in normalized],
    }


def _integer(values, shape, dtype="torch.int64"):
    bits = int(dtype.removeprefix("torch.int"))
    return {
        "dtype": dtype,
        "shape": list(shape),
        "strides": _strides(shape),
        "storage_offset": 0,
        "layout": "torch.strided",
        "encoding": "integer_decimal",
        "bit_width": bits,
        "signed": True,
        "values": [str(int(value)) for value in values],
    }


def _complex_arithmetic():
    left = [complex(1, 2), complex(-3, 0.5), complex(0.25, -2)]
    right = [complex(2, -1), complex(0.5, 4), complex(-1, -1)]
    multiplied = [a * b for a, b in zip(left, right)]
    foreach_base = [complex(1, 2), complex(-1, 0.5)]
    foreach_other = [complex(3, -1), complex(2, 4)]
    foreach_add = [a + Fraction(1, 2) * b for a, b in zip(foreach_base, foreach_other)]
    foreach_mul = [a + b * c for a, b, c in zip(foreach_base, left[:2], right[:2])]
    exponent_cases = [
        (complex(2, 0), True),
        (complex(-1, 0), False),
        (complex(1.5, 0), False),
        (complex(3, 1), False),
    ]
    return {
        "left": _complex(left, [3]),
        "right": _complex(right, [3]),
        "multiplied": _complex(multiplied, [3]),
        "exponent_cases": [
            {"input": _complex([value], [1]), "expected": expected}
            for value, expected in exponent_cases
        ],
        "foreach_base": _complex(foreach_base, [2]),
        "foreach_other": _complex(foreach_other, [2]),
        "foreach_add_alpha_half": _complex(foreach_add, [2]),
        "foreach_addcmul": _complex(foreach_mul, [2]),
    }


def _complex_loss():
    logits = [-2, 0, 2]
    target = [0, Fraction(1, 2), 1]
    weight = [1, 2, Fraction(1, 2)]
    pos_weight = [1, 3, 2]
    with mp.workdps(100):
        values = []
        for x, y, w, p in zip(logits, target, weight, pos_weight):
            xv, yv, wv, pv = map(_mpf, (x, y, w, p))
            positive = yv * pv * mp.log1p(mp.exp(-xv))
            negative = (1 - yv) * mp.log1p(mp.exp(xv))
            values.append(wv * (positive + negative))
    return {
        "logits": _float(logits, [3]),
        "target": _float(target, [3]),
        "weight": _float(weight, [3]),
        "pos_weight": _float(pos_weight, [3]),
        "expected_none": _float(values, [3]),
        "expected_sum": _float([sum(values)], []),
        "expected_mean": _float([sum(values) / 3], []),
        "decimal": [mp.nstr(value, 80) for value in values],
    }


def _complex_unary():
    inputs = [complex(0.5, 0.25), complex(-1, 0.75), complex(2, -1.5)]
    with mp.workdps(100):
        values = [mp.mpc(value.real, value.imag) for value in inputs]
        sigmoid = [1 / (1 + mp.exp(-value)) for value in values]
        rsqrt = [1 / mp.sqrt(value) for value in values]
        expm1 = [mp.exp(value) - 1 for value in values]
    return {
        "input": _complex(inputs, [3]),
        "sigmoid": _complex(sigmoid, [3]),
        "rsqrt": _complex(rsqrt, [3]),
        "expm1": _complex(expm1, [3]),
    }


def _ldexp_cumprod():
    values = [complex(1, 2), complex(-0.5, 1), complex(2, -1)]
    powers = [complex(2, 0.5), complex(-1, -0.25), complex(0.5, 1)]
    with mp.workdps(100):
        expected = [
            mp.mpc(value.real, value.imag)
            * mp.exp(mp.mpc(power.real, power.imag) * mp.log(2))
            for value, power in zip(values, powers)
        ]
    integral_powers = [2, -1, 3]
    integral = [
        complex(math.ldexp(value.real, power), math.ldexp(value.imag, power))
        for value, power in zip(values, integral_powers)
    ]
    accumulated, running = [], complex(1, 0)
    for value in values:
        running *= value
        accumulated.append(running)
    wide_input = [1.5, -2.0, 0.0, -0.0]
    wide_powers = [3, -2, 2**40, -(2**40)]
    wide_expected = [12.0, -0.5, 0.0, -0.0]
    return {
        "input": _complex(values, [3]),
        "complex_exponent": _complex(powers, [3]),
        "complex_expected": _complex(expected, [3]),
        "integral_exponent": _integer(integral_powers, [3]),
        "integral_expected": _complex(integral, [3]),
        "cumprod_expected": _complex(accumulated, [3]),
        "wide_input": _float(wide_input, [4]),
        "wide_exponent": _integer(wide_powers, [4]),
        "wide_expected": _float(wide_expected, [4]),
    }


def _gradient_covariance():
    gradient_input = [complex(1, 0), complex(3, 2), complex(7, 6), complex(13, 12)]
    gradient = [complex(2, 2), complex(3, 3), complex(5, 5), complex(6, 6)]
    rows = [
        [complex(1, 1), complex(2, 0), complex(3, -1)],
        [complex(0, 2), complex(1, 1), complex(2, 0)],
    ]
    means = [sum(row) / 3 for row in rows]
    covariance = []
    for left, left_mean in zip(rows, means):
        for right, right_mean in zip(rows, means):
            value = sum(
                (a - left_mean) * (b - right_mean).conjugate()
                for a, b in zip(left, right)
            ) / 2
            covariance.append(complex(value.real, 0) if left is right else value)
    return {
        "gradient_input": _complex(gradient_input, [4]),
        "spacing": 1.0,
        "edge_order": 1,
        "gradient_expected": _complex(gradient, [4]),
        "covariance_input": _complex([value for row in rows for value in row], [2, 3]),
        "correction": 1,
        "covariance_expected": _complex(covariance, [2, 2]),
    }


def _lanczos_coefficient(distance):
    if abs(distance) >= 3 or (distance != 0 and float(distance).is_integer()):
        return 0.0
    if distance == 0:
        return 1.0
    distance = _mpf(distance)
    return mp.sin(mp.pi * distance) / (mp.pi * distance) * mp.sin(mp.pi * distance / 3) / (mp.pi * distance / 3)


def _lanczos_weights(input_size, output_size):
    scale = Fraction(input_size, output_size)
    support = 3 * scale if scale >= 1 else Fraction(3)
    inverse_scale = 1 / scale if scale >= 1 else Fraction(1)
    result = []
    for output_index in range(output_size):
        center = scale * Fraction(2 * output_index + 1, 2)
        first = max(math.floor(center - support + Fraction(1, 2)), 0)
        count = min(math.floor(center + support + Fraction(1, 2)), input_size) - first
        weights = [
            _lanczos_coefficient((source - center + Fraction(1, 2)) * inverse_scale)
            for source in range(first, first + count)
        ]
        total = mp.fsum(weights)
        result.append([(source, weight / total) for source, weight in zip(range(first, first + count), weights)])
    return result


def _lanczos():
    distances = [0, Fraction(1, 2), 1, Fraction(5, 4), 3]
    image = [1, 2, 3, 4, 5, 6]
    height_weights = _lanczos_weights(2, 1)
    width_weights = _lanczos_weights(3, 2)
    output = []
    for y_terms in height_weights:
        for x_terms in width_weights:
            output.append(mp.fsum(image[y * 3 + x] * yw * xw for y, yw in y_terms for x, xw in x_terms if yw and xw))
    return {
        "distances": _float(distances, [5], "torch.float64"),
        "coefficients": _float([_lanczos_coefficient(value) for value in distances], [5], "torch.float64"),
        "input": _float(image, [1, 1, 2, 3], "torch.float64"),
        "output_size": [1, 2],
        "align_corners": False,
        "output": _float(output, [1, 1, 1, 2], "torch.float64"),
    }


def _grid():
    return {
        "input": _float([1, 2, 3, 4], [1, 1, 2, 2]),
        "grid": _float([0, 0, -1, -1], [1, 1, 2, 2]),
        "grad_output": _float([2, 3], [1, 1, 1, 2]),
        "interpolation_mode": 0,
        "padding_mode": 0,
        "align_corners": True,
        "forward": _float([Fraction(5, 2), 1], [1, 1, 1, 2]),
        "grad_input": _float([Fraction(7, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)], [1, 1, 2, 2]),
        "grad_grid": _float([1, 2, Fraction(3, 2), 3], [1, 1, 2, 2]),
    }


def _convolution():
    return {
        "transpose1d_input": _float([1, 2], [1, 1, 2]),
        "transpose1d_weight": _float([3, 4], [1, 1, 2]),
        "transpose1d_bias": _float([1], [1]),
        "transpose1d_expected": _float([4, 11, 9], [1, 1, 3]),
        "transpose1d_kwargs": {"stride": 1, "padding": 0, "output_padding": 0, "groups": 1, "dilation": 1},
        "transpose3d_input": _float([2], [1, 1, 1, 1, 1]),
        "transpose3d_weight": _float([3], [1, 1, 1, 1, 1]),
        "transpose3d_bias": _float([1], [1]),
        "transpose3d_expected": _float([7], [1, 1, 1, 1, 1]),
        "transpose3d_kwargs": {"stride": 1, "padding": 0, "output_padding": 0, "groups": 1, "dilation": 1},
    }


def _matrix_exp():
    with mp.workdps(100):
        values = [mp.e, 0, 0, mp.e ** -1]
    return {
        "input": _float([1, 0, 0, -1], [2, 2], "torch.float64"),
        "expected": _float(values, [2, 2], "torch.float64"),
        "decimal": [mp.nstr(value, 80) for value in values],
    }


def _special():
    with mp.workdps(100):
        dirichlet_inputs = [(mp.mpf("0.25"), mp.mpf("1.5"), mp.mpf("4")), (mp.mpf("0.6"), mp.mpf("2"), mp.mpf("5"))]
        dirichlet = []
        for x, alpha, total in dirichlet_inputs:
            beta = total - alpha
            derivative = mp.diff(lambda shape: mp.betainc(shape, beta, 0, x, regularized=True), alpha)
            density = x ** (alpha - 1) * (1 - x) ** (beta - 1) / mp.beta(alpha, beta)
            dirichlet.append(-derivative / (density * (1 - x)))
        gamma_inputs = [(mp.mpf("1.5"), mp.mpf("0.75")), (mp.mpf("3"), mp.mpf("2"))]
        gamma_grad = []
        for alpha, output in gamma_inputs:
            derivative = mp.diff(lambda shape: mp.gammainc(shape, 0, output, regularized=True), alpha)
            density = output ** (alpha - 1) * mp.exp(-output) / mp.gamma(alpha)
            gamma_grad.append(-derivative / density)
        i0_input = [mp.mpf("0"), mp.mpf("1"), mp.mpf("3.5")]
        polygamma_input = [mp.mpf("0.5"), mp.mpf("1"), mp.mpf("2.5")]
        gamma_shape = [mp.mpf("0.5"), mp.mpf("2"), mp.mpf("5")]
        gamma_value = [mp.mpf("0.25"), mp.mpf("1.5"), mp.mpf("7")]
        lower = [mp.gammainc(a, 0, x, regularized=True) for a, x in zip(gamma_shape, gamma_value)]
        upper = [mp.gammainc(a, x, mp.inf, regularized=True) for a, x in zip(gamma_shape, gamma_value)]
        length, beta = 5, mp.mpf("8.6")
        center = mp.mpf(length - 1) / 2
        kaiser = [
            mp.besseli(0, beta * mp.sqrt(1 - ((index - center) / center) ** 2)) / mp.besseli(0, beta)
            for index in range(length)
        ]
    return {
        "dirichlet_x": _float([item[0] for item in dirichlet_inputs], [2], "torch.float64"),
        "dirichlet_alpha": _float([item[1] for item in dirichlet_inputs], [2], "torch.float64"),
        "dirichlet_total": _float([item[2] for item in dirichlet_inputs], [2], "torch.float64"),
        "dirichlet_expected": _float(dirichlet, [2], "torch.float64"),
        "gamma_alpha": _float([item[0] for item in gamma_inputs], [2], "torch.float64"),
        "gamma_output": _float([item[1] for item in gamma_inputs], [2], "torch.float64"),
        "gamma_grad_expected": _float(gamma_grad, [2], "torch.float64"),
        "i0_input": _float(i0_input, [3], "torch.float64"),
        "i0_expected": _float([mp.besseli(0, value) for value in i0_input], [3], "torch.float64"),
        "polygamma_order": 2,
        "polygamma_input": _float(polygamma_input, [3], "torch.float64"),
        "polygamma_expected": _float([mp.polygamma(2, value) for value in polygamma_input], [3], "torch.float64"),
        "regularized_shape": _float(gamma_shape, [3], "torch.float64"),
        "regularized_value": _float(gamma_value, [3], "torch.float64"),
        "regularized_lower": _float(lower, [3], "torch.float64"),
        "regularized_upper": _float(upper, [3], "torch.float64"),
        "kaiser_length": length,
        "kaiser_periodic": False,
        "kaiser_beta": float(beta),
        "kaiser_expected": _float(kaiser, [length], "torch.float64"),
    }


def main():
    records = {
        "CP-COMPLEX-ARITH": _complex_arithmetic(),
        "CP-COMPLEX-LOSS": _complex_loss(),
        "CP-COMPLEX-UNARY": _complex_unary(),
        "CP-LDEXP-CUMPROD": _ldexp_cumprod(),
        "CP-GRAD-COV": _gradient_covariance(),
        "CP-LANCZOS": _lanczos(),
        "CP-GRID": _grid(),
        "CP-CONV": _convolution(),
        "CP-MATRIXEXP": _matrix_exp(),
        "CP-SPECIAL": _special(),
    }
    print(json.dumps({"schema_version": 1, "tools": {"mpmath": mp.__version__}, "records": records}, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


if __name__ == "__main__":
    main()
