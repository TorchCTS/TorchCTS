#!/usr/bin/env python3
"""Generate hand-enumerated FFT shape and contributor-contract records."""

from __future__ import annotations

import json
import struct


def _strides(shape):
    result, stride = [], 1
    for size in reversed(shape):
        result.append(stride)
        stride *= size
    return list(reversed(result))


def _f32_hex(value):
    return f"0x{struct.unpack('>I', struct.pack('>f', float(value)))[0]:08x}"


def _float(values, shape):
    return {
        "dtype": "torch.float32", "shape": list(shape), "strides": _strides(shape),
        "storage_offset": 0, "layout": "torch.strided", "encoding": "ieee754_bits",
        "values": [_f32_hex(value) for value in values],
    }


def _complex(values, shape):
    values = [complex(value) for value in values]
    return {
        "dtype": "torch.complex64", "shape": list(shape), "strides": _strides(shape),
        "storage_offset": 0, "layout": "torch.strided", "encoding": "complex_ieee754_bits",
        "real": [_f32_hex(value.real) for value in values],
        "imag": [_f32_hex(value.imag) for value in values],
    }


def _bool(values, shape):
    return {
        "dtype": "torch.bool", "shape": list(shape), "strides": _strides(shape),
        "storage_offset": 0, "layout": "torch.strided", "encoding": "boolean",
        "values": list(values),
    }


def main():
    c2c_values = [complex(index, -index) for index in range(8)]
    c2c_mask = []
    for _batch in range(2):
        for index in range(4):
            c2c_mask.extend([index < 3, index < 3])
    comparison_source = [float("nan"), 1, 2, 3]
    payload = {
        "schema_version": 1,
        "records": {
            "CP-FFT": {
                "public_c2c": {
                    "op_name": "fft.fft",
                    "source": _complex(c2c_values, [2, 4]),
                    "args": [3, 1],
                    "kwargs": {},
                    "spec": {
                        "family": "fft", "dimensions": [1], "full_lengths": [3],
                        "retained_source_lengths": [3], "kind": "c2c",
                        "half_spectrum_dimension": None,
                    },
                    "contributor_mask": _bool(c2c_mask, [2, 4, 2]),
                },
                "public_c2r": {
                    "op_name": "fft.irfft",
                    "source": _complex([0, 1 + 2j, 3], [3]),
                    "args": [4, 0],
                    "kwargs": {},
                    "spec": {
                        "family": "irfft", "dimensions": [0], "full_lengths": [4],
                        "retained_source_lengths": [3], "kind": "c2r",
                        "half_spectrum_dimension": 0,
                    },
                    "contributor_mask": _bool([True, False, True, True, True, False], [3, 2]),
                },
                "generated_c2c": {
                    "source": _complex([0, 1, 2, 3, 4, 5], [2, 3]),
                    "dimensions": [-1],
                    "spec": {
                        "family": "_fft_c2c", "dimensions": [1], "full_lengths": [3],
                        "retained_source_lengths": [3], "kind": "c2c",
                        "half_spectrum_dimension": None,
                    },
                    "contributor_mask": _bool([True] * 12, [2, 3, 2]),
                },
                "comparison": {
                    "source": _float(comparison_source, [2, 2]),
                    "expected": _complex([1, 2, 3, 4], [2, 2]),
                    "accepted_actual": _complex([complex(float("nan"), 0), 0, 3, 4], [2, 2]),
                    "rejected_actual": _complex([0, 0, 3, 4], [2, 2]),
                    "op_name": "fft.fft",
                    "args": [2, 1],
                    "kwargs": {},
                },
            }
        },
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


if __name__ == "__main__":
    main()
