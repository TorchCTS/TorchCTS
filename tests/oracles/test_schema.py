import copy

import pytest
import torch

from tests.oracles.schema import (
    FixtureValidationError,
    decode_tensor,
    encode_tensor,
    resolve_under,
)


@pytest.mark.oracle_contract(id="fixture-dense-codec", validation_class="V1_FIXED_VALUE")
def test_dense_tensor_codec_round_trips_bits_and_strides():
    float_record = {
        "dtype": "torch.float32",
        "shape": [4],
        "strides": [2],
        "storage_offset": 1,
        "layout": "torch.strided",
        "encoding": "ieee754_bits",
        "values": ["0x80000000", "0x00000000", "0x7f800000", "0x7fc00001"],
    }
    decoded = decode_tensor(float_record)

    assert encode_tensor(decoded) == float_record

    complex_record = {
        "dtype": "torch.complex64",
        "shape": [2],
        "strides": [1],
        "storage_offset": 0,
        "layout": "torch.strided",
        "encoding": "complex_ieee754_bits",
        "real": ["0x80000000", "0x3f800000"],
        "imag": ["0x00000000", "0xff800000"],
    }
    assert encode_tensor(decode_tensor(complex_record)) == complex_record


@pytest.mark.oracle_contract(id="fixture-structured-codec", validation_class="V4_PROPERTY")
def test_quantized_and_sparse_tensor_codecs_preserve_contract_metadata():
    quantized = torch.quantize_per_tensor(
        torch.tensor([-1.0, 0.0, 1.0]), scale=0.25, zero_point=3, dtype=torch.quint8
    )
    quantized_record = encode_tensor(quantized)
    decoded_quantized = decode_tensor(quantized_record)
    assert encode_tensor(decoded_quantized) == quantized_record

    with torch.sparse.check_sparse_tensor_invariants():
        sparse = torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [1, 0]]),
            torch.tensor([2.0, -3.0]),
            size=(2, 2),
        ).coalesce()
    sparse_record = encode_tensor(sparse)
    decoded_sparse = decode_tensor(sparse_record)
    assert encode_tensor(decoded_sparse) == sparse_record


@pytest.mark.oracle_contract(id="fixture-path-containment", validation_class="V4_PROPERTY")
def test_resolve_under_rejects_parent_and_symlink_escape(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="ascii")
    (allowed / "link.json").symlink_to(outside)

    with pytest.raises(FixtureValidationError, match="escapes"):
        resolve_under(allowed, "../outside.json")
    with pytest.raises(FixtureValidationError, match="escapes"):
        resolve_under(allowed, "link.json")


@pytest.mark.oracle_contract(id="fixture-codec-rejects-mutation", validation_class="V4_PROPERTY")
def test_dense_codec_rejects_noncanonical_hex():
    record = encode_tensor(torch.tensor([1.0], dtype=torch.float32))
    mutated = copy.deepcopy(record)
    mutated["values"] = ["0X3F800000"]

    with pytest.raises(FixtureValidationError, match="canonical"):
        decode_tensor(mutated)
