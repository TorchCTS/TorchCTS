from dataclasses import asdict

import pytest
import torch

from tests.oracles.schema import decode_tensor, encode_tensor
from tests.oracles.test_fixed_values import _case
from torchcts.core.fft_contract import (
    compare_fft_nonfinite_groups,
    fft_source_contributor_mask,
    generated_c2c_fft_contract_spec,
    public_fft_contract_spec,
)


@pytest.mark.oracle_contract(id="cp-fft-contracts", validation_class="V2_ADMISSIBILITY")
def test_fft_specs_and_contributor_masks_match_hand_enumeration():
    case = _case("cp-fft-contracts")
    inputs, expected = case["inputs"], case["expected"]
    for name in ("public_c2c", "public_c2r"):
        row = inputs[name]
        source = decode_tensor(row["source"])
        spec = public_fft_contract_spec(
            row["op_name"], source, tuple(row["args"]), row["kwargs"]
        )
        assert asdict(spec) == {
            **expected[name]["spec"],
            "dimensions": tuple(expected[name]["spec"]["dimensions"]),
            "full_lengths": tuple(expected[name]["spec"]["full_lengths"]),
            "retained_source_lengths": tuple(
                expected[name]["spec"]["retained_source_lengths"]
            ),
        }
        assert encode_tensor(fft_source_contributor_mask(source, spec)) == expected[name][
            "contributor_mask"
        ]

    row = inputs["generated_c2c"]
    source = decode_tensor(row["source"])
    spec = generated_c2c_fft_contract_spec(source, row["dimensions"])
    wanted = expected["generated_c2c"]["spec"]
    assert asdict(spec) == {
        **wanted,
        "dimensions": tuple(wanted["dimensions"]),
        "full_lengths": tuple(wanted["full_lengths"]),
        "retained_source_lengths": tuple(wanted["retained_source_lengths"]),
    }
    assert encode_tensor(fft_source_contributor_mask(source, spec)) == expected[
        "generated_c2c"
    ]["contributor_mask"]


@pytest.mark.oracle_contract(id="cp-fft-contracts", validation_class="V2_ADMISSIBILITY")
def test_fft_nonfinite_comparison_ignores_only_contributing_nonfinite_groups():
    case = _case("cp-fft-contracts")
    row = case["inputs"]["comparison"]
    source = decode_tensor(row["source"])
    expected = decode_tensor(case["expected"]["comparison"]["expected"])
    spec = public_fft_contract_spec(
        row["op_name"], source, tuple(row["args"]), row["kwargs"]
    )
    compared = []

    def compare(actual, wanted, **_metadata):
        compared.append((actual.clone(), wanted.clone()))
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)

    compare_fft_nonfinite_groups(
        decode_tensor(row["accepted_actual"]),
        expected,
        source,
        spec,
        dtype=torch.complex64,
        compare=compare,
        label="fixture",
    )
    assert len(compared) == 1

    with pytest.raises(AssertionError, match="dropped every contributing nonfinite lane"):
        compare_fft_nonfinite_groups(
            decode_tensor(row["rejected_actual"]),
            expected,
            source,
            spec,
            dtype=torch.complex64,
            compare=compare,
            label="fixture",
        )
