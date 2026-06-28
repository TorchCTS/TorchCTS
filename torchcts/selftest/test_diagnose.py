# Copyright (c) 2026 Kris Bailey. MIT License.
# See LICENSE file in the project root for full license information.

"""Selftests for the diagnostic engine."""

import pytest
from torchcts.core.diagnose import diagnose

pytestmark = pytest.mark.covers_category("selftest")


def test_diagnose_op_not_registered():
    d = diagnose('RuntimeError', "Could not run 'aten::_fft_c2c' with arguments from the 'mps' backend")
    assert d is not None
    assert 'not registered' in d.likely_cause
    assert d.confidence == 'high'


def test_diagnose_numerical_precision():
    d = diagnose('AssertionError', 'maxerr=0.15 > tolerance=0.01 for op torch.mm')
    assert d is not None
    assert 'precision' in d.likely_cause.lower()


def test_diagnose_segfault():
    d = diagnose('ProcessCrash', 'SEGFAULT (exit code -11)', exit_code=-11)
    assert d is not None
    assert 'segmentation' in d.likely_cause.lower() or 'corruption' in d.likely_cause.lower()
    assert d.confidence == 'high'


def test_diagnose_oom():
    d = diagnose('ProcessCrash', 'OOM KILLED (exit code -9)', exit_code=-9)
    assert d is not None
    assert 'memory' in d.likely_cause.lower()


def test_diagnose_complex_unsupported():
    d = diagnose('RuntimeError', "MPS does not support complex types for this operation")
    assert d is not None
    assert 'complex' in d.likely_cause.lower()


def test_diagnose_gradcheck():
    d = diagnose('AssertionError', 'Jacobian mismatch for output 0 with respect to input 0')
    assert d is not None
    assert 'backward' in d.likely_cause.lower() or 'derivative' in d.likely_cause.lower()


def test_diagnose_timeout():
    d = diagnose('TimeoutError', 'TIMEOUT (exceeded 30 seconds)')
    assert d is not None
    assert 'hung' in d.likely_cause.lower() or 'deadlock' in d.likely_cause.lower()


def test_diagnose_stride():
    d = diagnose('RuntimeError', 'expected tensor to be contiguous, got stride=(128, 1)')
    assert d is not None
    assert 'contiguous' in d.likely_cause.lower() or 'non-contiguous' in d.likely_cause.lower()


def test_diagnose_no_match():
    d = diagnose('ValueError', 'some completely unrelated error message xyz123')
    assert d is None
