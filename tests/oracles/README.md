# TorchCTS Oracle Validation

This development-only suite validates TorchCTS-owned reference implementations,
legality contracts, routing, properties, and direct backend dispositions. It is
separate from `torchcts/selftest` and from backend conformance collection.

Run the inventory gate with:

```bash
python scripts/oracle_fixtures/inventory.py --check
python scripts/oracle_fixtures/verify_manifest.py --check
python -m pytest -q tests/oracles
```

Expected values in this suite must come from reviewed frozen records. Tests may
not generate expected values by calling the TorchCTS implementation under test
or the same PyTorch CPU operation that the oracle replaces.

Hashes are deliberately limited to four ownership boundaries: accepted case
files, independent generators, raw evidence artifacts, and the generator
dependency lock. Cases refer to generators and evidence by path and do not
duplicate their digests. The many hashes in `requirements-oracles.lock` are
pip's standard cross-platform distribution hashes, not separate TorchCTS
integrity rules. None of these checks is imported by `torchcts`.
