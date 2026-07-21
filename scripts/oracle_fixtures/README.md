# Oracle fixture generators

These development-only tools create staged evidence for the frozen suite in
`tests/oracles`. They never run during pytest collection or backend conformance.

Create a Python 3.12 environment from the hash-locked dependency file:

```bash
python3.12 -m venv scratch/oracle-generator-venv
scratch/oracle-generator-venv/bin/pip install --require-hashes \
  -r scripts/oracle_fixtures/requirements-oracles.lock
```

Generators write canonical data to stdout or to an explicitly supplied staging
directory under `scratch/`. Promotion into `tests/oracles/cases` and
`evidence/oracles` is a reviewed source change; no generator overwrites an
accepted record.

For the C17 integer bootstrap record:

```bash
clang -std=c17 -O0 -Wall -Wextra -Werror \
  scripts/oracle_fixtures/generate_integer_c17.c \
  -o scratch/generate_integer_c17
scratch/generate_integer_c17
```

Verify all accepted records without running candidates:

```bash
python scripts/oracle_fixtures/verify_manifest.py --check
```
