# Release Checklist

Use this checklist before publishing a public TorchCTS release. It is intended
to protect package quality, coverage accounting, and repository hygiene.

## Validation Commands

Run the release gate from the repository root:

```bash
.venv/bin/python -m pytest -q torchcts/selftest --validation
.venv/bin/python -m torchcts coverage audit
.venv/bin/python -m torchcts coverage check --fail-on-unknown
.venv/bin/python -m compileall -q torchcts
.venv/bin/python scripts/check_release_hygiene.py
git diff --check
```

Run backend hardware jobs on the hosts that support the corresponding backend
families. Backend-pack coverage is only accepted from a build that can execute
the direct dispatcher path.

## Updating PyTorch Compatibility

TorchCTS only claims compatibility with PyTorch versions that are collected,
reduced, verified, and reflected in package dependency metadata. Do not add a
new PyTorch version by manually editing reduced JSON artifacts.

To add a new stable PyTorch patch release, run the compatibility updater from
the repository root. For example, if PyTorch `2.12.2` becomes available:

```bash
source .venv/bin/activate

python scripts/update_pytorch_compatibility.py \
  --add-version 2.12.2 \
  --family cpu \
  --update-tracked \
  --verify \
  --max-runtime-bytes 2000000
```

The updater is the source of truth for PyTorch compatibility updates. It adds
the version to `scripts/pytorch_version_matrix.json`, checks for missing stable
patch releases inside the claimed range, creates isolated collection venvs,
installs the exact PyTorch version, collects dispatcher and dtype evidence,
regenerates `torchcts/op_metadata.json`, regenerates compact
`torchcts/op_dtype_contracts.json`, regenerates source evidence under
`data/pytorch-version-matrix/`, updates the `torch` dependency upper bound in
`pyproject.toml`, runs artifact verification, runs selftests, builds package
artifacts, and verifies wheel/sdist contents.

If the raw matrix artifacts were already collected and only the tracked
artifacts need to be regenerated and verified, use:

```bash
python scripts/update_pytorch_compatibility.py \
  --selection torch-2.7-through-2.12-cpu \
  --update-tracked \
  --verify \
  --skip-collection \
  --max-runtime-bytes 2000000
```

After a successful update, confirm the summary reports the expected version
set, no unresolved version holes, runtime contracts under the 2 MB ceiling, and
the next-patch `torch` dependency upper bound.

## Package Build And PyPI README Validation

Build release artifacts into a temporary output directory:

```bash
tmpdist="$(mktemp -d)"
.venv/bin/python -m build --sdist --wheel --outdir "$tmpdist"
```

Validate package metadata:

```bash
.venv/bin/python -m twine check "$tmpdist"/*
```

PyPI README validation is required for every public release.

If `twine` is not installed in the release environment, install it or inspect
the generated metadata with the standard library before upload. The built
metadata must use the root `README.md` as the long description because
`pyproject.toml` publishes `README.md` as the package readme.

Inspect the wheel and sdist contents before upload. They must include required
TorchCTS package data and must not include generated results, build outputs,
caches, local manifests, temporary planning files, publish credentials, or local
helper scripts.

## Repository Hygiene

The hygiene script is part of the release gate:

```bash
.venv/bin/python scripts/check_release_hygiene.py
```

It rejects tracked or staged artifacts that should remain local, including
build outputs, cache directories, results, virtual environments, package
metadata outputs, local environment files, credential-like files, and
backend-specific contamination in tracked text.

Publishing credentials and local package helper scripts stay ignored and local.
Do not document credential names, token formats, or local upload helpers in
public package copy.

## Final Release Steps

Only after validation passes:

- update the package version;
- rebuild wheel and sdist from a clean tree;
- run package metadata checks again;
- upload the artifacts through the project's release process;
- tag the release after the published artifact is confirmed.
