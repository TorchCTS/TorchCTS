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
