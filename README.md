# TorchCTS - Validate PyTorch Backends

[![PyPI Version](https://img.shields.io/pypi/v/torchcts?style=flat-square&color=3B8BF6&label=PyPI)](https://pypi.org/project/torchcts/)
[![License](https://img.shields.io/github/license/TorchCTS/TorchCTS?style=flat-square&color=gray)](LICENSE)

TorchCTS is a manifest-driven PyTorch backend validation suite for backend
developers. It imports PyTorch's own OpInfo database from the installed PyTorch
build, builds tests from OpInfo operator metadata, dtype metadata, sample-input
generators, and error-input generators, then augments that matrix with TorchCTS
metadata for known CPU reference failures and undefined IEEE 754 NaN/Inf cases.
TorchCTS also includes hand-authored coverage suites for behavior PyTorch's
dynamic OpInfo list does not fully express, including layout, stride, memory
format, sparse/nested tensors, dtype-specific behavior, compiler behavior,
training workflows, device APIs, memory behavior, stress cases, and model
workloads.

The suite compares backend behavior against CPU references where the built-in
test has a CPU oracle. Unsupported features are skipped when tests declare the
matching capability requirement, and saved JSON keeps compact skip records so
reports can distinguish passed, failed, skipped, and not-run behavior.

## Why TorchCTS?

- **Conformance-focused checks**: Every operator/dtype tuple TorchCTS can build
  from PyTorch OpInfo for the installed PyTorch build, plus hand-authored suites
  that complete coverage for layout, stride, dtype, device, compiler, training,
  memory, and workload behavior.
- **Manifest-driven gating**: A `manifest.py` declares supported dtypes,
  capabilities, resource limits, tolerance overrides, container formats, and
  custom test directories.
- **Honest reports**: Results preserve pass/fail/skip data and generate
  scorecards that separate unsupported features from failing features.
- **Backend-oriented controls**: Resource caps, explicit tolerance overrides,
  capability filters, custom quantized decode hooks, and CPU harness validation
  are built into the normal workflow.

TorchCTS's operator matrix starts from PyTorch OpInfo rather than raw dispatcher
enumeration of every internal `aten::` overload. Full backend coverage comes
from that OpInfo matrix plus the hand-authored TorchCTS suites that exercise the
coverage gaps OpInfo does not dynamically generate.

## Quick Start

### 1. Install

TorchCTS requires Python >= 3.10 and PyTorch >= 2.12.

```bash
pip install torchcts
```

Run TorchCTS from the Python environment that contains the PyTorch and backend
build you want to validate. The CLI does not silently switch into a project
`.venv` by default. If you explicitly want that behavior, set
`TORCHCTS_USE_PROJECT_VENV=1`.

### 2. Init

Create a manifest from one of the shipped templates:

```bash
torchcts init --template smoke --non-interactive
```

Available templates are `smoke`, `minimal`, `inference`, `training`, and
`complete`.

### 3. Check the manifest

Validate the manifest before a long run:

```bash
torchcts check-manifest --manifest manifest.py
```

The checker rejects unknown top-level keys, stale capability names such as
`generator` and `quantized`, unsupported dtype keys, invalid tolerance
overrides, invalid quantized container formats, and malformed custom decoder
paths.

### 4. Run

Execute the suite against the target backend:

```bash
torchcts run --device mps
```

Each manifest also declares a semantic run depth with `semantic_level` from `1`
to `8`. A run at level `N` collects the normal manifest-valid test set, then
skips cases whose published `semantic_level` is greater than `N`. The CLI can
override the manifest for one run:

```bash
torchcts run --device mps --level 4
```

Semantic level is not a capability claim and does not replace dtype, layout,
resource, or capability gating. It is a priority/depth axis: level 1 is the fast
primitive baseline, level 4 is broad production behavior, and level 8 is
release-depth stress and adversarial coverage.

For a collection-only skip audit:

```bash
torchcts show-skips --device mps --level 4
```

### 5. Report

Generate HTML/Markdown scorecards and validation reports from saved JSON
results:

```bash
torchcts report
```

### 6. Audit Coverage

Inventory the installed PyTorch dispatch surface and map each `aten::` overload
to OpInfo coverage, hand-authored markers, generated coverage, exclusions, or an
unknown status:

```bash
torchcts coverage audit
torchcts coverage report
torchcts coverage check
torchcts coverage check --fail-on-unknown
```

Coverage commands use default paths. Built-in exclusions are packaged with
TorchCTS, an optional project `./coverage_exclusions.json` is merged after them,
and audit artifacts are written under `./results/coverage/`.

Unknown tensor-touching surfaces warn loudly and exit `0` by default for
compatibility; release checks should use `coverage check --fail-on-unknown`.
Malformed exclusion JSON, invalid exclusion names, and inconsistent audit metadata exit nonzero.
Coverage summaries also include semantic-level counts for covered surfaces and
generated sample case families.

Coverage policy, oracle-authoring rules, backend-pack rules, exclusion policy,
and accepted contract evidence are documented in
[`docs/coverage/`](docs/coverage/README.md).

## Manifest Notes

Current capability names include:

- `inference`, `training`, `serialization`, `compile`
- `rng`, `device_generator`, `rng_distributions`
- `double_backward`, `gradcheck`, `gradient_checkpointing`
- `autocast`, `fused_optimizer`, `dataloader`, `module_hooks`
- `channels_last`, `sparse`, `nested`, `foreach`, `fp8`
- `quantized_container_plumbing`, `native_quantization`,
  `custom_quantized_decode`
- `pinned_memory`, `streams`, `events`, `deterministic`, `guard_alloc`
- `device_api`, `multi_device`, `ieee754`

Quantized support is intentionally split:

- `quantized_container_plumbing` validates the CPU codec registry, packed byte
  transfer, and scale/zero-point tensor transfer.
- `native_quantization` covers native PyTorch quantized tensor support.
- `custom_quantized_decode` runs user-provided semantic decode hooks and
  compares their output against the CPU container codec.

Custom quantized decoder entries use `module:function` import paths:

```python
"custom_container_decoders": {
    "uint8": "my_backend.quant:decode_uint8",
}
```

The callable receives `(packed, scale, zero_point, shape, dtype, device)` and
returns a decoded `torch.Tensor`.

The built-in suite uses CPU references where a test supports reference
comparison. There is currently no public `reference_device` manifest key or
`--ref-device` CLI option.

## CLI Reference

TorchCTS provides these subcommands:

- `init`: Initialize `manifest.py` from a template.
- `run`: Run the test suite against the target backend. Pass `--level N` to
  override the manifest semantic run depth for that run.
- `show-skips`: Dry-run collection to show skipped tests and reasons. Pass
  `--level N` to audit a specific semantic run depth.
- `report`: Regenerate scorecards and reports from JSON results.
- `sync-opinfo`: Force-rebuild the OpInfo registry cache.
- `check-manifest`: Validate manifest syntax and schema.
- `coverage inventory`: Write `./results/coverage/inventory.json`.
- `coverage audit`: Write inventory, audit, unknowns, unmapped-tests, and summary
  artifacts under `./results/coverage/`.
- `coverage report`: Render the default coverage audit summary.
- `coverage check`: Validate the default coverage audit. Unknowns warn by
  default; `--fail-on-unknown` or `--strict-unknowns` makes them nonzero.

`--validation` is a CPU harness validation mode. It validates the harness and
CPU-compatible tests without probing an accelerator; it is not a substitute for
running the suite on the backend you intend to ship.

## Project Structure

- The package entry point is `torchcts`.
- Manifest templates are in `torchcts/templates/`.
- Test execution results are saved under `./results/`.
