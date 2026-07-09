# TorchCTS - PyTorch Backend Conformance

[![PyPI Version](https://img.shields.io/pypi/v/torchcts?style=flat-square&color=3B8BF6&label=PyPI)](https://pypi.org/project/torchcts/)
[![License](https://img.shields.io/github/license/TorchCTS/TorchCTS?style=flat-square&color=gray)](LICENSE)

TorchCTS is an open-source conformance suite for PyTorch backends.

If you are building or shipping a PyTorch device backend, accelerator backend,
PrivateUse1 backend, or compiler backend, TorchCTS verifies that the backend
behaves the way its manifest says it does.

Instead of only asking "did the selected tests pass?", TorchCTS asks whether a
backend can prove every dtype, capability, and coverage claim it declares. It
builds tests from PyTorch's installed OpInfo database, generated dispatcher
coverage, and hand-authored suites for behavior OpInfo does not fully express:
layout, stride, memory format, sparse and nested tensors, dtype-specific
behavior, compiler behavior, training workflows, device APIs, memory behavior,
stress cases, and model workloads.

A TorchCTS run produces structured evidence: what passed, what failed, what was
explicitly not claimed, what was deselected by policy, and what still needs
coverage work. Crash isolation can protect the pytest parent process, but it
does not turn backend failures into skips, xfails, or passes.

## At a Glance

These numbers are generated from this checkout and installed PyTorch build, not
from a backend pass/fail run. Refresh them with
`python scripts/generate_site_stats.py` before using them in release or website
copy.

| Metric | Current value |
| --- | ---: |
| Generated at | 2026-07-09T15:59:27Z |
| PyTorch version | 2.12.1 |
| Pytest nodes collected | 19,416 |
| Pytest executable nodes | 19,053 |
| ATen overloads inventoried | 3,225 |
| Backend-relevant overloads | 3,214 |
| Covered backend-relevant overloads | 3,062 |
| Dispatcher coverage | 95.3% |
| Unknown tensor-touching surfaces | 0 |
| Known crash isolation rules | 13 |

The generated source for these values is
[`docs/site-stats.md`](https://github.com/TorchCTS/TorchCTS/blob/main/docs/site-stats.md).

## Who Should Use TorchCTS?

- Backend developers validating a PyTorch device implementation.
- Hardware vendors and accelerator teams preparing release evidence.
- Compiler teams checking `torch.compile` and backend integration behavior.
- CI engineers who need reproducible backend support accounting.
- Organizations shipping custom PyTorch builds or PrivateUse1 integrations.

## Why TorchCTS?

Backend validation is difficult because a passing test run can still leave the
important questions unanswered:

- Which operator and dtype combinations actually ran?
- Which failures are backend bugs rather than unsupported features?
- Which declared capabilities became executable requirements?
- Which tests did not run because of manifest policy, resource policy, semantic
  depth, or unfinished coverage strategy?
- What changed between releases, hardware targets, or PyTorch versions?

TorchCTS answers those questions with manifest-driven validation, structured
not-run accounting, coverage audits, and backend-oriented reports.

## How It Works

```text
manifest.py
  -> declared dtypes, capabilities, resources, tolerances
  -> OpInfo tests, generated dispatcher tests, hand-authored suites
  -> backend execution
  -> JSON results, scorecards, coverage audits, evidence packs
```

A `manifest.py` describes what the backend claims to support. Positive
declarations become test requirements. Negative declarations remain visible as
structured accounting records instead of disappearing from the report.

TorchCTS's operator matrix starts from PyTorch OpInfo rather than raw dispatcher
enumeration of every internal `aten::` overload. Full backend coverage comes
from that OpInfo matrix plus TorchCTS-owned generated and hand-authored suites
that cover behavior OpInfo does not dynamically generate.

## What TorchCTS Adds

| Question | Ordinary backend test run | TorchCTS |
| --- | --- | --- |
| Did selected tests pass? | Yes | Yes |
| Are backend declarations executable requirements? | Usually manual | Manifest-driven |
| Are unsupported claims accounted for? | Often hidden in skips | Structured not-run records |
| Can reports separate failure, manifest policy, and coverage gaps? | Usually ad hoc | Built in |
| Dispatcher coverage audit? | Usually separate work | Built in |
| Can release evidence be regenerated from saved JSON? | Usually custom | Built in |

TorchCTS is not another backend, compiler, benchmark, or replacement for
upstream PyTorch tests. It is a conformance framework and evidence generator for
backend teams.

## Quick Start

TorchCTS requires Python >= 3.10 and PyTorch >= 2.7.

Run TorchCTS from the Python environment that contains the PyTorch and backend
build you want to validate. The CLI does not silently switch into a project
`.venv` by default. If you explicitly want that behavior, set
`TORCHCTS_USE_PROJECT_VENV=1`.

```bash
pip install torchcts

torchcts init --template smoke --non-interactive
torchcts check-manifest --manifest manifest.py
torchcts run --device mps
torchcts report
```

Available manifest templates are `smoke`, `minimal`, `inference`, `training`,
and `complete`.

The manifest checker rejects unknown top-level keys, stale capability names
such as `generator` and `quantized`, unsupported dtype keys, invalid tolerance
overrides, invalid quantized container formats, and malformed custom decoder
paths.

## Common Run Controls

Each manifest declares a semantic run depth with `semantic_level` from `1` to
`8`. A run at level `N` collects the normal manifest-valid test set, then skips
cases whose published `semantic_level` is greater than `N`.

```bash
torchcts run --device mps --level 4
```

Use `--dtype` to narrow one run to specific dtypes. Short and fully qualified
names are accepted:

```bash
torchcts run --device mps --level 4 --dtype float32 --dtype torch.bfloat16
```

The dtype filter rewrites the effective manifest for that run only. Selected
dtypes collect as supported even if the original manifest used a narrower dtype
declaration.

Semantic level is not a capability claim and does not replace dtype, layout,
resource, or capability gating. It is a priority/depth axis: level 1 is the fast
primitive baseline, level 4 is broad production behavior, and level 8 is
release-depth stress and adversarial coverage.

For a collection-only skip audit:

```bash
torchcts show-skips --device mps --level 4
```

`show-skips` reports structured manifest and semantic-level accounting without
executing tests.

## What Reports Show

`torchcts report` regenerates Markdown scorecards and validation reports from
saved JSON results under `./results/`.

Reports include:

- backend, hardware key, PyTorch version, run timestamp, and duration;
- operator coverage split across pass, fail/error, manifest policy, selection,
  coverage policy, CPU contract, and runtime availability;
- capability results for manifest-declared feature areas;
- dtype coverage;
- semantic-level execution accounting;
- failure summaries and baseline regressions when baseline history exists.

Full-run scorecards require enough runnable tests to support a meaningful
backend support percentage. Partial or interrupted runs still produce reports,
but they are explicitly marked as partial and do not get a backend support
percentage.

## Coverage Audit

Inventory the installed PyTorch dispatch surface and map each `aten::` overload
to OpInfo coverage, hand-authored markers, generated coverage, exclusions,
backend-pack coverage, or an unknown status:

```bash
torchcts coverage audit
torchcts coverage report
torchcts coverage check --fail-on-unknown
```

Coverage commands use default paths. Built-in exclusions are packaged with
TorchCTS, an optional project `./coverage_exclusions.json` is merged after them,
and audit artifacts are written under `./results/coverage/`.

Unknown tensor-touching surfaces warn loudly and exit `0` by default for
compatibility; release checks should use `coverage check --fail-on-unknown`.
Malformed exclusion JSON, invalid exclusion names, and inconsistent audit
metadata exit nonzero. Coverage summaries also include semantic-level counts
for covered surfaces and generated sample case families.

Coverage policy, oracle-authoring rules, backend-pack rules, exclusion policy,
and accepted contract evidence are documented in
[`docs/coverage/`](https://github.com/TorchCTS/TorchCTS/blob/main/docs/coverage/README.md).

## Runtime Policy

Manifest dtype and capability settings have strict meanings:

- `True` means the backend claims support. TorchCTS collects matching tests and
  any runtime unsupported-operation error is a test failure or error.
- `False` means the backend does not claim support. TorchCTS records structured
  manifest accounting and removes matching tests from execution.
- Dtype regex declarations allow only matching operators; non-matching operator
  dtypes become structured accounting records.
- Missing dtypes in concrete hand-authored tests are recorded as
  `dtype_not_listed`.

TorchCTS still runs small diagnostic probes for declared dtypes and
capabilities. Probe failures are written to the result JSON and diagnostic JSONL
artifacts, but probes do not rewrite the manifest, skip tests, or abort a run.

Crash-prone tests can be isolated in subprocesses:

```bash
torchcts run --device mps --adaptive-isolation auto
python -m pytest --collect-only --known-segfault-audit --device mps --level 8
```

Known crash rules come from the packaged reviewed ledger and adaptive isolation
comes from matching prior result/runlog evidence on the same hardware key,
device, and PyTorch minor-version family. Both mechanisms only choose where a
test executes. Passing, failing, timing out, or crashing keeps the same result
semantics it would have had without isolation.

More detail is in
[`docs/harness.md`](https://github.com/TorchCTS/TorchCTS/blob/main/docs/harness.md).

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
  override the manifest semantic run depth for that run. Pass `--dtype DTYPE`
  one or more times to narrow the effective manifest for that run.
- `show-skips`: Dry-run collection to show skipped tests and reasons. Pass
  `--level N` to audit a specific semantic run depth.
- `report`: Regenerate scorecards and reports from JSON results.
- `sync-opinfo`: Force-rebuild the OpInfo registry cache.
- `check-manifest`: Validate manifest syntax and schema.
- `coverage inventory`: Write `./results/coverage/inventory.json`.
- `coverage audit`: Write inventory, audit, unknowns, unmapped-tests, and
  summary artifacts under `./results/coverage/`.
- `coverage report`: Render the default coverage audit summary.
- `coverage materialize`: Write deterministic generated coverage cases.
- `coverage non-unique-audit`: Audit non-unique coverage identifiers.
- `coverage evidence-pack`: Build backend evidence artifacts.
- `coverage check`: Validate the default coverage audit. Unknowns warn by
  default; `--fail-on-unknown` or `--strict-unknowns` makes them nonzero.
- `path-shapes validate`: Validate the curated path-shape corpus.
- `path-shapes summary`: Summarize corpus families, resource tiers, semantic
  levels, budgets, and waivers.
- `path-shapes list`: List corpus cases by selector.
- `path-shapes run`: Run selected path-shape cases through pytest.
- `triage mps`: Classify MPS failures and optional crash repros.

`--validation` is a CPU harness validation mode. It validates the harness and
CPU-compatible tests without probing an accelerator; it is not a substitute for
running the suite on the backend you intend to ship.

Pytest-level controls used by the CLI include:

- `--adaptive-isolation {auto,off}`: isolate tests with matching prior crash,
  timeout, or suspected-hang evidence. CLI runs default to `auto`.
- `--known-segfault-policy {isolate,off}`: enable or disable reviewed known
  crash subprocess isolation.
- `--known-segfault-audit`: collect tests, validate active known-crash rules,
  print rule coverage, and exit without running tests.

## Project Structure

- The package entry point is `torchcts`.
- Manifest templates are in `torchcts/templates/`.
- Test execution results are saved under `./results/`.
- Generated current-checkout statistics are in
  [`docs/site-stats.md`](https://github.com/TorchCTS/TorchCTS/blob/main/docs/site-stats.md).
- Runtime harness policy is documented in
  [`docs/harness.md`](https://github.com/TorchCTS/TorchCTS/blob/main/docs/harness.md).
- Release validation is documented in
  [`docs/release.md`](https://github.com/TorchCTS/TorchCTS/blob/main/docs/release.md).
