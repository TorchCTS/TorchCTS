# TorchCTS - Focused PyTorch Backend Testing

[![PyPI Version](https://img.shields.io/pypi/v/torchcts?style=flat-square&color=3B8BF6&label=PyPI)](https://pypi.org/project/torchcts/)
[![License](https://img.shields.io/github/license/TorchCTS/TorchCTS?style=flat-square&color=gray)](LICENSE)

TorchCTS is an open-source test suite for PyTorch backend development and
conformance.

PyTorch provides the operator definitions, dispatcher semantics, OpInfo
metadata, reference behavior, and testing foundations that backend
implementations rely on. TorchCTS organizes the backend-relevant parts of that
foundation into a focused suite for operator bring-up, debugging, regression
testing, and release validation.

Backend engineers can start with one dtype, operator family, semantic level,
suite, or path-shape selection while implementing a feature. The same tests can
then become broader regression coverage in CI and release validation.

TorchCTS combines PyTorch OpInfo-sourced tests, generated dispatcher coverage,
hand-authored backend semantic tests, and targeted path-shape cases. It covers
values, dtypes, shapes, layouts, strides, aliasing, mutation, autograd, compiler
behavior, device APIs, memory behavior, workloads, and crash handling.

A run records what passed, failed, crashed, was filtered, or remained pending.
Crash isolation protects the parent process so the rest of the suite can
continue, but it does not turn a backend crash into a skip, xfail, or pass.

## What TorchCTS Is For

- Use a ready-made correctness suite while implementing a backend.
- Reproduce one operator, dtype, semantic, or path-shape problem quickly.
- Keep backend fixes covered as the implementation grows.
- Track filtered work separately from pytest skip-marked tests.
- Measure coverage of backend-relevant dispatcher surfaces.
- Preserve reports, runlogs, coverage audits, and crash evidence for CI and
  release review.

## At a Glance

These figures describe the current TorchCTS package and installed PyTorch
build. They are suite inventory and coverage statistics, not results from a
backend validation run.

| Metric | Current value |
| --- | ---: |
| Generated at | 2026-07-11T18:17:01.726986Z |
| TorchCTS version | 0.4.0 |
| PyTorch version | 2.12.1 |
| Pytest nodes collected | 19,481 |
| Pytest executable nodes | 18,666 |
| Structured filtered nodes | 812 |
| Pytest skip-marked nodes | 3 |
| Backend-relevant overloads | 3,214 |
| Covered backend-relevant overloads | 3,062 |
| Dispatcher coverage | 95.3% |
| Unknown tensor-touching surfaces | 0 |
| Targeted path-shape cases | 1,320 |
| Known crash isolation rules | 13 |

The generated source for these values is
[`docs/site-stats.md`](https://github.com/TorchCTS/TorchCTS/blob/main/docs/site-stats.md).

Test count and dispatcher coverage describe different parts of the suite.
Generated and hand-authored tests also cover backend semantics that a
dispatcher-overload count does not express by itself.

## Who Should Use TorchCTS?

- Backend developers implementing PyTorch device behavior.
- Backend maintainers adding regression coverage and preparing releases.
- PrivateUse1 backend teams integrating an out-of-tree device.
- Hardware vendors validating a backend on target systems.
- Compiler and runtime teams checking dispatch and `torch.compile` behavior.
- CI and release owners who need reproducible backend test artifacts.
- Engineers evaluating what a backend currently supports.

## Focused Backend Testing

PyTorch's framework-wide test suite serves the full project across many
subsystems, devices, build configurations, and development needs. An individual
backend team needs a smaller test surface centered on the behavior its backend
implements.

TorchCTS provides that focused surface without treating a smaller suite as
invisible coverage. Dispatcher audits record which backend-relevant surfaces
are covered, pending, excluded, or unknown. Structured run accounting records
which cases executed and why other cases were filtered.

The result is practical during implementation and reviewable during release.

- Which operator, dtype, layout, and shape behavior should I implement next?
- How can I reproduce one backend failure without running the broadest suite?
- Which fixes are now protected by regression coverage?
- Which cases did not execute, and why?
- Which dispatcher surfaces still need a test strategy or target-hardware
  evidence?
- What changed between backend builds, hardware targets, or PyTorch versions?

## How TorchCTS Fits With PyTorch

PyTorch defines the framework behavior. TorchCTS builds on PyTorch operator
schemas, dispatcher semantics, OpInfo metadata, CPU reference behavior, and
testing conventions.

TorchCTS adds a backend-focused workflow around those foundations: manifest
configuration, focused run controls, generated and hand-authored backend tests,
structured filtered accounting, dispatcher coverage audits, crash isolation,
and reusable result artifacts.

TorchCTS is not a replacement for PyTorch's upstream test suite and does not
define alternative PyTorch semantics. Backend teams can use TorchCTS alongside
the upstream tests appropriate to their development environment. PyTorch
remains the source of truth for the framework semantics a backend implements.

## How It Works

```text
backend environment and manifest
  -> focused dtype, level, suite, operator, or path-shape selection
  -> PyTorch OpInfo-sourced, generated, and hand-authored tests
  -> backend execution and structured filtered accounting
  -> fix and rerun during development
  -> broader regression and release runs
  -> JSON results, reports, runlogs, coverage audits, and crash evidence
```

A `manifest.py` configures the device, backend import, dtypes, capabilities,
semantic depth, hardware, resource limits, and tolerances TorchCTS should use
for a run.

During development, the manifest defines the focused implementation surface
under test. In release and conformance results, it also records the support
represented by the run.

TorchCTS starts from PyTorch OpInfo metadata, then adds generated and
hand-authored suites for backend behavior OpInfo does not fully express. This
includes layout, stride, mutation, aliasing, `out=` behavior, RNG, compiler
integration, device APIs, memory behavior, workloads, and crash-sensitive
paths.

## What TorchCTS Adds To A Backend Workflow

| Backend need | TorchCTS support |
| --- | --- |
| Focused implementation feedback | Narrow runs by dtype, semantic level, suite, operator selection, and path-shape filters |
| PyTorch-version-aware expectations | Versioned operator contracts for supported PyTorch releases |
| Broad backend semantics | OpInfo-sourced, generated, and hand-authored tests |
| Clear non-execution accounting | Filtered cases retain structured reasons outside pytest skip totals |
| Coverage visibility | Dispatcher audit with covered, pending, excluded, and unknown dispositions |
| Crash containment | Known and adaptive subprocess isolation without changing the result |
| Regression review | Saved JSON, markdown reports, runlogs, history, and comparison metadata |
| Release evidence | Coverage checks and reproducible result artifacts from the same suite used during development |

TorchCTS is a backend development and conformance system. It is not a backend,
compiler, benchmark, or replacement for PyTorch's upstream tests.

## Quick Start

Install TorchCTS in the Python environment that contains the PyTorch build and
backend package you are developing or validating. For a backend engineer, this
is usually the project's normal development virtual environment.

TorchCTS requires Python >= 3.10 and currently supports PyTorch 2.7.0 through
2.12.1 (`torch>=2.7.0,<2.12.2`). The CLI does not silently switch into a
project `.venv` by default. If you explicitly want that behavior, set
`TORCHCTS_USE_PROJECT_VENV=1`.

```bash
pip install torchcts

torchcts init --template smoke --non-interactive
torchcts check-manifest --manifest manifest.py
torchcts run --device my_backend --level 1 --report-skips
torchcts report
```

Replace `my_backend` with the runtime device name. PrivateUse1 backends
generally also set `backend_import` and `device_name` in the manifest.

Available manifest templates are `smoke`, `minimal`, `inference`, `training`,
and `complete`.

The manifest checker rejects unknown top-level keys, stale capability names
such as `generator` and `quantized`, unsupported dtype keys, invalid tolerance
overrides, invalid quantized container formats, and malformed custom decoder
paths.

## Use TorchCTS During Development

TorchCTS can bootstrap a test-driven backend workflow. Start with the behavior
currently being implemented, keep the run narrow while working, then preserve
the appropriate broader run as regression coverage.

1. Choose an operator, dtype, capability, semantic level, or path-shape case.
2. Configure the current support surface in `manifest.py`.
3. Run a focused selection.
4. Inspect the result, Markdown report, and runlog.
5. Fix the backend and rerun the same selection.
6. Expand one test axis at a time.
7. Keep the broader command in backend CI after the implementation is reliable.

```bash
torchcts run --device my_backend --level 2 --dtype torch.float32 --report-skips -k matmul
```

This command narrows the run to float32 cases through semantic level 2 whose
pytest selection matches `matmul`. Replace the selection with the behavior
currently under development.

## Common Run Controls

Each test has a semantic level from 1 to 8. `--level N` includes levels 1
through N. Cases above the selected level are filtered from pytest execution
where possible and retained in TorchCTS structured accounting. They do not
become thousands of pytest `SKIPPED` results.

```bash
torchcts run --device my_backend --level 4
torchcts run --device my_backend --level-exact 4
torchcts run --device my_backend --level-range 2:4
```

Use `--dtype` to narrow one run to specific dtypes. Short and fully qualified
names are accepted:

```bash
torchcts run --device my_backend --level 4 --dtype float32 --dtype torch.bfloat16
```

The dtype filter rewrites the effective manifest for that run only. Selected
dtypes collect as supported even if the original manifest used a narrower dtype
declaration.

Use `--suite` to select one TorchCTS suite and normal pytest selectors such as
`-k` to focus the forwarded pytest collection:

```bash
torchcts run --device my_backend --suite generated --level 4
torchcts run --device my_backend --level 4 -k 'matmul or conv'
```

Semantic level controls test depth, not backend capability. Lower levels are
useful for fast development feedback. Deeper levels add broader framework
semantics, workloads, multi-device behavior, stress, and release-depth
coverage.

For a collection-only skip audit:

```bash
torchcts show-skips --device my_backend --level 4
```

`show-skips` reports TorchCTS filtered and pytest skip-marked accounting without
executing tests.

## What Reports Show

Reports are part of the development loop, not only a release deliverable. A
focused run can identify the next implementation action, while broader runs
show regression clusters and release-level coverage.

`torchcts report` regenerates Markdown scorecards and validation reports from
saved result artifacts under `./results/`. Completed runs store one canonical
history artifact and a small latest reference that TorchCTS resolves
automatically.

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

Committed examples are available under
[`sample-results/`](https://github.com/TorchCTS/TorchCTS/tree/main/sample-results).

## Coverage Audit

Coverage auditing makes the focused-suite scope explicit. A smaller backend
suite remains credible when every backend-relevant dispatcher surface has a
visible covered, pending, excluded, or unknown disposition.

Inventory the installed PyTorch dispatch surface and map each `aten::` overload
to OpInfo coverage, hand-authored markers, generated coverage, exclusions,
backend-specific coverage, or an unknown status:

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

The release target is zero unknown tensor-touching backend-relevant surfaces.
The 95.3% dispatcher figure measures dispatcher coverage; it is not a claim of
95.3% total backend correctness. Pending and excluded coverage remain visible.

Coverage policy, oracle-authoring rules, backend-evidence rules, exclusion policy,
and accepted contract evidence are documented in
[`docs/coverage/`](https://github.com/TorchCTS/TorchCTS/blob/main/docs/coverage/README.md).

## Runtime Policy

Manifest dtype and capability settings have strict meanings:

- `True` includes matching in-contract tests in backend execution. Runtime
  unsupported or not-implemented results remain failures or errors.
- `False` leaves the dtype or capability outside the configured support
  surface. Matching cases are recorded in structured filtered accounting where
  applicable.
- A dtype regex includes the dtype only for matching operator names.
- Missing dtypes in concrete hand-authored tests are recorded with the
  applicable structured reason.

TorchCTS still runs small diagnostic probes for declared dtypes and
capabilities. Probe failures are written to the result artifact and diagnostic
JSONL artifacts, but probes do not rewrite the manifest, skip tests, or abort a
run.

Crash-prone tests can be isolated in subprocesses:

```bash
torchcts run --device mps --adaptive-isolation auto
python -m pytest --collect-only --known-segfault-audit --device mps --level 8
```

Known crash rules come from the packaged reviewed ledger and adaptive isolation
comes from matching prior result/runlog evidence on the same hardware key,
device, and PyTorch minor-version family. Isolation changes where a test
executes, not how its result is classified. A passing, failing, timed-out, or
crashed test keeps the same result semantics it would have had without
isolation.

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
- `show-skips`: Dry-run collection to show filtered and pytest skip-marked
  tests with their reasons. Pass `--level N` to audit a specific semantic run
  depth.
- `report`: Regenerate scorecards and reports from result artifacts.
- `sync-opinfo`: Force-rebuild the OpInfo registry cache.
- `check-manifest`: Validate manifest syntax and schema.
- `coverage inventory`: Write `./results/coverage/inventory.json`.
- `coverage audit`: Write inventory, audit, unknowns, unmapped-tests, and
  summary artifacts under `./results/coverage/`.
- `coverage report`: Render the default coverage audit summary.
- `coverage materialize`: Write deterministic generated coverage cases.
- `coverage non-unique-audit`: Audit non-unique coverage identifiers.
- `coverage collect-backend-evidence`: Collect backend-specific observations
  directly into the canonical tracked evidence store.
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
- Committed result examples are under
  [`sample-results/`](https://github.com/TorchCTS/TorchCTS/tree/main/sample-results).
- Generated current-checkout statistics are in
  [`docs/site-stats.md`](https://github.com/TorchCTS/TorchCTS/blob/main/docs/site-stats.md).
- Runtime harness policy is documented in
  [`docs/harness.md`](https://github.com/TorchCTS/TorchCTS/blob/main/docs/harness.md).
- Release validation is documented in
  [`docs/release.md`](https://github.com/TorchCTS/TorchCTS/blob/main/docs/release.md).
