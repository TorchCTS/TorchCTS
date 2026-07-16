# Harness Runtime Policy

TorchCTS treats the manifest as a set of backend claims and accounting choices.
The harness should not turn a backend runtime failure into a passing, skipped,
or hidden result.

## Manifest Claims

Manifest dtype and capability values have strict meanings:

- `True` means the backend claims support. Matching tests collect and execute.
  Setup probes do not remove the claim.
- `False` means the backend does not claim support. TorchCTS records structured
  not-run accounting and removes matching tests from execution.
- Dtype regex values allow only matching operators. Non-matching concrete
  dtype/operator pairs are recorded as `dtype_regex_filtered`.
- A concrete hand-authored test that names a dtype missing from the manifest is
  recorded as `dtype_not_listed`.

The `--dtype` option narrows the effective manifest for one run. It accepts
short names such as `float32` and fully qualified names such as
`torch.float32`. CLI-selected dtypes intentionally become supported for that
run.

## Probe Evidence

TorchCTS may probe declared dtypes, capabilities, and compiler behavior during
setup. Probe failures are diagnostic only.

Probe failures are recorded in:

- `metadata.harness_probe_failure_count` in the latest resolved result;
- `metadata.harness_probe_failure_artifact` in the latest resolved result;
- top-level `harness_probe_failures` in the latest resolved result;
- `results/<hardware-key>_harness_probe_failures_<pid>.jsonl`.

Probe evidence does not rewrite the manifest, skip tests, xfail tests, or abort
the session. If a declared capability is broken, the capability tests run and
fail normally.

## Result Artifacts

While a run is active, `<hardware-key>_latest.json` is an atomic recovery
snapshot. When the run completes, TorchCTS writes the canonical result once in
`<hardware-key>_history/` and replaces `latest.json` with a small relative
reference to it.

For large completed results, TorchCTS chooses a self-describing compact JSON
representation when it is smaller than ordinary compact JSON. Repeated field
names and exact repeated string values are stored once in tables; decoding
restores the existing `metadata`, `results`, `skips`, and diagnostic strings
without changing their values. Small results remain ordinary compact JSON when
the table envelope would cost more. Legacy expanded result JSON remains
readable.

Internal tools resolve both formats through the shared loader:

```python
from torchcts.core.result_artifacts import load_result_artifact

result = load_result_artifact("results/my_backend_latest.json")
```

Markdown reports contain the scorecard and aggregate audit summaries. Full
tracebacks, stdout, and stderr remain verbatim in the canonical JSON instead of
being duplicated into the report. Run logs retain only the latest 32 test
starts needed for crash and hang diagnosis.

## Structured Accounting

TorchCTS uses structured records for not-run behavior that comes from the
manifest or coverage policy. These records appear in saved results and reports,
but they are not executable pytest items.

Common accounting reasons include:

- `dtype_not_supported`;
- `dtype_regex_filtered`;
- `dtype_not_listed`;
- `capability_not_declared`;
- `op_excluded`;
- semantic-level filtering.

Runtime backend errors are different. If a test reaches execution and the CPU
reference path succeeds, a backend unsupported-operation exception is a test
failure or error.

## Crash Isolation

TorchCTS has two subprocess isolation mechanisms:

- reviewed known-crash rules from the packaged crash ledger;
- adaptive isolation from prior matching crash, timeout, or suspected-hang
  evidence.

Both mechanisms only decide whether a test runs in the parent pytest process or
in a subprocess. They never skip, xfail, downgrade, or hide results.
subprocess isolation never skips a test.

Known-crash audit mode validates rule coverage without running tests:

```bash
python -m pytest --collect-only --known-segfault-audit --device mps --level 8
```

Adaptive isolation is controlled with:

```bash
python -m pytest torchcts --device mps --adaptive-isolation auto
python -m pytest torchcts --device mps --adaptive-isolation off
```

Child subprocesses disable adaptive isolation so isolation decisions do not
recursively depend on child output.

## OpInfo Oracle Evidence

CPU oracle failures discovered while building OpInfo-backed samples are written
as diagnostic JSONL records:

```text
results/<hardware-key>_opinfo_oracle_failures_<pid>.jsonl
```

These records help explain CPU-reference invalidity or sample construction
failures. They do not change collection, skipping, xfail behavior, pass
semantics, or failure semantics.

## Permanent Contract References

Some PyTorch CPU implementations are executable but are not valid semantic
oracles for a backend. TorchCTS uses narrowly routed, permanent references for
those cases. Routing is based on the exact operation, dtype, input condition,
and argument contract; it is not gated by the installed PyTorch version.

The current permanent references cover real-unit-alpha complex add/subtract,
exact non-negative complex tensor integer powers, complex L1 loss, complex
`log2`, complex convolution special values, low-precision 3-D grid-sampler
backward, bf16 transposed 3-D convolution, and dense EmbeddingBag frequency
scaling. A matched reference never silently falls back to the CPU operation if
reference construction fails.

Contract-backed NaN/Inf cases use normal value comparison, including finite
lanes, infinity signs, and complex phase. They are not reduced to propagation
mask checks.

When CPU is the selected target, TorchCTS preserves validation-mode behavior:
the operation is executed as a smoke check, but known-bad CPU numerical output
is not compared against the permanent contract reference. Reference
mathematics is established with review-time development proofs; only the most
critical, nonredundant invariants remain in the fixed-budget selftest suite.
