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

- `metadata.harness_probe_failure_count` in the latest result JSON;
- `metadata.harness_probe_failure_artifact` in the latest result JSON;
- top-level `harness_probe_failures` in the latest result JSON;
- `results/<hardware-key>_harness_probe_failures_<pid>.jsonl`.

Probe evidence does not rewrite the manifest, skip tests, xfail tests, or abort
the session. If a declared capability is broken, the capability tests run and
fail normally.

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
