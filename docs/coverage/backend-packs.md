# Backend Packs

Backend packs cover dispatcher surfaces that are specific to a backend,
vendor library, or build family. They are not global exclusions. They are
backend-gated coverage strategies.

## Definition

A backend pack includes:

- exact dispatcher surfaces;
- a backend or build gate;
- a safe sample builder;
- a source-derived reference or property;
- a direct dispatcher invocation;
- a structured `backend_not_available` skip path.

Backend-private coverage requires a run on a build that supports the target
backend and direct dispatcher path. A local run on an unsupported host can
validate skip behavior and reference-helper selftests, but it does not prove the
backend-private surface is covered.

## Backend Families

### CPU Build: MKLDNN And NNPACK

MKLDNN and NNPACK surfaces are CPU backend-library surfaces. Their references
come from public dense CPU operations such as convolution, linear, pooling,
RNN, and dense metadata transforms.

Promotion requires a CPU build that can execute the direct MKLDNN or NNPACK
dispatcher path.

### MPS

MPS backend packs cover MPS-specific dispatcher surfaces. Value surfaces use CPU
references where the contract defines comparable math. Property surfaces use
backend-specific observable properties.

MPS TinyGEMM int4 pack and matmul are covered by a backend-pack oracle in
TorchCTS. The accepted contract is recorded in `contract-evidence.md`.

`aten::mps_convolution_backward.out` is intentionally still pending in the
current macOS MPS validation build. The Python schema exists, but a direct probe
on 2026-06-28 rejected every real third out tensor as a nullable/uninitialized
slot and rejected `None` as lacking a device. This surface must not be counted as
covered until a safe direct invocation path exists.

### FBGEMM

FBGEMM packs cover FBGEMM packed linear and quantized recurrent cell surfaces.
Packed weight objects must be produced through PyTorch pack operators or public
quantization flows. Fabricated opaque packed objects are not accepted.

Promotion requires an FBGEMM-enabled build and direct dispatcher validation.

### CUDA

CUDA packs cover cuDNN, cuSparseLt, Triton, fused dropout, and
semi-structured sparse surfaces.

References use public CPU operations or CPU autograd where the contract defines
equivalent math. RNG and sparse formats require property checks specific to the
surface contract.

Promotion requires CUDA hardware and a PyTorch build that executes the direct
dispatcher path.

### ROCm

ROCm packs cover MIOpen convolution, batch norm, CTC, depthwise convolution,
and RNN surfaces.

Promotion requires a ROCm build that executes the direct MIOpen dispatcher
path.

### XLA

XLA-specific surfaces require an XLA build and an accepted property contract.
They are not covered by CPU, MPS, CUDA, or ROCm runs.

### PrivateUse1 Override Hooks

PrivateUse1 override hooks require exact-dispatch evidence from a backend that
implements the hook. Public API behavior does not count unless a formal proxy
proof records exact dispatcher reachability.

## Validation Commands

Run targeted backend-pack tests during development. Full validation belongs at
the end of a closure batch.

CUDA family:

```bash
.venv/bin/python -m pytest -q torchcts/generated/test_oracle_surfaces.py --device cuda --level 8 -k 'cudnn or triton or cslt or dropout or sparse'
```

ROCm family:

```bash
.venv/bin/python -m pytest -q torchcts/generated/test_oracle_surfaces.py --device cuda --level 8 -k 'miopen'
```

PyTorch ROCm builds expose HIP devices through the `cuda` device namespace, so
the TorchCTS device argument for ROCm backend-pack validation is `--device cuda`.

FBGEMM family:

```bash
.venv/bin/python -m pytest -q torchcts/generated/test_oracle_surfaces.py --device cpu --level 8 -k 'fbgemm or quantized'
```

MKLDNN and NNPACK family:

```bash
.venv/bin/python -m pytest -q torchcts/generated/test_oracle_surfaces.py --device cpu --level 8 -k 'mkldnn or nnpack'
```

## Accepted Result Records

Use the packaged evidence command when collecting data from hardware that is not
available to the maintainer doing the promotion:

```bash
python -m torchcts coverage evidence-pack --device cuda
```

Use `--backend-gate` when the evidence target is a build family rather than a
plain `torch.device(...).type`, or when a backend reports as another device type:

```bash
python -m torchcts coverage evidence-pack \
  --device cuda \
  --backend-gate cuda+rocm

python -m torchcts coverage evidence-pack \
  --device cpu \
  --backend-gate cpu+fbgemm+cpu_build
```

Use `--backend-gate all` or `--include-all-backend-packs` only when the archive
should include every backend-pack row regardless of the current machine.

Use `--run-pending-candidates` only for promotion work. It executes pending
backend-pack specs that already have real runners, while generated conformance
tests continue to skip `pending_backend_pack` rows. Combine it with
`--require-oracle-results` and `--fail-on-oracle-failure` when collecting
promotion evidence:

```bash
python -m torchcts coverage evidence-pack \
  --device cuda \
  --backend-gate cuda \
  --run-pending-candidates \
  --require-oracle-results \
  --fail-on-oracle-failure
```

For a focused bundle, repeat `--surface` with exact dispatcher names:

```bash
python -m torchcts coverage evidence-pack \
  --device cuda \
  --surface aten::_fused_dropout \
  --surface aten::_fused_dropout.out
```

The command writes both an unpacked directory and a `.tar.gz` archive under
`results/coverage/evidence-packs/` by default. The archive includes host and
PyTorch environment facts, CUDA/MPS device state, the live coverage audit,
pending-review records, oracle metadata, dispatcher schemas, dispatcher tables,
and current oracle results. Pending backend-pack rows that do not yet have an
`OracleSpec` are still included from the coverage audit with schema, dispatch,
exclusion, and pending-review evidence; their oracle result is recorded as
skipped because no runner exists yet. Registered oracle specs are also skipped
when the selected evidence device cannot run that backend gate, so an all-gates
archive from a CUDA machine does not report MPS or CPU-only oracles as failures.

Accepted backend-pack evidence records must include:

- backend family;
- device;
- PyTorch version;
- host/build description;
- command;
- direct dispatcher surfaces exercised;
- reference or property used;
- result.

## Feasibility Ledger

`torchcts coverage audit` also writes a tracked backend-pack feasibility ledger
to `docs/coverage/backend-pack-feasibility.json`. The ledger is generated from
the live audit and oracle registry and assigns each backend-pack row to exactly
one review bucket:

- `promote_now`: already covered by an accepted contract, runner, and promotion
  evidence;
- `candidate_only`: contract accepted or candidate, real runner exists, but
  matching backend evidence has not promoted it yet;
- `blocked_contract`: no accepted source-derived contract;
- `blocked_schema`: no safe exact dispatcher invocation path;
- `blocked_hardware`: requires an unavailable backend/build;
- `blocked_runtime`: known runtime blocker such as OOM or unsupported layout.

Record accepted results in `contract-evidence.md` or another reviewed public
evidence file. Do not record partial investigation logs in public docs.
