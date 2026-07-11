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

## Backend Evidence Collection

Collect evidence directly into the canonical tracked store from a checkout on
the target backend:

```bash
python -m torchcts coverage collect-backend-evidence \
  --store evidence/backends \
  --device cuda \
  --backend-gate cuda
```

Backend selection is explicit and independent of PyTorch's device spelling.
ROCm uses a CUDA-namespaced device but records evidence under `rocm`. CPU-build
and FBGEMM evidence can be collected together because both gates execute on
CPU:

```bash
python -m torchcts coverage collect-backend-evidence \
  --store evidence/backends \
  --device cuda \
  --backend-gate rocm

python -m torchcts coverage collect-backend-evidence \
  --store evidence/backends \
  --device cpu \
  --backend-gate cpu-build \
  --backend-gate fbgemm
```

The collector rejects incompatible backend/device selections before writing.
It does not have an all-backends mode: diagnostic skips for unavailable
backends belong in the live coverage audit, not in canonical evidence for that
backend.

Use `--run-pending-candidates` only for promotion work. It executes pending
backend-pack specs that already have real runners, while generated conformance
tests continue to skip `pending_backend_pack` rows. Combine it with
`--require-oracle-results` and `--fail-on-oracle-failure` when collecting
promotion evidence:

```bash
python -m torchcts coverage collect-backend-evidence \
  --store evidence/backends \
  --device cuda \
  --backend-gate cuda \
  --run-pending-candidates \
  --require-oracle-results \
  --fail-on-oracle-failure
```

For a focused collection, repeat `--surface` with exact dispatcher names:

```bash
python -m torchcts coverage collect-backend-evidence \
  --store evidence/backends \
  --device cuda \
  --backend-gate cuda \
  --surface aten::_fused_dropout \
  --surface aten::_fused_dropout.out
```

Use `--runtime-modification` to record a path-free semantic label for a required
runtime modification, such as `sm89-guard-bypass`. Never record a shim path,
hostname, checkout path, environment path, or executable path.

The command validates the existing store, adds one anonymous collection source,
deduplicates identical observations within each backend, and replaces the store
transactionally. It stores backend-specific schema, dispatch, oracle, and
result evidence. It does not store a full repository audit or produce an
archive, README, report, or secondary export.

Accepted backend-pack evidence records must include:

- backend family;
- device;
- PyTorch version;
- path-free build and device capabilities;
- direct dispatcher surfaces exercised;
- reference or property used;
- result.

Collection does not promote a surface automatically. A promotion change must
explicitly mark a passing source in the backend record, update the corresponding
`OracleSpec`, and pass `scripts/verify_backend_evidence.py`.

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
