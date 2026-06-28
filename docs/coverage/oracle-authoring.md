# Oracle Authoring

TorchCTS oracles define correct behavior for dispatcher surfaces that are not
adequately represented by OpInfo, generic generated tests, or hand-authored
surface markers.

An oracle must validate behavior. Shape-only, dtype-only, finite-only, and
does-not-crash checks are not sufficient for data-touching dispatcher coverage.

## Required Assertions

Choose assertions based on the dispatcher contract:

- Value equality or value tolerance against a reference.
- Shape and dtype.
- Device and layout.
- Aliasing and storage identity.
- Mutation and in-place return identity.
- `out=` object identity and written value.
- Sparse indices, values, layout, coalescing state, and dense equivalence.
- Nested tensor values, padded equivalence, and metadata.
- RNG determinism, distribution bounds, key/counter protocol, and in-place or
  out-variant identity.
- Allocator and stream lifetime properties for memory-management surfaces.

## IEEE754 Handling

NaN, Inf, and signed-zero behavior must be handled explicitly when the public
or dispatcher contract defines those values.

Use `equal_nan=True` only when NaN equivalence is part of the accepted
semantics. Do not hide NaN or Inf disagreements by broadening tolerances.

For floating-point oracles, include edge-value cases when they are relevant to
the operation family:

- finite ordinary values;
- positive and negative infinity;
- NaN propagation or NaN preservation;
- positive and negative zero where sign is observable;
- subnormal values where the backend contract preserves them.

## Backend Availability

Use `OracleUnavailable("backend_not_available: ...")` when TorchCTS has a
strategy but the selected backend or build cannot execute the direct dispatcher
path.

Use `coverage_strategy_pending` only when TorchCTS does not yet have a strategy.

These are different states. Backend absence is an execution filter. Strategy
absence is unfinished TorchCTS coverage work.

## Source-Derived Contracts

Opaque, private, packed, fused, or vendor-specific surfaces require an accepted
contract before promotion to covered status.

Accepted contract evidence can come from:

- PyTorch or backend source code;
- a public API contract that defines the same behavior;
- a validated backend run against a reference oracle;
- a stable in-repository TorchCTS reference implementation that is tied to a
  reviewed source contract.

Black-box probes can confirm an accepted contract. They must not invent one.

## Selftests

Every new oracle family needs focused selftests. Use targeted tests during
development.

Required selftest coverage:

- a positive reference case;
- a case that catches an incorrect reference result where practical;
- backend-unavailable skip classification;
- `out=` identity when relevant;
- aliasing and mutation checks when relevant;
- invalid-shape or invalid-contract guards when relevant.

## Promotion Standard

A surface can move out of pending only when the exact dispatcher overload has:

- a safe invocation strategy;
- meaningful value or property assertions;
- backend/build gating for unavailable paths;
- focused selftests for the strategy or reference;
- refreshed coverage audit and materialized generated artifacts.
