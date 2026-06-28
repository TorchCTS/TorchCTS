# Coverage Exclusions

Coverage exclusions are reviewed dispatcher surfaces that do not count as
covered. Exclusions keep the ledger explicit without converting out-of-scope or
unsupported surfaces into coverage.

The machine-readable exclusion ledger is `torchcts/coverage_exclusions.json`.

## Required Fields

Each exclusion requires:

- `name`
- `match`
- `surface`
- `category`
- `reason`
- `owner`
- `review_after`

`match` is one of:

- `exact`
- `base`
- `regex`

Regex exclusions require a reason explaining why exact names are impractical.

## Categories

- `backend_specific_internal`: backend or vendor internals that require a
  backend-pack strategy before they can become covered.
- `dispatcher_plumbing`: framework or dispatcher implementation details that
  are not backend tensor semantics.
- `distributed_or_c10d`: distributed runtime behavior outside generic tensor
  backend conformance.
- `cpu_reference_invalid`: surfaces where the CPU reference path is not a valid
  oracle for generic backend behavior.
- `unsafe_direct_invocation`: surfaces that bypass public validation or require
  private invariants.
- `covered_by_public_surface`: surfaces that require a formal proxy proof or a
  direct runner before they can count as exact coverage.
- `deprecated_or_removed`: legacy surfaces that PyTorch reports as removed or
  unsupported.
- `manual_future_scope`: reviewed future-scope surfaces.

## Review Rules

Unknown exclusion names fail validation.

Expired `review_after` dates warn in the current phase.

Exclusions never count as covered.

If an excluded surface becomes safely invokable and backend-relevant, it must
move into pending or covered work. It must not remain excluded for convenience.

## Relationship To Reports

The coverage audit loads packaged exclusions and an optional project
`./coverage_exclusions.json`. Project exclusions are merged after packaged
exclusions.

The generated pending-review artifacts group excluded surfaces by category,
owner, review date, status, blocker, backend gate, and source category.
