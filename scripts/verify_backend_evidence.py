#!/usr/bin/env python3
"""Verify the canonical tracked backend evidence store and promotion links."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torchcts.core.backend_evidence import (
    canonical_backend,
    load_backend_evidence_store,
    outcome_counts,
    parse_promotion_reference,
)
from torchcts.core.oracles import all_oracle_specs


DEFAULT_STORE = Path("evidence/backends")


def verify_backend_evidence(path: str | Path) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    store = load_backend_evidence_store(path, verify_canonical=True)
    expected_promotions: set[tuple[str, str]] = set()
    for spec in all_oracle_specs():
        backend = parse_promotion_reference(spec.promotion_evidence)
        if backend is None:
            continue
        expected_promotions.add((backend, spec.surface))
        try:
            promotion_backend = canonical_backend(spec.promotion_backend or spec.backend_gate)
        except ValueError as exc:
            errors.append(f"{spec.surface}: invalid promotion backend: {exc}")
            continue
        if promotion_backend != backend:
            errors.append(
                f"{spec.surface}: promotion reference {backend!r} disagrees with "
                f"promotion backend {promotion_backend!r}"
            )
            continue
        contracts = store.observations.get(backend, {}).get(spec.surface)
        accepted_sources = store.promotions.get(backend, {}).get(spec.surface, set())
        if not contracts:
            errors.append(f"{spec.surface}: backend evidence {backend!r} has no observation")
            continue
        if not accepted_sources:
            errors.append(f"{spec.surface}: backend evidence {backend!r} has no accepted passing source")
            continue
        for source_id in sorted(accepted_sources):
            contract = contracts.get(source_id)
            if contract is None:
                errors.append(f"{spec.surface}: accepted source {source_id} has no observation")
                continue
            result = contract.get("oracle_result") or {}
            if result.get("ok") is not True:
                errors.append(f"{spec.surface}: accepted source {source_id} is not passing")
            oracle = contract.get("oracle") or {}
            for key, expected in (
                ("oracle_id", spec.oracle_id),
                ("runner", spec.runner),
                ("contract_ref", spec.contract_ref),
            ):
                actual = oracle.get(key)
                if key == "contract_ref" and not actual:
                    # The earliest accepted CUDA collection predates captured
                    # contract references. Its oracle identity and runner are
                    # still exact, and the current OracleSpec owns the link.
                    continue
                if actual != expected:
                    errors.append(
                        f"{spec.surface}: accepted source {source_id} has {key}={actual!r}; "
                        f"expected {expected!r}"
                    )

    actual_promotions = {
        (backend, surface)
        for backend, surfaces in store.promotions.items()
        for surface, source_ids in surfaces.items()
        if source_ids
    }
    for backend, surface in sorted(actual_promotions - expected_promotions):
        errors.append(f"{backend}/{surface}: accepted promotion evidence is not referenced by an OracleSpec")
    return errors, outcome_counts(store)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    args = parser.parse_args(argv)
    try:
        errors, counts = verify_backend_evidence(args.store)
    except Exception as exc:
        print(f"Backend evidence verification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if errors:
        for error in errors:
            print(f"Error: {error}", file=sys.stderr)
        return 1
    total = sum(counts.values())
    print(
        f"Backend evidence verified: {total} observations "
        f"({counts['passed']} passed, {counts['failed']} failed, {counts['skipped']} skipped)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
