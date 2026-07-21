#!/usr/bin/env python3
"""Generate reviewed admissibility and routing records for non-unique outputs."""

from __future__ import annotations

import json


ROWS = [
    ("eigh_eigenvalues", "representation_ambiguous", "strict_by_api", "linalg.eigh.eigenvalues", None, "Eigenvalues are unique up to numeric tolerance."),
    ("eigh", "representation_ambiguous", "covered_by_contract", "linalg.eigh", "_check_eigh", "Eigenvectors may change sign or unit phase; require eigen equation, orthonormality, and reconstruction."),
    ("eig", "representation_ambiguous", "covered_by_contract", "linalg.eig", "_check_eig", "Eigenpairs may permute and eigenvectors may scale; require finite nonzero vectors and eigen equations."),
    ("eigvals", "representation_ambiguous", "covered_by_contract", "linalg.eigvals", "_check_eigvals", "Eigenvalues may permute; compare the minimum-cost value matching."),
    ("svd", "representation_ambiguous", "covered_by_contract", "linalg.svd", "_check_svd", "Singular vectors may change sign or phase; require values, orthogonality, and reconstruction."),
    ("qr", "representation_ambiguous", "covered_by_contract", "linalg.qr", "_check_qr", "Q/R signs may move together; require orthogonality, triangular R, and reconstruction."),
    ("lu", "representation_ambiguous", "covered_by_contract", "linalg.lu", "_check_lu", "Pivot choices may differ; require permutation legality and P@L@U reconstruction."),
    ("lu_factor", "representation_ambiguous", "covered_by_contract", "linalg.lu_factor", "_check_lu", "Packed pivots may differ; require valid pivots and unpacked reconstruction."),
    ("pinv", "representation_ambiguous", "covered_by_contract", "linalg.pinv", "_check_pinv", "Require all four Moore-Penrose identities."),
    ("lstsq", "representation_ambiguous", "covered_by_contract", "linalg.lstsq", "_check_lstsq", "Rank-deficient solutions may differ by a null-space vector; require equal projection and residual norm."),
    ("sort", "tie_index_ambiguous", "covered_by_contract", "sort", "_check_sortlike", "Tied indices may permute unless stable; values and gathered index legality stay exact."),
    ("arg_reduce", "tie_index_ambiguous", "covered_by_contract", "argmax", "_check_arg_reduce", "Any in-bounds index that selects the extremal value is legal."),
    ("value_index", "tie_index_ambiguous", "covered_by_contract", "cummax", "_check_cumminmax", "Returned values remain exact and indices must select those values."),
    ("mode", "tie_index_ambiguous", "covered_by_contract", "mode", "_check_mode", "Any value with maximal frequency and any index selecting it is legal."),
    ("complex_matmul_determinate", "representation_ambiguous", "covered_by_contract", "matmul", "_check_complex_matmul_determinate", "Only lanes proven determinate by expanded real arithmetic remain strict."),
    ("fft_special", "representation_ambiguous", "covered_by_contract", "fft.fft", "_check_fft_special_contract", "Finite groups compare normally; affected groups must retain a nonfinite."),
    ("complex_product", "representation_ambiguous", "covered_by_contract", "prod", "_check_complex_product", "Result must belong to the finite set of legal multiplication trees."),
    ("welford_mean", "representation_ambiguous", "covered_by_contract", "std_mean", "_check_welford_mean", "Mean follows public arithmetic-mean semantics even when dispersion is nonfinite."),
    ("fractional_max_pool", "tie_index_ambiguous", "covered_by_contract", "fractional_max_pool2d", "_check_max_pool_indices", "Values remain exact and indices must point to those values."),
    ("max_pool_indices", "tie_index_ambiguous", "covered_by_contract", "max_pool2d", "_check_max_pool_indices", "Tied maxima may choose any in-plane index selecting the returned value."),
    ("max_unpool_writers", "tie_index_ambiguous", "covered_by_contract", "max_unpool2d", "_check_max_unpool", "Duplicate destinations may contain any source writer; unwritten destinations must be zero."),
    ("rrelu", "random_value", "covered_by_contract", "nn.functional.rrelu", "_check_rrelu", "Nonnegative values are exact and negative slopes stay within the requested interval."),
    ("random", "random_value", "structural_only", "rand", "_check_structural", "Only output structure and finite-value class are contractual without seeded equivalence."),
    ("randomized_linalg", "random_value", "covered_by_contract", "svd_lowrank", "_check_randomized_linalg", "Require orthonormal factors and a useful approximation, not identical random bases."),
    ("uninitialized", "uninitialized_value", "structural_only", "empty", "_check_structural", "Only output structure is contractual; values are deliberately unconstrained."),
    ("geqrf", "representation_ambiguous", "intentionally_exact", "geqrf", None, "Packed reflector state has no portable standalone legality checker."),
    ("orgqr_ormqr", "representation_ambiguous", "intentionally_exact", "orgqr", None, "Consumers of packed reflector state retain exact comparison."),
    ("value_only_linalg", "representation_ambiguous", "strict_by_api", "linalg.svdvals", None, "Value-only linalg results retain numeric comparison."),
]


def main():
    records = [
        {
            "family": family,
            "ambiguity_type": ambiguity,
            "status": status,
            "representative_op": representative,
            "checker": checker,
            "admissibility_rule": rule,
        }
        for family, ambiguity, status, representative, checker, rule in ROWS
    ]
    print(json.dumps({"schema_version": 1, "records": {"CP-NON-UNIQUE": records}}, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


if __name__ == "__main__":
    main()
