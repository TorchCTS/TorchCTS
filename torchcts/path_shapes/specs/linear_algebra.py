"""Linear-algebra path-shape specs."""

from torchcts.path_shapes.specs.common import case, limit


def cases():
    solve = []
    cholesky = []
    eigh = []
    qr = []
    svdvals = []
    for n in (2, 3, 4, 5, 7, 9, 11):
        for batch_shape in ([], [2], [3]):
            solve.append(case(
                runner="linear_algebra.solve",
                family="linear_algebra",
                name=f"solve_b{'x'.join(map(str, batch_shape)) or 'none'}_n{n}",
                shape={"batch_shape": batch_shape, "n": n, "rhs": 2},
                cost_class="small",
                model_role="batched_solve",
                branch_intent=["solve", f"n_{n}", f"batch_{batch_shape or 'none'}"],
            ))
            cholesky.append(case(
                runner="linear_algebra.cholesky",
                family="linear_algebra",
                name=f"cholesky_b{'x'.join(map(str, batch_shape)) or 'none'}_n{n}",
                shape={"batch_shape": batch_shape, "n": n},
                cost_class="small",
                model_role="spd_factorization",
                branch_intent=["cholesky", f"n_{n}", f"batch_{batch_shape or 'none'}"],
            ))
            eigh.append(case(
                runner="linear_algebra.eigh",
                family="linear_algebra",
                name=f"eigh_b{'x'.join(map(str, batch_shape)) or 'none'}_n{n}",
                shape={"batch_shape": batch_shape, "n": n},
                cost_class="small",
                model_role="symmetric_eigendecomposition",
                branch_intent=["eigh", f"n_{n}", f"batch_{batch_shape or 'none'}"],
            ))
    for m, n in [(5, 3), (7, 4), (9, 5), (4, 7), (3, 3)]:
        qr.append(case(
            runner="linear_algebra.qr",
            family="linear_algebra",
            name=f"qr_m{m}_n{n}",
            shape={"batch_shape": [], "m": m, "n": n, "mode": "reduced"},
            model_role="qr_factorization",
            branch_intent=["qr", f"m_{m}", f"n_{n}"],
        ))
        svdvals.append(case(
            runner="linear_algebra.svdvals",
            family="linear_algebra",
            name=f"svdvals_m{m}_n{n}",
            shape={"batch_shape": [], "m": m, "n": n},
            model_role="singular_value_path",
            branch_intent=["svdvals", f"m_{m}", f"n_{n}"],
        ))
    standard = limit(solve, 9) + limit(cholesky, 8) + limit(eigh, 8) + qr + svdvals
    heavy = []
    for item in (solve[9:21] + cholesky[8:17] + eigh[8:17]):
        clone = dict(item)
        clone["case_id"] = clone["case_id"].replace("_standard", "_heavy")
        clone["resource_tier"] = "heavy"
        clone["semantic_level"] = 8
        heavy.append(clone)
    return standard + heavy
