"""Matmul path-shape specs."""

from torchcts.path_shapes.specs.common import case, limit


def _mm_cases(tier="standard"):
    shapes = [
        (1, 31, 33), (1, 64, 65), (1, 127, 129), (2, 33, 31), (7, 65, 33),
        (8, 32, 64), (16, 63, 65), (17, 64, 17), (31, 33, 65), (33, 65, 31),
        (35, 48, 49), (64, 17, 33), (65, 65, 17), (9, 96, 40), (12, 80, 24),
        (24, 24, 96), (48, 40, 12), (96, 16, 8), (5, 128, 37), (37, 5, 128),
    ]
    layouts = ("nn", "nt", "tn")
    result = []
    for m, k, n in shapes:
        for layout in layouts:
            for stride_mode in ("contiguous", "padded"):
                result.append(case(
                    runner="matmul.mm",
                    family="matmul",
                    name=f"m{m}_k{k}_n{n}_{layout}_{stride_mode}",
                    shape={"m": m, "k": k, "n": n},
                    tier=tier,
                    cost_class="small" if tier == "standard" else "medium",
                    layout=layout,
                    stride_mode=stride_mode,
                    model_role="decode_or_tile_boundary",
                    branch_intent=[f"m_{m}", f"k_{k}", f"n_{n}", f"layout_{layout}", stride_mode],
                ))
    return result


def _bmm_cases(tier="standard"):
    shapes = [(1, 8, 33, 17), (2, 7, 31, 33), (3, 16, 65, 17), (7, 17, 33, 65), (17, 5, 16, 31)]
    result = []
    for batch, m, k, n in shapes:
        for stride_mode in ("contiguous", "zero_stride_batch"):
            result.append(case(
                runner="matmul.bmm",
                family="matmul",
                name=f"b{batch}_m{m}_k{k}_n{n}_{stride_mode}",
                shape={"batch": batch, "m": m, "k": k, "n": n},
                tier=tier,
                cost_class="small",
                layout="batch_broadcast" if stride_mode == "zero_stride_batch" else "batch_contiguous",
                stride_mode=stride_mode,
                model_role="batched_projection",
                branch_intent=[f"batch_{batch}", stride_mode, "tile_tail"],
            ))
    return result


def _other_cases(tier="standard"):
    result = []
    for m, k, n in [(4, 17, 31), (9, 33, 17), (16, 31, 65), (33, 17, 9), (7, 65, 33)]:
        result.append(case(
            runner="matmul.addmm",
            family="matmul",
            name=f"addmm_m{m}_k{k}_n{n}",
            shape={"m": m, "k": k, "n": n, "alpha": 0.75, "beta": 0.25},
            tier=tier,
            model_role="linear_projection_with_bias",
            branch_intent=["addmm", f"m_{m}", f"k_{k}", f"n_{n}"],
        ))
        result.append(case(
            runner="matmul.linear",
            family="matmul",
            name=f"linear_m{m}_in{k}_out{n}",
            shape={"input_shape": [m, k], "in_features": k, "out_features": n, "bias": True},
            tier=tier,
            model_role="linear_layer",
            branch_intent=["linear", f"in_{k}", f"out_{n}"],
        ))
    for batch, m, k, n in [(2, 3, 17, 31), (4, 1, 33, 17), (3, 5, 31, 9), (7, 2, 16, 33)]:
        result.append(case(
            runner="matmul.matmul",
            family="matmul",
            name=f"matmul_b{batch}_m{m}_k{k}_n{n}",
            shape={"a_shape": [batch, m, k], "b_shape": [batch, k, n]},
            tier=tier,
            model_role="broadcastable_matmul",
            branch_intent=["matmul_nd", f"batch_{batch}", "tail_shape"],
        ))
    return result


def cases():
    standard_mm = _mm_cases()
    standard_bmm = _bmm_cases()
    standard_other = _other_cases()
    standard = limit(standard_mm, 116) + limit(standard_bmm, 10) + standard_other

    heavy_mm = _mm_cases("heavy")
    heavy_bmm = _bmm_cases("heavy")
    heavy_other = _other_cases("heavy")
    heavy = limit(heavy_mm, 46) + limit(heavy_bmm, 10) + heavy_other
    return standard + heavy
