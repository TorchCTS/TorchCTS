"""Reduction path-shape specs."""

from torchcts.path_shapes.specs.common import case, limit


def cases():
    result = []
    dims_list = [(3, 1025), (17, 33), (8, 7, 65), (5, 9, 11), (2, 3, 4, 33)]
    runners = ("reduction.sum", "reduction.mean", "reduction.amax", "reduction.argmax", "reduction.prod")
    for dims in dims_list:
        for reduce_dim in range(len(dims)):
            for runner in runners:
                if runner == "reduction.prod" and len(dims) > 2:
                    continue
                for keepdim in (False, True):
                    result.append(case(
                        runner=runner,
                        family="reduction",
                        name=f"{runner.split('.')[1]}_dims{'x'.join(map(str, dims))}_dim{reduce_dim}_{'keep' if keepdim else 'drop'}",
                        shape={"dims": list(dims), "reduce_dim": reduce_dim, "keepdim": keepdim},
                        layout="transposed" if len(dims) == 2 and reduce_dim == 0 else "contiguous",
                        stride_mode="transpose" if len(dims) == 2 and reduce_dim == 0 else "contiguous",
                        model_role="reduction_boundary",
                        branch_intent=[runner.split(".")[1], f"dim_{reduce_dim}", f"dims_{dims}", "keepdim" if keepdim else "dropdim"],
                    ))
    standard = limit(result, 85)
    heavy = []
    for item in result[85:120]:
        clone = dict(item)
        clone["case_id"] = clone["case_id"].replace("_standard", "_heavy")
        clone["resource_tier"] = "heavy"
        clone["semantic_level"] = 8
        heavy.append(clone)
    return standard + heavy
