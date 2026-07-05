"""Sorting path-shape specs."""

from torchcts.path_shapes.specs.common import case, limit


def cases():
    result = []
    dims_list = [(17, 33), (8, 65), (3, 7, 31), (129,), (5, 9, 11), (4, 6), (2, 3, 5)]
    for dims in dims_list:
        for dim in range(len(dims)):
            size = dims[dim]
            for descending in (False, True):
                result.append(case(
                    runner="sorting.sort",
                    family="sorting",
                    name=f"sort_dims{'x'.join(map(str, dims))}_dim{dim}_{'desc' if descending else 'asc'}",
                    shape={"dims": list(dims), "dim": dim, "descending": descending, "ties": True, "stable": False},
                    layout="transposed" if len(dims) == 2 and dim == 0 else "contiguous",
                    stride_mode="transposed" if len(dims) == 2 and dim == 0 else "contiguous",
                    model_role="ordering_with_ties",
                    branch_intent=["sort", f"dim_{dim}", "ties", "descending" if descending else "ascending"],
                ))
                result.append(case(
                    runner="sorting.topk",
                    family="sorting",
                    name=f"topk_dims{'x'.join(map(str, dims))}_dim{dim}_k{max(1, size // 2)}_{'desc' if descending else 'asc'}",
                    shape={"dims": list(dims), "dim": dim, "k": max(1, size // 2), "largest": descending, "sorted": True, "ties": True},
                    model_role="partial_ordering",
                    branch_intent=["topk", f"k_{max(1, size // 2)}", f"dim_{dim}", "ties"],
                ))
            result.append(case(
                runner="sorting.kthvalue",
                family="sorting",
                name=f"kthvalue_dims{'x'.join(map(str, dims))}_dim{dim}_k{max(1, size // 3)}",
                shape={"dims": list(dims), "dim": dim, "k": max(1, size // 3), "ties": True},
                model_role="rank_selection",
                branch_intent=["kthvalue", f"k_{max(1, size // 3)}", f"dim_{dim}", "ties"],
            ))
    standard = limit(result, 55)
    heavy = []
    for item in result[55:80]:
        clone = dict(item)
        clone["case_id"] = clone["case_id"].replace("_standard", "_heavy")
        clone["resource_tier"] = "heavy"
        clone["semantic_level"] = 8
        heavy.append(clone)
    return standard + heavy
