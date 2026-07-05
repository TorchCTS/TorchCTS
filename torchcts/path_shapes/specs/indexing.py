"""Indexing path-shape specs."""

from torchcts.path_shapes.specs.common import case, limit


def cases():
    result = []
    dims_list = [
        (129, 33),
        (17, 65),
        (8, 9),
        (7, 11),
        (5, 97),
        (64, 3),
        (21, 22),
        (31, 4),
    ]
    for dims in dims_list:
        rows, cols = dims
        for dim in (0, 1):
            indices = [0, 0, (rows if dim == 0 else cols) // 2, (rows if dim == 0 else cols) - 1]
            result.append(case(
                runner="indexing.index_select",
                family="indexing",
                name=f"index_select_{rows}x{cols}_dim{dim}",
                shape={"dims": [rows, cols], "dim": dim, "indices": indices},
                model_role="embedding_like_lookup",
                branch_intent=["index_select", "repeated_indices", f"dim_{dim}"],
            ))
            gather_index = [[min(j, cols - 1) for j in (0, 0, cols // 2)] for _ in range(rows)] if dim == 1 else [[min(i, rows - 1) for _ in range(cols)] for i in (0, 0, rows // 2)]
            result.append(case(
                runner="indexing.gather",
                family="indexing",
                name=f"gather_{rows}x{cols}_dim{dim}",
                shape={"dims": [rows, cols], "dim": dim, "index": gather_index},
                model_role="gather_duplicate_index",
                branch_intent=["gather", "duplicate_indices", f"dim_{dim}"],
            ))
            result.append(case(
                runner="indexing.scatter_add",
                family="indexing",
                name=f"scatter_add_{rows}x{cols}_dim{dim}",
                shape={"dims": [rows, cols], "dim": dim},
                model_role="duplicate_index_accumulation",
                branch_intent=["scatter_add", "duplicate_destinations", f"dim_{dim}"],
            ))
            result.append(case(
                runner="indexing.scatter_reduce",
                family="indexing",
                name=f"scatter_reduce_{rows}x{cols}_dim{dim}",
                shape={"dims": [rows, cols], "dim": dim, "reduce": "sum"},
                model_role="duplicate_index_reduction",
                branch_intent=["scatter_reduce", "duplicate_destinations", f"dim_{dim}"],
            ))
        result.append(case(
            runner="indexing.take",
            family="indexing",
            name=f"take_{rows}x{cols}",
            shape={"dims": [rows, cols], "indices": [0, 1, cols, rows * cols - 1]},
            model_role="flattened_indexing",
            branch_intent=["take", "flattened_indices", "boundary_index"],
        ))
        result.append(case(
            runner="indexing.masked_select",
            family="indexing",
            name=f"masked_select_{rows}x{cols}",
            shape={"dims": [rows, cols], "step": 3},
            stride_mode="transposed" if rows % 2 else "contiguous",
            layout="transposed" if rows % 2 else "contiguous",
            model_role="boolean_masking",
            branch_intent=["masked_select", "bool_mask", "strided_input"],
        ))
    standard = limit(result, 80)
    heavy = []
    for item in standard[:30]:
        clone = dict(item)
        clone["case_id"] = clone["case_id"].replace("_standard", "_heavy")
        clone["resource_tier"] = "heavy"
        clone["semantic_level"] = 8
        heavy.append(clone)
    return standard + heavy
