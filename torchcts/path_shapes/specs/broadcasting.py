"""Broadcasting path-shape specs."""

from torchcts.path_shapes.specs.common import case, limit


def cases():
    result = []
    shape_pairs = [
        ([1, 33], [17, 33], [17, 33], [17, 33]),
        ([3, 1, 65], [1, 7, 65], [3, 7, 65], [3, 7, 65]),
        ([2, 5, 1, 17], [1, 5, 9, 1], [2, 5, 9, 17], [2, 5, 9, 17]),
        ([1, 1, 31], [4, 7, 31], [4, 7, 31], [4, 7, 31]),
        ([8, 1], [1, 11], [8, 11], [8, 11]),
        ([2, 1, 3, 1], [1, 5, 1, 7], [2, 5, 3, 7], [2, 5, 3, 7]),
        ([1, 97], [9, 97], [9, 97], [9, 97]),
        ([3, 4, 1], [1, 4, 13], [3, 4, 13], [3, 4, 13]),
        ([5, 1, 1], [1, 6, 7], [5, 6, 7], [5, 6, 7]),
        ([1, 2, 3, 1], [4, 1, 1, 5], [4, 2, 3, 5], [4, 2, 3, 5]),
        ([7, 1, 9], [1, 11, 1], [7, 11, 9], [7, 11, 9]),
        ([1, 3, 1, 13], [2, 1, 5, 1], [2, 3, 5, 13], [2, 3, 5, 13]),
        ([6, 1], [1, 17], [6, 17], [6, 17]),
    ]
    for index, (a_shape, b_shape, expanded_b_shape, mask_shape) in enumerate(shape_pairs):
        for runner, op_name in (("broadcasting.add", "add"), ("broadcasting.mul", "mul")):
            for stride_mode in ("contiguous", "zero_stride"):
                result.append(case(
                    runner=runner,
                    family="broadcasting",
                    name=f"{op_name}_case{index}_{stride_mode}",
                    shape={"a_shape": a_shape, "b_shape": b_shape, "expanded_b_shape": expanded_b_shape},
                    layout="broadcast",
                    stride_mode=stride_mode,
                    model_role="elementwise_broadcast",
                    branch_intent=[op_name, "implicit_expand", stride_mode],
                ))
        result.append(case(
            runner="broadcasting.where",
            family="broadcasting",
            name=f"where_case{index}",
            shape={"a_shape": a_shape, "b_shape": b_shape, "expanded_b_shape": expanded_b_shape, "mask_shape": mask_shape},
            layout="broadcast",
            stride_mode="broadcast",
            model_role="masked_blend_broadcast",
            branch_intent=["where", "mask_broadcast", "implicit_expand"],
        ))
        result.append(case(
            runner="broadcasting.masked_fill",
            family="broadcasting",
            name=f"masked_fill_case{index}",
            shape={"dims": expanded_b_shape, "mask_shape": mask_shape, "step": 3, "value": 2.5},
            layout="broadcast_mask",
            stride_mode="contiguous",
            model_role="masked_update",
            branch_intent=["masked_fill", "mask_shape", "scalar_fill"],
        ))

    standard = limit(result, 55)
    heavy = []
    for item in result[55:75]:
        clone = dict(item)
        clone["case_id"] = clone["case_id"].replace("_standard", "_heavy")
        clone["resource_tier"] = "heavy"
        clone["semantic_level"] = 8
        heavy.append(clone)
    return standard + heavy
