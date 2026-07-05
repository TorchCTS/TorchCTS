"""Normalization path-shape specs."""

from torchcts.path_shapes.specs.common import case, limit


def cases():
    layer_norm = []
    group_norm = []
    batch_norm = []
    for dims, normalized_shape in [
        ((2, 7, 5, 9), (5, 9)),
        ((3, 17), (17,)),
        ((2, 3, 33), (33,)),
        ((4, 8, 16), (8, 16)),
        ((2, 5, 11, 13), (11, 13)),
        ((3, 4, 7), (4, 7)),
        ((2, 9, 5), (5,)),
        ((5, 3, 8, 6), (8, 6)),
        ((2, 6, 7, 9), (7, 9)),
        ((4, 12), (12,)),
    ]:
        for eps in (1e-5, 1e-3):
            layer_norm.append(case(
                runner="normalization.layer_norm",
                family="normalization",
                name=f"layer_norm_dims{'x'.join(map(str, dims))}_norm{'x'.join(map(str, normalized_shape))}_eps{str(eps).replace('-', 'm')}",
                shape={"dims": list(dims), "normalized_shape": list(normalized_shape), "eps": eps},
                model_role="transformer_norm",
                branch_intent=["layer_norm", "multi_dim" if len(normalized_shape) > 1 else "single_dim", f"eps_{eps}"],
            ))
    for c in (4, 5, 6, 8, 9, 16, 32):
        groups_list = [1, c]
        if c % 2 == 0:
            groups_list.append(2)
        if c % 3 == 0:
            groups_list.append(3)
        for groups in groups_list:
            for layout in ("nchw", "channels_last"):
                group_norm.append(case(
                    runner="normalization.group_norm",
                    family="normalization",
                    name=f"group_norm_c{c}_g{groups}_{layout}",
                    shape={"n": 2, "c": c, "h": 15, "w": 17, "groups": groups, "eps": 1e-5},
                    layout=layout,
                    stride_mode="channels_last" if layout == "channels_last" else "contiguous",
                    model_role="vision_norm",
                    branch_intent=["group_norm", f"channels_{c}", f"groups_{groups}", layout],
                ))
    for c in (3, 5, 8, 16, 32):
        for training in (False, True):
            for layout in ("nchw", "channels_last"):
                batch_norm.append(case(
                    runner="normalization.batch_norm",
                    family="normalization",
                    name=f"batch_norm_c{c}_{'train' if training else 'eval'}_{layout}",
                    shape={"n": 3, "c": c, "h": 11, "w": 13, "training": training, "eps": 1e-5},
                    layout=layout,
                    stride_mode="channels_last" if layout == "channels_last" else "contiguous",
                    model_role="batch_norm",
                    branch_intent=["batch_norm", "training" if training else "eval", layout],
                ))
    standard = limit(layer_norm, 20) + limit(group_norm, 25) + limit(batch_norm, 20)
    heavy = []
    for item in (layer_norm[:8] + group_norm[25:34] + batch_norm[:8]):
        clone = dict(item)
        clone["case_id"] = clone["case_id"].replace("_standard", "_heavy")
        clone["resource_tier"] = "heavy"
        clone["semantic_level"] = 8
        heavy.append(clone)
    return standard + heavy
