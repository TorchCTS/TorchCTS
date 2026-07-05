"""Spatial path-shape specs."""

from torchcts.path_shapes.specs.common import case, limit


def cases():
    result = []
    images = [
        (1, 3, 15, 17),
        (2, 5, 31, 33),
        (1, 7, 16, 19),
        (3, 4, 29, 25),
        (2, 8, 23, 27),
        (1, 11, 13, 21),
        (4, 3, 32, 30),
        (2, 6, 17, 35),
        (1, 9, 37, 19),
        (3, 2, 11, 15),
    ]
    for n, c, h, w in images:
        for layout in ("nchw", "channels_last"):
            result.append(case(
                runner="spatial.avg_pool2d",
                family="spatial",
                name=f"avg_pool2d_n{n}_c{c}_h{h}_w{w}_{layout}",
                shape={"n": n, "c": c, "h": h, "w": w, "kernel": 3, "stride": 2, "padding": 1, "ceil_mode": True, "count_include_pad": True},
                layout=layout,
                stride_mode="channels_last" if layout == "channels_last" else "contiguous",
                model_role="vision_pooling",
                branch_intent=["avg_pool2d", "ceil_mode", "count_include_pad", layout],
            ))
            result.append(case(
                runner="spatial.max_pool2d",
                family="spatial",
                name=f"max_pool2d_n{n}_c{c}_h{h}_w{w}_{layout}",
                shape={"n": n, "c": c, "h": h, "w": w, "kernel": 3, "stride": 2, "padding": 1, "ceil_mode": True},
                layout=layout,
                stride_mode="channels_last" if layout == "channels_last" else "contiguous",
                model_role="vision_pooling",
                branch_intent=["max_pool2d", "ceil_mode", layout],
            ))
            result.append(case(
                runner="spatial.adaptive_avg_pool2d",
                family="spatial",
                name=f"adaptive_avg_pool2d_n{n}_c{c}_h{h}_w{w}_{layout}",
                shape={"n": n, "c": c, "h": h, "w": w, "output_size": [1, 3]},
                layout=layout,
                stride_mode="channels_last" if layout == "channels_last" else "contiguous",
                model_role="adaptive_pooling",
                branch_intent=["adaptive_avg_pool2d", "odd_output", layout],
            ))
            for mode in ("nearest", "bilinear"):
                result.append(case(
                    runner="spatial.interpolate",
                    family="spatial",
                    name=f"interpolate_{mode}_n{n}_c{c}_h{h}_w{w}_{layout}",
                    shape={"n": n, "c": c, "h": h, "w": w, "size": [h + 3, w + 5], "mode": mode, "align_corners": False},
                    layout=layout,
                    stride_mode="channels_last" if layout == "channels_last" else "contiguous",
                    model_role="resize_path",
                    branch_intent=["interpolate", mode, "odd_output", layout],
                ))
    standard = limit(result, 65)
    heavy = []
    for item in result[65:100]:
        clone = dict(item)
        clone["case_id"] = clone["case_id"].replace("_standard", "_heavy")
        clone["resource_tier"] = "heavy"
        clone["semantic_level"] = 8
        heavy.append(clone)
    return standard + heavy
