"""Convolution path-shape specs."""

from torchcts.path_shapes.specs.common import case, limit


def cases():
    conv2d = []
    conv1d = []
    conv3d = []
    conv_transpose2d = []
    spatial = [
        (1, 3, 15, 17),
        (2, 5, 31, 33),
        (1, 7, 16, 19),
        (3, 4, 29, 25),
        (2, 8, 13, 21),
        (1, 6, 23, 27),
        (2, 9, 19, 35),
        (1, 11, 37, 13),
        (4, 2, 11, 15),
        (2, 10, 25, 31),
    ]
    for n, c, h, w in spatial:
        for kernel in (1, 3):
            for stride in (1, 2):
                for layout in ("nchw", "channels_last"):
                    groups = c if kernel == 3 and c in {3, 5, 7, 8} else 1
                    conv2d.append(case(
                        runner="convolution.conv2d",
                        family="convolution",
                        name=f"conv2d_n{n}_c{c}_h{h}_w{w}_k{kernel}_s{stride}_{layout}_g{groups}",
                        shape={"n": n, "c": c, "h": h, "w": w, "out_channels": c if groups == c else c + 2, "kernel": kernel, "stride": stride, "padding": kernel // 2, "dilation": 1, "groups": groups},
                        layout=layout,
                        stride_mode="channels_last" if layout == "channels_last" else "contiguous",
                        model_role="vision_convolution",
                        branch_intent=["conv2d", f"kernel_{kernel}", f"stride_{stride}", f"groups_{groups}", layout],
                    ))
    for n, c, length in [(1, 3, 31), (2, 5, 65), (3, 4, 33), (1, 7, 97)]:
        for stride in (1, 2):
            conv1d.append(case(
                runner="convolution.conv1d",
                family="convolution",
                name=f"conv1d_n{n}_c{c}_l{length}_s{stride}",
                shape={"n": n, "c": c, "length": length, "out_channels": c + 1, "kernel": 3, "stride": stride, "padding": 1, "dilation": 1, "groups": 1},
                model_role="temporal_convolution",
                branch_intent=["conv1d", f"length_{length}", f"stride_{stride}"],
            ))
    for n, c, d, h, w in [(1, 2, 7, 9, 11), (1, 3, 5, 7, 9), (2, 2, 4, 6, 8)]:
        conv3d.append(case(
            runner="convolution.conv3d",
            family="convolution",
            name=f"conv3d_n{n}_c{c}_d{d}_h{h}_w{w}",
            shape={"n": n, "c": c, "d": d, "h": h, "w": w, "out_channels": c + 1, "kernel": 3, "stride": 1, "padding": 1, "dilation": 1, "groups": 1},
            cost_class="small",
            model_role="volumetric_convolution",
            branch_intent=["conv3d", "odd_volume"],
        ))
    for n, c, h, w in [(1, 3, 8, 9), (2, 5, 13, 15), (1, 4, 7, 11)]:
        conv_transpose2d.append(case(
            runner="convolution.conv_transpose2d",
            family="convolution",
            name=f"conv_transpose2d_n{n}_c{c}_h{h}_w{w}",
            shape={"n": n, "c": c, "h": h, "w": w, "out_channels": c + 1, "kernel": 3, "stride": 2, "padding": 1, "output_padding": 1, "dilation": 1, "groups": 1},
            model_role="decoder_upsample",
            branch_intent=["conv_transpose2d", "output_padding", "stride2"],
        ))
    standard = (
        limit(conv2d, 76)
        + conv1d
        + conv3d
        + conv_transpose2d
    )
    heavy = []
    for item in standard[:50]:
        clone = dict(item)
        clone["case_id"] = clone["case_id"].replace("_standard", "_heavy")
        clone["resource_tier"] = "heavy"
        clone["cost_class"] = "medium"
        clone["limits"] = {"max_tensor_mb": 64, "max_workspace_mb": 256}
        heavy.append(clone)
    return standard + heavy
