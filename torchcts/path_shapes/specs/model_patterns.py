"""Composed model-pattern path-shape specs."""

from torchcts.path_shapes.specs.common import case, limit


def cases():
    result = []

    for n, c, h, w, out_channels in [
        (1, 3, 15, 17, 8),
        (2, 5, 31, 33, 16),
        (1, 7, 16, 19, 14),
        (3, 4, 29, 25, 12),
        (2, 8, 23, 27, 16),
        (1, 11, 13, 21, 22),
        (2, 6, 17, 35, 18),
        (1, 9, 37, 19, 18),
    ]:
        for layout in ("nchw", "channels_last"):
            result.append(case(
                runner="model_patterns.vision_block",
                family="model_patterns",
                name=f"vision_block_n{n}_c{c}_h{h}_w{w}_out{out_channels}_{layout}",
                shape={"n": n, "c": c, "h": h, "w": w, "out_channels": out_channels},
                suite="workloads",
                layout=layout,
                stride_mode="channels_last" if layout == "channels_last" else "contiguous",
                model_role="vision_conv_relu_block",
                branch_intent=["vision_block", "conv_bias_relu", layout],
            ))
            result.append(case(
                runner="model_patterns.depthwise_separable",
                family="model_patterns",
                name=f"depthwise_separable_n{n}_c{c}_h{h}_w{w}_out{out_channels}_{layout}",
                shape={"n": n, "c": c, "h": h, "w": w, "out_channels": out_channels},
                suite="workloads",
                layout=layout,
                stride_mode="channels_last" if layout == "channels_last" else "contiguous",
                model_role="mobile_vision_block",
                branch_intent=["depthwise_separable", "groups_equal_channels", layout],
            ))

    for n, c, h, w, kernel, stride, embed_dim in [
        (1, 3, 16, 16, 4, 4, 32),
        (2, 3, 18, 22, 3, 3, 24),
        (1, 5, 17, 19, 2, 2, 20),
        (3, 4, 21, 25, 5, 4, 28),
        (2, 8, 15, 23, 3, 2, 36),
        (1, 3, 31, 33, 7, 4, 40),
        (2, 6, 13, 15, 3, 3, 18),
        (1, 9, 27, 29, 3, 2, 48),
    ]:
        result.append(case(
            runner="model_patterns.patch_embedding",
            family="model_patterns",
            name=f"patch_embedding_n{n}_c{c}_h{h}_w{w}_k{kernel}_s{stride}_e{embed_dim}",
            shape={"n": n, "c": c, "h": h, "w": w, "kernel": kernel, "stride": stride, "embed_dim": embed_dim},
            suite="workloads",
            model_role="vision_transformer_patch_embedding",
            branch_intent=["patch_embedding", f"kernel_{kernel}", f"stride_{stride}", "flatten_transpose"],
        ))

    for batch in (1, 2, 4):
        for heads in (1, 2, 4):
            for sq, sk, head_dim, mask, causal in [
                (1, 33, 32, "none", False),
                (17, 17, 40, "bool", False),
                (31, 33, 64, "float", False),
                (33, 33, 32, "none", True),
                (8, 65, 24, "noncontiguous_bool", False),
            ]:
                result.append(case(
                    runner="model_patterns.transformer_block_fragment",
                    family="model_patterns",
                    name=f"transformer_b{batch}_h{heads}_sq{sq}_sk{sk}_d{head_dim}_mask{mask}_causal{int(causal)}",
                    shape={"batch": batch, "heads": heads, "sq": sq, "sk": sk, "head_dim": head_dim, "mask": mask, "causal": causal},
                    suite="workloads",
                    cost_class="small",
                    layout="bhsd",
                    stride_mode="contiguous",
                    model_role="transformer_attention_fragment",
                    branch_intent=["transformer_fragment", "sdpa", f"sq_{sq}", f"sk_{sk}", f"mask_{mask}"],
                ))

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
        result.append(case(
            runner="model_patterns.residual_norm",
            family="model_patterns",
            name=f"residual_norm_dims{'x'.join(map(str, dims))}_norm{'x'.join(map(str, normalized_shape))}",
            shape={"dims": list(dims), "normalized_shape": list(normalized_shape)},
            suite="workloads",
            model_role="residual_layernorm_block",
            branch_intent=["residual_add", "layer_norm", "transformer_mlp_boundary"],
        ))

    standard = limit(result, 55)
    heavy = []
    for item in result[55:95]:
        clone = dict(item)
        clone["case_id"] = clone["case_id"].replace("_standard", "_heavy")
        clone["resource_tier"] = "heavy"
        clone["semantic_level"] = 8
        clone["cost_class"] = "small"
        heavy.append(clone)
    return standard + heavy
