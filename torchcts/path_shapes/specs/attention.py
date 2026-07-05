"""Attention path-shape specs."""

from torchcts.path_shapes.specs.common import case, limit


def cases():
    result = []
    for sq, sk in [(1, 33), (1, 129), (8, 8), (17, 17), (33, 33), (16, 65), (7, 31), (31, 7)]:
        for head_dim in (32, 64, 80, 96, 128):
            for mask in ("none", "bool", "noncontiguous_bool", "float"):
                causal = sq == sk and mask in {"none", "bool"}
                result.append(case(
                    runner="attention.sdpa",
                    family="attention",
                    name=f"sdpa_sq{sq}_sk{sk}_d{head_dim}_{mask}_{'causal' if causal else 'noncausal'}",
                    shape={"batch": 1 if sq == 1 else 2, "heads": 2, "sq": sq, "sk": sk, "head_dim": head_dim, "causal": causal, "mask": mask},
                    suite="workloads",
                    semantic_level=7,
                    cost_class="small",
                    layout="bhsd",
                    model_role="attention_kernel_selection",
                    branch_intent=["sdpa", f"sq_{sq}", f"sk_{sk}", f"head_dim_{head_dim}", f"mask_{mask}"],
                ))
    standard = limit(result, 80)
    heavy = []
    for item in result[80:150]:
        clone = dict(item)
        clone["case_id"] = clone["case_id"].replace("_standard", "_heavy")
        clone["resource_tier"] = "heavy"
        clone["semantic_level"] = 8
        clone["cost_class"] = "medium"
        clone["limits"] = {"max_tensor_mb": 64, "max_workspace_mb": 256}
        heavy.append(clone)
    return standard + heavy
