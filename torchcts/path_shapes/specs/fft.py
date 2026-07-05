"""FFT path-shape specs."""

from torchcts.path_shapes.specs.common import case, limit


def cases():
    result = []
    lengths = [16, 17, 32, 33, 64, 65, 75, 97, 128]
    for length in lengths:
        for batch in (1, 3, 7):
            for runner in ("fft.rfft", "fft.irfft"):
                result.append(case(
                    runner=runner,
                    family="fft",
                    name=f"{runner.split('.')[1]}_b{batch}_l{length}",
                    shape={"batch": batch, "length": length},
                    stride_mode="every_other" if length in {17, 97} else "contiguous",
                    layout="strided" if length in {17, 97} else "contiguous",
                    model_role="fft_plan_boundary",
                    branch_intent=[runner.split(".")[1], f"length_{length}", f"batch_{batch}", "prime_or_tail"],
                ))
            result.append(case(
                runner="fft.fft",
                family="fft",
                name=f"fft_b{batch}_l{length}",
                shape={"batch": batch, "length": length},
                model_role="complex_fft_plan",
                branch_intent=["fft", f"length_{length}", f"batch_{batch}"],
            ))
    for h, w in [(8, 9), (15, 17), (16, 16), (7, 11)]:
        result.append(case(
            runner="fft.fft2",
            family="fft",
            name=f"fft2_h{h}_w{w}",
            shape={"batch": 2, "h": h, "w": w},
            model_role="two_dimensional_fft",
            branch_intent=["fft2", f"h_{h}", f"w_{w}"],
        ))
    standard = limit(result, 45)
    heavy = []
    for item in result[45:85]:
        clone = dict(item)
        clone["case_id"] = clone["case_id"].replace("_standard", "_heavy")
        clone["resource_tier"] = "heavy"
        clone["semantic_level"] = 8
        clone["cost_class"] = "small"
        heavy.append(clone)
    return standard + heavy
