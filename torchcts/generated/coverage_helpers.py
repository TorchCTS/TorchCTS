# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import pytest
import torch

from torchcts.core.coverage import generated_entries_for
from torchcts.core.comparer import compare_inf_propagation, compare_nan_propagation
from torchcts.core.device import synchronize
from torchcts.core.oracles import OracleUnavailable, run_oracle_for_surface
from torchcts.core.opinfo_adapter import (
    InputCondition,
    get_forward_op_tests,
    get_live_opinfo,
    get_op_sample_inputs,
    is_cpu_reference_failure,
    prepare_sample,
    record_known_failure,
    str_to_dtype,
)
from torchcts.core.reference_oracles import matmul_family_reference
from torchcts.sample_generation import (
    bitwise_args_and_template as sample_bitwise_args_and_template,
    bitwise_dtype_supported as sample_bitwise_dtype_supported,
    convolution_sample as sample_convolution,
    elementwise_sample as sample_elementwise,
    factory_args as sample_factory_args,
    factory_dtype_supported as sample_factory_dtype_supported,
    factory_out_args as sample_factory_out_args,
    factory_out_call_parts as sample_factory_out_call_parts,
    factory_out_shape as sample_factory_out_shape,
    fft_sample as sample_fft,
    foreach_sample as sample_foreach,
    get_inputs_for_op,
    grid_backward_sample as sample_grid_backward,
    grid_sample as sample_grid,
    indexing_sample as sample_indexing,
    input_conditions_for as sample_input_conditions_for,
    linalg_sample as sample_linalg,
    loss_sample as sample_loss,
    metadata_sample as sample_metadata,
    manifest_dtype_items as sample_manifest_dtype_items,
    multi_output_reduction_sample as sample_multi_output_reduction,
    padding_sample as sample_padding,
    pooling_sample as sample_pooling,
    reduction_sample as sample_reduction,
    rng_call_parts as sample_rng_call_parts,
    rng_uses_target_device_generator as sample_rng_uses_target_device_generator,
    rng_output_shape as sample_rng_output_shape,
    rnn_cell_sample as sample_rnn_cell,
    SampleGenerationError,
    sample_case_specs_for_entry,
    shape_sample as sample_shape,
    special_math_sample as sample_special_math,
    upsample_sample as sample_upsample,
)


EXCLUDED_OR_PENDING_STATUSES = {
    "excluded",
    "excluded_framework_plumbing",
    "excluded_deprecated_or_removed",
    "excluded_unsupported_public_api",
    "excluded_distributed_scope",
    "excluded_host_storage",
    "pending_oracle",
    "pending_backend_pack",
    "pending_property",
}


def generated_cases(surface_kind: str) -> list[dict | None]:
    cases = generated_entries_for(surface_kind)
    return cases or [None]


def generated_case_id(entry: dict | None) -> str:
    if entry is None:
        return "no-audit"
    base = entry["name"].replace("aten::", "").replace("/", "_")
    level = entry.get("semantic_level")
    return f"{base}[L{level}]" if level is not None else base


def skip_until_strategy_exists(entry: dict | None, strategy_name: str) -> None:
    if entry is None:
        pytest.skip(f"No default coverage audit found for generated {strategy_name} tests")

    status = entry.get("status")
    if status == "unknown":
        pytest.skip("coverage_unknown")
    if status == "excluded":
        pytest.skip("coverage_excluded")
    if status in EXCLUDED_OR_PENDING_STATUSES:
        pytest.skip(status)

    pytest.skip(
        "coverage_strategy_pending: "
        f"{strategy_name} strategy is not implemented for {entry.get('name')}"
    )


def run_oracle_strategy(entry: dict | None, device: str) -> None:
    if entry is None:
        pytest.skip("No default coverage audit found for oracle-backed tests")

    status = entry.get("status")
    if status == "unknown":
        pytest.skip("coverage_unknown")
    if status == "excluded":
        pytest.skip("coverage_excluded")
    if status in EXCLUDED_OR_PENDING_STATUSES:
        pytest.skip(status)
    if status not in {"covered_oracle", "covered_backend_pack", "covered_property"}:
        pytest.skip(f"coverage_strategy_pending: {entry.get('name')} is not oracle-backed")

    try:
        run_oracle_for_surface(entry["name"], device)
    except OracleUnavailable as exc:
        pytest.skip(str(exc))


_FORWARD_CASE_CACHE = {}


def _forward_cases_for_op(op_name: str, manifest: dict) -> list[tuple[str, str]]:
    key = (id(manifest), op_name)
    if key not in _FORWARD_CASE_CACHE:
        _FORWARD_CASE_CACHE[key] = [
            (dtype_str, input_condition)
            for candidate, dtype_str, input_condition in get_forward_op_tests(manifest)
            if candidate == op_name
        ]
    return _FORWARD_CASE_CACHE[key]


def _manifest_dtype_items(manifest: dict) -> list[tuple[torch.dtype, str]]:
    return sample_manifest_dtype_items(manifest)


def _manifest_dtype_items_or(manifest: dict, fallback: list[torch.dtype]) -> list[tuple[torch.dtype, str]]:
    items = _manifest_dtype_items(manifest)
    if items:
        return items
    return [(dtype, str(dtype)) for dtype in fallback]


def _move_to_device(obj, device: str):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(_move_to_device(item, device) for item in obj)
    if isinstance(obj, list):
        return [_move_to_device(item, device) for item in obj]
    if isinstance(obj, dict):
        return {key: _move_to_device(value, device) for key, value in obj.items()}
    return obj


def _assert_out_identity(actual, out, dispatcher_name: str) -> None:
    if actual is not out:
        raise AssertionError(f"{dispatcher_name} returned a different object than the provided out tensor")
    if isinstance(actual, torch.Tensor) and actual.data_ptr() != out.data_ptr():
        raise AssertionError(f"{dispatcher_name} returned a tensor with different storage than out")


def _assert_inplace_identity(actual, mutated_input, dispatcher_name: str) -> None:
    if actual is not mutated_input:
        raise AssertionError(f"{dispatcher_name} returned a different object than the mutated input tensor")
    if isinstance(actual, torch.Tensor) and actual.data_ptr() != mutated_input.data_ptr():
        raise AssertionError(f"{dispatcher_name} returned a tensor with different storage than the mutated input")


def _clone_writable_input(obj):
    if not isinstance(obj, torch.Tensor):
        return obj
    return obj.detach().clone(memory_format=torch.preserve_format)


def _shares_storage_alias(a, b) -> bool:
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return False
    try:
        return bool(torch._C._is_alias_of(a, b))
    except Exception:
        return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def _storage_data_ptr(obj) -> int | None:
    if isinstance(obj, torch.Tensor):
        try:
            return int(obj.untyped_storage().data_ptr())
        except Exception:
            return None
    if hasattr(obj, "data_ptr"):
        try:
            return int(obj.data_ptr())
        except Exception:
            return None
    return None


def _assert_storage_set_alias(entry: dict, result: torch.Tensor, call_args: tuple, dispatcher_name: str) -> None:
    name = entry["name"]
    if entry.get("surface_kind") == "out_variant":
        return
    if name in {"aten::set", "aten::set.out", "aten::set_"}:
        return
    if name == "aten::set_data" or ".source_" in name:
        if len(call_args) < 2:
            raise AssertionError(f"{dispatcher_name} storage-alias sample did not include a source argument")
        source = call_args[1]
    else:
        return

    result_ptr = _storage_data_ptr(result)
    source_ptr = _storage_data_ptr(source)
    if result_ptr is None or source_ptr is None:
        raise AssertionError(f"{dispatcher_name} could not inspect storage alias pointers")
    if result_ptr != source_ptr:
        raise AssertionError(f"{dispatcher_name} did not rebind to the expected source storage")


def _assert_view_metadata_matches(actual, expected, dispatcher_name: str) -> None:
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(f"{dispatcher_name} shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}")
    if actual.dtype != expected.dtype:
        raise AssertionError(f"{dispatcher_name} dtype mismatch: {actual.dtype} vs {expected.dtype}")
    if tuple(actual.stride()) != tuple(expected.stride()):
        raise AssertionError(f"{dispatcher_name} stride mismatch: {tuple(actual.stride())} vs {tuple(expected.stride())}")
    if actual.storage_offset() != expected.storage_offset():
        raise AssertionError(
            f"{dispatcher_name} storage_offset mismatch: {actual.storage_offset()} vs {expected.storage_offset()}"
        )


def _first_index(tensor: torch.Tensor):
    if tensor.ndim == 0:
        return ()
    return (0,) * tensor.ndim


def _replacement_value(current):
    if isinstance(current, bool):
        return not current
    if isinstance(current, complex):
        if current != complex(3.25, -1.5):
            return complex(3.25, -1.5)
        return complex(-2.5, 4.0)
    try:
        if current != 3:
            return 3
        return 7
    except Exception:
        return 3


def _tensor_content_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    a_cpu = a.detach().cpu()
    b_cpu = b.detach().cpu()
    if a_cpu.shape != b_cpu.shape or a_cpu.dtype != b_cpu.dtype:
        return False
    if a_cpu.is_complex():
        a_cpu = torch.view_as_real(a_cpu)
        b_cpu = torch.view_as_real(b_cpu)
    if a_cpu.is_floating_point():
        a_nan = torch.isnan(a_cpu)
        b_nan = torch.isnan(b_cpu)
        if not torch.equal(a_nan, b_nan):
            return False
        a_cpu = torch.nan_to_num(a_cpu)
        b_cpu = torch.nan_to_num(b_cpu)
    return bool(torch.equal(a_cpu, b_cpu))


def _assert_mutation_reflects(view: torch.Tensor, base: torch.Tensor, dispatcher_name: str, device: str) -> None:
    if view.numel() == 0:
        return
    if view.is_conj() or view.is_neg():
        return
    before = base.detach().clone(memory_format=torch.preserve_format)
    index = _first_index(view)
    current = view[index].detach().cpu().item()
    try:
        view[index] = _replacement_value(current)
        synchronize(device)
    except Exception:
        return
    if _tensor_content_equal(base, before):
        raise AssertionError(f"{dispatcher_name} returned an alias but mutation did not reflect in the input")


def _compare_special_tier(actual, expected, input_condition: str) -> None:
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} vs {expected.shape}"
        assert actual.dtype == expected.dtype, f"Dtype mismatch: {actual.dtype} vs {expected.dtype}"
        if input_condition == InputCondition.HAS_NAN:
            compare_nan_propagation(actual, expected)
            compare_inf_propagation(actual, expected)
        elif input_condition == InputCondition.HAS_INF:
            compare_inf_propagation(actual, expected)
        return
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), f"Output sequence lengths differ: {len(actual)} vs {len(expected)}"
        for actual_item, expected_item in zip(actual, expected):
            _compare_special_tier(actual_item, expected_item, input_condition)
        return
    if isinstance(actual, dict) and isinstance(expected, dict):
        assert set(actual) == set(expected), f"Output dict keys differ: {set(actual)} vs {set(expected)}"
        for key in actual:
            _compare_special_tier(actual[key], expected[key], input_condition)


def _run_opinfo_out_sample(
    *,
    entry: dict,
    op_fn,
    sample,
    input_condition: str,
    dtype,
    dtype_str: str,
    device: str,
    compare,
    category: str,
) -> bool:
    cpu_input = _move_to_device(sample.input, "cpu")
    cpu_args = _move_to_device(sample.args, "cpu")
    cpu_kwargs = _move_to_device(sample.kwargs, "cpu")

    try:
        expected = op_fn(cpu_input, *cpu_args, **cpu_kwargs)
    except Exception as exc:
        if is_cpu_reference_failure(exc):
            opinfo_name = entry["generated"]["strategy"]["opinfo_name"]
            record_known_failure("forward", opinfo_name, dtype_str, f"{type(exc).__name__}: {exc}")
        return False

    if not isinstance(expected, torch.Tensor):
        pytest.skip(f"coverage_strategy_pending: {entry['name']} produced non-tensor output")

    dev_input = _move_to_device(sample.input, device)
    dev_args = _move_to_device(sample.args, device)
    dev_kwargs = _move_to_device(sample.kwargs, device)
    out = torch.empty_strided(
        tuple(expected.shape),
        tuple(expected.stride()),
        dtype=expected.dtype,
        device=device,
    )

    try:
        actual = op_fn(dev_input, *dev_args, **dev_kwargs, out=out)
        synchronize(device)
    except Exception as exc:
        if input_condition != InputCondition.CLEAN:
            raise AssertionError(
                f"{entry['name']} device raised {type(exc).__name__} for {input_condition} "
                "after CPU functional reference succeeded"
            ) from exc
        raise RuntimeError(f"{entry['name']} out= execution failed on {device}: {exc}") from exc

    _assert_out_identity(actual, out, entry["name"])
    if input_condition != InputCondition.CLEAN:
        _compare_special_tier(actual, expected, input_condition)
    else:
        compare(actual, expected, category=category, dtype=dtype)
    return True


def run_opinfo_out_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    if entry is None:
        pytest.skip("No default coverage audit found for generated out_variant tests")
    if entry.get("status") == "unknown":
        pytest.skip("coverage_unknown")
    if entry.get("status") == "excluded":
        pytest.skip("coverage_excluded")

    strategy = entry.get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") != "opinfo_out":
        skip_until_strategy_exists(entry, "out_variant")

    op_name = strategy["opinfo_name"]
    op_info = get_live_opinfo(op_name)
    if op_info is None:
        pytest.skip(f"coverage_strategy_pending: no live OpInfo for {op_name}")
    if not getattr(op_info, "supports_out", False):
        pytest.skip(f"coverage_strategy_pending: OpInfo {op_name} does not support out=")

    cases = _forward_cases_for_op(op_name, manifest)
    if not cases:
        pytest.skip(f"coverage_strategy_pending: no manifest-enabled OpInfo cases for {op_name}")

    max_samples = manifest.get("max_samples", 10)
    max_samples_ieee = manifest.get("max_samples_ieee754", 3)
    ieee754_seed = manifest.get("ieee754_seed", 67)
    tested_any = False

    for dtype_str, input_condition in cases:
        dtype = str_to_dtype(dtype_str)
        if dtype is None:
            continue
        sample_cap = max_samples_ieee if input_condition != InputCondition.CLEAN else max_samples
        passed_count = 0
        for sample_index, raw_sample in enumerate(get_op_sample_inputs(op_name, device, dtype)):
            if sample_cap and passed_count >= sample_cap:
                break
            sample = prepare_sample(
                raw_sample,
                input_condition,
                ieee754_seed=ieee754_seed,
                sample_index=sample_index,
                op_name=op_name,
            )
            if _run_opinfo_out_sample(
                entry=entry,
                op_fn=op_info.op,
                sample=sample,
                input_condition=input_condition,
                dtype=dtype,
                dtype_str=dtype_str,
                device=device,
                compare=compare,
                category="elementwise",
            ):
                tested_any = True
                passed_count += 1

    if not tested_any:
        pytest.skip(f"coverage_strategy_pending: all generated out= samples skipped for {op_name}")


def _run_opinfo_inplace_unary_sample(
    *,
    entry: dict,
    inplace_fn,
    sample,
    input_condition: str,
    dtype,
    device: str,
    compare,
    category: str,
) -> bool:
    cpu_input = _move_to_device(sample.input, "cpu")
    if not isinstance(cpu_input, torch.Tensor):
        pytest.skip(f"coverage_strategy_pending: {entry['name']} sample input is not a tensor")
    cpu_input = _clone_writable_input(cpu_input)
    cpu_args = _move_to_device(sample.args, "cpu")
    cpu_kwargs = _move_to_device(sample.kwargs, "cpu")

    try:
        cpu_actual = inplace_fn(cpu_input, *cpu_args, **cpu_kwargs)
    except Exception:
        return False

    _assert_inplace_identity(cpu_actual, cpu_input, entry["name"])

    dev_input = _move_to_device(sample.input, device)
    if not isinstance(dev_input, torch.Tensor):
        pytest.skip(f"coverage_strategy_pending: {entry['name']} sample input is not a tensor")
    dev_input = _clone_writable_input(dev_input)
    dev_args = _move_to_device(sample.args, device)
    dev_kwargs = _move_to_device(sample.kwargs, device)

    try:
        actual = inplace_fn(dev_input, *dev_args, **dev_kwargs)
        synchronize(device)
    except Exception as exc:
        if input_condition != InputCondition.CLEAN:
            raise AssertionError(
                f"{entry['name']} device raised {type(exc).__name__} for {input_condition} "
                "after CPU in-place reference succeeded"
            ) from exc
        raise RuntimeError(f"{entry['name']} in-place execution failed on {device}: {exc}") from exc

    _assert_inplace_identity(actual, dev_input, entry["name"])
    if input_condition != InputCondition.CLEAN:
        _compare_special_tier(dev_input, cpu_input, input_condition)
    else:
        compare(dev_input, cpu_input, category=category, dtype=dtype)
    return True


def run_opinfo_inplace_unary_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    if entry is None:
        pytest.skip("No default coverage audit found for generated in-place tests")
    if entry.get("status") == "unknown":
        pytest.skip("coverage_unknown")
    if entry.get("status") == "excluded":
        pytest.skip("coverage_excluded")

    strategy = entry.get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") != "opinfo_inplace_unary":
        skip_until_strategy_exists(entry, "inplace")

    op_name = strategy["opinfo_name"]
    op_info = get_live_opinfo(op_name)
    if op_info is None:
        pytest.skip(f"coverage_strategy_pending: no live OpInfo for {op_name}")
    inplace_fn = getattr(op_info, "inplace_variant", None)
    if inplace_fn is None:
        pytest.skip(f"coverage_strategy_pending: OpInfo {op_name} has no in-place variant")

    cases = _forward_cases_for_op(op_name, manifest)
    if not cases:
        pytest.skip(f"coverage_strategy_pending: no manifest-enabled OpInfo cases for {op_name}")

    max_samples = manifest.get("max_samples", 10)
    max_samples_ieee = manifest.get("max_samples_ieee754", 3)
    ieee754_seed = manifest.get("ieee754_seed", 67)
    tested_any = False

    for dtype_str, input_condition in cases:
        dtype = str_to_dtype(dtype_str)
        if dtype is None:
            continue
        sample_cap = max_samples_ieee if input_condition != InputCondition.CLEAN else max_samples
        passed_count = 0
        for sample_index, raw_sample in enumerate(get_op_sample_inputs(op_name, device, dtype)):
            if sample_cap and passed_count >= sample_cap:
                break
            sample = prepare_sample(
                raw_sample,
                input_condition,
                ieee754_seed=ieee754_seed,
                sample_index=sample_index,
                op_name=op_name,
            )
            if _run_opinfo_inplace_unary_sample(
                entry=entry,
                inplace_fn=inplace_fn,
                sample=sample,
                input_condition=input_condition,
                dtype=dtype,
                device=device,
                compare=compare,
                category="elementwise",
            ):
                tested_any = True
                passed_count += 1

    if not tested_any:
        pytest.skip(f"coverage_strategy_pending: all generated in-place samples skipped for {op_name}")


def _run_opinfo_view_alias_sample(
    *,
    entry: dict,
    op_fn,
    sample,
    input_condition: str,
    dtype,
    dtype_str: str,
    device: str,
    compare,
    category: str,
) -> bool:
    cpu_input = _move_to_device(sample.input, "cpu")
    if not isinstance(cpu_input, torch.Tensor):
        pytest.skip(f"coverage_strategy_pending: {entry['name']} sample input is not a tensor")
    cpu_args = _move_to_device(sample.args, "cpu")
    cpu_kwargs = _move_to_device(sample.kwargs, "cpu")

    try:
        expected = op_fn(cpu_input, *cpu_args, **cpu_kwargs)
    except Exception as exc:
        if is_cpu_reference_failure(exc):
            opinfo_name = entry["generated"]["strategy"]["opinfo_name"]
            record_known_failure("forward", opinfo_name, dtype_str, f"{type(exc).__name__}: {exc}")
        return False

    if not isinstance(expected, torch.Tensor):
        pytest.skip(f"coverage_strategy_pending: {entry['name']} produced non-tensor output")

    dev_input = _move_to_device(sample.input, device)
    if not isinstance(dev_input, torch.Tensor):
        pytest.skip(f"coverage_strategy_pending: {entry['name']} sample input is not a tensor")
    dev_args = _move_to_device(sample.args, device)
    dev_kwargs = _move_to_device(sample.kwargs, device)

    try:
        actual = op_fn(dev_input, *dev_args, **dev_kwargs)
        synchronize(device)
    except Exception as exc:
        if input_condition != InputCondition.CLEAN:
            raise AssertionError(
                f"{entry['name']} device raised {type(exc).__name__} for {input_condition} "
                "after CPU view reference succeeded"
            ) from exc
        raise RuntimeError(f"{entry['name']} view execution failed on {device}: {exc}") from exc

    if not isinstance(actual, torch.Tensor):
        raise AssertionError(f"{entry['name']} produced {type(actual).__name__}, expected Tensor")

    _assert_view_metadata_matches(actual, expected, entry["name"])
    expected_alias = _shares_storage_alias(expected, cpu_input)
    actual_alias = _shares_storage_alias(actual, dev_input)
    if actual_alias != expected_alias:
        raise AssertionError(
            f"{entry['name']} alias mismatch: device alias={actual_alias}, CPU alias={expected_alias}"
        )

    if input_condition != InputCondition.CLEAN:
        _compare_special_tier(actual, expected, input_condition)
    else:
        compare(actual, expected, category=category, dtype=dtype)

    if expected_alias and actual_alias:
        _assert_mutation_reflects(actual, dev_input, entry["name"], device)
    return True


def run_opinfo_view_alias_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    if entry is None:
        pytest.skip("No default coverage audit found for generated view/alias tests")
    if entry.get("status") == "unknown":
        pytest.skip("coverage_unknown")
    if entry.get("status") == "excluded":
        pytest.skip("coverage_excluded")

    strategy = entry.get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") == "manual_shape":
        run_manual_shape_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_indexing":
        run_manual_indexing_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") != "opinfo_view_alias":
        skip_until_strategy_exists(entry, "view_alias")

    op_name = strategy["opinfo_name"]
    op_info = get_live_opinfo(op_name)
    if op_info is None:
        pytest.skip(f"coverage_strategy_pending: no live OpInfo for {op_name}")

    cases = _forward_cases_for_op(op_name, manifest)
    if not cases:
        pytest.skip(f"coverage_strategy_pending: no manifest-enabled OpInfo cases for {op_name}")

    max_samples = manifest.get("max_samples", 10)
    max_samples_ieee = manifest.get("max_samples_ieee754", 3)
    ieee754_seed = manifest.get("ieee754_seed", 67)
    tested_any = False

    for dtype_str, input_condition in cases:
        dtype = str_to_dtype(dtype_str)
        if dtype is None:
            continue
        sample_cap = max_samples_ieee if input_condition != InputCondition.CLEAN else max_samples
        passed_count = 0
        for sample_index, raw_sample in enumerate(get_op_sample_inputs(op_name, device, dtype)):
            if sample_cap and passed_count >= sample_cap:
                break
            sample = prepare_sample(
                raw_sample,
                input_condition,
                ieee754_seed=ieee754_seed,
                sample_index=sample_index,
                op_name=op_name,
            )
            if _run_opinfo_view_alias_sample(
                entry=entry,
                op_fn=op_info.op,
                sample=sample,
                input_condition=input_condition,
                dtype=dtype,
                dtype_str=dtype_str,
                device=device,
                compare=compare,
                category="copy",
            ):
                tested_any = True
                passed_count += 1

    if not tested_any:
        pytest.skip(f"coverage_strategy_pending: all generated view/alias samples skipped for {op_name}")


def _dispatcher_callable(entry: dict):
    packet = getattr(torch.ops.aten, entry["base_name"])
    overload = entry.get("overload") or "default"
    return getattr(packet, overload)


def _functional_dispatcher_callable(entry: dict):
    packet = getattr(torch.ops.aten, entry["base_name"])
    overload = entry.get("overload") or "default"
    candidates = []
    if overload.endswith("_out"):
        candidates.append(overload[: -len("_out")])
    if overload == "out":
        candidates.append("default")
        candidates.append("dim")
    candidates.append("default")
    for candidate in candidates:
        if hasattr(packet, candidate):
            return getattr(packet, candidate)
    return _dispatcher_callable(entry)


def _is_integral_or_bool_dtype(dtype: torch.dtype) -> bool:
    return dtype in {
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }


def _bitwise_dtype_supported(family: str, dtype: torch.dtype) -> bool:
    return sample_bitwise_dtype_supported(family, dtype)


def _bitwise_base_tensor(dtype: torch.dtype, device: str) -> torch.Tensor:
    if dtype == torch.bool:
        values = torch.tensor(
            [[True, False, True, False], [False, True, False, True]],
            dtype=dtype,
            device=device,
        )
    else:
        values = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            dtype=dtype,
            device=device,
        )
    return values.t().contiguous().t()


def _bitwise_other_tensor(family: str, dtype: torch.dtype, device: str) -> torch.Tensor:
    if dtype == torch.bool:
        return torch.tensor(
            [[False, True, True, False], [True, False, True, False]],
            dtype=dtype,
            device=device,
        )
    if family in {"bitwise_left_shift", "bitwise_right_shift"}:
        return torch.tensor(
            [[0, 1, 2, 3], [1, 0, 2, 1]],
            dtype=dtype,
            device=device,
        )
    return torch.tensor(
        [[3, 1, 6, 2], [4, 7, 5, 9]],
        dtype=dtype,
        device=device,
    )


def _bitwise_scalar(family: str, dtype: torch.dtype):
    if dtype == torch.bool:
        if family == "bitwise_and":
            return False
        return True
    if family in {"bitwise_left_shift", "bitwise_right_shift"}:
        return 1
    if family == "bitwise_and":
        return 3
    if family == "bitwise_or":
        return 8
    if family == "bitwise_xor":
        return 5
    return 1


def _bitwise_args_and_template(entry: dict, dtype: torch.dtype, device: str):
    return sample_bitwise_args_and_template(entry, dtype, device)


def _assert_exact_tensor_match(actual: torch.Tensor, expected: torch.Tensor, dispatcher_name: str) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"{dispatcher_name} shape mismatch: {actual.shape} vs {expected.shape}")
    if actual.dtype != expected.dtype:
        raise AssertionError(f"{dispatcher_name} dtype mismatch: {actual.dtype} vs {expected.dtype}")
    if not _tensor_content_equal(actual, expected):
        raise AssertionError(f"{dispatcher_name} produced different bitwise values than the CPU reference")


def run_manual_bitwise_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    if entry is None:
        pytest.skip("No default coverage audit found for generated bitwise tests")
    if entry.get("status") == "unknown":
        pytest.skip("coverage_unknown")
    if entry.get("status") == "excluded":
        pytest.skip("coverage_excluded")

    strategy = entry.get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") != "manual_bitwise":
        skip_until_strategy_exists(entry, "bitwise")

    family = strategy["family"]
    callable_op = _dispatcher_callable(entry)
    surface_kind = entry.get("surface_kind")
    tested_any = False

    for dtype, _dtype_str in _manifest_dtype_items(manifest):
        if not _bitwise_dtype_supported(family, dtype):
            continue

        cpu_args, cpu_template = _bitwise_args_and_template(entry, dtype, "cpu")
        dev_args, dev_template = _bitwise_args_and_template(entry, dtype, device)

        try:
            if surface_kind == "out_variant":
                expected = torch.empty_strided(
                    tuple(cpu_template.shape),
                    tuple(cpu_template.stride()),
                    dtype=cpu_template.dtype,
                    device="cpu",
                )
                returned = callable_op(*cpu_args, out=expected)
                _assert_out_identity(returned, expected, entry["name"])
            elif surface_kind == "mutating_or_inplace":
                expected = _clone_writable_input(cpu_args[0])
                expected_before = expected.detach().clone(memory_format=torch.preserve_format)
                returned = callable_op(expected, *cpu_args[1:])
                _assert_inplace_identity(returned, expected, entry["name"])
                expected_changed = not _tensor_content_equal(expected, expected_before)
            else:
                pytest.skip(f"coverage_strategy_pending: unsupported bitwise surface kind {surface_kind}")
        except Exception:
            continue

        try:
            if surface_kind == "out_variant":
                actual = torch.empty_strided(
                    tuple(dev_template.shape),
                    tuple(dev_template.stride()),
                    dtype=dev_template.dtype,
                    device=device,
                )
                returned = callable_op(*dev_args, out=actual)
                synchronize(device)
                _assert_out_identity(returned, actual, entry["name"])
            else:
                actual = _clone_writable_input(dev_args[0])
                before = actual.detach().clone(memory_format=torch.preserve_format)
                returned = callable_op(actual, *dev_args[1:])
                synchronize(device)
                _assert_inplace_identity(returned, actual, entry["name"])
                actual_changed = not _tensor_content_equal(actual, before)
                if expected_changed and not actual_changed:
                    raise AssertionError(f"{entry['name']} did not mutate the input tensor")
        except Exception as exc:
            raise RuntimeError(f"{entry['name']} bitwise execution failed on {device}: {exc}") from exc

        _assert_exact_tensor_match(actual, expected, entry["name"])
        tested_any = True

    if not tested_any:
        pytest.skip(f"coverage_strategy_pending: no manifest-enabled bitwise cases for {entry['name']}")


def _functional_matmul_callable(entry: dict):
    base_name = entry["base_name"].rstrip("_")
    packet = getattr(torch.ops.aten, base_name)
    return getattr(packet, "default")


def _native_or_reference_matmul_expected(entry: dict, callable_op, cpu_args: tuple, cpu_kwargs: dict) -> tuple[torch.Tensor, bool]:
    surface_kind = entry.get("surface_kind")
    try:
        if surface_kind == "out_variant":
            functional_op = _functional_matmul_callable(entry)
            functional_expected = functional_op(*cpu_args, **cpu_kwargs)
            if not isinstance(functional_expected, torch.Tensor):
                raise TypeError("functional CPU result is not a tensor")
            expected = torch.empty_strided(
                tuple(functional_expected.shape),
                tuple(functional_expected.stride()),
                dtype=functional_expected.dtype,
                device="cpu",
            )
            returned = callable_op(*cpu_args, **cpu_kwargs, out=expected)
            _assert_out_identity(returned, expected, entry["name"])
            return expected, True
        if surface_kind == "mutating_or_inplace":
            if not cpu_args or not isinstance(cpu_args[0], torch.Tensor):
                raise TypeError("in-place matmul-family sample has no writable tensor input")
            expected = _clone_writable_input(cpu_args[0])
            returned = callable_op(expected, *cpu_args[1:], **cpu_kwargs)
            _assert_inplace_identity(returned, expected, entry["name"])
            return expected, True
        if surface_kind == "functional_data":
            expected = callable_op(*cpu_args, **cpu_kwargs)
            if not isinstance(expected, torch.Tensor):
                raise TypeError("functional CPU result is not a tensor")
            return expected, True
    except Exception:
        expected = matmul_family_reference(entry["name"], cpu_args, cpu_kwargs)
        if not isinstance(expected, torch.Tensor):
            raise TypeError("TorchCTS matmul-family reference result is not a tensor")
        return expected, False
    raise TypeError(f"unsupported matmul-family surface kind {surface_kind!r}")


def _run_manual_matmul_case(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    input_condition: str,
    device: str,
    compare,
    manifest: dict,
    case_spec,
) -> bool:
    schema = entry.get("schema", "")
    case_id = getattr(case_spec, "case_id", "default")
    case_kwargs = dict(getattr(case_spec, "params", {}) or {})
    try:
        cpu_inputs = get_inputs_for_op(
            entry["name"],
            dtype=dtype,
            device="cpu",
            input_condition=input_condition,
            seed=manifest.get("ieee754_seed", 67),
            audit={"entries": [entry]},
            **case_kwargs,
        )
    except Exception as exc:
        if getattr(case_spec, "required", True):
            raise RuntimeError(
                f"{entry['name']} required matmul-family case {case_id!r} could not be generated "
                f"on CPU: {type(exc).__name__}: {exc}; schema={schema}"
            ) from exc
        return False

    cpu_args = cpu_inputs.positional_args()
    cpu_kwargs = cpu_inputs.kwargs()
    surface_kind = entry.get("surface_kind")

    try:
        if surface_kind not in {"out_variant", "mutating_or_inplace", "functional_data"}:
            return False
        before = None
        if surface_kind == "mutating_or_inplace" and cpu_args and isinstance(cpu_args[0], torch.Tensor):
            before = cpu_args[0].detach().clone(memory_format=torch.preserve_format)
        expected, _used_native_cpu = _native_or_reference_matmul_expected(entry, callable_op, cpu_args, cpu_kwargs)
        expected_changed = before is not None and not _tensor_content_equal(expected, before)
    except Exception as exc:
        if getattr(case_spec, "required", True):
            raise RuntimeError(
                f"{entry['name']} required matmul-family case {case_id!r} failed CPU reference: "
                f"{type(exc).__name__}: {exc}; schema={schema}"
            ) from exc
        return False

    try:
        dev_inputs = get_inputs_for_op(
            entry["name"],
            dtype=dtype,
            device=device,
            input_condition=input_condition,
            seed=manifest.get("ieee754_seed", 67),
            audit={"entries": [entry]},
            **case_kwargs,
        )
        dev_args = dev_inputs.positional_args()
        dev_kwargs = dev_inputs.kwargs()
        if surface_kind == "out_variant":
            actual = torch.empty_strided(
                tuple(expected.shape),
                tuple(expected.stride()),
                dtype=expected.dtype,
                device=device,
            )
            returned = callable_op(*dev_args, **dev_kwargs, out=actual)
            synchronize(device)
            _assert_out_identity(returned, actual, entry["name"])
        elif surface_kind == "mutating_or_inplace":
            actual = _clone_writable_input(dev_args[0])
            before = actual.detach().clone(memory_format=torch.preserve_format)
            returned = callable_op(actual, *dev_args[1:], **dev_kwargs)
            synchronize(device)
            _assert_inplace_identity(returned, actual, entry["name"])
            actual_changed = not _tensor_content_equal(actual, before)
            if expected_changed and not actual_changed:
                raise AssertionError(f"{entry['name']} did not mutate the input tensor")
        else:
            actual = callable_op(*dev_args, **dev_kwargs)
            synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            f"{entry['name']} matmul-family execution failed on {device}: "
            f"{type(exc).__name__}: {exc}; schema={schema}; case_id={case_id}; input_condition={input_condition}"
        ) from exc

    if input_condition != InputCondition.CLEAN:
        _compare_special_tier(actual, expected, input_condition)
    else:
        compare(actual, expected, category="matmul", dtype=dtype)
    return True


def run_manual_matmul_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    if entry is None:
        pytest.skip("No default coverage audit found for generated matmul-family tests")
    if entry.get("status") == "unknown":
        pytest.skip("coverage_unknown")
    if entry.get("status") == "excluded":
        pytest.skip("coverage_excluded")

    strategy = entry.get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") != "manual_matmul":
        skip_until_strategy_exists(entry, "matmul")

    callable_op = _dispatcher_callable(entry)
    case_specs = sample_case_specs_for_entry(entry)
    tested_any = False
    for dtype, _dtype_str in _manifest_dtype_items(manifest):
        if dtype == torch.bool:
            continue
        for input_condition in _manual_input_conditions(manifest, entry["base_name"], dtype):
            for case_spec in case_specs:
                if _run_manual_matmul_case(
                    entry,
                    callable_op,
                    dtype,
                    input_condition,
                    device,
                    compare,
                    manifest,
                    case_spec,
                ):
                    tested_any = True

    if not tested_any:
        pytest.skip(f"coverage_strategy_pending: no manifest-enabled matmul-family cases for {entry['name']}")


def _is_tensor_sequence(value) -> bool:
    return isinstance(value, (list, tuple)) and all(isinstance(item, torch.Tensor) for item in value)


def _shape_uses_tensor_list_out(entry: dict) -> bool:
    return any(arg.get("is_out") and arg.get("type") == "List[Tensor]" for arg in entry.get("args", []))


def _shape_out_dtype(entry: dict, dtype: torch.dtype) -> torch.dtype:
    if entry["name"] in {
        "aten::argsort.stable_out",
        "aten::nonzero_static.out",
        "aten::repeat_interleave.Tensor_out",
        "aten::tril_indices.out",
        "aten::triu_indices.out",
    }:
        return torch.long
    return dtype


def _shape_tensor_list_out_count(entry: dict, args: tuple) -> int:
    name = entry["name"]
    if not args or not isinstance(args[0], torch.Tensor):
        return 0
    if "unbind_copy" in name:
        dim = int(args[1]) if len(args) > 1 else 0
        return int(args[0].shape[dim])
    if "split_with_sizes" in name:
        return len(args[1])
    if "split_copy" in name:
        split_size = int(args[1])
        dim = int(args[2]) if len(args) > 2 else 0
        size = int(args[0].shape[dim])
        return (size + split_size - 1) // split_size
    return 0


def _empty_shape_tensor_list(entry: dict, args: tuple, device: str) -> list[torch.Tensor]:
    count = _shape_tensor_list_out_count(entry, args)
    dtype = args[0].dtype if args and isinstance(args[0], torch.Tensor) else torch.float32
    return [torch.empty(0, dtype=dtype, device=device) for _ in range(count)]


def _compare_shape_tensor_sequence(
    entry: dict,
    actual,
    expected,
    input_condition: str,
    dtype: torch.dtype,
    compare,
) -> None:
    if not _is_tensor_sequence(actual) or not _is_tensor_sequence(expected):
        raise AssertionError(
            f"{entry['name']} expected Tensor sequence outputs, got {type(actual).__name__} and {type(expected).__name__}"
        )
    if len(actual) != len(expected):
        raise AssertionError(f"{entry['name']} Tensor sequence length mismatch: {len(actual)} vs {len(expected)}")
    for actual_item, expected_item in zip(actual, expected):
        if input_condition != InputCondition.CLEAN:
            _compare_special_tier(actual_item, expected_item, input_condition)
        else:
            compare(actual_item, expected_item, category="copy", dtype=dtype)


def _run_manual_shape_case(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    input_condition: str,
    device: str,
    compare,
    manifest: dict,
) -> bool:
    try:
        cpu_sample = sample_shape(
            entry,
            dtype,
            device="cpu",
            input_condition=input_condition,
            seed=manifest.get("ieee754_seed", 67),
        )
    except Exception:
        return False

    cpu_args = cpu_sample.call_args()
    cpu_kwargs = cpu_sample.kwargs
    surface_kind = entry.get("surface_kind")

    try:
        if surface_kind == "out_variant":
            if _shape_uses_tensor_list_out(entry):
                expected = _empty_shape_tensor_list(entry, cpu_args, "cpu")
                returned = callable_op(*cpu_args, **cpu_kwargs, out=expected)
                if returned is not None:
                    raise AssertionError(f"{entry['name']} returned {returned!r}, expected None for Tensor-list out")
            else:
                expected = torch.empty(0, dtype=_shape_out_dtype(entry, dtype), device="cpu")
                returned = callable_op(*cpu_args, **cpu_kwargs, out=expected)
                _assert_out_identity(returned, expected, entry["name"])
        elif surface_kind == "mutating_or_inplace":
            expected = _clone_writable_input(cpu_args[0])
            returned = callable_op(expected, *cpu_args[1:], **cpu_kwargs)
            if entry["name"] == "aten::set_data":
                if returned is not None:
                    raise AssertionError(f"{entry['name']} returned {returned!r}, expected None")
            else:
                _assert_inplace_identity(returned, expected, entry["name"])
        elif surface_kind == "view_or_alias":
            expected = callable_op(*cpu_args, **cpu_kwargs)
        elif surface_kind == "functional_data":
            expected = callable_op(*cpu_args, **cpu_kwargs)
        else:
            return False
    except Exception:
        return False

    try:
        dev_sample = sample_shape(
            entry,
            dtype,
            device=device,
            input_condition=input_condition,
            seed=manifest.get("ieee754_seed", 67),
        )
        dev_args = dev_sample.call_args()
        dev_kwargs = dev_sample.kwargs

        if surface_kind == "out_variant":
            if _shape_uses_tensor_list_out(entry):
                actual = _empty_shape_tensor_list(entry, dev_args, device)
                returned = callable_op(*dev_args, **dev_kwargs, out=actual)
                synchronize(device)
                if returned is not None:
                    raise AssertionError(f"{entry['name']} returned {returned!r}, expected None for Tensor-list out")
            else:
                actual = torch.empty(0, dtype=expected.dtype, device=device)
                returned = callable_op(*dev_args, **dev_kwargs, out=actual)
                synchronize(device)
                _assert_out_identity(returned, actual, entry["name"])
        elif surface_kind == "mutating_or_inplace":
            actual = _clone_writable_input(dev_args[0])
            returned = callable_op(actual, *dev_args[1:], **dev_kwargs)
            synchronize(device)
            if entry["name"] == "aten::set_data":
                if returned is not None:
                    raise AssertionError(f"{entry['name']} returned {returned!r}, expected None")
            else:
                _assert_inplace_identity(returned, actual, entry["name"])
        else:
            actual = callable_op(*dev_args, **dev_kwargs)
            synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            f"{entry['name']} shape strategy failed on {device}: "
            f"{type(exc).__name__}: {exc}; input_condition={input_condition}"
        ) from exc

    should_check_mutation_reflection = False
    mutation_reflection_views = []
    if _is_tensor_sequence(actual) or _is_tensor_sequence(expected):
        _compare_shape_tensor_sequence(entry, actual, expected, input_condition, dtype, compare)
        if surface_kind == "view_or_alias":
            for actual_item, expected_item in zip(actual, expected):
                _assert_view_metadata_matches(actual_item, expected_item, entry["name"])
                actual_alias = _shares_storage_alias(actual_item, dev_args[0])
                expected_alias = _shares_storage_alias(expected_item, cpu_args[0])
                if actual_alias != expected_alias:
                    raise AssertionError(
                        f"{entry['name']} alias mismatch: device alias={actual_alias}, CPU alias={expected_alias}"
                    )
                if expected_alias and actual_alias:
                    mutation_reflection_views.append(actual_item)
        elif surface_kind == "out_variant":
            for actual_item, out_item in zip(actual, actual):
                _assert_out_identity(actual_item, out_item, entry["name"])
        for view in mutation_reflection_views:
            _assert_mutation_reflects(view, dev_args[0], entry["name"], device)
        return True

    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        raise AssertionError(f"{entry['name']} manual shape strategy expected tensor outputs")

    if entry["base_name"] in {"set", "set_", "set_data"}:
        _assert_view_metadata_matches(actual, expected, entry["name"])

    if surface_kind == "view_or_alias":
        _assert_view_metadata_matches(actual, expected, entry["name"])
        actual_alias = _shares_storage_alias(actual, dev_args[0])
        expected_alias = _shares_storage_alias(expected, cpu_args[0])
        if actual_alias != expected_alias:
            raise AssertionError(
                f"{entry['name']} alias mismatch: device alias={actual_alias}, CPU alias={expected_alias}"
            )
        should_check_mutation_reflection = bool(expected_alias and actual_alias)

    if input_condition != InputCondition.CLEAN:
        _compare_special_tier(actual, expected, input_condition)
    else:
        compare(actual, expected, category="copy", dtype=dtype)
    if entry["base_name"] in {"set", "set_", "set_data"}:
        _assert_storage_set_alias(entry, actual, dev_args, entry["name"])
    if should_check_mutation_reflection:
        _assert_mutation_reflects(actual, dev_args[0], entry["name"], device)
    return True


def _manual_shape_input_conditions(manifest: dict, entry: dict, dtype: torch.dtype) -> list[str]:
    name = entry["name"]
    if entry["base_name"] in {"set", "set_", "set_data"}:
        return [InputCondition.CLEAN]
    if entry["base_name"].startswith("_cast_"):
        return [InputCondition.CLEAN]
    if name.startswith("aten::view_as_"):
        return [InputCondition.CLEAN]
    if name in {"aten::view.dtype", "aten::view_copy.dtype", "aten::view_copy.dtype_out"}:
        return [InputCondition.CLEAN]
    return _manual_input_conditions(manifest, entry["base_name"], dtype)


def _manual_shape_dtype_supported(entry: dict, dtype: torch.dtype) -> bool:
    if entry["name"] == "aten::_neg_view":
        return dtype != torch.bool
    return True


def run_manual_shape_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    if entry is None:
        pytest.skip("No default coverage audit found for generated shape tests")
    if entry.get("status") == "unknown":
        pytest.skip("coverage_unknown")
    if entry.get("status") == "excluded":
        pytest.skip("coverage_excluded")

    strategy = entry.get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") != "manual_shape":
        skip_until_strategy_exists(entry, "shape")

    callable_op = _dispatcher_callable(entry)
    tested_any = False
    for dtype, _dtype_str in _manifest_dtype_items(manifest):
        if not _manual_shape_dtype_supported(entry, dtype):
            continue
        for input_condition in _manual_shape_input_conditions(manifest, entry, dtype):
            if _run_manual_shape_case(entry, callable_op, dtype, input_condition, device, compare, manifest):
                tested_any = True

    if not tested_any:
        pytest.skip(f"coverage_strategy_pending: no manifest-enabled shape cases for {entry['name']}")


def run_generated_out_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    strategy = (entry or {}).get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") == "manual_shape":
        run_manual_shape_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_matmul":
        run_manual_matmul_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_bitwise":
        run_manual_bitwise_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_special_math":
        run_manual_special_math_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_elementwise":
        run_manual_elementwise_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_reduction":
        run_manual_reduction_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_indexing":
        run_manual_indexing_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_rng":
        run_manual_rng_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_multi_output_reduction":
        run_manual_multi_output_reduction_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_upsample":
        run_manual_upsample_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_pooling":
        run_manual_pooling_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_convolution":
        run_manual_convolution_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_grid":
        run_manual_grid_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_grid_backward":
        run_manual_grid_backward_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_rnn_cell":
        run_manual_rnn_cell_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_fft":
        run_manual_fft_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_loss":
        run_manual_loss_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_linalg":
        run_manual_linalg_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_padding":
        run_manual_padding_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_factory_out":
        run_manual_factory_out_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_fft":
        run_manual_fft_strategy(entry, device, compare, manifest)
        return
    run_opinfo_out_strategy(entry, device, compare, manifest)


def run_generated_inplace_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    strategy = (entry or {}).get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") == "manual_shape":
        run_manual_shape_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_matmul":
        run_manual_matmul_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_bitwise":
        run_manual_bitwise_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_special_math":
        run_manual_special_math_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_elementwise":
        run_manual_elementwise_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_reduction":
        run_manual_reduction_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_pooling":
        run_manual_pooling_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_convolution":
        run_manual_convolution_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_linalg":
        run_manual_linalg_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_indexing":
        run_manual_indexing_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_rng":
        run_manual_rng_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_multi_output_reduction":
        run_manual_multi_output_reduction_strategy(entry, device, compare, manifest)
        return
    run_opinfo_inplace_unary_strategy(entry, device, compare, manifest)


def run_generated_functional_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    strategy = (entry or {}).get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") == "manual_shape":
        run_manual_shape_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_special_math":
        run_manual_special_math_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_elementwise":
        run_manual_elementwise_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_reduction":
        run_manual_reduction_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_indexing":
        run_manual_indexing_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_rng":
        run_manual_rng_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_multi_output_reduction":
        run_manual_multi_output_reduction_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_upsample":
        run_manual_upsample_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_pooling":
        run_manual_pooling_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_convolution":
        run_manual_convolution_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_grid":
        run_manual_grid_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_grid_backward":
        run_manual_grid_backward_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_rnn_cell":
        run_manual_rnn_cell_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_fft":
        run_manual_fft_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_loss":
        run_manual_loss_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_linalg":
        run_manual_linalg_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_metadata":
        run_manual_metadata_strategy(entry, device, compare, manifest)
        return
    if strategy.get("strategy") == "manual_padding":
        run_manual_padding_strategy(entry, device, compare, manifest)
        return
    skip_until_strategy_exists(entry, "functional_data")


def _factory_args(entry_name: str):
    return sample_factory_args(entry_name)


def _factory_dtype_supported(family: str, dtype: torch.dtype) -> bool:
    return sample_factory_dtype_supported(family, dtype)


EMPTY_FACTORY_BASES = {
    "empty",
    "empty_like",
    "empty_permuted",
    "empty_strided",
    "new_empty",
    "new_empty_strided",
}


def _assert_factory_metadata(entry: dict, actual: torch.Tensor, expected: torch.Tensor, device: str, dtype: torch.dtype) -> None:
    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        raise AssertionError(f"{entry['name']} did not return a Tensor")
    if actual.dtype != expected.dtype:
        raise AssertionError(f"{entry['name']} dtype mismatch: {actual.dtype} vs {expected.dtype}")
    if actual.device.type != torch.device(device).type:
        raise AssertionError(f"{entry['name']} returned tensor on {actual.device}, expected {device}")
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(f"{entry['name']} shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}")
    if tuple(getattr(actual, "names", ())) != tuple(getattr(expected, "names", ())):
        raise AssertionError(f"{entry['name']} names mismatch: {actual.names} vs {expected.names}")
    if entry.get("base_name") in EMPTY_FACTORY_BASES:
        if actual.layout != expected.layout:
            raise AssertionError(f"{entry['name']} layout mismatch: {actual.layout} vs {expected.layout}")
        if actual.layout == torch.strided and expected.layout == torch.strided:
            if tuple(actual.stride()) != tuple(expected.stride()):
                raise AssertionError(f"{entry['name']} stride mismatch: {tuple(actual.stride())} vs {tuple(expected.stride())}")
            try:
                actual_offset = actual.storage_offset()
                expected_offset = expected.storage_offset()
            except Exception:
                actual_offset = expected_offset = None
            if actual_offset != expected_offset:
                raise AssertionError(f"{entry['name']} storage_offset mismatch: {actual_offset} vs {expected_offset}")


def run_manual_factory_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    if entry is None:
        pytest.skip("No default coverage audit found for generated factory tests")
    if entry.get("status") == "unknown":
        pytest.skip("coverage_unknown")
    if entry.get("status") == "excluded":
        pytest.skip("coverage_excluded")

    strategy = entry.get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") != "manual_factory":
        skip_until_strategy_exists(entry, "factory")

    family = strategy["family"]
    callable_op = _dispatcher_callable(entry)
    args = _factory_args(entry["name"])
    tested_any = False

    for dtype, _dtype_str in _manifest_dtype_items(manifest):
        if not _factory_dtype_supported(family, dtype):
            continue
        try:
            expected = callable_op(*args, dtype=dtype, device="cpu")
        except Exception:
            continue
        try:
            actual = callable_op(*args, dtype=dtype, device=device)
            synchronize(device)
        except Exception as exc:
            raise RuntimeError(f"{entry['name']} factory execution failed on {device}: {exc}") from exc

        _assert_factory_metadata(entry, actual, expected, device, dtype)
        if entry.get("base_name") in EMPTY_FACTORY_BASES:
            tested_any = True
            continue
        compare(actual, expected, category="elementwise", dtype=dtype)
        tested_any = True

    if not tested_any:
        pytest.skip(f"coverage_strategy_pending: no manifest-enabled factory cases for {entry['name']}")


def _factory_out_arg_value(arg: dict, dtype: torch.dtype, device: str):
    name = arg.get("name", "")
    if arg.get("tensor"):
        if name == "start":
            return torch.tensor(0.0, dtype=dtype, device=device)
        if name == "end":
            return torch.tensor(6.0, dtype=dtype, device=device)
        return torch.tensor(1.0, dtype=dtype, device=device)
    if name in {"window_length", "n"}:
        return 8
    if name == "m":
        return 5
    if name == "size":
        return [2, 3]
    if name == "start":
        return 0
    if name == "end":
        return 6
    if name == "step":
        return 2
    if name == "steps":
        return 8
    if name == "base":
        return 10.0
    if name == "d":
        return 0.5
    if name == "periodic":
        return True
    if name == "alpha":
        return 0.54
    if name == "beta":
        return 0.46
    if name == "fill_value":
        return 3
    if name == "memory_format":
        return None
    return None


def _factory_out_args(entry: dict, dtype: torch.dtype, device: str):
    return sample_factory_out_args(entry, dtype, device)


def _factory_out_call_parts(entry: dict, dtype: torch.dtype, device: str):
    return sample_factory_out_call_parts(entry, dtype, device)


def _factory_out_arg_map(entry: dict, args: tuple) -> dict:
    names = [arg.get("name") for arg in entry.get("args", []) if arg.get("name") != "out"]
    return dict(zip(names, args))


def _factory_out_shape(entry: dict, args: tuple) -> tuple[int, ...]:
    return sample_factory_out_shape(entry, args)


def _run_manual_factory_out_case(entry: dict, callable_op, dtype: torch.dtype, device: str, compare) -> bool:
    schema = entry.get("schema", "")
    try:
        cpu_args = _factory_out_args(entry, dtype, "cpu")
        cpu_call_args, cpu_call_kwargs = _factory_out_call_parts(entry, dtype, "cpu")
        shape = _factory_out_shape(entry, cpu_args)
        expected = torch.empty_strided(
            shape,
            torch.empty(shape).stride(),
            dtype=dtype,
            device="cpu",
        )
        returned = callable_op(*cpu_call_args, **cpu_call_kwargs, out=expected)
        _assert_out_identity(returned, expected, entry["name"])
    except Exception:
        return False

    dev_call_args, dev_call_kwargs = _factory_out_call_parts(entry, dtype, device)
    actual = torch.empty_strided(
        tuple(expected.shape),
        tuple(expected.stride()),
        dtype=expected.dtype,
        device=device,
    )
    try:
        returned = callable_op(*dev_call_args, **dev_call_kwargs, out=actual)
        synchronize(device)
        _assert_out_identity(returned, actual, entry["name"])
    except Exception as exc:
        raise RuntimeError(
            f"{entry['name']} factory out= execution failed on {device}: "
            f"{type(exc).__name__}: {exc}; schema={schema}; dtype={dtype}"
        ) from exc

    _assert_factory_metadata(entry, actual, expected, device, dtype)
    if entry["base_name"] not in EMPTY_FACTORY_BASES:
        compare(actual, expected, category="elementwise", dtype=dtype)
    return True


def run_manual_factory_out_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    if entry is None:
        pytest.skip("No default coverage audit found for generated factory out= tests")
    if entry.get("status") == "unknown":
        pytest.skip("coverage_unknown")
    if entry.get("status") == "excluded":
        pytest.skip("coverage_excluded")

    strategy = entry.get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") != "manual_factory_out":
        skip_until_strategy_exists(entry, "factory_out")

    callable_op = _dispatcher_callable(entry)
    tested_any = False
    for dtype, _dtype_str in _manifest_dtype_items(manifest):
        if _run_manual_factory_out_case(entry, callable_op, dtype, device, compare):
            tested_any = True

    if not tested_any:
        pytest.skip(f"coverage_strategy_pending: no manifest-enabled factory out= cases for {entry['name']}")


def _run_fft_once(
    entry: dict,
    callable_op,
    functional_op,
    dtype: torch.dtype,
    device: str,
    manifest: dict,
) -> torch.Tensor:
    sample = sample_fft(
        entry,
        dtype,
        device=device,
        input_condition=InputCondition.CLEAN,
        seed=manifest.get("ieee754_seed", 67),
    )
    args = sample.call_args()
    kwargs = dict(sample.kwargs)
    functional_result = functional_op(*args, **kwargs)
    if not isinstance(functional_result, torch.Tensor):
        raise AssertionError(f"{entry['name']} functional reference returned {type(functional_result).__name__}")
    if entry.get("surface_kind") != "out_variant":
        returned = callable_op(*args, **kwargs)
        if not isinstance(returned, torch.Tensor):
            raise AssertionError(f"{entry['name']} returned {type(returned).__name__}, expected Tensor")
        return returned
    out = torch.empty_strided(
        tuple(functional_result.shape),
        tuple(functional_result.stride()),
        dtype=functional_result.dtype,
        device=device,
    )
    returned = callable_op(*args, **kwargs, out=out)
    _assert_out_identity(returned, out, entry["name"])
    return out


def _run_manual_fft_case(
    entry: dict,
    callable_op,
    functional_op,
    dtype: torch.dtype,
    device: str,
    compare,
    manifest: dict,
) -> bool:
    schema = entry.get("schema", "")
    try:
        expected = _run_fft_once(entry, callable_op, functional_op, dtype, "cpu", manifest)
    except Exception:
        return False

    try:
        actual = _run_fft_once(entry, callable_op, functional_op, dtype, device, manifest)
        synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            f"{entry['name']} FFT out= execution failed on {device}: "
            f"{type(exc).__name__}: {exc}; schema={schema}; dtype={dtype}"
        ) from exc

    if actual.device.type != torch.device(device).type:
        raise AssertionError(f"{entry['name']} returned tensor on {actual.device}, expected {device}")
    if actual.dtype != expected.dtype:
        raise AssertionError(f"{entry['name']} dtype mismatch: {actual.dtype} vs {expected.dtype}")
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(f"{entry['name']} shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}")
    compare(actual, expected, category="fft", dtype=dtype)
    return True


def run_manual_fft_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    if entry is None:
        pytest.skip("No default coverage audit found for generated FFT tests")
    if entry.get("status") == "unknown":
        pytest.skip("coverage_unknown")
    if entry.get("status") == "excluded":
        pytest.skip("coverage_excluded")

    strategy = entry.get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") != "manual_fft":
        skip_until_strategy_exists(entry, "fft")

    callable_op = _dispatcher_callable(entry)
    functional_op = _functional_dispatcher_callable(entry)
    tested_any = False
    for dtype, _dtype_str in _manifest_dtype_items_or(manifest, [torch.float32]):
        if dtype not in {torch.float32, torch.float64}:
            continue
        if _run_manual_fft_case(entry, callable_op, functional_op, dtype, device, compare, manifest):
            tested_any = True

    if not tested_any:
        pytest.skip(f"coverage_strategy_pending: no manifest-enabled FFT cases for {entry['name']}")


def _manual_ieee754_enabled(manifest: dict, op_name: str) -> bool:
    cap = manifest.get("capabilities", {}).get("ieee754", True)
    if cap is True:
        return True
    if cap is False or cap is None:
        return False
    if isinstance(cap, str):
        import re
        return bool(re.search(cap, op_name))
    if isinstance(cap, (list, tuple)):
        import re
        return any(re.search(pattern, op_name) for pattern in cap)
    return False


def _manual_input_conditions(manifest: dict, op_name: str, dtype: torch.dtype) -> list[str]:
    return sample_input_conditions_for(manifest, op_name, dtype)


def _special_math_domain(base_name: str, arg_name: str) -> str:
    if arg_name in {"n", "ord", "p"}:
        return "integer"
    if base_name == "_dirichlet_grad":
        if arg_name == "x":
            return "probability"
        return "positive_large"
    if "ndtri" in base_name:
        return "probability"
    if "polynomial" in base_name:
        return "unit"
    if any(token in base_name for token in ("gamm", "zeta", "polygamma", "digamma")):
        return "positive_large"
    if any(token in base_name for token in ("log", "bessel_y", "modified_bessel_k")):
        return "positive"
    if base_name in {"special_xlogy", "xlogy"} and arg_name == "other":
        return "positive"
    return "mixed"


def _special_tensor_values(dtype: torch.dtype, device: str, domain: str, offset: float = 0.0) -> torch.Tensor:
    if domain == "integer":
        return torch.full((3, 4), 3, dtype=torch.int64, device=device)
    if domain == "probability":
        base = torch.linspace(0.1 + offset, 0.9 - offset, 12, dtype=torch.float32).reshape(3, 4)
    elif domain == "positive_large":
        base = torch.linspace(2.5 + offset, 4.25 + offset, 12, dtype=torch.float32).reshape(3, 4)
    else:
        base = _manual_tensor_values(dtype, "cpu", offset=offset, domain=domain).cpu()

    if domain != "integer":
        if dtype.is_complex:
            if base.is_complex():
                base = base.to(dtype)
            else:
                base = torch.complex(base, base / 8).to(dtype)
        elif dtype.is_floating_point:
            base = base.to(dtype)
        else:
            base = torch.round(base * 4).to(dtype)
    return base.to(device)


def _special_scalar_value(arg: dict, base_name: str):
    arg_name = arg.get("name", "")
    arg_type = arg.get("type", "")
    if arg_name in {"n", "ord", "p"} or "int" in arg_type:
        return 3
    if "bool" in arg_type:
        return True
    if _special_math_domain(base_name, arg_name) in {"positive", "positive_large", "probability"}:
        return 1.5
    return 0.75


def _special_sample(entry: dict, dtype: torch.dtype, device: str, input_condition: str, seed: int):
    return sample_special_math(entry, dtype, device=device, input_condition=input_condition, seed=seed)


def _run_manual_special_math_case(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    input_condition: str,
    device: str,
    compare,
    manifest: dict,
) -> bool:
    sample = _special_sample(entry, dtype, "cpu", input_condition, manifest.get("ieee754_seed", 67))
    cpu_input = sample.input
    cpu_args = sample.args
    cpu_kwargs = sample.kwargs
    surface_kind = entry.get("surface_kind")
    schema = entry.get("schema", "")

    try:
        if surface_kind == "out_variant":
            functional_op = _functional_dispatcher_callable(entry)
            functional_expected = functional_op(cpu_input, *cpu_args, **cpu_kwargs)
            if not isinstance(functional_expected, torch.Tensor):
                return False
            expected = torch.empty_strided(
                tuple(functional_expected.shape),
                tuple(functional_expected.stride()),
                dtype=functional_expected.dtype,
                device="cpu",
            )
            returned = callable_op(cpu_input, *cpu_args, **cpu_kwargs, out=expected)
            _assert_out_identity(returned, expected, entry["name"])
        elif surface_kind == "mutating_or_inplace":
            if not isinstance(cpu_input, torch.Tensor):
                return False
            expected = _clone_writable_input(cpu_input)
            returned = callable_op(expected, *cpu_args, **cpu_kwargs)
            _assert_inplace_identity(returned, expected, entry["name"])
        else:
            expected = callable_op(cpu_input, *cpu_args, **cpu_kwargs)
    except Exception:
        return False

    if not isinstance(expected, torch.Tensor):
        return False

    dev_input = _move_to_device(sample.input, device)
    dev_args = _move_to_device(sample.args, device)
    dev_kwargs = _move_to_device(sample.kwargs, device)

    try:
        if surface_kind == "out_variant":
            actual = torch.empty_strided(
                tuple(expected.shape),
                tuple(expected.stride()),
                dtype=expected.dtype,
                device=device,
            )
            returned = callable_op(dev_input, *dev_args, **dev_kwargs, out=actual)
            synchronize(device)
            _assert_out_identity(returned, actual, entry["name"])
        elif surface_kind == "mutating_or_inplace":
            if not isinstance(dev_input, torch.Tensor):
                return False
            actual = _clone_writable_input(dev_input)
            returned = callable_op(actual, *dev_args, **dev_kwargs)
            synchronize(device)
            _assert_inplace_identity(returned, actual, entry["name"])
        else:
            actual = callable_op(dev_input, *dev_args, **dev_kwargs)
            synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            f"{entry['name']} special-math execution failed on {device}: "
            f"{type(exc).__name__}: {exc}; schema={schema}; input_condition={input_condition}"
        ) from exc

    if not isinstance(actual, torch.Tensor):
        raise AssertionError(f"{entry['name']} returned {type(actual).__name__}, expected Tensor; schema={schema}")
    if input_condition != InputCondition.CLEAN:
        _compare_special_tier(actual, expected, input_condition)
    else:
        compare(actual, expected, category="elementwise", dtype=dtype)
    return True


def run_manual_special_math_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    if entry is None:
        pytest.skip("No default coverage audit found for generated special-math tests")
    if entry.get("status") == "unknown":
        pytest.skip("coverage_unknown")
    if entry.get("status") == "excluded":
        pytest.skip("coverage_excluded")

    strategy = entry.get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") != "manual_special_math":
        skip_until_strategy_exists(entry, "special_math")

    callable_op = _dispatcher_callable(entry)
    tested_any = False
    for dtype, _dtype_str in _manifest_dtype_items(manifest):
        if not (dtype.is_floating_point or dtype.is_complex):
            continue
        for input_condition in _manual_input_conditions(manifest, entry["base_name"], dtype):
            if _run_manual_special_math_case(entry, callable_op, dtype, input_condition, device, compare, manifest):
                tested_any = True

    if not tested_any:
        pytest.skip(f"coverage_strategy_pending: no manifest-enabled special-math cases for {entry['name']}")


def _elementwise_domain(base_name: str, arg_name: str) -> str:
    if arg_name in {"condition"}:
        return "bool"
    if base_name in {"acos", "arccos", "asin", "arcsin", "atanh", "arctanh"}:
        return "unit"
    if base_name in {"acosh", "arccosh"}:
        return "positive_large"
    if base_name in {"log", "log10", "log2", "reciprocal", "rsqrt", "sqrt"}:
        return "positive"
    if base_name == "log1p":
        return "nonzero"
    if base_name in {"div", "divide", "true_divide", "floor_divide", "fmod", "remainder", "addcdiv"}:
        if arg_name in {"other", "tensor2"}:
            return "nonzero"
    if base_name in {"pow", "float_power"}:
        if arg_name in {"self"}:
            return "positive"
        if arg_name in {"exponent"}:
            return "small"
    if base_name in {"clamp", "clip", "clamp_min", "clamp_max"}:
        if arg_name == "min":
            return "lower_bound"
        if arg_name == "max":
            return "upper_bound"
    return "mixed"


def _elementwise_tensor_values(dtype: torch.dtype, device: str, domain: str, offset: float = 0.0) -> torch.Tensor:
    if domain == "bool":
        return torch.tensor(
            [[True, False, True, False], [False, True, False, True], [True, True, False, False]],
            dtype=torch.bool,
            device=device,
        )
    if domain == "small":
        return torch.full((3, 4), 2, dtype=dtype, device=device)
    if domain == "positive_large":
        base = torch.linspace(1.25 + offset, 3.25 + offset, 12, dtype=torch.float32).reshape(3, 4)
        if dtype.is_complex:
            return torch.complex(base, base / 8).to(dtype).to(device)
        if dtype.is_floating_point:
            return base.to(dtype).to(device)
        return torch.round(base * 2).to(dtype).to(device)
    if domain == "lower_bound":
        return torch.full((3, 4), -0.5, dtype=dtype, device=device)
    if domain == "upper_bound":
        return torch.full((3, 4), 0.5, dtype=dtype, device=device)
    return _manual_tensor_values(dtype, device, offset=offset, domain=domain)


def _elementwise_scalar_value(arg: dict, base_name: str):
    arg_name = arg.get("name", "")
    arg_type = arg.get("type", "")
    if arg_name == "rounding_mode":
        return None
    if arg_name == "condition" or "bool" in arg_type:
        return True
    if arg_name == "alpha":
        return 1
    if arg_name == "value":
        return 0.5
    if arg_name == "min":
        return -0.5
    if arg_name == "max":
        return 0.5
    if arg_name in {"other", "tensor2"} and base_name in {"div", "divide", "true_divide", "floor_divide", "fmod", "remainder", "addcdiv"}:
        return 2
    if arg_name == "exponent":
        return 2
    if "int" in arg_type:
        return 2
    if "float" in arg_type or arg_type == "number":
        return 1.25
    return None


def _elementwise_sample(entry: dict, dtype: torch.dtype, input_condition: str, seed: int):
    return sample_elementwise(entry, dtype, device="cpu", input_condition=input_condition, seed=seed)


def _run_manual_elementwise_case(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    input_condition: str,
    device: str,
    compare,
    manifest: dict,
) -> bool:
    sample = _elementwise_sample(entry, dtype, input_condition, manifest.get("ieee754_seed", 67))
    cpu_input = sample.input
    cpu_args = sample.args
    cpu_kwargs = sample.kwargs
    surface_kind = entry.get("surface_kind")
    schema = entry.get("schema", "")

    try:
        if surface_kind == "out_variant":
            functional_op = _functional_dispatcher_callable(entry)
            functional_expected = functional_op(cpu_input, *cpu_args, **cpu_kwargs)
            if not isinstance(functional_expected, torch.Tensor):
                return False
            expected = torch.empty_strided(
                tuple(functional_expected.shape),
                tuple(functional_expected.stride()),
                dtype=functional_expected.dtype,
                device="cpu",
            )
            returned = callable_op(cpu_input, *cpu_args, **cpu_kwargs, out=expected)
            _assert_out_identity(returned, expected, entry["name"])
        elif surface_kind == "mutating_or_inplace":
            if not isinstance(cpu_input, torch.Tensor):
                return False
            expected = _clone_writable_input(cpu_input)
            returned = callable_op(expected, *cpu_args, **cpu_kwargs)
            _assert_inplace_identity(returned, expected, entry["name"])
        else:
            expected = callable_op(cpu_input, *cpu_args, **cpu_kwargs)
    except Exception:
        return False

    if not isinstance(expected, torch.Tensor):
        return False

    dev_input = _move_to_device(sample.input, device)
    dev_args = _move_to_device(sample.args, device)
    dev_kwargs = _move_to_device(sample.kwargs, device)
    try:
        if surface_kind == "out_variant":
            actual = torch.empty_strided(
                tuple(expected.shape),
                tuple(expected.stride()),
                dtype=expected.dtype,
                device=device,
            )
            returned = callable_op(dev_input, *dev_args, **dev_kwargs, out=actual)
            synchronize(device)
            _assert_out_identity(returned, actual, entry["name"])
        elif surface_kind == "mutating_or_inplace":
            if not isinstance(dev_input, torch.Tensor):
                return False
            actual = _clone_writable_input(dev_input)
            returned = callable_op(actual, *dev_args, **dev_kwargs)
            synchronize(device)
            _assert_inplace_identity(returned, actual, entry["name"])
        else:
            actual = callable_op(dev_input, *dev_args, **dev_kwargs)
            synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            f"{entry['name']} elementwise execution failed on {device}: "
            f"{type(exc).__name__}: {exc}; schema={schema}; input_condition={input_condition}"
        ) from exc

    if not isinstance(actual, torch.Tensor):
        raise AssertionError(f"{entry['name']} returned {type(actual).__name__}, expected Tensor; schema={schema}")
    if input_condition != InputCondition.CLEAN:
        _compare_special_tier(actual, expected, input_condition)
    else:
        compare(actual, expected, category="elementwise", dtype=dtype)
    return True


def run_manual_elementwise_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    if entry is None:
        pytest.skip("No default coverage audit found for generated elementwise tests")
    if entry.get("status") == "unknown":
        pytest.skip("coverage_unknown")
    if entry.get("status") == "excluded":
        pytest.skip("coverage_excluded")

    strategy = entry.get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") != "manual_elementwise":
        skip_until_strategy_exists(entry, "elementwise")

    callable_op = _dispatcher_callable(entry)
    tested_any = False
    for dtype, _dtype_str in _manifest_dtype_items(manifest):
        for input_condition in _manual_input_conditions(manifest, entry["base_name"], dtype):
            if _run_manual_elementwise_case(entry, callable_op, dtype, input_condition, device, compare, manifest):
                tested_any = True

    if not tested_any:
        pytest.skip(f"coverage_strategy_pending: no manifest-enabled elementwise cases for {entry['name']}")


def _reduction_arg_value(arg: dict):
    name = arg.get("name", "")
    arg_type = arg.get("type", "")
    if name == "dim":
        if arg_type in {"List[int]", "Optional[List[int]]"}:
            return [1]
        if arg_type in {"int", "Optional[int]"}:
            return 1
    if name == "dtype":
        if arg_type == "Optional[int]":
            return None
        return torch.float32
    if name == "p":
        return 2
    if name == "correction":
        return 0
    if name == "unbiased":
        return False
    if name == "keepdim":
        return False
    return None


def _reduction_sample(entry: dict, dtype: torch.dtype, input_condition: str, seed: int):
    return sample_reduction(entry, dtype, device="cpu", input_condition=input_condition, seed=seed)


def _manual_reduction_dtype_supported(entry: dict, dtype: torch.dtype) -> bool:
    base_name = entry["base_name"].rstrip("_")
    if base_name in {"_softmax_backward_data", "_log_softmax_backward_data", "_segment_reduce_backward"}:
        return dtype.is_floating_point
    return True


def _single_out_arg_name(entry: dict) -> str | None:
    out_args = [arg for arg in entry.get("args", []) if arg.get("is_out")]
    if len(out_args) != 1:
        return None
    return out_args[0].get("name")


def _run_manual_reduction_case(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    input_condition: str,
    device: str,
    compare,
    manifest: dict,
) -> bool:
    sample = _reduction_sample(entry, dtype, input_condition, manifest.get("ieee754_seed", 67))
    cpu_input = sample.input
    cpu_args = sample.args
    cpu_kwargs = sample.kwargs
    surface_kind = entry.get("surface_kind")
    schema = entry.get("schema", "")

    try:
        if surface_kind == "out_variant":
            functional_op = _functional_dispatcher_callable(entry)
            functional_expected = functional_op(cpu_input, *cpu_args, **cpu_kwargs)
            if not isinstance(functional_expected, torch.Tensor):
                return False
            out_arg_name = _single_out_arg_name(entry)
            if not out_arg_name:
                return False
            expected = torch.empty_strided(
                tuple(functional_expected.shape),
                tuple(functional_expected.stride()),
                dtype=functional_expected.dtype,
                device="cpu",
            )
            returned = callable_op(cpu_input, *cpu_args, **cpu_kwargs, **{out_arg_name: expected})
            _assert_out_identity(returned, expected, entry["name"])
        elif surface_kind == "mutating_or_inplace":
            if not isinstance(cpu_input, torch.Tensor):
                return False
            expected = _clone_writable_input(cpu_input)
            returned = callable_op(expected, *cpu_args, **cpu_kwargs)
            _assert_inplace_identity(returned, expected, entry["name"])
        else:
            expected = callable_op(cpu_input, *cpu_args, **cpu_kwargs)
            if not isinstance(expected, torch.Tensor):
                return False
    except Exception:
        return False

    dev_input = _move_to_device(sample.input, device)
    dev_args = _move_to_device(sample.args, device)
    dev_kwargs = _move_to_device(sample.kwargs, device)
    try:
        if surface_kind == "out_variant":
            out_arg_name = _single_out_arg_name(entry)
            if not out_arg_name:
                return False
            actual = torch.empty_strided(
                tuple(expected.shape),
                tuple(expected.stride()),
                dtype=expected.dtype,
                device=device,
            )
            returned = callable_op(dev_input, *dev_args, **dev_kwargs, **{out_arg_name: actual})
            synchronize(device)
            _assert_out_identity(returned, actual, entry["name"])
        elif surface_kind == "mutating_or_inplace":
            if not isinstance(dev_input, torch.Tensor):
                return False
            actual = _clone_writable_input(dev_input)
            returned = callable_op(actual, *dev_args, **dev_kwargs)
            synchronize(device)
            _assert_inplace_identity(returned, actual, entry["name"])
        else:
            actual = callable_op(dev_input, *dev_args, **dev_kwargs)
            synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            f"{entry['name']} reduction execution failed on {device}: "
            f"{type(exc).__name__}: {exc}; schema={schema}; input_condition={input_condition}"
        ) from exc

    if not isinstance(actual, torch.Tensor):
        raise AssertionError(f"{entry['name']} returned {type(actual).__name__}, expected Tensor; schema={schema}")

    if input_condition != InputCondition.CLEAN:
        _compare_special_tier(actual, expected, input_condition)
    else:
        compare(actual, expected, category="reduction", dtype=dtype)
    return True


def run_manual_reduction_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    if entry is None:
        pytest.skip("No default coverage audit found for generated reduction tests")
    if entry.get("status") == "unknown":
        pytest.skip("coverage_unknown")
    if entry.get("status") == "excluded":
        pytest.skip("coverage_excluded")

    strategy = entry.get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") != "manual_reduction":
        skip_until_strategy_exists(entry, "reduction")

    callable_op = _dispatcher_callable(entry)
    tested_any = False
    for dtype, _dtype_str in _manifest_dtype_items(manifest):
        if not _manual_reduction_dtype_supported(entry, dtype):
            continue
        for input_condition in _manual_input_conditions(manifest, entry["base_name"], dtype):
            if _run_manual_reduction_case(entry, callable_op, dtype, input_condition, device, compare, manifest):
                tested_any = True

    if not tested_any:
        pytest.skip(f"coverage_strategy_pending: no manifest-enabled reduction cases for {entry['name']}")


def _run_manual_indexing_case(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    device: str,
    compare,
    manifest: dict,
) -> bool:
    sample = sample_indexing(entry, dtype, device="cpu", input_condition=InputCondition.CLEAN, seed=manifest.get("ieee754_seed", 67))
    cpu_input = sample.input
    cpu_args = sample.args
    cpu_kwargs = sample.kwargs
    surface_kind = entry.get("surface_kind")
    schema = entry.get("schema", "")

    try:
        if surface_kind == "out_variant":
            functional_op = _functional_dispatcher_callable(entry)
            functional_expected = functional_op(cpu_input, *cpu_args, **cpu_kwargs)
            if not isinstance(functional_expected, torch.Tensor):
                return False
            expected = torch.empty_strided(
                tuple(functional_expected.shape),
                tuple(functional_expected.stride()),
                dtype=functional_expected.dtype,
                device="cpu",
            )
            returned = callable_op(cpu_input, *cpu_args, **cpu_kwargs, out=expected)
            _assert_out_identity(returned, expected, entry["name"])
        elif surface_kind == "mutating_or_inplace":
            if not isinstance(cpu_input, torch.Tensor):
                return False
            expected = _clone_writable_input(cpu_input)
            returned = callable_op(expected, *cpu_args, **cpu_kwargs)
            _assert_inplace_identity(returned, expected, entry["name"])
        else:
            expected = callable_op(cpu_input, *cpu_args, **cpu_kwargs)
    except Exception:
        return False

    if not isinstance(expected, torch.Tensor):
        return False

    dev_input = _move_to_device(sample.input, device)
    dev_args = _move_to_device(sample.args, device)
    dev_kwargs = _move_to_device(sample.kwargs, device)
    try:
        if surface_kind == "out_variant":
            actual = torch.empty_strided(
                tuple(expected.shape),
                tuple(expected.stride()),
                dtype=expected.dtype,
                device=device,
            )
            returned = callable_op(dev_input, *dev_args, **dev_kwargs, out=actual)
            synchronize(device)
            _assert_out_identity(returned, actual, entry["name"])
        elif surface_kind == "mutating_or_inplace":
            if not isinstance(dev_input, torch.Tensor):
                return False
            actual = _clone_writable_input(dev_input)
            returned = callable_op(actual, *dev_args, **dev_kwargs)
            synchronize(device)
            _assert_inplace_identity(returned, actual, entry["name"])
        else:
            actual = callable_op(dev_input, *dev_args, **dev_kwargs)
            synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            f"{entry['name']} indexing execution failed on {device}: "
            f"{type(exc).__name__}: {exc}; schema={schema}"
        ) from exc

    if not isinstance(actual, torch.Tensor):
        raise AssertionError(f"{entry['name']} returned {type(actual).__name__}, expected Tensor; schema={schema}")

    if surface_kind == "view_or_alias":
        expected_alias = _shares_storage_alias(expected, cpu_input)
        actual_alias = _shares_storage_alias(actual, dev_input)
        if actual_alias != expected_alias:
            raise AssertionError(
                f"{entry['name']} alias mismatch: device alias={actual_alias}, CPU alias={expected_alias}"
            )
        if expected_alias:
            _assert_view_metadata_matches(actual, expected, entry["name"])
            _assert_mutation_reflects(actual, dev_input, entry["name"], device)

    compare(actual, expected, category="copy", dtype=dtype)
    return True


def run_manual_indexing_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    if entry is None:
        pytest.skip("No default coverage audit found for generated indexing tests")
    if entry.get("status") == "unknown":
        pytest.skip("coverage_unknown")
    if entry.get("status") == "excluded":
        pytest.skip("coverage_excluded")

    strategy = entry.get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") != "manual_indexing":
        skip_until_strategy_exists(entry, "indexing")

    callable_op = _dispatcher_callable(entry)
    tested_any = False
    for dtype, _dtype_str in _manifest_dtype_items(manifest):
        if _run_manual_indexing_case(entry, callable_op, dtype, device, compare, manifest):
            tested_any = True

    if not tested_any:
        pytest.skip(f"coverage_strategy_pending: no manifest-enabled indexing cases for {entry['name']}")


def _rng_has_generator_arg(entry: dict) -> bool:
    return any(arg.get("name") == "generator" for arg in entry.get("args", []))


def _rng_requires_enabled_capability(entry: dict, manifest: dict) -> None:
    capabilities = manifest.get("capabilities", {})
    if not capabilities.get("rng", True):
        pytest.skip("coverage_capability_disabled: rng")
    if (
        _rng_has_generator_arg(entry)
        and sample_rng_uses_target_device_generator(entry)
        and not capabilities.get("device_generator", True)
    ):
        pytest.skip("coverage_capability_disabled: device_generator")


def _rng_seed_for_call(entry: dict, device: str, seed: int) -> None:
    if not _rng_has_generator_arg(entry):
        torch.manual_seed(seed)


def _rng_integral_bounds(entry: dict) -> tuple[int | None, int | None]:
    overload = entry.get("overload", "")
    base_name = entry["base_name"].rstrip("_")
    if base_name == "randint":
        if overload.startswith("low"):
            return 2, 7
        return 0, 7
    if base_name == "randint_like":
        if overload.startswith("low"):
            return 2, 7
        return 0, 7
    if base_name in {"random", "random_"}:
        if overload in {"from", "from_out"}:
            return 2, 11
        if overload in {"to", "to_out"}:
            return 0, 11
        return 0, None
    return None, None


def _as_real_for_rng_checks(tensor: torch.Tensor) -> torch.Tensor:
    candidate = tensor.detach().cpu()
    if candidate.is_complex():
        candidate = torch.view_as_real(candidate)
    return candidate


def _rng_tensor_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool(torch.equal(_as_real_for_rng_checks(a), _as_real_for_rng_checks(b)))


def _assert_rng_domain(entry: dict, tensor: torch.Tensor) -> None:
    base_name = entry["base_name"].rstrip("_")
    values = _as_real_for_rng_checks(tensor)

    if base_name in {"rand", "rand_like"}:
        if not bool(((values >= 0) & (values < 1)).all()):
            raise AssertionError(f"{entry['name']} produced values outside [0, 1)")
        return

    if base_name in {"randn", "randn_like", "normal"}:
        if not bool(torch.isfinite(values).all()):
            raise AssertionError(f"{entry['name']} produced non-finite normal samples")
        return

    if base_name == "uniform":
        if not bool(((values >= 2) & (values < 11)).all()):
            raise AssertionError(f"{entry['name']} produced uniform values outside [2, 11)")
        return

    if base_name == "log_normal":
        if not bool((torch.isfinite(values) & (values > 0)).all()):
            raise AssertionError(f"{entry['name']} produced non-positive or non-finite log-normal samples")
        return

    if base_name == "poisson":
        if not bool(torch.isfinite(values).all()):
            raise AssertionError(f"{entry['name']} produced non-finite Poisson samples")
        if not bool((values >= 0).all()):
            raise AssertionError(f"{entry['name']} produced negative Poisson samples")
        if not bool(torch.equal(values, values.floor())):
            raise AssertionError(f"{entry['name']} produced non-integral Poisson samples")
        return

    if base_name == "cauchy":
        if not bool(torch.isfinite(values).all()):
            raise AssertionError(f"{entry['name']} produced non-finite Cauchy samples")
        return

    if base_name == "exponential":
        if not bool(torch.isfinite(values).all()):
            raise AssertionError(f"{entry['name']} produced non-finite exponential samples")
        if not bool((values >= 0).all()):
            raise AssertionError(f"{entry['name']} produced negative exponential samples")
        return

    if base_name == "geometric":
        if not bool(torch.isfinite(values).all()):
            raise AssertionError(f"{entry['name']} produced non-finite geometric samples")
        if not bool((values >= 1).all()):
            raise AssertionError(f"{entry['name']} produced geometric samples below 1")
        if not bool(torch.equal(values, values.floor())):
            raise AssertionError(f"{entry['name']} produced non-integral geometric samples")
        return

    if base_name == "binomial":
        if not bool(torch.isfinite(values).all()):
            raise AssertionError(f"{entry['name']} produced non-finite binomial samples")
        if not bool((values >= 0).all()):
            raise AssertionError(f"{entry['name']} produced negative binomial samples")
        if not bool(torch.equal(values, values.floor())):
            raise AssertionError(f"{entry['name']} produced non-integral binomial samples")
        return

    if base_name == "multinomial":
        if values.dtype != torch.int64:
            raise AssertionError(f"{entry['name']} produced {values.dtype}, expected torch.int64")
        if not bool(((values >= 0) & (values < 4)).all()):
            raise AssertionError(f"{entry['name']} produced class indices outside [0, 4)")
        if values.ndim >= 2:
            for row in values.reshape(-1, values.shape[-1]):
                if len(set(int(item) for item in row.tolist())) != row.numel():
                    raise AssertionError(f"{entry['name']} produced duplicate class indices without replacement")
        return

    if base_name == "_standard_gamma":
        if not bool(torch.isfinite(values).all()):
            raise AssertionError(f"{entry['name']} produced non-finite gamma samples")
        if not bool((values >= 0).all()):
            raise AssertionError(f"{entry['name']} produced negative gamma samples")
        return

    if base_name == "_sample_dirichlet":
        if not bool(torch.isfinite(values).all()):
            raise AssertionError(f"{entry['name']} produced non-finite Dirichlet samples")
        if not bool((values >= 0).all()):
            raise AssertionError(f"{entry['name']} produced negative Dirichlet samples")
        row_sums = values.sum(dim=-1)
        expected = torch.ones_like(row_sums)
        if not bool(torch.allclose(row_sums, expected, rtol=1e-4, atol=1e-5)):
            raise AssertionError(f"{entry['name']} produced Dirichlet rows that do not sum to 1")
        return

    if base_name == "randperm":
        if values.ndim != 1:
            raise AssertionError(f"{entry['name']} produced randperm output with shape {tuple(values.shape)}")
        if not bool(torch.equal(values, values.floor())):
            raise AssertionError(f"{entry['name']} produced non-integral randperm values")
        sorted_values = torch.sort(values.to(torch.int64)).values
        expected = torch.arange(values.numel(), dtype=torch.int64)
        if not bool(torch.equal(sorted_values, expected)):
            raise AssertionError(f"{entry['name']} did not produce a permutation of [0, {values.numel()})")
        return

    if base_name == "bernoulli":
        if not bool(((values == 0) | (values == 1)).all()):
            raise AssertionError(f"{entry['name']} produced Bernoulli values outside {{0, 1}}")
        return

    if base_name in {"randint", "randint_like", "random", "random_"}:
        if values.dtype == torch.bool:
            return
        if not bool(torch.equal(values, values.floor())):
            raise AssertionError(f"{entry['name']} produced non-integral random integer values")
        low, high = _rng_integral_bounds(entry)
        if low is not None and not bool((values >= low).all()):
            raise AssertionError(f"{entry['name']} produced values below {low}")
        if high is not None and not bool((values < high).all()):
            raise AssertionError(f"{entry['name']} produced values >= {high}")


def _rng_out_dtype(entry: dict, dtype: torch.dtype) -> torch.dtype:
    base_name = entry["base_name"].rstrip("_")
    if base_name == "multinomial":
        return torch.int64
    return dtype


def _run_rng_once(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    device: str,
    seed: int,
) -> torch.Tensor:
    surface_kind = entry.get("surface_kind")
    args, kwargs = sample_rng_call_parts(entry, dtype, device, seed)
    _rng_seed_for_call(entry, device, seed)

    if surface_kind == "out_variant":
        shape = sample_rng_output_shape(entry, args)
        out_dtype = _rng_out_dtype(entry, dtype)
        out = torch.empty_strided(
            shape,
            torch.empty(shape).stride(),
            dtype=out_dtype,
            device=device,
        )
        returned = callable_op(*args, **kwargs, out=out)
        _assert_out_identity(returned, out, entry["name"])
        return out

    if surface_kind == "mutating_or_inplace":
        if not args or not isinstance(args[0], torch.Tensor):
            raise SampleGenerationError(f"{entry['name']} in-place RNG sample has no tensor self argument")
        target = _clone_writable_input(args[0])
        returned = callable_op(target, *args[1:], **kwargs)
        _assert_inplace_identity(returned, target, entry["name"])
        return target

    result = callable_op(*args, **kwargs)
    if not isinstance(result, torch.Tensor):
        raise AssertionError(f"{entry['name']} returned {type(result).__name__}, expected Tensor")
    return result


def _run_manual_rng_case(entry: dict, callable_op, dtype: torch.dtype, device: str, manifest: dict) -> bool:
    seed = int(manifest.get("ieee754_seed", 67))
    schema = entry.get("schema", "")
    try:
        cpu_result = _run_rng_once(entry, callable_op, dtype, "cpu", seed)
        _assert_rng_domain(entry, cpu_result)
    except Exception:
        return False

    try:
        actual = _run_rng_once(entry, callable_op, dtype, device, seed)
        repeated = _run_rng_once(entry, callable_op, dtype, device, seed)
        synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            f"{entry['name']} RNG execution failed on {device}: "
            f"{type(exc).__name__}: {exc}; schema={schema}; dtype={dtype}"
        ) from exc

    if actual.dtype != cpu_result.dtype:
        raise AssertionError(f"{entry['name']} dtype mismatch: {actual.dtype} vs {cpu_result.dtype}")
    if tuple(actual.shape) != tuple(cpu_result.shape):
        raise AssertionError(f"{entry['name']} shape mismatch: {tuple(actual.shape)} vs {tuple(cpu_result.shape)}")
    if actual.device.type != torch.device(device).type:
        raise AssertionError(f"{entry['name']} returned tensor on {actual.device}, expected {device}")
    _assert_rng_domain(entry, actual)
    if not _rng_tensor_equal(actual, repeated):
        raise AssertionError(f"{entry['name']} is not reproducible for same seed/generator on {device}")
    return True


def run_manual_rng_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    if entry is None:
        pytest.skip("No default coverage audit found for generated RNG tests")
    if entry.get("status") == "unknown":
        pytest.skip("coverage_unknown")
    if entry.get("status") == "excluded":
        pytest.skip("coverage_excluded")

    strategy = entry.get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") != "manual_rng":
        skip_until_strategy_exists(entry, "rng")
    _rng_requires_enabled_capability(entry, manifest)

    callable_op = _dispatcher_callable(entry)
    tested_any = False
    for dtype, _dtype_str in _manifest_dtype_items(manifest):
        if _run_manual_rng_case(entry, callable_op, dtype, device, manifest):
            tested_any = True

    if not tested_any:
        pytest.skip(f"coverage_strategy_pending: no manifest-enabled RNG cases for {entry['name']}")


def _multi_output_return_tuple(value) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, (tuple, list)):
        tensor_items = tuple(item for item in value if isinstance(item, torch.Tensor))
        if tensor_items:
            return tensor_items
    raise AssertionError(f"Expected tensor or tuple/list of tensors, got {type(value).__name__}")


def _multi_output_out_dtype(entry: dict, dtype: torch.dtype, out_index: int) -> torch.dtype:
    base_name = entry["base_name"].rstrip("_")
    if base_name == "_linalg_det":
        return torch.int32 if out_index == 2 else dtype
    if base_name == "_linalg_slogdet":
        if out_index == 1:
            return _real_dtype_for_complex(dtype)
        if out_index == 3:
            return torch.int32
        return dtype
    if base_name == "_linalg_solve_ex":
        return dtype if out_index in {0, 1} else torch.int32
    if base_name == "_linalg_eigh":
        return _real_dtype_for_complex(dtype) if out_index == 0 else dtype
    if base_name in {"nll_loss_forward", "nll_loss2d_forward"}:
        return dtype
    if base_name in {"_ctc_loss", "_ctc_loss_backward"}:
        return dtype
    if base_name == "multilabel_margin_loss_forward":
        return dtype
    if base_name in {"linalg_slogdet", "slogdet"}:
        return dtype if out_index == 0 else _real_dtype_for_complex(dtype)
    if base_name in {"linalg_cholesky_ex", "linalg_inv_ex", "linalg_solve_ex"}:
        return dtype if out_index == 0 else torch.int32
    if base_name in {"linalg_lu_factor", "linalg_lu_factor_ex"}:
        return dtype if out_index == 0 else torch.int32
    if "histogram" in base_name:
        return dtype
    if base_name in {"_aminmax", "aminmax"}:
        return dtype
    if base_name in {"_unique", "_unique2", "unique_dim", "unique_dim_consecutive", "unique_consecutive"}:
        return dtype if out_index == 0 else torch.int64
    if base_name == "frexp":
        return dtype if out_index == 0 else torch.int32
    if base_name in {"geqrf", "qr"}:
        return dtype
    if base_name in {"std_mean", "var_mean"}:
        if out_index == 0:
            return _real_dtype_for_complex(dtype)
        return dtype
    if "batch_norm" in base_name:
        if base_name in {"_batch_norm_no_update", "_batch_norm_with_update"} and out_index == 3:
            return torch.uint8
        return dtype
    if "embedding_bag" in base_name:
        return dtype if out_index == 0 else torch.int64
    if "fake_quantize" in base_name and "cachemask" in base_name:
        return dtype if out_index == 0 else torch.bool
    if base_name == "log_sigmoid_forward":
        return dtype
    if out_index == 1:
        return torch.int64
    return dtype


def _multi_output_out_kwargs(entry: dict, dtype: torch.dtype, device: str) -> dict[str, torch.Tensor]:
    kwargs = {}
    for index, arg in enumerate(arg for arg in entry.get("args", []) if arg.get("is_out")):
        name = arg.get("name")
        if not name:
            continue
        kwargs[name] = torch.empty(0, dtype=_multi_output_out_dtype(entry, dtype, index), device=device)
    return kwargs


def _assert_multi_output_identity(returned, outs: dict[str, torch.Tensor], entry_name: str) -> None:
    if not outs:
        return
    returned_items = _multi_output_return_tuple(returned)
    out_items = tuple(outs.values())
    if len(returned_items) != len(out_items):
        raise AssertionError(f"{entry_name} returned {len(returned_items)} tensors for {len(out_items)} out tensors")
    for actual, out in zip(returned_items, out_items):
        _assert_out_identity(actual, out, entry_name)


def _run_multi_output_once(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    input_condition: str,
    device: str,
    manifest: dict,
):
    sample = sample_multi_output_reduction(
        entry,
        dtype,
        device=device,
        input_condition=input_condition,
        seed=manifest.get("ieee754_seed", 67),
    )
    args = sample.call_args()
    kwargs = dict(sample.kwargs)
    out_kwargs = _multi_output_out_kwargs(entry, dtype, device)
    kwargs.update(out_kwargs)
    returned = callable_op(*args, **kwargs)
    _assert_multi_output_identity(returned, out_kwargs, entry["name"])
    return _multi_output_return_tuple(returned)


def _compare_multi_output_results(
    entry: dict,
    actual: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
    input_condition: str,
    dtype: torch.dtype,
    device: str,
    compare,
) -> None:
    if len(actual) != len(expected):
        raise AssertionError(f"{entry['name']} returned {len(actual)} tensors, expected {len(expected)}")
    for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
        if tuple(actual_item.shape) != tuple(expected_item.shape):
            raise AssertionError(
                f"{entry['name']} output {index} shape mismatch: "
                f"{tuple(actual_item.shape)} vs {tuple(expected_item.shape)}"
            )
        if actual_item.dtype != expected_item.dtype:
            raise AssertionError(
                f"{entry['name']} output {index} dtype mismatch: {actual_item.dtype} vs {expected_item.dtype}"
            )
        if actual_item.device.type != torch.device(device).type:
            raise AssertionError(f"{entry['name']} output {index} is on {actual_item.device}, expected {device}")
        if expected_item.dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool}:
            if not torch.equal(actual_item.detach().cpu(), expected_item.detach().cpu()):
                raise AssertionError(f"{entry['name']} output {index} integer/index values differ")
        elif input_condition != InputCondition.CLEAN:
            _compare_special_tier(actual_item, expected_item, input_condition)
        else:
            compare(actual_item, expected_item, category="reduction", dtype=dtype)


def _run_manual_multi_output_reduction_case(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    input_condition: str,
    device: str,
    compare,
    manifest: dict,
) -> bool:
    schema = entry.get("schema", "")
    try:
        expected = _run_multi_output_once(entry, callable_op, dtype, input_condition, "cpu", manifest)
    except Exception:
        return False

    try:
        actual = _run_multi_output_once(entry, callable_op, dtype, input_condition, device, manifest)
        synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            f"{entry['name']} multi-output reduction failed on {device}: "
            f"{type(exc).__name__}: {exc}; schema={schema}; input_condition={input_condition}; dtype={dtype}"
        ) from exc

    _compare_multi_output_results(entry, actual, expected, input_condition, dtype, device, compare)
    return True


def _multi_output_input_conditions(manifest: dict, entry: dict, dtype: torch.dtype) -> list[str]:
    base_name = entry["base_name"].rstrip("_")
    if base_name in {"_unique", "_unique2", "unique_dim", "unique_dim_consecutive", "unique_consecutive"}:
        return [InputCondition.CLEAN]
    if base_name in {"_ctc_loss", "_ctc_loss_backward", "frexp", "geqrf", "qr", "std_mean", "var_mean"}:
        return [InputCondition.CLEAN]
    if base_name.startswith("linalg_") or base_name.startswith("_linalg_"):
        return [InputCondition.CLEAN]
    if base_name == "slogdet":
        return [InputCondition.CLEAN]
    if "histogram" in base_name:
        return [InputCondition.CLEAN]
    if "batch_norm" in base_name:
        return [InputCondition.CLEAN]
    if "embedding_bag" in base_name:
        return [InputCondition.CLEAN]
    if base_name in {"nll_loss_forward", "nll_loss2d_forward"}:
        return [InputCondition.CLEAN]
    if "fake_quantize" in base_name and "cachemask" in base_name:
        return [InputCondition.CLEAN]
    return _manual_input_conditions(manifest, entry["base_name"], dtype)


def run_manual_multi_output_reduction_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    if entry is None:
        pytest.skip("No default coverage audit found for generated multi-output reduction tests")
    if entry.get("status") == "unknown":
        pytest.skip("coverage_unknown")
    if entry.get("status") == "excluded":
        pytest.skip("coverage_excluded")

    strategy = entry.get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") != "manual_multi_output_reduction":
        skip_until_strategy_exists(entry, "multi_output_reduction")

    callable_op = _dispatcher_callable(entry)
    tested_any = False
    for dtype, _dtype_str in _manifest_dtype_items(manifest):
        for input_condition in _multi_output_input_conditions(manifest, entry, dtype):
            if _run_manual_multi_output_reduction_case(
                entry,
                callable_op,
                dtype,
                input_condition,
                device,
                compare,
                manifest,
            ):
                tested_any = True

    if not tested_any:
        pytest.skip(f"coverage_strategy_pending: no manifest-enabled multi-output cases for {entry['name']}")


def _run_manual_upsample_case(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    input_condition: str,
    device: str,
    compare,
    manifest: dict,
) -> bool:
    try:
        cpu_sample = sample_upsample(
            entry,
            dtype,
            device="cpu",
            input_condition=input_condition,
            seed=manifest.get("ieee754_seed", 67),
        )
    except Exception:
        return False

    cpu_args = cpu_sample.call_args()
    cpu_kwargs = cpu_sample.kwargs
    surface_kind = entry.get("surface_kind")
    schema = entry.get("schema", "")

    try:
        if surface_kind == "out_variant":
            expected = torch.empty(0, dtype=dtype, device="cpu")
            returned = callable_op(*cpu_args, **cpu_kwargs, out=expected)
            _assert_out_identity(returned, expected, entry["name"])
        else:
            expected = callable_op(*cpu_args, **cpu_kwargs)
    except Exception:
        return False

    if not isinstance(expected, torch.Tensor):
        return False

    dev_sample = sample_upsample(
        entry,
        dtype,
        device=device,
        input_condition=input_condition,
        seed=manifest.get("ieee754_seed", 67),
    )
    dev_args = dev_sample.call_args()
    dev_kwargs = dev_sample.kwargs
    try:
        if surface_kind == "out_variant":
            actual = torch.empty(0, dtype=expected.dtype, device=device)
            returned = callable_op(*dev_args, **dev_kwargs, out=actual)
            synchronize(device)
            _assert_out_identity(returned, actual, entry["name"])
        else:
            actual = callable_op(*dev_args, **dev_kwargs)
            synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            f"{entry['name']} upsample execution failed on {device}: "
            f"{type(exc).__name__}: {exc}; schema={schema}; input_condition={input_condition}; dtype={dtype}"
        ) from exc

    if actual.dtype != expected.dtype:
        raise AssertionError(f"{entry['name']} dtype mismatch: {actual.dtype} vs {expected.dtype}")
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(f"{entry['name']} shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}")
    if actual.device.type != torch.device(device).type:
        raise AssertionError(f"{entry['name']} returned tensor on {actual.device}, expected {device}")

    if input_condition != InputCondition.CLEAN:
        _compare_special_tier(actual, expected, input_condition)
    else:
        compare(actual, expected, category="elementwise", dtype=dtype)
    return True


def run_manual_upsample_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    if entry is None:
        pytest.skip("No default coverage audit found for generated upsample tests")
    if entry.get("status") == "unknown":
        pytest.skip("coverage_unknown")
    if entry.get("status") == "excluded":
        pytest.skip("coverage_excluded")

    strategy = entry.get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") != "manual_upsample":
        skip_until_strategy_exists(entry, "upsample")

    callable_op = _dispatcher_callable(entry)
    tested_any = False
    for dtype, _dtype_str in _manifest_dtype_items(manifest):
        for input_condition in _manual_input_conditions(manifest, entry["base_name"], dtype):
            if _run_manual_upsample_case(entry, callable_op, dtype, input_condition, device, compare, manifest):
                tested_any = True

    if not tested_any:
        pytest.skip(f"coverage_strategy_pending: no manifest-enabled upsample cases for {entry['name']}")


def _run_pooling_once(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    input_condition: str,
    device: str,
    manifest: dict,
):
    sample = sample_pooling(
        entry,
        dtype,
        device=device,
        input_condition=input_condition,
        seed=manifest.get("ieee754_seed", 67),
    )
    args = sample.call_args()
    kwargs = dict(sample.kwargs)
    out_kwargs = _multi_output_out_kwargs(entry, dtype, device)
    kwargs.update(out_kwargs)
    returned = callable_op(*args, **kwargs)
    _assert_multi_output_identity(returned, out_kwargs, entry["name"])
    return _multi_output_return_tuple(returned)


def _run_manual_pooling_case(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    input_condition: str,
    device: str,
    compare,
    manifest: dict,
) -> bool:
    schema = entry.get("schema", "")
    try:
        expected = _run_pooling_once(entry, callable_op, dtype, input_condition, "cpu", manifest)
    except Exception:
        return False

    try:
        actual = _run_pooling_once(entry, callable_op, dtype, input_condition, device, manifest)
        synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            f"{entry['name']} pooling execution failed on {device}: "
            f"{type(exc).__name__}: {exc}; schema={schema}; input_condition={input_condition}; dtype={dtype}"
        ) from exc

    _compare_multi_output_results(entry, actual, expected, input_condition, dtype, device, compare)
    return True


def run_manual_pooling_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    if entry is None:
        pytest.skip("No default coverage audit found for generated pooling tests")
    if entry.get("status") == "unknown":
        pytest.skip("coverage_unknown")
    if entry.get("status") == "excluded":
        pytest.skip("coverage_excluded")

    strategy = entry.get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") != "manual_pooling":
        skip_until_strategy_exists(entry, "pooling")

    callable_op = _dispatcher_callable(entry)
    tested_any = False
    for dtype, _dtype_str in _manifest_dtype_items(manifest):
        for input_condition in _manual_input_conditions(manifest, entry["base_name"], dtype):
            if _run_manual_pooling_case(entry, callable_op, dtype, input_condition, device, compare, manifest):
                tested_any = True

    if not tested_any:
        pytest.skip(f"coverage_strategy_pending: no manifest-enabled pooling cases for {entry['name']}")


def _run_convolution_once(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    input_condition: str,
    device: str,
    manifest: dict,
):
    sample = sample_convolution(
        entry,
        dtype,
        device=device,
        input_condition=input_condition,
        seed=manifest.get("ieee754_seed", 67),
    )
    args = sample.call_args()
    kwargs = dict(sample.kwargs)
    out_kwargs = _multi_output_out_kwargs(entry, dtype, device)
    kwargs.update(out_kwargs)
    returned = callable_op(*args, **kwargs)
    _assert_multi_output_identity(returned, out_kwargs, entry["name"])
    return _multi_output_return_tuple(returned)


def _run_manual_convolution_case(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    input_condition: str,
    device: str,
    compare,
    manifest: dict,
) -> bool:
    schema = entry.get("schema", "")
    try:
        expected = _run_convolution_once(entry, callable_op, dtype, input_condition, "cpu", manifest)
    except Exception:
        return False

    try:
        actual = _run_convolution_once(entry, callable_op, dtype, input_condition, device, manifest)
        synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            f"{entry['name']} convolution execution failed on {device}: "
            f"{type(exc).__name__}: {exc}; schema={schema}; input_condition={input_condition}; dtype={dtype}"
        ) from exc

    if len(actual) != len(expected):
        raise AssertionError(f"{entry['name']} returned {len(actual)} tensors, expected {len(expected)}")
    for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
        if actual_item.device.type != torch.device(device).type:
            raise AssertionError(f"{entry['name']} output {index} is on {actual_item.device}, expected {device}")
        if tuple(actual_item.shape) != tuple(expected_item.shape):
            raise AssertionError(
                f"{entry['name']} output {index} shape mismatch: "
                f"{tuple(actual_item.shape)} vs {tuple(expected_item.shape)}"
            )
        if actual_item.dtype != expected_item.dtype:
            raise AssertionError(
                f"{entry['name']} output {index} dtype mismatch: {actual_item.dtype} vs {expected_item.dtype}"
            )
        if input_condition != InputCondition.CLEAN:
            _compare_special_tier(actual_item, expected_item, input_condition)
        else:
            compare(actual_item, expected_item, category="conv", dtype=dtype)
    return True


def run_manual_convolution_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    if entry is None:
        pytest.skip("No default coverage audit found for generated convolution tests")
    if entry.get("status") == "unknown":
        pytest.skip("coverage_unknown")
    if entry.get("status") == "excluded":
        pytest.skip("coverage_excluded")

    strategy = entry.get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") != "manual_convolution":
        skip_until_strategy_exists(entry, "convolution")

    callable_op = _dispatcher_callable(entry)
    tested_any = False
    for dtype, _dtype_str in _manifest_dtype_items(manifest):
        for input_condition in _manual_input_conditions(manifest, entry["base_name"], dtype):
            if _run_manual_convolution_case(entry, callable_op, dtype, input_condition, device, compare, manifest):
                tested_any = True

    if not tested_any:
        pytest.skip(f"coverage_strategy_pending: no manifest-enabled convolution cases for {entry['name']}")


def _run_grid_once(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    device: str,
    manifest: dict,
) -> torch.Tensor:
    sample = sample_grid(
        entry,
        dtype,
        device=device,
        input_condition=InputCondition.CLEAN,
        seed=manifest.get("ieee754_seed", 67),
    )
    args = sample.call_args()
    kwargs = dict(sample.kwargs)
    out_kwargs = {}
    if entry.get("surface_kind") == "out_variant":
        out_kwargs["out"] = torch.empty(0, dtype=dtype, device=device)
    kwargs.update(out_kwargs)
    returned = callable_op(*args, **kwargs)
    _assert_multi_output_identity(returned, out_kwargs, entry["name"])
    if not isinstance(returned, torch.Tensor):
        raise AssertionError(f"{entry['name']} returned {type(returned).__name__}, expected Tensor")
    return returned


def _run_manual_grid_case(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    device: str,
    compare,
    manifest: dict,
) -> bool:
    schema = entry.get("schema", "")
    try:
        expected = _run_grid_once(entry, callable_op, dtype, "cpu", manifest)
    except Exception:
        return False

    try:
        actual = _run_grid_once(entry, callable_op, dtype, device, manifest)
        synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            f"{entry['name']} grid execution failed on {device}: "
            f"{type(exc).__name__}: {exc}; schema={schema}; dtype={dtype}"
        ) from exc

    if actual.device.type != torch.device(device).type:
        raise AssertionError(f"{entry['name']} returned tensor on {actual.device}, expected {device}")
    if actual.dtype != expected.dtype:
        raise AssertionError(f"{entry['name']} dtype mismatch: {actual.dtype} vs {expected.dtype}")
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(f"{entry['name']} shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}")
    compare(actual, expected, category="elementwise", dtype=dtype)
    return True


def run_manual_grid_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    if entry is None:
        pytest.skip("No default coverage audit found for generated grid tests")
    if entry.get("status") == "unknown":
        pytest.skip("coverage_unknown")
    if entry.get("status") == "excluded":
        pytest.skip("coverage_excluded")

    strategy = entry.get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") != "manual_grid":
        skip_until_strategy_exists(entry, "grid")

    callable_op = _dispatcher_callable(entry)
    tested_any = False
    for dtype, _dtype_str in _manifest_dtype_items(manifest):
        if _run_manual_grid_case(entry, callable_op, dtype, device, compare, manifest):
            tested_any = True

    if not tested_any:
        pytest.skip(f"coverage_strategy_pending: no manifest-enabled grid cases for {entry['name']}")


def _grid_backward_out_kwargs_from(expected_items: tuple[torch.Tensor, ...], device: str) -> dict[str, torch.Tensor]:
    return {
        f"out{index}": torch.empty_strided(
            tuple(expected.shape),
            tuple(expected.stride()),
            dtype=expected.dtype,
            device=device,
        )
        for index, expected in enumerate(expected_items)
    }


def _run_grid_backward_once(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    device: str,
    manifest: dict,
    *,
    expected_items: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor, ...]:
    sample = sample_grid_backward(
        entry,
        dtype,
        device=device,
        input_condition=InputCondition.CLEAN,
        seed=manifest.get("ieee754_seed", 67),
    )
    args = sample.call_args()
    kwargs = dict(sample.kwargs)
    out_kwargs = {}
    if entry.get("surface_kind") == "out_variant":
        if expected_items is None:
            functional_op = _functional_dispatcher_callable(entry)
            expected_items = _multi_output_return_tuple(functional_op(*args, **kwargs))
        out_kwargs = _grid_backward_out_kwargs_from(expected_items, device)
        kwargs.update(out_kwargs)
    returned = callable_op(*args, **kwargs)
    _assert_multi_output_identity(returned, out_kwargs, entry["name"])
    returned_items = _multi_output_return_tuple(returned)
    if len(returned_items) != 2:
        raise AssertionError(f"{entry['name']} returned {len(returned_items)} tensors, expected 2")
    return returned_items


def _run_manual_grid_backward_case(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    device: str,
    compare,
    manifest: dict,
) -> bool:
    schema = entry.get("schema", "")
    try:
        expected_items = _run_grid_backward_once(entry, callable_op, dtype, "cpu", manifest)
    except Exception:
        return False

    try:
        actual_items = _run_grid_backward_once(
            entry,
            callable_op,
            dtype,
            device,
            manifest,
            expected_items=expected_items,
        )
        synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            f"{entry['name']} grid backward execution failed on {device}: "
            f"{type(exc).__name__}: {exc}; schema={schema}; dtype={dtype}"
        ) from exc

    if len(actual_items) != len(expected_items):
        raise AssertionError(f"{entry['name']} returned {len(actual_items)} tensors, expected {len(expected_items)}")
    for actual, expected in zip(actual_items, expected_items):
        if actual.device.type != torch.device(device).type:
            raise AssertionError(f"{entry['name']} returned tensor on {actual.device}, expected {device}")
        if actual.dtype != expected.dtype:
            raise AssertionError(f"{entry['name']} dtype mismatch: {actual.dtype} vs {expected.dtype}")
        if tuple(actual.shape) != tuple(expected.shape):
            raise AssertionError(f"{entry['name']} shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}")
        compare(actual, expected, category="elementwise", dtype=dtype)
    return True


def run_manual_grid_backward_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    if entry is None:
        pytest.skip("No default coverage audit found for generated grid backward tests")
    if entry.get("status") == "unknown":
        pytest.skip("coverage_unknown")
    if entry.get("status") == "excluded":
        pytest.skip("coverage_excluded")

    strategy = entry.get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") != "manual_grid_backward":
        skip_until_strategy_exists(entry, "grid_backward")

    callable_op = _dispatcher_callable(entry)
    tested_any = False
    for dtype, _dtype_str in _manifest_dtype_items(manifest):
        if not dtype.is_floating_point:
            continue
        if _run_manual_grid_backward_case(entry, callable_op, dtype, device, compare, manifest):
            tested_any = True

    if not tested_any:
        pytest.skip(f"coverage_strategy_pending: no manifest-enabled grid backward cases for {entry['name']}")


def _run_rnn_cell_once(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    device: str,
    manifest: dict,
) -> tuple[torch.Tensor, ...]:
    sample = sample_rnn_cell(
        entry,
        dtype,
        device=device,
        input_condition=InputCondition.CLEAN,
        seed=manifest.get("ieee754_seed", 67),
    )
    returned = callable_op(*sample.call_args(), **sample.kwargs)
    return _multi_output_return_tuple(returned)


def _run_manual_rnn_cell_case(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    device: str,
    compare,
    manifest: dict,
) -> bool:
    schema = entry.get("schema", "")
    try:
        expected = _run_rnn_cell_once(entry, callable_op, dtype, "cpu", manifest)
    except Exception:
        return False

    try:
        actual = _run_rnn_cell_once(entry, callable_op, dtype, device, manifest)
        synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            f"{entry['name']} RNN cell execution failed on {device}: "
            f"{type(exc).__name__}: {exc}; schema={schema}; dtype={dtype}"
        ) from exc

    if len(actual) != len(expected):
        raise AssertionError(f"{entry['name']} returned {len(actual)} tensors, expected {len(expected)}")
    for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
        if actual_item.device.type != torch.device(device).type:
            raise AssertionError(f"{entry['name']} output {index} is on {actual_item.device}, expected {device}")
        if actual_item.dtype != expected_item.dtype:
            raise AssertionError(
                f"{entry['name']} output {index} dtype mismatch: {actual_item.dtype} vs {expected_item.dtype}"
            )
        if tuple(actual_item.shape) != tuple(expected_item.shape):
            raise AssertionError(
                f"{entry['name']} output {index} shape mismatch: "
                f"{tuple(actual_item.shape)} vs {tuple(expected_item.shape)}"
            )
        compare(actual_item, expected_item, category="matmul", dtype=dtype)
    return True


def run_manual_rnn_cell_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    if entry is None:
        pytest.skip("No default coverage audit found for generated RNN cell tests")
    if entry.get("status") == "unknown":
        pytest.skip("coverage_unknown")
    if entry.get("status") == "excluded":
        pytest.skip("coverage_excluded")

    strategy = entry.get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") != "manual_rnn_cell":
        skip_until_strategy_exists(entry, "rnn_cell")

    callable_op = _dispatcher_callable(entry)
    tested_any = False
    for dtype, _dtype_str in _manifest_dtype_items_or(manifest, [torch.float32]):
        if _run_manual_rnn_cell_case(entry, callable_op, dtype, device, compare, manifest):
            tested_any = True

    if not tested_any:
        pytest.skip(f"coverage_strategy_pending: no manifest-enabled RNN cell cases for {entry['name']}")


def _run_loss_once(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    input_condition: str,
    device: str,
    manifest: dict,
) -> torch.Tensor:
    sample = sample_loss(
        entry,
        dtype,
        device=device,
        input_condition=input_condition,
        seed=manifest.get("ieee754_seed", 67),
    )
    args = sample.call_args()
    kwargs = dict(sample.kwargs)
    out_kwargs = _multi_output_out_kwargs(entry, dtype, device)
    kwargs.update(out_kwargs)
    returned = callable_op(*args, **kwargs)
    _assert_multi_output_identity(returned, out_kwargs, entry["name"])
    if not isinstance(returned, torch.Tensor):
        raise AssertionError(f"{entry['name']} returned {type(returned).__name__}, expected Tensor")
    return returned


def _run_manual_loss_case(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    input_condition: str,
    device: str,
    compare,
    manifest: dict,
) -> bool:
    schema = entry.get("schema", "")
    try:
        expected = _run_loss_once(entry, callable_op, dtype, input_condition, "cpu", manifest)
    except Exception:
        return False

    try:
        actual = _run_loss_once(entry, callable_op, dtype, input_condition, device, manifest)
        synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            f"{entry['name']} loss execution failed on {device}: "
            f"{type(exc).__name__}: {exc}; schema={schema}; input_condition={input_condition}; dtype={dtype}"
        ) from exc

    if actual.device.type != torch.device(device).type:
        raise AssertionError(f"{entry['name']} returned tensor on {actual.device}, expected {device}")
    if actual.dtype != expected.dtype:
        raise AssertionError(f"{entry['name']} dtype mismatch: {actual.dtype} vs {expected.dtype}")
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(f"{entry['name']} shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}")
    if input_condition != InputCondition.CLEAN:
        _compare_special_tier(actual, expected, input_condition)
    else:
        compare(actual, expected, category="reduction", dtype=dtype)
    return True


def run_manual_loss_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    if entry is None:
        pytest.skip("No default coverage audit found for generated loss tests")
    if entry.get("status") == "unknown":
        pytest.skip("coverage_unknown")
    if entry.get("status") == "excluded":
        pytest.skip("coverage_excluded")

    strategy = entry.get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") != "manual_loss":
        skip_until_strategy_exists(entry, "loss")

    callable_op = _dispatcher_callable(entry)
    tested_any = False
    for dtype, _dtype_str in _manifest_dtype_items(manifest):
        input_conditions = (
            [InputCondition.CLEAN]
            if entry["base_name"] == "ctc_loss"
            else _manual_input_conditions(manifest, entry["base_name"], dtype)
        )
        for input_condition in input_conditions:
            if _run_manual_loss_case(entry, callable_op, dtype, input_condition, device, compare, manifest):
                tested_any = True

    if not tested_any:
        pytest.skip(f"coverage_strategy_pending: no manifest-enabled loss cases for {entry['name']}")


LINALG_IEEE754_SAFE_BASES = frozenset({
    "linalg_cross",
    "linalg_matmul",
    "linalg_norm",
    "linalg_vecdot",
    "linalg_vector_norm",
    "native_norm",
})

LINALG_REAL_OUTPUT_BASES = frozenset({
    "linalg_cond",
    "linalg_matrix_norm",
    "linalg_norm",
    "linalg_svdvals",
    "linalg_vector_norm",
    "native_norm",
    "nuclear_norm",
})

LINALG_INT_OUTPUT_BASES = frozenset({
    "linalg_matrix_rank",
})


def _real_dtype_for_complex(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex128:
        return torch.float64
    if dtype == torch.complex64:
        return torch.float32
    if getattr(torch, "complex32", None) is not None and dtype == torch.complex32:
        return torch.float16
    return dtype


def _linalg_out_dtype(entry: dict, dtype: torch.dtype) -> torch.dtype:
    if entry["base_name"].rstrip("_") in LINALG_INT_OUTPUT_BASES:
        return torch.int64
    if entry["base_name"].rstrip("_") in LINALG_REAL_OUTPUT_BASES:
        return _real_dtype_for_complex(dtype)
    return dtype


def _linalg_out_kwargs(entry: dict, dtype: torch.dtype, device: str) -> dict[str, torch.Tensor]:
    kwargs = {}
    out_dtype = _linalg_out_dtype(entry, dtype)
    for arg in entry.get("args", []):
        if arg.get("is_out") and arg.get("name"):
            kwargs[arg["name"]] = torch.empty(0, dtype=out_dtype, device=device)
    return kwargs


def _manual_linalg_input_conditions(manifest: dict, entry: dict, dtype: torch.dtype) -> list[str]:
    if entry["base_name"].rstrip("_") not in LINALG_IEEE754_SAFE_BASES:
        return [InputCondition.CLEAN]
    return _manual_input_conditions(manifest, entry["base_name"], dtype)


def _linalg_compare_category(entry: dict) -> str:
    if entry["base_name"].rstrip("_") == "linalg_matmul":
        return "matmul"
    return "linalg"


def _run_linalg_once(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    input_condition: str,
    device: str,
    manifest: dict,
) -> torch.Tensor:
    sample = sample_linalg(
        entry,
        dtype,
        device=device,
        input_condition=input_condition,
        seed=manifest.get("ieee754_seed", 67),
    )
    args = sample.call_args()
    kwargs = dict(sample.kwargs)
    out_kwargs = _linalg_out_kwargs(entry, dtype, device)
    kwargs.update(out_kwargs)
    if entry.get("surface_kind") == "mutating_or_inplace":
        if not args or not isinstance(args[0], torch.Tensor):
            raise AssertionError(f"{entry['name']} in-place linalg sample did not provide a tensor input")
        mutated_input = _clone_writable_input(args[0])
        returned = callable_op(mutated_input, *args[1:], **kwargs)
        _assert_inplace_identity(returned, mutated_input, entry["name"])
    else:
        returned = callable_op(*args, **kwargs)
    _assert_multi_output_identity(returned, out_kwargs, entry["name"])
    returned_items = _multi_output_return_tuple(returned)
    if len(returned_items) != 1:
        raise AssertionError(f"{entry['name']} returned {len(returned_items)} tensors, expected 1")
    return returned_items[0]


def _run_manual_linalg_case(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    input_condition: str,
    device: str,
    compare,
    manifest: dict,
) -> bool:
    schema = entry.get("schema", "")
    try:
        expected = _run_linalg_once(entry, callable_op, dtype, input_condition, "cpu", manifest)
    except Exception:
        return False

    try:
        actual = _run_linalg_once(entry, callable_op, dtype, input_condition, device, manifest)
        synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            f"{entry['name']} linalg execution failed on {device}: "
            f"{type(exc).__name__}: {exc}; schema={schema}; input_condition={input_condition}; dtype={dtype}"
        ) from exc

    if actual.device.type != torch.device(device).type:
        raise AssertionError(f"{entry['name']} returned tensor on {actual.device}, expected {device}")
    if actual.dtype != expected.dtype:
        raise AssertionError(f"{entry['name']} dtype mismatch: {actual.dtype} vs {expected.dtype}")
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(f"{entry['name']} shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}")
    if input_condition != InputCondition.CLEAN:
        _compare_special_tier(actual, expected, input_condition)
    else:
        compare(actual, expected, category=_linalg_compare_category(entry), dtype=dtype)
    return True


def run_manual_linalg_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    if entry is None:
        pytest.skip("No default coverage audit found for generated linalg tests")
    if entry.get("status") == "unknown":
        pytest.skip("coverage_unknown")
    if entry.get("status") == "excluded":
        pytest.skip("coverage_excluded")

    strategy = entry.get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") != "manual_linalg":
        skip_until_strategy_exists(entry, "linalg")

    callable_op = _dispatcher_callable(entry)
    tested_any = False
    for dtype, _dtype_str in _manifest_dtype_items(manifest):
        for input_condition in _manual_linalg_input_conditions(manifest, entry, dtype):
            if _run_manual_linalg_case(entry, callable_op, dtype, input_condition, device, compare, manifest):
                tested_any = True

    if not tested_any:
        pytest.skip(f"coverage_strategy_pending: no manifest-enabled linalg cases for {entry['name']}")


def _metadata_return_equal(actual, expected, *, entry_name: str, dtype: torch.dtype, compare) -> None:
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        if tuple(actual.shape) != tuple(expected.shape):
            raise AssertionError(f"{entry_name} metadata tensor shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}")
        if actual.dtype != expected.dtype:
            raise AssertionError(f"{entry_name} metadata tensor dtype mismatch: {actual.dtype} vs {expected.dtype}")
        compare(actual, expected, category="copy", dtype=dtype)
        return
    if isinstance(actual, torch.dtype) or isinstance(expected, torch.dtype):
        if actual != expected:
            raise AssertionError(f"{entry_name} dtype result mismatch: {actual} vs {expected}")
        return
    if isinstance(actual, torch.device) or isinstance(expected, torch.device):
        if torch.device(actual) != torch.device(expected):
            raise AssertionError(f"{entry_name} device result mismatch: {actual} vs {expected}")
        return
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError(f"{entry_name} metadata sequence length mismatch: {len(actual)} vs {len(expected)}")
        for actual_item, expected_item in zip(actual, expected):
            _metadata_return_equal(actual_item, expected_item, entry_name=entry_name, dtype=dtype, compare=compare)
        return
    if actual != expected:
        raise AssertionError(f"{entry_name} metadata result mismatch: {actual!r} vs {expected!r}")


def _run_metadata_once(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    device: str,
    manifest: dict,
):
    sample = sample_metadata(
        entry,
        dtype,
        device=device,
        input_condition=InputCondition.CLEAN,
        seed=manifest.get("ieee754_seed", 67),
    )
    return callable_op(*sample.call_args(), **sample.kwargs)


def _run_manual_metadata_case(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    device: str,
    compare,
    manifest: dict,
) -> bool:
    schema = entry.get("schema", "")
    try:
        expected = _run_metadata_once(entry, callable_op, dtype, "cpu", manifest)
    except Exception:
        return False

    try:
        actual = _run_metadata_once(entry, callable_op, dtype, device, manifest)
        synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            f"{entry['name']} metadata execution failed on {device}: "
            f"{type(exc).__name__}: {exc}; schema={schema}; dtype={dtype}"
        ) from exc

    _metadata_return_equal(actual, expected, entry_name=entry["name"], dtype=dtype, compare=compare)
    return True


def run_manual_metadata_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    if entry is None:
        pytest.skip("No default coverage audit found for generated metadata tests")
    if entry.get("status") == "unknown":
        pytest.skip("coverage_unknown")
    if entry.get("status") == "excluded":
        pytest.skip("coverage_excluded")

    strategy = entry.get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") != "manual_metadata":
        skip_until_strategy_exists(entry, "metadata")

    callable_op = _dispatcher_callable(entry)
    tested_any = False
    for dtype, _dtype_str in _manifest_dtype_items(manifest):
        if _run_manual_metadata_case(entry, callable_op, dtype, device, compare, manifest):
            tested_any = True

    if not tested_any:
        pytest.skip(f"coverage_strategy_pending: no manifest-enabled metadata cases for {entry['name']}")


def _run_padding_once(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    input_condition: str,
    device: str,
    manifest: dict,
) -> torch.Tensor:
    sample = sample_padding(
        entry,
        dtype,
        device=device,
        input_condition=input_condition,
        seed=manifest.get("ieee754_seed", 67),
    )
    args = sample.call_args()
    kwargs = dict(sample.kwargs)
    out_kwargs = _multi_output_out_kwargs(entry, dtype, device)
    kwargs.update(out_kwargs)
    returned = callable_op(*args, **kwargs)
    _assert_multi_output_identity(returned, out_kwargs, entry["name"])
    if not isinstance(returned, torch.Tensor):
        raise AssertionError(f"{entry['name']} returned {type(returned).__name__}, expected Tensor")
    return returned


def _run_manual_padding_case(
    entry: dict,
    callable_op,
    dtype: torch.dtype,
    input_condition: str,
    device: str,
    compare,
    manifest: dict,
) -> bool:
    schema = entry.get("schema", "")
    try:
        expected = _run_padding_once(entry, callable_op, dtype, input_condition, "cpu", manifest)
    except Exception:
        return False

    try:
        actual = _run_padding_once(entry, callable_op, dtype, input_condition, device, manifest)
        synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            f"{entry['name']} padding execution failed on {device}: "
            f"{type(exc).__name__}: {exc}; schema={schema}; input_condition={input_condition}; dtype={dtype}"
        ) from exc

    if actual.device.type != torch.device(device).type:
        raise AssertionError(f"{entry['name']} returned tensor on {actual.device}, expected {device}")
    if actual.dtype != expected.dtype:
        raise AssertionError(f"{entry['name']} dtype mismatch: {actual.dtype} vs {expected.dtype}")
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(f"{entry['name']} shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}")
    if input_condition != InputCondition.CLEAN:
        _compare_special_tier(actual, expected, input_condition)
    else:
        compare(actual, expected, category="copy", dtype=dtype)
    return True


def run_manual_padding_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    if entry is None:
        pytest.skip("No default coverage audit found for generated padding tests")
    if entry.get("status") == "unknown":
        pytest.skip("coverage_unknown")
    if entry.get("status") == "excluded":
        pytest.skip("coverage_excluded")

    strategy = entry.get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") != "manual_padding":
        skip_until_strategy_exists(entry, "padding")

    callable_op = _dispatcher_callable(entry)
    tested_any = False
    for dtype, _dtype_str in _manifest_dtype_items(manifest):
        for input_condition in _manual_input_conditions(manifest, entry["base_name"], dtype):
            if _run_manual_padding_case(entry, callable_op, dtype, input_condition, device, compare, manifest):
                tested_any = True

    if not tested_any:
        pytest.skip(f"coverage_strategy_pending: no manifest-enabled padding cases for {entry['name']}")


def _foreach_domain(foreach_name: str) -> str:
    if foreach_name in {"acos", "asin", "atanh"}:
        return "unit"
    if foreach_name in {"lgamma", "log", "log10", "log1p", "log2", "pow", "reciprocal", "rsqrt", "sqrt"}:
        return "positive"
    if foreach_name in {"div"}:
        return "nonzero"
    return "mixed"


def _manual_tensor_values(dtype: torch.dtype, device: str, offset: float = 0.0, domain: str = "mixed") -> torch.Tensor:
    if domain == "unit":
        base = torch.linspace(-0.75 + offset, 0.75 + offset, 12, dtype=torch.float32).reshape(3, 4)
        base = base.clamp(-0.9, 0.9)
    elif domain in {"positive", "nonzero"}:
        base = torch.linspace(0.25 + offset, 1.75 + offset, 12, dtype=torch.float32).reshape(3, 4)
    else:
        base = torch.linspace(-1.25 + offset, 1.25 + offset, 12, dtype=torch.float32).reshape(3, 4)

    if dtype == torch.bool:
        tensor = base > 0
    elif dtype.is_complex:
        tensor = torch.complex(base, base / 4).to(dtype)
    elif dtype.is_floating_point:
        tensor = base.to(dtype)
    else:
        tensor = torch.round(base * 4).to(dtype)
        if domain == "nonzero":
            tensor = torch.where(tensor == 0, torch.ones_like(tensor), tensor)
    return tensor.to(device)


def _manual_scalar_tensor(dtype: torch.dtype, value, device: str = "cpu") -> torch.Tensor:
    if dtype == torch.bool:
        value = bool(value)
    elif dtype.is_complex:
        value = complex(value, value / 4)
    elif not dtype.is_floating_point:
        value = int(value)
    return torch.tensor(value, dtype=dtype, device=device)


def _manual_scalar_list(dtype: torch.dtype, values):
    if dtype == torch.bool:
        return [bool(value) for value in values]
    if dtype.is_complex:
        return [complex(value, value / 4) for value in values]
    if dtype.is_floating_point:
        return [float(value) for value in values]
    return [int(value) for value in values]


def _manual_packed_scalars(dtype: torch.dtype, values, device: str = "cpu") -> torch.Tensor:
    if dtype == torch.bool:
        values = [bool(value) for value in values]
    elif dtype.is_complex:
        values = [complex(value, value / 4) for value in values]
    elif not dtype.is_floating_point:
        values = [int(value) for value in values]
    return torch.tensor(values, dtype=dtype, device=device)


def _manual_foreach_sample(entry: dict, dtype: torch.dtype, input_condition: str, seed: int):
    return sample_foreach(entry, dtype, device="cpu", input_condition=input_condition, seed=seed)


def _move_foreach_args_to_device(entry: dict, cpu_args, device: str):
    dev_args = _move_to_device(cpu_args, device)
    strategy = entry.get("generated", {}).get("strategy") or {}
    if strategy.get("family") == "ternary" and strategy.get("overload") == "Tensor" and len(cpu_args) >= 3:
        items = list(dev_args)
        items[2] = cpu_args[2]
        return tuple(items)
    return dev_args


def _empty_like_tensor_list(tensors, device: str):
    return [
        torch.empty_strided(
            tuple(tensor.shape),
            tuple(tensor.stride()),
            dtype=tensor.dtype,
            device=device,
        )
        for tensor in tensors
    ]


def _compare_tensor_list(actual, expected, input_condition: str, dtype, compare, category: str) -> None:
    if not isinstance(actual, (list, tuple)) or not isinstance(expected, (list, tuple)):
        raise AssertionError(f"Expected Tensor list outputs, got {type(actual).__name__} and {type(expected).__name__}")
    if len(actual) != len(expected):
        raise AssertionError(f"Tensor list output length mismatch: {len(actual)} vs {len(expected)}")
    for actual_item, expected_item in zip(actual, expected):
        if not isinstance(actual_item, torch.Tensor) or not isinstance(expected_item, torch.Tensor):
            raise AssertionError("Foreach output list contains a non-Tensor value")
        if input_condition != InputCondition.CLEAN:
            _compare_special_tier(actual_item, expected_item, input_condition)
        else:
            compare(actual_item, expected_item, category=category, dtype=dtype)


def run_manual_foreach_strategy(entry: dict | None, device: str, compare, manifest: dict) -> None:
    if entry is None:
        pytest.skip("No default coverage audit found for generated foreach/fused tests")
    if entry.get("status") == "unknown":
        pytest.skip("coverage_unknown")
    if entry.get("status") == "excluded":
        pytest.skip("coverage_excluded")

    strategy = entry.get("generated", {}).get("strategy") or {}
    if strategy.get("strategy") != "manual_foreach":
        skip_until_strategy_exists(entry, "foreach_fused")

    callable_op = _dispatcher_callable(entry)
    tested_any = False
    ieee754_seed = manifest.get("ieee754_seed", 67)
    surface_kind = entry.get("surface_kind")

    for dtype, _dtype_str in _manifest_dtype_items(manifest):
        for input_condition in _manual_input_conditions(manifest, entry["base_name"], dtype):
            sample = _manual_foreach_sample(entry, dtype, input_condition, ieee754_seed)
            cpu_self = sample.input
            cpu_args = sample.args
            cpu_kwargs = sample.kwargs
            try:
                if surface_kind == "out_variant":
                    expected_out = _empty_like_tensor_list(cpu_self, "cpu")
                    returned = callable_op(cpu_self, *cpu_args, **cpu_kwargs, out=expected_out)
                    if returned is not None:
                        raise AssertionError(f"{entry['name']} returned {returned!r}, expected None for out variant")
                    expected = expected_out
                elif surface_kind == "mutating_or_inplace":
                    expected = [_clone_writable_input(item) for item in cpu_self]
                    expected_before = [item.detach().clone(memory_format=torch.preserve_format) for item in expected]
                    returned = callable_op(expected, *cpu_args, **cpu_kwargs)
                    if returned is not None:
                        raise AssertionError(f"{entry['name']} returned {returned!r}, expected None for in-place variant")
                    expected_changed = any(
                        not _tensor_content_equal(after, before_item)
                        for after, before_item in zip(expected, expected_before)
                    )
                else:
                    expected = callable_op(cpu_self, *cpu_args, **cpu_kwargs)
                    expected_changed = False
            except Exception:
                continue

            dev_self = _move_to_device(cpu_self, device)
            dev_args = _move_foreach_args_to_device(entry, cpu_args, device)
            dev_kwargs = _move_to_device(cpu_kwargs, device)
            try:
                if surface_kind == "out_variant":
                    actual_out = _empty_like_tensor_list(dev_self, device)
                    returned = callable_op(dev_self, *dev_args, **dev_kwargs, out=actual_out)
                    if returned is not None:
                        raise AssertionError(f"{entry['name']} returned {returned!r}, expected None for out variant")
                    actual = actual_out
                elif surface_kind == "mutating_or_inplace":
                    actual = [_clone_writable_input(item).to(device) for item in cpu_self]
                    before = [item.detach().clone(memory_format=torch.preserve_format) for item in actual]
                    returned = callable_op(actual, *dev_args, **dev_kwargs)
                    if returned is not None:
                        raise AssertionError(f"{entry['name']} returned {returned!r}, expected None for in-place variant")
                    actual_changed = any(
                        not _tensor_content_equal(after, before_item)
                        for after, before_item in zip(actual, before)
                    )
                    if expected_changed and not actual_changed:
                        raise AssertionError(f"{entry['name']} did not mutate any input tensor")
                else:
                    actual = callable_op(dev_self, *dev_args, **dev_kwargs)
                synchronize(device)
            except Exception as exc:
                raise RuntimeError(f"{entry['name']} foreach execution failed on {device}: {exc}") from exc

            _compare_tensor_list(actual, expected, input_condition, dtype, compare, "elementwise")
            if surface_kind == "functional_data":
                if isinstance(dev_self, (list, tuple)):
                    for actual_item, input_item in zip(actual, dev_self):
                        if _shares_storage_alias(actual_item, input_item):
                            raise AssertionError(f"{entry['name']} returned an alias from a functional foreach overload")
            elif surface_kind == "out_variant":
                for actual_item, out_item in zip(actual, actual_out):
                    if not _shares_storage_alias(actual_item, out_item):
                        raise AssertionError(f"{entry['name']} did not write through the provided out tensor")
            tested_any = True

    if not tested_any:
        pytest.skip(f"coverage_strategy_pending: no manifest-enabled foreach cases for {entry['name']}")
