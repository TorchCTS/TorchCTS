# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import torchcts.conftest as harness
from torchcts.core import adaptive_isolation
from torchcts.core import known_segfaults
from torchcts.core import triage
from torchcts.core.device import stream_context
from torchcts.core.report import get_hardware_key
from torchcts.generated import coverage_helpers


pytestmark = pytest.mark.covers_category("selftest")
_SOURCE_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_CHECKOUT_ONLY = pytest.mark.skipif(
    not (_SOURCE_REPO_ROOT / "pyproject.toml").exists(),
    reason="source checkout test requires repository files",
)


def _adaptive_result_payload(results, *, device="mps", hardware="hw", version="2.12.1", completed=False):
    return {
        "metadata": {
            "device_name": device,
            "hardware_key": hardware,
            "pytorch_version": version,
            "timestamp": "2026-06-28T00:00:00Z",
            "session_completed": completed,
        },
        "results": results,
        "skips": {},
    }


def _known_segfault_entry(**overrides):
    entry = {
        "id": "mps-example",
        "backend": "mps",
        "match": "nodeid",
        "nodeid": "torchcts/example.py::test_crash",
        "dispatcher": "aten::example.default",
        "evidence_scope": "exact_node",
        "classification": "confirmed_backend_crash",
        "expected_signal": "SIGSEGV",
        "repro": {"script": "repro.py", "case": "case0"},
        "reason": "standalone repro crashes",
        "owner": "torchcts",
        "pytorch_min": "2.12.0",
        "pytorch_max": None,
        "hardware": "any",
        "review_after": "2026-09-30",
    }
    for key, value in overrides.items():
        if value is None:
            entry.pop(key, None)
        else:
            entry[key] = value
    return entry


def _generated_item(nodeid, entry, *, fspath="torchcts/generated/test_out_variants.py", dtype=None):
    params = {"entry": entry}
    if dtype is not None:
        params["dtype"] = dtype
    return SimpleNamespace(
        nodeid=nodeid,
        fspath=fspath,
        name=nodeid.rsplit("::", 1)[-1],
        callspec=SimpleNamespace(params=params),
        iter_markers=lambda name=None: iter(()),
    )


def _reflection_pad3d_out_entry():
    return {
        "name": "aten::reflection_pad3d.out",
        "schema": "aten::reflection_pad3d.out(Tensor self, SymInt[6] padding, *, Tensor(a!) out) -> Tensor(a!)",
        "status": "covered_generated",
        "coverage_kind": "generated",
        "surface_kind": "out_variant",
        "variant_kind": "out_variant",
        "semantic_level": 3,
        "generated": {
            "strategy": {
                "strategy": "manual_padding",
                "family": "reflection_pad3d",
            }
        },
    }


def _unfold_view_alias_entry():
    return {
        "name": "aten::unfold",
        "schema": "aten::unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)",
        "status": "covered_generated",
        "coverage_kind": "generated",
        "surface_kind": "view_or_alias",
        "variant_kind": "view",
        "semantic_level": 3,
        "generated": {
            "strategy": {
                "strategy": "opinfo_view_alias",
                "opinfo_name": "unfold",
            }
        },
    }


def _hamming_window_periodic_factory_entry():
    return {
        "name": "aten::hamming_window.periodic",
        "schema": (
            "aten::hamming_window.periodic(int window_length, bool periodic, *, "
            "ScalarType? dtype=None, Layout? layout=None, Device? device=None, "
            "bool? pin_memory=None) -> Tensor"
        ),
        "status": "covered_generated",
        "coverage_kind": "generated",
        "surface_kind": "factory",
        "variant_kind": "factory",
        "semantic_level": 2,
        "generated": {
            "strategy": {
                "strategy": "manual_factory",
                "family": "window",
            }
        },
    }


def _col2im_functional_entry():
    return {
        "name": "aten::col2im",
        "schema": (
            "aten::col2im(Tensor self, SymInt[2] output_size, int[2] kernel_size, "
            "int[2] dilation, int[2] padding, int[2] stride) -> Tensor"
        ),
        "status": "covered_generated",
        "coverage_kind": "generated",
        "surface_kind": "functional_data",
        "variant_kind": "functional",
        "semantic_level": 4,
        "generated": {
            "strategy": {
                "strategy": "manual_convolution",
                "family": "col2im",
            }
        },
    }


def _foreach_add_list_functional_entry():
    return {
        "name": "aten::_foreach_add.List",
        "schema": "aten::_foreach_add.List(Tensor[] self, Tensor[] other, *, Scalar alpha=1) -> Tensor[]",
        "status": "covered_generated",
        "coverage_kind": "generated",
        "surface_kind": "functional_data",
        "variant_kind": "functional",
        "semantic_level": 4,
        "generated": {
            "strategy": {
                "strategy": "manual_foreach",
                "family": "binary",
                "foreach_name": "add",
                "overload": "List",
            }
        },
    }


def _foreach_add_scalarlist_functional_entry():
    return {
        "name": "aten::_foreach_add.ScalarList",
        "schema": "aten::_foreach_add.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[]",
        "status": "covered_generated",
        "coverage_kind": "generated",
        "surface_kind": "functional_data",
        "variant_kind": "functional",
        "semantic_level": 4,
        "generated": {
            "strategy": {
                "strategy": "manual_foreach",
                "family": "binary",
                "foreach_name": "add",
                "overload": "ScalarList",
            }
        },
    }


def _foreach_addcdiv_scalar_functional_entry():
    return {
        "name": "aten::_foreach_addcdiv.Scalar",
        "schema": (
            "aten::_foreach_addcdiv.Scalar(Tensor[] self, Tensor[] tensor1, "
            "Tensor[] tensor2, Scalar value=1) -> Tensor[]"
        ),
        "status": "covered_generated",
        "coverage_kind": "generated",
        "surface_kind": "functional_data",
        "variant_kind": "functional",
        "semantic_level": 4,
        "generated": {
            "strategy": {
                "strategy": "manual_foreach",
                "family": "ternary",
                "foreach_name": "addcdiv",
                "overload": "Scalar",
            }
        },
    }


def _native_batch_norm_backward_entry():
    return {
        "name": "aten::native_batch_norm_backward",
        "schema": (
            "aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, "
            "Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, "
            "bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)"
        ),
        "status": "covered_generated",
        "coverage_kind": "generated",
        "surface_kind": "autograd_backward",
        "variant_kind": "functional",
        "semantic_level": 3,
        "generated": {
            "strategy": {
                "strategy": "manual_multi_output_reduction",
                "family": "native_batch_norm_backward",
            }
        },
    }


def test_mps_triage_queue_loads_failures_and_crashers(tmp_path):
    runlog = tmp_path / "Apple_M3_Max_128gb_runlog.txt"
    runlog.write_text(
        "    1.0s  torchcts/a.py::test_ok\n"
        "    2.0s  torchcts/b.py::test_last\n",
        encoding="utf-8",
    )
    result = {
        "results": {
            "torchcts/c.py::test_pass": {"status": "PASS"},
            "torchcts/d.py::test_fail": {"status": "FAIL"},
            "torchcts/e.py::test_error": {"status": "ERROR"},
        }
    }

    queue = triage.build_triage_queue(result, include_crashers=True, runlog_path=runlog)

    assert triage.DEFAULT_MPS_CRASH_NODES[0] in queue
    assert "torchcts/b.py::test_last" in queue
    assert "torchcts/d.py::test_fail" in queue
    assert "torchcts/e.py::test_error" in queue
    assert "torchcts/c.py::test_pass" not in queue


def test_adaptive_isolation_loads_crash_signal_and_timeout_candidates(tmp_path):
    latest = tmp_path / "hw_latest.json"
    latest.write_text(
        json.dumps(
            _adaptive_result_payload(
                {
                    "torchcts/a.py::test_crash": {
                        "status": "ERROR",
                        "error_type": "ProcessCrash",
                        "subprocess": {"signal": "SIGSEGV"},
                    },
                    "torchcts/b.py::test_signal": {
                        "status": "ERROR",
                        "error_type": "SubprocessFailure",
                        "subprocess": {"signal": "SIGABRT"},
                    },
                    "torchcts/c.py::test_timeout": {
                        "status": "ERROR",
                        "error_type": "TimeoutError",
                        "phase": "subprocess",
                        "failure_stage": "subprocess_timeout",
                        "subprocess": {"timed_out": True},
                    },
                }
            )
        ),
        encoding="utf-8",
    )

    loaded = adaptive_isolation.load_adaptive_isolation(
        tmp_path,
        hardware_key="hw",
        device_name="mps",
        torch_version="2.12.1",
    )

    assert loaded.candidates["torchcts/a.py::test_crash"].isolation_source == "adaptive_previous_crash"
    assert loaded.candidates["torchcts/b.py::test_signal"].prior_signal == "SIGABRT"
    assert loaded.candidates["torchcts/c.py::test_timeout"].isolation_source == "adaptive_previous_timeout"


def test_adaptive_isolation_ignores_mismatched_and_malformed_artifacts(tmp_path):
    (tmp_path / "hw_latest.json").write_text("{not json", encoding="utf-8")
    history = tmp_path / "hw_history"
    history.mkdir()
    (history / "2026-06-27T00-00-00Z.json").write_text(
        json.dumps(
            _adaptive_result_payload(
                {
                    "torchcts/a.py::test_crash": {
                        "status": "ERROR",
                        "error_type": "ProcessCrash",
                    }
                },
                device="cpu",
            )
        ),
        encoding="utf-8",
    )

    loaded = adaptive_isolation.load_adaptive_isolation(
        tmp_path,
        hardware_key="hw",
        device_name="mps",
        torch_version="2.12.1",
    )

    assert not loaded.candidates
    assert loaded.warnings
    assert loaded.rejected[0]["reason"] == "metadata_mismatch"


def test_adaptive_isolation_requires_matching_latest_for_runlog_hang(tmp_path):
    (tmp_path / "hw_latest.json").write_text(
        json.dumps(_adaptive_result_payload({}, device="cpu", completed=False)),
        encoding="utf-8",
    )
    history = tmp_path / "hw_history"
    history.mkdir()
    (history / "2026-06-27T00-00-00Z.json").write_text(
        json.dumps(_adaptive_result_payload({}, completed=False)),
        encoding="utf-8",
    )
    (tmp_path / "hw_runlog.txt").write_text(
        "     1.0s  torchcts/hangs.py::test_hang\n",
        encoding="utf-8",
    )

    loaded = adaptive_isolation.load_adaptive_isolation(
        tmp_path,
        hardware_key="hw",
        device_name="mps",
        torch_version="2.12.1",
    )

    assert not loaded.candidates


def test_adaptive_isolation_scans_only_five_history_artifacts(tmp_path):
    history = tmp_path / "hw_history"
    history.mkdir()
    for index in range(6):
        (history / f"2026-06-27T00-00-0{index}Z.json").write_text(
            json.dumps(_adaptive_result_payload({}, completed=True)),
            encoding="utf-8",
        )

    loaded = adaptive_isolation.load_adaptive_isolation(
        tmp_path,
        hardware_key="hw",
        device_name="mps",
        torch_version="2.12.1",
    )

    assert len(loaded.artifacts_considered) == 5


def test_adaptive_isolation_uses_runlog_for_high_confidence_hang(tmp_path):
    (tmp_path / "hw_latest.json").write_text(
        json.dumps(
            _adaptive_result_payload(
                {
                    "torchcts/a.py::test_done": {"status": "PASS"},
                },
                completed=False,
            )
        ),
        encoding="utf-8",
    )
    (tmp_path / "hw_runlog.txt").write_text(
        "     1.0s  torchcts/a.py::test_done\n"
        "     2.0s  torchcts/hangs.py::test_hang\n",
        encoding="utf-8",
    )

    loaded = adaptive_isolation.load_adaptive_isolation(
        tmp_path,
        hardware_key="hw",
        device_name="mps",
        torch_version="2.12.1",
    )

    candidate = loaded.candidates["torchcts/hangs.py::test_hang"]
    assert candidate.isolation_source == "adaptive_suspected_hang"


def test_adaptive_isolation_does_not_infer_hang_from_result_or_completed_session(tmp_path):
    runlog = tmp_path / "hw_runlog.txt"
    runlog.write_text("     1.0s  torchcts/a.py::test_done\n", encoding="utf-8")
    latest = tmp_path / "hw_latest.json"
    latest.write_text(
        json.dumps(
            _adaptive_result_payload(
                {"torchcts/a.py::test_done": {"status": "PASS"}},
                completed=False,
            )
        ),
        encoding="utf-8",
    )

    loaded = adaptive_isolation.load_adaptive_isolation(
        tmp_path,
        hardware_key="hw",
        device_name="mps",
        torch_version="2.12.1",
    )
    assert not loaded.candidates

    latest.write_text(
        json.dumps(_adaptive_result_payload({}, completed=True)),
        encoding="utf-8",
    )
    loaded = adaptive_isolation.load_adaptive_isolation(
        tmp_path,
        hardware_key="hw",
        device_name="mps",
        torch_version="2.12.1",
    )
    assert not loaded.candidates
    assert any(item["reason"] == "latest_json_reports_completed_session" for item in loaded.rejected)


def test_adaptive_isolation_newer_pass_resolves_older_crash(tmp_path):
    (tmp_path / "hw_latest.json").write_text(
        json.dumps(
            _adaptive_result_payload(
                {"torchcts/a.py::test_flaky": {"status": "PASS"}},
                completed=True,
            )
        ),
        encoding="utf-8",
    )
    history = tmp_path / "hw_history"
    history.mkdir()
    (history / "2026-06-27T00-00-00Z.json").write_text(
        json.dumps(
            _adaptive_result_payload(
                {
                    "torchcts/a.py::test_flaky": {
                        "status": "ERROR",
                        "error_type": "ProcessCrash",
                    }
                }
            )
        ),
        encoding="utf-8",
    )

    loaded = adaptive_isolation.load_adaptive_isolation(
        tmp_path,
        hardware_key="hw",
        device_name="mps",
        torch_version="2.12.1",
    )

    assert not loaded.candidates
    assert loaded.rejected[0]["reason"] == "newer_nonisolating_record_resolved_candidate"


def test_adaptive_isolation_filters_stale_nodeids_from_collection(tmp_path):
    loaded = adaptive_isolation.AdaptiveIsolationLoadResult(
        candidates={
            "torchcts/a.py::test_keep": adaptive_isolation.AdaptiveIsolationCandidate(
                nodeid="torchcts/a.py::test_keep",
                canonical_nodeid="torchcts/a.py::test_keep",
                isolation_source="adaptive_previous_crash",
                reason="crashed",
                evidence_path="results/hw_latest.json",
            ),
            "torchcts/b.py::test_drop": adaptive_isolation.AdaptiveIsolationCandidate(
                nodeid="torchcts/b.py::test_drop",
                canonical_nodeid="torchcts/b.py::test_drop",
                isolation_source="adaptive_previous_crash",
                reason="crashed",
                evidence_path="results/hw_latest.json",
            ),
        },
        rejected=[],
        warnings=[],
        artifacts_considered=[],
    )

    accepted, rejected = adaptive_isolation.filter_candidates_for_collection(
        loaded,
        ["torchcts/a.py::test_keep"],
    )

    assert sorted(accepted) == ["torchcts/a.py::test_keep"]
    assert rejected == [
        {
            "nodeid": "torchcts/b.py::test_drop",
            "canonical_nodeid": "torchcts/b.py::test_drop",
            "path": "results/hw_latest.json",
            "reason": "not_collected_in_current_run",
        }
    ]


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"status": "ERROR", "error_type": "ProcessCrash", "error_message": "Fatal Python error"}, "confirmed_mps_crash"),
        ({"status": "ERROR", "error_type": "RuntimeError", "error_message": "DispatchStub: missing kernel for mps"}, "confirmed_mps_missing_kernel"),
        ({"status": "ERROR", "error_type": "TypeError", "error_message": "Trying to convert Float8_e4m3fn to the MPS backend"}, "manifest_overclaim"),
        ({"status": "ERROR", "error_type": "RuntimeError", "error_message": "NYI: named tensors only support CPU, CUDA, XPU or privateuseone tensors."}, "manifest_overclaim"),
        ({"status": "ERROR", "error_type": "RuntimeError", "error_message": "NestedTensorImpl storage must be either CPU, CUDA, XPU, HPU or privateuseone but got MPS"}, "manifest_overclaim"),
        ({"status": "ERROR", "error_type": "RuntimeError", "error_message": "NestedTensorImpl storage must be either CUDA, CPU, XPU or privateuseone but got mps:0"}, "manifest_overclaim"),
        ({"status": "ERROR", "error_type": "RuntimeError", "error_message": "device type of values (mps) must be one of CPU, CUDA, XPU, Meta or PrivateUse1"}, "confirmed_mps_unsupported_dtype_or_layout"),
        ({"status": "ERROR", "error_type": "RuntimeError", "error_message": "Expected N % 32 == 0 && K % 32 == 0 to be true, but got false."}, "confirmed_mps_unsupported_dtype_or_layout"),
        ({"status": "ERROR", "error_type": "RuntimeError", "error_message": "Promotion for uint16, uint32, uint64 types is not supported, attempted to promote UInt16 and Long"}, "confirmed_mps_unsupported_dtype_or_layout"),
        (
            {
                "nodeid": "torchcts/generated/test_foreach_fused.py::test_generated_foreach_or_fused[_foreach_acos[L4]-torch.uint16]",
                "status": "ERROR",
                "error_type": "RuntimeError",
                "error_message": "aten::_foreach_acos foreach execution failed on mps: Failed to create function state object for: acos_dense_float_ushort",
            },
            "confirmed_mps_missing_kernel",
        ),
        (
            {
                "nodeid": "torchcts/generated/test_foreach_fused.py::test_generated_foreach_or_fused[_foreach_addcmul.Tensor[L4]-torch.complex64]",
                "status": "ERROR",
                "error_type": "RuntimeError",
                "error_message": "aten::_foreach_addcmul.Tensor foreach execution failed on mps: value cannot be converted to type double without overflow",
            },
            "confirmed_mps_wrong_value",
        ),
        (
            {
                "nodeid": "torchcts/generated/test_functional_variants.py::test_generated_functional_variant[adaptive_avg_pool2d[L2]]",
                "status": "ERROR",
                "error_type": "RuntimeError",
                "error_message": "Adaptive pool MPS: input sizes must be divisible by output sizes. Non-divisible input sizes are not implemented on MPS device yet.",
            },
            "confirmed_mps_missing_kernel",
        ),
        (
            {
                "nodeid": "torchcts/generated/test_functional_variants.py::test_generated_functional_variant[addr[L2]]",
                "status": "ERROR",
                "error_type": "RuntimeError",
                "error_message": "MPS device does not support addr for non-float input",
            },
            "confirmed_mps_unsupported_dtype_or_layout",
        ),
        (
            {
                "nodeid": "torchcts/generated/test_out_variants.py::test_generated_out_variant[index_reduce.out[L2]]",
                "status": "ERROR",
                "error_type": "RuntimeError",
                "error_message": "Failed to create function state object for: index_reduce_amax_bool_long",
            },
            "confirmed_mps_missing_kernel",
        ),
        ({"status": "ERROR", "error_type": "NotImplementedError", "error_message": "\"arange_cpu\" not implemented for 'UInt16'"}, "torchcts_bad_oracle"),
        ({"status": "ERROR", "error_type": "NotImplementedError", "error_message": "\"equal_cpu\" not implemented for 'ComplexHalf'"}, "torchcts_bad_assertion"),
        ({"status": "ERROR", "error_type": "RuntimeError", "error_message": "Expected a 'cpu' device type for generator but found 'mps'"}, "torchcts_invalid_sample"),
        ({"status": "ERROR", "error_type": "RuntimeError", "error_message": "high must be a scalar tensor and on CPU"}, "torchcts_invalid_sample"),
        ({"status": "ERROR", "error_type": "RuntimeError", "error_message": "tensor_split expected tensor_indices_or_sections to be on cpu, but it's on mps:0"}, "torchcts_invalid_sample"),
        ({"status": "ERROR", "error_type": "RuntimeError", "error_message": "normal expects mean to be non-complex"}, "confirmed_mps_unsupported_dtype_or_layout"),
        ({"status": "ERROR", "error_type": "RuntimeError", "error_message": "mode only supports CPU, CUDA and XPU device type, got: mps"}, "confirmed_mps_missing_kernel"),
        ({"status": "ERROR", "error_type": "RuntimeError", "error_message": "Expected tensor to have CPU Backend, but got tensor with MPS Backend"}, "confirmed_mps_missing_kernel"),
        ({"status": "ERROR", "error_type": "NotImplementedError", "error_message": "\"nan_to_num_mps\" not implemented for 'ComplexFloat'"}, "confirmed_mps_missing_kernel"),
        (
            {
                "nodeid": "torchcts/generated/test_out_variants.py::test_generated_out_variant[conj_physical.out[L1]]",
                "status": "ERROR",
                "error_type": "RuntimeError",
                "error_message": "Expected self.is_complex() to be true, but got false.",
            },
            "confirmed_mps_wrong_value",
        ),
        ({"status": "FAIL", "error_type": "AssertionError", "error_message": "Expected exception RuntimeError not raised for op dot"}, "confirmed_mps_wrong_value"),
        ({"status": "FAIL", "error_type": "AssertionError", "error_message": "CPU raised RuntimeError but device succeeded for nn.functional.binary_cross_entropy (has_nan): all elements of input should be between 0 and 1"}, "confirmed_mps_wrong_value"),
        ({"status": "FAIL", "error_type": "AssertionError", "error_message": "Inf propagation mismatch: 1 positions differ."}, "confirmed_mps_wrong_value"),
        ({"status": "FAIL", "error_type": "AssertionError", "error_message": "Inf sign mismatch at some positions."}, "confirmed_mps_wrong_value"),
        ({"status": "FAIL", "error_type": "AssertionError", "error_message": "aten::embedding_bag output 1 shape mismatch: (4,) vs (0,)"}, "confirmed_mps_wrong_value"),
        ({"status": "FAIL", "error_type": "AssertionError", "error_message": "aten::max.dim output 1 integer/index values differ"}, "confirmed_mps_wrong_value"),
        ({"status": "FAIL", "error_type": "AssertionError", "error_message": "aten::median.dim_values returned a different object than the provided out tensor"}, "confirmed_mps_wrong_value"),
        ({"status": "ERROR", "error_type": "RuntimeError", "error_message": "Expected out tensor to have dtype c10::complex<Half>, but got c10::Half instead"}, "confirmed_mps_wrong_value"),
        ({"status": "FAIL", "error_type": "AssertionError", "error_message": "aten::random_ produced values below 0"}, "confirmed_mps_wrong_value"),
        ({"status": "FAIL", "error_type": "AssertionError", "error_message": "Scalars are not close! Expected nan but got 0.6328125."}, "confirmed_mps_wrong_value"),
        ({"status": "ERROR", "error_type": "RuntimeError", "error_message": "shape '[72057599010901125, 0]' is invalid for input of size 12"}, "confirmed_mps_wrong_value"),
        ({"status": "ERROR", "error_type": "RuntimeError", "error_message": "INTERNAL ASSERT FAILED: Placeholder tensor is empty!"}, "confirmed_mps_wrong_value"),
        ({"status": "FAIL", "error_type": "AssertionError", "error_message": "aten::ravel alias mismatch: device alias=True, CPU alias=False"}, "confirmed_mps_wrong_value"),
        (
            {
                "nodeid": "torchcts/strides/test_noncontiguous.py::test_stride_and_storage_offset_metadata",
                "status": "FAIL",
                "error_type": "AssertionError",
                "error_message": "assert 2 == 1\nwhere 2 = view_dev.stride(0)\nwhere 1 = view_cpu.stride(0)",
            },
            "confirmed_mps_wrong_value",
        ),
        ({"status": "FAIL", "error_type": "AssertionError", "error_message": "Tensor-likes are not close!"}, "confirmed_mps_wrong_value"),
        ({"status": "FAIL", "error_type": "AssertionError", "error_message": "Tensor-likes are not equal!"}, "confirmed_mps_wrong_value"),
        ({"status": "ERROR", "error_type": "RuntimeError", "error_message": "Sparse CSR tensors do not have is_contiguous"}, "torchcts_bad_generated_strategy"),
        ({"status": "ERROR", "error_type": "Failed", "error_message": "Library not loaded: @rpath/libc++.1.dylib"}, "local_toolchain_or_environment"),
        ({"status": "ERROR", "error_type": "AttributeError", "error_message": "module 'torch' has no attribute 'stream'"}, "torchcts_bad_assertion"),
        ({"status": "ERROR", "error_type": "RuntimeError", "error_message": "unrecognized failure"}, "needs_more_evidence"),
    ],
)
def test_mps_triage_classifier(record, expected):
    assert triage.classify_record(record)["classification"] == expected


def test_mps_triage_classifier_identifies_fused_rms_norm_missing_return():
    record = {
        "nodeid": "torchcts/operators/test_norm.py::test_fused_rms_norm_dispatcher_variant",
        "status": "FAIL",
        "error_type": "AssertionError",
        "error_message": (
            "Tensor comparison requires tensor values: actual NoneType vs expected "
            "Tensor(shape=(2, 1), dtype=torch.float32, device=cpu)"
        ),
    }

    assert triage.classify_record(record)["classification"] == "confirmed_mps_wrong_value"


def test_mps_triage_classifier_identifies_saturate_weight_mismatch():
    record = {
        "nodeid": "torchcts/operators/test_misc.py::test_low_level_misc_dispatcher_helpers",
        "status": "FAIL",
        "error_type": "AssertionError",
        "error_message": "Tensor-likes are not close! Mismatched elements: 2 / 12",
    }

    assert triage.classify_record(record)["classification"] == "confirmed_mps_wrong_value"


@pytest.mark.parametrize(
    "nodeid",
    [
        "torchcts/generated/test_functional_variants.py::test_generated_functional_variant[_safe_softmax[L2]]",
        "torchcts/generated/test_out_variants.py::test_generated_out_variant[_softmax.out[L2]]",
        "torchcts/opinfo/test_opinfo_forward.py::test_op_forward[has_nan-softmax-torch.float32]",
        "torchcts/generated/test_functional_variants.py::test_generated_functional_variant[softmax.int[L2]]",
        "torchcts/generated/test_functional_variants.py::test_generated_functional_variant[_logcumsumexp[L2]]",
        "torchcts/generated/test_functional_variants.py::test_generated_functional_variant[logcumsumexp[L2]]",
        "torchcts/generated/test_functional_variants.py::test_generated_functional_variant[prod.dim_int[L2]]",
    ],
)
def test_mps_triage_classifier_identifies_reproduced_generated_reduction_ieee_mismatch(nodeid):
    record = {
        "nodeid": nodeid,
        "status": "FAIL",
        "error_type": "AssertionError",
        "error_message": "NaN propagation mismatch: 3 positions differ.",
    }

    assert triage.classify_record(record)["classification"] == "confirmed_mps_wrong_value"


@pytest.mark.parametrize(
    "record",
    [
        {
            "nodeid": "torchcts/opinfo/test_opinfo_forward.py::test_op_forward[clean-std-torch.complex64]",
            "status": "FAIL",
            "error_type": "AssertionError",
            "error_message": "Shape mismatch after comparison normalization: actual torch.Size([3, 2]) vs expected torch.Size([3])",
        },
        {
            "nodeid": "torchcts/opinfo/test_opinfo_forward.py::test_op_forward[has_nan-var_mean-torch.complex64]",
            "status": "FAIL",
            "error_type": "AssertionError",
            "error_message": "Dtype mismatch: got torch.complex64, expected torch.float32",
        },
    ],
)
def test_mps_triage_classifier_identifies_complex_reduction_contract_mismatch(record):
    assert triage.classify_record(record)["classification"] == "confirmed_mps_wrong_value"


def test_mps_triage_classifier_identifies_complex_cholesky_rejecting_cpu_valid_sample():
    record = {
        "nodeid": "torchcts/opinfo/test_opinfo_forward.py::test_op_forward[clean-linalg.cholesky-torch.complex64]",
        "status": "ERROR",
        "error_type": "RuntimeError",
        "error_message": "Execution failed on device mps: linalg.cholesky: The factorization could not be completed because the input is not positive-definite",
    }

    assert triage.classify_record(record)["classification"] == "confirmed_mps_wrong_value"


def test_mps_triage_classifier_prioritizes_grid_backward_wrong_value_over_crash():
    record = {
        "nodeid": "grid2_cpu_fallback_backward",
        "status": "ERROR",
    }
    subprocess_result = {
        "returncode": -11,
        "signal": "SIGSEGV",
        "timed_out": False,
        "stdout_tail": "",
        "stderr_tail": (
            "AssertionError: Tensor-likes are not close!\n"
            "Mismatched elements: 9 / 9 (100.0%)"
        ),
    }

    classification = triage.classify_record(record, subprocess_result)

    assert classification["classification"] == "confirmed_mps_wrong_value"


def test_mps_triage_subprocess_records_signal(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=-11,
            stdout="stdout before crash",
            stderr="Fatal Python error: Segmentation fault",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = triage.run_pytest_node(
        "torchcts/example.py::test_crash",
        triage_dir=tmp_path,
        timeout=1,
    )

    assert result["returncode"] == -11
    assert result["signal"] == "SIGSEGV"
    assert result["classification"]["classification"] == "confirmed_mps_crash"
    assert Path(result["stdout_path"]).exists()
    assert Path(result["stderr_path"]).exists()


def test_mps_triage_timeout_records_inconclusive(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], timeout=1, output="partial out", stderr="partial err")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = triage.run_subprocess_command(["python", "-c", "pass"], timeout=1)

    assert result["timed_out"] is True
    assert result["returncode"] is None
    assert "partial out" in result["stdout_tail"]
    assert "partial err" in result["stderr_tail"]


def test_known_segfault_schema_accepts_and_matches_entry():
    payload = {
        "version": 1,
        "known_segfaults": [_known_segfault_entry()],
    }

    entries = known_segfaults.validate_known_segfaults([payload])
    active = known_segfaults.active_known_segfaults(
        entries,
        backend="mps",
        torch_version="2.12.1+cpu",
        hardware_key="Apple_M3_Max_128gb",
    )
    item = SimpleNamespace(nodeid="torchcts/example.py::test_crash")

    assert known_segfaults.match_known_segfault(item, active)["id"] == "mps-example"


def test_known_segfault_matching_canonicalizes_installed_package_nodeids():
    payload = {
        "version": 1,
        "known_segfaults": [
            _known_segfault_entry(
                id="mps-installed-nodeid",
                nodeid="torchcts/generated/test_out_variants.py::test_generated_out_variant[range.out_[L2]]",
                dispatcher="aten::range.out_",
                repro={"script": "repro.py", "case": "range_out_"},
            )
        ],
    }

    entries = known_segfaults.validate_known_segfaults([payload])
    active = known_segfaults.active_known_segfaults(
        entries,
        backend="mps",
        torch_version="2.12.1",
        hardware_key="Apple_M3_Max_128gb",
    )
    item = SimpleNamespace(
        nodeid=(
            ".venv/lib/python3.14/site-packages/torchcts/generated/"
            "test_out_variants.py::test_generated_out_variant[range.out_[L2]]"
        )
    )

    assert known_segfaults.canonicalize_nodeid(item.nodeid) == payload["known_segfaults"][0]["nodeid"]
    assert known_segfaults.match_known_segfault(item, active)["id"] == "mps-installed-nodeid"


def test_packaged_known_segfaults_cover_generated_grid_sampler_crash_nodes():
    entries = known_segfaults.load_known_segfaults(Path.cwd())
    active = known_segfaults.active_known_segfaults(
        entries,
        backend="mps",
        torch_version="2.12.1",
        hardware_key="Apple_M3_Max_128gb",
    )

    expected = {
        "torchcts/generated/test_functional_variants.py::test_generated_functional_variant[_grid_sampler_2d_cpu_fallback[L3]]":
            ("mps-grid-sampler-2d-cpu-fallback-default-pytorch-2-12", "aten::_grid_sampler_2d_cpu_fallback"),
        "torchcts/generated/test_out_variants.py::test_generated_out_variant[_grid_sampler_2d_cpu_fallback.out[L3]]":
            ("mps-grid-sampler-2d-cpu-fallback-out-pytorch-2-12", "aten::_grid_sampler_2d_cpu_fallback.out"),
    }
    for nodeid, (expected_id, dispatcher_name) in expected.items():
        match = known_segfaults.match_known_segfault(
            SimpleNamespace(
                nodeid=nodeid,
                metadata={"dispatcher_name": dispatcher_name, "coverage_id": dispatcher_name},
            ),
            active,
        )
        assert match is not None
        assert match["id"] == expected_id


def test_packaged_known_segfaults_cover_generated_reflection_pad3d_out_node():
    entries = known_segfaults.load_known_segfaults(Path.cwd())
    active = known_segfaults.active_known_segfaults(
        entries,
        backend="mps",
        torch_version="2.12.1",
        hardware_key="Apple_M3_Max_128gb",
    )
    nodeid = "torchcts/generated/test_out_variants.py::test_generated_out_variant[reflection_pad3d.out[L3]]"
    item = _generated_item(nodeid, _reflection_pad3d_out_entry())

    match = known_segfaults.match_known_segfault(
        item,
        active,
        metadata=harness._extract_result_metadata(item),
    )

    assert match is not None
    assert match["id"] == "mps-reflection-pad3d-out-pytorch-2-12"
    assert match["matched_by"] == "dispatcher"
    assert match["evidence_scope"] == "dispatcher_surface"


def test_packaged_known_segfaults_cover_generated_unfold_view_alias_node():
    entries = known_segfaults.load_known_segfaults(Path.cwd())
    active = known_segfaults.active_known_segfaults(
        entries,
        backend="mps",
        torch_version="2.12.1",
        hardware_key="Apple_M3_Max_128gb",
    )
    nodeid = "torchcts/generated/test_view_aliases.py::test_generated_view_alias[unfold[L3]]"
    item = _generated_item(nodeid, _unfold_view_alias_entry())

    match = known_segfaults.match_known_segfault(
        item,
        active,
        metadata=harness._extract_result_metadata(item),
    )

    assert match is not None
    assert match["id"] == "mps-generated-unfold-view-copy-pytorch-2-12"
    assert match["matched_by"] == "dispatcher"
    assert match["evidence_scope"] == "dispatcher_surface"


def test_packaged_known_segfaults_cover_generated_hamming_window_periodic_node():
    entries = known_segfaults.load_known_segfaults(Path.cwd())
    active = known_segfaults.active_known_segfaults(
        entries,
        backend="mps",
        torch_version="2.12.1",
        hardware_key="Apple_M3_Max_128gb",
    )
    nodeid = "torchcts/generated/test_factories.py::test_generated_factory[hamming_window.periodic[L2]]"
    item = _generated_item(
        nodeid,
        _hamming_window_periodic_factory_entry(),
        fspath="torchcts/generated/test_factories.py",
    )

    match = known_segfaults.match_known_segfault(
        item,
        active,
        metadata=harness._extract_result_metadata(item),
    )

    assert match is not None
    assert match["id"] == "mps-hamming-window-periodic-generated-factory-pytorch-2-12"
    assert match["matched_by"] == "dispatcher"
    assert match["evidence_scope"] == "constrained_metadata"
    assert match["constraints"]["strategy_family"] == ["window"]


def test_packaged_known_segfaults_cover_generated_col2im_node():
    entries = known_segfaults.load_known_segfaults(Path.cwd())
    active = known_segfaults.active_known_segfaults(
        entries,
        backend="mps",
        torch_version="2.12.1",
        hardware_key="Apple_M3_Max_128gb",
    )
    nodeid = "torchcts/generated/test_functional_variants.py::test_generated_functional_variant[col2im[L4]]"
    item = _generated_item(
        nodeid,
        _col2im_functional_entry(),
        fspath="torchcts/generated/test_functional_variants.py",
    )

    match = known_segfaults.match_known_segfault(
        item,
        active,
        metadata=harness._extract_result_metadata(item),
    )

    assert match is not None
    assert match["id"] == "mps-col2im-generated-functional-pytorch-2-12"
    assert match["matched_by"] == "dispatcher"
    assert match["evidence_scope"] == "constrained_metadata"
    assert match["constraints"]["strategy_family"] == ["col2im"]


def test_packaged_known_segfaults_cover_generated_manual_foreach_nodes():
    entries = known_segfaults.load_known_segfaults(Path.cwd())
    active = known_segfaults.active_known_segfaults(
        entries,
        backend="mps",
        torch_version="2.12.1",
        hardware_key="Apple_M3_Max_128gb",
    )
    crash_entries = {
        "_foreach_add.List[L4]": _foreach_add_list_functional_entry(),
        "_foreach_add.ScalarList[L4]": _foreach_add_scalarlist_functional_entry(),
        "_foreach_addcdiv.Scalar[L4]": _foreach_addcdiv_scalar_functional_entry(),
    }
    for case_id, entry in crash_entries.items():
        for dtype in (torch.float64, torch.complex128):
            dtype_str = str(dtype)
            nodeid = (
                "torchcts/generated/test_foreach_fused.py::"
                f"test_generated_foreach_or_fused[{case_id}-{dtype_str}]"
            )
            item = _generated_item(
                nodeid,
                entry,
                fspath=nodeid.split("::", 1)[0],
                dtype=dtype,
            )
            metadata = harness._extract_result_metadata(item)
            match = known_segfaults.match_known_segfault(
                item,
                active,
                metadata=metadata,
            )

            assert match is not None
            assert match["id"] == "mps-generated-manual-foreach-pytorch-2-12"
            assert match["matched_by"] == "coverage_id"
            assert match["evidence_scope"] == "constrained_metadata"
            assert match["constraints"]["coverage_id_glob"] == ["aten::_foreach_*"]
            assert match["constraints"]["dtype"] == ["torch.float64", "torch.complex128"]
            assert match["matched_metadata"]["dtype"] == dtype_str
            assert any(
                glob.startswith(nodeid.split("::", 1)[0])
                for glob in match["constraints"]["nodeid_glob"]
            )

    safe_entry = _foreach_add_list_functional_entry()
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        dtype_str = str(dtype)
        nodeid = (
            "torchcts/generated/test_foreach_fused.py::"
            f"test_generated_foreach_or_fused[_foreach_add.List[L4]-{dtype_str}]"
        )
        item = _generated_item(
            nodeid,
            safe_entry,
            fspath=nodeid.split("::", 1)[0],
            dtype=dtype,
        )

        assert known_segfaults.match_known_segfault(
            item,
            active,
            metadata=harness._extract_result_metadata(item),
        ) is None


def test_packaged_known_segfaults_cover_generated_autograd_backward_family():
    entries = known_segfaults.load_known_segfaults(Path.cwd())
    active = known_segfaults.active_known_segfaults(
        entries,
        backend="mps",
        torch_version="2.12.1",
        hardware_key="Apple_M3_Max_128gb",
    )
    nodeid = (
        "torchcts/generated/test_autograd_backward_variants.py::"
        "test_generated_autograd_backward_variant[native_batch_norm_backward[L3]]"
    )
    item = _generated_item(
        nodeid,
        _native_batch_norm_backward_entry(),
        fspath="torchcts/generated/test_autograd_backward_variants.py",
    )

    match = known_segfaults.match_known_segfault(
        item,
        active,
        metadata=harness._extract_result_metadata(item),
    )

    assert match is not None
    assert match["id"] == "mps-generated-autograd-backward-teardown-pytorch-2-12"
    assert match["matched_by"] == "coverage_id"
    assert match["evidence_scope"] == "constrained_metadata"
    assert match["constraints"]["surface_kind"] == ["autograd_backward"]


def test_known_segfault_schema_accepts_dispatcher_and_coverage_id_entries():
    payload = {
        "version": 1,
        "known_segfaults": [
            _known_segfault_entry(
                id="mps-dispatcher",
                match="dispatcher",
                nodeid=None,
                dispatcher="aten::dispatcher.default",
                evidence_scope="dispatcher_surface",
            ),
            _known_segfault_entry(
                id="mps-coverage",
                match="coverage_id",
                nodeid=None,
                dispatcher="aten::coverage.default",
                coverage_id="aten::coverage.default",
                evidence_scope="constrained_metadata",
                constraints={"suite": ["generated"]},
            ),
        ],
    }

    entries = known_segfaults.validate_known_segfaults([payload])

    assert [entry["match"] for entry in entries] == ["dispatcher", "coverage_id"]


def test_known_segfault_schema_rejects_nonsensical_metadata_rules():
    bad_dispatcher = _known_segfault_entry(
        match="dispatcher",
        nodeid=None,
        evidence_scope="constrained_metadata",
    )
    bad_scope = _known_segfault_entry(
        match="dispatcher",
        nodeid=None,
        evidence_scope="exact_node",
    )

    with pytest.raises(known_segfaults.KnownSegfaultError, match="non-empty constraints"):
        known_segfaults.validate_known_segfaults([
            {"version": 1, "known_segfaults": [bad_dispatcher]}
        ])
    with pytest.raises(known_segfaults.KnownSegfaultError, match="requires match=nodeid"):
        known_segfaults.validate_known_segfaults([
            {"version": 1, "known_segfaults": [bad_scope]}
        ])


def test_known_segfault_schema_rejects_bad_constraints():
    cases = [
        (_known_segfault_entry(constraints={"nope": ["generated"]}), "unknown"),
        (_known_segfault_entry(constraints={"suite": []}), "must not be empty"),
        (_known_segfault_entry(constraints={"semantic_level": [9]}), "semantic levels"),
        (_known_segfault_entry(constraints={"dtype": []}), "must not be empty"),
        (_known_segfault_entry(constraints={"dtype": [1]}), "non-empty strings"),
    ]

    for entry, pattern in cases:
        with pytest.raises(known_segfaults.KnownSegfaultError, match=pattern):
            known_segfaults.validate_known_segfaults([
                {"version": 1, "known_segfaults": [entry]}
            ])


def test_known_segfault_dispatcher_match_uses_metadata_and_constraints():
    entry = known_segfaults.validate_known_segfaults([
        {
            "version": 1,
            "known_segfaults": [
                _known_segfault_entry(
                    match="dispatcher",
                    nodeid=None,
                    dispatcher="aten::reflection_pad3d.out",
                    evidence_scope="constrained_metadata",
                    constraints={
                        "suite": ["generated"],
                        "variant_kind": ["out_variant"],
                        "strategy_family": ["reflection_pad3d"],
                    },
                )
            ],
        }
    ])[0]
    nodeid = "torchcts/generated/test_out_variants.py::test_generated_out_variant[reflection_pad3d.out[L3]]"
    metadata = harness._extract_result_metadata(_generated_item(nodeid, _reflection_pad3d_out_entry()))
    other_metadata = dict(metadata, strategy_family="reflection_pad2d")

    assert known_segfaults.entry_matches(entry, nodeid, metadata)
    assert not known_segfaults.entry_matches(entry, nodeid, other_metadata)


def test_known_segfault_dtype_constraint_uses_metadata():
    entry = known_segfaults.validate_known_segfaults([
        {
            "version": 1,
            "known_segfaults": [
                _known_segfault_entry(
                    match="dispatcher",
                    nodeid=None,
                    dispatcher="aten::example.default",
                    evidence_scope="constrained_metadata",
                    constraints={
                        "suite": ["generated"],
                        "dtype": ["torch.float64"],
                    },
                )
            ],
        }
    ])[0]
    nodeid = "torchcts/generated/test_foreach_fused.py::test_generated_foreach_or_fused[example-torch.float64]"

    assert known_segfaults.entry_matches(
        entry,
        nodeid,
        {"dispatcher_name": "aten::example.default", "suite": "generated", "dtype": "torch.float64"},
    )
    assert not known_segfaults.entry_matches(
        entry,
        nodeid,
        {"dispatcher_name": "aten::example.default", "suite": "generated", "dtype": "torch.float32"},
    )


def test_known_segfault_nodeid_wins_over_dispatcher():
    exact = _known_segfault_entry(id="exact")
    dispatcher = _known_segfault_entry(
        id="dispatcher",
        match="dispatcher",
        nodeid=None,
        dispatcher="aten::example.default",
        evidence_scope="dispatcher_surface",
    )
    entries = known_segfaults.validate_known_segfaults([
        {"version": 1, "known_segfaults": [dispatcher, exact]}
    ])

    match = known_segfaults.match_known_segfault(
        SimpleNamespace(
            nodeid="torchcts/example.py::test_crash",
            metadata={"dispatcher_name": "aten::example.default"},
        ),
        entries,
    )

    assert match["id"] == "exact"


def test_known_segfault_equal_specificity_ambiguity_fails():
    first = _known_segfault_entry(
        id="first",
        match="dispatcher",
        nodeid=None,
        dispatcher="aten::example.default",
        evidence_scope="dispatcher_surface",
    )
    second = dict(first, id="second")
    entries = known_segfaults.validate_known_segfaults([
        {"version": 1, "known_segfaults": [first, second]}
    ])

    with pytest.raises(known_segfaults.KnownSegfaultError, match="ambiguous"):
        known_segfaults.match_known_segfault(
            SimpleNamespace(
                nodeid="torchcts/example.py::test_case",
                metadata={"dispatcher_name": "aten::example.default"},
            ),
            entries,
        )


def test_known_segfault_schema_rejects_duplicate_ids():
    entry = _known_segfault_entry(id="dup")

    with pytest.raises(known_segfaults.KnownSegfaultError, match="duplicate"):
        known_segfaults.validate_known_segfaults([
            {"version": 1, "known_segfaults": [entry, dict(entry)]}
        ])


def test_harness_known_segfault_fields_preserve_failure_semantics():
    match = {
        "id": "mps-example",
        "match": "dispatcher",
        "matched_by": "dispatcher",
        "dispatcher": "aten::example.default",
        "classification": "confirmed_backend_crash",
        "evidence_scope": "dispatcher_surface",
        "constraints": {"suite": ["generated"]},
        "matched_nodeid": "torchcts/example.py::test_crash",
        "matched_metadata": {"dispatcher_name": "aten::example.default", "suite": "generated"},
        "reason": "standalone repro crashes",
        "expected_signal": "SIGSEGV",
        "repro": {"script": "repro.py", "case": "case0"},
    }

    fields = harness._known_segfault_result_fields(match, actual_signal="SIGABRT")

    assert fields["known_segfault_id"] == "mps-example"
    assert fields["known_segfault_classification"] == "confirmed_backend_crash"
    assert fields["known_segfault_expected_signal"] == "SIGSEGV"
    assert fields["known_segfault_unexpected_signal"] == "SIGABRT"
    assert fields["known_segfault_match"] == "dispatcher"
    assert fields["known_segfault_evidence_scope"] == "dispatcher_surface"
    assert fields["known_segfault_constraints"] == {"suite": ["generated"]}
    assert fields["known_segfault_matched_nodeid"] == "torchcts/example.py::test_crash"
    assert fields["known_segfault_matched_metadata"]["dispatcher_name"] == "aten::example.default"
    assert "status" not in fields


def test_harness_known_segfault_process_classification_is_backend_generic():
    crash = _known_segfault_entry()
    wrong_value = _known_segfault_entry(
        dispatcher="aten::_grid_sampler_2d_cpu_fallback_backward",
    )

    assert harness._known_segfault_process_classification(crash) == "confirmed_backend_crash"
    assert (
        harness._known_segfault_process_classification(
            wrong_value,
            stdout="Tensor-likes are not close! Mismatched elements: 4 / 4",
        )
        == "confirmed_backend_wrong_value"
    )


def test_harness_known_segfault_validation_accepts_matching_dispatcher_rule(monkeypatch):
    entry = known_segfaults.validate_known_segfaults([
        {
            "version": 1,
            "known_segfaults": [
                _known_segfault_entry(
                    match="dispatcher",
                    nodeid=None,
                    dispatcher="aten::reflection_pad3d.out",
                    evidence_scope="constrained_metadata",
                    constraints={"suite": ["generated"], "strategy_family": ["reflection_pad3d"]},
                )
            ],
        }
    ])[0]
    item = _generated_item(
        "torchcts/generated/test_out_variants.py::test_generated_out_variant[reflection_pad3d.out[L3]]",
        _reflection_pad3d_out_entry(),
    )
    config = SimpleNamespace(args=["torchcts/generated"], getoption=lambda name: None)

    descriptors = harness._validate_known_segfault_collection(config, [entry], [item])

    assert descriptors[0]["metadata"]["dispatcher_name"] == "aten::reflection_pad3d.out"


def test_harness_known_segfault_validation_rejects_stale_in_scope_rule():
    entry = known_segfaults.validate_known_segfaults([
        {
            "version": 1,
            "known_segfaults": [
                _known_segfault_entry(
                    match="dispatcher",
                    nodeid=None,
                    dispatcher="aten::missing.out",
                    evidence_scope="constrained_metadata",
                    constraints={"suite": ["generated"]},
                )
            ],
        }
    ])[0]
    item = _generated_item(
        "torchcts/generated/test_out_variants.py::test_generated_out_variant[reflection_pad3d.out[L3]]",
        _reflection_pad3d_out_entry(),
    )
    config = SimpleNamespace(args=["torchcts/generated"], getoption=lambda name: "generated")

    with pytest.raises(pytest.exit.Exception, match="stale in-scope rule"):
        harness._validate_known_segfault_collection(config, [entry], [item])


def test_harness_known_segfault_validation_ignores_dtype_excluded_rule():
    entry = known_segfaults.validate_known_segfaults([
        {
            "version": 1,
            "known_segfaults": [
                _known_segfault_entry(
                    match="coverage_id",
                    nodeid="torchcts/generated/test_foreach_fused.py::test_generated_foreach_or_fused[_foreach_add.List[L4]-torch.float64]",
                    dispatcher="aten::_foreach_*",
                    evidence_scope="constrained_metadata",
                    constraints={
                        "suite": ["generated"],
                        "coverage_kind": ["generated"],
                        "strategy": ["manual_foreach"],
                        "coverage_id_glob": ["aten::_foreach_*"],
                        "dtype": ["torch.float64", "torch.complex128"],
                    },
                )
            ],
        }
    ])[0]
    nodeid = (
        "torchcts/generated/test_foreach_fused.py::"
        "test_generated_foreach_or_fused[_foreach_add.List[L4]-torch.float32]"
    )
    item = _generated_item(
        nodeid,
        _foreach_add_list_functional_entry(),
        fspath="torchcts/generated/test_foreach_fused.py",
        dtype=torch.float32,
    )
    config = SimpleNamespace(args=["torchcts/generated"], getoption=lambda name: "generated")

    descriptors = harness._validate_known_segfault_collection(config, [entry], [item])

    assert descriptors[0]["metadata"]["dtype"] == "torch.float32"


def test_harness_known_segfault_validation_ignores_unrelated_targeted_node():
    entry = known_segfaults.validate_known_segfaults([
        {
            "version": 1,
            "known_segfaults": [
                _known_segfault_entry(
                    match="dispatcher",
                    nodeid=None,
                    dispatcher="aten::reflection_pad3d.out",
                    evidence_scope="constrained_metadata",
                    constraints={"suite": ["generated"], "strategy_family": ["reflection_pad3d"]},
                )
            ],
        }
    ])[0]
    config = SimpleNamespace(
        args=["torchcts/selftest/test_mps_triage.py::test_mps_triage_timeout_records_inconclusive"],
        getoption=lambda name: None,
    )
    item = SimpleNamespace(
        nodeid="torchcts/selftest/test_mps_triage.py::test_mps_triage_timeout_records_inconclusive",
        fspath="torchcts/selftest/test_mps_triage.py",
        name="test_mps_triage_timeout_records_inconclusive",
        iter_markers=lambda name=None: iter(()),
    )

    descriptors = harness._validate_known_segfault_collection(config, [entry], [item])

    assert descriptors[0]["metadata"]["suite"] == "selftest"


def test_harness_known_segfault_validation_ignores_unrelated_targeted_generated_file():
    entry = known_segfaults.validate_known_segfaults([
        {
            "version": 1,
            "known_segfaults": [
                _known_segfault_entry(
                    match="dispatcher",
                    nodeid="torchcts/generated/test_factories.py::test_generated_factory[hamming_window.periodic[L2]]",
                    dispatcher="aten::hamming_window.periodic",
                    evidence_scope="constrained_metadata",
                    constraints={
                        "suite": ["generated"],
                        "coverage_kind": ["generated"],
                        "surface_kind": ["factory"],
                        "variant_kind": ["factory"],
                        "strategy": ["manual_factory"],
                        "strategy_family": ["window"],
                        "semantic_level": [2],
                    },
                )
            ],
        }
    ])[0]
    item = _generated_item(
        "torchcts/generated/test_functional_variants.py::test_generated_functional_variant[_add_relu.Scalar[L1]]",
        {
            "name": "aten::_add_relu.Scalar",
            "status": "covered_generated",
            "coverage_kind": "generated",
            "surface_kind": "functional_data",
            "variant_kind": "functional",
            "semantic_level": 1,
            "generated": {
                "strategy": {
                    "strategy": "manual_elementwise",
                    "family": "_add_relu",
                }
            },
        },
        fspath="torchcts/generated/test_functional_variants.py",
    )
    config = SimpleNamespace(
        args=["torchcts/generated/test_functional_variants.py"],
        getoption=lambda name: "generated",
    )

    descriptors = harness._validate_known_segfault_collection(config, [entry], [item])

    assert descriptors[0]["metadata"]["dispatcher_name"] == "aten::_add_relu.Scalar"


def test_harness_known_segfault_audit_prints_rule_counts(capsys):
    entry = _known_segfault_entry(
        match="dispatcher",
        nodeid=None,
        dispatcher="aten::reflection_pad3d.out",
        evidence_scope="constrained_metadata",
        constraints={"suite": ["generated"], "strategy_family": ["reflection_pad3d"]},
    )
    item = _generated_item(
        "torchcts/generated/test_out_variants.py::test_generated_out_variant[reflection_pad3d.out[L3]]",
        _reflection_pad3d_out_entry(),
    )
    descriptor = harness._known_segfault_descriptor_for_item(item)
    config = SimpleNamespace(option=SimpleNamespace(verbose=1))

    harness._print_known_segfault_audit(config, [entry], [descriptor])

    out = capsys.readouterr().out
    assert "Known segfault audit: 1 active rule(s)" in out
    assert "matched=1" in out
    assert "reflection_pad3d.out" in out


def test_harness_adaptive_match_canonicalizes_nodeids(monkeypatch):
    candidate = {
        "nodeid": "torchcts/generated/test_out_variants.py::test_generated_out_variant[range.out_[L2]]",
        "canonical_nodeid": "torchcts/generated/test_out_variants.py::test_generated_out_variant[range.out_[L2]]",
        "isolation_source": "adaptive_previous_crash",
        "reason": "previous run recorded a process crash",
        "evidence_path": "results/hw_latest.json",
        "prior_status": "ERROR",
        "prior_signal": "SIGSEGV",
    }
    monkeypatch.setattr(
        harness,
        "_ADAPTIVE_ISOLATION_ACTIVE",
        {candidate["canonical_nodeid"]: candidate},
    )
    item = SimpleNamespace(
        nodeid=(
            ".venv/lib/python3.14/site-packages/torchcts/generated/"
            "test_out_variants.py::test_generated_out_variant[range.out_[L2]]"
        )
    )

    assert harness._adaptive_isolation_match_for_item(item) is candidate


def test_harness_adaptive_fields_preserve_known_segfault_precedence():
    adaptive = {
        "isolation_source": "adaptive_previous_crash",
        "reason": "previous run recorded a process crash (SIGSEGV)",
        "evidence_path": "results/hw_latest.json",
        "prior_status": "ERROR",
        "prior_signal": "SIGSEGV",
        "prior_error_type": "ProcessCrash",
        "prior_timestamp": "2026-06-28T00:00:00Z",
    }

    fields = harness._adaptive_isolation_result_fields(
        adaptive,
        known_segfault_match={"id": "known"},
        resolved=True,
    )

    assert fields["isolation_source"] == "known_segfault"
    assert fields["adaptive_isolation_source"] == "adaptive_previous_crash"
    assert fields["adaptive_isolation_prior_signal"] == "SIGSEGV"
    assert fields["adaptive_isolation_resolved"] is True
    assert "status" not in fields


def test_harness_child_command_disables_adaptive_isolation(monkeypatch, tmp_path):
    monkeypatch.setattr(harness, "_DEVICE_NAME", "mps")
    monkeypatch.setattr(harness, "_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(harness, "_REQUESTED_SEMANTIC_LEVEL", 8)

    options = {
        "--level-exact": None,
        "--level-range": None,
        "--dtype": [],
        "--memory-mode": "balanced",
        "--max-device-memory": None,
        "--max-tensor-size": None,
        "--validation": True,
    }
    config = SimpleNamespace(getoption=lambda name: options.get(name))
    item = SimpleNamespace(nodeid="torchcts/example.py::test_crash", config=config)

    cmd = harness._subprocess_child_command(item)

    assert cmd[cmd.index("--adaptive-isolation") + 1] == "off"
    assert cmd[cmd.index("--known-segfault-policy") + 1] == "off"


def test_harness_finalizes_adaptive_candidates_for_collected_items(monkeypatch, tmp_path, capsys):
    loaded = adaptive_isolation.AdaptiveIsolationLoadResult(
        candidates={
            "torchcts/a.py::test_keep": adaptive_isolation.AdaptiveIsolationCandidate(
                nodeid="torchcts/a.py::test_keep",
                canonical_nodeid="torchcts/a.py::test_keep",
                isolation_source="adaptive_previous_timeout",
                reason="previous subprocess isolation timed out",
                evidence_path="results/hw_latest.json",
                prior_status="ERROR",
                prior_error_type="TimeoutError",
            ),
            "torchcts/b.py::test_drop": adaptive_isolation.AdaptiveIsolationCandidate(
                nodeid="torchcts/b.py::test_drop",
                canonical_nodeid="torchcts/b.py::test_drop",
                isolation_source="adaptive_previous_crash",
                reason="previous run recorded a process crash",
                evidence_path="results/hw_latest.json",
                prior_status="ERROR",
            ),
        },
        rejected=[],
        warnings=[],
        artifacts_considered=["results/hw_latest.json"],
    )
    monkeypatch.setattr(harness, "_ADAPTIVE_ISOLATION_MODE", "auto")
    monkeypatch.setattr(harness, "_ADAPTIVE_ISOLATION_LOAD", loaded)
    monkeypatch.setattr(harness, "_ADAPTIVE_ISOLATION_WARNINGS", [])
    monkeypatch.setattr(harness, "_COLLECT_ONLY", False)
    monkeypatch.setattr(harness, "_SHOW_SKIPS", False)
    monkeypatch.setattr(harness, "_ARTIFACT_WRITES_ENABLED", True)
    monkeypatch.setattr(harness, "_IS_XDIST_WORKER", False)
    monkeypatch.setattr(harness, "_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(harness, "_HARDWARE_KEY", "hw")
    monkeypatch.setattr(harness, "_DEVICE_NAME", "mps")
    monkeypatch.setattr(harness, "_is_child_process", lambda: False)

    config = SimpleNamespace(option=SimpleNamespace(verbose=1))
    item = SimpleNamespace(nodeid="torchcts/a.py::test_keep")

    harness._finalize_adaptive_isolation_for_collection(config, [item])

    assert sorted(harness._ADAPTIVE_ISOLATION_ACTIVE) == ["torchcts/a.py::test_keep"]
    out = capsys.readouterr().out
    assert "Adaptive isolation: 1 node(s)" in out
    artifact = json.loads((tmp_path / "hw_adaptive_isolation.json").read_text(encoding="utf-8"))
    assert sorted(artifact["accepted"]) == ["torchcts/a.py::test_keep"]
    assert artifact["rejected"][0]["reason"] == "not_collected_in_current_run"


@_SOURCE_CHECKOUT_ONLY
def test_adaptive_isolation_synthetic_replay_runs_candidate_in_subprocess(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    target_file = Path(__file__).resolve()
    target_root = target_file.parents[2]
    target_path = target_file.relative_to(target_root).as_posix()
    target = f"{target_path}::test_mps_triage_timeout_records_inconclusive"
    hardware_key = get_hardware_key("cpu", harness.load_manifest())
    latest_path = results_dir / f"{hardware_key}_latest.json"
    latest_path.write_text(
        json.dumps(
            _adaptive_result_payload(
                {
                    target: {
                        "status": "ERROR",
                        "error_type": "ProcessCrash",
                        "subprocess": {"signal": "SIGSEGV"},
                    }
                },
                device="cpu",
                hardware=hardware_key,
                version=torch.__version__,
                completed=False,
            )
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            target,
            "--validation",
            "--results-dir",
            str(results_dir),
            "--tb=short",
        ],
        cwd=target_root,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    record = payload["results"][target]
    command_args = (record.get("subprocess") or {}).get("command_args")
    adaptive_artifact = json.loads(
        (results_dir / f"{hardware_key}_adaptive_isolation.json").read_text(encoding="utf-8")
    )

    assert record["status"] == "PASS"
    assert record["phase"] == "subprocess_child"
    assert record["adaptive_isolation_source"] == "adaptive_previous_crash"
    assert record["adaptive_isolation_resolved"] is True
    if command_args is not None:
        assert command_args[command_args.index("--adaptive-isolation") + 1] == "off"
    assert sorted(adaptive_artifact["accepted"]) == [target]


def test_harness_subprocess_error_record_preserves_crash_evidence():
    item = SimpleNamespace(
        nodeid="torchcts/example.py::test_crash",
        fspath="torchcts/example.py",
        name="test_crash",
        iter_markers=lambda name=None: iter(()),
    )

    record = harness._subprocess_error_record(
        item,
        "ERROR",
        "ProcessCrash",
        "crashed",
        1234.5,
        ["python", "-m", "pytest", item.nodeid],
        returncode=-11,
        stdout="stdout before crash",
        stderr=b"Fatal Python error: Segmentation fault",
    )

    assert record["nodeid"] == item.nodeid
    assert record["phase"] == "subprocess"
    assert record["failure_stage"] == "process"
    assert record["subprocess"]["command_args"] == ["python", "-m", "pytest", item.nodeid]
    assert record["subprocess"]["signal"] == "SIGSEGV"
    assert record["subprocess"]["duration_seconds"] == pytest.approx(1.2345)
    assert "Segmentation fault" in record["subprocess"]["stderr_tail"]


def test_mps_triage_repro_writer_is_stable(tmp_path):
    scripts = triage.write_repro_scripts(tmp_path)

    assert sorted(scripts) == ["factory_out", "grid_segment_dense_split"]
    for path in scripts.values():
        text = Path(path).read_text(encoding="utf-8")
        assert "import torch" in text
        assert "mps" in text


def test_mps_triage_known_repros_mark_passing_cases_as_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(
        triage,
        "write_repro_scripts",
        lambda triage_dir: {
            "factory_out": "factory_out.py",
            "grid_segment_dense_split": "grid.py",
        },
    )
    monkeypatch.setattr(
        triage,
        "run_subprocess_command",
        lambda *args, **kwargs: {
            "returncode": 0,
            "signal": None,
            "timed_out": False,
            "stdout_tail": "ok",
            "stderr_tail": "",
        },
    )

    runs = triage.run_known_repros(triage_dir=tmp_path, timeout=1)

    assert runs
    assert {run["status"] for run in runs} == {"PASS"}
    assert {run["classification"]["classification"] for run in runs} == {"expected_unsupported"}


def test_mps_triage_repros_only_skips_queue_and_support_probe(tmp_path, monkeypatch):
    def fail_probe(**kwargs):
        raise AssertionError("support probe should not run in repros-only mode")

    monkeypatch.setattr(triage, "run_mps_support_probe", fail_probe)
    monkeypatch.setattr(
        triage,
        "run_known_repros",
        lambda **kwargs: [
            {
                "repro": "grid_segment_dense_split",
                "case": "grid2_cpu_fallback_default",
                "status": "ERROR",
                "returncode": -11,
                "signal": "SIGSEGV",
                "timed_out": False,
                "classification": {"classification": "confirmed_mps_crash"},
            }
        ],
    )

    payload = triage.run_mps_triage(triage_dir=tmp_path, repros_only=True)

    assert payload["input_result"] is None
    assert payload["input_snapshot"] == []
    assert payload["queue"] == []
    assert payload["subprocess_runs"] == []
    assert payload["support_probe"] is None
    assert payload["repro_counts"]["confirmed_mps_crash"] == 1
    assert (tmp_path / "adjudication_queue.json").exists()
    assert "Standalone Repros" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_mps_triage_summary_and_classification_artifacts(tmp_path, monkeypatch):
    result_file = tmp_path / "Apple_M3_Max_128gb_latest.json"
    result_file.write_text(
        json.dumps(
            {
                "metadata": {"device_name": "mps", "hardware_key": "Apple_M3_Max_128gb"},
                "results": {
                    "torchcts/dtypes/test_fp8.py::test_fp8": {
                        "status": "ERROR",
                        "error_type": "TypeError",
                        "error_message": "Trying to convert Float8_e4m3fn to the MPS backend",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(triage, "run_mps_support_probe", lambda **kwargs: {"probes": {}})
    monkeypatch.setattr(triage, "run_known_repros", lambda **kwargs: [])

    payload = triage.run_mps_triage(
        from_file=result_file,
        triage_dir=tmp_path / "triage",
        run_nodes=False,
    )

    classification = payload["classifications"]["torchcts/dtypes/test_fp8.py::test_fp8"]["classification"]
    assert classification == "manifest_overclaim"
    assert (tmp_path / "triage" / "summary.md").exists()
    assert (tmp_path / "triage" / "classifications.json").exists()
    assert (tmp_path / "triage" / "input_snapshot" / result_file.name).exists()


def test_mps_triage_snapshot_accepts_existing_snapshot_file(tmp_path):
    triage_dir = tmp_path / "results" / "mps_triage"
    snapshot_dir = triage_dir / "input_snapshot"
    snapshot_dir.mkdir(parents=True)
    result_file = snapshot_dir / "Apple_M3_Max_128gb_latest.json"
    result_file.write_text(
        json.dumps({"metadata": {"hardware_key": "Apple_M3_Max_128gb"}, "results": {}}),
        encoding="utf-8",
    )

    copied = triage.freeze_input_snapshot(
        result_file,
        {"metadata": {"hardware_key": "Apple_M3_Max_128gb"}, "results": {}},
        triage_dir,
    )

    assert copied == [str(result_file)]


def test_stream_context_uses_stream_object_context_manager():
    class DummyStream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    stream = DummyStream()

    assert stream_context(stream) is stream


def test_stream_context_raises_clear_error_without_context():
    with pytest.raises(RuntimeError, match="stream context API unavailable"):
        stream_context(object())


def test_empty_factory_metadata_does_not_compare_uninitialized_values():
    entry = {"name": "aten::empty_permuted", "base_name": "empty_permuted"}
    expected = torch.empty_strided((2, 3), (1, 2), dtype=torch.float32)
    actual = torch.empty_strided((2, 3), (1, 2), dtype=torch.float32)
    expected.fill_(1.0)
    actual.fill_(2.0)

    coverage_helpers._assert_factory_metadata(entry, actual, expected, "cpu", torch.float32)


def test_empty_factory_metadata_catches_stride_mismatch():
    entry = {"name": "aten::empty_permuted", "base_name": "empty_permuted"}
    expected = torch.empty_strided((2, 3), (1, 2), dtype=torch.float32)
    actual = torch.empty_strided((2, 3), (3, 1), dtype=torch.float32)

    with pytest.raises(AssertionError, match="stride mismatch"):
        coverage_helpers._assert_factory_metadata(entry, actual, expected, "cpu", torch.float32)


def test_harness_metadata_includes_generated_strategy_fields(monkeypatch):
    entry = {
        "name": "aten::_native_batch_norm_legit.no_stats",
        "schema": "aten::_native_batch_norm_legit.no_stats(Tensor input) -> Tensor",
        "status": "covered_generated",
        "surface_kind": "functional_data",
        "variant_kind": "functional",
        "semantic_level": 3,
        "generated": {"strategy": {"strategy": "manual_multi_output_reduction", "family": "_native_batch_norm_legit"}},
    }

    item = SimpleNamespace(
        fspath="torchcts/generated/test_functional_variants.py",
        name="test_generated_functional_variant",
        callspec=SimpleNamespace(params={"entry": entry}),
        iter_markers=lambda name=None: iter(()),
    )
    monkeypatch.setattr(harness, "_REQUESTED_SEMANTIC_LEVEL", 8)

    metadata = harness._extract_result_metadata(item)

    assert metadata["dispatcher_name"] == "aten::_native_batch_norm_legit.no_stats"
    assert metadata["schema"] == entry["schema"]
    assert metadata["strategy"] == "manual_multi_output_reduction"
    assert metadata["strategy_family"] == "_native_batch_norm_legit"
    assert metadata["coverage_kind"] == "generated"
