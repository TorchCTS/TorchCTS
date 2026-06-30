# TorchCTS Site Stats

This generated file is a statistics source for website copy and AI-assisted site updates.
It describes the current checkout and installed PyTorch build; it is not a backend pass/fail report.

| Field | Value |
| --- | --- |
| Generated at | 2026-06-30T04:15:51.876265Z |
| TorchCTS version | 0.3.0 |
| PyTorch version | 2.12.1 |
| Python version | 3.14.2 |
| Platform | macOS-26.3-arm64-arm-64bit-Mach-O |
| Coverage audit timestamp | 2026-06-30T04:15:30.533835Z |
| Pytest collection included | yes |

## Headline Stats

| Metric | Value |
| --- | --- |
| Pytest nodes collected | 18867 |
| ATen overloads inventoried | 3225 |
| Backend-relevant overloads | 3214 |
| Covered backend-relevant overloads | 2938 |
| Dispatcher coverage | 91.4% |
| Unknown tensor-touching surfaces | 0 |
| Pending surfaces | 220 |
| Excluded surfaces | 56 |
| Generated coverage surfaces | 1910 |
| Generated semantic cases | 1921 |
| Required generated semantic cases | 1921 |
| Known crash isolation rules | 10 |
| CPU dtype contract records | 3053 |

## Pytest Collection Summary

| Metric | Value |
| --- | --- |
| Collection command | `python -m pytest --collect-only -q torchcts --validation` |
| Node IDs parsed | 18867 |
| Pytest summary count | 18867 |
| Parameterized node IDs | 18433 |
| Unparameterized node IDs | 434 |

## Pytest Nodes By Suite

| Name | Count |
| --- | --- |
| opinfo | 13098 |
| generated | 4258 |
| operators | 434 |
| selftest | 353 |
| compiler | 144 |
| dtypes | 135 |
| workloads | 120 |
| strides | 87 |
| autograd | 77 |
| training | 56 |
| stress | 30 |
| rng | 28 |
| memory | 24 |
| device_api | 12 |
| errors | 5 |
| multi_device | 3 |
| serialization | 3 |

## Pytest Nodes By Test Kind

| Name | Count |
| --- | --- |
| opinfo | 13098 |
| generated | 4258 |
| handwritten | 1158 |
| selftest | 353 |

## Pytest Nodes By File

| Name | Count |
| --- | --- |
| torchcts/opinfo/test_opinfo_forward.py | 10577 |
| torchcts/opinfo/test_opinfo_backward.py | 2319 |
| torchcts/generated/test_foreach_fused.py | 1969 |
| torchcts/generated/test_out_variants.py | 898 |
| torchcts/generated/test_functional_variants.py | 578 |
| torchcts/generated/test_inplace_variants.py | 346 |
| torchcts/generated/test_oracle_surfaces.py | 273 |
| torchcts/selftest/test_harness_reporting.py | 213 |
| torchcts/opinfo/test_opinfo_errors.py | 202 |
| torchcts/selftest/test_mps_triage.py | 109 |
| torchcts/generated/test_view_aliases.py | 100 |
| torchcts/strides/test_noncontiguous.py | 73 |
| torchcts/workloads/test_sdpa.py | 73 |
| torchcts/compiler/test_compile_ops.py | 72 |
| torchcts/operators/test_unary.py | 66 |
| torchcts/dtypes/test_quantized.py | 50 |
| torchcts/dtypes/test_copy_cast.py | 49 |
| torchcts/compiler/test_compile_dynamic.py | 48 |
| torchcts/autograd/test_backward.py | 41 |
| torchcts/generated/test_autograd_backward_variants.py | 38 |
| torchcts/operators/test_matmul.py | 37 |
| torchcts/operators/test_binary.py | 36 |
| torchcts/operators/test_sparse.py | 34 |
| torchcts/operators/test_view_shape.py | 33 |
| torchcts/rng/test_generator.py | 25 |
| torchcts/compiler/test_compile_training.py | 24 |
| torchcts/generated/test_layout_storage_variants.py | 24 |
| torchcts/operators/test_comparison.py | 22 |
| torchcts/operators/test_activation.py | 21 |
| torchcts/operators/test_loss.py | 21 |
| torchcts/workloads/test_model_shapes.py | 21 |
| torchcts/generated/test_factories.py | 20 |
| torchcts/dtypes/test_native_quantization.py | 19 |
| torchcts/operators/test_creation.py | 19 |
| torchcts/operators/test_misc.py | 19 |
| torchcts/operators/test_norm.py | 19 |
| torchcts/autograd/test_inplace_safety.py | 18 |
| torchcts/operators/test_reduction.py | 18 |
| torchcts/training/test_training_pipeline.py | 18 |
| torchcts/operators/test_conv.py | 17 |
| torchcts/operators/test_nested.py | 15 |
| torchcts/operators/test_index_scatter.py | 14 |
| torchcts/selftest/test_install_plan.py | 14 |
| torchcts/stress/test_adversarial.py | 14 |
| torchcts/operators/test_pooling.py | 13 |
| torchcts/autograd/test_double_backward.py | 12 |
| torchcts/generated/test_rng_variants.py | 12 |
| torchcts/memory/test_allocator.py | 12 |
| torchcts/dtypes/test_complex.py | 9 |
| torchcts/selftest/test_diagnose.py | 9 |
| torchcts/stress/test_large_tensors.py | 9 |
| torchcts/training/test_mixed_precision.py | 8 |
| torchcts/dtypes/test_fp8.py | 7 |
| torchcts/operators/test_linalg.py | 7 |
| torchcts/device_api/test_device_module.py | 6 |
| torchcts/device_api/test_streams_events.py | 6 |
| torchcts/memory/test_determinism.py | 6 |
| torchcts/memory/test_guard_alloc.py | 6 |
| torchcts/selftest/test_docs_public.py | 6 |
| torchcts/strides/test_advanced_indexing.py | 6 |
| torchcts/strides/test_strided_inplace.py | 6 |
| torchcts/training/test_dataloader.py | 6 |
| torchcts/training/test_grad_clipping.py | 6 |
| torchcts/training/test_lr_scheduler.py | 6 |
| torchcts/training/test_module_hooks.py | 6 |
| torchcts/workloads/test_e2e_models.py | 6 |
| torchcts/workloads/test_lora.py | 6 |
| torchcts/workloads/test_vision.py | 6 |
| torchcts/errors/test_error_handling.py | 5 |
| torchcts/operators/test_fft.py | 5 |
| torchcts/operators/test_padding.py | 5 |
| torchcts/stress/test_edge_numerics.py | 5 |
| torchcts/workloads/test_transformer.py | 5 |
| torchcts/autograd/test_gradcheck.py | 4 |
| torchcts/operators/test_named_tensors.py | 4 |
| torchcts/operators/test_upsample.py | 4 |
| torchcts/training/test_fused_optimizer_raw.py | 4 |
| torchcts/multi_device/test_multi_device.py | 3 |
| torchcts/operators/test_foreach.py | 3 |
| torchcts/rng/test_dropout_rrelu_variants.py | 3 |
| torchcts/serialization/test_save_load.py | 3 |
| torchcts/workloads/test_attention_dispatcher.py | 3 |
| torchcts/autograd/test_tensor_metadata_mutation.py | 2 |
| torchcts/operators/test_fractional_pooling_backward.py | 2 |
| torchcts/selftest/test_site_stats.py | 2 |
| torchcts/stress/test_rapid_alloc.py | 2 |
| torchcts/dtypes/test_metadata_ops.py | 1 |
| torchcts/strides/test_channels_last.py | 1 |
| torchcts/strides/test_memory_format.py | 1 |
| torchcts/training/test_checkpoint.py | 1 |
| torchcts/training/test_grad_accumulation.py | 1 |

## Top Pytest Test Functions

| Name | Count |
| --- | --- |
| test_op_forward | 10577 |
| test_op_backward | 2319 |
| test_generated_foreach_or_fused | 1969 |
| test_generated_out_variant | 898 |
| test_generated_functional_variant | 578 |
| test_generated_inplace_variant | 346 |
| test_oracle_surface | 273 |
| test_op_errors | 202 |
| test_generated_view_alias | 100 |
| test_unary_float_op | 54 |
| test_copy_cast_grid | 49 |
| test_compile_dynamic_shapes | 45 |
| test_mps_triage_classifier | 44 |
| test_generated_autograd_backward_variant | 38 |
| test_compile_unary_op | 33 |
| test_quantized_plumbing | 25 |
| test_custom_quantized_decoder | 25 |
| test_generated_layout_storage_variant | 24 |
| test_binary_float_op | 24 |
| test_generated_factory | 20 |
| test_binary_op_noncontiguous | 18 |
| test_reduction_noncontiguous | 18 |
| test_sdpa_gqa_enable_gqa | 16 |
| test_first_order_backward | 15 |
| test_activations_basic | 15 |
| test_unary_op_noncontiguous | 15 |
| test_comparison_op | 14 |
| test_double_backward_ops | 12 |
| test_inplace_safety_checks | 12 |
| test_compile_binary_op | 12 |
| test_compile_reduction_op | 12 |
| test_compile_training_step | 12 |
| test_generated_rng_variant | 12 |
| test_binary_int_op | 12 |
| test_mm_layouts | 12 |
| test_bmm_layouts | 12 |
| test_unary_int_op | 12 |
| test_gemv_m1_shapes | 12 |
| test_internal_dispatcher_surface | 9 |
| test_compile_training_optimizer | 9 |
| test_optimizer_pipelines | 9 |
| test_fused_optimizer_pipelines | 9 |
| test_complex_operations | 8 |
| test_sort_topk_kthvalue_median | 8 |
| test_sdpa_gqa_manual_expand | 8 |
| test_sdpa_mqa_decode | 8 |
| test_mps_triage_classifier_identifies_reproduced_generated_reduction_ieee_mismatch | 7 |
| test_inplace_saved_for_backward_error | 6 |
| test_allocator_tracking_and_cache | 6 |
| test_oom_recovery | 6 |
| test_determinism_stale_buffers | 6 |
| test_guard_alloc_canary | 6 |
| test_softmax_log_softmax | 6 |
| test_smooth_l1_huber_loss | 6 |
| test_dot_mv | 6 |
| test_cumsum_cumprod_unique | 6 |
| test_zero_element_and_scalar_tensors | 6 |
| test_advanced_indexing_mixed | 6 |
| test_ops_on_permuted_4d | 6 |
| test_strided_inplace_updates | 6 |
| test_dataloader_pin_memory | 6 |
| test_gradient_clipping | 6 |
| test_lr_schedulers | 6 |
| test_lora_forward_backward | 6 |
| test_sdpa_asymmetric_lengths | 6 |
| test_sdpa_nested_forward | 6 |
| test_sdpa_gqa_model_configs | 6 |
| test_vision_components | 6 |
| test_amax_amin | 5 |
| test_argmax_argmin | 5 |
| test_shipped_manifest_templates_validate | 5 |
| test_shipped_manifest_templates_list_every_known_capability | 5 |
| test_compare_tensors_sparse_layouts_pass | 5 |
| test_compare_tensors_sparse_structure_mismatch_records_failure | 5 |
| test_sdpa_non_power_of_2 | 5 |

## Top Suite And Function Pairs

| Name | Count |
| --- | --- |
| opinfo::test_op_forward | 10577 |
| opinfo::test_op_backward | 2319 |
| generated::test_generated_foreach_or_fused | 1969 |
| generated::test_generated_out_variant | 898 |
| generated::test_generated_functional_variant | 578 |
| generated::test_generated_inplace_variant | 346 |
| generated::test_oracle_surface | 273 |
| opinfo::test_op_errors | 202 |
| generated::test_generated_view_alias | 100 |
| operators::test_unary_float_op | 54 |
| dtypes::test_copy_cast_grid | 49 |
| compiler::test_compile_dynamic_shapes | 45 |
| selftest::test_mps_triage_classifier | 44 |
| generated::test_generated_autograd_backward_variant | 38 |
| compiler::test_compile_unary_op | 33 |
| dtypes::test_quantized_plumbing | 25 |
| dtypes::test_custom_quantized_decoder | 25 |
| generated::test_generated_layout_storage_variant | 24 |
| operators::test_binary_float_op | 24 |
| generated::test_generated_factory | 20 |
| strides::test_binary_op_noncontiguous | 18 |
| strides::test_reduction_noncontiguous | 18 |
| workloads::test_sdpa_gqa_enable_gqa | 16 |
| autograd::test_first_order_backward | 15 |
| operators::test_activations_basic | 15 |
| strides::test_unary_op_noncontiguous | 15 |
| operators::test_comparison_op | 14 |
| autograd::test_double_backward_ops | 12 |
| autograd::test_inplace_safety_checks | 12 |
| compiler::test_compile_binary_op | 12 |
| compiler::test_compile_reduction_op | 12 |
| compiler::test_compile_training_step | 12 |
| generated::test_generated_rng_variant | 12 |
| operators::test_binary_int_op | 12 |
| operators::test_mm_layouts | 12 |
| operators::test_bmm_layouts | 12 |
| operators::test_unary_int_op | 12 |
| workloads::test_gemv_m1_shapes | 12 |
| autograd::test_internal_dispatcher_surface | 9 |
| compiler::test_compile_training_optimizer | 9 |
| training::test_optimizer_pipelines | 9 |
| training::test_fused_optimizer_pipelines | 9 |
| dtypes::test_complex_operations | 8 |
| operators::test_sort_topk_kthvalue_median | 8 |
| workloads::test_sdpa_gqa_manual_expand | 8 |
| workloads::test_sdpa_mqa_decode | 8 |
| selftest::test_mps_triage_classifier_identifies_reproduced_generated_reduction_ieee_mismatch | 7 |
| autograd::test_inplace_saved_for_backward_error | 6 |
| memory::test_allocator_tracking_and_cache | 6 |
| memory::test_oom_recovery | 6 |
| memory::test_determinism_stale_buffers | 6 |
| memory::test_guard_alloc_canary | 6 |
| operators::test_softmax_log_softmax | 6 |
| operators::test_smooth_l1_huber_loss | 6 |
| operators::test_dot_mv | 6 |
| operators::test_cumsum_cumprod_unique | 6 |
| stress::test_zero_element_and_scalar_tensors | 6 |
| strides::test_advanced_indexing_mixed | 6 |
| strides::test_ops_on_permuted_4d | 6 |
| strides::test_strided_inplace_updates | 6 |
| training::test_dataloader_pin_memory | 6 |
| training::test_gradient_clipping | 6 |
| training::test_lr_schedulers | 6 |
| workloads::test_lora_forward_backward | 6 |
| workloads::test_sdpa_asymmetric_lengths | 6 |
| workloads::test_sdpa_nested_forward | 6 |
| workloads::test_sdpa_gqa_model_configs | 6 |
| workloads::test_vision_components | 6 |
| operators::test_amax_amin | 5 |
| operators::test_argmax_argmin | 5 |
| selftest::test_shipped_manifest_templates_validate | 5 |
| selftest::test_shipped_manifest_templates_list_every_known_capability | 5 |
| selftest::test_compare_tensors_sparse_layouts_pass | 5 |
| selftest::test_compare_tensors_sparse_structure_mismatch_records_failure | 5 |
| workloads::test_sdpa_non_power_of_2 | 5 |

## Visible Dtype Tokens In Node IDs

| Name | Count |
| --- | --- |
| torch.float32 | 2385 |
| torch.float64 | 2375 |
| torch.float16 | 2007 |
| torch.bfloat16 | 2003 |
| torch.complex128 | 1443 |
| torch.complex64 | 1443 |
| torch.int64 | 571 |
| torch.int32 | 560 |
| torch.uint8 | 560 |
| torch.int16 | 558 |
| torch.int8 | 558 |
| torch.bool | 403 |
| torch.ops | 22 |

## Visible Generated Level Tokens In Node IDs

| Name | Count |
| --- | --- |
| L4 | 2214 |
| L3 | 971 |
| L1 | 422 |
| L5 | 391 |
| L2 | 247 |
| L6 | 19 |

## Dispatcher Coverage Summary

| Metric | Value |
| --- | --- |
| ATen overloads | 3225 |
| Backend-relevant overloads | 3214 |
| Covered backend-relevant overloads | 2938 |
| Coverage percent | 91.4% |
| Pending surfaces | 220 |
| Excluded surfaces | 56 |
| Unknown surfaces | 0 |

## Coverage Status Counts

| Name | Count |
| --- | --- |
| covered_generated | 1910 |
| covered_handwritten | 710 |
| covered_opinfo | 265 |
| pending_backend_pack | 143 |
| pending_property | 77 |
| excluded_framework_plumbing | 32 |
| covered_backend_pack | 20 |
| covered_property | 20 |
| excluded_unsupported_public_api | 16 |
| covered_oracle | 13 |
| not_backend_relevant | 11 |
| excluded_deprecated_or_removed | 4 |
| excluded_distributed_scope | 2 |
| excluded_host_storage | 2 |

## Coverage Kind Counts

| Name | Count |
| --- | --- |
| generated | 1910 |
| handwritten | 710 |
| opinfo | 265 |
| backend_pack | 163 |
| property | 97 |
| excluded | 56 |
| oracle | 13 |

## Surface Kind Counts

| Name | Count |
| --- | --- |
| functional_data | 1190 |
| out_variant | 1056 |
| mutating_or_inplace | 454 |
| autograd_backward | 145 |
| layout_storage | 139 |
| view_or_alias | 137 |
| rng | 51 |
| factory | 39 |
| not_backend_relevant | 11 |
| metadata_device | 3 |

## Variant Kind Counts

| Name | Count |
| --- | --- |
| functional | 1536 |
| out | 1056 |
| inplace | 454 |
| view | 104 |
| factory | 61 |
| metadata | 14 |

## Tensor Input And Return Shape Counts

| Name | Count |
| --- | --- |
| tensor_args_and_returns | 2896 |
| tensor_args_only | 254 |
| tensor_returns_only | 61 |
| no_tensor_io | 14 |

## Dispatch Key Availability Counts

| Name | Count |
| --- | --- |
| Meta | 1434 |
| CompositeExplicitAutograd | 1076 |
| CPU | 1075 |
| MPS | 885 |
| CompositeImplicitAutograd | 844 |

## Coverage Source Combination Counts

| Name | Count |
| --- | --- |
| generated | 1608 |
| handwritten | 555 |
| opinfo+generated | 302 |
| opinfo | 265 |
| exclusion+pending_review | 247 |
| opinfo+handwritten+generated | 66 |
| oracle+exclusion | 52 |
| handwritten+generated | 46 |
| opinfo+handwritten | 43 |
| oracle+exclusion+pending_review | 29 |
| none | 9 |
| exclusion | 2 |
| generated+oracle+exclusion | 1 |

## Semantic Level Counts

| Name | Count |
| --- | --- |
| 1 | 422 |
| 2 | 952 |
| 3 | 1051 |
| 4 | 384 |
| 5 | 380 |
| 6 | 19 |
| 7 | 6 |

## Semantic Level Descriptions

| Level | Description |
| --- | --- |
| 1 | Core primitive behavior that every backend should run continuously. |
| 2 | Normal correctness coverage for common tensor-producing and tensor-consuming surfaces. |
| 3 | Mainstream framework semantics such as mutation, aliasing, RNG, metadata, and generated variants. |
| 4 | Broad production behavior including training/autograd-adjacent and family-specialized cases. |
| 5 | Advanced numeric, layout, storage, sparse, nested, and stride-sensitive behavior. |
| 6 | Specialized backend integration such as compiler, device API, allocator, quantization-adjacent, and low-level implementation surfaces. |
| 7 | Heavy integration and workload coverage that validates realistic model or multi-device behavior. |
| 8 | Release-depth stress and adversarial coverage intended for exhaustive validation passes. |

## Semantic Level By Status

### Level 1

| Status | Count |
| --- | --- |
| covered_generated | 422 |

### Level 2

| Status | Count |
| --- | --- |
| covered_backend_pack | 6 |
| covered_generated | 145 |
| covered_handwritten | 419 |
| covered_opinfo | 265 |
| covered_oracle | 10 |
| covered_property | 4 |
| excluded_deprecated_or_removed | 4 |
| excluded_distributed_scope | 2 |
| excluded_framework_plumbing | 20 |
| excluded_host_storage | 1 |
| excluded_unsupported_public_api | 1 |
| pending_backend_pack | 55 |
| pending_property | 20 |

### Level 3

| Status | Count |
| --- | --- |
| covered_backend_pack | 9 |
| covered_generated | 713 |
| covered_handwritten | 184 |
| covered_oracle | 3 |
| covered_property | 9 |
| excluded_framework_plumbing | 11 |
| excluded_host_storage | 1 |
| excluded_unsupported_public_api | 13 |
| pending_backend_pack | 65 |
| pending_property | 43 |

### Level 4

| Status | Count |
| --- | --- |
| covered_backend_pack | 5 |
| covered_generated | 264 |
| covered_handwritten | 92 |
| covered_property | 1 |
| excluded_framework_plumbing | 1 |
| pending_backend_pack | 14 |
| pending_property | 7 |

### Level 5

| Status | Count |
| --- | --- |
| covered_generated | 347 |
| covered_handwritten | 9 |
| covered_property | 6 |
| excluded_unsupported_public_api | 2 |
| pending_backend_pack | 9 |
| pending_property | 7 |

### Level 6

| Status | Count |
| --- | --- |
| covered_generated | 19 |

### Level 7

| Status | Count |
| --- | --- |
| covered_handwritten | 6 |

## Semantic Level By Surface Kind

### Level 1

| Surface kind | Count |
| --- | --- |
| functional_data | 123 |
| mutating_or_inplace | 140 |
| out_variant | 159 |

### Level 2

| Surface kind | Count |
| --- | --- |
| autograd_backward | 35 |
| factory | 39 |
| functional_data | 512 |
| layout_storage | 108 |
| mutating_or_inplace | 32 |
| out_variant | 177 |
| rng | 20 |
| view_or_alias | 29 |

### Level 3

| Surface kind | Count |
| --- | --- |
| autograd_backward | 77 |
| functional_data | 240 |
| mutating_or_inplace | 184 |
| out_variant | 413 |
| rng | 31 |
| view_or_alias | 106 |

### Level 4

| Surface kind | Count |
| --- | --- |
| autograd_backward | 33 |
| functional_data | 132 |
| metadata_device | 3 |
| mutating_or_inplace | 88 |
| out_variant | 126 |
| view_or_alias | 2 |

### Level 5

| Surface kind | Count |
| --- | --- |
| functional_data | 169 |
| layout_storage | 31 |
| mutating_or_inplace | 10 |
| out_variant | 170 |

### Level 6

| Surface kind | Count |
| --- | --- |
| functional_data | 10 |
| out_variant | 9 |

### Level 7

| Surface kind | Count |
| --- | --- |
| functional_data | 4 |
| out_variant | 2 |

## Generated Coverage Depth

| Metric | Value |
| --- | --- |
| Generated surfaces with case plans | 1910 |
| Generated semantic cases | 1921 |
| Required generated semantic cases | 1921 |
| Optional generated semantic cases | 0 |

## Generated Semantic Cases By Strategy

| Name | Count |
| --- | --- |
| manual_bitwise | 41 |
| manual_convolution | 26 |
| manual_elementwise | 381 |
| manual_factory | 19 |
| manual_factory_out | 49 |
| manual_fft | 38 |
| manual_foreach | 219 |
| manual_grid | 9 |
| manual_grid_backward | 5 |
| manual_indexing | 87 |
| manual_linalg | 119 |
| manual_loss | 26 |
| manual_matmul | 23 |
| manual_metadata | 39 |
| manual_multi_output_reduction | 158 |
| manual_padding | 14 |
| manual_pooling | 26 |
| manual_reduction | 85 |
| manual_rng | 63 |
| manual_rnn_cell | 4 |
| manual_shape | 186 |
| manual_special_math | 188 |
| manual_upsample | 37 |
| opinfo_inplace_unary | 31 |
| opinfo_out | 31 |
| opinfo_view_alias | 17 |

## Generated Semantic Cases By Semantic Level

| Name | Count |
| --- | --- |
| 1 | 422 |
| 2 | 145 |
| 3 | 713 |
| 4 | 275 |
| 5 | 347 |
| 6 | 19 |

## Generated Covered Surfaces By Strategy

| Name | Count |
| --- | --- |
| manual_elementwise | 411 |
| manual_foreach | 222 |
| manual_special_math | 188 |
| manual_shape | 187 |
| manual_multi_output_reduction | 166 |
| manual_linalg | 125 |
| manual_reduction | 104 |
| manual_indexing | 97 |
| manual_rng | 71 |
| manual_factory_out | 49 |
| manual_fft | 42 |
| manual_bitwise | 41 |
| manual_upsample | 41 |
| manual_metadata | 40 |
| opinfo_out | 31 |
| opinfo_inplace_unary | 31 |
| manual_pooling | 30 |
| manual_loss | 30 |
| manual_convolution | 27 |
| opinfo_view_alias | 24 |
| manual_factory | 19 |
| manual_padding | 17 |
| manual_matmul | 12 |
| manual_grid | 9 |
| manual_grid_backward | 5 |
| manual_rnn_cell | 4 |

## Generated Covered Surfaces By Strategy Family

| Name | Count |
| --- | --- |
| unary | 97 |
| unknown | 86 |
| binary | 45 |
| extrema | 36 |
| ternary | 18 |
| pow | 18 |
| lerp | 15 |
| window | 13 |
| set | 13 |
| norm | 12 |
| div | 12 |
| scatter | 12 |
| divide | 10 |
| bitwise_left_shift | 9 |
| bitwise_right_shift | 9 |
| normal | 9 |
| random | 9 |
| bernoulli | 8 |
| float_power | 8 |
| linalg_matrix_rank | 8 |
| linalg_pinv | 8 |
| max | 8 |
| min | 8 |
| remainder | 8 |
| sort | 8 |
| xlogy | 8 |
| bitwise_and | 7 |
| bitwise_or | 7 |
| bitwise_xor | 7 |
| _add_relu | 6 |
| add | 6 |
| all | 6 |
| any | 6 |
| clamp | 6 |
| clamp_max | 6 |
| clamp_min | 6 |
| clip | 6 |
| copysign | 6 |
| eq | 6 |
| fill | 6 |
| floor_divide | 6 |
| fmod | 6 |
| ge | 6 |
| greater | 6 |
| greater_equal | 6 |
| gt | 6 |
| index_fill | 6 |
| isin | 6 |
| le | 6 |
| less | 6 |
| less_equal | 6 |
| lt | 6 |
| masked_fill | 6 |
| median | 6 |
| mul | 6 |
| nanmedian | 6 |
| ne | 6 |
| not_equal | 6 |
| randint_like | 6 |
| round | 6 |
| special_chebyshev_polynomial_t | 6 |
| special_chebyshev_polynomial_u | 6 |
| special_chebyshev_polynomial_v | 6 |
| special_chebyshev_polynomial_w | 6 |
| special_hermite_polynomial_h | 6 |
| special_hermite_polynomial_he | 6 |
| special_laguerre_polynomial_l | 6 |
| special_legendre_polynomial_p | 6 |
| special_shifted_chebyshev_polynomial_t | 6 |
| special_shifted_chebyshev_polynomial_u | 6 |
| special_shifted_chebyshev_polynomial_v | 6 |
| special_shifted_chebyshev_polynomial_w | 6 |
| special_xlog1py | 6 |
| special_xlogy | 6 |
| special_zeta | 6 |
| squeeze | 6 |
| squeeze_copy | 6 |
| sub | 6 |
| where | 6 |
| copy | 5 |
| multiply | 5 |
| std | 5 |
| subtract | 5 |
| true_divide | 5 |
| var | 5 |
| _aminmax | 4 |
| _ctc_loss | 4 |
| _native_batch_norm_legit | 4 |
| bucketize | 4 |
| count_nonzero | 4 |
| hamming_window | 4 |
| kthvalue | 4 |
| linalg_cond | 4 |
| linalg_matrix_norm | 4 |
| linalg_norm | 4 |
| linspace | 4 |
| logspace | 4 |
| mean | 4 |
| mode | 4 |
| nanquantile | 4 |
| native_norm | 4 |
| nuclear_norm | 4 |
| prod | 4 |
| quantile | 4 |
| randint | 4 |
| range | 4 |
| result_type | 4 |
| rsub | 4 |
| searchsorted | 4 |
| std_mean | 4 |
| sum | 4 |
| to | 4 |
| upsample_bilinear2d | 4 |
| upsample_nearest2d | 4 |
| var_mean | 4 |
| view_copy | 4 |
| _convolution | 3 |
| _ctc_loss_backward | 3 |
| _index_put_impl | 3 |
| _upsample_bicubic2d_aa | 3 |
| _upsample_bilinear2d_aa | 3 |
| _upsample_lanczos2d_aa | 3 |
| _upsample_nearest_exact1d | 3 |
| _upsample_nearest_exact2d | 3 |
| _upsample_nearest_exact3d | 3 |
| absolute | 3 |
| addcdiv | 3 |
| addcmul | 3 |
| addmv | 3 |
| addr | 3 |
| arccos | 3 |
| arccosh | 3 |
| arcsin | 3 |
| arcsinh | 3 |
| arctan | 3 |
| arctan2 | 3 |
| arctanh | 3 |
| atan2 | 3 |
| cauchy | 3 |
| celu | 3 |
| conj_physical | 3 |
| cumprod | 3 |
| cumsum | 3 |
| deg2rad | 3 |
| digamma | 3 |
| elu | 3 |
| embedding_renorm | 3 |
| empty | 3 |
| erfinv | 3 |
| exponential | 3 |
| fix | 3 |
| gcd | 3 |
| gelu | 3 |
| geometric | 3 |
| hardsigmoid | 3 |
| hardswish | 3 |
| hardtanh | 3 |
| heaviside | 3 |
| hypot | 3 |
| i0 | 3 |
| igamma | 3 |
| igammac | 3 |
| index_add | 3 |
| index_copy | 3 |
| index_put | 3 |
| index_reduce | 3 |
| kaiser_window | 3 |
| lcm | 3 |
| ldexp | 3 |
| leaky_relu | 3 |
| lgamma | 3 |
| log_normal | 3 |
| logical_and | 3 |
| logical_not | 3 |
| logical_or | 3 |
| logical_xor | 3 |
| logit | 3 |
| masked_scatter | 3 |
| mish | 3 |
| mvlgamma | 3 |
| nan_to_num | 3 |
| negative | 3 |
| nextafter | 3 |
| polygamma | 3 |
| put | 3 |
| rad2deg | 3 |
| relu | 3 |
| renorm | 3 |
| scatter_add | 3 |
| scatter_reduce | 3 |
| sgn | 3 |
| sign | 3 |
| silu | 3 |
| sinc | 3 |
| square | 3 |
| tensor_split | 3 |
| threshold | 3 |
| tril | 3 |
| triu | 3 |
| uniform | 3 |
| upsample_bicubic2d | 3 |
| upsample_linear1d | 3 |
| upsample_nearest1d | 3 |
| upsample_nearest3d | 3 |
| upsample_trilinear3d | 3 |
| zero | 3 |
| _adaptive_avg_pool2d | 2 |
| _adaptive_avg_pool3d | 2 |
| _addmm_activation | 2 |
| _assert_async | 2 |
| _batch_norm_no_update | 2 |
| _batch_norm_with_update | 2 |
| _cdist_forward | 2 |
| _cholesky_solve_helper | 2 |
| _conj_copy | 2 |
| _conj_physical | 2 |
| _conv_depthwise2d | 2 |
| _copy_from | 2 |
| _copy_from_and_resize | 2 |
| _dirichlet_grad | 2 |
| _embedding_bag | 2 |
| _embedding_bag_forward_only | 2 |
| _euclidean_dist | 2 |
| _fake_quantize_learnable_per_channel_affine | 2 |
| _fake_quantize_learnable_per_tensor_affine | 2 |
| _fake_quantize_per_tensor_affine_cachemask_tensor_qparams | 2 |
| _fft_c2c | 2 |
| _fft_c2r | 2 |
| _fft_r2c | 2 |
| _grid_sampler_2d_cpu_fallback | 2 |
| _histogramdd_from_bin_cts | 2 |
| _histogramdd_from_bin_tensors | 2 |
| _linalg_det | 2 |
| _linalg_eigh | 2 |
| _linalg_slogdet | 2 |
| _linalg_solve_ex | 2 |
| _log_softmax | 2 |
| _log_softmax_backward_data | 2 |
| _logcumsumexp | 2 |
| _masked_scale | 2 |
| _masked_softmax | 2 |
| _native_batch_norm_legit_no_training | 2 |
| _neg_view_copy | 2 |
| _new_zeros_with_same_feature_meta | 2 |
| _pdist_forward | 2 |
| _reshape_alias_copy | 2 |
| _sample_dirichlet | 2 |
| _segment_reduce_backward | 2 |
| _slow_conv2d_forward | 2 |
| _softmax | 2 |
| _softmax_backward_data | 2 |
| _stack | 2 |
| _standard_gamma | 2 |
| _standard_gamma_grad | 2 |
| _to_copy | 2 |
| _unique | 2 |
| _unique2 | 2 |
| adaptive_avg_pool1d | 2 |
| adaptive_avg_pool2d | 2 |
| adaptive_avg_pool3d | 2 |
| adaptive_max_pool2d | 2 |
| adaptive_max_pool3d | 2 |
| affine_grid_generator | 2 |
| alias_copy | 2 |
| amax | 2 |
| amin | 2 |
| aminmax | 2 |
| angle | 2 |
| arange | 2 |
| argmax | 2 |
| argmin | 2 |
| avg_pool1d | 2 |
| avg_pool2d | 2 |
| avg_pool3d | 2 |
| bartlett_window | 2 |
| batch_norm_update_stats | 2 |
| binary_cross_entropy | 2 |
| binary_cross_entropy_with_logits | 2 |
| bincount | 2 |
| binomial | 2 |
| bitwise_not | 2 |
| blackman_window | 2 |
| chain_matmul | 2 |
| channel_shuffle | 2 |
| cholesky | 2 |
| cholesky_inverse | 2 |
| cholesky_solve | 2 |
| col2im | 2 |
| complex | 2 |
| constant_pad_nd | 2 |
| conv_depthwise3d | 2 |
| convolution | 2 |
| convolution_overrideable | 2 |
| cross | 2 |
| ctc_loss | 2 |
| cummax | 2 |
| cummin | 2 |
| detach_copy | 2 |
| diagonal_copy | 2 |
| diagonal_scatter | 2 |
| dist | 2 |
| dot | 2 |
| dsplit | 2 |
| embedding | 2 |
| embedding_bag | 2 |
| expand_copy | 2 |
| eye | 2 |
| fake_quantize_per_channel_affine_cachemask | 2 |
| fake_quantize_per_tensor_affine_cachemask | 2 |
| fft_fft | 2 |
| fft_fft2 | 2 |
| frequency | 2 |
| fft_fftn | 2 |
| fft_hfft | 2 |
| fft_hfft2 | 2 |
| fft_hfftn | 2 |
| fft_ifft | 2 |
| fft_ifft2 | 2 |
| fft_ifftn | 2 |
| fft_ihfft | 2 |
| fft_ihfft2 | 2 |
| fft_ihfftn | 2 |
| fft_irfft | 2 |
| fft_irfft2 | 2 |
| fft_irfftn | 2 |
| fft_rfft | 2 |
| fft_rfft2 | 2 |
| fft_rfftn | 2 |
| fmax | 2 |
| fmin | 2 |
| fractional_max_pool2d | 2 |
| fractional_max_pool3d | 2 |
| frexp | 2 |
| frobenius_norm | 2 |
| full | 2 |
| gather | 2 |
| geqrf | 2 |
| ger | 2 |
| grid_sampler_2d | 2 |
| grid_sampler_2d_backward | 2 |
| grid_sampler_3d | 2 |
| grid_sampler_3d_backward | 2 |
| hann_window | 2 |
| hardshrink | 2 |
| histc | 2 |
| histogram | 2 |
| hsplit | 2 |
| huber_loss | 2 |
| im2col | 2 |
| index | 2 |
| index_select | 2 |
| inner | 2 |
| inverse | 2 |
| is_contiguous | 2 |
| isinf | 2 |
| isnan | 2 |
| isneginf | 2 |
| isposinf | 2 |
| kron | 2 |
| lift | 2 |
| lift_fresh_copy | 2 |
| linalg_cholesky | 2 |
| linalg_cholesky_ex | 2 |
| linalg_cross | 2 |
| linalg_det | 2 |
| linalg_inv | 2 |
| linalg_inv_ex | 2 |
| linalg_lu | 2 |
| linalg_lu_factor | 2 |
| linalg_lu_factor_ex | 2 |
| linalg_matmul | 2 |
| linalg_matrix_exp | 2 |
| linalg_matrix_power | 2 |
| linalg_qr | 2 |
| linalg_slogdet | 2 |
| linalg_solve | 2 |
| linalg_solve_ex | 2 |
| linalg_solve_triangular | 2 |
| linalg_svdvals | 2 |
| linalg_tensorinv | 2 |
| linalg_tensorsolve | 2 |
| linalg_vecdot | 2 |
| linalg_vector_norm | 2 |
| log_sigmoid | 2 |
| log_sigmoid_forward | 2 |
| log_softmax | 2 |
| logaddexp | 2 |
| logaddexp2 | 2 |
| logcumsumexp | 2 |
| logsumexp | 2 |
| masked_select | 2 |
| matrix_power | 2 |
| max_pool2d_with_indices | 2 |
| max_pool3d_with_indices | 2 |
| maximum | 2 |
| minimum | 2 |
| moveaxis | 2 |
| movedim | 2 |
| mse_loss | 2 |
| multi_margin_loss | 2 |
| multilabel_margin_loss | 2 |
| multilabel_margin_loss_forward | 2 |
| multinomial | 2 |
| mv | 2 |
| nanmean | 2 |
| nansum | 2 |
| native_batch_norm | 2 |
| native_batch_norm_backward | 2 |
| nll_loss | 2 |
| nll_loss2d | 2 |
| nll_loss2d_forward | 2 |
| nll_loss_forward | 2 |
| nonzero | 2 |
| ones | 2 |
| orgqr | 2 |
| ormqr | 2 |
| outer | 2 |
| permute_copy | 2 |
| pixel_shuffle | 2 |
| pixel_unshuffle | 2 |
| poisson | 2 |
| polar | 2 |
| qr | 2 |
| rand | 2 |
| rand_like | 2 |
| randn | 2 |
| randn_like | 2 |
| randperm | 2 |
| reflection_pad1d | 2 |
| reflection_pad2d | 2 |
| reflection_pad3d | 2 |
| relu6 | 2 |
| replication_pad1d | 2 |
| replication_pad2d | 2 |
| replication_pad3d | 2 |
| segment_reduce | 2 |
| select_copy | 2 |
| select_scatter | 2 |
| selu | 2 |
| signbit | 2 |
| slice_copy | 2 |
| slice_scatter | 2 |
| slogdet | 2 |
| slow_conv3d | 2 |
| slow_conv3d_forward | 2 |
| slow_conv_dilated2d | 2 |
| slow_conv_dilated3d | 2 |
| smooth_l1_loss | 2 |
| soft_margin_loss | 2 |
| softmax | 2 |
| softplus | 2 |
| softshrink | 2 |
| special_airy_ai | 2 |
| special_bessel_j0 | 2 |
| special_bessel_j1 | 2 |
| special_bessel_y0 | 2 |
| special_bessel_y1 | 2 |
| special_digamma | 2 |
| special_entr | 2 |
| special_erf | 2 |
| special_erfc | 2 |
| special_erfcx | 2 |
| special_erfinv | 2 |
| special_exp2 | 2 |
| special_expit | 2 |
| special_expm1 | 2 |
| special_gammainc | 2 |
| special_gammaincc | 2 |
| special_gammaln | 2 |
| special_i0 | 2 |
| special_i0e | 2 |
| special_i1 | 2 |
| special_i1e | 2 |
| special_log1p | 2 |
| special_log_ndtr | 2 |
| special_logit | 2 |
| special_logsumexp | 2 |
| special_modified_bessel_i0 | 2 |
| special_modified_bessel_i1 | 2 |
| special_modified_bessel_k0 | 2 |
| special_modified_bessel_k1 | 2 |
| special_multigammaln | 2 |
| special_ndtr | 2 |
| special_ndtri | 2 |
| special_polygamma | 2 |
| special_psi | 2 |
| special_round | 2 |
| special_scaled_modified_bessel_k0 | 2 |
| special_scaled_modified_bessel_k1 | 2 |
| special_sinc | 2 |
| special_spherical_bessel_j0 | 2 |
| split_copy | 2 |
| split_with_sizes_copy | 2 |
| swapaxes | 2 |
| swapdims | 2 |
| take | 2 |
| take_along_dim | 2 |
| tensordot | 2 |
| thnn_conv2d | 2 |
| topk | 2 |
| trace | 2 |
| transpose_copy | 2 |
| unfold_copy | 2 |
| unique_consecutive | 2 |
| unique_dim | 2 |
| unique_dim_consecutive | 2 |
| unsqueeze_copy | 2 |
| vdot | 2 |
| view_as_complex_copy | 2 |
| view_as_real_copy | 2 |
| vsplit | 2 |
| zeros | 2 |
| _assert_tensor_metadata | 1 |
| _batch_norm_impl_index | 1 |
| _batch_norm_impl_index_backward | 1 |
| _batch_norm_with_update_functional | 1 |
| _cast_Byte | 1 |
| _cast_Char | 1 |
| _cast_Double | 1 |
| _cast_Float | 1 |
| _cast_Half | 1 |
| _cast_Int | 1 |
| _cast_Long | 1 |
| _cast_Short | 1 |
| _chunk_cat | 1 |
| _debug_has_internal_overlap | 1 |
| _dimI | 1 |
| _dimV | 1 |
| _dim_arange | 1 |
| zero_tensor | 1 |
| _grid_sampler_2d_cpu_fallback_backward | 1 |
| _has_compatible_shallow_copy_type | 1 |
| _histogramdd_bin_edges | 1 |
| _is_all_true | 1 |
| _is_any_true | 1 |
| _is_zerotensor | 1 |
| _local_scalar_dense | 1 |
| _native_batch_norm_legit_functional | 1 |
| _neg_view | 1 |
| _pad_circular | 1 |
| _pad_enum | 1 |
| _reshape_alias | 1 |
| _reshape_copy | 1 |
| _reshape_from_tensor | 1 |
| _safe_softmax | 1 |
| _shape_as_tensor | 1 |
| _version | 1 |
| adaptive_max_pool1d | 1 |
| addbmm | 1 |
| addbmm_ | 1 |
| addmm | 1 |
| addmm_ | 1 |
| adjoint | 1 |
| alias | 1 |
| argsort | 1 |
| as_strided | 1 |
| as_strided_copy | 1 |
| as_strided_scatter | 1 |
| baddbmm | 1 |
| baddbmm_ | 1 |
| batch_norm_backward | 1 |
| block_diag | 1 |
| bmm | 1 |
| cat | 1 |
| chunk | 1 |
| clone | 1 |
| column_stack | 1 |
| concat | 1 |
| concatenate | 1 |
| contiguous | 1 |
| cosine_embedding_loss | 1 |
| cross_entropy_loss | 1 |
| data | 1 |
| dense_dim | 1 |
| detach | 1 |
| diag | 1 |
| diag_embed | 1 |
| diff | 1 |
| dim | 1 |
| dstack | 1 |
| empty_like | 1 |
| empty_permuted | 1 |
| empty_strided | 1 |
| fake_quantize_per_tensor_affine | 1 |
| fft_fftfreq | 1 |
| fft_fftshift | 1 |
| fft_ifftshift | 1 |
| fft_rfftfreq | 1 |
| fill_diagonal | 1 |
| flatten_dense_tensors | 1 |
| flip | 1 |
| full_like | 1 |
| glu | 1 |
| grid_sampler | 1 |
| gru_cell | 1 |
| hinge_embedding_loss | 1 |
| hstack | 1 |
| is_complex | 1 |
| is_conj | 1 |
| is_floating_point | 1 |
| is_inference | 1 |
| is_leaf | 1 |
| is_neg | 1 |
| is_nonzero | 1 |
| is_pinned | 1 |
| is_same_size | 1 |
| is_set_to | 1 |
| is_signed | 1 |
| kl_div | 1 |
| l1_loss | 1 |
| linalg__powsum | 1 |
| linalg_diagonal | 1 |
| linear | 1 |
| lstm_cell | 1 |
| margin_ranking_loss | 1 |
| matmul | 1 |
| matrix_H | 1 |
| max_pool1d_with_indices | 1 |
| mm | 1 |
| msort | 1 |
| narrow | 1 |
| narrow_copy | 1 |
| native_channel_shuffle | 1 |
| new_empty | 1 |
| new_empty_strided | 1 |
| new_full | 1 |
| new_ones | 1 |
| new_zeros | 1 |
| nll_loss_nd | 1 |
| nonzero_numpy | 1 |
| nonzero_static | 1 |
| norm_except_dim | 1 |
| numel | 1 |
| numpy_T | 1 |
| one_hot | 1 |
| ones_like | 1 |
| output_nr | 1 |
| pad | 1 |
| pad_sequence | 1 |
| pairwise_distance | 1 |
| pdist | 1 |
| prelu | 1 |
| repeat | 1 |
| repeat_interleave | 1 |
| resolve_conj | 1 |
| resolve_neg | 1 |
| retains_grad | 1 |
| rnn_relu_cell | 1 |
| rnn_tanh_cell | 1 |
| roll | 1 |
| rot90 | 1 |
| row_stack | 1 |
| scalar_tensor | 1 |
| set_data | 1 |
| size | 1 |
| slice_inverse | 1 |
| special_log_softmax | 1 |
| special_softmax | 1 |
| split | 1 |
| split_with_sizes | 1 |
| stack | 1 |
| sym_is_contiguous | 1 |
| sym_numel | 1 |
| sym_size | 1 |
| t | 1 |
| t_copy | 1 |
| to_dense | 1 |
| transpose | 1 |
| tril_indices | 1 |
| triplet_margin_loss | 1 |
| triu_indices | 1 |
| type_as | 1 |
| unbind_copy | 1 |
| unflatten | 1 |
| unflatten_dense_tensors | 1 |
| unsafe_split | 1 |
| unsafe_split_with_sizes | 1 |
| unsqueeze | 1 |
| vander | 1 |
| view | 1 |
| view_as_complex | 1 |
| vstack | 1 |
| zeros_like | 1 |

## CPU Dtype Contract Stats

| Metric | Value |
| --- | --- |
| Contracted dispatcher entries | 3053 |
| CPU-supported dtype cases | 22757 |
| CPU-unsupported dtype cases | 7820 |
| CPU-pending dtype cases | 787 |
| CPU-unknown dtype cases | 0 |
| Oracle-supported dtype cases | 0 |
| Source-expected ops | 2104 |
| Source-expected dtype entries | 34512 |
| Source/probe mismatches | 8802 |
| Local PyTorch source available | True |
| Local PyTorch ufunc source entries | 1 |

## CPU Dtype Contract Last Run Probe Counts

| Name | Count |
| --- | --- |
| generated_pending | 787 |
| generated_supported | 16254 |
| generated_total | 24830 |
| generated_unsupported | 7789 |
| opinfo_backward_supported | 2656 |
| opinfo_backward_total | 2663 |
| opinfo_backward_unsupported | 7 |
| opinfo_forward_supported | 5894 |
| opinfo_forward_total | 5918 |
| opinfo_forward_unsupported | 24 |
| pytorch_src_ufunc_seeded_ops | 1 |
| source_seeded_ops | 1795 |

## CPU Dtype Contract Version Rules

| Name | Count |
| --- | --- |
| 2.12 | 3053 |

## CPU Dtype Contract Buckets By Dtype

| Name | Count |
| --- | --- |
| cpu_supported:torch.float32 | 2717 |
| cpu_supported:torch.float64 | 2700 |
| cpu_supported:torch.float16 | 2256 |
| cpu_supported:torch.bfloat16 | 2254 |
| cpu_supported:torch.int64 | 1565 |
| cpu_supported:torch.uint8 | 1564 |
| cpu_supported:torch.complex128 | 1562 |
| cpu_supported:torch.complex64 | 1560 |
| cpu_supported:torch.int32 | 1547 |
| cpu_supported:torch.int16 | 1545 |
| cpu_supported:torch.int8 | 1545 |
| cpu_unsupported:torch.complex32 | 1308 |
| cpu_supported:torch.bool | 1223 |
| cpu_unsupported:torch.complex64 | 880 |
| cpu_unsupported:torch.complex128 | 878 |
| cpu_unsupported:torch.bool | 861 |
| cpu_supported:torch.complex32 | 719 |
| cpu_unsupported:torch.int16 | 621 |
| cpu_unsupported:torch.int32 | 621 |
| cpu_unsupported:torch.int8 | 621 |
| cpu_unsupported:torch.int64 | 614 |
| cpu_unsupported:torch.uint8 | 592 |
| cpu_unsupported:torch.bfloat16 | 339 |
| cpu_unsupported:torch.float16 | 337 |
| cpu_unsupported:torch.float64 | 78 |
| cpu_unsupported:torch.float32 | 70 |
| cpu_pending:torch.uint8 | 69 |
| cpu_pending:torch.bool | 66 |
| cpu_pending:torch.float64 | 66 |
| cpu_pending:torch.bfloat16 | 62 |
| cpu_pending:torch.float16 | 62 |
| cpu_pending:torch.complex128 | 58 |
| cpu_pending:torch.complex64 | 58 |
| cpu_pending:torch.float32 | 58 |
| cpu_pending:torch.int16 | 58 |
| cpu_pending:torch.int32 | 58 |
| cpu_pending:torch.int64 | 58 |
| cpu_pending:torch.int8 | 58 |
| cpu_pending:torch.complex32 | 56 |

## CPU Dtype Contract Source Conditions

| Name | Count |
| --- | --- |
| * | 19080 |
| forward:* | 13018 |
| backward:* | 2414 |

## CPU Dtype Contract Source Probe Mismatches

| Name | Count |
| --- | --- |
| cpu_supported_but_missing_from_source | 8332 |
| source_expected_but_cpu_unsupported | 470 |

## Marker And Source Coverage Stats

| Metric | Value |
| --- | --- |
| Coverage markers discovered | 620 |
| Category markers discovered | 755 |
| Unmapped hand-authored tests | 0 |
| Audit warnings | 0 |
| Audit errors | 0 |

## Coverage Markers By Suite

| Name | Count |
| --- | --- |
| operators | 461 |
| autograd | 150 |
| dtypes | 84 |
| strides | 49 |
| rng | 40 |
| training | 30 |
| workloads | 6 |

## Category Markers By Suite

| Name | Count |
| --- | --- |
| selftest | 275 |
| operators | 187 |
| workloads | 57 |
| strides | 40 |
| device_api | 33 |
| training | 31 |
| dtypes | 30 |
| autograd | 29 |
| rng | 24 |
| compiler | 13 |
| generated | 10 |
| stress | 10 |
| memory | 4 |
| multi_device | 4 |
| opinfo | 3 |
| serialization | 3 |
| errors | 2 |

## Pending Blocker Counts

| Name | Count |
| --- | --- |
| needs_backend_pack | 143 |
| kernel_unavailable_in_host_build | 60 |
| out_of_backend_conformance_scope | 56 |
| needs_public_proxy_proof | 16 |
| needs_valid_internal_inputs | 1 |

## Pending Backend Gate Counts

| Name | Count |
| --- | --- |
| any | 125 |
| cuda | 55 |
| cpu_build | 40 |
| rocm | 21 |
| fbgemm | 16 |
| cpu | 9 |
| mps | 9 |
| xla | 1 |

## Pending Required Closure Counts

| Name | Count |
| --- | --- |
| implement_backend_gated_runner | 143 |
| validate_on_backend_build_or_keep_pending | 60 |
| none_dispatcher_plumbing | 32 |
| prove_public_proxy_or_add_direct_runner | 16 |
| none_unsupported_public_api | 16 |
| none_deprecated_or_removed | 4 |
| none_host_storage_only | 2 |
| none_distributed_scope | 2 |
| construct_valid_internal_inputs_and_property_runner | 1 |

## Pending Next Family Counts

| Name | Count |
| --- | --- |
| backend_specific_internal | 119 |
| cpu_reference_invalid | 61 |
| dispatcher_plumbing | 32 |
| covered_by_public_surface | 16 |
| manual_future_scope | 12 |
| philox_mps_rng | 8 |
| semi_structured_sparse_backend_pack | 7 |
| unsafe_direct_invocation | 5 |
| quantized_legacy_rnn_removed | 4 |
| quantized_static_rnn_cell | 4 |
| fused_dropout_backend_pack | 3 |
| wrapped_quantized_linear_backend_pack | 2 |
| distributed_or_c10d | 2 |
| int4_scales_zeros_meta | 1 |

## Pending Source Category Counts

| Name | Count |
| --- | --- |
| backend_specific_internal | 119 |
| cpu_reference_invalid | 70 |
| dispatcher_plumbing | 32 |
| unsafe_direct_invocation | 25 |
| covered_by_public_surface | 16 |
| manual_future_scope | 12 |
| distributed_or_c10d | 2 |

## Exclusion Category Counts

| Name | Count |
| --- | --- |
| backend_specific_internal | 137 |
| cpu_reference_invalid | 77 |
| unsafe_direct_invocation | 53 |
| dispatcher_plumbing | 32 |
| covered_by_public_surface | 18 |
| manual_future_scope | 12 |
| distributed_or_c10d | 2 |

## Exclusion Match Counts

| Name | Count |
| --- | --- |
| exact | 177 |
| regex | 154 |

## Exclusion Surface Counts

| Name | Count |
| --- | --- |
| functional_data | 210 |
| out_variant | 65 |
| layout_storage | 22 |
| autograd_backward | 11 |
| mutating_or_inplace | 10 |
| view_or_alias | 6 |
| factory | 5 |
| rng | 1 |
| metadata_device | 1 |

## Known Crash Isolation Stats

| Metric | Value |
| --- | --- |
| Rules | 10 |
| Rules with constraints | 4 |

## Known Crash Rules By Backend

| Name | Count |
| --- | --- |
| mps | 10 |

## Known Crash Rules By Match Mode

| Name | Count |
| --- | --- |
| dispatcher | 6 |
| nodeid | 2 |
| coverage_id | 2 |

## Known Crash Rules By Evidence Scope

| Name | Count |
| --- | --- |
| dispatcher_surface | 4 |
| constrained_metadata | 4 |
| exact_node | 2 |

## Known Crash Rules By Classification

| Name | Count |
| --- | --- |
| confirmed_backend_crash | 10 |

## Known Crash Rules By Expected Signal

| Name | Count |
| --- | --- |
| SIGSEGV | 10 |

## Known Crash Constraint Key Counts

| Name | Count |
| --- | --- |
| suite | 4 |
| coverage_kind | 4 |
| nodeid_glob | 4 |
| surface_kind | 3 |
| variant_kind | 3 |
| strategy | 3 |
| semantic_level | 3 |
| strategy_family | 2 |
| coverage_id_glob | 2 |
| dtype | 1 |

## Notes For Website Use

- Use coverage and collection numbers as current-checkout statistics, not universal PyTorch promises.
- `unknown=0` means TorchCTS has an explicit disposition for every tensor-touching backend-relevant ATen surface in this audit.
- Pending backend-pack counts are intentional hardware/build gates, not claimed coverage.
- Known crash rules are subprocess isolation policy only; they do not skip, xfail, or downgrade failures.
- Re-run this script after changing tests, generated coverage, coverage exclusions, known crash rules, or PyTorch versions.
