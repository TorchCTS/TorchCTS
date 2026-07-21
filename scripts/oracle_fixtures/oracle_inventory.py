# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software.

"""Stable development-only metadata for mapping runtime oracles to QA packs.

This module deliberately lives outside ``torchcts`` and has no torch
dependency. Runtime routing does not import it; the inventory gate compares
these reviewed maps with runtime discovery so drift is visible without making
QA metadata a software dependency.
"""

from __future__ import annotations


V1_FIXED_VALUE = "V1_FIXED_VALUE"
V2_ADMISSIBILITY = "V2_ADMISSIBILITY"
V3_ROUTING = "V3_ROUTING"
V4_PROPERTY = "V4_PROPERTY"
V5_BACKEND_SEMANTIC = "V5_BACKEND_SEMANTIC"
V6_DISPOSITION = "V6_DISPOSITION"

VALIDATION_CLASSES = frozenset({
    V1_FIXED_VALUE,
    V2_ADMISSIBILITY,
    V3_ROUTING,
    V4_PROPERTY,
    V5_BACKEND_SEMANTIC,
    V6_DISPOSITION,
})


REFERENCE_FUNCTION_CASE_PACKS = {
    "quantized_opmath_tensor": "CP-CAST",
    "cast_public_result": "CP-CAST",
    "matmul_family_determinate_reference": "CP-MATMUL",
    "matmul_family_reference": "CP-MATMUL",
    "soft_margin_loss_reference": "CP-SOFTMARGIN",
    "soft_margin_loss_backward_reference": "CP-SOFTMARGIN",
    "segment_reduce_prod_backward_reference": "CP-SEGMENT",
    "segment_reduce_prod_reference": "CP-SEGMENT",
    "linear_backward_reference": "CP-LINEAR-BWD",
    "max_pool2d_backward_reference": "CP-POOL",
    "pack_int4_values": "CP-INT4",
    "unpack_int4_values": "CP-INT4",
    "tinygemm_int4_dequantize_reference": "CP-INT4",
    "tinygemm_int4_matmul_reference": "CP-INT4",
    "unpack_dynamic_int4_weight_bytes": "CP-INT4",
    "dynamic_int4_dequantize_reference": "CP-INT4",
    "dynamic_int4_matmul_reference": "CP-INT4",
    "complex_unit_alpha_add_sub_reference": "CP-COMPLEX-ARITH",
    "has_nonnegative_integral_complex_exponent": "CP-COMPLEX-ARITH",
    "complex_tensor_integer_power_reference": "CP-COMPLEX-ARITH",
    "complex_l1_loss_reference": "CP-COMPLEX-LOSS",
    "binary_cross_entropy_with_logits_reference": "CP-COMPLEX-LOSS",
    "complex_log2_reference": "CP-COMPLEX-UNARY",
    "complex_sigmoid_reference": "CP-COMPLEX-UNARY",
    "complex_rsqrt_reference": "CP-COMPLEX-UNARY",
    "complex_expm1_reference": "CP-COMPLEX-UNARY",
    "complex_ldexp_reference": "CP-LDEXP-CUMPROD",
    "complex_integral_ldexp_reference": "CP-LDEXP-CUMPROD",
    "complex_mul_reference": "CP-COMPLEX-ARITH",
    "foreach_complex_compound_reference": "CP-COMPLEX-ARITH",
    "complex_cumprod_reference": "CP-LDEXP-CUMPROD",
    "complex_gradient_reference": "CP-GRAD-COV",
    "complex_covariance_reference": "CP-GRAD-COV",
    "laguerre_polynomial_reference": "CP-POLYNOMIAL",
    "shifted_chebyshev_polynomial_reference": "CP-POLYNOMIAL",
    "lanczos3_coefficient": "CP-LANCZOS",
    "lanczos2d_aa_reference": "CP-LANCZOS",
    "float_to_uint8_reference": "CP-INTEGER",
    "unsigned_negation_reference": "CP-INTEGER",
    "saturate_weight_to_fp16_reference": "CP-CAST",
    "weight_int8pack_mm_reference": "CP-QUANT",
    "gcd_integer_reference": "CP-INTEGER",
    "histc_integer_count_reference": "CP-HISTC",
    "im2col_reference": "CP-IM2COL",
    "col2im_reference": "CP-IM2COL",
    "wide_ldexp_reference": "CP-LDEXP-CUMPROD",
    "logit_backward_reference": "CP-LOGIT",
    "grid_sampler_backward_f32_reference": "CP-GRID",
    "grid_sampler_3d_backward_f32_reference": "CP-GRID",
    "grid_sampler_forward_f32_reference": "CP-GRID",
    "conv_transpose_f32_reference": "CP-CONV",
    "conv_transpose3d_f32_reference": "CP-CONV",
    "matrix_exp_f64_reference": "CP-MATRIXEXP",
    "embedding_bag_scale_grad_by_freq_reference": "CP-EMBED",
    "complex_convolution_reference": "CP-CONV",
    "slow_complex_convolution_reference": "CP-CONV",
}


HIGH_PRECISION_FUNCTION_CASE_PACKS = {
    "dirichlet_grad_reference": "CP-SPECIAL",
    "standard_gamma_grad_reference": "CP-SPECIAL",
    "i0_reference": "CP-SPECIAL",
    "polygamma_reference": "CP-SPECIAL",
    "regularized_gamma_reference": "CP-SPECIAL",
    "kaiser_window_reference": "CP-SPECIAL",
}


BACKWARD_FUNCTION_CASE_PACKS = {
    "has_opinfo_backward_reference": "CP-ROUTING",
    "vander_backward_reference": "CP-LINALG-BWD",
    "complex_rpow_backward_reference": "CP-LINALG-BWD",
    "condition_number_backward_reference": "CP-LINALG-BWD",
    "group_norm_backward_reference": "CP-NORM-SPARSE-BWD",
    "sampled_addmm_backward_reference": "CP-NORM-SPARSE-BWD",
    "renorm_inf_backward_reference": "CP-NORM-SPARSE-BWD",
    "resolve_opinfo_backward_reference": "CP-ROUTING",
}


FFT_FUNCTION_CASE_PACKS = {
    "public_fft_contract_spec": "CP-FFT",
    "generated_c2c_fft_contract_spec": "CP-FFT",
    "fft_source_contributor_mask": "CP-FFT",
    "compare_fft_nonfinite_groups": "CP-FFT",
}


FORWARD_REFERENCE_CASE_PACKS = {
    "addbmm_f32_opmath": "CP-MATMUL",
    "bce_with_logits_stable": "CP-COMPLEX-LOSS",
    "complex_convolution_four_real": "CP-CONV",
    "complex_covariance_real_diagonal": "CP-GRAD-COV",
    "complex_cumprod_inclusive": "CP-LDEXP-CUMPROD",
    "complex_expm1_lane_aware": "CP-COMPLEX-UNARY",
    "complex_gradient_real_spacing": "CP-GRAD-COV",
    "complex_l1_loss": "CP-COMPLEX-LOSS",
    "complex_ldexp_integral_lane_scale": "CP-LDEXP-CUMPROD",
    "complex_ldexp_phase_scale": "CP-LDEXP-CUMPROD",
    "complex_log2": "CP-COMPLEX-UNARY",
    "complex_matmul_wide_semantic": "CP-MATMUL",
    "complex_rsqrt_c99": "CP-COMPLEX-UNARY",
    "complex_sigmoid_stable": "CP-COMPLEX-UNARY",
    "complex_tensor_integer_power": "CP-COMPLEX-ARITH",
    "complex_unit_alpha_add_sub": "CP-COMPLEX-ARITH",
    "conv_transpose_f32_opmath": "CP-CONV",
    "dirichlet_grad_mpmath": "CP-SPECIAL",
    "float_to_uint8_signed_truncate": "CP-INTEGER",
    "foreach_complex_c99_compound": "CP-COMPLEX-ARITH",
    "foreach_complex_c99_tensor_mul": "CP-COMPLEX-ARITH",
    "foreach_complex_real_alpha_add_sub": "CP-COMPLEX-ARITH",
    "grid_sampler_f32": "CP-GRID",
    "i0_mpmath": "CP-SPECIAL",
    "laguerre_initialized_recurrence": "CP-POLYNOMIAL",
    "lanczos3_exact_zero": "CP-LANCZOS",
    "matrix_exp_f64_scaling_squaring": "CP-MATRIXEXP",
    "polygamma_mpmath_poles": "CP-SPECIAL",
    "shifted_chebyshev_initialized_recurrence": "CP-POLYNOMIAL",
    "soft_margin_f32_opmath": "CP-SOFTMARGIN",
    "standard_gamma_grad_mpmath": "CP-SPECIAL",
}


BACKWARD_REFERENCE_CASE_PACKS = {
    "segment_prod_exclusive_f32": "CP-SEGMENT",
    "linalg_cond_recomputed": "CP-LINALG-BWD",
    "linalg_vander_column_power": "CP-LINALG-BWD",
    "complex_rpow_c128": "CP-LINALG-BWD",
    "group_norm_explicit_reduction": "CP-NORM-SPARSE-BWD",
    "sampled_addmm_conjugate_vjp": "CP-NORM-SPARSE-BWD",
    "renorm_inf_unique_max": "CP-NORM-SPARSE-BWD",
}


NON_UNIQUE_VALIDATION_CLASSES = {
    "eigh_eigenvalues": V3_ROUTING,
    "eigh": V2_ADMISSIBILITY,
    "eig": V2_ADMISSIBILITY,
    "eigvals": V2_ADMISSIBILITY,
    "svd": V2_ADMISSIBILITY,
    "qr": V2_ADMISSIBILITY,
    "lu": V2_ADMISSIBILITY,
    "lu_factor": V2_ADMISSIBILITY,
    "pinv": V2_ADMISSIBILITY,
    "lstsq": V2_ADMISSIBILITY,
    "sort": V2_ADMISSIBILITY,
    "arg_reduce": V2_ADMISSIBILITY,
    "value_index": V2_ADMISSIBILITY,
    "mode": V2_ADMISSIBILITY,
    "complex_matmul_determinate": V2_ADMISSIBILITY,
    "fft_special": V2_ADMISSIBILITY,
    "complex_product": V2_ADMISSIBILITY,
    "welford_mean": V2_ADMISSIBILITY,
    "fractional_max_pool": V2_ADMISSIBILITY,
    "max_pool_indices": V2_ADMISSIBILITY,
    "max_unpool_writers": V2_ADMISSIBILITY,
    "rrelu": V2_ADMISSIBILITY,
    "random": V4_PROPERTY,
    "randomized_linalg": V2_ADMISSIBILITY,
    "uninitialized": V4_PROPERTY,
    "geqrf": V3_ROUTING,
    "orgqr_ormqr": V3_ROUTING,
    "value_only_linalg": V3_ROUTING,
}


NON_UNIQUE_CASE_PACKS = {
    "eigh_eigenvalues": "CP-ROUTING",
    "eigh": "CP-LINALG-LEGAL",
    "eig": "CP-LINALG-LEGAL",
    "eigvals": "CP-LINALG-LEGAL",
    "svd": "CP-LINALG-LEGAL",
    "qr": "CP-LINALG-LEGAL",
    "lu": "CP-LINALG-LEGAL",
    "lu_factor": "CP-LINALG-LEGAL",
    "pinv": "CP-LINALG-LEGAL",
    "lstsq": "CP-LINALG-LEGAL",
    "sort": "CP-TIES",
    "arg_reduce": "CP-TIES",
    "value_index": "CP-TIES",
    "mode": "CP-TIES",
    "complex_matmul_determinate": "CP-MATMUL",
    "fft_special": "CP-FFT",
    "complex_product": "CP-COMPLEX-ARITH",
    "welford_mean": "CP-STRUCTURAL",
    "fractional_max_pool": "CP-TIES",
    "max_pool_indices": "CP-TIES",
    "max_unpool_writers": "CP-TIES",
    "rrelu": "CP-STRUCTURAL",
    "random": "CP-STRUCTURAL",
    "randomized_linalg": "CP-LINALG-LEGAL",
    "uninitialized": "CP-STRUCTURAL",
    "geqrf": "CP-ROUTING",
    "orgqr_ormqr": "CP-ROUTING",
    "value_only_linalg": "CP-ROUTING",
}


DIRECT_ORACLE_CASE_PACKS = {
    "sobol_engine_state": "CP-SOBOL",
    "quantized_affine_allocation": "CP-STRUCTURAL",
    "linear_backward": "CP-LINEAR-BWD",
    "max_pool2d_backward": "CP-POOL",
    "unsafe_valid_input_semantics": "CP-METADATA",
    "autocast_cast_policy": "CP-METADATA",
    "native_batch_norm_no_stats_out": "CP-BN",
    "forward_ad_inference_copy": "CP-METADATA",
    "nested_select_backward": "CP-NESTED",
    "sparse_unsafe_valid_constructor": "CP-METADATA",
    "fused_dropout_backend_pack": "CP-DROPOUT",
    "semi_structured_sparse_backend_pack": "CP-SPARSE24",
    "semi_structured_sparse_thread_mask_backend_pack": "CP-SPARSE24",
    "cuda_cslt_backend_pack": "CP-SPARSE24",
    "cuda_cudnn_attention_backend_pack": "CP-SDPA",
    "cuda_cudnn_convolution_backend_pack": "CP-CONV",
    "cuda_cudnn_grid_backend_pack": "CP-GRID",
    "cuda_cudnn_batch_norm_backend_pack": "CP-BN",
    "cuda_cudnn_ctc_backend_pack": "CP-CTC",
    "cuda_cudnn_dropout_state_backend_pack": "CP-OPAQUE-STATE",
    "cuda_cudnn_is_acceptable_backend_pack": "CP-OPAQUE-STATE",
    "cuda_cudnn_rnn_backend_pack": "CP-RNN",
    "cuda_triton_attention_backend_pack": "CP-SDPA",
    "cuda_flash_attention_no_dropout_inplace_backend_pack": "CP-SDPA",
    "cuda_fused_rms_norm_backward_backend_pack": "CP-NORM-SPARSE-BWD",
    "cuda_dtype_out_matmul_backend_pack": "CP-MATMUL",
    "cuda_batch_norm_internal_backend_pack": "CP-BN",
    "cuda_thnn_cell_backward_backend_pack": "CP-RNN",
    "cuda_mixed_dtypes_linear_backend_pack": "CP-QUANT",
    "cuda_scaled_grouped_mm_backend_pack": "CP-MATMUL",
    "mkldnn_shape_backend_pack": "CP-METADATA",
    "mkldnn_linear_backend_pack": "CP-LINEAR-BWD",
    "mkldnn_convolution_backend_pack": "CP-CONV",
    "mkldnn_pooling_backend_pack": "CP-POOL",
    "mkldnn_rnn_backend_pack": "CP-RNN",
    "nnpack_convolution_backend_pack": "CP-CONV",
    "rocm_miopen_convolution_backend_pack": "CP-CONV",
    "rocm_miopen_batch_norm_backend_pack": "CP-BN",
    "rocm_miopen_ctc_backend_pack": "CP-CTC",
    "rocm_miopen_rnn_backend_pack": "CP-RNN",
    "xla_data_propagation_bridge_blocked": "CP-METADATA",
    "fbgemm_linear_backend_pack": "CP-QUANT",
    "wrapped_quantized_linear_backend_pack": "CP-QUANT",
    "cpu_flash_attention_public_sdpa": "CP-SDPA",
    "privateuse1_attention_public_sdpa": "CP-SDPA",
    "quantized_flash_attention_public_sdpa": "CP-SDPA",
    "privateuse1_matmul_backward_formula": "CP-LINEAR-BWD",
    "privateuse1_resize_output_property": "CP-METADATA",
    "privateuse1_batch_norm_forward_formula": "CP-BN",
    "privateuse1_thnn_cell_formula": "CP-RNN",
    "privateuse1_pin_memory_noop": "CP-METADATA",
    "mps_convolution_cpu_reference": "CP-CONV",
    "mps_convolution_backward_out_blocked_schema": "CP-CONV",
    "mps_sdpa_math_public_reference": "CP-SDPA",
    "mps_lstm_cpu_reference": "CP-RNN",
    "mps_philox_rng_backend_pack": "CP-DROPOUT",
    "quantized_dynamic_rnn": "CP-QUANT-RNN",
    "quantized_legacy_rnn_removed": "CP-ROUTING",
    "quantized_static_rnn_cell": "CP-QUANT-RNN",
    "int4_cpu_pack_value_oracle": "CP-INT4",
    "dynamic_int4_pack_matmul_value_oracle": "CP-INT4",
    "int4_mps_pack": "CP-INT4",
    "int4_scales_zeros_meta": "CP-INT4",
}


_DIRECT_VALUE_IDS = frozenset({"sobol_engine_state"})
_DIRECT_PROPERTY_IDS = frozenset({
    "quantized_affine_allocation",
    "unsafe_valid_input_semantics",
    "autocast_cast_policy",
    "native_batch_norm_no_stats_out",
    "forward_ad_inference_copy",
    "nested_select_backward",
    "sparse_unsafe_valid_constructor",
    "privateuse1_attention_public_sdpa",
    "quantized_flash_attention_public_sdpa",
    "privateuse1_matmul_backward_formula",
    "privateuse1_resize_output_property",
    "privateuse1_batch_norm_forward_formula",
    "privateuse1_thnn_cell_formula",
    "privateuse1_pin_memory_noop",
})


def direct_validation_class(oracle_id: str, coverage_status: str) -> str:
    """Return the primary QA class for one direct registry disposition."""

    if coverage_status.startswith("pending_") or coverage_status.startswith("excluded_"):
        return V6_DISPOSITION
    if oracle_id in _DIRECT_VALUE_IDS:
        return V1_FIXED_VALUE
    if oracle_id in _DIRECT_PROPERTY_IDS or coverage_status == "covered_property":
        return V4_PROPERTY
    if coverage_status in {"covered_backend_pack", "covered_oracle"}:
        return V5_BACKEND_SEMANTIC
    raise ValueError(
        f"No direct oracle validation class for {oracle_id!r} with status {coverage_status!r}"
    )


ALL_CASE_PACKS = frozenset(
    set(REFERENCE_FUNCTION_CASE_PACKS.values())
    | set(HIGH_PRECISION_FUNCTION_CASE_PACKS.values())
    | set(BACKWARD_FUNCTION_CASE_PACKS.values())
    | set(FFT_FUNCTION_CASE_PACKS.values())
    | set(FORWARD_REFERENCE_CASE_PACKS.values())
    | set(BACKWARD_REFERENCE_CASE_PACKS.values())
    | set(NON_UNIQUE_CASE_PACKS.values())
    | set(DIRECT_ORACLE_CASE_PACKS.values())
)
