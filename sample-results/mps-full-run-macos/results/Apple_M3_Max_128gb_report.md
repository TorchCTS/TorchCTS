# Validator Scorecard for mps

```
  Backend asserts via manifest.py that it supports:
  15703 out of 19210 tests / 81.74%
  [██████████████████████████████░░░░░░░]

============================================================
  Backend: mps        | Hardware: Apple_M3_Max_128gb
  PyTorch: 2.12.1     | Run: 2026-07-09T17:21:31.188410Z
  Duration: 2:13:51
============================================================

  OPERATOR COVERAGE
  ─────────────────
  Op categories overlap; an op can pass one case and skip another.
  OpInfo ops represented:    625
  Ops with PASS coverage:    611  (97.8%)
  Ops with FAIL/ERROR:       219  (35.0%)
  Ops with not-run cases (manifest): 608  (97.3%)
  Ops with not-run cases (selection): 0    (0.0%)
  Ops with not-run cases (coverage): 0    (0.0%)
  Ops with not-run cases (CPU contract): 1    (0.2%)
  Ops with not-run cases (backend pack): 0    (0.0%)
  Ops with not-run cases (known unsafe): 1    (0.2%)
  Ops with not-run cases (runtime): 0    (0.0%)

  CAPABILITY RESULTS
  ──────────────────
  ❌  inference       5490/5844 passed
  ❌  training        1626/1714 passed
  ✅  serialization   3/3 passed
  ❌  rng             17/18 passed
  ❌  device_generator 4/5 passed
  ✅  rng_distributions 4/4 passed
  ✅  double_backward 12/12 passed
  ⬚  gradcheck       DECLINED
  ✅  gradient_checkpointing 3/3 passed
  ✅  autocast        10/10 passed
  ✅  fused_optimizer 13/13 passed
  ⬚  dataloader      DECLINED
  ✅  module_hooks    3/3 passed
  ❌  channels_last   86/87 passed
  ❌  sparse          22/52 passed
  ❌  nested          8/35 passed
  ❌  named_tensor    0/5 passed
  ✅  foreach         3/3 passed
  ❌  fp8             3/7 passed
  ⬚  quantized_container_plumbing 0/0 passed
  ❌  native_quantization 4/15 passed
  ⬚  custom_quantized_decode DECLINED
  ❌  compile         138/144 passed
  ⬚  pinned_memory   0/0 passed
  ❌  streams         1/3 passed
  ❌  events          1/3 passed
  ✅  deterministic   6/6 passed
  ⬚  guard_alloc     0/0 passed
  ✅  device_api      6/6 passed
  ⬚  multi_device    DECLINED
  ⬚  ieee754         0/0 passed

  DTYPE COVERAGE
  ──────────────
  bfloat16   2048/2184 ❌ skip=4    bool       405/420 ❌ skip=1
  complex128 0/0 ⬚ declined=1297    complex32  434/477 ❌ skip=4
  complex64  1117/1444 ❌ skip=4    float16    2046/2178 ❌ skip=4
  float32    3286/3486 ❌ skip=4    float64    0/0 ⬚ declined=2166
  int16      539/557 ❌ skip=1    int32      556/574 ❌ skip=1
  int64      600/625 ❌ skip=1    int8       539/557 ❌ skip=1
  uint16     6/35 ❌    uint32     6/35 ❌
  uint64     6/34 ❌    uint8      541/559 ❌ skip=1

  SEMANTIC LEVELS
  ───────────────
  requested <= 8
  L1  pass=290  fail=129  deselected=6    total=425
  L2  pass=9765 fail=960  deselected=3608 total=14333
  L3  pass=668  fail=142  deselected=375  total=1185
  L4  pass=1704 fail=144  deselected=403  total=2251
  L5  pass=1137 fail=53   deselected=86   total=1276
  L6  pass=173  fail=16   deselected=14   total=203
  L7  pass=174  fail=26   deselected=13   total=213
  L8  pass=27   fail=0    deselected=3    total=30

  IEEE 754 COMPLIANCE
  ───────────────────
  ❌  NaN/Inf propagation  3298/3753 passed

  QUALITY WARNINGS: 268 tests passed at usable tolerance but failed golden tier

  FAILURE RECORDS (1470)
  ────────────
  test_workload_path_shape_case[attention_sdpa_sdpa_sq17_sk17_d128_bool_causal_f32_standard] float32   ERROR: Known confirmed backend crash was record
  test_workload_path_shape_case[attention_sdpa_sdpa_sq17_sk17_d32_bool_causal_f32_standard] float32   ERROR: Known confirmed backend crash was record
  test_workload_path_shape_case[attention_sdpa_sdpa_sq17_sk17_d64_bool_causal_f32_standard] float32   ERROR: Known confirmed backend crash was record
  test_workload_path_shape_case[attention_sdpa_sdpa_sq17_sk17_d80_bool_causal_f32_standard] float32   ERROR: Known confirmed backend crash was record
  test_workload_path_shape_case[attention_sdpa_sdpa_sq17_sk17_d96_bool_causal_f32_standard] float32   ERROR: Known confirmed backend crash was record
  test_workload_path_shape_case[attention_sdpa_sdpa_sq8_sk8_d128_bool_causal_f32_standard] float32   ERROR: Known confirmed backend crash was record
  test_workload_path_shape_case[attention_sdpa_sdpa_sq8_sk8_d32_bool_causal_f32_standard] float32   ERROR: Known confirmed backend crash was record
  test_workload_path_shape_case[attention_sdpa_sdpa_sq8_sk8_d64_bool_causal_f32_standard] float32   ERROR: Known confirmed backend crash was record
  test_workload_path_shape_case[attention_sdpa_sdpa_sq8_sk8_d80_bool_causal_f32_standard] float32   ERROR: Known confirmed backend crash was record
  test_workload_path_shape_case[attention_sdpa_sdpa_sq8_sk8_d96_bool_causal_f32_standard] float32   ERROR: Known confirmed backend crash was record
  aten::_grid_sampler_2d_cpu_fallback unknown   ERROR: Known confirmed backend crash was record
  aten::_grid_sampler_2d_cpu_fallback.out unknown   ERROR: Known confirmed backend crash was record
  test_internal_dispatcher_surface[grid2_cpu_fallback_backward] unknown   ERROR: Known confirmed backend crash was record
  test_internal_dispatcher_surface[grid2_cpu_fallback_default] unknown   ERROR: Known confirmed backend crash was record
  test_internal_dispatcher_surface[grid2_cpu_fallback_out] unknown   ERROR: Known confirmed backend crash was record
  aten::range.out_       unknown   ERROR: Known confirmed backend crash was record
  py::test_direct_sparse_and_embedding_backward_dispatcher_surfaces unknown   ERROR: Could not run 'new_compressed_tensor' fr
    ↳ Hint: Operator not registered for the target backend device.
  py::test_compile_training_optimizer float32   ERROR: torch.compile training with adam/torch.f
  py::test_compile_training_optimizer float32   ERROR: torch.compile training with adamw/torch.
  py::test_compile_training_optimizer float16   ERROR: torch.compile training with adam/torch.f
  ... and 1450 more failure records

```

Full per-test records and verbatim diagnostics are available through `Apple_M3_Max_128gb_latest.json`.