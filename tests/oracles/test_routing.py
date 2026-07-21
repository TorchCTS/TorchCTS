from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import torchcts.core.contract_references as routes


def _sample(input_value, *args, **kwargs):
    return SimpleNamespace(input=input_value, args=args, kwargs=kwargs)


@dataclass(frozen=True)
class RouteCall:
    resolver: str
    name: str
    sample: object
    dtype: torch.dtype
    condition: str


def _opinfo(name, sample, dtype, condition):
    return RouteCall("opinfo", name, sample, dtype, condition)


def _generated(name, sample, dtype, condition):
    return RouteCall("generated", name, sample, dtype, condition)


def _routing_cases():
    real = torch.tensor([1.0, 2.0], dtype=torch.float32)
    real_half = real.to(torch.float16)
    integer = torch.tensor([1, 2], dtype=torch.int64)
    complex_value = torch.tensor([1.0 + 2.0j, 2.0 - 1.0j], dtype=torch.complex64)
    complex_other = torch.tensor([2.0 - 1.0j, 1.0 + 0.0j], dtype=torch.complex64)
    complex_exponent = torch.tensor([2.0 + 0.0j, 3.0 + 0.0j], dtype=torch.complex64)
    nonintegral_exponent = torch.tensor([2.5 + 0.0j, 3.0 + 1.0j], dtype=torch.complex64)
    complex_matrix = complex_value.reshape(1, 2)
    complex_weight = complex_other.reshape(2, 1)
    grid_input = real_half.reshape(1, 1, 1, 2)
    grid = torch.zeros((1, 1, 1, 2), dtype=torch.float16)
    conv_input_half = real_half.reshape(1, 1, 2)
    conv_weight_half = torch.ones((1, 1, 1), dtype=torch.float16)
    conv_input_complex = complex_value.reshape(1, 1, 2)
    conv_weight_complex = torch.ones((1, 1, 1), dtype=torch.complex64)

    return [
        (
            "addbmm_f32_opmath",
            _opinfo("addbmm", _sample(real_half, real_half, real_half), torch.float16, "clean"),
            _opinfo("addbmm", _sample(real_half, real_half, real_half), torch.float16, "has_inf"),
        ),
        (
            "bce_with_logits_stable",
            _opinfo(
                "nn.functional.binary_cross_entropy_with_logits",
                _sample(real, torch.zeros_like(real)),
                torch.float32,
                "has_inf",
            ),
            _opinfo(
                "nn.functional.binary_cross_entropy_with_logits",
                _sample(real, torch.zeros_like(real)),
                torch.float32,
                "clean",
            ),
        ),
        (
            "complex_convolution_four_real",
            _opinfo(
                "nn.functional.conv1d",
                _sample(conv_input_complex, conv_weight_complex),
                torch.complex64,
                "has_inf",
            ),
            _opinfo(
                "nn.functional.conv1d",
                _sample(conv_input_complex, conv_weight_complex),
                torch.complex64,
                "clean",
            ),
        ),
        (
            "complex_covariance_real_diagonal",
            _opinfo("cov", _sample(complex_matrix), torch.complex64, "has_inf"),
            _opinfo("cov", _sample(complex_matrix), torch.complex64, "clean"),
        ),
        (
            "complex_cumprod_inclusive",
            _opinfo("cumprod", _sample(complex_value, 0), torch.complex64, "has_inf"),
            _opinfo("cumprod", _sample(complex_value, 0), torch.complex64, "clean"),
        ),
        (
            "complex_expm1_lane_aware",
            _opinfo("expm1", _sample(complex_value), torch.complex64, "has_inf"),
            _opinfo("expm1", _sample(complex_value), torch.complex64, "clean"),
        ),
        (
            "complex_gradient_real_spacing",
            _opinfo("gradient", _sample(complex_value), torch.complex64, "has_inf"),
            _opinfo("gradient", _sample(complex_value), torch.complex64, "clean"),
        ),
        (
            "complex_l1_loss",
            _opinfo(
                "nn.functional.l1_loss",
                _sample(complex_value, complex_other),
                torch.complex64,
                "has_inf",
            ),
            _opinfo(
                "nn.functional.l1_loss",
                _sample(complex_value, complex_other),
                torch.complex64,
                "clean",
            ),
        ),
        (
            "complex_ldexp_integral_lane_scale",
            _generated(
                "aten::ldexp.Tensor",
                _sample(complex_value, integer),
                torch.complex64,
                "has_inf",
            ),
            _generated(
                "aten::ldexp.Tensor",
                _sample(complex_value, integer.to(torch.bool)),
                torch.complex64,
                "has_inf",
            ),
        ),
        (
            "complex_ldexp_phase_scale",
            _opinfo("ldexp", _sample(complex_value, complex_other), torch.complex64, "has_inf"),
            _opinfo("ldexp", _sample(complex_value, complex_other), torch.complex64, "clean"),
        ),
        (
            "complex_log2",
            _opinfo("log2", _sample(complex_value), torch.complex64, "has_inf"),
            _opinfo("log2", _sample(complex_value), torch.complex64, "clean"),
        ),
        (
            "complex_matmul_wide_semantic",
            _opinfo(
                "matmul",
                _sample(complex_matrix, complex_weight),
                torch.complex64,
                "clean",
            ),
            _opinfo(
                "matmul",
                _sample(complex_matrix, complex_weight),
                torch.complex64,
                "has_inf",
            ),
        ),
        (
            "complex_rsqrt_c99",
            _opinfo("rsqrt", _sample(complex_value), torch.complex64, "has_inf"),
            _opinfo("rsqrt", _sample(complex_value), torch.complex64, "clean"),
        ),
        (
            "complex_sigmoid_stable",
            _opinfo("sigmoid", _sample(complex_value), torch.complex64, "has_inf"),
            _opinfo("sigmoid", _sample(complex_value), torch.complex64, "clean"),
        ),
        (
            "complex_tensor_integer_power",
            _opinfo(
                "pow",
                _sample(complex_value, complex_exponent),
                torch.complex64,
                "has_inf",
            ),
            _opinfo(
                "pow",
                _sample(complex_value, nonintegral_exponent),
                torch.complex64,
                "has_inf",
            ),
        ),
        (
            "complex_unit_alpha_add_sub",
            _opinfo("add", _sample(complex_value, complex_other), torch.complex64, "has_inf"),
            _opinfo(
                "add",
                _sample(complex_value, complex_other, alpha=2),
                torch.complex64,
                "has_inf",
            ),
        ),
        (
            "conv_transpose_f32_opmath",
            _opinfo(
                "nn.functional.conv_transpose1d",
                _sample(conv_input_half, conv_weight_half),
                torch.float16,
                "clean",
            ),
            _opinfo(
                "nn.functional.conv_transpose1d",
                _sample(conv_input_half, conv_weight_half),
                torch.float16,
                "has_inf",
            ),
        ),
        (
            "dirichlet_grad_mpmath",
            _generated("aten::_dirichlet_grad", _sample(real, real, real), torch.float32, "clean"),
            _generated("aten::_dirichlet_grad", _sample(real, real), torch.float32, "clean"),
        ),
        (
            "float_to_uint8_signed_truncate",
            _generated("aten::_cast_Byte", _sample(real), torch.float32, "clean"),
            _generated("aten::_cast_Byte", _sample(integer), torch.int64, "clean"),
        ),
        (
            "foreach_complex_c99_compound",
            _generated(
                "aten::_foreach_addcmul.Scalar",
                _sample([complex_value], [complex_other], [complex_other], value=1),
                torch.complex64,
                "has_inf",
            ),
            _generated(
                "aten::_foreach_addcmul.Scalar",
                _sample([complex_value], [complex_other], [complex_other], value=1),
                torch.complex64,
                "clean",
            ),
        ),
        (
            "foreach_complex_c99_tensor_mul",
            _generated(
                "aten::_foreach_mul.Tensor",
                _sample([complex_value], complex_other),
                torch.complex64,
                "has_inf",
            ),
            _generated(
                "aten::_foreach_mul.Tensor",
                _sample([complex_value], complex_other),
                torch.complex64,
                "clean",
            ),
        ),
        (
            "foreach_complex_real_alpha_add_sub",
            _generated(
                "aten::_foreach_add.List",
                _sample([complex_value], [complex_other], alpha=1),
                torch.complex64,
                "has_inf",
            ),
            _generated(
                "aten::_foreach_add.List",
                _sample([complex_value], [complex_other], alpha=1),
                torch.complex64,
                "clean",
            ),
        ),
        (
            "grid_sampler_f32",
            _opinfo(
                "nn.functional.grid_sample",
                _sample(grid_input, grid),
                torch.float16,
                "clean",
            ),
            _opinfo(
                "nn.functional.grid_sample",
                _sample(grid_input, grid),
                torch.float16,
                "has_inf",
            ),
        ),
        (
            "i0_mpmath",
            _opinfo("i0", _sample(real), torch.float32, "clean"),
            _opinfo("cos", _sample(real), torch.float32, "clean"),
        ),
        (
            "laguerre_initialized_recurrence",
            _opinfo("special.laguerre_polynomial_l", _sample(real, integer), torch.float32, "has_inf"),
            _opinfo("special.laguerre_polynomial_l", _sample(real, integer), torch.float32, "clean"),
        ),
        (
            "lanczos3_exact_zero",
            _generated(
                "aten::_upsample_lanczos2d_aa",
                _sample(real.reshape(1, 1, 1, 2), [2, 2], False),
                torch.float32,
                "has_inf",
            ),
            _generated(
                "aten::_upsample_lanczos2d_aa",
                _sample(real.reshape(1, 1, 1, 2), [2, 2], False),
                torch.float32,
                "clean",
            ),
        ),
        (
            "matrix_exp_f64_scaling_squaring",
            _opinfo("matrix_exp", _sample(torch.eye(2)), torch.float32, "clean"),
            _opinfo("matrix_exp", _sample(torch.eye(2)), torch.float32, "has_inf"),
        ),
        (
            "polygamma_mpmath_poles",
            _opinfo("polygamma", _sample(real, 1), torch.float32, "clean"),
            _opinfo("polygamma", _sample(real, 2), torch.float32, "clean"),
        ),
        (
            "shifted_chebyshev_initialized_recurrence",
            _opinfo(
                "special.shifted_chebyshev_polynomial_t",
                _sample(real, integer),
                torch.float32,
                "has_inf",
            ),
            _opinfo(
                "special.shifted_chebyshev_polynomial_t",
                _sample(real, integer),
                torch.float32,
                "clean",
            ),
        ),
        (
            "soft_margin_f32_opmath",
            _opinfo(
                "nn.functional.soft_margin_loss",
                _sample(real_half, torch.ones_like(real_half)),
                torch.float16,
                "clean",
            ),
            _opinfo(
                "nn.functional.soft_margin_loss",
                _sample(real_half, torch.ones_like(real_half)),
                torch.float16,
                "has_inf",
            ),
        ),
        (
            "standard_gamma_grad_mpmath",
            _generated("aten::_standard_gamma_grad", _sample(real, real), torch.float32, "clean"),
            _generated("aten::_standard_gamma_grad", _sample(real), torch.float32, "clean"),
        ),
    ]


def _resolve(call: RouteCall):
    if call.resolver == "opinfo":
        return routes.resolve_opinfo_forward_reference(
            call.name, call.sample, call.dtype, call.condition
        )
    return routes.resolve_generated_forward_reference(
        call.name, call.sample, call.dtype, call.condition
    )


@pytest.mark.oracle_contract(id="forward-routing", validation_class="V3_ROUTING")
@pytest.mark.parametrize(
    "reference_id,positive,negative",
    _routing_cases(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_every_forward_reference_id_has_a_positive_and_adjacent_negative_route(
    monkeypatch, reference_id, positive, negative
):
    monkeypatch.setattr(
        routes,
        "_build",
        lambda routed_id, category, _builder: routes.ContractReference(
            value=None,
            reference_id=routed_id,
            category=category,
        ),
    )

    matched = _resolve(positive)
    assert matched is not None
    assert matched.reference_id == reference_id
    assert _resolve(negative) is None
