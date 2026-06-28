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

import warnings

import pytest
import torch
from torchcts.core.device import synchronize


@pytest.mark.smoke
@pytest.mark.requires("rng")
@pytest.mark.covers("aten::randn")
@pytest.mark.parametrize("seed", [42, 1234])
def test_rng_reproducibility(seed, device, manifest):
    # 1. Same manual seed -> identical outputs
    torch.manual_seed(seed)
    # Note: torch.randn with generator or global seed
    x1 = torch.randn(10, 10, device=device)
    
    torch.manual_seed(seed)
    x2 = torch.randn(10, 10, device=device)
    synchronize(device)
    
    assert torch.equal(x1.cpu(), x2.cpu()), "manual_seed did not yield reproducible random outputs"


@pytest.mark.smoke
@pytest.mark.requires("rng")
@pytest.mark.covers("aten::rand")
def test_rand_factory_reproducibility(device, manifest):
    torch.manual_seed(42)
    a = torch.rand((8, 8), device=device)
    torch.manual_seed(42)
    b = torch.rand((8, 8), device=device)
    synchronize(device)

    a_cpu = a.cpu()
    assert torch.equal(a_cpu, b.cpu()), "torch.rand() not reproducible with same seed"
    assert bool((a_cpu >= 0).all() and (a_cpu < 1).all())


@pytest.mark.smoke
@pytest.mark.requires("device_generator")
@pytest.mark.covers("aten::rand.generator")
def test_rand_generator_factory_reproducibility(device, manifest):
    g1 = torch.Generator(device=device)
    g1.manual_seed(42)
    a = torch.rand((8, 8), device=device, generator=g1)

    g2 = torch.Generator(device=device)
    g2.manual_seed(42)
    b = torch.rand((8, 8), device=device, generator=g2)
    synchronize(device)

    assert torch.equal(a.cpu(), b.cpu()), "torch.rand(generator=...) not reproducible"


@pytest.mark.smoke
@pytest.mark.requires("rng")
@pytest.mark.requires("named_tensor")
@pytest.mark.covers("aten::rand.names")
def test_rand_named_factory_metadata(device, manifest):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        x = torch.rand((2, 3), names=("batch", "feature"), device=device)

    synchronize(device)
    assert x.device.type == device
    assert x.names == ("batch", "feature")


@pytest.mark.smoke
@pytest.mark.requires("device_generator")
@pytest.mark.requires("named_tensor")
@pytest.mark.covers("aten::rand.generator_with_names")
def test_rand_generator_named_factory_metadata(device, manifest):
    g = torch.Generator(device=device)
    g.manual_seed(42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        x = torch.rand((2, 3), names=("batch", "feature"), device=device, generator=g)

    synchronize(device)
    assert x.device.type == device
    assert x.names == ("batch", "feature")


@pytest.mark.smoke
@pytest.mark.requires("device_generator")
@pytest.mark.covers("aten::randn.generator")
@pytest.mark.parametrize("seed", [123, 456])
def test_rng_generator_seeding(seed, device, manifest):
    # 2. Per-Generator seeding
    g1 = torch.Generator(device=device)
    g1.manual_seed(seed)
    y1 = torch.randn(5, 5, device=device, generator=g1)

    g2 = torch.Generator(device=device)
    g2.manual_seed(seed)
    y2 = torch.randn(5, 5, device=device, generator=g2)
    synchronize(device)

    assert torch.equal(y1.cpu(), y2.cpu()), "per-generator manual_seed was not reproducible"


@pytest.mark.smoke
@pytest.mark.requires("rng")
@pytest.mark.covers("aten::randn")
def test_rng_sequential_calls_differ(device, manifest):
    """Two consecutive randn calls with the same seed context must produce different tensors."""
    torch.manual_seed(42)
    a = torch.randn(100, device=device).cpu()
    b = torch.randn(100, device=device).cpu()
    assert not torch.equal(a, b), "Sequential randn calls returned identical tensors"


@pytest.mark.smoke
@pytest.mark.requires("rng")
@pytest.mark.covers("aten::randperm")
def test_randperm_factory(device, manifest):
    perm = torch.randperm(32, device=device)
    synchronize(device)

    assert perm.device.type == device
    assert perm.dtype == torch.int64
    assert torch.equal(torch.sort(perm.cpu()).values, torch.arange(32, dtype=torch.int64))


@pytest.mark.smoke
@pytest.mark.requires("device_generator")
@pytest.mark.covers("aten::randperm.generator")
def test_randperm_generator_factory_reproducibility(device, manifest):
    g1 = torch.Generator(device=device)
    g1.manual_seed(123)
    a = torch.randperm(32, device=device, generator=g1)

    g2 = torch.Generator(device=device)
    g2.manual_seed(123)
    b = torch.randperm(32, device=device, generator=g2)
    synchronize(device)

    assert torch.equal(a.cpu(), b.cpu()), "torch.randperm(generator=...) not reproducible"


@pytest.mark.smoke
@pytest.mark.requires("rng")
@pytest.mark.covers("aten::random")
@pytest.mark.covers("aten::random.to")
@pytest.mark.covers("aten::random.from")
def test_random_inplace_overloads_reproducibility_and_range(device, manifest):
    cases = [
        ((), None, None),
        ((7,), 0, 7),
        ((2, 11), 2, 11),
    ]
    for args, low, high in cases:
        torch.manual_seed(42)
        a = torch.empty(256, dtype=torch.float32, device=device).random_(*args)
        torch.manual_seed(42)
        b = torch.empty(256, dtype=torch.float32, device=device).random_(*args)
        synchronize(device)

        a_cpu = a.cpu()
        assert torch.equal(a_cpu, b.cpu()), f"random_{args} not reproducible"
        assert torch.equal(a_cpu, a_cpu.floor()), f"random_{args} produced non-integral values"
        if low is not None and high is not None:
            assert bool((a_cpu >= low).all() and (a_cpu < high).all())


@pytest.mark.smoke
@pytest.mark.requires("rng")
@pytest.mark.covers("aten::uniform_")
def test_uniform_reproducibility(device, manifest):
    """uniform_() must be reproducible across manual_seed resets."""
    torch.manual_seed(42)
    a = torch.empty(100, device=device).uniform_().cpu()
    torch.manual_seed(42)
    b = torch.empty(100, device=device).uniform_().cpu()
    assert torch.allclose(a, b), "uniform_() not reproducible with same seed"


@pytest.mark.smoke
@pytest.mark.requires("rng")
@pytest.mark.covers("aten::bernoulli_.float")
def test_bernoulli_reproducibility(device, manifest):
    """bernoulli_() must be reproducible across manual_seed resets."""
    torch.manual_seed(42)
    a = torch.empty(100, device=device).bernoulli_(0.5).cpu()
    torch.manual_seed(42)
    b = torch.empty(100, device=device).bernoulli_(0.5).cpu()
    assert torch.equal(a, b), "bernoulli_() not reproducible with same seed"


@pytest.mark.smoke
@pytest.mark.requires("rng")
@pytest.mark.covers("aten::normal.Tensor_Tensor")
@pytest.mark.parametrize("seed", [42, 1234])
def test_normal_reproducibility(seed, device, manifest):
    mean = torch.zeros(100, device=device)
    std = torch.ones(100, device=device)
    torch.manual_seed(seed)
    a = torch.normal(mean, std).cpu()
    torch.manual_seed(seed)
    b = torch.normal(mean, std).cpu()
    assert torch.equal(a, b), "torch.normal() not reproducible with same seed"


@pytest.mark.smoke
@pytest.mark.requires("rng")
@pytest.mark.covers("aten::normal_functional")
def test_normal_functional_reproducibility(device, manifest):
    base = torch.empty(128, dtype=torch.float32, device=device)
    torch.manual_seed(42)
    a = torch.ops.aten.normal_functional.default(base, mean=2.0, std=0.5)
    torch.manual_seed(42)
    b = torch.ops.aten.normal_functional.default(base, mean=2.0, std=0.5)
    synchronize(device)

    assert torch.equal(a.cpu(), b.cpu()), "aten::normal_functional not reproducible"
    assert a.shape == base.shape
    assert a.dtype == base.dtype


@pytest.mark.smoke
@pytest.mark.requires("rng")
@pytest.mark.covers("aten::multinomial")
@pytest.mark.parametrize("seed", [42, 1234])
def test_multinomial_reproducibility(seed, device, manifest):
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4], device=device)
    torch.manual_seed(seed)
    a = torch.multinomial(probs, num_samples=50, replacement=True).cpu()
    torch.manual_seed(seed)
    b = torch.multinomial(probs, num_samples=50, replacement=True).cpu()
    assert torch.equal(a, b), "torch.multinomial() not reproducible with same seed"


@pytest.mark.smoke
@pytest.mark.requires("rng")
@pytest.mark.covers("aten::poisson")
def test_poisson_reproducibility(device, manifest):
    rates = torch.full((256,), 2.0, dtype=torch.float32, device=device)
    torch.manual_seed(42)
    a = torch.poisson(rates)
    torch.manual_seed(42)
    b = torch.poisson(rates)
    synchronize(device)

    a_cpu = a.cpu()
    assert torch.equal(a_cpu, b.cpu()), "torch.poisson() not reproducible"
    assert bool((a_cpu >= 0).all())
    assert torch.equal(a_cpu, a_cpu.floor()), "torch.poisson() produced non-integral samples"


@pytest.mark.smoke
@pytest.mark.requires("rng")
@pytest.mark.covers("aten::dropout")
@pytest.mark.covers("aten::native_dropout")
@pytest.mark.covers("aten::alpha_dropout")
@pytest.mark.covers("aten::feature_dropout")
@pytest.mark.covers("aten::feature_alpha_dropout")
def test_dropout_dispatch_surfaces_reproducibility(device, manifest):
    x = torch.ones((4, 3, 8), dtype=torch.float32, device=device)

    torch.manual_seed(42)
    dropout_a = torch.ops.aten.dropout.default(x, 0.25, True)
    torch.manual_seed(42)
    dropout_b = torch.ops.aten.dropout.default(x, 0.25, True)

    torch.manual_seed(42)
    native_a, mask_a = torch.ops.aten.native_dropout.default(x, 0.25, True)
    torch.manual_seed(42)
    native_b, mask_b = torch.ops.aten.native_dropout.default(x, 0.25, True)

    torch.manual_seed(42)
    alpha_a = torch.ops.aten.alpha_dropout.default(x, 0.25, True)
    torch.manual_seed(42)
    alpha_b = torch.ops.aten.alpha_dropout.default(x, 0.25, True)

    torch.manual_seed(42)
    feature_a = torch.ops.aten.feature_dropout.default(x, 0.25, True)
    torch.manual_seed(42)
    feature_b = torch.ops.aten.feature_dropout.default(x, 0.25, True)

    torch.manual_seed(42)
    feature_alpha_a = torch.ops.aten.feature_alpha_dropout.default(x, 0.25, True)
    torch.manual_seed(42)
    feature_alpha_b = torch.ops.aten.feature_alpha_dropout.default(x, 0.25, True)
    synchronize(device)

    assert torch.equal(dropout_a.cpu(), dropout_b.cpu())
    assert torch.equal(native_a.cpu(), native_b.cpu())
    assert torch.equal(mask_a.cpu(), mask_b.cpu())
    assert mask_a.dtype == torch.bool
    assert torch.equal(alpha_a.cpu(), alpha_b.cpu())
    assert torch.equal(feature_a.cpu(), feature_b.cpu())
    assert torch.equal(feature_alpha_a.cpu(), feature_alpha_b.cpu())
    assert torch.equal(torch.ops.aten.dropout.default(x, 0.25, False).cpu(), x.cpu())


@pytest.mark.medium
@pytest.mark.requires("rng_distributions")
@pytest.mark.covers("aten::uniform_")
def test_uniform_distribution_properties(device, manifest):
    torch.manual_seed(42)
    x = torch.empty(10000, device=device).uniform_().cpu().float()
    assert x.min() >= 0.0, f"uniform_() produced value below 0: {x.min()}"
    assert x.max() <= 1.0, f"uniform_() produced value above 1: {x.max()}"
    assert abs(x.mean().item() - 0.5) < 0.05, f"uniform_() mean {x.mean().item()} not ≈ 0.5"
    expected_std = 1.0 / (12 ** 0.5)
    assert abs(x.std().item() - expected_std) < 0.05, (
        f"uniform_() std {x.std().item()} not ≈ {expected_std}"
    )


@pytest.mark.medium
@pytest.mark.requires("rng_distributions")
@pytest.mark.covers("aten::randn")
def test_normal_distribution_properties(device, manifest):
    torch.manual_seed(42)
    x = torch.randn(10000, device=device).cpu().float()
    assert abs(x.mean().item()) < 0.1, f"randn mean {x.mean().item()} not ≈ 0.0"
    assert abs(x.std().item() - 1.0) < 0.1, f"randn std {x.std().item()} not ≈ 1.0"


@pytest.mark.medium
@pytest.mark.requires("rng_distributions")
@pytest.mark.covers("aten::bernoulli_.float")
def test_bernoulli_distribution_properties(device, manifest):
    p = 0.3
    torch.manual_seed(42)
    x = torch.empty(10000, device=device).bernoulli_(p).cpu().float()
    freq = x.mean().item()
    assert abs(freq - p) < 0.05, f"bernoulli_({p}) frequency {freq} not ≈ {p}"


@pytest.mark.medium
@pytest.mark.requires("rng_distributions")
@pytest.mark.covers("aten::multinomial")
def test_multinomial_distribution_properties(device, manifest):
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4], device=device)
    torch.manual_seed(42)
    samples = torch.multinomial(probs, num_samples=10000, replacement=True).cpu()
    for i, expected_p in enumerate(probs.cpu().tolist()):
        empirical_p = (samples == i).float().mean().item()
        assert abs(empirical_p - expected_p) < 0.05, (
            f"multinomial category {i}: empirical {empirical_p} vs expected {expected_p}"
        )
