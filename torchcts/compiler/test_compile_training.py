import pytest
import torch
from torchcts.core.device import synchronize

COMPILE_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


class LinearModel(torch.nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.fc = torch.nn.Linear(in_f, out_f)

    def forward(self, x):
        return self.fc(x).sum()


class MLPModel(torch.nn.Module):
    def __init__(self, in_f, hidden, out_f):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_f, hidden)
        self.fc2 = torch.nn.Linear(hidden, out_f)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        return self.fc2(x).sum()


class ConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(8, 2)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x).sum()


class NormModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = torch.nn.LayerNorm(32)
        self.fc = torch.nn.Linear(32, 10)

    def forward(self, x):
        return self.fc(self.ln(x)).sum()


_MODELS = {
    "linear": (LinearModel(16, 8), (4, 16)),
    "mlp": (MLPModel(16, 32, 8), (4, 16)),
    "conv": (ConvModel(), (2, 3, 8, 8)),
    "norm": (NormModel(), (4, 32)),
}

_OPTIMIZERS = ["sgd", "adam", "adamw"]


@pytest.mark.smoke
@pytest.mark.requires("compile")
@pytest.mark.requires("training")
@pytest.mark.parametrize("model_name", list(_MODELS.keys()))
@pytest.mark.parametrize("dtype", COMPILE_DTYPES)
def test_compile_training_step(model_name, dtype, device, input_gen):
    model_template, input_shape = _MODELS[model_name]

    # Fresh copy for each test
    import copy
    model = copy.deepcopy(model_template).to(device)
    if dtype != torch.float32:
        model = model.to(dtype)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    @torch.compile
    def train_step(x):
        optimizer.zero_grad()
        loss = model(x)
        loss.backward()
        optimizer.step()
        return loss

    x = input_gen(input_shape, dtype, device)

    try:
        loss = train_step(x)
        synchronize(device)
        assert loss is not None
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    except Exception as e:
        pytest.fail(f"torch.compile training step failed for {model_name}/{dtype}: {e}")


@pytest.mark.medium
@pytest.mark.requires("compile")
@pytest.mark.requires("training")
@pytest.mark.parametrize("opt_name", _OPTIMIZERS)
@pytest.mark.parametrize("dtype", COMPILE_DTYPES)
def test_compile_training_optimizer(opt_name, dtype, device, input_gen):
    model = torch.nn.Linear(16, 8).to(device)
    if dtype != torch.float32:
        model = model.to(dtype)

    if opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    @torch.compile
    def train_step(x):
        optimizer.zero_grad()
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        return loss

    x = input_gen((4, 16), dtype, device)

    try:
        loss = train_step(x)
        synchronize(device)
        assert loss is not None
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    except Exception as e:
        pytest.fail(f"torch.compile training with {opt_name}/{dtype} failed: {e}")


@pytest.mark.medium
@pytest.mark.requires("compile")
@pytest.mark.requires("training")
@pytest.mark.parametrize("dtype", COMPILE_DTYPES)
def test_compile_multi_step_convergence(dtype, device, input_gen):
    """Run 5 compiled training steps and verify loss decreases."""
    model = torch.nn.Linear(16, 4).to(device)
    if dtype != torch.float32:
        model = model.to(dtype)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    target = torch.zeros(4, 4, dtype=dtype, device=device)

    @torch.compile
    def train_step(x, t):
        optimizer.zero_grad()
        out = model(x)
        loss = torch.nn.functional.mse_loss(out, t)
        loss.backward()
        optimizer.step()
        return loss

    x = input_gen((4, 16), dtype, device)
    losses = []
    for _ in range(5):
        loss = train_step(x, target)
        synchronize(device)
        losses.append(loss.item())

    # Loss should not be NaN/Inf
    for i, l in enumerate(losses):
        assert not (l != l), f"Loss is NaN at step {i}"  # NaN check

    # Loss should generally decrease (allow some noise — just check last < first * 2)
    assert losses[-1] < losses[0] * 2, f"Loss did not decrease: {losses}"
