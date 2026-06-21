import pytest
import torch
from torchcts.core.device import synchronize

COMPILE_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# Elementwise ops that take a single tensor
_UNARY_OPS = ["relu", "gelu", "silu", "tanh", "sigmoid", "abs", "neg", "exp", "log", "sin", "cos"]

# Binary ops that take two tensors of the same shape
_BINARY_OPS = ["add", "sub", "mul", "div"]

# Reduction ops
_REDUCTION_OPS = ["sum", "mean", "amax", "amin"]

@pytest.mark.smoke
@pytest.mark.requires("compile")
@pytest.mark.parametrize("op_name", _UNARY_OPS)
@pytest.mark.parametrize("dtype", COMPILE_DTYPES)
def test_compile_unary_op(op_name, dtype, device, compare, input_gen):
    op_fn = getattr(torch, op_name, None) or getattr(torch.nn.functional, op_name, None)
    if op_fn is None:
        pytest.fail(f"Op {op_name} not found")

    shape = (32, 32)
    x = input_gen(shape, dtype, device)

    def func(a):
        return op_fn(a)

    compiled_func = torch.compile(func, fullgraph=True)

    expected = func(x)
    actual = compiled_func(x)
    synchronize(device)

    compare(actual, expected, category="compile", dtype=dtype)


@pytest.mark.smoke
@pytest.mark.requires("compile")
@pytest.mark.parametrize("op_name", _BINARY_OPS)
@pytest.mark.parametrize("dtype", COMPILE_DTYPES)
def test_compile_binary_op(op_name, dtype, device, compare, input_gen):
    op_fn = getattr(torch, op_name)

    shape = (32, 32)
    x = input_gen(shape, dtype, device)
    y = input_gen(shape, dtype, device)

    def func(a, b):
        return op_fn(a, b)

    compiled_func = torch.compile(func, fullgraph=True)

    expected = func(x, y)
    actual = compiled_func(x, y)
    synchronize(device)

    compare(actual, expected, category="compile", dtype=dtype)


@pytest.mark.smoke
@pytest.mark.requires("compile")
@pytest.mark.parametrize("op_name", _REDUCTION_OPS)
@pytest.mark.parametrize("dtype", COMPILE_DTYPES)
def test_compile_reduction_op(op_name, dtype, device, compare, input_gen):
    op_fn = getattr(torch, op_name)

    shape = (32, 32)
    x = input_gen(shape, dtype, device)

    def func(a):
        return op_fn(a)

    compiled_func = torch.compile(func, fullgraph=True)

    expected = func(x)
    actual = compiled_func(x)
    synchronize(device)

    compare(actual, expected, category="compile", dtype=dtype)


@pytest.mark.smoke
@pytest.mark.requires("compile")
@pytest.mark.parametrize("dtype", COMPILE_DTYPES)
def test_compile_matmul(dtype, device, compare, input_gen):
    shape = (32, 32)
    x = input_gen(shape, dtype, device)
    y = input_gen(shape, dtype, device)

    def func(a, b):
        return torch.mm(a, b)

    compiled_func = torch.compile(func, fullgraph=True)

    expected = func(x, y)
    actual = compiled_func(x, y)
    synchronize(device)

    compare(actual, expected, category="compile", dtype=dtype)


@pytest.mark.smoke
@pytest.mark.requires("compile")
@pytest.mark.parametrize("dtype", COMPILE_DTYPES)
def test_compile_softmax(dtype, device, compare, input_gen):
    shape = (16, 32)
    x = input_gen(shape, dtype, device)

    def func(a):
        return torch.nn.functional.softmax(a, dim=-1)

    compiled_func = torch.compile(func, fullgraph=True)

    expected = func(x)
    actual = compiled_func(x)
    synchronize(device)

    compare(actual, expected, category="compile", dtype=dtype)


@pytest.mark.smoke
@pytest.mark.requires("compile")
@pytest.mark.parametrize("dtype", COMPILE_DTYPES)
def test_compile_layer_norm(dtype, device, compare, input_gen):
    shape = (4, 8, 32)
    x = input_gen(shape, dtype, device)

    ln = torch.nn.LayerNorm(32).to(device)
    if dtype == torch.float16 or dtype == torch.bfloat16:
        ln = ln.to(dtype)

    def func(a):
        return ln(a)

    compiled_func = torch.compile(func, fullgraph=True)

    expected = func(x)
    actual = compiled_func(x)
    synchronize(device)

    compare(actual, expected, category="compile", dtype=dtype)


@pytest.mark.smoke
@pytest.mark.requires("compile")
@pytest.mark.parametrize("dtype", COMPILE_DTYPES)
def test_compile_conv2d(dtype, device, compare, input_gen):
    x = input_gen((2, 3, 16, 16), dtype, device)
    conv = torch.nn.Conv2d(3, 8, 3, padding=1).to(device)
    if dtype == torch.float16 or dtype == torch.bfloat16:
        conv = conv.to(dtype)

    def func(a):
        return conv(a)

    compiled_func = torch.compile(func, fullgraph=True)

    expected = func(x)
    actual = compiled_func(x)
    synchronize(device)

    compare(actual, expected, category="compile", dtype=dtype)


@pytest.mark.medium
@pytest.mark.requires("compile")
@pytest.mark.parametrize("dtype", COMPILE_DTYPES)
def test_compile_chained_ops(dtype, device, compare, input_gen):
    shape = (32, 32)
    x = input_gen(shape, dtype, device)
    y = input_gen(shape, dtype, device)

    def func(a, b):
        c = torch.mm(a, b)
        c = torch.nn.functional.relu(c)
        c = c + a
        c = torch.nn.functional.gelu(c)
        return c.sum()

    compiled_func = torch.compile(func, fullgraph=True)

    expected = func(x, y)
    actual = compiled_func(x, y)
    synchronize(device)

    compare(actual, expected, category="compile", dtype=dtype)
