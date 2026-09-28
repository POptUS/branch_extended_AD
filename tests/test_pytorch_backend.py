import subprocess
import sys

import pytest


torch = pytest.importorskip("torch")

from branch_extended_AD import pytorch as bead
import branch_extended_AD.torch_numpy as bnp


def tensor(values):
    return torch.tensor(values, dtype=torch.float64)


def test_record_and_replay_elementwise_maximum():
    left = tensor([1.0, 2.0, 2.5])
    right = tensor([0.95, 1.5, 3.0])

    def function(x, y):
        return bnp.maximum(x, y)

    value, paths = bead.record(function, atol=0.1)(left, right)
    assert len(paths) == 2
    assert torch.allclose(value, tensor([1.0, 2.0, 3.0]))
    replayed = [bead.replay(function, path)(left, right) for path in paths]
    assert sorted(float(result[0]) for result in replayed) == [0.95, 1.0]


def test_minimum_sum_and_scalar_operands():
    x = tensor([[1.0, -2.0], [0.0, 3.0]])

    def function(value):
        clipped = bnp.minimum(bnp.maximum(value, -1.0), 2.0)
        return bnp.sum(clipped)

    (value, gradient), paths = bead.value_and_grad(function)(x)
    assert value == 2.0
    assert gradient.shape == x.shape
    assert torch.allclose(gradient, tensor([[1.0, 0.0], [1.0, 0.0]]))
    assert len(paths) == 1

    reversed_value = bead.replay(
        lambda value: bnp.maximum(0.0, value),
        bead.record(lambda value: bnp.maximum(0.0, value))(x)[1][0],
    )(x)
    assert torch.allclose(reversed_value, torch.maximum(torch.zeros_like(x), x))


def test_grad_and_value_and_grad():
    x = tensor([1.0, 3.0, 2.0])

    def function(value):
        return bnp.max(value)

    gradient, paths = bead.grad(function)(x)
    assert len(paths) == 1
    assert torch.allclose(gradient, tensor([0.0, 1.0, 0.0]))

    (value, gradient), paths = bead.value_and_grad(function)(x)
    assert value == 3.0
    assert torch.allclose(gradient, tensor([0.0, 1.0, 0.0]))


def test_min_reduction_records_multidimensional_ties():
    x = tensor([[1.0, -2.0], [-2.0, 3.0]])
    results, paths = bead.all_value_and_grad(lambda value: bnp.min(value))(x)
    assert len(paths) == 2
    gradients = {tuple(gradient.reshape(-1).tolist()) for _, gradient in results}
    assert gradients == {
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
    }

def test_all_value_and_grad_and_batch_replay():
    x = tensor([1.0, 1.0, 0.0])

    def function(value):
        return bnp.max(value)

    results, paths = bead.all_value_and_grad(function)(x)
    assert len(results) == len(paths) == 2
    gradients = {tuple(gradient.tolist()) for _, gradient in results}
    assert gradients == {(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)}

    values, batch_gradients = bead.replay_value_and_grad_batch(function, paths)(x)
    assert values.shape == (2,)
    assert batch_gradients.shape == (2, 3)
    assert {tuple(gradient.tolist()) for gradient in batch_gradients} == gradients


def test_replay_grad_and_value_and_grad():
    x = tensor([1.0, 1.0, 0.0])

    def function(value):
        return bnp.max(value)

    _, paths = bead.record(function)(x)
    for path in paths:
        gradient = bead.replay_grad(function, path)(x)
        value, value_gradient = bead.replay_value_and_grad(function, path)(x)
        assert value == 1.0
        assert torch.allclose(gradient, value_gradient)


def test_abs_policies():
    x = tensor([0.0, -2.0, 3.0])

    def function(value):
        return bnp.sum(bnp.abs(value))

    gradient, paths = bead.grad(function)(x)
    assert len(paths) == 1
    assert torch.allclose(gradient, tensor([0.0, -1.0, 1.0]))

    results, paths = bead.all_value_and_grad(function, abs_policy="enumerate")(x)
    assert len(paths) == 2
    gradients = {tuple(gradient.tolist()) for _, gradient in results}
    assert gradients == {(1.0, -1.0, 1.0), (-1.0, -1.0, 1.0)}


def test_relative_tolerance():
    x = tensor([100.0, 105.0, 1.0])
    _, paths = bead.record(lambda value: bnp.max(value), rtol=0.1)(x)
    assert len(paths) == 2


def test_argnums_and_auxiliary_output():
    x = tensor([1.0, 2.0])
    y = tensor([3.0, 1.0])

    def function(left, right):
        value = bnp.max(left + right)
        return value, {"sum": left + right}

    ((value, aux), gradients), paths = bead.value_and_grad(
        function,
        argnums=(0, 1),
        has_aux=True,
    )(x, y)
    assert value == 4.0
    assert torch.allclose(aux["sum"], tensor([4.0, 3.0]))
    assert len(gradients) == 2
    assert torch.allclose(gradients[0], tensor([1.0, 0.0]))
    assert torch.allclose(gradients[1], tensor([1.0, 0.0]))
    assert len(paths) == 1


def test_single_nonzero_argnum():
    x = tensor([1.0, 2.0])
    y = tensor([3.0, 1.0])
    gradient, _ = bead.grad(
        lambda left, right: bnp.max(left + right),
        argnums=1,
    )(x, y)
    assert torch.allclose(gradient, tensor([1.0, 0.0]))


def test_batch_replay_with_auxiliary_output():
    x = tensor([1.0, 1.0])

    def function(value):
        result = bnp.max(value)
        return result, {"double": 2 * result}

    _, paths = bead.record(function)(x)
    (values, aux), gradients = bead.replay_value_and_grad_batch(
        function,
        paths,
        has_aux=True,
    )(x)
    assert values.shape == (2,)
    assert aux["double"].shape == (2,)
    assert gradients.shape == (2, 2)


def test_batch_replay_rejects_mismatched_paths():
    x = tensor([0.0, 0.0])
    max_path = bead.record(lambda value: bnp.max(value))(x)[1][0]
    abs_path = bead.record(lambda value: bnp.sum(bnp.abs(value)))(x)[1][0]
    with pytest.raises(ValueError, match="same branch operations"):
        bead.replay_value_and_grad_batch(
            lambda value: bnp.max(value),
            [max_path, abs_path],
        )


def test_input_scaled_mode_is_explicitly_unsupported():
    with pytest.raises(NotImplementedError, match="input_scaled"):
        bead.record(lambda value: bnp.max(value), tol_mode="input_scaled")


def test_invalid_abs_policy():
    with pytest.raises(ValueError, match="abs_policy"):
        bead.record(lambda value: bnp.abs(value), abs_policy="invalid")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gradient_preserves_dtype_and_device(dtype):
    x = torch.tensor([1.0, 3.0, 2.0], dtype=dtype)
    gradient, _ = bead.grad(lambda value: bnp.max(value))(x)
    assert gradient.dtype == dtype
    assert gradient.device == x.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_gradient_preserves_cuda_device():
    x = torch.tensor([1.0, 3.0, 2.0], device="cuda")
    gradient, _ = bead.grad(lambda value: bnp.max(value))(x)
    assert gradient.device == x.device


def test_pytorch_backend_import_does_not_require_jax():
    code = r'''
import importlib.abc
import sys

class BlockJax(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "jax" or fullname.startswith("jax."):
            raise ModuleNotFoundError("JAX import blocked", name=fullname)
        return None

sys.meta_path.insert(0, BlockJax())
from branch_extended_AD import pytorch as bead
import branch_extended_AD.torch_numpy as bnp
import torch
x = torch.tensor([1.0, 2.0])
gradient, _ = bead.grad(lambda value: bnp.max(value))(x)
assert torch.equal(gradient, torch.tensor([0.0, 1.0]))
'''
    subprocess.run([sys.executable, "-c", code], check=True)
