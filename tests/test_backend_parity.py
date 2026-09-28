import jax.numpy as jnp
import pytest

import branch_extended_AD as jax_bead
import branch_extended_AD.numpy as jax_numpy


torch = pytest.importorskip("torch")
from branch_extended_AD import pytorch as torch_bead
import branch_extended_AD.torch_numpy as torch_numpy


def gradient_set(results):
    return {
        tuple(float(item) for item in gradient.reshape(-1))
        for _, gradient in results
    }


@pytest.mark.parametrize(
    ("jax_function", "torch_function", "values", "expected_paths"),
    [
        (jax_numpy.max, torch_numpy.max, [1.0, 1.0, 0.0], 2),
        (
            lambda value: jax_numpy.sum(jax_numpy.maximum(value, 0.0)),
            lambda value: torch_numpy.sum(torch_numpy.maximum(value, 0.0)),
            [0.0, 0.0],
            4,
        ),
        (
            lambda value: jax_numpy.sum(jax_numpy.abs(value)),
            lambda value: torch_numpy.sum(torch_numpy.abs(value)),
            [0.0, -2.0],
            2,
        ),
    ],
)
def test_jax_and_pytorch_branch_gradients_match(
    jax_function,
    torch_function,
    values,
    expected_paths,
):
    jax_results, jax_paths = jax_bead.all_value_and_grad(
        jax_function,
        abs_policy="enumerate",
    )(jnp.asarray(values))
    torch_results, torch_paths = torch_bead.all_value_and_grad(
        torch_function,
        abs_policy="enumerate",
    )(torch.tensor(values, dtype=torch.float64))

    assert len(jax_paths) == len(torch_paths) == expected_paths
    assert gradient_set(jax_results) == gradient_set(torch_results)
