import jax.numpy as jnp
import pytest

import branch_extended_AD as bead
import branch_extended_AD.numpy as bnp


@pytest.mark.parametrize(
    ("operation", "jax_operation", "arguments"),
    [
        (bnp.max, jnp.max, (jnp.array([1.0, 3.0, 2.0]),)),
        (bnp.min, jnp.min, (jnp.array([1.0, 3.0, 2.0]),)),
        (bnp.sum, jnp.sum, (jnp.array([1.0, 3.0, 2.0]),)),
        (bnp.abs, jnp.abs, (jnp.array([-1.0, 2.0]),)),
        (bnp.maximum, jnp.maximum, (jnp.array([1.0, 3.0]), jnp.array([2.0, 2.0]))),
        (bnp.minimum, jnp.minimum, (jnp.array([1.0, 3.0]), jnp.array([2.0, 2.0]))),
    ],
)
def test_overrides_match_jax_outside_transform(operation, jax_operation, arguments):
    assert jnp.allclose(operation(*arguments), jax_operation(*arguments))


def test_non_overridden_array_operations_match_jax():
    x = jnp.array([1.0, 2.0, 3.0])
    assert jnp.allclose(bnp.sin(x), jnp.sin(x))
    assert jnp.allclose(bnp.exp(x), jnp.exp(x))
    assert bnp.array([1, 2, 3]).dtype == jnp.array([1, 2, 3]).dtype


@pytest.mark.parametrize("operation", [bnp.max, bnp.min, bnp.sum])
def test_reduction_signatures_are_deliberately_narrow(operation):
    with pytest.raises(TypeError):
        operation(jnp.ones((2, 2)), axis=0)


def test_array_operations_compose_with_branch_primitives():
    def function(x):
        reshaped = bnp.reshape(x, (2, 2))
        transformed = bnp.sin(reshaped) + bnp.exp(reshaped) * 0.1
        return bnp.max(transformed)

    x = jnp.array([0.0, 0.5, 1.0, 1.5])
    expected = jnp.max(jnp.sin(x.reshape(2, 2)) + 0.1 * jnp.exp(x.reshape(2, 2)))
    (value, gradient), paths = bead.value_and_grad(function)(x)
    assert jnp.allclose(value, expected)
    assert gradient.shape == x.shape
    assert len(paths) == 1
