import importlib
from itertools import product

import jax.numpy as jnp
import numpy as np
import pytest

import branch_extended_AD as bead
import branch_extended_AD.numpy as bnp

ht = importlib.import_module("branch_extended_AD.HashTensor")


@pytest.mark.parametrize("operation", [bnp.maximum, bnp.minimum])
def test_elementwise_minmax_records_every_nearby_choice(operation):
    left = jnp.array([1.0, 2.0, 2.5])
    separated = jnp.array([0.5, 1.5, 3.0])
    one_nearby = jnp.array([0.95, 1.5, 3.0]) if operation is bnp.maximum else jnp.array([1.05, 1.5, 3.0])
    all_nearby = jnp.array([0.95, 1.95, 2.45]) if operation is bnp.maximum else jnp.array([1.05, 2.05, 2.55])

    def function(x, y):
        return operation(x, y)

    value, paths = bead.record(function, atol=0.1)(left, separated)
    assert len(paths) == 1
    assert jnp.allclose(value, operation(left, separated))
    assert jnp.allclose(bead.replay(function, paths[0])(left, separated), value)

    _, paths = bead.record(function, atol=0.1)(left, one_nearby)
    assert len(paths) == 2
    expected = {
        tuple(operation(left, one_nearby).at[0].set(candidate).tolist())
        for candidate in (left[0], one_nearby[0])
    }
    assert {tuple(bead.replay(function, path)(left, one_nearby).tolist()) for path in paths} == expected

    _, paths = bead.record(function, atol=0.1)(left, all_nearby)
    assert len(paths) == 8
    expected = {
        tuple(float(value) for value in values)
        for values in product(*[(left[index], all_nearby[index]) for index in range(left.size)])
    }
    assert {tuple(bead.replay(function, path)(left, all_nearby).tolist()) for path in paths} == expected


def test_abs():
    def f(x):
        return bnp.abs(x)

    x = jnp.array([-1.0, -2.0, 2.5])
    val, paths = bead.record(f, atol=0.1)(x)
    assert jnp.allclose(val, jnp.array([1.0, 2.0, 2.5]))
    assert len(paths) == 1

    replayed = bead.replay(f, paths[0])(x)
    assert jnp.allclose(replayed, jnp.array([1.0, 2.0, 2.5]))

    x2 = jnp.array([0.05, -2.0, 2.5])
    val, paths = bead.record(f, atol=0.1)(x2)
    assert val[0] == 0.05
    assert jnp.allclose(val[1:], jnp.array([2.0, 2.5]))
    assert len(paths) == 1

    replayed = bead.replay(f, paths[0])(x2)
    assert jnp.allclose(replayed, jnp.array([0.05, 2.0, 2.5]))

    x3 = jnp.array([0.05, -0.05, 0.03])
    val, paths = bead.record(f, atol=0.1)(x3)
    assert len(paths) == 1
    assert jnp.allclose(val, jnp.array([0.05, 0.05, 0.03]))

    replayed = bead.replay(f, paths[0])(x3)
    assert jnp.allclose(replayed, jnp.array([0.05, 0.05, 0.03]))


def test_abs_ambiguous_gradient_is_zero():
    def f(x):
        return bnp.sum(bnp.abs(x))

    x = jnp.array([0.05, -2.0, 2.5])
    g, paths = bead.grad(f, atol=0.1)(x)
    assert len(paths) == 1
    assert jnp.allclose(g, jnp.array([0.0, -1.0, 1.0]))


def test_abs_ambiguity_policy_can_enumerate_limiting_gradients():
    def f(x):
        return bnp.sum(bnp.abs(x))

    x = jnp.zeros((2, 2))
    results, paths = bead.all_value_and_grad(f, abs_policy="enumerate")(x)
    assert len(paths) == 16
    gradients = {tuple(np.asarray(gradient).ravel()) for _, gradient in results}
    assert gradients == set(product((-1.0, 1.0), repeat=x.size))
    assert all(value == 0.0 for value, _ in results)

    batch_values, batch_gradients = bead.replay_value_and_grad_batch(f, list(paths))(x)
    assert batch_values.shape == (16,)
    assert batch_gradients.shape == (16, 2, 2)
    assert {tuple(np.asarray(gradient).ravel()) for gradient in batch_gradients} == gradients


def test_abs_ambiguity_policy_defaults_to_zero_gradient():
    def f(x):
        return bnp.sum(bnp.abs(x))

    gradient, paths = bead.grad(f)(jnp.zeros(2))
    assert len(paths) == 1
    assert jnp.allclose(gradient, jnp.zeros(2))


def test_invalid_abs_ambiguity_policy():
    with pytest.raises(ValueError, match="abs_policy"):
        bead.record(lambda x: x, abs_policy="invalid")


def test_multidimensional_primitives_record_replay_and_gradients():
    x = jnp.array([[1.0, 3.0], [3.0, -2.0]])

    def reduce_max(x):
        return bnp.max(x)

    def reduce_min(x):
        return bnp.min(x)

    for function, expected_value, expected_gradients in (
        (
            reduce_max,
            3.0,
            [
                jnp.array([[0.0, 1.0], [0.0, 0.0]]),
                jnp.array([[0.0, 0.0], [1.0, 0.0]]),
            ],
        ),
        (reduce_min, -2.0, [jnp.array([[0.0, 0.0], [0.0, 1.0]])]),
    ):
        results, paths = bead.all_value_and_grad(function)(x)
        assert len(paths) == len(expected_gradients)
        assert all(jnp.isclose(value, expected_value) for value, _ in results)
        assert all(
            any(jnp.allclose(gradient, expected) for expected in expected_gradients)
            for _, gradient in results
        )

    y = jnp.array([[1.0, 2.0], [4.0, -3.0]])

    def elementwise(x):
        return bnp.sum(bnp.minimum(bnp.maximum(x, y), 3.0))

    _, paths = bead.record(elementwise, atol=1.0)(x)
    for path in paths:
        value, gradient = bead.replay_value_and_grad(elementwise, path)(x)
        assert jnp.ndim(value) == 0
        assert gradient.shape == x.shape

    def absolute(x):
        return bnp.sum(bnp.abs(x))

    zero_matrix = jnp.zeros((2, 2))
    value, paths = bead.record(absolute)(zero_matrix)
    assert value == 0.0
    assert len(paths.trace[-1].choices[0][0]) == zero_matrix.size
    _, gradient = bead.replay_value_and_grad(absolute, paths[0])(zero_matrix)
    assert gradient.shape == zero_matrix.shape


def test_abs_replay_at_different_point():
    def f(x):
        return bnp.abs(x)

    x = jnp.array([0.05, -2.0, 2.5])
    _, paths = bead.record(f, atol=0.1)(x)
    path = paths[0]

    x_prime = jnp.array([0.03, 2.0, 2.5])
    replayed = bead.replay(f, path)(x_prime)
    assert jnp.allclose(replayed, jnp.array([0.03, -2.0, 2.5]))

    def f_sum(x):
        return bnp.sum(bnp.abs(x))

    _, sum_paths = bead.record(f_sum, atol=0.1)(x)
    sum_path = sum_paths[0]
    v, g = bead.replay_value_and_grad(f_sum, sum_path)(x_prime)
    assert jnp.allclose(v, 0.03 - 2.0 + 2.5)
    assert jnp.allclose(g, jnp.array([0.0, -1.0, 1.0]))
