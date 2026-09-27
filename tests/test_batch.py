import importlib

import jax.numpy as jnp
import numpy as np
import pytest

import branch_extended_AD as bead
import branch_extended_AD.numpy as bnp

ht = importlib.import_module("branch_extended_AD.HashTensor")

def test_replay_value_and_grad_batch_matches_loop_max_min():
    def f_max(x):
        return bnp.max(x)

    def f_min(x):
        return bnp.min(x)

    z = jnp.array([2.0, 4.0, 6.0])

    for f in (f_max, f_min):
        xs = [
            jnp.array([1.0, 3.0, 2.0]),
            jnp.array([5.0, 1.0, 5.0]),
            jnp.array([0.0, -1.0, 2.0]),
        ]
        paths = [list(bead.record(f, atol=0.0)(x)[1])[0] for x in xs]

        values, grads = bead.replay_value_and_grad_batch(f, paths)(z)
        for k, path in enumerate(paths):
            manual_v, manual_g = bead.replay_value_and_grad(f, path)(z)
            assert jnp.allclose(values[k], manual_v)
            assert jnp.allclose(grads[k], manual_g)


def test_replay_value_and_grad_batch_matches_loop_abs():
    def f(x):
        return bnp.sum(bnp.abs(x))

    xs = [
        jnp.array([1.0, -0.5, 0.005, 2.0, -0.005]),
        jnp.array([-3.0, 0.5, 0.5, -0.001, 4.0]),
        jnp.array([2.0, 2.0, 2.0, 2.0, 2.0]),
    ]
    paths = [list(bead.record(f, atol=0.01)(x)[1])[0] for x in xs]
    z = jnp.array([1.0, -2.0, 0.5, 3.0, -0.2])

    values, grads = bead.replay_value_and_grad_batch(f, paths)(z)
    for k, path in enumerate(paths):
        manual_v, manual_g = bead.replay_value_and_grad(f, path)(z)
        assert jnp.allclose(values[k], manual_v)
        assert jnp.allclose(grads[k], manual_g)


def test_replay_value_and_grad_batch_matches_loop_maximum_minimum():
    def f(x, y):
        return bnp.sum(bnp.maximum(x, y))

    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.05, 1.95, 3.05])
    _, path_set = bead.record(f, atol=0.1)(x, y)
    paths = list(path_set)
    assert len(paths) > 1

    z1 = jnp.array([2.0, -1.0, 0.5])
    z2 = jnp.array([1.9, -1.5, 0.6])

    values, grads = bead.replay_value_and_grad_batch(f, paths, argnums=(0, 1))(z1, z2)
    for k, path in enumerate(paths):
        manual_v, manual_g = bead.replay_value_and_grad(f, path, argnums=(0, 1))(z1, z2)
        assert jnp.allclose(values[k], manual_v)
        assert jnp.allclose(grads[0][k], manual_g[0])
        assert jnp.allclose(grads[1][k], manual_g[1])


def test_replay_value_and_grad_batch_large_J_no_blowup():
    def f(x):
        return bnp.max(x)

    n = 44
    x = jnp.ones(n) * 5.0
    _, path_set = bead.record(f, atol=0.0)(x)
    paths = list(path_set)
    assert len(paths) == n

    z = jnp.arange(n, dtype=jnp.float32)
    values, grads = bead.replay_value_and_grad_batch(f, paths)(z)

    assert values.shape == (n,)
    assert grads.shape == (n, n)


def test_replay_value_and_grad_batch_padding_matches_unpadded():
    def f(x):
        return bnp.max(x)

    n = 10
    z = jnp.arange(n, dtype=jnp.float32)

    for J in (3, 5, 6, 7):
        xs = [jnp.asarray(np.random.RandomState(i).rand(n)) for i in range(J)]
        paths = [list(bead.record(f, atol=0.0)(x)[1])[0] for x in xs]

        values, grads = bead.replay_value_and_grad_batch(f, paths)(z)
        assert values.shape == (J,)
        assert grads.shape == (J, n)

        for k, path in enumerate(paths):
            manual_v, manual_g = bead.replay_value_and_grad(f, path)(z)
            assert jnp.allclose(values[k], manual_v)
            assert jnp.allclose(grads[k], manual_g)


def test_replay_value_and_grad_batch_mismatched_paths_raises():
    def f_max(x):
        return bnp.max(x)

    def f_maximum(y, z):
        return bnp.sum(bnp.maximum(y, z))

    x = jnp.array([1.0, 3.0, 2.0])
    _, max_paths = bead.record(f_max, atol=0.0)(x)
    max_path = list(max_paths)[0]

    y = jnp.array([1.0, 3.0])
    z = jnp.array([1.0, 3.0])
    _, maximum_paths = bead.record(f_maximum, atol=0.5)(y, z)
    maximum_path = list(maximum_paths)[0]

    with pytest.raises(ValueError):
        bead.replay_value_and_grad_batch(f_max, [max_path, maximum_path])(x)


def test_replay_value_and_grad_batch_rejects_empty_paths_and_kwargs():
    def function(x):
        return bnp.max(x)

    x = jnp.array([1.0, 2.0])
    with pytest.raises(ValueError, match="at least one path"):
        bead.replay_value_and_grad_batch(function, [])

    path = bead.record(function)(x)[1][0]
    with pytest.raises(TypeError, match="does not support kwargs"):
        bead.replay_value_and_grad_batch(function, [path])(x=x)


def test_replay_value_and_grad_batch_has_aux_matches_loop():
    def function(x):
        value = bnp.max(x)
        return value, {"double": 2 * value}

    x = jnp.array([1.0, 1.0, 0.0])
    _, path_set = bead.record(function)(x)
    paths = list(path_set)

    (values, aux), gradients = bead.replay_value_and_grad_batch(
        function,
        paths,
        has_aux=True,
    )(x)
    assert values.shape == (2,)
    assert aux["double"].shape == (2,)
    assert gradients.shape == (2, 3)

    for index, path in enumerate(paths):
        (manual_value, manual_aux), manual_gradient = bead.replay_value_and_grad(
            function,
            path,
            has_aux=True,
        )(x)
        assert jnp.allclose(values[index], manual_value)
        assert jnp.allclose(aux["double"][index], manual_aux["double"])
        assert jnp.allclose(gradients[index], manual_gradient)


def test_batched_replay_reuses_jitted_wrapper(monkeypatch):
    def function(x):
        return bnp.max(x)

    x = jnp.ones(4)
    paths = list(bead.record(function)(x)[1])
    ht._jit_batch_cache.clear()
    jit_calls = 0
    original_jit = ht.jax.jit

    def counting_jit(*args, **kwargs):
        nonlocal jit_calls
        jit_calls += 1
        return original_jit(*args, **kwargs)

    monkeypatch.setattr(ht.jax, "jit", counting_jit)
    bead.replay_value_and_grad_batch(function, paths[:3])(x)
    cached = next(iter(ht._jit_batch_cache.values()))
    bead.replay_value_and_grad_batch(function, paths)(x)

    assert jit_calls == 1
    assert next(iter(ht._jit_batch_cache.values())) is cached
    assert hasattr(cached, "lower")

    names = [node.name for node in paths[0]]
    leaves, layout = ht._build_batch_leaves(paths, names)
    padded, _ = ht._pad_batch_leaves(leaves, len(paths))
    cached.lower(x, *padded).compile()
