import importlib

import jax.numpy as jnp
import pytest

import branch_extended_AD as bead
import branch_extended_AD.numpy as bnp

ht = importlib.import_module("branch_extended_AD.HashTensor")


def test_record_and_replay():
    x1 = jnp.array([1.0, 2.0, 2.5])
    x2 = jnp.array([0.95, 1.5, 3.0])

    def f(x, y):
        return bnp.maximum(x, y)

    val, paths = bead.record(f, atol=0.1)(x1, x2)
    assert isinstance(paths, bead.PathSet)
    assert len(paths) == 2

    results = []
    for p in paths:
        assert isinstance(p, list)
        replayed_val = bead.replay(f, p)(x1, x2)
        results.append(replayed_val)

    assert len(results) == 2
    first_vals = sorted([float(r[0]) for r in results])
    assert jnp.allclose(first_vals[0], 0.95, atol=1e-6)
    assert jnp.allclose(first_vals[1], 1.0, atol=1e-6)
    for r in results:
        assert jnp.allclose(r[1], 2.0)
        assert jnp.allclose(r[2], 3.0)


def test_record_no_tolerance():
    x = jnp.array([1.0, 5.0, 10.0])

    def f(x):
        return bnp.max(x)

    val, paths = bead.record(f, atol=0.0)(x)
    assert len(paths) == 1
    assert float(val) == 10.0


def test_grad_simple():
    x = jnp.array([1.0, 3.0, 2.0])

    def f(x):
        return bnp.max(x)

    g, paths = bead.grad(f, atol=0.0)(x)
    assert g.shape == x.shape
    assert jnp.allclose(g, jnp.array([0.0, 1.0, 0.0])), f"Expected [0, 1, 0], got {g}"
    assert len(paths) == 1


def test_grad_with_tolerance():
    x = jnp.array([1.0, 1.05, 0.5])

    def f(x):
        return bnp.max(x)

    g, paths = bead.grad(f, atol=0.1)(x)
    assert len(paths) == 2
    assert g.shape == x.shape


def test_value_and_grad_simple():
    x = jnp.array([1.0, 3.0, 2.0])

    def f(x):
        return bnp.sum(x)

    (val, g), paths = bead.value_and_grad(f, atol=0.0)(x)
    assert jnp.allclose(val, 6.0)
    assert jnp.allclose(g, jnp.array([1.0, 1.0, 1.0])), f"Expected [1, 1, 1], got {g}"
    assert len(paths) == 1


def test_value_and_grad_maximum():
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.5, 1.5, 1.5])

    def f(x, y):
        return bnp.sum(bnp.maximum(x, y))

    (val, g), paths = bead.value_and_grad(f, argnums=0, atol=0.0)(x, y)
    assert float(val) == 1.5 + 2.0 + 3.0
    assert jnp.allclose(g, jnp.array([0.0, 1.0, 1.0])), f"Expected [0, 1, 1], got {g}"


def test_replay_grad():
    x = jnp.array([1.0, 1.05, 0.5])

    def f(x):
        return bnp.max(x)

    _, paths = bead.record(f, atol=0.1)(x)
    assert len(paths) >= 2

    grads = []
    for p in paths:
        g = bead.replay_grad(f, p)(x)
        grads.append(g)

    grad_tuples = set(tuple(float(v) for v in g) for g in grads)
    assert (0.0, 1.0, 0.0) in grad_tuples or (1.0, 0.0, 0.0) in grad_tuples


def test_replay_value_and_grad():
    x = jnp.array([1.0, 2.0, 2.5])
    y = jnp.array([0.95, 1.5, 3.0])

    def f(x, y):
        return bnp.sum(bnp.maximum(x, y))

    _, paths = bead.record(f, atol=0.1)(x, y)

    for p in paths:
        replayed_val = bead.replay(f, p)(x, y)
        v, g = bead.replay_value_and_grad(f, p, argnums=0)(x, y)
        assert jnp.allclose(v, replayed_val), f"value_and_grad value {v} != replay value {replayed_val}"
        assert g.shape == x.shape


def test_replay_accepts_singleton_path_set():
    def function(x):
        return bnp.max(x)

    x = jnp.array([1.0, 2.0])
    value, paths = bead.record(function)(x)
    assert bead.replay(function, paths)(x) == value


def test_replay_rejects_ambiguous_or_invalid_paths():
    def function(x):
        return bnp.max(x)

    x = jnp.array([1.0, 1.0])
    _, paths = bead.record(function)(x)
    with pytest.raises(ValueError, match="multiple paths"):
        bead.replay(function, paths)(x)
    with pytest.raises(TypeError, match="Unexpected replay_path type"):
        bead.replay(function, "invalid")(x)
    with pytest.raises(ValueError, match="Path exhausted"):
        bead.replay(function, [])(x)

    wrong = [ht._TraceNode("min", [0])]
    with pytest.raises(ValueError, match="Expected trace node max"):
        bead.replay(function, wrong)(x)


def test_grad_argnums():
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.5, 1.5, 1.5])

    def f(x, y):
        return bnp.sum(bnp.maximum(x, y))

    g_x, _ = bead.grad(f, argnums=0, atol=0.0)(x, y)
    assert jnp.allclose(g_x, jnp.array([0.0, 1.0, 1.0]))

    g_y, _ = bead.grad(f, argnums=1, atol=0.0)(x, y)
    assert jnp.allclose(g_y, jnp.array([1.0, 0.0, 0.0]))

    (g_x2, g_y2), _ = bead.grad(f, argnums=(0, 1), atol=0.0)(x, y)
    assert jnp.allclose(g_x2, g_x)
    assert jnp.allclose(g_y2, g_y)


def test_has_aux():
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.5, 1.5, 1.5])

    def f_aux(x, y):
        result = bnp.maximum(x, y)
        s = bnp.sum(result)
        aux = {"max_val": bnp.max(result)}
        return s, aux

    (g, aux), paths = bead.grad(f_aux, argnums=0, atol=0.0, has_aux=True)(x, y)
    assert "max_val" in aux
    assert g.shape == x.shape
    assert jnp.allclose(g, jnp.array([0.0, 1.0, 1.0]))

    ((val, aux2), g2), paths2 = bead.value_and_grad(f_aux, argnums=0, atol=0.0, has_aux=True)(x, y)
    assert float(val) == 1.5 + 2.0 + 3.0
    assert jnp.allclose(g2, g)
    assert "max_val" in aux2


def test_replay_grad_has_aux():
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.05, 1.95, 2.95])

    def f_aux(x, y):
        result = bnp.maximum(x, y)
        s = bnp.sum(result)
        return s, {"count": len(x)}

    _, paths = bead.record(f_aux, atol=0.1)(x, y)

    for p in paths:
        g, aux = bead.replay_grad(f_aux, p, argnums=0, has_aux=True)(x, y)
        assert g.shape == x.shape
        assert aux["count"] == 3

        (v, aux2), g2 = bead.replay_value_and_grad(f_aux, p, argnums=0, has_aux=True)(x, y)
        assert jnp.allclose(g, g2)
        assert aux2["count"] == 3


def test_all_value_and_grad_simple():
    x = jnp.array([1.0, 3.0, 2.0])

    def f(x):
        return bnp.max(x)

    results, paths = bead.all_value_and_grad(f, atol=0.0)(x)
    assert len(results) == 1
    assert len(paths) == 1
    val, grad = results[0]
    assert jnp.allclose(val, 3.0)
    assert jnp.allclose(grad, jnp.array([0.0, 1.0, 0.0]))


def test_all_value_and_grad_with_tolerance():
    x = jnp.array([1.0, 1.05, 0.5])

    def f(x):
        return bnp.max(x)

    results, paths = bead.all_value_and_grad(f, atol=0.1)(x)
    assert len(paths) == 2
    assert len(results) == 2

    values = set()
    grad_tuples = set()
    for val, grad in results:
        values.add(float(val))
        grad_tuples.add(tuple(float(v) for v in grad))

    assert (0.0, 1.0, 0.0) in grad_tuples
    assert (1.0, 0.0, 0.0) in grad_tuples


def test_all_value_and_grad_matches_manual():
    x = jnp.array([1.0, 2.0, 2.5])
    y = jnp.array([0.95, 1.5, 3.0])

    def f(x, y):
        return bnp.sum(bnp.maximum(x, y))

    results, paths = bead.all_value_and_grad(f, argnums=0, atol=0.1)(x, y)

    for i, path in enumerate(paths):
        manual_v, manual_g = bead.replay_value_and_grad(f, path, argnums=0)(x, y)
        api_v, api_g = results[i]
        assert jnp.allclose(api_v, manual_v)
        assert jnp.allclose(api_g, manual_g)


def test_all_value_and_grad_has_aux():
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.05, 1.95, 2.95])

    def f_aux(x, y):
        result = bnp.maximum(x, y)
        s = bnp.sum(result)
        return s, {"count": len(x)}

    results, paths = bead.all_value_and_grad(f_aux, argnums=0, atol=0.1, has_aux=True)(x, y)
    assert len(results) == len(paths)

    for (val, aux), grad in results:
        assert aux["count"] == 3
        assert grad.shape == x.shape


def test_all_value_and_grad_argnums():
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.5, 1.5, 1.5])

    def f(x, y):
        return bnp.sum(bnp.maximum(x, y))

    results_x, _ = bead.all_value_and_grad(f, argnums=0, atol=0.0)(x, y)
    assert len(results_x) == 1
    val, g_x = results_x[0]
    assert jnp.allclose(g_x, jnp.array([0.0, 1.0, 1.0]))

    results_y, _ = bead.all_value_and_grad(f, argnums=1, atol=0.0)(x, y)
    val, g_y = results_y[0]
    assert jnp.allclose(g_y, jnp.array([1.0, 0.0, 0.0]))


def test_all_value_and_grad_supports_function_keyword_arguments():
    def function(x, *, offset):
        return bnp.max(x + offset)

    results, paths = bead.all_value_and_grad(function)(
        jnp.array([1.0, 2.0]),
        offset=1.0,
    )
    assert len(paths) == 1
    value, gradient = results[0]
    assert value == 3.0
    assert jnp.allclose(gradient, jnp.array([0.0, 1.0]))


def test_clean_public_api_exports():
    assert not isinstance(bead.HashTensor, type)
    assert not hasattr(bead, "_HashTensor")
    assert not hasattr(bead, "h_fun")
    assert not hasattr(bead, "path_key")


def test_value_and_grad_has_aux_matches_jax_nesting():
    def f(x):
        return bnp.max(x), {"size": x.size}

    x = jnp.array([1.0, 2.0])
    ((value, aux), gradient), paths = bead.value_and_grad(f, has_aux=True)(x)
    assert value == 2.0
    assert aux == {"size": 2}
    assert jnp.allclose(gradient, jnp.array([0.0, 1.0]))

    replay_result = bead.replay_value_and_grad(f, paths[0], has_aux=True)(x)
    assert replay_result[0][1] == aux
    assert jnp.allclose(replay_result[1], gradient)

    results, _ = bead.all_value_and_grad(f, has_aux=True)(x)
    assert results[0][0][1] == aux
    assert jnp.allclose(results[0][1], gradient)
