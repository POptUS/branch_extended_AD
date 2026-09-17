import importlib
import time
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxnp_hash as jnph
import jaxnp_hash.numpy as jnph_np

# jaxnp_hash/__init__.py does `from .HashTensor import HashTensor` (the class), which
# shadows the `jaxnp_hash.HashTensor` submodule attribute on the package -- import via
# importlib to reach private module-level helpers like `_bucket_size`.
ht = importlib.import_module("jaxnp_hash.HashTensor")


def test_maximum():
    x1 = jnp.array([1.0, 2.0, 2.5])
    x2 = jnp.array([0.5, 1.5, 3.0])

    def f(x, y):
        return jnph_np.maximum(x, y)

    val, paths = jnph.record(f, tol=0.1)(x1, x2)
    assert jnp.allclose(val, jnp.array([1.0, 2.0, 3.0])), f"Expected [1.0, 2.0, 3.0], got {val}"
    assert len(paths) == 1

    replayed = jnph.replay(f, paths[0])(x1, x2)
    assert jnp.allclose(replayed, jnp.array([1.0, 2.0, 3.0])), f"Expected [1.0, 2.0, 3.0], got {replayed}"

    x2b = jnp.array([0.95, 1.5, 3.0])

    val, paths = jnph.record(f, tol=0.1)(x1, x2b)
    assert jnp.allclose(val[1:], jnp.array([2.0, 3.0]))
    assert val[0] in [1.0, 0.95], f"Expected first value to be 1.0 or 0.95, got {val[0]}"
    assert len(paths) == 2

    expected_results = set()
    for first_val in [1.0, 0.95]:
        expected_results.add(tuple(jnp.array([first_val, 2.0, 3.0]).tolist()))

    actual_results = set()
    for p in paths:
        result = jnph.replay(f, p)(x1, x2b)
        actual_results.add(tuple(result.tolist()))

    assert actual_results == expected_results, f"Expected {expected_results}, got {actual_results}"

    x2c = jnp.array([0.95, 1.95, 2.45])

    val, paths = jnph.record(f, tol=0.1)(x1, x2c)
    for i in range(3):
        assert val[i] in [x1[i], x2c[i]]
    assert len(paths) == 8

    expected_results = set()
    for vals in product([x1[0], x2c[0]], [x1[1], x2c[1]], [x1[2], x2c[2]]):
        expected_results.add(tuple(jnp.array(vals).tolist()))

    actual_results = set()
    for p in paths:
        result = jnph.replay(f, p)(x1, x2c)
        actual_results.add(tuple(result.tolist()))

    assert actual_results == expected_results, f"Expected {expected_results}, got {actual_results}"


def test_minimum():
    x1 = jnp.array([1.0, 2.0, 2.5])
    x2 = jnp.array([0.5, 1.5, 3.0])

    def f(x, y):
        return jnph_np.minimum(x, y)

    val, paths = jnph.record(f, tol=0.1)(x1, x2)
    assert jnp.allclose(val, jnp.array([0.5, 1.5, 2.5]))
    assert len(paths) == 1

    replayed = jnph.replay(f, paths[0])(x1, x2)
    assert jnp.allclose(replayed, jnp.array([0.5, 1.5, 2.5]))

    x2b = jnp.array([1.05, 1.5, 3.0])

    val, paths = jnph.record(f, tol=0.1)(x1, x2b)
    assert jnp.allclose(val[1:], jnp.array([1.5, 2.5]))
    assert val[0] in [1.0, 1.05]
    assert len(paths) == 2

    expected_results = set()
    for first_val in [1.0, 1.05]:
        expected_results.add(tuple(jnp.array([first_val, 1.5, 2.5]).tolist()))

    actual_results = set()
    for p in paths:
        result = jnph.replay(f, p)(x1, x2b)
        actual_results.add(tuple(result.tolist()))

    assert actual_results == expected_results

    x2c = jnp.array([1.05, 2.05, 2.55])

    val, paths = jnph.record(f, tol=0.1)(x1, x2c)
    for i in range(3):
        assert val[i] in [x1[i], x2c[i]]
    assert len(paths) == 8

    expected_results = set()
    for vals in product([x1[0], x2c[0]], [x1[1], x2c[1]], [x1[2], x2c[2]]):
        expected_results.add(tuple(jnp.array(vals).tolist()))

    actual_results = set()
    for p in paths:
        result = jnph.replay(f, p)(x1, x2c)
        actual_results.add(tuple(result.tolist()))

    assert actual_results == expected_results


def test_abs():
    def f(x):
        return jnph_np.abs(x)

    x = jnp.array([-1.0, -2.0, 2.5])
    val, paths = jnph.record(f, tol=0.1)(x)
    assert jnp.allclose(val, jnp.array([1.0, 2.0, 2.5]))
    assert len(paths) == 1

    replayed = jnph.replay(f, paths[0])(x)
    assert jnp.allclose(replayed, jnp.array([1.0, 2.0, 2.5]))

    x2 = jnp.array([0.05, -2.0, 2.5])
    val, paths = jnph.record(f, tol=0.1)(x2)
    assert val[0] == 0.05
    assert jnp.allclose(val[1:], jnp.array([2.0, 2.5]))
    assert len(paths) == 1

    replayed = jnph.replay(f, paths[0])(x2)
    assert jnp.allclose(replayed, jnp.array([0.05, 2.0, 2.5]))

    x3 = jnp.array([0.05, -0.05, 0.03])
    val, paths = jnph.record(f, tol=0.1)(x3)
    assert len(paths) == 1
    assert jnp.allclose(val, jnp.array([0.05, 0.05, 0.03]))

    replayed = jnph.replay(f, paths[0])(x3)
    assert jnp.allclose(replayed, jnp.array([0.05, 0.05, 0.03]))


def test_abs_ambiguous_gradient_is_zero():
    def f(x):
        return jnph_np.sum(jnph_np.abs(x))

    x = jnp.array([0.05, -2.0, 2.5])
    g, paths = jnph.grad(f, tol=0.1)(x)
    assert len(paths) == 1
    assert jnp.allclose(g, jnp.array([0.0, -1.0, 1.0]))


def test_abs_replay_at_different_point():
    def f(x):
        return jnph_np.abs(x)

    x = jnp.array([0.05, -2.0, 2.5])
    _, paths = jnph.record(f, tol=0.1)(x)
    path = paths[0]

    x_prime = jnp.array([0.03, 2.0, 2.5])
    replayed = jnph.replay(f, path)(x_prime)
    assert jnp.allclose(replayed, jnp.array([0.03, -2.0, 2.5]))

    def f_sum(x):
        return jnph_np.sum(jnph_np.abs(x))

    _, sum_paths = jnph.record(f_sum, tol=0.1)(x)
    sum_path = sum_paths[0]
    v, g = jnph.replay_value_and_grad(f_sum, sum_path)(x_prime)
    assert jnp.allclose(v, 0.03 - 2.0 + 2.5)
    assert jnp.allclose(g, jnp.array([0.0, -1.0, 1.0]))


def test_path_set_operations():
    arr1 = jnp.array([1.01, 2.02, 3.03])
    arr2 = jnp.array([0.02, -0.01, 0.03])

    def f(x, y):
        r1 = jnph_np.max(x)
        r2 = jnph_np.abs(y)
        return jnph_np.sum(r2)

    _, tight_paths = jnph.record(f, tol=0.01)(arr1, arr2)
    _, loose_paths = jnph.record(f, tol=0.1)(arr1, arr2)

    print(f"Tight tolerance: {len(tight_paths)} paths")
    print(f"Loose tolerance: {len(loose_paths)} paths")

    union_set = tight_paths.union(loose_paths)
    print(f"Union: {len(union_set)} paths")
    assert len(union_set) >= len(tight_paths) and len(union_set) >= len(loose_paths)

    intersection_set = tight_paths.intersection(loose_paths)
    print(f"Intersection: {len(intersection_set)} paths")
    assert len(intersection_set) <= len(tight_paths) and len(intersection_set) <= len(loose_paths)

    diff_set = loose_paths.difference(tight_paths)
    print(f"Difference (loose - tight): {len(diff_set)} paths")

    for path in tight_paths:
        assert path in union_set or any(jnph.PathSet._paths_equal(path, u) for u in union_set)

    for path in loose_paths:
        assert path in union_set or any(jnph.PathSet._paths_equal(path, u) for u in union_set)

    for path in intersection_set:
        in_tight = path in tight_paths or any(jnph.PathSet._paths_equal(path, t) for t in tight_paths)
        in_loose = path in loose_paths or any(jnph.PathSet._paths_equal(path, l) for l in loose_paths)
        assert in_tight and in_loose, "Intersection element should be in both sets"

    print("All set operations tests passed!")


def test_path_set_with_no_tolerance():
    print("Testing PathSet with no tolerance effects...")

    x1 = jnp.array([1.0, 5.0, 10.0])
    x2 = jnp.array([2.0, 6.0, 11.0])

    def f(x, y):
        return jnph_np.maximum(x, y)

    val, paths = jnph.record(f, tol=0.1)(x1, x2)
    assert len(paths) == 1

    path_list = list(paths)
    assert len(path_list) == 1

    replay_result = jnph.replay(f, path_list[0])(x1, x2)
    expected = jnp.array([2.0, 6.0, 11.0])
    assert jnp.allclose(replay_result, expected), f"Expected {expected}, got {replay_result}"

    print("✓ No tolerance test passed!")


def test_path_set_membership():
    print("Testing PathSet membership...")

    x1 = jnp.array([1.0, 2.0])
    x2 = jnp.array([1.05, 1.95])

    def f(x, y):
        return jnph_np.maximum(x, y)

    _, paths = jnph.record(f, tol=0.1)(x1, x2)
    assert len(paths) == 4

    iteration1 = list(paths)
    iteration2 = list(paths)
    iteration3 = list(paths)

    assert len(iteration1) == len(iteration2) == len(iteration3) == 4

    valid_results = set()
    for i, p in enumerate(iteration1):
        _, fresh_paths = jnph.record(f, tol=0.1)(x1, x2)
        fresh_list = list(fresh_paths)

        replay_result = jnph.replay(f, fresh_list[i])(x1, x2)
        result_tuple = tuple(replay_result.tolist())
        valid_results.add(result_tuple)

        for j, val in enumerate(replay_result):
            assert val in [x1[j], x2[j]], \
                f"Result value {val} at index {j} should be from x1 or x2"

    assert len(valid_results) == 4, \
        f"Expected 4 unique results from 4 paths, got {len(valid_results)}"

    print("✓ Membership test passed!")


def test_path_set_random_access():
    print("Testing PathSet random access...")

    x1 = jnp.array([1.0, 2.0])
    x2 = jnp.array([1.05, 1.95])

    def f(x, y):
        return jnph_np.maximum(x, y)

    _, paths = jnph.record(f, tol=0.1)(x1, x2)
    assert len(paths) == 4

    accessed = []
    for i in range(len(paths)):
        p = paths[i]
        accessed.append(p)
        print(f"Path at index {i}: {p}")

    iteration = list(paths)
    assert len(accessed) == len(iteration)

    for i, (a, it) in enumerate(zip(accessed, iteration)):
        assert jnph.PathSet._paths_equal(a, it), \
            f"Path at index {i} should be the same from random access and iteration"

    assert paths[-1] == paths[3]
    assert paths[-2] == paths[2]

    try:
        _ = paths[4]
        assert False, "Should have raised IndexError for index 4"
    except IndexError:
        pass

    try:
        _ = paths[-5]
        assert False, "Should have raised IndexError for index -5"
    except IndexError:
        pass

    try:
        _ = paths["invalid"]
        assert False, "Should have raised TypeError for string index"
    except TypeError:
        pass

    first_access = paths[1]
    second_access = paths[1]
    assert first_access == second_access

    print("✓ Random access test passed!")


def test_record_and_replay():
    x1 = jnp.array([1.0, 2.0, 2.5])
    x2 = jnp.array([0.95, 1.5, 3.0])

    def f(x, y):
        return jnph_np.maximum(x, y)

    val, paths = jnph.record(f, tol=0.1)(x1, x2)
    assert isinstance(paths, jnph.PathSet)
    assert len(paths) == 2

    results = []
    for p in paths:
        assert isinstance(p, list)
        replayed_val = jnph.replay(f, p)(x1, x2)
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
        return jnph_np.max(x)

    val, paths = jnph.record(f, tol=0.0)(x)
    assert len(paths) == 1
    assert float(val) == 10.0


def test_grad_simple():
    x = jnp.array([1.0, 3.0, 2.0])

    def f(x):
        return jnph_np.max(x)

    g, paths = jnph.grad(f, tol=0.0)(x)
    assert g.shape == x.shape
    assert jnp.allclose(g, jnp.array([0.0, 1.0, 0.0])), f"Expected [0, 1, 0], got {g}"
    assert len(paths) == 1


def test_grad_with_tolerance():
    x = jnp.array([1.0, 1.05, 0.5])

    def f(x):
        return jnph_np.max(x)

    g, paths = jnph.grad(f, tol=0.1)(x)
    assert len(paths) == 2
    assert g.shape == x.shape


def test_value_and_grad_simple():
    x = jnp.array([1.0, 3.0, 2.0])

    def f(x):
        return jnph_np.sum(x)

    (val, g), paths = jnph.value_and_grad(f, tol=0.0)(x)
    assert jnp.allclose(val, 6.0)
    assert jnp.allclose(g, jnp.array([1.0, 1.0, 1.0])), f"Expected [1, 1, 1], got {g}"
    assert len(paths) == 1


def test_value_and_grad_maximum():
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.5, 1.5, 1.5])

    def f(x, y):
        return jnph_np.sum(jnph_np.maximum(x, y))

    (val, g), paths = jnph.value_and_grad(f, argnums=0, tol=0.0)(x, y)
    assert float(val) == 1.5 + 2.0 + 3.0
    assert jnp.allclose(g, jnp.array([0.0, 1.0, 1.0])), f"Expected [0, 1, 1], got {g}"


def test_replay_grad():
    x = jnp.array([1.0, 1.05, 0.5])

    def f(x):
        return jnph_np.max(x)

    _, paths = jnph.record(f, tol=0.1)(x)
    assert len(paths) >= 2

    grads = []
    for p in paths:
        g = jnph.replay_grad(f, p)(x)
        grads.append(g)

    grad_tuples = set(tuple(float(v) for v in g) for g in grads)
    assert (0.0, 1.0, 0.0) in grad_tuples or (1.0, 0.0, 0.0) in grad_tuples


def test_replay_value_and_grad():
    x = jnp.array([1.0, 2.0, 2.5])
    y = jnp.array([0.95, 1.5, 3.0])

    def f(x, y):
        return jnph_np.sum(jnph_np.maximum(x, y))

    _, paths = jnph.record(f, tol=0.1)(x, y)

    for p in paths:
        replayed_val = jnph.replay(f, p)(x, y)
        v, g = jnph.replay_value_and_grad(f, p, argnums=0)(x, y)
        assert jnp.allclose(v, replayed_val), f"value_and_grad value {v} != replay value {replayed_val}"
        assert g.shape == x.shape


def test_grad_argnums():
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.5, 1.5, 1.5])

    def f(x, y):
        return jnph_np.sum(jnph_np.maximum(x, y))

    g_x, _ = jnph.grad(f, argnums=0, tol=0.0)(x, y)
    assert jnp.allclose(g_x, jnp.array([0.0, 1.0, 1.0]))

    g_y, _ = jnph.grad(f, argnums=1, tol=0.0)(x, y)
    assert jnp.allclose(g_y, jnp.array([1.0, 0.0, 0.0]))

    (g_x2, g_y2), _ = jnph.grad(f, argnums=(0, 1), tol=0.0)(x, y)
    assert jnp.allclose(g_x2, g_x)
    assert jnp.allclose(g_y2, g_y)


def test_has_aux():
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.5, 1.5, 1.5])

    def f_aux(x, y):
        result = jnph_np.maximum(x, y)
        s = jnph_np.sum(result)
        aux = {"max_val": jnph_np.max(result)}
        return s, aux

    (g, aux), paths = jnph.grad(f_aux, argnums=0, tol=0.0, has_aux=True)(x, y)
    assert "max_val" in aux
    assert g.shape == x.shape
    assert jnp.allclose(g, jnp.array([0.0, 1.0, 1.0]))

    (val, g2, aux2), paths2 = jnph.value_and_grad(f_aux, argnums=0, tol=0.0, has_aux=True)(x, y)
    assert float(val) == 1.5 + 2.0 + 3.0
    assert jnp.allclose(g2, g)
    assert "max_val" in aux2


def test_replay_grad_has_aux():
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.05, 1.95, 2.95])

    def f_aux(x, y):
        result = jnph_np.maximum(x, y)
        s = jnph_np.sum(result)
        return s, {"count": len(x)}

    _, paths = jnph.record(f_aux, tol=0.1)(x, y)

    for p in paths:
        g, aux = jnph.replay_grad(f_aux, p, argnums=0, has_aux=True)(x, y)
        assert g.shape == x.shape
        assert aux["count"] == 3

        v, g2, aux2 = jnph.replay_value_and_grad(f_aux, p, argnums=0, has_aux=True)(x, y)
        assert jnp.allclose(g, g2)
        assert aux2["count"] == 3


def test_path_types():
    x1 = jnp.array([1.0, 2.0])
    x2 = jnp.array([1.05, 1.95])

    def f(x, y):
        return jnph_np.maximum(x, y)

    _, paths = jnph.record(f, tol=0.1)(x1, x2)

    assert isinstance(paths, jnph.PathSet)
    assert len(paths) == 4

    for p in paths:
        assert isinstance(p, list)
        assert len(p) == 1

    p0 = paths[0]
    assert isinstance(p0, list)

    assert paths[-1] == paths[3]
    assert paths[0] != paths[1]


def test_branch_path_equality():
    x1 = jnp.array([1.0, 2.0])
    x2 = jnp.array([1.05, 1.95])

    def f(x, y):
        return jnph_np.maximum(x, y)

    _, paths1 = jnph.record(f, tol=0.1)(x1, x2)
    _, paths2 = jnph.record(f, tol=0.1)(x1, x2)

    for p1, p2 in zip(paths1, paths2):
        assert p1 == p2

    all_paths = list(paths1)
    assert all_paths[0] != all_paths[1]


def test_all_value_and_grad_simple():
    x = jnp.array([1.0, 3.0, 2.0])

    def f(x):
        return jnph_np.max(x)

    results, paths = jnph.all_value_and_grad(f, tol=0.0)(x)
    assert len(results) == 1
    assert len(paths) == 1
    val, grad = results[0]
    assert jnp.allclose(val, 3.0)
    assert jnp.allclose(grad, jnp.array([0.0, 1.0, 0.0]))


def test_all_value_and_grad_with_tolerance():
    x = jnp.array([1.0, 1.05, 0.5])

    def f(x):
        return jnph_np.max(x)

    results, paths = jnph.all_value_and_grad(f, tol=0.1)(x)
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
        return jnph_np.sum(jnph_np.maximum(x, y))

    results, paths = jnph.all_value_and_grad(f, argnums=0, tol=0.1)(x, y)

    for i, path in enumerate(paths):
        manual_v, manual_g = jnph.replay_value_and_grad(f, path, argnums=0)(x, y)
        api_v, api_g = results[i]
        assert jnp.allclose(api_v, manual_v)
        assert jnp.allclose(api_g, manual_g)


def test_all_value_and_grad_has_aux():
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.05, 1.95, 2.95])

    def f_aux(x, y):
        result = jnph_np.maximum(x, y)
        s = jnph_np.sum(result)
        return s, {"count": len(x)}

    results, paths = jnph.all_value_and_grad(f_aux, argnums=0, tol=0.1, has_aux=True)(x, y)
    assert len(results) == len(paths)

    for val, grad, aux in results:
        assert aux["count"] == 3
        assert grad.shape == x.shape


def test_all_value_and_grad_argnums():
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.5, 1.5, 1.5])

    def f(x, y):
        return jnph_np.sum(jnph_np.maximum(x, y))

    results_x, _ = jnph.all_value_and_grad(f, argnums=0, tol=0.0)(x, y)
    assert len(results_x) == 1
    val, g_x = results_x[0]
    assert jnp.allclose(g_x, jnp.array([0.0, 1.0, 1.0]))

    results_y, _ = jnph.all_value_and_grad(f, argnums=1, tol=0.0)(x, y)
    val, g_y = results_y[0]
    assert jnp.allclose(g_y, jnp.array([1.0, 0.0, 0.0]))


def test_replay_value_and_grad_batch_matches_loop_max_min():
    def f_max(x):
        return jnph_np.max(x)

    def f_min(x):
        return jnph_np.min(x)

    z = jnp.array([2.0, 4.0, 6.0])

    for f in (f_max, f_min):
        xs = [
            jnp.array([1.0, 3.0, 2.0]),
            jnp.array([5.0, 1.0, 5.0]),
            jnp.array([0.0, -1.0, 2.0]),
        ]
        paths = [list(jnph.record(f, tol=0.0)(x)[1])[0] for x in xs]

        values, grads = jnph.replay_value_and_grad_batch(f, paths)(z)
        for k, path in enumerate(paths):
            manual_v, manual_g = jnph.replay_value_and_grad(f, path)(z)
            assert jnp.allclose(values[k], manual_v)
            assert jnp.allclose(grads[k], manual_g)


def test_replay_value_and_grad_batch_matches_loop_abs():
    def f(x):
        return jnph_np.sum(jnph_np.abs(x))

    xs = [
        jnp.array([1.0, -0.5, 0.005, 2.0, -0.005]),
        jnp.array([-3.0, 0.5, 0.5, -0.001, 4.0]),
        jnp.array([2.0, 2.0, 2.0, 2.0, 2.0]),
    ]
    paths = [list(jnph.record(f, tol=0.01)(x)[1])[0] for x in xs]
    z = jnp.array([1.0, -2.0, 0.5, 3.0, -0.2])

    values, grads = jnph.replay_value_and_grad_batch(f, paths)(z)
    for k, path in enumerate(paths):
        manual_v, manual_g = jnph.replay_value_and_grad(f, path)(z)
        assert jnp.allclose(values[k], manual_v)
        assert jnp.allclose(grads[k], manual_g)


def test_replay_value_and_grad_batch_matches_loop_maximum_minimum():
    def f(x, y):
        return jnph_np.sum(jnph_np.maximum(x, y))

    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.05, 1.95, 3.05])
    _, path_set = jnph.record(f, tol=0.1)(x, y)
    paths = list(path_set)
    assert len(paths) > 1

    z1 = jnp.array([2.0, -1.0, 0.5])
    z2 = jnp.array([1.9, -1.5, 0.6])

    values, grads = jnph.replay_value_and_grad_batch(f, paths, argnums=(0, 1))(z1, z2)
    for k, path in enumerate(paths):
        manual_v, manual_g = jnph.replay_value_and_grad(f, path, argnums=(0, 1))(z1, z2)
        assert jnp.allclose(values[k], manual_v)
        assert jnp.allclose(grads[0][k], manual_g[0])
        assert jnp.allclose(grads[1][k], manual_g[1])


def test_replay_value_and_grad_batch_large_J_no_blowup():
    def f(x):
        return jnph_np.max(x)

    n = 44
    x = jnp.ones(n) * 5.0
    _, path_set = jnph.record(f, tol=0.0)(x)
    paths = list(path_set)
    assert len(paths) == n

    z = jnp.arange(n, dtype=jnp.float32)
    t0 = time.time()
    values, grads = jnph.replay_value_and_grad_batch(f, paths)(z)
    dt = time.time() - t0

    assert dt < 5.0, f"batched replay over J={n} took {dt:.2f}s, expected well under a second"
    assert values.shape == (n,)
    assert grads.shape == (n, n)


def test_bucket_size():
    expected = {1: 1, 2: 2, 3: 4, 4: 4, 5: 8, 8: 8, 9: 16, 44: 64, 87: 128}
    for J, want in expected.items():
        assert ht._bucket_size(J) == want


def test_replay_value_and_grad_batch_padding_matches_unpadded():
    # J values that are NOT powers of two, so _bucket_size's padding is exercised.
    def f(x):
        return jnph_np.max(x)

    n = 10
    z = jnp.arange(n, dtype=jnp.float32)

    for J in (3, 5, 6, 7):
        xs = [jnp.asarray(np.random.RandomState(i).rand(n)) for i in range(J)]
        paths = [list(jnph.record(f, tol=0.0)(x)[1])[0] for x in xs]

        values, grads = jnph.replay_value_and_grad_batch(f, paths)(z)
        assert values.shape == (J,)
        assert grads.shape == (J, n)

        for k, path in enumerate(paths):
            manual_v, manual_g = jnph.replay_value_and_grad(f, path)(z)
            assert jnp.allclose(values[k], manual_v)
            assert jnp.allclose(grads[k], manual_g)


def test_replay_value_and_grad_batch_mismatched_paths_raises():
    def f_max(x):
        return jnph_np.max(x)

    def f_maximum(y, z):
        return jnph_np.sum(jnph_np.maximum(y, z))

    x = jnp.array([1.0, 3.0, 2.0])
    _, max_paths = jnph.record(f_max, tol=0.0)(x)
    max_path = list(max_paths)[0]

    y = jnp.array([1.0, 3.0])
    z = jnp.array([1.0, 3.0])
    _, maximum_paths = jnph.record(f_maximum, tol=0.5)(y, z)
    maximum_path = list(maximum_paths)[0]

    with pytest.raises(ValueError):
        jnph.replay_value_and_grad_batch(f_max, [max_path, maximum_path])(x)


def test_h_fun_batch_replay_matches_loop():
    def f(x):
        return jnph_np.sum(jnph_np.abs(x)) + jnph_np.max(x)

    hfun = jnph.h_fun(f, tol=0.01)
    z = np.array([1.0, -0.5, 0.005, 2.0, -0.005])

    defaultresult, grads0, paths = hfun(z)
    H0 = list(paths)
    assert len(H0) >= 1

    h, grads = hfun(z, H0=H0)
    assert h.shape == (len(H0),)
    assert grads.shape == (z.shape[0], len(H0))

    for k, path in enumerate(H0):
        manual_v, manual_g = jnph.replay_value_and_grad(f, path)(jnp.asarray(z))
        assert np.allclose(h[k], float(manual_v))
        assert np.allclose(grads[:, k], np.asarray(manual_g))

    h0, g0 = hfun(z, H0=[])
    assert h0.shape == (0,)
    assert g0.shape == (z.shape[0], 0)


def test_h_fun_record_branch_matches_manual():
    def f(x):
        return jnph_np.max(x)

    hfun = jnph.h_fun(f, tol=0.1)
    z = np.array([1.0, 1.05, 0.5])

    defaultresult, grads, paths = hfun(z)
    full_paths = list(paths)
    assert len(full_paths) == 2
    assert grads.shape == (z.shape[0], 2)

    for k, path in enumerate(full_paths):
        manual_v, manual_g = jnph.replay_value_and_grad(f, path)(jnp.asarray(z))
        assert np.allclose(grads[:, k], np.asarray(manual_g))


def test_h_fun_record_branch_cache_correctness_across_different_ties():
    # Regression guard: the record (H0=None) branch now routes through the same
    # jit-cached replay_value_and_grad_batch machinery as the H0-given branch. That
    # cache is keyed on (fun, leaf_layout, n_args, argnums, has_aux) -- NOT on J or
    # the concrete tie locations -- so calls with different npaths/choices but the
    # same op-sequence must each get correct, non-stale results.
    def f(x):
        return jnph_np.max(x)

    hfun = jnph.h_fun(f, tol=0.0)

    z1 = np.array([1.0, 1.0, 1.0, 0.0])  # 3-way tie -> npaths=3
    _, grads1, paths1 = hfun(z1)
    full_paths1 = list(paths1)
    assert len(full_paths1) == 3
    for k, path in enumerate(full_paths1):
        manual_v, manual_g = jnph.replay_value_and_grad(f, path)(jnp.asarray(z1))
        assert np.allclose(grads1[:, k], np.asarray(manual_g))

    z2 = np.array([5.0, 0.0, 0.0, 0.0])  # unique max -> npaths=1
    _, grads2, paths2 = hfun(z2)
    full_paths2 = list(paths2)
    assert len(full_paths2) == 1
    manual_v2, manual_g2 = jnph.replay_value_and_grad(f, full_paths2[0])(jnp.asarray(z2))
    assert np.allclose(grads2[:, 0], np.asarray(manual_g2))

    # Re-run z1 to confirm the J=1 call above didn't corrupt the (same-cache-key) J=3 case.
    _, grads1b, _ = hfun(z1)
    assert np.allclose(grads1b, grads1)


def test_passthrough_outside_mode():
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.5, 1.5, 1.5])

    result = jnph_np.maximum(x, y)
    assert jnp.allclose(result, jnp.array([1.5, 2.0, 3.0]))

    result = jnph_np.sum(x)
    assert float(result) == 6.0

    result = jnph_np.max(x)
    assert float(result) == 3.0

    result = jnph_np.abs(jnp.array([-1.0, 2.0]))
    assert jnp.allclose(result, jnp.array([1.0, 2.0]))

    result = jnph_np.min(x)
    assert float(result) == 1.0

    result = jnph_np.minimum(x, y)
    assert jnp.allclose(result, jnp.array([1.0, 1.5, 1.5]))


def test_numpy_non_overridden_functions():
    x = jnp.array([1.0, 2.0, 3.0])
    assert jnp.allclose(jnph_np.sin(x), jnp.sin(x))
    assert jnp.allclose(jnph_np.exp(x), jnp.exp(x))
    assert jnph_np.array([1, 2, 3]).dtype == jnp.array([1, 2, 3]).dtype


if __name__ == "__main__":
    test_maximum()
    test_minimum()
    test_abs()
    test_path_set_operations()
    test_path_set_with_no_tolerance()
    test_path_set_membership()
    test_path_set_random_access()
    test_record_and_replay()
    test_record_no_tolerance()
    test_grad_simple()
    test_grad_with_tolerance()
    test_value_and_grad_simple()
    test_value_and_grad_maximum()
    test_replay_grad()
    test_replay_value_and_grad()
    test_grad_argnums()
    test_has_aux()
    test_replay_grad_has_aux()
    test_path_types()
    test_branch_path_equality()
    test_all_value_and_grad_simple()
    test_all_value_and_grad_with_tolerance()
    test_all_value_and_grad_matches_manual()
    test_all_value_and_grad_has_aux()
    test_all_value_and_grad_argnums()
    test_replay_value_and_grad_batch_matches_loop_max_min()
    test_replay_value_and_grad_batch_matches_loop_abs()
    test_replay_value_and_grad_batch_matches_loop_maximum_minimum()
    test_replay_value_and_grad_batch_large_J_no_blowup()
    test_replay_value_and_grad_batch_mismatched_paths_raises()
    test_h_fun_batch_replay_matches_loop()
    test_passthrough_outside_mode()
    test_numpy_non_overridden_functions()
    print("All tests passed!")
