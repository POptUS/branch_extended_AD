import jax.numpy as jnp
import numpy as np

import branch_extended_AD as bead
import branch_extended_AD.numpy as bnp
from branch_extended_AD.integrations.ibcdfo import h_fun

def test_h_fun_batch_replay_matches_loop():
    def f(x):
        return bnp.sum(bnp.abs(x)) + bnp.max(x)

    hfun = h_fun(f, atol=0.01)
    z = np.array([1.0, -0.5, 0.005, 2.0, -0.005])

    default_value, discovery_gradients, paths = hfun(z)
    H0 = list(paths)
    assert len(H0) >= 1
    assert np.isclose(default_value, np.sum(np.abs(z)) + np.max(z))
    assert discovery_gradients.shape == (z.size, len(H0))

    h, grads = hfun(z, H0=H0)
    assert h.shape == (len(H0),)
    assert grads.shape == (z.shape[0], len(H0))

    for k, path in enumerate(H0):
        manual_v, manual_g = bead.replay_value_and_grad(f, path)(jnp.asarray(z))
        assert np.allclose(h[k], float(manual_v))
        assert np.allclose(grads[:, k], np.asarray(manual_g))

    h0, g0 = hfun(z, H0=[])
    assert h0.shape == (0,)
    assert g0.shape == (z.shape[0], 0)


def test_h_fun_record_branch_matches_manual():
    def f(x):
        return bnp.max(x)

    hfun = h_fun(f, atol=0.1)
    z = np.array([1.0, 1.05, 0.5])

    defaultresult, grads, paths = hfun(z)
    full_paths = list(paths)
    assert len(full_paths) == 2
    assert grads.shape == (z.shape[0], 2)

    for k, path in enumerate(full_paths):
        manual_v, manual_g = bead.replay_value_and_grad(f, path)(jnp.asarray(z))
        assert np.allclose(grads[:, k], np.asarray(manual_g))


def test_h_fun_record_branch_cache_correctness_across_different_ties():
    def f(x):
        return bnp.max(x)

    hfun = h_fun(f, atol=0.0)

    z1 = np.array([1.0, 1.0, 1.0, 0.0])
    _, grads1, paths1 = hfun(z1)
    full_paths1 = list(paths1)
    assert len(full_paths1) == 3
    for k, path in enumerate(full_paths1):
        manual_v, manual_g = bead.replay_value_and_grad(f, path)(jnp.asarray(z1))
        assert np.allclose(grads1[:, k], np.asarray(manual_g))

    z2 = np.array([5.0, 0.0, 0.0, 0.0])
    _, grads2, paths2 = hfun(z2)
    full_paths2 = list(paths2)
    assert len(full_paths2) == 1
    manual_v2, manual_g2 = bead.replay_value_and_grad(f, full_paths2[0])(jnp.asarray(z2))
    assert np.allclose(grads2[:, 0], np.asarray(manual_g2))

    _, grads1b, _ = hfun(z1)
    assert np.allclose(grads1b, grads1)


def test_h_fun_forwards_recording_options():
    def function(x):
        return bnp.sum(bnp.abs(x))

    hfun = h_fun(function, abs_policy="enumerate", rtol=0.1)
    value, gradients, paths = hfun(np.zeros(2))
    assert value == 0.0
    assert len(paths) == 4
    assert gradients.shape == (2, 4)

    scaled = h_fun(lambda x: bnp.max(10.0 * x), atol=1.0, tol_mode="input_scaled")
    _, _, scaled_paths = scaled(np.array([1.0, 1.2]))
    assert len(scaled_paths) == 2


def test_h_fun_supports_argnums_and_aux():
    def function(x):
        value = bnp.max(2.0 * x)
        return value, {"size": x.size}

    hfun = h_fun(function, argnums=0, has_aux=True)
    value, gradients, paths = hfun(np.array([1.0, 2.0]))
    assert value == 4.0
    assert gradients.shape == (2, 1)
    assert len(paths) == 1

    replay_values, replay_gradients = hfun(np.array([1.0, 2.0]), H0=list(paths))
    assert np.allclose(replay_values, np.array([4.0]))
    assert replay_gradients.shape == (2, 1)


def test_h_fun_without_branch_primitives_returns_one_gradient_column():
    def smooth_function(x):
        return bnp.sum(x * x)

    hfun = h_fun(smooth_function)
    value, gradients, paths = hfun(np.array([1.0, 2.0]))
    assert value == 5.0
    assert gradients.shape == (2, 1)
    assert np.allclose(gradients[:, 0], np.array([2.0, 4.0]))
    assert len(paths) == 1
