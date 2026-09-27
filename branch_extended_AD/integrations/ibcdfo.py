import jax
import jax.numpy as jnp
import numpy as np

from ..HashTensor import record, replay_value_and_grad_batch
from ..paths import path_key


def hash_key(value):
    """Return a key for either a BEAD path or a manual IBCDFO hash."""
    return path_key(value) if isinstance(value, list) else value


def hashes_any_in(needles, haystack):
    """Return whether any requested IBCDFO hash occurs in a collection."""
    haystack_keys = {hash_key(value) for value in haystack}
    return any(hash_key(value) in haystack_keys for value in needles)


def hashes_all_in(needles, haystack):
    """Return whether every requested IBCDFO hash occurs in a collection."""
    haystack_keys = {hash_key(value) for value in haystack}
    return all(hash_key(value) in haystack_keys for value in needles)


def unique_hashes(values):
    """Remove duplicate IBCDFO hashes while keeping their original order."""
    result = []
    seen = set()
    for value in values:
        key = hash_key(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def h_fun(fun, argnums=0, has_aux=False, *, atol=0.0, rtol=0.0,
          tol_mode="local", abs_policy="zero"):
    """Adapt a JAX function to the h-function interface used by IBCDFO.

    Calling the result without ``H0`` returns a value, gradient matrix, and
    active paths. Calling it with ``H0`` returns values and gradients for those
    paths.
    """
    def wrapped(z, H0=None):
        z_jax = jnp.asarray(z)

        if H0 is None:
            defaultresult, paths = record(
                fun,
                atol=atol,
                rtol=rtol,
                tol_mode=tol_mode,
                abs_policy=abs_policy,
            )(z_jax)
            if has_aux:
                defaultresult, _ = defaultresult

            if not paths.has_decisions:
                jax_vg_fn = jax.value_and_grad(fun, argnums=argnums, has_aux=has_aux)
                vg_result = jax_vg_fn(z_jax)
                if has_aux:
                    (_, _aux), gradient = vg_result
                else:
                    _, gradient = vg_result
                gradients = np.asarray(gradient, dtype=float).reshape(z_jax.shape[0], 1)
                return defaultresult, gradients, paths

            full_paths = list(paths)
            batch_result = replay_value_and_grad_batch(
                fun,
                full_paths,
                argnums=argnums,
                has_aux=has_aux,
            )(z_jax)
            if has_aux:
                (_, _aux), gradient = batch_result
            else:
                _, gradient = batch_result
            gradients = np.asarray(gradient, dtype=float).T
            return defaultresult, gradients, paths

        if len(H0) == 0:
            return np.zeros(0, dtype=float), np.zeros((z_jax.shape[0], 0), dtype=float)

        batch_result = replay_value_and_grad_batch(
            fun,
            H0,
            argnums=argnums,
            has_aux=has_aux,
        )(z_jax)
        if has_aux:
            (values, _aux), gradient = batch_result
        else:
            values, gradient = batch_result
        return np.asarray(values, dtype=float), np.asarray(gradient, dtype=float).T

    return wrapped
