from jax.numpy import *
import jax.numpy as _jnp

from .HashTensor import (
    _is_recording,
    _is_vmap_replay,
    _is_batch_replay,
    _replay_path,
    HashTensor,
    max as _ht_max,
    min as _ht_min,
    maximum as _ht_maximum,
    minimum as _ht_minimum,
    sum as _ht_sum,
    abs as _ht_abs,
)


def _is_active():
    return (
        _is_recording.get()
        or _replay_path.get() is not None
        or _is_vmap_replay.get()
        or _is_batch_replay.get()
    )


def _unwrap(val):
    if isinstance(val, HashTensor):
        return val.value
    return val


def max(a, **kwargs):
    if _is_active() and not kwargs:
        return _unwrap(_ht_max(HashTensor(a)))
    return _jnp.max(a, **kwargs)


def min(a, **kwargs):
    if _is_active() and not kwargs:
        return _unwrap(_ht_min(HashTensor(a)))
    return _jnp.min(a, **kwargs)


def maximum(x1, x2):
    if _is_active():
        return _unwrap(_ht_maximum(HashTensor(x1), HashTensor(x2)))
    return _jnp.maximum(x1, x2)


def minimum(x1, x2):
    if _is_active():
        return _unwrap(_ht_minimum(HashTensor(x1), HashTensor(x2)))
    return _jnp.minimum(x1, x2)


def sum(a, **kwargs):
    if _is_active() and not kwargs:
        return _unwrap(_ht_sum(HashTensor(a)))
    return _jnp.sum(a, **kwargs)


def abs(a):
    if _is_active():
        return _unwrap(_ht_abs(HashTensor(a)))
    return _jnp.abs(a)
