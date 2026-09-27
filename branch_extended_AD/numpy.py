from jax.numpy import *
import jax.numpy as _jnp

from .HashTensor import (
    _is_recording,
    _is_vmap_replay,
    _is_batch_replay,
    _replay_path,
    _HashTensor,
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
    if isinstance(val, _HashTensor):
        if val.sensitivity is not None:
            from .HashTensor import _SensitivityTensor
            return _SensitivityTensor(val.value, val.sensitivity)
        return val.value
    return val


def _reject_active_kwargs(name, kwargs):
    if kwargs:
        arguments = ", ".join(sorted(kwargs))
        raise NotImplementedError(
            f"branch_extended_AD.numpy.{name} does not support keyword arguments "
            f"during record/replay: {arguments}"
        )


def max(a, **kwargs):
    if _is_active():
        _reject_active_kwargs("max", kwargs)
        return _unwrap(_ht_max(_HashTensor(a)))
    return _jnp.max(a, **kwargs)


def min(a, **kwargs):
    if _is_active():
        _reject_active_kwargs("min", kwargs)
        return _unwrap(_ht_min(_HashTensor(a)))
    return _jnp.min(a, **kwargs)


def maximum(x1, x2):
    if _is_active():
        return _unwrap(_ht_maximum(_HashTensor(x1), _HashTensor(x2)))
    return _jnp.maximum(x1, x2)


def minimum(x1, x2):
    if _is_active():
        return _unwrap(_ht_minimum(_HashTensor(x1), _HashTensor(x2)))
    return _jnp.minimum(x1, x2)


def sum(a, **kwargs):
    if _is_active():
        _reject_active_kwargs("sum", kwargs)
        return _unwrap(_ht_sum(_HashTensor(a)))
    return _jnp.sum(a, **kwargs)


def abs(a):
    if _is_active():
        return _unwrap(_ht_abs(_HashTensor(a)))
    return _jnp.abs(a)
