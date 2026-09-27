from jax.numpy import *
import jax.numpy as _jnp

from .HashTensor import (
    _branch_state,
    _HashTensor,
    max as _ht_max,
    min as _ht_min,
    maximum as _ht_maximum,
    minimum as _ht_minimum,
    sum as _ht_sum,
    abs as _ht_abs,
)


def _is_active():
    return _branch_state.get().mode != "inactive"


def _unwrap(val):
    if isinstance(val, _HashTensor):
        if val.sensitivity is not None:
            from .HashTensor import _SensitivityTensor
            return _SensitivityTensor(val.value, val.sensitivity)
        return val.value
    return val


def max(a):
    if _is_active():
        return _unwrap(_ht_max(_HashTensor(a)))
    return _jnp.max(a)


def min(a):
    if _is_active():
        return _unwrap(_ht_min(_HashTensor(a)))
    return _jnp.min(a)


def maximum(x1, x2):
    if _is_active():
        return _unwrap(_ht_maximum(_HashTensor(x1), _HashTensor(x2)))
    return _jnp.maximum(x1, x2)


def minimum(x1, x2):
    if _is_active():
        return _unwrap(_ht_minimum(_HashTensor(x1), _HashTensor(x2)))
    return _jnp.minimum(x1, x2)


def sum(a):
    if _is_active():
        return _unwrap(_ht_sum(_HashTensor(a)))
    return _jnp.sum(a)


def abs(a):
    if _is_active():
        return _unwrap(_ht_abs(_HashTensor(a)))
    return _jnp.abs(a)
