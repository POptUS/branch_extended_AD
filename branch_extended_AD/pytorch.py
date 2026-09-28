from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace

import torch

from ._path_data import PathSet, _AbsChoice, _ElementwiseChoice, _ReductionChoice, _TraceNode


@dataclass(frozen=True)
class _State:
    mode: str = "inactive"
    trace: object = None
    path: object = None
    position: int = 0
    atol: float = 0.0
    rtol: float = 0.0
    abs_policy: str = "zero"


_state = ContextVar("_torch_branch_state", default=_State())


def _validate_options(tol_mode, abs_policy):
    if tol_mode != "local":
        raise NotImplementedError(
            "PyTorch does not yet support tol_mode='input_scaled'; use tol_mode='local'"
        )
    if abs_policy not in {"zero", "enumerate"}:
        raise ValueError("abs_policy must be 'zero' or 'enumerate'")


@contextmanager
def _mode(mode, *, path=None, atol=0.0, rtol=0.0, abs_policy="zero"):
    if mode == "record":
        state = _State(mode=mode, trace=[], atol=atol, rtol=rtol, abs_policy=abs_policy)
    elif mode == "replay":
        if not isinstance(path, list):
            raise TypeError("path must be a resolved path list")
        state = _State(mode=mode, path=tuple(path))
    else:
        raise ValueError(f"unknown mode {mode}")
    token = _state.set(state)
    try:
        yield state.trace
    finally:
        _state.reset(token)


def _append(name, choices):
    _state.get().trace.append(_TraceNode(name, choices))


def _pop(name):
    state = _state.get()
    if state.position >= len(state.path):
        raise ValueError("Path exhausted")
    node = state.path[state.position]
    if node.name != name:
        raise ValueError(f"Expected trace node {name}, got {node.name}")
    _state.set(replace(state, position=state.position + 1))
    return node.choices[0]


def _tolerance(reference):
    state = _state.get()
    return state.atol + state.rtol * reference.abs()


def _stack(values):
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(values)
    if isinstance(first, tuple):
        return tuple(_stack([value[index] for value in values]) for index in range(len(first)))
    if isinstance(first, list):
        return [_stack([value[index] for value in values]) for index in range(len(first))]
    if isinstance(first, dict):
        return {key: _stack([value[key] for value in values]) for key in first}
    return values


def record(fun, *, atol=0.0, rtol=0.0, tol_mode="local", abs_policy="zero"):
    """Wrap a PyTorch function to return its value and nearby branch paths."""
    _validate_options(tol_mode, abs_policy)

    def recorded(*args, **kwargs):
        with _mode("record", atol=atol, rtol=rtol, abs_policy=abs_policy) as trace:
            value = fun(*args, **kwargs)
        return value, PathSet.from_trace(trace)

    return recorded


def replay(fun, path):
    """Wrap a PyTorch function so it follows one recorded path."""
    def replayed(*args, **kwargs):
        with _mode("replay", path=path):
            return fun(*args, **kwargs)

    return replayed


def grad(fun, argnums=0, has_aux=False, *, atol=0.0, rtol=0.0, tol_mode="local", abs_policy="zero"):
    """Wrap a PyTorch function to return its default gradient and paths."""
    transformed = torch.func.grad(fun, argnums=argnums, has_aux=has_aux)

    def wrapped(*args, **kwargs):
        _, paths = record(fun, atol=atol, rtol=rtol, tol_mode=tol_mode, abs_policy=abs_policy)(*args, **kwargs)
        with _mode("replay", path=paths[0]):
            result = transformed(*args, **kwargs)
        return result, paths

    return wrapped


def value_and_grad(fun, argnums=0, has_aux=False, *, atol=0.0, rtol=0.0, tol_mode="local", abs_policy="zero"):
    """Wrap a PyTorch function to return its default value, gradient, and paths."""
    transformed = torch.func.grad_and_value(fun, argnums=argnums, has_aux=has_aux)

    def wrapped(*args, **kwargs):
        _, paths = record(fun, atol=atol, rtol=rtol, tol_mode=tol_mode, abs_policy=abs_policy)(*args, **kwargs)
        with _mode("replay", path=paths[0]):
            gradient, value = transformed(*args, **kwargs)
        if has_aux:
            value, aux = value
            return ((value, aux), gradient), paths
        return (value, gradient), paths

    return wrapped


def replay_grad(fun, path, argnums=0, has_aux=False):
    """Wrap a PyTorch function to return its gradient along one path."""
    transformed = torch.func.grad(fun, argnums=argnums, has_aux=has_aux)

    def wrapped(*args, **kwargs):
        with _mode("replay", path=path):
            return transformed(*args, **kwargs)

    return wrapped


def replay_value_and_grad(fun, path, argnums=0, has_aux=False):
    """Wrap a PyTorch function to return its value and gradient along one path."""
    transformed = torch.func.grad_and_value(fun, argnums=argnums, has_aux=has_aux)

    def wrapped(*args, **kwargs):
        with _mode("replay", path=path):
            gradient, value = transformed(*args, **kwargs)
        if has_aux:
            value, aux = value
            return (value, aux), gradient
        return value, gradient

    return wrapped


def replay_value_and_grad_batch(fun, paths, argnums=0, has_aux=False):
    """Wrap a PyTorch function to replay several paths."""
    paths = list(paths)
    if not paths:
        raise ValueError("replay_value_and_grad_batch requires at least one path")
    names = [node.name for node in paths[0]]
    if any([node.name for node in path] != names for path in paths[1:]):
        raise ValueError(
            "replay_value_and_grad_batch requires all paths to share the same "
            "branch operations"
        )
    transformed = replay_value_and_grad

    def wrapped(*args, **kwargs):
        results = [
            transformed(fun, path, argnums=argnums, has_aux=has_aux)(*args, **kwargs)
            for path in paths
        ]
        if has_aux:
            values = [result[0][0] for result in results]
            aux = [result[0][1] for result in results]
            gradients = [result[1] for result in results]
            return (_stack(values), _stack(aux)), _stack(gradients)
        return _stack([result[0] for result in results]), _stack([result[1] for result in results])

    return wrapped


def all_value_and_grad(fun, argnums=0, has_aux=False, *, atol=0.0, rtol=0.0, tol_mode="local", abs_policy="zero"):
    """Wrap a PyTorch function to return values and gradients for every path."""
    def wrapped(*args, **kwargs):
        _, paths = record(fun, atol=atol, rtol=rtol, tol_mode=tol_mode, abs_policy=abs_policy)(*args, **kwargs)
        results = [
            replay_value_and_grad(fun, path, argnums=argnums, has_aux=has_aux)(*args, **kwargs)
            for path in paths
        ]
        return results, paths

    return wrapped
