import builtins
import itertools
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from .paths import path_key

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class _BranchState:
    mode: str = "inactive"
    trace: object = None
    replay_path: object = None
    replay_pos: int = 0
    atol: float = 0.0
    rtol: float = 0.0
    tol_mode: str = "local"
    abs_policy: str = "zero"
    selectors: object = None
    batch_arrays: object = None


_branch_state: ContextVar[_BranchState] = ContextVar(
    "_branch_state",
    default=_BranchState(),
)

# Cache compiled batched replay functions by function and traced operation layout.
_jit_batch_cache: dict = {}


class _TraceNode:
    def __init__(self, name, choices):
        logger.debug("_TraceNode.__init__: name=%s, num_choices=%s", name, len(choices))
        self.name = name
        self.choices = list(choices)
        self.num = len(self.choices)

    def __repr__(self):
        return f'_TraceNode(name="{self.name}", choices={self.choices!r})'

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if not isinstance(other, _TraceNode):
            return NotImplemented
        return self.name == other.name and self.choices == other.choices

class PathSet:
    def __init__(self, *, trace=None, paths=None):
        if (trace is None) == (paths is None):
            raise ValueError("provide exactly one of trace or paths")
        self._trace = None if trace is None else tuple(trace)
        self._paths = None if paths is None else tuple(dict.fromkeys(path_key(path) for path in paths))

    @classmethod
    def from_trace(cls, trace):
        return cls(trace=trace)

    @classmethod
    def from_paths(cls, paths):
        paths = tuple(paths)
        if not all(isinstance(path, list) for path in paths):
            raise TypeError("paths must contain resolved path lists")
        return cls(paths=paths)

    @classmethod
    def empty(cls):
        return cls(paths=())

    @property
    def has_decisions(self):
        if self._trace is not None:
            return bool(self._trace)
        return any(key for key in self._paths)

    def choices(self, decision_index):
        if self._trace is not None:
            return tuple(self._trace[decision_index].choices)
        return tuple(dict.fromkeys(key[decision_index][1] for key in self._paths))

    def __iter__(self):
        if self._paths is not None:
            yield from ([_TraceNode(name, [choice]) for name, choice in key] for key in self._paths)
            return
        if not self._trace:
            yield []
            return
        for choices in itertools.product(*(node.choices for node in self._trace)):
            yield [_TraceNode(node.name, [choice]) for node, choice in zip(self._trace, choices)]

    def _iter_positions(self):
        if self._paths is not None:
            raise ValueError("explicit PathSet does not have factorized trace positions")
        if not self._trace:
            yield ()
            return
        yield from itertools.product(*(range(node.num) for node in self._trace))

    def __len__(self):
        if self._paths is not None:
            return len(self._paths)
        total = 1
        for node in self._trace:
            total *= node.num
        return total

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError("index must be an integer")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"index {index} is out of range for PathSet with {len(self)} elements")
        if self._paths is not None:
            return [_TraceNode(name, [choice]) for name, choice in self._paths[index]]
        if not self._trace:
            return []
        positions = []
        remaining = index
        for current, node in enumerate(self._trace):
            combinations_after = 1
            for later in self._trace[current + 1:]:
                combinations_after *= later.num
            positions.append(remaining // combinations_after)
            remaining %= combinations_after
        return [_TraceNode(node.name, [node.choices[position]]) for node, position in zip(self._trace, positions)]

    def __contains__(self, path):
        if not isinstance(path, list):
            return False
        if self._paths is not None:
            return path_key(path) in self._paths
        if len(path) != len(self._trace):
            return False
        return all(
            decision.name == node.name and len(decision.choices) == 1 and decision.choices[0] in node.choices
            for decision, node in zip(path, self._trace)
        )

    def __bool__(self):
        return len(self) > 0

    def __repr__(self):
        if self._paths is not None:
            return f"PathSet(paths={len(self._paths)})"
        return f"PathSet(decisions={len(self._trace)}, paths={len(self)})"

    def __str__(self):
        if self._paths is not None:
            return f"PathSet with {len(self)} explicit paths"
        return f"PathSet with {len(self)} possible paths from {len(self._trace)} decisions"

    def union(self, other):
        if not isinstance(other, PathSet):
            raise TypeError("union requires another PathSet")
        return PathSet.from_paths([*self, *other])

    def intersection(self, other):
        if not isinstance(other, PathSet):
            raise TypeError("intersection requires another PathSet")
        other_keys = {path_key(path) for path in other}
        return PathSet.from_paths(path for path in self if path_key(path) in other_keys)

    def difference(self, other):
        if not isinstance(other, PathSet):
            raise TypeError("difference requires another PathSet")
        other_keys = {path_key(path) for path in other}
        return PathSet.from_paths(path for path in self if path_key(path) not in other_keys)

    def format_path(self, path):
        if not isinstance(path, list):
            raise TypeError("path must be a list of resolved trace nodes")
        if not path:
            return "No decision points"
        lines = []
        for step, decision in enumerate(path, start=1):
            choice = decision.choices[0]
            prefix = f"  Step {step} ({decision.name}): "
            if isinstance(choice, tuple) and len(choice) == 2 and all(isinstance(item, tuple) for item in choice):
                nearby_indices, _ = choice
                description = "standard absolute value" if not nearby_indices else f"ambiguous indices {list(nearby_indices)}"
            elif isinstance(choice, tuple) and len(choice) >= 2 and isinstance(choice[0], tuple):
                nearby_indices, choice_int = choice[:2]
                flipped = [index for bit, index in enumerate(nearby_indices) if (choice_int >> bit) & 1]
                description = "standard choice" if not flipped else f"flip indices {flipped}"
            elif isinstance(choice, tuple):
                indices = [index for index, selected in enumerate(choice) if selected]
                description = "standard absolute value" if not indices else f"negate at indices {indices}"
            else:
                description = f"scalar choice = {choice}"
            lines.append(prefix + description)
        return "\n".join(lines)


class _SensitivityTensor:
    __array_priority__ = 1000

    def __init__(self, value, sensitivity=None):
        self.value = value
        self.sensitivity = sensitivity

    def __len__(self):
        return len(self.value)

    def __getitem__(self, index):
        if self.sensitivity is None:
            sensitivity = None
        elif isinstance(index, tuple):
            sensitivity = self.sensitivity[(slice(None),) + index]
        else:
            sensitivity = self.sensitivity[(slice(None), index)]
        return _SensitivityTensor(self.value[index], sensitivity)

    @staticmethod
    def _value(other):
        return other.value if isinstance(other, _SensitivityTensor) else other

    @staticmethod
    def _sensitivity(other, value, directions):
        if isinstance(other, _SensitivityTensor) and other.sensitivity is not None:
            return other.sensitivity
        return jnp.zeros((directions,) + jnp.shape(value), dtype=jnp.result_type(value, float))

    def _binary(self, other, operation, sensitivity_operation):
        other_value = self._value(other)
        value = operation(self.value, other_value)
        directions = _sensitivity_directions(self.sensitivity, getattr(other, "sensitivity", None))
        if directions is None:
            sensitivity = None
        else:
            left = _align_sensitivity(self._sensitivity(self, self.value, directions), value)
            right = _align_sensitivity(self._sensitivity(other, other_value, directions), value)
            sensitivity = sensitivity_operation(self.value, other_value, left, right)
        return _SensitivityTensor(value, sensitivity)

    def _rbinary(self, other, operation, sensitivity_operation):
        other_value = self._value(other)
        value = operation(other_value, self.value)
        directions = _sensitivity_directions(getattr(other, "sensitivity", None), self.sensitivity)
        if directions is None:
            sensitivity = None
        else:
            left = _align_sensitivity(self._sensitivity(other, other_value, directions), value)
            right = _align_sensitivity(self._sensitivity(self, self.value, directions), value)
            sensitivity = sensitivity_operation(other_value, self.value, left, right)
        return _SensitivityTensor(value, sensitivity)

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b, lambda a, b, da, db: da + db)

    def __radd__(self, other):
        return self._rbinary(other, lambda a, b: a + b, lambda a, b, da, db: da + db)

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b, lambda a, b, da, db: da + db)

    def __rsub__(self, other):
        return self._rbinary(other, lambda a, b: a - b, lambda a, b, da, db: da + db)

    def __mul__(self, other):
        return self._binary(other, lambda a, b: a * b, lambda a, b, da, db: jnp.abs(b) * da + jnp.abs(a) * db)

    def __rmul__(self, other):
        return self._rbinary(other, lambda a, b: a * b, lambda a, b, da, db: jnp.abs(b) * da + jnp.abs(a) * db)

    def __truediv__(self, other):
        return self._binary(other, lambda a, b: a / b, _division_sensitivity)

    def __rtruediv__(self, other):
        return self._rbinary(other, lambda a, b: a / b, _division_sensitivity)

    def __neg__(self):
        return _SensitivityTensor(-self.value, self.sensitivity)


def _sensitivity_directions(*sensitivities):
    for sensitivity in sensitivities:
        if sensitivity is not None:
            return sensitivity.shape[0]
    return None


def _align_sensitivity(sensitivity, value):
    while sensitivity.ndim < jnp.ndim(value) + 1:
        sensitivity = jnp.expand_dims(sensitivity, -1)
    return jnp.broadcast_to(sensitivity, (sensitivity.shape[0],) + jnp.shape(value))


def _division_sensitivity(a, b, da, db):
    denominator = jnp.maximum(jnp.abs(b), jnp.finfo(jnp.result_type(b, float)).eps)
    return da / denominator + jnp.abs(a) * db / denominator ** 2


def _sensitivity_scale(sensitivity):
    if sensitivity is None:
        return 0.0
    return jnp.max(sensitivity, axis=0)


def _tolerance(reference, sensitivity_scale=0.0):
    state = _branch_state.get()
    tolerance = state.atol + state.rtol * jnp.abs(reference)
    if state.tol_mode == "input_scaled":
        tolerance = tolerance * sensitivity_scale
    return tolerance


def _near(difference, tolerance):
    result = difference <= tolerance
    if _branch_state.get().tol_mode == "input_scaled":
        result = jnp.logical_and(result, tolerance > 0)
    return result


def _input_size(value):
    if isinstance(value, tuple) or isinstance(value, list):
        return builtins.sum(_input_size(item) for item in value)
    if isinstance(value, dict):
        return builtins.sum(_input_size(item) for item in value.values())
    if hasattr(value, "shape"):
        return int(jnp.size(value))
    return 0


def _wrap_sensitive_inputs(value, directions, position):
    if isinstance(value, tuple):
        wrapped = []
        for item in value:
            wrapped_item, position = _wrap_sensitive_inputs(item, directions, position)
            wrapped.append(wrapped_item)
        return tuple(wrapped), position
    if isinstance(value, list):
        wrapped = []
        for item in value:
            wrapped_item, position = _wrap_sensitive_inputs(item, directions, position)
            wrapped.append(wrapped_item)
        return wrapped, position
    if isinstance(value, dict):
        wrapped = {}
        for key, item in value.items():
            wrapped[key], position = _wrap_sensitive_inputs(item, directions, position)
        return wrapped, position
    if hasattr(value, "shape"):
        size = int(jnp.size(value))
        sensitivity = jnp.zeros((directions, size), dtype=jnp.result_type(value, float))
        indices = jnp.arange(size)
        sensitivity = sensitivity.at[position + indices, indices].set(1.0)
        return _SensitivityTensor(value, sensitivity.reshape((directions,) + value.shape)), position + size
    return value, position


def _unwrap_sensitive(value):
    if isinstance(value, _SensitivityTensor):
        return value.value
    if isinstance(value, tuple):
        return tuple(_unwrap_sensitive(item) for item in value)
    if isinstance(value, list):
        return [_unwrap_sensitive(item) for item in value]
    if isinstance(value, dict):
        return {key: _unwrap_sensitive(item) for key, item in value.items()}
    return value


def _resolve_options(atol, rtol, tol_mode, abs_policy):
    if tol_mode not in {"local", "input_scaled"}:
        raise ValueError("tol_mode must be 'local' or 'input_scaled'")
    if abs_policy not in {"zero", "enumerate"}:
        raise ValueError("abs_policy must be 'zero' or 'enumerate'")
    return atol, rtol, tol_mode, abs_policy


@contextmanager
def _branch_mode(mode, atol=0, rtol=0, tol_mode="local", abs_policy="zero", replay_path=None, trace=None, selectors=None, batch_arrays=None):
    logger.debug("Entering _branch_mode: mode=%s, atol=%s, rtol=%s, tol_mode=%s", mode, atol, rtol, tol_mode)

    state = _BranchState(mode=mode)
    if mode == "record":
        state = replace(
            state,
            trace=[],
            atol=atol,
            rtol=rtol,
            tol_mode=tol_mode,
            abs_policy=abs_policy,
        )
    elif mode == "replay":
        if replay_path is None:
            raise ValueError("replay_path must be provided in replay mode")
        if isinstance(replay_path, list):
            path = tuple(replay_path)
        else:
            raise TypeError("replay_path must be a resolved path list")
        state = replace(state, replay_path=path)
    elif mode == "vmap_replay":
        if trace is None or selectors is None:
            raise ValueError("trace and selectors must be provided in vmap_replay mode")
        state = replace(state, trace=tuple(trace), selectors=tuple(selectors))
    elif mode == "batch_replay":
        if batch_arrays is None:
            raise ValueError("batch_arrays must be provided in batch_replay mode")
        state = replace(state, batch_arrays=tuple(batch_arrays))
    else:
        raise ValueError(f"Unexpected branch recording mode {mode}.")

    token = _branch_state.set(state)
    try:
        yield state.trace if mode == "record" else None
    finally:
        _branch_state.reset(token)
        logger.debug("Exiting _branch_mode")


def _trace_append(name, choices):
    trace = _branch_state.get().trace
    logger.debug("_trace_append: name=%s, num_choices=%s", name, len(choices))
    trace.append(_TraceNode(name, choices))


def _trace_popf(name):
    state = _branch_state.get()
    replay = state.replay_path
    pos = state.replay_pos
    if replay is None:
        raise ValueError("No path provided for replay mode")
    if pos >= len(replay):
        raise ValueError("Path exhausted")
    node = replay[pos]
    _branch_state.set(replace(state, replay_pos=pos + 1))
    if node.name != name:
        raise ValueError(f"Expected trace node {name}, got {node.name}")
    return node.choices[0]


def _trace_popf_vmap(name):
    state = _branch_state.get()
    trace = state.trace
    pos = state.replay_pos
    selectors = state.selectors
    if trace is None:
        raise ValueError("No trace provided for vmap_replay mode")
    if pos >= len(trace):
        raise ValueError("Path exhausted")
    node = trace[pos]
    selector = selectors[pos]
    _branch_state.set(replace(state, replay_pos=pos + 1))
    if node.name != name:
        raise ValueError(f"Expected trace node {name}, got {node.name}")
    return node, selector


def _batch_popf(name):
    state = _branch_state.get()
    arrays = state.batch_arrays
    pos = state.replay_pos
    if arrays is None:
        raise ValueError("No batch arrays provided for batch_replay mode")
    if pos >= len(arrays):
        raise ValueError("Batch arrays exhausted")
    entry = arrays[pos]
    _branch_state.set(replace(state, replay_pos=pos + 1))
    if entry[0] != name:
        raise ValueError(f"Expected trace node {name}, got {entry[0]}")
    return entry[1:]


class _HashTensor:
    def __init__(self, value, sensitivity=None):
        if isinstance(value, _SensitivityTensor):
            sensitivity = value.sensitivity
            value = value.value
        logger.debug("_HashTensor.__init__: value=%s", value)
        self.value = value
        self.sensitivity = sensitivity

    def __repr__(self):
        return f'_HashTensor({self.value})'

    def __str__(self):
        return f'_HashTensor({self.value})'

    @staticmethod
    def _unwrap(other):
        if isinstance(other, _HashTensor):
            return other.value
        return other

    def __add__(self, other):
        return _HashTensor(self.value + self._unwrap(other))

    def __radd__(self, other):
        return _HashTensor(self._unwrap(other) + self.value)

    def __sub__(self, other):
        return _HashTensor(self.value - self._unwrap(other))

    def __rsub__(self, other):
        return _HashTensor(self._unwrap(other) - self.value)

    def __mul__(self, other):
        return _HashTensor(self.value * self._unwrap(other))

    def __rmul__(self, other):
        return _HashTensor(self._unwrap(other) * self.value)

    def __truediv__(self, other):
        return _HashTensor(self.value / self._unwrap(other))

    def __rtruediv__(self, other):
        return _HashTensor(self._unwrap(other) / self.value)


def max(inval):
    logger.debug("max: input=%s", inval.value)
    flat_value = jnp.ravel(inval.value)
    if _branch_state.get().mode == "record":
        loc = jnp.argmax(flat_value)
        val = flat_value[loc]
        sensitivity_scale = _sensitivity_scale(inval.sensitivity)
        flat_scale = jnp.ravel(sensitivity_scale)
        selected_scale = flat_scale[loc] if jnp.ndim(sensitivity_scale) else sensitivity_scale
        scale = jnp.maximum(sensitivity_scale, selected_scale)
        tolerance = _tolerance(val, scale)
        nearby_locs, = jnp.where(jnp.ravel(_near(val - inval.value, tolerance)))
        nearby_locs = tuple(int(x) for x in nearby_locs.tolist())
        logger.debug("max: recording - loc=%s, val=%s, nearby_locs=%s", loc, val, nearby_locs)
        _trace_append("max", nearby_locs)
    elif _branch_state.get().mode == "vmap_replay":
        node, selector = _trace_popf_vmap("max")
        nearby_locs_arr = jnp.asarray(node.choices)
        loc = nearby_locs_arr[selector]
        val = flat_value[loc]
        logger.debug("max: vmap replaying - loc=%s, val=%s", loc, val)
    elif _branch_state.get().mode == "batch_replay":
        (loc,) = _batch_popf("max")
        val = flat_value[loc]
        logger.debug("max: batch replaying - loc=%s, val=%s", loc, val)
    else:
        loc = _trace_popf("max")
        val = flat_value[loc]
        logger.debug("max: replaying - loc=%s, val=%s", loc, val)
    sensitivity = None
    if inval.sensitivity is not None:
        if _branch_state.get().mode == "record":
            flat_sensitivity = inval.sensitivity.reshape((inval.sensitivity.shape[0], -1))
            sensitivity = jnp.max(flat_sensitivity[:, jnp.array(nearby_locs)], axis=1)
        else:
            sensitivity = inval.sensitivity.reshape((inval.sensitivity.shape[0], -1))[:, loc]
    return _HashTensor(val, sensitivity)


def min(inval):
    logger.debug("min: input=%s", inval.value)
    flat_value = jnp.ravel(inval.value)
    if _branch_state.get().mode == "record":
        loc = jnp.argmin(flat_value)
        val = flat_value[loc]
        sensitivity_scale = _sensitivity_scale(inval.sensitivity)
        flat_scale = jnp.ravel(sensitivity_scale)
        selected_scale = flat_scale[loc] if jnp.ndim(sensitivity_scale) else sensitivity_scale
        scale = jnp.maximum(sensitivity_scale, selected_scale)
        tolerance = _tolerance(val, scale)
        nearby_locs, = jnp.where(jnp.ravel(_near(inval.value - val, tolerance)))
        nearby_locs = tuple(int(x) for x in nearby_locs.tolist())
        logger.debug("min: recording - loc=%s, val=%s, nearby_locs=%s", loc, val, nearby_locs)
        _trace_append("min", nearby_locs)
    elif _branch_state.get().mode == "vmap_replay":
        node, selector = _trace_popf_vmap("min")
        nearby_locs_arr = jnp.asarray(node.choices)
        loc = nearby_locs_arr[selector]
        val = flat_value[loc]
        logger.debug("min: vmap replaying - loc=%s, val=%s", loc, val)
    elif _branch_state.get().mode == "batch_replay":
        (loc,) = _batch_popf("min")
        val = flat_value[loc]
        logger.debug("min: batch replaying - loc=%s, val=%s", loc, val)
    else:
        loc = _trace_popf("min")
        val = flat_value[loc]
        logger.debug("min: replaying - loc=%s, val=%s", loc, val)
    sensitivity = None
    if inval.sensitivity is not None:
        if _branch_state.get().mode == "record":
            flat_sensitivity = inval.sensitivity.reshape((inval.sensitivity.shape[0], -1))
            sensitivity = jnp.max(flat_sensitivity[:, jnp.array(nearby_locs)], axis=1)
        else:
            sensitivity = inval.sensitivity.reshape((inval.sensitivity.shape[0], -1))[:, loc]
    return _HashTensor(val, sensitivity)


def _elementwise_minmax(one, two, name, jnp_op, prefer_first):
    logger.debug("%s: one=%s, two=%s", name, one.value, two.value)
    if _branch_state.get().mode == "record":
        directions = _sensitivity_directions(one.sensitivity, two.sensitivity)
        sensitivity = None
        if directions is not None:
            one_sensitivity = _SensitivityTensor._sensitivity(one, one.value, directions)
            two_sensitivity = _SensitivityTensor._sensitivity(two, two.value, directions)
            sensitivity = jnp.maximum(one_sensitivity, two_sensitivity)
        reference = jnp.maximum(jnp.abs(one.value), jnp.abs(two.value))
        near = _near(jnp.abs(one.value - two.value), _tolerance(reference, _sensitivity_scale(sensitivity)))
        nearby_indices = tuple(int(x) for x in jnp.where(jnp.ravel(near))[0].tolist())
        logger.debug("%s: recording - nearby_indices=%s", name, nearby_indices)
        base_pick_two = tuple(bool(x) for x in jnp.ravel(jnp_op(one.value, two.value) == two.value).tolist())
        # Each nearby element contributes an independent binary branch.
        n_choices = 2 ** len(nearby_indices)
        choices = [(nearby_indices, i, base_pick_two) for i in range(n_choices)]
        _trace_append(name, choices)
    elif _branch_state.get().mode == "vmap_replay":
        node, selector = _trace_popf_vmap(name)
        nearby_indices, _, base_pick_two = node.choices[0]
        m = len(base_pick_two)
        nearby_arr = np.array(nearby_indices, dtype=np.intp)
        base_arr = np.array(base_pick_two)
        bits = (selector >> jnp.arange(len(nearby_indices))) & 1
        flip = jnp.zeros(m, dtype=bool).at[nearby_arr].set(bits.astype(bool))
        pick_two = _reshape_pick_two(jnp.logical_xor(base_arr, flip), one.value, two.value)
        result = _HashTensor(jnp.where(pick_two, two.value, one.value))
        logger.debug("%s: vmap replaying - pick_two=%s, result=%s", name, pick_two, result.value)
        return result
    elif _branch_state.get().mode == "batch_replay":
        (pick_two,) = _batch_popf(name)
        pick_two = _reshape_pick_two(pick_two, one.value, two.value)
        result = _HashTensor(jnp.where(pick_two, two.value, one.value))
        logger.debug("%s: batch replaying - pick_two=%s, result=%s", name, pick_two, result.value)
        return result
    else:
        nearby_indices, choice_int, base_pick_two = _trace_popf(name)
        pick_two = _reshape_pick_two(
            _resolve_pick_two(nearby_indices, choice_int, base_pick_two),
            one.value,
            two.value,
        )
        result = _HashTensor(jnp.where(pick_two, two.value, one.value))
        logger.debug("%s: replaying - pick_two=%s, result=%s", name, pick_two, result.value)
        return result
    value = jnp_op(one.value, two.value)
    sensitivity = None
    directions = _sensitivity_directions(one.sensitivity, two.sensitivity)
    if directions is not None:
        one_sensitivity = _SensitivityTensor._sensitivity(one, one.value, directions)
        two_sensitivity = _SensitivityTensor._sensitivity(two, two.value, directions)
        if prefer_first:
            selected = jnp.where(jnp.expand_dims(one.value >= two.value, 0), one_sensitivity, two_sensitivity)
        else:
            selected = jnp.where(jnp.expand_dims(one.value <= two.value, 0), one_sensitivity, two_sensitivity)
        nearby = jnp.zeros_like(one.value, dtype=bool)
        if nearby.ndim == 0:
            nearby = jnp.asarray(len(nearby_indices) > 0)
        elif len(nearby_indices) > 0:
            nearby = nearby.ravel().at[jnp.array(nearby_indices)].set(True).reshape(nearby.shape)
        sensitivity = jnp.where(
            jnp.expand_dims(nearby, 0),
            jnp.maximum(one_sensitivity, two_sensitivity),
            selected,
        )
    return _HashTensor(value, sensitivity)


def _resolve_pick_two(nearby_indices, choice_int, base_pick_two):
    pick_two = list(base_pick_two)
    for j, idx in enumerate(nearby_indices):
        if (choice_int >> j) & 1:
            pick_two[idx] = not pick_two[idx]
    return np.array(pick_two, dtype=bool)


def _reshape_pick_two(pick_two, one, two):
    shape = jnp.broadcast_shapes(jnp.shape(one), jnp.shape(two))
    return jnp.reshape(pick_two, shape)


def maximum(one, two):
    return _elementwise_minmax(one, two, "maximum", jnp.maximum, prefer_first=True)


def minimum(one, two):
    return _elementwise_minmax(one, two, "minimum", jnp.minimum, prefer_first=False)


def sum(inval):
    logger.debug("sum: input=%s", inval.value)
    sensitivity = None
    if inval.sensitivity is not None:
        axes = tuple(range(1, inval.sensitivity.ndim))
        sensitivity = jnp.sum(inval.sensitivity, axis=axes)
    result = _HashTensor(jnp.sum(inval.value), sensitivity)
    logger.debug("sum: result=%s", result.value)
    return result


def _abs_from_masks(value, ambiguous, negate):
    linear_part = jnp.where(negate, -value, value)
    zero_grad_part = jax.lax.stop_gradient(jnp.abs(value))
    return jnp.where(ambiguous, zero_grad_part, linear_part)


def _abs_from_choice(value, nearby_indices, base_negate):
    shape = jnp.shape(value)
    negate = np.array(base_negate).reshape(-1)
    ambiguous = np.zeros(len(base_negate), dtype=bool)
    if len(nearby_indices) > 0:
        ambiguous[np.array(nearby_indices, dtype=np.intp)] = True

    return _abs_from_masks(value, ambiguous.reshape(shape), negate.reshape(shape))


def _abs_from_branch(value, nearby_indices, choice_int, base_negate):
    negate = jnp.asarray(base_negate, dtype=bool)
    if nearby_indices:
        nearby = jnp.asarray(nearby_indices, dtype=jnp.int32)
        flips = ((choice_int >> jnp.arange(len(nearby_indices))) & 1).astype(bool)
        negate = negate.at[nearby].set(jnp.logical_xor(negate[nearby], flips))
    return jnp.where(negate.reshape(jnp.shape(value)), -value, value)


def abs(inval):
    logger.debug("abs: input=%s", inval.value)
    if _branch_state.get().mode == "record":
        tolerance = _tolerance(inval.value, _sensitivity_scale(inval.sensitivity))
        nearby_indices = jnp.where(jnp.ravel(_near(jnp.abs(inval.value), tolerance)))[0]
        nearby_indices = tuple(int(x) for x in nearby_indices.tolist())
        logger.debug("abs: recording - nearby_indices=%s", nearby_indices)
        base_negate = tuple(bool(x) for x in jnp.ravel(inval.value < 0).tolist())
        if _branch_state.get().abs_policy == "enumerate":
            choices = [
                (nearby_indices, choice_int, base_negate)
                for choice_int in range(2 ** len(nearby_indices))
            ]
        else:
            choices = [(nearby_indices, base_negate)]
        _trace_append("abs", choices)
        if _branch_state.get().abs_policy == "enumerate":
            value = _abs_from_branch(inval.value, nearby_indices, 0, base_negate)
        else:
            value = _abs_from_choice(inval.value, nearby_indices, base_negate)
        result = _HashTensor(value, inval.sensitivity)
        logger.debug("abs: recording - result=%s", result.value)
        return result
    elif _branch_state.get().mode == "vmap_replay":
        node, selector = _trace_popf_vmap("abs")
        choice = node.choices[0]
        if len(choice) == 3:
            nearby_indices, _, base_negate = choice
            result = _HashTensor(_abs_from_branch(inval.value, nearby_indices, selector, base_negate))
        else:
            nearby_indices, base_negate = choice
            result = _HashTensor(_abs_from_choice(inval.value, nearby_indices, base_negate))
        logger.debug("abs: vmap replaying - result=%s", result.value)
        return result
    elif _branch_state.get().mode == "batch_replay":
        ambiguous, negate = _batch_popf("abs")
        ambiguous = jnp.reshape(ambiguous, jnp.shape(inval.value))
        negate = jnp.reshape(negate, jnp.shape(inval.value))
        result = _HashTensor(_abs_from_masks(inval.value, ambiguous, negate))
        logger.debug("abs: batch replaying - result=%s", result.value)
        return result
    else:
        choice = _trace_popf("abs")
        if len(choice) == 3:
            nearby_indices, choice_int, base_negate = choice
            result = _HashTensor(_abs_from_branch(inval.value, nearby_indices, choice_int, base_negate))
        else:
            nearby_indices, base_negate = choice
            result = _HashTensor(_abs_from_choice(inval.value, nearby_indices, base_negate))
        logger.debug("abs: replaying - result=%s", result.value)
        return result


def record(fun, *, atol=0.0, rtol=0.0, tol_mode="local", abs_policy="zero"):
    atol, rtol, tol_mode, abs_policy = _resolve_options(atol, rtol, tol_mode, abs_policy)

    def recorded(*args, **kwargs):
        with _branch_mode("record", atol=atol, rtol=rtol, tol_mode=tol_mode, abs_policy=abs_policy) as trace:
            if tol_mode == "input_scaled":
                directions = _input_size(args) + _input_size(kwargs)
                wrapped_args, position = _wrap_sensitive_inputs(args, directions, 0)
                wrapped_kwargs, _ = _wrap_sensitive_inputs(kwargs, directions, position)
                value = _unwrap_sensitive(fun(*wrapped_args, **wrapped_kwargs))
            else:
                value = fun(*args, **kwargs)
        paths = PathSet.from_trace(trace)
        return value, paths
    return recorded


def replay(fun, path):
    def replayed(*args, **kwargs):
        with _branch_mode("replay", replay_path=path):
            value = fun(*args, **kwargs)
        return value
    return replayed


def grad(fun, argnums=0, has_aux=False, *, atol=0.0, rtol=0.0, tol_mode="local", abs_policy="zero"):
    def grad_fn(*args, **kwargs):
        _, paths = record(fun, atol=atol, rtol=rtol, tol_mode=tol_mode, abs_policy=abs_policy)(*args, **kwargs)
        default_path = paths[0]

        with _branch_mode("replay", replay_path=default_path):
            jax_grad_fn = jax.grad(fun, argnums=argnums, has_aux=has_aux)
            grad_result = jax_grad_fn(*args, **kwargs)

        if has_aux:
            grads, aux = grad_result
            return (grads, aux), paths
        else:
            return grad_result, paths
    return grad_fn


def value_and_grad(fun, argnums=0, has_aux=False, *, atol=0.0, rtol=0.0, tol_mode="local", abs_policy="zero"):
    def val_grad_fn(*args, **kwargs):
        record_result, paths = record(fun, atol=atol, rtol=rtol, tol_mode=tol_mode, abs_policy=abs_policy)(*args, **kwargs)
        default_path = paths[0]

        if has_aux:
            record_value = record_result[0] if isinstance(record_result, tuple) else record_result
        else:
            record_value = record_result

        with _branch_mode("replay", replay_path=default_path):
            jax_grad_fn = jax.grad(fun, argnums=argnums, has_aux=has_aux)
            grad_result = jax_grad_fn(*args, **kwargs)

        if has_aux:
            grads, aux = grad_result
            return ((record_value, aux), grads), paths
        else:
            return (record_value, grad_result), paths
    return val_grad_fn


def replay_grad(fun, path, argnums=0, has_aux=False):
    def replayed_grad(*args, **kwargs):
        with _branch_mode("replay", replay_path=path):
            jax_grad_fn = jax.grad(fun, argnums=argnums, has_aux=has_aux)
            grad_result = jax_grad_fn(*args, **kwargs)

        if has_aux:
            grads, aux = grad_result
            return grads, aux
        else:
            return grad_result
    return replayed_grad


def _replay_value_and_grad(fun, path, jax_vg_fn, has_aux):

    def replayed_val_grad(*args, **kwargs):
        with _branch_mode("replay", replay_path=path):
            vg_result = jax_vg_fn(*args, **kwargs)

        if has_aux:
            (value, aux), grads = vg_result
            return (value, aux), grads
        else:
            value, grads = vg_result
            return value, grads
    return replayed_val_grad


def replay_value_and_grad(fun, path, argnums=0, has_aux=False):
    return _replay_value_and_grad(
        fun,
        path,
        jax.value_and_grad(fun, argnums=argnums, has_aux=has_aux),
        has_aux,
    )


def _build_batch_leaves(paths, names):
    """Convert J already-resolved paths' choices into dense, vmap-able arrays.

    Unlike all_value_and_grad's selector arrays (which index into a PathSet's
    Cartesian-product enumeration), this only ever reads path[i].choices[0] for each
    of the J given paths -- there is no enumeration here, so this cannot blow up
    combinatorially regardless of how many ties a record-time trace had.
    """
    J = len(paths)
    flat_leaves = []
    leaf_layout = []
    for i, name in enumerate(names):
        choices = [path[i].choices[0] for path in paths]
        if name in ("max", "min"):
            loc_arr = jnp.asarray(choices, dtype=jnp.int32)
            flat_leaves.append(loc_arr)
            leaf_layout.append((name, 1))
        elif name == "abs":
            enumerated = len(choices[0]) == 3
            p = len(choices[0][2] if enumerated else choices[0][1])
            negate = np.zeros((J, p), dtype=bool)
            ambiguous = np.zeros((J, p), dtype=bool)
            for k, choice in enumerate(choices):
                if enumerated:
                    nearby_indices, choice_int, base_negate = choice
                    negate[k] = np.asarray(base_negate, dtype=bool)
                    for bit, index in enumerate(nearby_indices):
                        if (choice_int >> bit) & 1:
                            negate[k, index] = not negate[k, index]
                else:
                    nearby_indices, base_negate = choice
                    negate[k] = np.asarray(base_negate, dtype=bool)
                    if len(nearby_indices) > 0:
                        ambiguous[k, np.array(nearby_indices, dtype=np.intp)] = True
            flat_leaves.append(jnp.asarray(ambiguous))
            flat_leaves.append(jnp.asarray(negate))
            leaf_layout.append((name, 2))
        elif name in ("maximum", "minimum"):
            pick_two = np.stack([_resolve_pick_two(*c) for c in choices])
            flat_leaves.append(jnp.asarray(pick_two))
            leaf_layout.append((name, 1))
        else:
            raise ValueError(f"replay_value_and_grad_batch: unsupported traced op {name!r}")
    return flat_leaves, leaf_layout


def _bucket_size(J):
    """Round J up to the next power of two so the jitted vmap_body only ever sees
    O(log2(J_max)) distinct batch shapes across a run, instead of a fresh XLA compile
    for every J value encountered (J varies a lot call-to-call in practice)."""
    if J <= 1:
        return 1
    return 1 << (J - 1).bit_length()


def _get_jit_batched_vg(fun, leaf_layout, n_args, argnums, has_aux):
    """Return a cached jax.jit-compiled (vmap . value_and_grad) callable for the given
    (fun, op-sequence) shape, building and caching it on first use.

    `leaf_layout` is a pure function of the traced op-name sequence (not of J or the
    concrete choice values), so it's a safe, stable cache key: every call whose fun and
    control-flow shape match reuses the same jitted wrapper. JAX's own shape-based
    dispatch then recompiles only when it sees a new J (batch size) and otherwise reuses
    the compiled program -- exactly what we want once J plateaus near convergence.

    This must be constructed once and reused, not rebuilt+jitted per call: jax.jit on a
    freshly-built function object every call gets zero cache reuse (jit's cache is keyed
    to the wrapped function's identity), so it would only add compilation overhead on
    top of the existing eager cost.

    Safe to cache because the per-path choice data always flows in as traced vmap
    arguments (`inner_args`/`flat_node_leaves` below), never as closed-over Python
    constants -- so reusing the compiled program across calls with different H0 entries
    cannot go stale.
    """
    key = (fun, tuple(leaf_layout), n_args, argnums, has_aux)
    cached = _jit_batch_cache.get(key)
    if cached is not None:
        return cached

    def vmap_body(*inner_args):
        call_args = inner_args[:n_args]
        flat_node_leaves = inner_args[n_args:]
        batch_arrays = []
        idx = 0
        for name, n_leaves in leaf_layout:
            batch_arrays.append((name, *flat_node_leaves[idx:idx + n_leaves]))
            idx += n_leaves
        with _branch_mode("batch_replay", batch_arrays=batch_arrays):
            return fun(*call_args)

    jax_vg_fn = jax.value_and_grad(vmap_body, argnums=argnums, has_aux=has_aux)
    # This module defines its own sum, so use a loop for Python integers.
    total_leaves = 0
    for _, n_leaves in leaf_layout:
        total_leaves += n_leaves
    in_axes = (None,) * n_args + (0,) * total_leaves
    vmap_vg_fn = jax.vmap(jax_vg_fn, in_axes=in_axes)
    jitted = jax.jit(vmap_vg_fn)
    _jit_batch_cache[key] = jitted
    return jitted


def replay_value_and_grad_batch(fun, paths, argnums=0, has_aux=False):
    """Batch replay+grad of `fun` over J independently-obtained, fully-resolved paths.

    This is the batching mechanism used by the IBCDFO integration: H0 is a Python list of
    length J of already-resolved paths (e.g. accumulated by choose_generator_set from
    several distinct nearby points), not a PathSet to enumerate. It vmaps a single
    jax.value_and_grad(fun) call over all J paths at once, instead of dispatching J
    separate unbatched jax calls (~11.5ms of dispatch overhead each). Because it only
    ever reads each path's already-narrowed choices[0] and never constructs a PathSet
    or calls _iter_positions, its cost is O(J) by construction -- it cannot reintroduce
    the combinatorial explosion that vmapping over a PathSet's full enumeration caused
    for the one-norm (abs-heavy) hfun.

    The vmap(value_and_grad(...)) callable itself is jit-compiled and cached across
    calls (see _get_jit_batched_vg) -- without that, every call pays full un-jitted JAX
    dispatch overhead for every op regardless of batching.
    """
    paths = list(paths)
    J = len(paths)
    if J == 0:
        raise ValueError("replay_value_and_grad_batch requires at least one path")
    if not all(isinstance(path, list) for path in paths):
        raise TypeError("paths must contain resolved path lists")
    names = [node.name for node in paths[0]]
    if any([node.name for node in path] != names for path in paths[1:]):
        raise ValueError(
            "replay_value_and_grad_batch requires all paths to share the same "
            "op-name sequence (same traced control flow)."
        )
    encoded_leaves, leaf_layout = _build_batch_leaves(paths, names)

    def batched_val_grad(*args, **kwargs):
        if kwargs:
            raise TypeError("replay_value_and_grad_batch does not support kwargs")
        n_args = len(args)
        flat_leaves = encoded_leaves

        # Pad the batch axis up to a power-of-two bucket so the jitted vmap_body only
        # recompiles for a new bucket size, not for every distinct J -- see
        # _bucket_size. Safe because vmap's batching is embarrassingly parallel (no
        # reduction across the batch axis inside vmap_body/fun), so repeating the last
        # path's leaf data into the padding lanes cannot affect the real lanes' values
        # or grads; slicing back to [:J] below reproduces an unpadded call exactly.
        padded_J = _bucket_size(J)
        if padded_J != J:
            pad_amount = padded_J - J
            flat_leaves = [
                jnp.pad(leaf, [(0, pad_amount)] + [(0, 0)] * (leaf.ndim - 1), mode="edge")
                for leaf in flat_leaves
            ]

        vmap_vg_fn = _get_jit_batched_vg(fun, leaf_layout, n_args, argnums, has_aux)
        vg_out = vmap_vg_fn(*args, *flat_leaves)

        if padded_J != J:
            vg_out = jax.tree_util.tree_map(lambda a: a[:J], vg_out)

        if has_aux:
            (values, aux), grads = vg_out
            return (values, aux), grads
        else:
            values, grads = vg_out
            return values, grads
    return batched_val_grad


def _evaluate_paths_sequentially(fun, paths, args, kwargs, argnums, has_aux):
    jax_vg_fn = jax.value_and_grad(fun, argnums=argnums, has_aux=has_aux)
    return [
        _replay_value_and_grad(fun, path, jax_vg_fn, has_aux)(*args, **kwargs)
        for path in paths
    ]


def _evaluate_factorized_paths(fun, paths, args, argnums, has_aux):
    trace = paths._trace
    positions = list(paths._iter_positions())
    selector_arrays = [
        jnp.asarray([position[index] for position in positions], dtype=jnp.int32)
        for index in range(len(trace))
    ]

    def vmap_body(*inner_args):
        call_args = inner_args[:len(args)]
        selectors = inner_args[len(args):]
        with _branch_mode("vmap_replay", trace=trace, selectors=selectors):
            return fun(*call_args)

    jax_vg_fn = jax.value_and_grad(vmap_body, argnums=argnums, has_aux=has_aux)
    vg_out = jax.vmap(jax_vg_fn, in_axes=(None,) * len(args) + (0,) * len(trace))(
        *args,
        *selector_arrays,
    )
    results = []
    if has_aux:
        (values, aux), grads = vg_out
        for index in range(len(positions)):
            item_aux = jax.tree_util.tree_map(lambda value: value[index], aux)
            results.append(((values[index], item_aux), grads[index]))
    else:
        values, grads = vg_out
        results.extend((values[index], grads[index]) for index in range(len(positions)))
    return results


def all_value_and_grad(fun, argnums=0, has_aux=False, *, atol=0.0, rtol=0.0, tol_mode="local", abs_policy="zero"):
    def all_vg_fn(*args, **kwargs):
        _defaultresult, paths = record(fun, atol=atol, rtol=rtol, tol_mode=tol_mode, abs_policy=abs_policy)(*args, **kwargs)

        if kwargs or not paths.has_decisions:
            results = _evaluate_paths_sequentially(fun, paths, args, kwargs, argnums, has_aux)
        else:
            results = _evaluate_factorized_paths(fun, paths, args, argnums, has_aux)
        return results, paths
    return all_vg_fn
