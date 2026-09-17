import logging
from contextlib import contextmanager
from contextvars import ContextVar

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

_is_recording: ContextVar[bool] = ContextVar('_is_recording', default=False)
_recorded_trace: ContextVar[list] = ContextVar('_recorded_trace', default=[])
_replay_path: ContextVar[list] = ContextVar('_replay_path', default=None)
_replay_pos: ContextVar[int] = ContextVar('_replay_pos', default=0)
_tolerance: ContextVar[float] = ContextVar('_tolerance', default=0)
_is_vmap_replay: ContextVar[bool] = ContextVar('_is_vmap_replay', default=False)
_vmap_trace: ContextVar[list] = ContextVar('_vmap_trace', default=None)
_vmap_selectors: ContextVar[tuple] = ContextVar('_vmap_selectors', default=None)
_is_batch_replay: ContextVar[bool] = ContextVar('_is_batch_replay', default=False)
_batch_arrays: ContextVar[tuple] = ContextVar('_batch_arrays', default=None)

# Cache of jax.jit-compiled (vmap . value_and_grad) callables for
# replay_value_and_grad_batch, keyed by (fun, op-sequence, ...) -- see
# _get_jit_batched_vg below. Keyed on `fun` itself (not id(fun)) so the cache entry
# holds a strong reference, ruling out id-reuse after GC.
_jit_batch_cache: dict = {}


class _TraceNode:
    def __init__(self, name, choices):
        logger.debug("_TraceNode.__init__: name=%s, num_choices=%s", name, len(choices))
        self.name = name
        self.choices = list(choices)
        self.num = len(self.choices)
        self.pos = 0

    def __repr__(self):
        return f'_TraceNode(name="{self.name}", pos={self.pos}/{self.num}, current_choice={self.currentChoice()})'

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if not isinstance(other, _TraceNode):
            return NotImplemented
        return self.name == other.name and self.choices == other.choices

    def currentChoice(self):
        choice = self.choices[self.pos]
        logger.debug("_TraceNode.currentChoice: name=%s, pos=%s, choice=%s", self.name, self.pos, choice)
        return choice

    def incrementChoice(self):
        logger.debug("_TraceNode.incrementChoice: name=%s, pos=%s, num=%s", self.name, self.pos, self.num)
        if self.pos + 1 >= self.num:
            self.pos = 0
            return False
        else:
            self.pos += 1
            return True


def path_key(path):
    """Canonical hashable key for a single resolved path.

    A "path" is a list of `_TraceNode`, each already narrowed to exactly one
    choice -- e.g. an element yielded by iterating a `PathSet`, or `PathSet[i]`.
    `_TraceNode` defines `__eq__` but not `__hash__`, so a bare `_TraceNode`
    (and any list containing one) is unhashable in plain Python: `set()`/`dict`
    keyed on paths, or a `path in some_list` check, silently falls back to an
    O(n) linear scan (or raises `TypeError` if you try to hash it directly).
    `path_key` gives callers a plain tuple of hashable primitives (the trace
    node names and their resolved choices, which `_branch_mode` always builds
    out of ints/bools/tuples) so raw paths can be deduplicated or tested for
    membership in O(1) amortized per path instead of O(n).
    """
    return tuple((node.name, node.choices[0]) for node in path)


def paths_equal(path1, path2):
    """Value equality for two individual paths, independent of identity/hashing."""
    if len(path1) != len(path2):
        return False
    for node1, node2 in zip(path1, path2):
        if node1.name != node2.name:
            return False
        if len(node1.choices) != 1 or len(node2.choices) != 1:
            return False
        if node1.choices[0] != node2.choices[0]:
            return False
    return True


def unique_paths(paths):
    """Deduplicate an arbitrary iterable of individual paths, in O(n).

    Keeps the first occurrence of each distinct path (by value, per
    `paths_equal`) and drops later duplicates. Because paths are unhashable,
    doing this by hand (`if p not in seen_list: seen_list.append(p)`) costs
    O(n^2) equality comparisons; `unique_paths` costs O(n) by hashing each
    path's `path_key` instead.
    """
    seen = set()
    out = []
    for p in paths:
        key = path_key(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def paths_any_in(needles, haystack):
    """True if any path in `needles` also appears (by value) in `haystack`."""
    haystack_keys = {path_key(p) for p in haystack}
    return any(path_key(p) in haystack_keys for p in needles)


def paths_all_in(needles, haystack):
    """True if every path in `needles` also appears (by value) in `haystack`."""
    haystack_keys = {path_key(p) for p in haystack}
    return all(path_key(p) in haystack_keys for p in needles)


class PathSet:
    def __init__(self, trace, _empty=False):
        self.trace = trace.copy()
        self._empty = _empty
        logger.debug("PathSet.__init__: trace with %s nodes", len(trace))

    def __iter__(self):
        if self._empty:
            return
        if not self.trace:
            yield []
            return

        for node in self.trace:
            node.pos = 0

        yield self._current_path()

        while self._increment_trace():
            yield self._current_path()

    def _current_path(self):
        return [_TraceNode(node.name, [node.currentChoice()]) for node in self.trace]

    def _increment_trace(self):
        for i in reversed(range(len(self.trace))):
            if self.trace[i].incrementChoice():
                return True
        return False

    def _iter_positions(self):
        if self._empty:
            return
        if not self.trace:
            yield ()
            return

        for node in self.trace:
            node.pos = 0

        yield tuple(node.pos for node in self.trace)

        while self._increment_trace():
            yield tuple(node.pos for node in self.trace)

    def __len__(self):
        if self._empty:
            return 0
        total = 1
        for node in self.trace:
            total *= node.num
        return total

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError("Index must be an integer")

        total_len = len(self)
        if index < 0:
            index = total_len + index

        if index < 0 or index >= total_len:
            raise IndexError(f"Index {index} is out of range for PathSet with {total_len} elements")

        if not self.trace:
            return []

        trace_copy = [_TraceNode(node.name, node.choices) for node in self.trace]

        remaining_index = index
        for i in range(len(trace_copy)):
            node = trace_copy[i]
            combinations_after = 1
            for j in range(i + 1, len(trace_copy)):
                combinations_after *= trace_copy[j].num
            node.pos = remaining_index // combinations_after
            remaining_index = remaining_index % combinations_after

        return [_TraceNode(node.name, [node.currentChoice()]) for node in trace_copy]

    def __contains__(self, item):
        if isinstance(item, list):
            nodes = item
        else:
            return False
        if len(nodes) != len(self.trace):
            return False
        if not nodes and not self.trace:
            return True
        for item_node, trace_node in zip(nodes, self.trace):
            if item_node.name != trace_node.name:
                return False
            if len(item_node.choices) != 1:
                return False
            if item_node.choices[0] not in trace_node.choices:
                return False
        return True

    @staticmethod
    def _paths_equal(path1, path2):
        return paths_equal(path1, path2)

    def __bool__(self):
        return len(self) > 0

    def __repr__(self):
        return f'PathSet(trace_length={len(self.trace)}, total_paths={len(self)})'

    def __str__(self):
        return f'PathSet with {len(self)} possible paths from {len(self.trace)} trace nodes'

    def _path_to_tuple(self, path):
        return path_key(path)

    def _create_trace_from_paths(self, path_tuples):
        if not path_tuples:
            return None

        node_choices = {}
        for path_tuple in path_tuples:
            for i, (node_name, choice) in enumerate(path_tuple):
                if i not in node_choices:
                    node_choices[i] = {'name': node_name, 'choices': []}
                if choice not in node_choices[i]['choices']:
                    node_choices[i]['choices'].append(choice)

        trace = []
        for i in sorted(node_choices.keys()):
            node_info = node_choices[i]
            trace.append(_TraceNode(node_info['name'], node_info['choices']))
        return trace

    def _build_from_tuples(self, path_tuples):
        trace = self._create_trace_from_paths(path_tuples)
        if trace is None:
            return PathSet([], _empty=True)
        return PathSet(trace)

    def union(self, other):
        if not isinstance(other, PathSet):
            raise TypeError("Union requires another PathSet")
        all_paths = set()
        for path in self:
            all_paths.add(self._path_to_tuple(path))
        for path in other:
            all_paths.add(self._path_to_tuple(path))
        return self._build_from_tuples(list(all_paths))

    def intersection(self, other):
        if not isinstance(other, PathSet):
            raise TypeError("Intersection requires another PathSet")
        other_paths = set()
        for path in other:
            other_paths.add(self._path_to_tuple(path))
        common_paths = []
        for path in self:
            pt = self._path_to_tuple(path)
            if pt in other_paths:
                common_paths.append(pt)
        return self._build_from_tuples(common_paths)

    def difference(self, other):
        if not isinstance(other, PathSet):
            raise TypeError("Difference requires another PathSet")
        other_paths = set()
        for path in other:
            other_paths.add(self._path_to_tuple(path))
        diff_paths = []
        for path in self:
            pt = self._path_to_tuple(path)
            if pt not in other_paths:
                diff_paths.append(pt)
        return self._build_from_tuples(diff_paths)

    def format_path(self, path):
        nodes = path
        if not nodes:
            return "No decision points"

        lines = []
        for step, node in enumerate(nodes):
            choice_val = node.choices[0]
            if isinstance(choice_val, tuple) and len(choice_val) == 2 and isinstance(choice_val[0], tuple) and isinstance(choice_val[1], tuple):
                nearby_indices, _base_negate = choice_val
                if len(nearby_indices) == 0:
                    lines.append(f"  Step {step+1} ({node.name}): standard absolute value (no ambiguous indices)")
                else:
                    lines.append(f"  Step {step+1} ({node.name}): ambiguous (zero-grad) indices {list(nearby_indices)}")
            elif isinstance(choice_val, tuple) and len(choice_val) >= 2 and isinstance(choice_val[0], tuple):
                nearby_indices, choice_int = choice_val[0], choice_val[1]
                if isinstance(nearby_indices, tuple):
                    if len(nearby_indices) == 0:
                        lines.append(f"  Step {step+1} ({node.name}): standard choice (no nearby values)")
                    else:
                        flipped_indices = [
                            nearby_indices[j] for j in range(len(nearby_indices))
                            if (choice_int >> j) & 1
                        ]
                        if len(flipped_indices) == 0:
                            lines.append(f"  Step {step+1} ({node.name}): standard choice (no flips)")
                        else:
                            lines.append(f"  Step {step+1} ({node.name}): flip indices {flipped_indices} (from nearby {list(nearby_indices)})")
                else:
                    lines.append(f"  Step {step+1} ({node.name}): indices {nearby_indices}, pattern {choice_int:b}")
            elif isinstance(choice_val, tuple):
                true_indices = [j for j, val in enumerate(choice_val) if val]
                if len(true_indices) == 0:
                    lines.append(f"  Step {step+1} ({node.name}): standard absolute value")
                else:
                    lines.append(f"  Step {step+1} ({node.name}): negate at indices {true_indices}")
            else:
                lines.append(f"  Step {step+1} ({node.name}): scalar choice = {choice_val}")
        return "\n".join(lines)


@contextmanager
def _branch_mode(mode, tol=0, replay_path=None, trace=None, selectors=None, batch_arrays=None):
    logger.debug("Entering _branch_mode: mode=%s, tolerance=%s", mode, tol)

    rec_token = _is_recording.set(False)
    trace_token = _recorded_trace.set([])
    path_token = _replay_path.set(None)
    pos_token = _replay_pos.set(0)
    tol_token = _tolerance.set(0)
    vmap_flag_token = _is_vmap_replay.set(False)
    vmap_trace_token = _vmap_trace.set(None)
    vmap_sel_token = _vmap_selectors.set(None)
    batch_flag_token = _is_batch_replay.set(False)
    batch_arrays_token = _batch_arrays.set(None)

    try:
        if mode == "record":
            _is_recording.set(True)
            _tolerance.set(tol)
            logger.debug("Starting recording mode")

            trace = _recorded_trace.get()

            try:
                yield trace
            finally:
                pass

        elif mode == "replay":
            if replay_path is None:
                raise ValueError("replay_path must be provided in replay mode")

            if isinstance(replay_path, list):
                _replay_path.set(list(replay_path))
            elif isinstance(replay_path, PathSet):
                if len(replay_path) == 1:
                    path = next(iter(replay_path))
                    _replay_path.set(list(path))
                else:
                    raise ValueError("PathSet with multiple paths cannot be used directly as replay_path. Iterate over it first.")
            else:
                raise TypeError(f"Unexpected replay_path type: {type(replay_path)}")

            _replay_pos.set(0)
            logger.debug("Starting replay mode")
            yield

        elif mode == "vmap_replay":
            if trace is None or selectors is None:
                raise ValueError("trace and selectors must be provided in vmap_replay mode")

            _is_vmap_replay.set(True)
            _vmap_trace.set(trace)
            _vmap_selectors.set(tuple(selectors))
            _replay_pos.set(0)
            logger.debug("Starting vmap_replay mode")
            yield

        elif mode == "batch_replay":
            if batch_arrays is None:
                raise ValueError("batch_arrays must be provided in batch_replay mode")

            _is_batch_replay.set(True)
            _batch_arrays.set(tuple(batch_arrays))
            _replay_pos.set(0)
            logger.debug("Starting batch_replay mode")
            yield
        else:
            raise Exception(f"Unexpected branch recording mode {mode}.")
    finally:
        _is_recording.reset(rec_token)
        _recorded_trace.reset(trace_token)
        _replay_path.reset(path_token)
        _replay_pos.reset(pos_token)
        _tolerance.reset(tol_token)
        _is_vmap_replay.reset(vmap_flag_token)
        _vmap_trace.reset(vmap_trace_token)
        _vmap_selectors.reset(vmap_sel_token)
        _is_batch_replay.reset(batch_flag_token)
        _batch_arrays.reset(batch_arrays_token)
        logger.debug("Exiting _branch_mode")


def _trace_append(name, choices):
    trace = _recorded_trace.get()
    logger.debug("_trace_append: name=%s, num_choices=%s", name, len(choices))
    trace.append(_TraceNode(name, choices))


def _trace_popf(name):
    replay = _replay_path.get()
    pos = _replay_pos.get()
    if replay is None:
        raise ValueError("No path provided for replay mode")
    if pos >= len(replay):
        raise ValueError("Path exhausted")
    node = replay[pos]
    _replay_pos.set(pos + 1)
    if node.name != name:
        raise ValueError(f"Expected trace node {name}, got {node.name}")
    return node.choices[0]


def _trace_popf_vmap(name):
    trace = _vmap_trace.get()
    pos = _replay_pos.get()
    selectors = _vmap_selectors.get()
    if trace is None:
        raise ValueError("No trace provided for vmap_replay mode")
    if pos >= len(trace):
        raise ValueError("Path exhausted")
    node = trace[pos]
    selector = selectors[pos]
    _replay_pos.set(pos + 1)
    if node.name != name:
        raise ValueError(f"Expected trace node {name}, got {node.name}")
    return node, selector


def _batch_popf(name):
    arrays = _batch_arrays.get()
    pos = _replay_pos.get()
    if arrays is None:
        raise ValueError("No batch arrays provided for batch_replay mode")
    if pos >= len(arrays):
        raise ValueError("Batch arrays exhausted")
    entry = arrays[pos]
    _replay_pos.set(pos + 1)
    if entry[0] != name:
        raise ValueError(f"Expected trace node {name}, got {entry[0]}")
    return entry[1:]


class HashTensor:
    def __init__(self, value):
        logger.debug("HashTensor.__init__: value=%s", value)
        self.value = value

    def __repr__(self):
        return f'HashTensor({self.value})'

    def __str__(self):
        return f'HashTensor({self.value})'

    @staticmethod
    def _unwrap(other):
        if isinstance(other, HashTensor):
            return other.value
        return other

    def __add__(self, other):
        return HashTensor(self.value + self._unwrap(other))

    def __radd__(self, other):
        return HashTensor(self._unwrap(other) + self.value)

    def __sub__(self, other):
        return HashTensor(self.value - self._unwrap(other))

    def __rsub__(self, other):
        return HashTensor(self._unwrap(other) - self.value)

    def __mul__(self, other):
        return HashTensor(self.value * self._unwrap(other))

    def __rmul__(self, other):
        return HashTensor(self._unwrap(other) * self.value)

    def __truediv__(self, other):
        return HashTensor(self.value / self._unwrap(other))

    def __rtruediv__(self, other):
        return HashTensor(self._unwrap(other) / self.value)


def max(inval):
    logger.debug("max: input=%s", inval.value)
    if _is_recording.get():
        loc = jnp.argmax(inval.value)
        val = inval.value[loc]
        nearby_locs, = jnp.where(inval.value >= val - _tolerance.get())
        nearby_locs = tuple(int(x) for x in nearby_locs.tolist())
        logger.debug("max: recording - loc=%s, val=%s, nearby_locs=%s", loc, val, nearby_locs)
        _trace_append("max", nearby_locs)
    elif _is_vmap_replay.get():
        node, selector = _trace_popf_vmap("max")
        nearby_locs_arr = jnp.asarray(node.choices)
        loc = nearby_locs_arr[selector]
        val = inval.value[loc]
        logger.debug("max: vmap replaying - loc=%s, val=%s", loc, val)
    elif _is_batch_replay.get():
        (loc,) = _batch_popf("max")
        val = inval.value[loc]
        logger.debug("max: batch replaying - loc=%s, val=%s", loc, val)
    else:
        loc = _trace_popf("max")
        val = inval.value[loc]
        logger.debug("max: replaying - loc=%s, val=%s", loc, val)
    return HashTensor(val)


def min(inval):
    logger.debug("min: input=%s", inval.value)
    if _is_recording.get():
        loc = jnp.argmin(inval.value)
        val = inval.value[loc]
        nearby_locs, = jnp.where(inval.value <= val + _tolerance.get())
        nearby_locs = tuple(int(x) for x in nearby_locs.tolist())
        logger.debug("min: recording - loc=%s, val=%s, nearby_locs=%s", loc, val, nearby_locs)
        _trace_append("min", nearby_locs)
    elif _is_vmap_replay.get():
        node, selector = _trace_popf_vmap("min")
        nearby_locs_arr = jnp.asarray(node.choices)
        loc = nearby_locs_arr[selector]
        val = inval.value[loc]
        logger.debug("min: vmap replaying - loc=%s, val=%s", loc, val)
    elif _is_batch_replay.get():
        (loc,) = _batch_popf("min")
        val = inval.value[loc]
        logger.debug("min: batch replaying - loc=%s, val=%s", loc, val)
    else:
        loc = _trace_popf("min")
        val = inval.value[loc]
        logger.debug("min: replaying - loc=%s, val=%s", loc, val)
    return HashTensor(val)


def _elementwise_minmax(one, two, name, jnp_op, prefer_first):
    logger.debug("%s: one=%s, two=%s", name, one.value, two.value)
    if _is_recording.get():
        nearby_indices = jnp.where(jnp.abs(one.value - two.value) <= _tolerance.get())[0]
        nearby_indices = tuple(int(x) for x in nearby_indices.tolist())
        logger.debug("%s: recording - nearby_indices=%s", name, nearby_indices)
        base_pick_two = tuple(bool(x) for x in (jnp_op(one.value, two.value) == two.value).tolist())
        # NOTE: this enumerates 2**len(nearby_indices) choices at record time (unlike
        # abs(), which was fixed in b527254 to always record exactly one choice). This
        # is a known, currently-latent risk: a maximum/minimum call over a vector with
        # many simultaneous ties (e.g. create_censored_L1_loss_hfun_jax) could still
        # explode combinatorially if ever fed through all_value_and_grad's PathSet
        # enumeration. replay_value_and_grad_batch below is NOT affected by this, since
        # it batches over already-resolved paths and never enumerates PathSet choices.
        n_choices = 2 ** len(nearby_indices)
        choices = [(nearby_indices, i, base_pick_two) for i in range(n_choices)]
        _trace_append(name, choices)
    elif _is_vmap_replay.get():
        node, selector = _trace_popf_vmap(name)
        nearby_indices, _, base_pick_two = node.choices[0]
        m = len(base_pick_two)
        nearby_arr = np.array(nearby_indices, dtype=np.intp)
        base_arr = np.array(base_pick_two)
        bits = (selector >> jnp.arange(len(nearby_indices))) & 1
        flip = jnp.zeros(m, dtype=bool).at[nearby_arr].set(bits.astype(bool))
        pick_two = jnp.logical_xor(base_arr, flip)
        result = HashTensor(jnp.where(pick_two, two.value, one.value))
        logger.debug("%s: vmap replaying - pick_two=%s, result=%s", name, pick_two, result.value)
        return result
    elif _is_batch_replay.get():
        (pick_two,) = _batch_popf(name)
        result = HashTensor(jnp.where(pick_two, two.value, one.value))
        logger.debug("%s: batch replaying - pick_two=%s, result=%s", name, pick_two, result.value)
        return result
    else:
        nearby_indices, choice_int, base_pick_two = _trace_popf(name)
        pick_two = _resolve_pick_two(nearby_indices, choice_int, base_pick_two)
        result = HashTensor(jnp.where(pick_two, two.value, one.value))
        logger.debug("%s: replaying - pick_two=%s, result=%s", name, pick_two, result.value)
        return result
    return HashTensor(jnp_op(one.value, two.value))


def _resolve_pick_two(nearby_indices, choice_int, base_pick_two):
    pick_two = list(base_pick_two)
    for j, idx in enumerate(nearby_indices):
        if (choice_int >> j) & 1:
            pick_two[idx] = not pick_two[idx]
    return np.array(pick_two, dtype=bool)


def maximum(one, two):
    return _elementwise_minmax(one, two, "maximum", jnp.maximum, prefer_first=True)


def minimum(one, two):
    return _elementwise_minmax(one, two, "minimum", jnp.minimum, prefer_first=False)


def sum(inval):
    logger.debug("sum: input=%s", inval.value)
    result = HashTensor(jnp.sum(inval.value))
    logger.debug("sum: result=%s", result.value)
    return result


def _abs_from_masks(value, ambiguous, negate):
    linear_part = jnp.where(negate, -value, value)
    zero_grad_part = jax.lax.stop_gradient(jnp.abs(value))
    return jnp.where(ambiguous, zero_grad_part, linear_part)


def _abs_from_choice(value, nearby_indices, base_negate):
    negate = np.array(base_negate)
    ambiguous = np.zeros(len(base_negate), dtype=bool)
    if len(nearby_indices) > 0:
        ambiguous[np.array(nearby_indices, dtype=np.intp)] = True

    return _abs_from_masks(value, ambiguous, negate)


def abs(inval):
    logger.debug("abs: input=%s", inval.value)
    if _is_recording.get():
        nearby_indices = jnp.where(jnp.abs(inval.value) <= _tolerance.get())[0]
        nearby_indices = tuple(int(x) for x in nearby_indices.tolist())
        logger.debug("abs: recording - nearby_indices=%s", nearby_indices)
        base_negate = tuple(bool(x) for x in (inval.value < 0).tolist())
        choices = [(nearby_indices, base_negate)]
        _trace_append("abs", choices)
        result = HashTensor(_abs_from_choice(inval.value, nearby_indices, base_negate))
        logger.debug("abs: recording - result=%s", result.value)
        return result
    elif _is_vmap_replay.get():
        node, _selector = _trace_popf_vmap("abs")
        nearby_indices, base_negate = node.choices[0]
        result = HashTensor(_abs_from_choice(inval.value, nearby_indices, base_negate))
        logger.debug("abs: vmap replaying - result=%s", result.value)
        return result
    elif _is_batch_replay.get():
        ambiguous, negate = _batch_popf("abs")
        result = HashTensor(_abs_from_masks(inval.value, ambiguous, negate))
        logger.debug("abs: batch replaying - result=%s", result.value)
        return result
    else:
        nearby_indices, base_negate = _trace_popf("abs")
        result = HashTensor(_abs_from_choice(inval.value, nearby_indices, base_negate))
        logger.debug("abs: replaying - result=%s", result.value)
        return result


def record(fun, tol=0.0):
    def recorded(*args, **kwargs):
        with _branch_mode("record", tol=tol) as trace:
            value = fun(*args, **kwargs)
        paths = PathSet(trace)
        return value, paths
    return recorded


def replay(fun, path):
    def replayed(*args, **kwargs):
        with _branch_mode("replay", replay_path=path):
            value = fun(*args, **kwargs)
        return value
    return replayed


def grad(fun, argnums=0, tol=0.0, has_aux=False):
    def grad_fn(*args, **kwargs):
        _, paths = record(fun, tol=tol)(*args, **kwargs)
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


def value_and_grad(fun, argnums=0, tol=0.0, has_aux=False):
    def val_grad_fn(*args, **kwargs):
        record_result, paths = record(fun, tol=tol)(*args, **kwargs)
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
            return (record_value, grads, aux), paths
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


def replay_value_and_grad(fun, path, argnums=0, has_aux=False, _jax_vg_fn=None):
    jax_vg_fn = _jax_vg_fn if _jax_vg_fn is not None else jax.value_and_grad(fun, argnums=argnums, has_aux=has_aux)

    def replayed_val_grad(*args, **kwargs):
        with _branch_mode("replay", replay_path=path):
            vg_result = jax_vg_fn(*args, **kwargs)

        if has_aux:
            (value, aux), grads = vg_result
            return value, grads, aux
        else:
            value, grads = vg_result
            return value, grads
    return replayed_val_grad


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
            p = len(choices[0][1])
            negate = np.zeros((J, p), dtype=bool)
            ambiguous = np.zeros((J, p), dtype=bool)
            for k, (nearby_indices, base_negate) in enumerate(choices):
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
    # NB: this module defines its own `sum` (for HashTensor values, see below), which
    # shadows the builtin -- use a plain loop instead of sum(...) here.
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

    This is the batching mechanism for h_fun's H0-given branch: H0 is a Python list of
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

    def batched_val_grad(*args, **kwargs):
        if kwargs:
            raise TypeError("replay_value_and_grad_batch does not support kwargs")
        n_args = len(args)

        if J == 0:
            raise ValueError("replay_value_and_grad_batch requires at least one path")

        names = [node.name for node in paths[0]]
        for path in paths[1:]:
            if [node.name for node in path] != names:
                raise ValueError(
                    "replay_value_and_grad_batch requires all paths to share the same "
                    "op-name sequence (same traced control flow)."
                )

        flat_leaves, leaf_layout = _build_batch_leaves(paths, names)

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
            return values, grads, aux
        else:
            values, grads = vg_out
            return values, grads
    return batched_val_grad


def all_value_and_grad(fun, argnums=0, tol=0.0, has_aux=False):
    def all_vg_fn(*args, **kwargs):
        _defaultresult, paths = record(fun, tol=tol)(*args, **kwargs)

        if kwargs or not paths.trace:
            jax_vg_fn = jax.value_and_grad(fun, argnums=argnums, has_aux=has_aux)
            results = []
            for path in paths:
                vg_fn = replay_value_and_grad(fun, path, argnums=argnums, has_aux=has_aux, _jax_vg_fn=jax_vg_fn)
                results.append(vg_fn(*args, **kwargs))
            return results, paths

        n_args = len(args)
        n_nodes = len(paths.trace)
        positions = list(paths._iter_positions())
        n_paths = len(positions)
        selector_arrays = [jnp.asarray([pos[i] for pos in positions], dtype=jnp.int32) for i in range(n_nodes)]

        def vmap_body(*inner_args):
            call_args = inner_args[:n_args]
            selectors = inner_args[n_args:]
            with _branch_mode("vmap_replay", trace=paths.trace, selectors=selectors):
                return fun(*call_args)

        jax_vg_fn = jax.value_and_grad(vmap_body, argnums=argnums, has_aux=has_aux)
        vmap_vg_fn = jax.vmap(jax_vg_fn, in_axes=(None,) * n_args + (0,) * n_nodes)
        vg_out = vmap_vg_fn(*args, *selector_arrays)

        results = []
        if has_aux:
            (values, aux), grads = vg_out
            for k in range(n_paths):
                aux_k = jax.tree_util.tree_map(lambda a, k=k: a[k], aux)
                results.append((values[k], grads[k], aux_k))
        else:
            values, grads = vg_out
            for k in range(n_paths):
                results.append((values[k], grads[k]))

        return results, paths
    return all_vg_fn


def h_fun(fun, argnums=0, tol=0.0, has_aux=False):

    def wrapped(z, H0=None):
        z_jax = jnp.asarray(z)

        if H0 is None:
            defaultresult, paths = record(fun, tol=tol)(z_jax)

            if not paths.trace:
                # No traced max/min/abs/maximum/minimum ops at all -- a single default
                # path, nothing to batch (jax.vmap needs at least one non-None in_axes
                # entry, so this can't go through replay_value_and_grad_batch below).
                jax_vg_fn = jax.value_and_grad(fun, argnums=argnums, has_aux=has_aux)
                vg_result = jax_vg_fn(z_jax)
                if has_aux:
                    (v, _aux), g = vg_result
                else:
                    v, g = vg_result
                grads = np.asarray(g, dtype=float).reshape(z_jax.shape[0], 1)
                return defaultresult, grads, paths

            full_paths = list(paths)
            v, g = replay_value_and_grad_batch(fun, full_paths, argnums=argnums, has_aux=has_aux)(z_jax)
            grads = np.asarray(g, dtype=float).T

            return defaultresult, grads, paths
        else:
            J = len(H0)
            if J == 0:
                return np.zeros(0, dtype=float), np.zeros((z_jax.shape[0], 0), dtype=float)

            v, g = replay_value_and_grad_batch(fun, H0, argnums=argnums, has_aux=has_aux)(z_jax)
            h = np.asarray(v, dtype=float)
            grads = np.asarray(g, dtype=float).T

            return h, grads

    return wrapped
