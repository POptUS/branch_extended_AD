import itertools
import logging
from dataclasses import dataclass

from .paths import path_key

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class _ReductionChoice:
    index: int


@dataclass(frozen=True)
class _ElementwiseChoice:
    nearby_indices: tuple
    choice_bits: int
    base_pick_two: tuple


@dataclass(frozen=True)
class _AbsChoice:
    nearby_indices: tuple
    base_negate: tuple
    choice_bits: int | None = None


class PathSet:
    """A collection of branch paths found while recording a function."""

    def __init__(self, *, trace=None, paths=None):
        if (trace is None) == (paths is None):
            raise ValueError("provide exactly one of trace or paths")
        self._trace = None if trace is None else tuple(trace)
        self._paths = None if paths is None else tuple(dict.fromkeys(path_key(path) for path in paths))

    @classmethod
    def from_trace(cls, trace):
        """Create a path set from recorded branch decisions."""
        return cls(trace=trace)

    @classmethod
    def from_paths(cls, paths):
        """Create a path set from resolved paths."""
        paths = tuple(paths)
        if not all(isinstance(path, list) for path in paths):
            raise TypeError("paths must contain resolved path lists")
        return cls(paths=paths)

    @classmethod
    def empty(cls):
        """Create a path set with no paths."""
        return cls(paths=())

    @property
    def has_decisions(self):
        """Return whether any path contains a branch decision."""
        if self._trace is not None:
            return bool(self._trace)
        return any(key for key in self._paths)

    def choices(self, decision_index):
        """Return the possible choices for one branch decision."""
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
        """Return paths that occur in either set."""
        if not isinstance(other, PathSet):
            raise TypeError("union requires another PathSet")
        return PathSet.from_paths([*self, *other])

    def intersection(self, other):
        """Return paths that occur in both sets."""
        if not isinstance(other, PathSet):
            raise TypeError("intersection requires another PathSet")
        other_keys = {path_key(path) for path in other}
        return PathSet.from_paths(path for path in self if path_key(path) in other_keys)

    def difference(self, other):
        """Return paths that occur in this set but not the other set."""
        if not isinstance(other, PathSet):
            raise TypeError("difference requires another PathSet")
        other_keys = {path_key(path) for path in other}
        return PathSet.from_paths(path for path in self if path_key(path) not in other_keys)

    def format_path(self, path):
        """Return a readable description of a resolved path."""
        if not isinstance(path, list):
            raise TypeError("path must be a list of resolved trace nodes")
        if not path:
            return "No decision points"
        lines = []
        for step, decision in enumerate(path, start=1):
            choice = decision.choices[0]
            prefix = f"  Step {step} ({decision.name}): "
            if isinstance(choice, _AbsChoice):
                if choice.choice_bits is None:
                    description = "standard absolute value" if not choice.nearby_indices else f"ambiguous indices {list(choice.nearby_indices)}"
                else:
                    flipped = [index for bit, index in enumerate(choice.nearby_indices) if (choice.choice_bits >> bit) & 1]
                    description = "standard choice" if not flipped else f"flip indices {flipped}"
            elif isinstance(choice, _ElementwiseChoice):
                flipped = [index for bit, index in enumerate(choice.nearby_indices) if (choice.choice_bits >> bit) & 1]
                description = "standard choice" if not flipped else f"flip indices {flipped}"
            elif isinstance(choice, _ReductionChoice):
                description = f"selected index = {choice.index}"
            else:
                description = f"scalar choice = {choice}"
            lines.append(prefix + description)
        return "\n".join(lines)
