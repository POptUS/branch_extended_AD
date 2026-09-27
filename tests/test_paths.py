import importlib

import jax.numpy as jnp
import pytest

import branch_extended_AD as bead
import branch_extended_AD.numpy as bnp
from branch_extended_AD.paths import (
    path_key,
    paths_all_in,
    paths_any_in,
    paths_equal,
    unique_paths,
)

ht = importlib.import_module("branch_extended_AD.HashTensor")


def make_path(*choices):
    return [ht._TraceNode(name, [choice]) for name, choice in choices]


def four_maximum_paths():
    x = jnp.array([1.0, 2.0])
    y = jnp.array([1.05, 1.95])
    _, paths = bead.record(lambda a, b: bnp.maximum(a, b), atol=0.1)(x, y)
    return paths


def test_factorized_path_set_iteration_indexing_and_membership():
    paths = four_maximum_paths()
    iterated = list(paths)

    assert len(paths) == 4
    assert bool(paths)
    assert all(path in paths for path in iterated)
    assert [path_key(paths[index]) for index in range(len(paths))] == [
        path_key(path) for path in iterated
    ]
    assert paths[-1] == paths[3]
    assert paths[-2] == paths[2]
    assert paths[1] == paths[1]

    with pytest.raises(IndexError):
        paths[4]
    with pytest.raises(IndexError):
        paths[-5]
    with pytest.raises(TypeError):
        paths["invalid"]
    assert "not a path" not in paths


def test_path_set_iterators_are_independent():
    paths = bead.PathSet.from_trace([
        ht._TraceNode("a", [0, 1]),
        ht._TraceNode("b", ["x", "y"]),
    ])
    first = iter(paths)
    assert path_key(next(first)) == (("a", 0), ("b", "x"))

    expected = [
        (("a", 0), ("b", "x")),
        (("a", 0), ("b", "y")),
        (("a", 1), ("b", "x")),
        (("a", 1), ("b", "y")),
    ]
    assert [path_key(path) for path in paths] == expected
    assert path_key(next(first)) == (("a", 0), ("b", "y"))


def test_path_set_algebra_preserves_correlations():
    diagonal = bead.PathSet.from_paths([
        make_path(("a", 0), ("b", 0)),
        make_path(("a", 1), ("b", 1)),
    ])
    first = bead.PathSet.from_paths([make_path(("a", 0), ("b", 0))])

    assert {path_key(path) for path in diagonal.union(first)} == {
        (("a", 0), ("b", 0)),
        (("a", 1), ("b", 1)),
    }
    assert {path_key(path) for path in diagonal.intersection(first)} == {
        (("a", 0), ("b", 0)),
    }
    assert {path_key(path) for path in diagonal.difference(first)} == {
        (("a", 1), ("b", 1)),
    }
    with pytest.raises(TypeError):
        diagonal.union([])
    with pytest.raises(TypeError):
        diagonal.intersection([])
    with pytest.raises(TypeError):
        diagonal.difference([])


def test_empty_and_single_path_sets():
    empty = bead.PathSet.empty()
    singleton = bead.PathSet.from_trace([])

    assert len(empty) == 0
    assert not empty
    assert list(empty) == []
    assert len(singleton) == 1
    assert bool(singleton)
    assert list(singleton) == [[]]
    assert singleton[0] == []
    assert [] in singleton
    assert repr(empty) == "PathSet(paths=0)"
    assert "possible paths" in str(singleton)


def test_explicit_path_set_string_representation():
    paths = bead.PathSet.from_paths([make_path(("max", 0))])
    assert repr(paths) == "PathSet(paths=1)"
    assert str(paths) == "PathSet with 1 explicit paths"


def test_path_helpers_support_paths():
    first = make_path(("a", 0), ("b", 1))
    duplicate = make_path(("a", 0), ("b", 1))
    other = make_path(("a", 1), ("b", 1))

    assert path_key(first) == (("a", 0), ("b", 1))
    assert paths_equal(first, duplicate)
    assert not paths_equal(first, other)
    assert unique_paths([first, duplicate, other]) == [first, other]
    assert paths_any_in([other], [first, other])
    assert paths_all_in([first, duplicate], [first])
    assert not paths_all_in([first, other], [first])

    with pytest.raises(TypeError):
        path_key("manual")
    with pytest.raises(TypeError):
        paths_equal("manual", "manual")


def test_format_path_describes_branch_choices():
    maximum_path = four_maximum_paths()[0]
    text = four_maximum_paths().format_path(maximum_path)
    assert "Step 1 (maximum)" in text
    assert "standard choice" in text

    max_path = make_path(("max", 2))
    formatter = bead.PathSet.from_trace([])
    assert "scalar choice = 2" in formatter.format_path(max_path)
    assert formatter.format_path([]) == "No decision points"
    with pytest.raises(TypeError):
        formatter.format_path("invalid")
