import jax.numpy as jnp
import pytest

import branch_extended_AD as bead
import branch_extended_AD.numpy as bnp

def test_relative_tolerance_recording():
    x = jnp.array([100.0, 105.0, 1.0])

    def f(x):
        return bnp.max(x)

    _, paths = bead.record(f, rtol=0.1)(x)
    assert len(paths) == 2


def test_input_scaled_tolerance_mode():
    x = jnp.array([1.0, 1.2])

    def f(x):
        return bnp.max(10.0 * x)

    _, local_paths = bead.record(f, atol=1.0)(x)
    _, scaled_paths = bead.record(f, atol=1.0, tol_mode="input_scaled")(x)
    assert len(local_paths) == 1
    assert len(scaled_paths) == 2


def test_input_scaled_tolerance_uses_local_derivatives():
    x = jnp.array([2.0, 2.2])

    def f(x):
        y = x * x
        return bnp.max(y)

    _, local_paths = bead.record(f, atol=0.5)(x)
    _, scaled_paths = bead.record(f, atol=0.5, tol_mode="input_scaled")(x)
    assert len(local_paths) == 1
    assert len(scaled_paths) == 2


def test_input_scaled_tolerance_avoids_flat_region_overbranching():
    radius = 1.0

    def f(x):
        radial_excess = bnp.maximum(bnp.sum(x * x) - radius ** 2, 0.0)
        return bnp.sum(bnp.abs(radial_excess * x))

    center = jnp.array([0.0, 0.0, 0.0])
    edge = jnp.array([1.02, 0.0, 0.0])
    _, local_center_paths = bead.record(f, atol=0.1)(center)
    _, scaled_center_paths = bead.record(f, atol=0.1, tol_mode="input_scaled")(center)
    _, scaled_edge_paths = bead.record(f, atol=0.1, tol_mode="input_scaled")(edge)
    local_center_nearby = local_center_paths.choices(-1)[0].nearby_indices
    scaled_center_nearby = scaled_center_paths.choices(-1)[0].nearby_indices
    scaled_edge_nearby = scaled_edge_paths.choices(-1)[0].nearby_indices
    assert len(local_center_nearby) == center.size
    assert len(scaled_center_nearby) == 0
    assert len(scaled_edge_nearby) > len(scaled_center_nearby)


def test_invalid_tolerance_arguments():
    with pytest.raises(ValueError):
        bead.record(lambda x: x, tol_mode="invalid")


@pytest.mark.parametrize(
    "transform",
    [bead.record, bead.grad, bead.value_and_grad, bead.all_value_and_grad],
)
def test_recording_transforms_forward_relative_tolerance(transform):
    def function(x):
        return bnp.max(x)

    result = transform(function, rtol=0.1)(jnp.array([100.0, 105.0, 1.0]))
    paths = result[1]
    assert len(paths) == 2


@pytest.mark.parametrize(
    "transform",
    [bead.record, bead.grad, bead.value_and_grad, bead.all_value_and_grad],
)
def test_recording_transforms_forward_abs_policy(transform):
    def function(x):
        return bnp.sum(bnp.abs(x))

    result = transform(function, abs_policy="enumerate")(jnp.zeros(2))
    paths = result[1]
    assert len(paths) == 4


@pytest.mark.parametrize(
    "transform",
    [bead.record, bead.grad, bead.value_and_grad, bead.all_value_and_grad],
)
def test_recording_transforms_forward_input_scaled_tolerance(transform):
    def function(x):
        return bnp.max(10.0 * x)

    result = transform(function, atol=1.0, tol_mode="input_scaled")(jnp.array([1.0, 1.2]))
    paths = result[1]
    assert len(paths) == 2
