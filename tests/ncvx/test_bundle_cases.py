from contextlib import ExitStack

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import branch_extended_AD as bead
import branch_extended_AD.numpy as bnp


torch = pytest.importorskip("torch")
pytest.importorskip("pygranso")
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct

from tests.ncvx.bundle_strategy import bundle_strategy


jax.config.update("jax_enable_x64", True)


def branch_abs(value):
    return bnp.maximum(-value, value)


class BeadEvaluator:
    def __init__(self, objective, x0, atol=1e-8):
        self.objective = objective
        self.current = bead.value_and_grad(objective, atol=atol)
        self.all_gradients = bead.all_value_and_grad(objective, atol=atol)

    def value_gradient(self, x):
        (value, gradient), _ = self.current(jnp.asarray(x))
        return float(value), np.asarray(gradient).copy()

    def bundle(self, x):
        results, _ = self.all_gradients(jnp.asarray(x))
        gradients = np.column_stack([np.asarray(gradient) for _, gradient in results])
        return np.unique(np.round(gradients.T, 14), axis=0).T


def solve(evaluator, x0, constraints, maxit=1000, opt_tol=1e-8):
    def combined(variables):
        x = variables.x.detach().cpu().numpy().reshape(-1)
        value, gradient = evaluator.value_gradient(x)
        ci, ci_gradient = constraints(x)
        return (
            value,
            torch.as_tensor(gradient, dtype=torch.double).reshape(-1, 1),
            None if ci is None else torch.as_tensor(ci, dtype=torch.double),
            None if ci_gradient is None else torch.as_tensor(ci_gradient, dtype=torch.double),
            None,
            None,
        )

    options = pygransoStruct()
    options.torch_device = torch.device("cpu")
    options.x0 = torch.as_tensor(x0, dtype=torch.double).reshape(-1, 1)
    options.globalAD = False
    options.print_level = 0
    options.quadprog_info_msg = False
    options.maxit = maxit
    options.opt_tol = opt_tol
    with ExitStack() as stack:
        stack.enter_context(bundle_strategy(evaluator))
        return pygranso({"x": [len(x0), 1]}, combined, options)


def no_constraints(x):
    return None, None


def rosenbrock_constraints(x):
    return (
        np.array([[np.sqrt(2) * x[0] - 1], [2 * x[1] - 1]]),
        np.array([[np.sqrt(2), 0], [0, 2]], dtype=float),
    )


def test_bundle_strategy_solves_constrained_nonsmooth_rosenbrock():
    def objective(x):
        return 8 * branch_abs(x[0] * x[0] - x[1]) + (1 - x[0]) * (1 - x[0])

    x0 = np.ones(2)
    evaluator = BeadEvaluator(objective, x0)
    assert evaluator.bundle(x0).shape == (2, 2)
    solution = solve(evaluator, x0, rosenbrock_constraints, maxit=100, opt_tol=1e-6)
    assert solution.termination_code == 0
    assert solution.final.feasible_to_tol
    assert solution.stat_value < 1e-6
    assert np.isclose(solution.final.f, (1 - 1 / np.sqrt(2)) ** 2, atol=1e-5)


def test_bundle_strategy_solves_fused_lasso_denoising():
    data = jnp.array([0.0, 2.0, 0.0])
    weight = 4.0 / 3.0

    def objective(x):
        residual = x - data
        differences = x[1:] - x[:-1]
        return bnp.sum(residual * residual) + weight * bnp.sum(branch_abs(differences))

    x0 = np.ones(3)
    evaluator = BeadEvaluator(objective, x0)
    assert evaluator.bundle(x0).shape == (3, 4)
    solution = solve(evaluator, x0, no_constraints)
    expected = np.full(3, 2.0 / 3.0)
    assert solution.termination_code == 0
    assert solution.stat_value < 1e-8
    assert np.allclose(solution.final.x.detach().cpu().numpy().reshape(-1), expected, atol=1e-8)
    assert np.isclose(solution.final.f, 8.0 / 3.0, atol=1e-10)
