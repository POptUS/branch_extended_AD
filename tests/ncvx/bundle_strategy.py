import contextlib

import numpy as np
import torch


class BundleStrategy:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.penalty = None

    def objective_bundle(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy().reshape(-1)
        return torch.as_tensor(self.evaluator.bundle(np.asarray(x)), dtype=torch.double)

    def penalty_bundle(self, x, gradient):
        objective_bundle = self.objective_bundle(x).to(device=gradient.device, dtype=gradient.dtype)
        _, objective_gradient = self.evaluator.value_gradient(x.detach().cpu().numpy().reshape(-1))
        objective_gradient = torch.as_tensor(
            objective_gradient,
            device=gradient.device,
            dtype=gradient.dtype,
        ).reshape(-1, 1)
        mu = self.penalty.getPenaltyParameter()
        _, selected_penalty_gradient = self.penalty.getPenaltyFunctionValue()
        remainder = selected_penalty_gradient - mu * objective_gradient
        return mu * objective_bundle + remainder

    def support_gradient(self, x, gradient, direction):
        bundle = self.penalty_bundle(x, gradient)
        products = torch.conj(bundle.t()) @ direction
        return bundle[:, torch.argmax(products)].reshape(-1, 1)


@contextlib.contextmanager
def bundle_strategy(evaluator):
    from pygranso.private import bfgssqp as bfgssqp_module
    from pygranso.private import linesearchWeakWolfe as line_search_module
    from pygranso.private.qpSteeringStrategy import qpSS
    from pygranso.private.solveQP import solveQP
    from pygranso.pygransoStruct import pygransoStruct

    strategy = BundleStrategy(evaluator)
    original_bfgssqp = bfgssqp_module.AlgBFGSSQP.bfgssqp
    original_check = bfgssqp_module.AlgBFGSSQP.checkDirection
    original_steering = qpSS.qpSteeringStrategy
    original_line_search = line_search_module.linesearchWeakWolfe
    original_nearby = bfgssqp_module.getNearbyGradients

    def patched_nearby(penaltyfn_obj, grad_nbd_fn):
        gradients = strategy.objective_bundle(penaltyfn_obj.getX())
        _, ci_gradient, ce_gradient = penaltyfn_obj.getGradients()
        samples = np.empty(gradients.shape[1], dtype=object)
        for index in range(gradients.shape[1]):
            sample = pygransoStruct()
            sample.F = gradients[:, index].reshape(-1, 1)
            sample.CI = ci_gradient
            sample.CE = ce_gradient
            samples[index] = sample
        return samples, 0

    def patched_bfgssqp(self, penaltyfn_obj, bfgs_obj, opts, printer, torch_device):
        strategy.penalty = penaltyfn_obj
        original_has_constraints = penaltyfn_obj.hasConstraints
        penaltyfn_obj.hasConstraints = lambda: True
        try:
            return original_bfgssqp(self, penaltyfn_obj, bfgs_obj, opts, printer, torch_device)
        finally:
            penaltyfn_obj.hasConstraints = original_has_constraints

    def patched_check(self, direction, gradient):
        support = strategy.support_gradient(strategy.penalty.getX(), gradient, direction)
        return original_check(self, direction, support)

    def patched_line_search(x0, f0, grad0, direction, obj_fn, c1, c2, fvalquit,
                            eval_limit, step_tol, init_step_size, linesearch_maxit,
                            is_backtrack_linesearch, torch_device):
        support0 = strategy.support_gradient(x0, grad0, direction)

        def bundle_objective(x, get_grad=True):
            result = obj_fn(x, get_grad)
            if not get_grad:
                return result
            value, gradient, feasible = result
            return value, strategy.support_gradient(x, gradient, direction), feasible

        return original_line_search(
            x0,
            f0,
            support0,
            direction,
            bundle_objective,
            c1,
            c2,
            fvalquit,
            eval_limit,
            step_tol,
            init_step_size,
            linesearch_maxit,
            is_backtrack_linesearch,
            torch_device,
        )

    def patched_steering(self, penaltyfn_at_x, apply_hinv, l1_model, ineq_margin,
                         maxit, c_viol, c_mu, qp_solver, torch_device, double_precision):
        self.device = torch_device
        self.double_precision = double_precision
        self.torch_dtype = torch.double if double_precision else torch.float
        self.QPsolver = qp_solver
        self.ineq = penaltyfn_at_x.ci
        self.ineq_grad = penaltyfn_at_x.ci_grad
        self.eq = penaltyfn_at_x.ce
        self.eq_grad = penaltyfn_at_x.ce_grad
        self.n_ineq = len(self.ineq)
        n_eq = len(self.eq)
        self.violation = penaltyfn_at_x.tv_l1 if l1_model else penaltyfn_at_x.tv
        predicted = self.predictedViolationReductionL1 if l1_model else self.predictedViolationReduction
        gradients = strategy.objective_bundle(penaltyfn_at_x.x).to(
            device=torch_device,
            dtype=self.torch_dtype,
        )
        constraint_gradients = torch.hstack((self.eq_grad, self.ineq_grad))
        all_gradients = torch.hstack((gradients, constraint_gradients))
        hinv_gradients = apply_hinv(all_gradients)
        hessian = torch.conj(all_gradients.t()) @ hinv_gradients
        hessian = (hessian + torch.conj(hessian.t())) / 2
        linear = torch.vstack((
            torch.zeros((gradients.shape[1], 1), device=torch_device, dtype=self.torch_dtype),
            -self.eq,
            -self.ineq,
        ))
        lower = torch.vstack((
            torch.zeros((gradients.shape[1], 1), device=torch_device, dtype=self.torch_dtype),
            -torch.ones((n_eq, 1), device=torch_device, dtype=self.torch_dtype),
            torch.zeros((self.n_ineq, 1), device=torch_device, dtype=self.torch_dtype),
        ))
        upper = torch.ones((all_gradients.shape[1], 1), device=torch_device, dtype=self.torch_dtype)
        equality = torch.zeros((1, all_gradients.shape[1]), device=torch_device, dtype=self.torch_dtype)
        equality[0, :gradients.shape[1]] = 1
        violation_tolerance = np.sqrt(np.finfo(np.float64).eps) * max(self.violation, 1)

        def solve(mu):
            equality_rhs = torch.as_tensor([[mu]], device=torch_device, dtype=self.torch_dtype)
            upper[:gradients.shape[1]] = mu
            dual = solveQP(
                hessian,
                linear,
                equality,
                equality_rhs,
                lower,
                upper,
                qp_solver,
                torch_device,
                double_precision,
            )
            return -(hinv_gradients @ dual)

        mu = penaltyfn_at_x.mu
        direction = solve(mu)
        reduction = predicted(direction)
        if reduction >= c_viol * self.violation - violation_tolerance:
            return direction, mu, reduction
        if ineq_margin != np.inf and not torch.any(self.ineq >= -ineq_margin) and n_eq == 0:
            return direction, mu, reduction
        reference = solve(0.0)
        reference_reduction = predicted(reference)
        if reduction >= c_viol * reference_reduction - violation_tolerance:
            return direction, mu, reduction
        for _ in range(maxit):
            mu *= c_mu
            direction = solve(mu)
            reduction = predicted(direction)
            if reduction >= c_viol * reference_reduction - violation_tolerance:
                return direction, mu, reduction
        return direction, mu, reduction

    bfgssqp_module.AlgBFGSSQP.bfgssqp = patched_bfgssqp
    bfgssqp_module.AlgBFGSSQP.checkDirection = patched_check
    bfgssqp_module.getNearbyGradients = patched_nearby
    qpSS.qpSteeringStrategy = patched_steering
    line_search_module.linesearchWeakWolfe = patched_line_search
    try:
        yield
    finally:
        line_search_module.linesearchWeakWolfe = original_line_search
        qpSS.qpSteeringStrategy = original_steering
        bfgssqp_module.getNearbyGradients = original_nearby
        bfgssqp_module.AlgBFGSSQP.checkDirection = original_check
        bfgssqp_module.AlgBFGSSQP.bfgssqp = original_bfgssqp
