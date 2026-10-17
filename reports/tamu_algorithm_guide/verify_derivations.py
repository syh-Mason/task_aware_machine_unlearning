"""Verify teaching identities only; does not rerun TX-123BT experiments."""
import json
from pathlib import Path
import numpy as np


def run():
    checks = []

    def check(name, condition, **details):
        if not bool(condition):
            raise AssertionError((name, details))
        checks.append(dict(name=name, passed=True, **details))

    x = np.array([1., 2., 3.])
    y = np.array([1., 2., 0.])
    theta0 = float(x @ y / (x @ x))
    hr = float(np.mean(x[:2] ** 2))
    gr = float(np.mean(x[:2] * (x[:2] * theta0 - y[:2])))
    retrain = float(x[:2] @ y[:2] / (x[:2] @ x[:2]))
    newton = theta0 - gr / hr
    wrong_h = theta0 - gr / np.mean(x ** 2)
    local_if = theta0 + x[2] * (x[2] * theta0-y[2]) / (len(x) * np.mean(x**2))
    check('remaining_Newton_equals_refit', abs(newton-retrain) < 1e-13,
          original=theta0, newton=newton, refit=retrain)
    check('full_Hessian_does_not_give_exact_remaining_update',
          abs(wrong_h-275/392) < 1e-13 and abs(wrong_h-retrain) > .1,
          value=wrong_h)
    check('classical_local_deletion_approximation', abs(local_if-115/196) < 1e-13,
          value=local_if)

    design = np.array([[1., .2], [.3, 1.], [1.5, -.4], [.2, 1.8]])
    target = np.array([.7, -.2, 1.4, -.6])
    count = len(target)

    def fit(weights):
        return np.linalg.solve(design.T @ (weights[:, None]*design),
                               design.T @ (weights*target))

    base_weights = np.ones(count)
    base = fit(base_weights)
    hessian = design.T @ design/count
    gradients = design*(design @ base-target)[:, None]
    analytic = -np.linalg.solve(hessian, gradients.T)/count
    finite = []
    step = 1e-5
    for i in range(count):
        delta = np.eye(count)[i]*step
        finite.append((fit(base_weights+delta)-fit(base_weights-delta))/(2*step))
    finite = np.array(finite).T
    param_error = float(np.max(np.abs(analytic-finite)))
    check('sample_weight_parameter_derivative', param_error < 1e-9,
          max_abs_error=param_error)
    evaluation_target = np.array([.1, .2])
    v = base-evaluation_target
    score = v @ analytic

    def performance(weights):
        return .5*float(np.sum((fit(weights)-evaluation_target)**2))

    fd_score = []
    for i in range(count):
        delta = np.eye(count)[i]*step
        fd_score.append((performance(base_weights+delta)-performance(base_weights-delta))/(2*step))
    score_error = float(np.max(np.abs(score-np.array(fd_score))))
    check('sample_weight_outer_loss_derivative', score_error < 1e-9,
          max_abs_error=score_error)

    weights = np.array([.2, 1.8, .9, 1.3])
    weighted_hessian = design.T @ (weights[:, None]*design)/count
    weighted_gradient = design.T @ (weights*(design @ base-target))/count
    weighted_newton = base-np.linalg.solve(weighted_hessian, weighted_gradient)
    weighted_error = float(np.max(np.abs(weighted_newton-fit(weights))))
    check('weighted_quadratic_Newton_equals_weighted_refit', weighted_error < 1e-12,
          max_abs_error=weighted_error)

    s = np.array([2., -1., .5])
    # L1 ball of radius 1 around ones: all vertices remain within [0,2].
    vertices = np.concatenate([np.ones((1, 3)),
                               np.ones((3, 3))+np.eye(3),
                               np.ones((3, 3))-np.eye(3)])
    best = vertices[np.argmin(vertices @ s)]
    check('linear_weight_budget_example',
          np.array_equal(best, [0., 1., 1.]) and float(best @ s) == -.5,
          weights=best.tolist(), objective=float(best @ s), baseline=1.5)

    # Active-inequality QP: min .5 z'Qz + q'z, with z0 <= u.
    # For 0 < u < 2, its unique solution is z=(u,0), beta=4-2u > 0.
    Q = np.diag([2., 1.])
    G = np.array([[1., 0.]])
    K = np.block([[Q, G.T], [G, np.zeros((1, 1))]])
    rhs = np.array([0., 0., 1.])
    dyd_u = np.linalg.solve(K, rhs)

    def qp_solution(u):
        return np.linalg.solve(K, np.array([4., 0., u]))

    u = .8
    finite_kkt = (qp_solution(u+step)-qp_solution(u-step))/(2*step)
    kkt_error = float(np.max(np.abs(finite_kkt-dyd_u)))
    check('active_set_KKT_sensitivity', kkt_error < 1e-9,
          derivative=dyd_u.tolist(), max_abs_error=kkt_error)
    loss_y = np.array([1.3, -.2, 0.])
    # F_u = (0,0,-1); adjoint gradient is -F_u' K^{-T} C_y.
    F_u = -rhs
    adjoint = -float(F_u @ np.linalg.solve(K.T, loss_y))
    check('adjoint_equals_forward_sensitivity',
          abs(adjoint-float(loss_y @ dyd_u)) < 1e-13,
          adjoint=adjoint)

    costs = []
    for q in [.9, 1.1]:
        costs.append(q+100*max(1-q, 0)+10*max(q-1, 0))
    check('equal_MSE_unequal_teaching_cost',
          np.allclose(costs, [10.9, 2.1], atol=1e-12),
          forecasts=[.9, 1.1], squared_errors=[.01, .01], costs=costs)

    result = dict(scope='Teaching examples, not new power-grid experiments',
                  passed=len(checks), failed=0, checks=checks)
    output = Path(__file__).with_name('derivation_checks.json')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    run()
