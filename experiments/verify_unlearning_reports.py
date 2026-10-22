"""Small mathematical checks for the three literature reports, NOT grid experiments.

Only NumPy is needed. All arrays here are explicitly synthetic teaching examples.
The checks do not run the original papers' experimental implementations.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def run():
    rng = np.random.default_rng(42)
    records = []

    def check(name, condition, **details):
        assert bool(condition), (name, details)
        records.append(dict(name=name, passed=True, **details))

    # A deterministic sharded ridge learner with a slice-specific training state.
    # State = sufficient statistics, so rollback/replay is unambiguous here.
    X = np.column_stack([np.ones(24), rng.normal(size=(24, 3))])
    y = X @ np.array([1.0, 2.0, -0.5, 0.7]) + rng.normal(0, .5, 24)
    shards = np.arange(24).reshape(3, 8)
    lam = .3

    def fit(ids):
        return np.linalg.solve(X[ids].T @ X[ids] + lam * np.eye(4), X[ids].T @ y[ids])

    deleted = {3, 6}
    initial = np.array([fit(ids) for ids in shards])
    resumed = initial.copy()
    affected = 0
    checkpoint = (X[:2].T @ X[:2], X[:2].T @ y[:2])
    for s, ids in enumerate(shards):
        if not deleted.intersection(ids):
            continue
        affected += 1
        # Earliest affected slice has indices [2,3]. Resume before it.
        prefix = ids[:2]
        assert set(prefix).isdisjoint(deleted)
        suffix = np.array([i for i in ids[2:] if i not in deleted])
        a = checkpoint[0] + X[suffix].T @ X[suffix] + lam * np.eye(4)
        b = checkpoint[1] + X[suffix].T @ y[suffix]
        resumed[s] = np.linalg.solve(a, b)
    retrained = np.array([fit(np.array([i for i in ids if i not in deleted])) for ids in shards])
    error = float(np.max(np.abs(resumed - retrained)))
    check('sisa_rollback_equals_same_pipeline_retraining', error < 1e-12,
          max_parameter_error=error, affected_shards=affected, total_shards=3)
    check('sisa_unaffected_shards_bit_identical', np.array_equal(initial[1:], resumed[1:]))
    remain = np.array([i for i in range(24) if i not in deleted])
    aggregate = X @ resumed.mean(0)
    pooled = X @ fit(remain)
    gap = float(np.max(np.abs(aggregate - pooled)))
    check('sisa_is_not_pooled_model_retraining', gap > 1e-3, max_prediction_gap=gap)

    # Influence derivative: sum loss + lambda*n/2 regularization; upweight one
    # sample loss without changing the fixed regularization in this derivative.
    ridge = .2
    n = len(X)
    H = X.T @ X + ridge * n * np.eye(4)
    theta = np.linalg.solve(H, X.T @ y)
    i, eps = 3, 1e-5
    gi = X[i] * (X[i] @ theta - y[i])
    derivative = -np.linalg.solve(H, gi)
    shifted = np.linalg.solve(H + eps * np.outer(X[i], X[i]), X.T @ y + eps * X[i] * y[i])
    err = float(np.max(np.abs((shifted - theta) / eps - derivative)))
    check('influence_upweight_derivative_has_negative_sign', err < 1e-6,
          finite_difference_max_error=err)

    forget = np.array([3, 6, 15, 20])
    keep = np.setdiff1d(np.arange(n), forget)
    HR = X[keep].T @ X[keep] + ridge * len(keep) * np.eye(4)
    delta = X[forget].T @ (X[forget] @ theta - y[forget]) + len(forget) * ridge * theta
    newton = theta + np.linalg.solve(HR, delta)
    target = np.linalg.solve(HR, X[keep].T @ y[keep])
    err = float(np.max(np.abs(newton - target)))
    check('remaining_hessian_ridge_batch_deletion_is_exact', err < 1e-12,
          max_parameter_error=err, removed_count=len(forget))
    wrong_h = theta + np.linalg.solve(H, delta)
    wrong_err = float(np.max(np.abs(wrong_h - target)))
    check('full_hessian_one_step_is_not_exact_batch_deletion', wrong_err > 1e-4,
          max_parameter_error=wrong_err)
    wrong_reg = theta + np.linalg.solve(HR, delta - len(forget) * ridge * theta)
    wrong_reg_err = float(np.max(np.abs(wrong_reg - target)))
    check('omitting_changed_regularization_is_wrong', wrong_reg_err > 1e-4,
          max_parameter_error=wrong_reg_err)

    # Strongly convex logistic regression, bounded feature norms, one deletion.
    Z = rng.normal(size=(30, 3))
    Z /= np.maximum(1, np.linalg.norm(Z, axis=1, keepdims=True))
    labels = np.where(Z @ np.array([1.0, -1.5, .8]) + rng.normal(0, .5, 30) > 0, 1., -1.)
    reg = .15

    def logistic(w, ids):
        z, t = Z[ids], labels[ids]
        a = t * (z @ w)
        q = 1 / (1 + np.exp(-a))
        g = z.T @ ((q - 1) * t) + reg * len(ids) * w
        h = z.T @ ((q * (1 - q))[:, None] * z) + reg * len(ids) * np.eye(3)
        return g, h

    def optimize(ids):
        w = np.zeros(3)
        for _ in range(30):
            g, h = logistic(w, ids)
            if np.linalg.norm(g) < 1e-12:
                break
            w -= np.linalg.solve(h, g)
        return w

    all_ids = np.arange(30)
    base = optimize(all_ids)
    retained = all_ids[1:]
    g, h = logistic(base, retained)
    update = -np.linalg.solve(h, g)
    approximate = base + update
    true_model = optimize(retained)
    residual = float(np.linalg.norm(logistic(approximate, retained)[0]))
    bound = float(.25 * np.linalg.norm(Z[retained], 2) * np.linalg.norm(update)
                  * np.linalg.norm(Z[retained] @ update))
    check('logistic_newton_step_has_nonzero_residual', residual > 1e-9,
          residual_norm=residual, max_parameter_error=float(np.max(np.abs(approximate - true_model))))
    check('logistic_data_dependent_residual_bound', residual <= bound + 1e-10,
          residual_norm=residual, data_dependent_bound=bound)

    # SSD-style ratio rule (default exponent=1, clipping upper bound=1).
    weights = np.array([2., -3., .5, 4., 7.])
    original_imp = np.array([1., 4., 0., 2., 0.])
    forget_imp = np.array([10., 1., 5., 0., 0.])
    alpha, damping = 2., 1.
    selected = forget_imp > alpha * original_imp
    scale = np.ones_like(weights)
    # Define only at selected coordinates, avoiding irrelevant 0/0.
    scale[selected] = np.minimum(1., damping * original_imp[selected] / forget_imp[selected])
    dampened = weights * scale
    check('ssd_ratio_selection_and_dampening',
          np.array_equal(selected, [True, False, True, False, False])
          and np.allclose(dampened, [.2, -3., 0., 4., 7.]),
          selected=selected.tolist(), multipliers=scale.tolist(), resulting_weights=dampened.tolist())
    check('ssd_preserves_unselected_parameters', np.array_equal(weights[~selected], dampened[~selected]))
    layer_gate = np.array([False, False, True, True, True])
    gated = weights.copy()
    gated[layer_gate] *= scale[layer_gate]
    check('ssd_explicit_layer_gate_protects_frozen_core', np.array_equal(gated[:2], weights[:2]),
          resulting_weights=gated.tolist())
    gradients = np.array([1., -1.])
    check('squared_batch_gradient_differs_from_mean_squared_sample_gradients',
          gradients.mean() ** 2 == 0 and (gradients ** 2).mean() == 1,
          squared_batch_mean=float(gradients.mean() ** 2),
          mean_sample_square=float((gradients ** 2).mean()))
    importance = np.array([100., 1.])
    ewc = importance / (1 + importance)
    check('ewc_penalizes_important_parameter_motion', 1 - ewc[0] < 1 - ewc[1],
          ewc_optimum=ewc.tolist())

    return dict(scope='Synthetic mathematical examples only; no official benchmark, grid solve, or privacy certification',
                random_seed=42, checks_passed=len(records), checks_failed=0, checks=records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('reports/unlearning_three_methods/numerical_checks.json'))
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
