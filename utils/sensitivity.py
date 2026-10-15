"""Finite interventions and audited costs for frozen-model sensitivity studies.

The project Operator supplies the unchanged dispatch/redispatch problems.
"""
import numpy as np


def coordinate_perturbations(values, fraction=0.1):
    """One row per coordinate, with x[j] replaced by (1+fraction)*x[j].

    Negative coordinates become more negative; zero coordinates stay zero.
    Preserve the input floating dtype so a float32 CNN sees the recorded input.
    """
    values = np.asarray(values)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError('values must be a nonempty finite vector')
    if not np.isfinite(fraction) or fraction < 0:
        raise ValueError('fraction must be finite and nonnegative')
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float64)
    scenarios = np.tile(values, (values.size, 1))
    indices = np.arange(values.size)
    scenarios[indices, indices] *= 1 + fraction
    if not np.isfinite(scenarios).all():
        raise ValueError('perturbation produced non-finite values')
    return scenarios


def audited_cost(operator, forecast, target):
    """Evaluate one unchanged-target scenario with both original QPs.

    Return objective components, actual solve statuses/solvers, constraint
    residuals and the decisions needed to inspect the complete cost path.
    """
    forecast = np.asarray(forecast, dtype=float)
    target = np.asarray(target, dtype=float)
    if forecast.shape != (operator.no_load,) or target.shape != (operator.no_load,):
        raise ValueError('audited_cost expects one forecast and target vector')
    pg, ls1, angle1, c1, status1 = operator.solve_one(forecast)
    solver1 = operator.prob_1._tamu_solver
    violation1 = max(float(np.max(c.violation())) for c in operator.prob_1.constraints)
    ls2, storage, angle2, c2, status2 = operator.solve_two(target, pg)
    solver2 = operator.prob_2._tamu_solver
    violation2 = max(float(np.max(c.violation())) for c in operator.prob_2.constraints)
    components = dict(
        dispatch_generation_quadratic=float(pg[0] ** 2 @ operator.second_coeff),
        dispatch_generation_linear=float(pg[0] @ operator.first_coeff),
        dispatch_shedding=float(ls1[0].sum() * operator.load_shed_coeff),
        redispatch_shedding_linear=float(ls2[0].sum() * operator.load_shed_coeff),
        redispatch_shedding_quadratic=float((ls2[0] ** 2).sum() * operator.load_shed_coeff_second),
        redispatch_storage_linear=float(storage[0].sum() * operator.gen_storage_coeff),
        redispatch_storage_quadratic=float((storage[0] ** 2).sum() * operator.gen_storage_coeff_second),
    )
    total = float(c1[0] + c2[0])
    objective_error = abs(sum(components.values()) - total)
    if not np.isclose(sum(components.values()), total, rtol=1e-9, atol=1e-6):
        raise AssertionError(f'Original objective decomposition disagrees by {objective_error}')
    return dict(cost1=float(c1[0]), cost2=float(c2[0]), cost_total=total,
                status1=str(status1[0]), status2=str(status2[0]), solver1=solver1, solver2=solver2,
                violation1=violation1, violation2=violation2, objective_error=objective_error,
                ls1_total=float(ls1.sum()), ls2_total=float(ls2.sum()), storage_total=float(storage.sum()),
                negative_forecast_count=int(np.sum(forecast < 0)), components=components,
                decisions=dict(pg=pg[0].tolist(), ls1=ls1[0].tolist(), ls2=ls2[0].tolist(),
                               storage=storage[0].tolist(), angle1=angle1[0].tolist(), angle2=angle2[0].tolist()))
