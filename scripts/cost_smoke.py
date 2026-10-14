"""Compute actual solver costs for PYPOWER case14 benchmarks, not TX-123BT experiments."""
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
import torch

from func_operation import return_objective
from utils.net import SPO
from utils.optimization import Operator


def main():
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / 'conf'), version_base=None):
        cfg = compose(config_name='config')
    records = []
    for solver in ['GUROBI', 'ECOS', 'OSQP', 'SCS']:
        operator = Operator(cfg.case, solver=solver)
        load = operator.default_load * 0.7
        pg, _, _, cost1, status1 = operator.solve_one(load)
        _, _, _, cost2, status2 = operator.solve_two(load * 1.02, pg)
        records.append(dict(solver=solver, forecast_scale=0.7, true_scale=0.714,
                            cost1=float(cost1[0]), cost2=float(cost2[0]),
                            cost_total=float((cost1 + cost2)[0]),
                            status1=str(status1[0]), status2=str(status2[0])))
    operator = Operator(cfg.case, solver='ECOS')
    forecast = torch.tensor(operator.default_load[None, :] * 3, dtype=torch.double, requires_grad=True)
    target = forecast.detach().clone()
    pg, shedding, _, cost1, _ = operator.solve_one(forecast)
    _, _, _, cost2, _ = operator.solve_two(target, pg)
    objective = return_objective({'train': 'mse', 'test': 'cost'}, False, operator=operator)
    cost = objective.test_loss(SPO(torch.nn.Identity(), operator, 0, 1), None, (forecast, target))
    cost.backward()
    result = dict(data_source='PYPOWER case14 benchmark loads scaled by the original case14 configuration; not TX-123BT data',
                  solver_comparison=records,
                  dispatch_shedding_regression=dict(forecast_scale=3, true_scale=3,
                      evaluated_cost=float((cost1 + cost2)[0]),
                      differentiable_cost=float(cost.detach()),
                      dispatch_shedding_cost=float(shedding.sum() * operator.load_shed_coeff),
                      finite_nonzero_gradient=bool(torch.isfinite(forecast.grad).all() and forecast.grad.norm() > 0)))
    print(json.dumps(result, indent=2))
    output = root / 'outputs' / 'cost-debug'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'cost-benchmarks.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
