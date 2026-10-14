"""Solver regression tests using published PYPOWER case14 loads, not TX-123BT data."""
from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from pathlib import Path

from func_operation import return_objective
from utils.funcs import evaluate, load_influences
from utils.net import SPO
from utils.optimization import Operator, solve_problem


@pytest.fixture
def cfg():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / 'conf'), version_base=None):
        return compose(config_name='config')


def test_two_sample_cost_equals_individual_costs(cfg):
    operator = Operator(cfg.case)
    loads = torch.tensor(np.stack([operator.default_load * 0.5, operator.default_load * 0.9]), dtype=torch.float32)
    dataset = SimpleNamespace(feature=loads, target=loads, is_scale=False)
    batched = evaluate(torch.nn.Identity(), {'test': dataset}, 'cost', case_config=cfg.case, is_average=False)['test']
    individual = []
    for i in range(2):
        one = SimpleNamespace(feature=loads[i:i+1], target=loads[i:i+1], is_scale=False)
        individual.append(evaluate(torch.nn.Identity(), {'test': one}, 'cost', case_config=cfg.case)['test'])
    assert batched.shape == (2,)
    np.testing.assert_allclose(batched, individual, rtol=1e-5, atol=1e-5)


def test_redispatch_rejects_unequal_batches(cfg):
    operator = Operator(cfg.case)
    loads = np.stack([operator.default_load, operator.default_load])
    with pytest.raises(ValueError, match='batch'):
        operator.solve_two(loads, np.ones((1, operator.no_gen)))


def test_differentiable_cost_includes_dispatch_shedding(cfg):
    operator = Operator(cfg.case)
    forecast = torch.tensor(operator.default_load[None, :] * 3, dtype=torch.double, requires_grad=True)
    target = forecast.detach().clone()
    model = SPO(torch.nn.Identity(), operator, 0, 1)
    objective = return_objective({'train': 'mse', 'test': 'cost'}, False, operator=operator)
    differentiable = objective.test_loss(model, None, (forecast, target))
    pg, shedding, _, cost1, _ = operator.solve_one(forecast.detach())
    _, _, _, cost2, _ = operator.solve_two(target, pg)
    assert shedding.sum() > 1
    np.testing.assert_allclose(differentiable.detach(), (cost1 + cost2).mean(), rtol=1e-4, atol=0.1)
    differentiable.backward()
    assert forecast.grad is not None and torch.isfinite(forecast.grad).all()


@pytest.mark.parametrize('solver', ['GUROBI', 'ECOS', 'OSQP', 'SCS'])
def test_same_objective_and_physical_constraints(cfg, solver):
    if solver == 'GUROBI':
        gp = pytest.importorskip('gurobipy')
        try:
            with gp.Env(empty=True) as environment:
                environment.setParam('OutputFlag', 0)
                environment.start()
        except gp.GurobiError as error:
            if error.errno == 10009:
                pytest.skip(f'Optional Gurobi license unavailable: {error}')
            raise
    operator = Operator(cfg.case, solver=solver)
    load = operator.default_load * 0.7
    pg, ls1, angle1, cost1, status1 = operator.solve_one(load)
    true_load = load * 1.02
    ls2, storage, angle2, cost2, status2 = operator.solve_two(true_load, pg)
    assert (status1 == cp.OPTIMAL).all() and (status2 == cp.OPTIMAL).all()
    np.testing.assert_allclose(operator.Bbus @ angle1[0], operator.Cg @ pg[0] - operator.Cl @ (load - ls1[0]), atol=1e-5)
    np.testing.assert_allclose(operator.Bbus @ angle2[0], operator.Cg @ (pg[0] - storage[0]) - operator.Cl @ (true_load - ls2[0]), atol=1e-5)
    assert (pg >= -1e-5).all() and (pg <= operator.pg_max + 1e-5).all()
    assert (ls1 >= -1e-5).all() and (ls2 >= -1e-5).all() and (storage >= -1e-5).all()
    for angle in (angle1[0], angle2[0]):
        assert (np.abs(operator.Bf @ angle) <= operator.pf_max + 1e-5).all()
        np.testing.assert_allclose(angle[operator.ref_index], operator.theta_slack, atol=1e-5)
    expected1 = (pg ** 2 @ operator.second_coeff + pg @ operator.first_coeff
                 + ls1.sum(axis=1) * operator.load_shed_coeff)
    expected2 = (ls2.sum(axis=1) * operator.load_shed_coeff
                 + (ls2 ** 2).sum(axis=1) * operator.load_shed_coeff_second
                 + storage.sum(axis=1) * operator.gen_storage_coeff
                 + (storage ** 2).sum(axis=1) * operator.gen_storage_coeff_second)
    np.testing.assert_allclose(cost1, expected1, rtol=1e-8, atol=1e-6)
    np.testing.assert_allclose(cost2, expected2, rtol=1e-8, atol=1e-6)
    reference = Operator(cfg.case, solver='ECOS')
    ref_pg, _, _, ref_cost1, _ = reference.solve_one(load)
    _, _, _, ref_cost2, _ = reference.solve_two(true_load, ref_pg)
    np.testing.assert_allclose(cost1 + cost2, ref_cost1 + ref_cost2, rtol=2e-5, atol=1e-4)


def test_missing_gurobi_uses_compatible_solver(cfg, monkeypatch):
    installed = [name for name in cp.installed_solvers() if name != 'GUROBI']
    monkeypatch.setattr(cp, 'installed_solvers', lambda: installed)
    monkeypatch.delenv('TAMU_SOLVER', raising=False)
    operator = Operator(cfg.case)
    with pytest.warns(RuntimeWarning, match='Using ECOS'):
        _, _, _, cost, status = operator.solve_one(operator.default_load)
    assert status[0] == cp.OPTIMAL and np.isfinite(cost).all()
    assert operator.prob_1._tamu_solver == 'ECOS'


def test_unusable_gurobi_license_falls_back_on_same_problem(monkeypatch):
    monkeypatch.delenv('TAMU_SOLVER', raising=False)
    monkeypatch.setattr(cp, 'installed_solvers', lambda: ['GUROBI', 'ECOS'])
    problem = cp.Problem(cp.Minimize(cp.square(cp.Variable() - 2)))
    original = problem.solve
    calls = []
    def solve(**kwargs):
        calls.append(kwargs['solver'])
        if kwargs['solver'] == 'GUROBI':
            raise cp.error.SolverError('License unavailable')
        return original(**kwargs)
    monkeypatch.setattr(problem, 'solve', solve)
    with pytest.warns(RuntimeWarning, match='License unavailable'):
        solve_problem(problem)
    assert calls == ['GUROBI', 'ECOS']
    assert problem.value == pytest.approx(0, abs=1e-8)


def test_failed_solve_does_not_return_stale_values(monkeypatch):
    x = cp.Variable()
    limit = cp.Parameter(value=1)
    problem = cp.Problem(cp.Minimize(x), [x >= 0, x <= limit])
    solve_problem(problem, solver='ECOS')
    limit.value = -1
    with pytest.raises(cp.error.SolverError, match='infeasible'):
        solve_problem(problem, solver='ECOS')


@pytest.mark.parametrize('load', [np.full(14, np.nan), np.ones((2, 7)), np.ones((1, 1, 14)), np.empty((0, 14))])
def test_invalid_load_is_rejected(cfg, load):
    with pytest.raises(ValueError):
        Operator(cfg.case).solve_one(load)


def test_negative_forecast_clipping_matches_differentiable_path(cfg):
    operator = Operator(cfg.case)
    forecast = operator.default_load[None, :] * 0.7
    forecast[0, 0] = -0.1
    target = operator.default_load[None, :] * 0.7
    tensor = torch.tensor(forecast, dtype=torch.double, requires_grad=True)
    true_tensor = torch.tensor(target, dtype=torch.double)
    objective = return_objective({'train': 'mse', 'test': 'cost'}, False, operator=operator)
    cost = objective.test_loss(SPO(torch.nn.Identity(), operator, 0, 1), None, (tensor, true_tensor))
    pg, _, _, c1, _ = operator.solve_one(forecast)
    _, _, _, c2, _ = operator.solve_two(target, pg)
    np.testing.assert_allclose(cost.detach(), (c1 + c2).mean(), rtol=2e-4, atol=0.1)


def test_missing_influence_scores_have_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match='gen_index.py'):
        load_influences(tmp_path / 'missing.npy', 3)


@pytest.mark.parametrize('scores', [np.ones(2), np.full(3, np.nan), np.ones((3, 1))])
def test_incompatible_influence_scores_are_rejected(tmp_path, scores):
    path = tmp_path / 'scores.npy'
    np.save(path, scores)
    with pytest.raises(ValueError, match='matching data settings'):
        load_influences(path, 3)


def test_valid_influence_scores_preserve_values(tmp_path):
    path = tmp_path / 'scores.npy'
    np.save(path, np.array([1.0, -2.0, 3.0]))
    torch.testing.assert_close(load_influences(path, 3), torch.tensor([1.0, -2.0, 3.0]))


def test_real_runner_reports_missing_data_without_fabrication(tmp_path):
    from run_real import audit_data
    audit = audit_data(tmp_path)
    assert audit['raw_present_files'] == 0 and audit['processed_files'] == 0
    assert not audit['raw_complete'] and not audit['processed_complete']
    assert list(tmp_path.iterdir()) == []


def test_climate_rows_match_zero_based_load_columns():
    import pandas as pd
    from clean_data import climate_row
    # Two rows are a unit fixture, not a replacement climate dataset.
    rows = pd.DataFrame({'Bus': [1, 2]})
    assert climate_row(rows, 0)['Bus'].tolist() == [1]
    assert climate_row(rows, 1)['Bus'].tolist() == [2]
    with pytest.raises(IndexError):
        climate_row(rows, 2)


def test_reweighting_problem_beyond_bundled_gurobi_size_limit(monkeypatch):
    monkeypatch.delenv('TAMU_SOLVER', raising=False)
    # Deterministic unit scores exercise the original LP, not an experiment.
    size = 2100
    weights = cp.Variable(size)
    scores = np.linspace(-1, 1, size)
    problem = cp.Problem(cp.Minimize(cp.scalar_product(weights, scores)),
                         [cp.norm(weights - 1, 1) <= 0.1 * size,
                          cp.norm(weights - 1, np.inf) <= 1])
    solve_problem(problem)
    assert problem.status == cp.OPTIMAL
    assert np.isfinite(weights.value).all()
    assert np.linalg.norm(weights.value - 1, 1) <= 0.1 * size + 1e-5
    assert np.linalg.norm(weights.value - 1, np.inf) <= 1 + 1e-5
