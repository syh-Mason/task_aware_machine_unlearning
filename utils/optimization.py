"""
cvxpy fomulation for generator scheduling problem
"""

import numpy as np
from pypower.api import ext2int, makeBdc, bustypes
from pypower.idx_brch import RATE_A, RATE_B, RATE_C, BR_X
from pypower.idx_bus import PD
from pypower.idx_gen import PMAX, PMIN
import cvxpy as cp
import torch
import os
import warnings
from utils.dataset import case_modifier

try:
    from gurobipy import GurobiError
    SOLVER_ERRORS = (cp.error.SolverError, GurobiError)
except ImportError:
    SOLVER_ERRORS = (cp.error.SolverError,)


def solve_problem(problem, solver=None):
    """Solve the unchanged CVXPY problem and reject unusable solutions.

    GUROBI and ECOS/OSQP/SCS solve the same convex QP (or its equivalent
    conic canonicalization). No objective, coefficient or constraint changes.
    An explicit solver never silently falls back; auto mode does.
    """
    selected = (solver or os.environ.get('TAMU_SOLVER', 'auto')).upper()
    installed = cp.installed_solvers()
    candidates = ['GUROBI', 'ECOS', 'OSQP', 'SCS'] if selected == 'AUTO' else [selected]
    previous = getattr(problem, '_tamu_solver', None)
    if selected == 'AUTO' and previous in candidates:
        candidates.remove(previous)
        candidates.insert(0, previous)
    options = {
        'ECOS': dict(abstol=1e-8, reltol=1e-8, feastol=1e-8, max_iters=1000),
        'OSQP': dict(eps_abs=1e-8, eps_rel=1e-8, max_iter=100000, polish=True),
        'SCS': dict(eps=1e-6, max_iters=100000),
    }
    failures = []
    for candidate in candidates:
        if candidate not in installed:
            failures.append(f'{candidate}: not installed')
            continue
        try:
            problem.solve(solver=candidate, verbose=False, **options.get(candidate, {}))
            if problem.status != cp.OPTIMAL:
                raise cp.error.SolverError(f'status={problem.status}')
            if problem.value is None or not np.isfinite(problem.value):
                raise cp.error.SolverError('non-finite objective')
            for variable in problem.variables():
                if variable.value is None or not np.isfinite(variable.value).all():
                    raise cp.error.SolverError(f'missing/non-finite variable {variable.name()}')
            violation = max((float(np.max(c.violation())) for c in problem.constraints), default=0)
            if violation > 1e-5:
                raise cp.error.SolverError(f'constraint violation={violation:g}')
        except SOLVER_ERRORS as error:
            failures.append(f'{candidate}: {error}')
            continue
        problem._tamu_solver = candidate
        if failures and selected == 'AUTO':
            warnings.warn(f'Using {candidate} for the same optimization problem; ' + '; '.join(failures), RuntimeWarning)
        return problem.value
    raise cp.error.SolverError('No valid optimal solution: ' + '; '.join(failures))


def as_batch(value, width, name):
    """Validate a vector or a batch without flattening/truncating samples."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=float)
    if value.ndim == 1 and value.shape == (width,):
        value = value[None, :]
    if value.ndim != 2 or value.shape[1] != width or not len(value):
        raise ValueError(f'{name} must have shape ({width},) or (batch, {width}); got {value.shape}')
    if not np.isfinite(value).all():
        raise ValueError(f'{name} must contain only finite values')
    return value

class Operator:
    
    def __init__(self, case_config, theta_slack = 0, solver=None):
        self.solver = solver
        
        case = case_modifier(case_config=case_config)
        self.baseMVA = case["baseMVA"]
        self.no_bus = case['bus'].shape[0]
        self.no_gen = case['gen'].shape[0]  
        self.no_branch = case['branch'].shape[0]
        self.no_load = np.sum(case['bus'][:, PD] > 0)
        
        self.no_brh = case['branch'].shape[0]
        self.first_coeff = case["first_coeff"] # in p.u.
        self.second_coeff = case["second_coeff"] # in p.u.^2
        
        self.load_shed_coeff = case["load_shed_coeff"]
        self.gen_storage_coeff = case["gen_storage_coeff"]
        self.load_shed_coeff_second = case["load_shed_coeff_second"]
        self.gen_storage_coeff_second = case["gen_storage_coeff_second"]
        
        self.default_load = case['bus'][:, PD] / self.baseMVA
        self.pg_max = case["gen"][:, PMAX] / self.baseMVA
        self.pf_max = case["branch"][:, RATE_A] / self.baseMVA
        
        # branch susceptance
        self.b = 1/np.array(case['branch'][:, BR_X])
        
        # matrices
        case_int = ext2int(case)
        self.ref_index, _, _ = bustypes(case_int['bus'], case_int['gen'])
        
        # generator incidence matrix
        self.Cg = np.zeros((self.no_bus,self.no_gen))
        for i in range(self.no_gen):
            self.Cg[int(case_int['gen'][i,0]),i] = 1
        
        # load incidence matrix
        self.Cl = np.zeros((self.no_bus,self.no_load))
        load_bus_index = np.where(case_int['bus'][:, PD] > 0)[0]
        for i in range(self.no_load):
            self.Cl[load_bus_index[i],i] = 1
        
        self.Bbus, self.Bf, self.Pbusinj, self.Pfinj = makeBdc(case_int['baseMVA'], case_int['bus'], case_int['branch'])
        if np.all(case['bus'][:, 4:6] == 0) and np.all(case['branch'][:, 4] == 0) and np.all(case['branch'][:, 8:10] == 0):
            assert np.all(self.Pbusinj == 0)
            assert np.all(self.Pfinj == 0)
        
        self.Bbus = self.Bbus.todense()
        self.Bf = self.Bf.todense()
        # convert matrix to array
        self.Bbus = np.array(self.Bbus)
        self.Bf = np.array(self.Bf)
        
        # Branch Incidence Matrix
        f_bus = case_int['branch'][:, 0].astype('int')        # list of "from" buses
        t_bus = case_int['branch'][:, 1].astype('int')        # list of "to" buses
        self.Cf = np.zeros((self.no_brh,self.no_bus))         # "from" bus incidence matrix
        self.Ct = np.zeros((self.no_brh,self.no_bus))         # "to" bus incidence matrix
        for i in range(self.no_brh):
            self.Cf[i,f_bus[i]] = 1
            self.Ct[i,t_bus[i]] = 1
        
        self.A = self.Cf-self.Ct
        
        # check the relationship
        assert np.all(self.A.T@self.Bf == self.Bbus), "the relationship between Bbus and Bf is not correct"
        assert np.all(self.Bf == np.diag(self.b) @ self.A), "the relationship on the Bf"

        self.theta_slack = theta_slack

        # define problems
        
        self.prob_1 = self.stage_one()
        self.prob_2 = self.stage_two()
        assert self.prob_1.is_dpp()  # dpp helps for repeat solving the same problem with varying parameters
        assert self.prob_2.is_dpp()
        
    def stage_one(self):
        """
        construct the one-stage scheduling problem, the operator dispatches the generator based on the forecasted load
        param:
            d: the forecasted load (unscaled in p.u.)
            theta_slack: the default slack bus angle
        return:
            a cvxpy problem
        """
        
        # parameters: the forecasted load is a parameter
        # the branch susceptance is a parameter to cover the uncertainty
        # other parameters are fixed (constant)
        d = cp.Parameter(self.no_load, name = 'load')         # load forecast
        
        ls = cp.Variable(self.no_load, name='load_shedding')
        pg = cp.Variable(self.no_gen, name='generation')
        theta = cp.Variable(self.no_bus, name='voltage_angle')
        self.load_forecast, self.generation, self.dispatch_shedding, self.dispatch_angle = d, pg, ls, theta
        
        # constraints
        constraints = []
        
        constraints += [pg <= self.pg_max, pg >= 0] # generation limit
        
        constraints += [ self.Bf @ theta <= self.pf_max, self.Bf @ theta >= -self.pf_max] # branch flow limit
        
        constraints += [self.Bbus @ theta == self.Cg @ pg - self.Cl@(d - ls)] # power balance
        
        constraints += [ls >= 0]
        
        constraints += [theta[self.ref_index] == self.theta_slack] # slack bus angle            
        
        objective = cp.Minimize(
                    # second order + first order + load shedding
                    cp.quad_form(pg, cp.diag(self.second_coeff)) + cp.scalar_product(self.first_coeff, pg) + self.load_shed_coeff * cp.sum(ls)
                    )

        prob = cp.Problem(objective, constraints)
        
        return prob
    
    def stage_two(self):
        d = cp.Parameter(self.no_load, name = 'load')                 # ground-truth load
        pg = cp.Parameter(self.no_gen, name = 'generation')           # generation (passed from stage one)
        ls = cp.Variable(self.no_load, name='load_shedding')
        gs = cp.Variable(self.no_gen, name='generation_storage')
        theta = cp.Variable(self.no_bus, name='voltage_angle')
        self.load_true, self.scheduled_generation = d, pg
        self.redispatch_shedding, self.generation_storage, self.redispatch_angle = ls, gs, theta
        
        # constraints
        constraints = []
        
        constraints += [ self.Bf @ theta <= self.pf_max, self.Bf @ theta >= -self.pf_max] # branch flow limit
        
        constraints += [self.Bbus @ theta == self.Cg @ (pg-gs) - self.Cl@(d - ls)] # power balance
        
        constraints += [ls >= 0]
        
        constraints += [gs >= 0]
        
        constraints += [theta[self.ref_index] == self.theta_slack] # slack bus angle
        
        objective = cp.Minimize(
            # generation storage + load shedding both first order and second order
            self.load_shed_coeff * cp.sum(ls) + self.gen_storage_coeff * cp.sum(gs) + cp.sum(cp.square(gs)) * self.gen_storage_coeff_second + cp.sum(cp.square(ls)) * self.load_shed_coeff_second
        )
        
        prob = cp.Problem(objective, constraints)
        
        return prob

    def solve_one(self, load_forecast):
        """
        solve on a single or a batch of load forecast
        for dpp, the problem can be solved repeatedly with different parameters (load_forecast) 
            faster than solving it from scratch each time
        """
        # assert len(load_forecast.shape) == 1, "load_forecast should be a vector"
        load_forecast = as_batch(load_forecast, self.no_load, 'load_forecast')
        
        # assert np.all(load_forecast >= 0), "negative load detected"
        
        # clip the load forecast to have min = 0
        load_forecast = np.clip(load_forecast, 0, None)
        
        # allow batch solving
        p,s,theta,cost,status = [],[],[],[],[]
        for load_forecast_ in load_forecast:
            self.load_forecast.value = load_forecast_
            solve_problem(self.prob_1, self.solver)
            p.append(self.generation.value.copy())
            s.append(self.dispatch_shedding.value.copy())
            theta.append(self.dispatch_angle.value.copy())
            cost.append(self.prob_1.value)
            status.append(self.prob_1.status)
        
        return np.array(p), np.array(s), np.array(theta), np.array(cost), np.array(status)
    
    def solve_two(self, load_true, pg):
        load_true = as_batch(load_true, self.no_load, 'load_true')
        pg = as_batch(pg, self.no_gen, 'generation')
        if len(load_true) != len(pg):
            raise ValueError('load_true and generation batch sizes must match')
        if np.any(load_true < 0):
            raise ValueError('negative true load detected')
        # assert np.all(pg >= 0), "negative generation detected"
        
        pg = np.clip(pg, 0, None)
        
        # allow batch solving
        ls, gs, theta, cost, status = [],[],[],[],[]
        for load_true_, pg_ in zip(load_true, pg):
            self.load_true.value = load_true_
            self.scheduled_generation.value = pg_
            solve_problem(self.prob_2, self.solver)
            ls.append(self.redispatch_shedding.value.copy())
            gs.append(self.generation_storage.value.copy())
            theta.append(self.redispatch_angle.value.copy())
            cost.append(self.prob_2.value)
            status.append(self.prob_2.status)
        
        return np.array(ls), np.array(gs), np.array(theta), np.array(cost), np.array(status)
                    
    # def loss(self, pg, ls, gs):
        
    #     # pg (batch_size, no_gen)
    #     # loss = generator cost (from stage one) + load shedding cost (from stage two) + generation storage cost (from stage two)
        
    #     loss = (pg @ np.array(self.first_coeff)[:, None]).flatten() + self.load_shed_coeff * ls.sum(axis=1) + self.gen_storage_coeff * gs.sum(axis=1)
        
    #     return loss.flatten()
