"""Frozen CNN temperature responses, output-node cost diagnostics and a cohort.

No model fitting. Every counterfactual cost uses the unchanged two-stage Operator.
Output-node diagnostics are forecast-vector interventions, not physical causality.
"""
import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.input_sensitivity import LEVELS, NAMES, MATH_FILES, temperature_cases
from func_operation import return_nn_model
from scripts.analyze_cnn_sensitivity import digest, model_digest, repository_commit, write_csv, write_json
from utils import return_dataset
from utils.funcs import ModelNNAffine
from utils.optimization import Operator
from utils.sensitivity import audited_cost


def choose_cohort(total_load, per_cell=3):
    """Outcome-independent, deterministic month/true-load tercile sampling.

    Reference hour 0 is held apart. Within each month and load tercile, choose
    hours closest to 10/50/90% load quantiles, reserving distinct calendar dates.
    """
    total_load=np.asarray(total_load,dtype=float)
    if total_load.shape!=(1752,) or not np.isfinite(total_load).all() or per_cell!=3:
        raise ValueError('Expected 1752 finite test loads and three samples per cell')
    start=datetime(2019,1,1)
    dates=[start+timedelta(hours=i) for i in range(1752)]
    months=np.asarray([d.month for d in dates])
    days=np.arange(1752)//24
    reserved={0}
    result=[dict(test_index=0,date='2019-01-01',source_hour=1,month=1,load_group='reference',
                 role='reference',true_total_mw=float(total_load[0]),selection_quantile=None)]
    thresholds={}
    for month in (1,2,3):
        indices=np.where(months==month)[0]
        q1,q2=np.quantile(total_load[indices],[1/3,2/3])
        thresholds[month]=dict(q1_mw=float(q1),q2_mw=float(q2),available_hours=len(indices))
        strata=np.digitize(total_load[indices],[q1,q2],right=True)
        for group,name in enumerate(('low','middle','high')):
            pool=indices[strata==group]
            for quantile in (.1,.5,.9):
                wanted=np.quantile(total_load[pool],quantile)
                ordered=pool[np.lexsort((pool,np.abs(total_load[pool]-wanted)))]
                index=next((int(i) for i in ordered if int(days[i]) not in reserved),None)
                if index is None:
                    raise ValueError(f'Cannot select distinct dates in month {month}, {name}')
                reserved.add(int(days[index]))
                result.append(dict(test_index=index,date=dates[index].strftime('%Y-%m-%d'),
                    source_hour=index%24+1,month=month,load_group=name,role='stratified',
                    true_total_mw=float(total_load[index]),selection_quantile=quantile))
    return result,thresholds


def mixed_forecast(baseline,full,mask):
    baseline,full=np.asarray(baseline),np.asarray(full)
    if baseline.shape!=(14,) or full.shape!=(14,) or not 0<=mask<2**14:
        raise ValueError('Expected two 14-output forecasts and a valid subset bitmask')
    active=(int(mask)>>np.arange(14))&1
    return np.where(active,full,baseline)


def paired_permutations(seed,pairs,n=14):
    if not isinstance(pairs,int) or pairs<2:
        raise ValueError('At least two antithetic permutation pairs are required')
    rng=np.random.default_rng(seed)
    values=[]
    for _ in range(pairs):
        order=rng.permutation(n)
        values.extend([order,order[::-1]])
    return np.asarray(values)


def shapley_paths(value,permutations,n=14):
    """Average paired permutation marginals; MC SE uses independent pairs.

    value(mask) returns [cost1,cost2,total]. Each permutation telescopes exactly.
    Pair-based uncertainty quantifies computation, not sampling of test dates.
    """
    contributions=np.zeros((len(permutations),n,3))
    for r,order in enumerate(permutations):
        mask=0
        previous=np.asarray(value(mask))
        for node in order:
            mask|=1<<int(node)
            current=np.asarray(value(mask))
            contributions[r,node]=current-previous
            previous=current
    total=np.asarray(value(2**n-1))-np.asarray(value(0))
    np.testing.assert_allclose(contributions.sum(axis=1),np.tile(total,(len(permutations),1)),rtol=0,atol=1e-9)
    pair_means=contributions.reshape(-1,2,n,3).mean(axis=1)
    mean=pair_means.mean(axis=0)
    se=pair_means.std(axis=0,ddof=1)/np.sqrt(len(pair_means))
    return mean,se,contributions


def prepare(output,replay=None):
    parent=ROOT/'research/cnn_temperature_response'
    if replay:
        manifest=json.loads((replay/'manifest.json').read_text())
        for name,checksum in manifest['snapshot_sha256'].items():
            if digest(replay/name)!=checksum:
                raise ValueError(f'Bad replay checksum: {name}')
        for name in ('frozen_models.npz','cohort_inputs.npz','selection.json','parent_provenance.json'):
            (output/name).write_bytes((replay/name).read_bytes())
        data=np.load(output/'cohort_inputs.npz',allow_pickle=False)
        selection=json.loads((output/'selection.json').read_text())
        provenance=json.loads((output/'parent_provenance.json').read_text())
    else:
        previous=json.loads((parent/'manifest.json').read_text())
        for name,checksum in previous['snapshot_sha256'].items():
            if digest(parent/name)!=checksum:
                raise ValueError(f'Existing snapshot checksum mismatch: {name}')
        cfg=OmegaConf.create(previous['model_restore']['config'])
        for name in MATH_FILES:
            if digest(ROOT/name)!=previous['source_sha256'][name]:
                raise ValueError(f'Original mathematical source has changed: {name}')
        train,test=return_dataset(cfg)
        for name,checksum in previous['model_restore']['processed_data_sha256'].items():
            if digest(ROOT/'data/data_case14'/name)!=checksum:
                raise ValueError(f'Processed data changed: {name}')
        mean,std=test.target_mean.numpy(),test.target_std.numpy()
        truth=test.target.numpy()*std+mean
        selection_pool=truth.astype(float).sum(axis=1)*100
        selection,thresholds=choose_cohort(selection_pool)
        sigma=float(np.load(parent/'reference_inputs.npz',allow_pickle=False)['train_sigma_encoded'])
        np.testing.assert_equal(sigma,train.feature.numpy()[:,0,0,4].astype(float).std(ddof=0))
        ids=np.asarray([r['test_index'] for r in selection])
        np.savez_compressed(output/'cohort_inputs.npz',raw=test.feature[ids].numpy(),target_pu=truth[ids],
            mean=mean,std=std,train_sigma_encoded=np.asarray(sigma),test_indices=ids,
            selection_pool_true_total_mw=selection_pool)
        (output/'frozen_models.npz').write_bytes((parent/'frozen_models.npz').read_bytes())
        provenance=dict(parent_commit='b5fd9f13296382a787f28cacbcb498dd9fff07bc',
            parent_manifest_sha256=digest(parent/'manifest.json'),
            frozen_model_sha256=digest(parent/'frozen_models.npz'),config=previous['model_restore']['config'],
            counts=previous['model_restore']['counts'],temperature_statistics=previous['temperature_statistics'],
            processed_data_sha256=previous['model_restore']['processed_data_sha256'],
            full_test_restoration=previous['model_restore']['full_test_validation'],
            selection_thresholds=thresholds,
            selection_protocol='Reference index 0 plus 3 months x 3 within-month true-load terciles x 3 quantile targets (10/50/90%); all 28 dates distinct; never use model forecasts or costs',
            retraining_scope=previous['model_restore']['retraining_scope'])
        write_json(output/'selection.json',selection)
        write_json(output/'parent_provenance.json',provenance)
        data=np.load(output/'cohort_inputs.npz',allow_pickle=False)
    cfg=OmegaConf.create(provenance['config'])
    core=return_nn_model(cfg,is_load=False).eval()
    snapshot=np.load(output/'frozen_models.npz',allow_pickle=False)
    core.load_state_dict({key:torch.from_numpy(snapshot[f'core.{key}']) for key in core.state_dict()})
    heads={name:ModelNNAffine(np.column_stack([snapshot[f'{name}.weight'],snapshot[f'{name}.bias']]).reshape(-1)).eval()
           for name in NAMES}
    return cfg,core,heads,data,selection,provenance


def compute(output,values,pairs,solver='auto',smoke=False):
    cfg,core,heads,data,selection,provenance=values
    if smoke:
        selection=selection[:1]
    immutable={k:model_digest(v) for k,v in [('core',core),*heads.items()]}
    operator=Operator(cfg.case,solver=solver)
    baseline_rows,scenario_rows,bus_rows,node_rows,coalition_rows,shapley_rows=[],[],[],[],[],[]
    predictions=np.empty((len(selection),4,5,14),dtype=float)
    all_cases=[]
    solve_count=0
    max_violation=max_error=0.
    statuses=set();solvers=set()
    trace=output.joinpath('all_cost_traces.jsonl').open('w',encoding='utf-8')

    def cost(prediction,truth,metadata):
        nonlocal solve_count,max_violation,max_error
        actual=audited_cost(operator,prediction,truth)
        np.testing.assert_array_equal(operator.load_true.value,truth)
        solve_count+=1
        max_violation=max(max_violation,actual['violation1'],actual['violation2'])
        max_error=max(max_error,actual['objective_error'])
        statuses.update([actual['status1'],actual['status2']]);solvers.update([actual['solver1'],actual['solver2']])
        trace.write(json.dumps(dict(**metadata,forecast_pu=prediction.tolist(),target_pu=truth.tolist(),**actual),allow_nan=False)+'\n')
        return actual

    def cost_vector(actual):
        return np.array([actual['cost1'],actual['cost2'],actual['cost_total']])

    permutations=paired_permutations(42,pairs)
    shapley_marginals=[]
    for context,sample in enumerate(selection):
        test_index=sample['test_index']
        raw=data['raw'][context:context+1]
        truth=data['target_pu'][context].astype(float)
        untouched=truth.copy()
        cases=temperature_cases(raw[0],float(data['train_sigma_encoded']),0)
        all_cases.append(cases)
        with torch.no_grad():
            latent=torch.cat([core(torch.from_numpy(case[None]))[0] for case in cases]).double()
            baseline_feature=core(torch.from_numpy(raw))[0].double()
        np.testing.assert_array_equal(latent[2:3],baseline_feature)
        original_costs=None
        for model_index,(name,head) in enumerate(heads.items()):
            with torch.no_grad():
                prediction=np.concatenate([head(z[None]).numpy() for z in latent])*data['std']+data['mean']
                independent=head(baseline_feature).numpy()[0]*data['std']+data['mean']
            np.testing.assert_array_equal(prediction[2],independent)
            predictions[context,model_index]=prediction
            meta=dict(kind='independent_baseline',context=context,test_index=test_index,model=name,sigma_level=0.)
            baseline=cost(independent,truth,meta)
            baseline_rows.append(dict(context=context,test_index=test_index,model=name,
                cost1=baseline['cost1'],cost2=baseline['cost2'],cost_total=baseline['cost_total']))
            scenario_costs=[]
            for level_index,level in enumerate(LEVELS):
                actual=cost(prediction[level_index],truth,dict(kind='scenario',context=context,
                    test_index=test_index,model=name,sigma_level=float(level)))
                scenario_costs.append(actual)
                delta=(prediction[level_index]-prediction[2])*100
                scenario_rows.append(dict(context=context,**sample,model=name,sigma_level=float(level),
                    total_prediction_mw=float(prediction[level_index].sum()*100),
                    total_prediction_delta_mw=float(delta.sum()),mean_abs_prediction_delta_mw=float(np.abs(delta).mean()),
                    mse_pu2=float(np.mean((prediction[level_index]-truth)**2)),
                    mape_percent=float(np.mean(np.abs(prediction[level_index]-truth)/truth)*100),
                    cost1=actual['cost1'],cost2=actual['cost2'],cost_total=actual['cost_total'],
                    cost_delta=actual['cost_total']-baseline['cost_total'],
                    status1=actual['status1'],status2=actual['status2'],
                    violation1=actual['violation1'],violation2=actual['violation2'],
                    **actual['components']))
                for bus in range(14):
                    bus_rows.append(dict(context=context,test_index=test_index,model=name,sigma_level=float(level),
                        grid_bus=bus+1,prediction_mw=float(prediction[level_index,bus]*100),
                        actual_load_mw=float(truth[bus]*100),delta_mw=float(delta[bus])))
            if name=='original':
                original_costs=scenario_costs
        # Isolate and revert output-coordinate changes with actual QP reoptimization.
        # These are diagnostic forecast vectors, not alternative CNN inputs.
        f0=predictions[context,0,2]
        for level_index,level in enumerate(LEVELS):
            if level==0:
                continue  # exact zero has no changed outputs; its two QPs ran above.
            full=predictions[context,0,level_index]
            cache={0:cost_vector(original_costs[2]),(1<<14)-1:cost_vector(original_costs[level_index])}
            for mask,value in cache.items():
                coalition_rows.append(dict(context=context,test_index=test_index,sigma_level=float(level),
                    subset_mask=mask,cost1=float(value[0]),cost2=float(value[1]),cost_total=float(value[2]),source='main_scenario'))
            def value(mask):
                mask=int(mask)
                if mask not in cache:
                    mixed=mixed_forecast(f0,full,mask)
                    actual=cost(mixed,truth,dict(kind='output_subset',context=context,test_index=test_index,
                        model='original',sigma_level=float(level),subset_mask=mask))
                    cache[mask]=cost_vector(actual)
                    coalition_rows.append(dict(context=context,test_index=test_index,sigma_level=float(level),
                        subset_mask=mask,cost1=actual['cost1'],cost2=actual['cost2'],cost_total=actual['cost_total'],source='new_solve'))
                return cache[mask]
            for bus in range(14):
                isolated=value(1<<bus)-value(0)
                reverted=value((1<<14)-1)-value(((1<<14)-1)^(1<<bus))
                node_rows.append(dict(context=context,test_index=test_index,sigma_level=float(level),grid_bus=bus+1,
                    prediction_delta_mw=float((full[bus]-f0[bus])*100),
                    isolated_cost1_delta=float(isolated[0]),isolated_cost2_delta=float(isolated[1]),
                    isolated_total_cost_delta=float(isolated[2]),
                    reset_cost1_effect=float(reverted[0]),reset_cost2_effect=float(reverted[1]),
                    reset_total_cost_effect=float(reverted[2]),
                    joint_cost_delta=float((value((1<<14)-1)-value(0))[2])))
            if context==0:
                mean,se,marginals=shapley_paths(value,permutations)
                shapley_marginals.append(marginals)
                for bus in range(14):
                    shapley_rows.append(dict(test_index=test_index,sigma_level=float(level),grid_bus=bus+1,
                        prediction_delta_mw=float((full[bus]-f0[bus])*100),
                        shapley_cost1=float(mean[bus,0]),shapley_cost2=float(mean[bus,1]),
                        shapley_total=float(mean[bus,2]),mc_se_total=float(se[bus,2]),
                        independent_pairs=pairs,permutations=2*pairs,
                        joint_cost_delta=float((value((1<<14)-1)-value(0))[2])))
        np.testing.assert_array_equal(truth,untouched)
        print(f'Context {context+1}/{len(selection)}: test {test_index}, {sample["date"]}, '
              f'{sample["load_group"]}; four models, five levels and node diagnostics solved',flush=True)
    trace.close()
    if immutable!={k:model_digest(v) for k,v in [('core',core),*heads.items()]}:
        raise AssertionError('Frozen weights were mutated')
    for row in scenario_rows:
        if row['sigma_level']==0 and abs(row['cost_delta'])>1e-6:
            raise AssertionError('Zero temperature perturbation changed cost')
    write_csv(output/'baseline_costs.csv',baseline_rows)
    write_csv(output/'results.csv',scenario_rows)
    write_csv(output/'predictions_14_buses.csv',bus_rows)
    write_csv(output/'node_counterfactuals.csv',node_rows)
    write_csv(output/'coalition_costs.csv',coalition_rows)
    write_csv(output/'reference_shapley.csv',shapley_rows)
    np.savez_compressed(output/'interventions.npz',raw_cases=np.asarray(all_cases),forecasts_pu=predictions)
    np.savez_compressed(output/'shapley_permutations.npz',orders=permutations,marginals=np.asarray(shapley_marginals))
    checks=dict(contexts=len(selection),stratified_contexts=sum(r['role']=='stratified' for r in selection),
        scenario_rows=len(scenario_rows),forecast_values=int(predictions.size),
        node_counterfactual_rows=len(node_rows),coalition_rows=len(coalition_rows),
        cost_calls=solve_count,qp_solves=2*solve_count,max_constraint_violation=max_violation,
        max_objective_error=max_error,statuses=sorted(statuses),solvers=sorted(solvers),
        zero_forecast_exact=True,zero_cost_max_error=max(abs(r['cost_delta']) for r in scenario_rows if r['sigma_level']==0),
        unchanged_model_hashes=immutable,unchanged_target=True,
        shapley_pairs=pairs,shapley_permutations=2*pairs,shapley_seed=42)
    write_json(output/'checks.json',checks)
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'research/cnn_cost_attribution')
    parser.add_argument('--replay-from',type=Path)
    parser.add_argument('--pairs',type=int,default=64)
    parser.add_argument('--solver',choices=['auto','GUROBI','ECOS','OSQP','SCS'],default='auto')
    parser.add_argument('--smoke',action='store_true',help='One real sample, four antithetic pairs')
    args=parser.parse_args()
    if args.pairs<2:
        parser.error('pairs must be at least 2')
    output=args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory must be empty')
    output.mkdir(parents=True,exist_ok=True)
    os.chdir(ROOT);torch.set_num_threads(1)
    values=prepare(output,args.replay_from.resolve() if args.replay_from else None)
    checks=compute(output,values,4 if args.smoke else args.pairs,args.solver,args.smoke)
    write_json(output/'manifest.json',dict(source_commit=repository_commit(ROOT),
        mode='snapshot_replay' if args.replay_from else 'official_processed_data',
        source_sha256={name:digest(ROOT/name) for name in [*MATH_FILES,'experiments/cost_attribution.py',
            'experiments/input_sensitivity.py','scripts/analyze_cnn_sensitivity.py','utils/sensitivity.py']},
        snapshot_sha256={name:digest(output/name) for name in ['frozen_models.npz','cohort_inputs.npz','selection.json','parent_provenance.json']},
        selection_sha256=digest(output/'selection.json'),checks=checks,
        attribution_note=f'Output forecast coordinates are isolated/reset; Shapley uses {checks["shapley_permutations"]} permutations in antithetic pairs for the reference sample, not physical causality or population confidence intervals'))
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':
    main()
