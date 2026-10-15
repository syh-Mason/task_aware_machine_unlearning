"""Five training-sigma temperature interventions with four frozen CNN heads.

Uses the existing core, original deletion rule, original TAMU LP and weighted
CG update. Dispatch objectives and constraints are supplied unchanged by Operator.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import cvxpy as cp
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_cnn_sensitivity import (
    rebuild_models, digest, model_digest, repository_commit, write_csv, write_json,
)
from func_operation import (return_core_datasets, return_dataset_for_nn_affine,
                            return_module, return_nn_model)
from utils import DatasetWithWeight, flatten_model, reconstruct_model, return_dataset
from utils.funcs import ModelNNAffine
from utils.optimization import Operator, solve_problem
from utils.sensitivity import audited_cost

LEVELS = np.array([-1., -.5, 0., .5, 1.])
NAMES = dict(original='Original CNN', unlearned20='Direct unlearning (20%)',
             retrained='Retrained head (20% removed)', tamu20='TAMU (20%, L1=0.1)')
MATH_FILES = ['clean_data.py', 'utils/dataset.py', 'utils/net.py', 'utils/optimization.py',
              'utils/funcs.py', 'func_operation.py', 'eval_unlearn.py', 'eval_unchange.py',
              'torch_influence/modules.py']


def temperature_cases(sample, train_sigma, bus_index0):
    """Add training sigma in encoded coordinates; preserve every other slot."""
    sample = np.asarray(sample)
    if sample.shape != (1, 14, 10) or not np.isfinite(sample).all() or not np.issubdtype(sample.dtype, np.floating):
        raise ValueError('Expected one finite (1,14,10) CNN input')
    if not 0 <= bus_index0 < 14 or not np.isfinite(train_sigma) or train_sigma <= 0:
        raise ValueError('Invalid bus or training standard deviation')
    cases = np.repeat(sample[None], len(LEVELS), axis=0)
    cases[:, 0, bus_index0, 4] = sample[0, bus_index0, 4] + LEVELS * train_sigma
    return cases


def read_temperature_year(tx_index0):
    """Read the actual numeric temperature cell from all 365 official XLSX files.

    Their suffix is .csv but their contents are XLSX. Read only the selected bus
    row; resolve sheet relationships instead of assuming worksheet filenames.
    """
    ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    relns = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
    row_number = tx_index0 + 2  # one header row, then zero-based TX rows
    values, hashes = [], {}
    for day in range(1, 366):
        path = ROOT / f'data/Data_public/Climate_2019/climate_2019_Day{day}.csv'
        hashes[str(path.relative_to(ROOT))] = digest(path)
        with ZipFile(path) as book:
            relations = ET.fromstring(book.read('xl/_rels/workbook.xml.rels'))
            targets = {r.attrib['Id']: r.attrib['Target'] for r in relations}
            sheets = ET.fromstring(book.read('xl/workbook.xml')).find(f'{ns}sheets')
            by_name = {s.attrib['name']: targets[s.attrib[f'{relns}id']] for s in sheets}
            if set(by_name) != {f'Hour {h}' for h in range(1, 25)}:
                raise ValueError(f'Incomplete hourly sheets: {path}')
            for hour in range(1, 25):
                target = by_name[f'Hour {hour}']
                target = target.lstrip('/') if target.startswith('/') else 'xl/' + target
                found = False
                with book.open(target) as stream:
                    for _, element in ET.iterparse(stream, events=['end']):
                        if element.tag != f'{ns}row' or int(element.attrib['r']) != row_number:
                            continue
                        cells = {c.attrib['r']: c for c in element}
                        bus = float(cells[f'A{row_number}'].find(f'{ns}v').text)
                        temperature = float(cells[f'B{row_number}'].find(f'{ns}v').text)
                        if bus != tx_index0 + 1 or not np.isfinite(temperature):
                            raise ValueError(f'Wrong bus or nonfinite temperature: {path}, Hour {hour}')
                        values.append(temperature)
                        found = True
                        break
                if not found:
                    raise ValueError(f'Missing temperature row: {path}, Hour {hour}')
        if day % 100 == 0:
            print(f'Official temperature audit: {day}/365 days', flush=True)
    return np.asarray(values), hashes


def full_restore(args, output):
    cfg, core, heads, raw, latent, target, mean, std, catalog, metadata = rebuild_models(
        SimpleNamespace(test_index=args.test_index, context_samples=1), output)
    snapshot = dict(np.load(output/'frozen_models.npz', allow_pickle=False))
    heads['retrained'] = ModelNNAffine(np.column_stack([
        snapshot['retrained.weight'], snapshot['retrained.bias']]).reshape(-1)).eval()
    train, test = return_dataset(cfg)
    _, sensitive = return_core_datasets(cfg, train)
    affine, affine_test = return_dataset_for_nn_affine(cfg, sensitive, test)
    remain_indices, delete_indices = snapshot['remain_indices'], snapshot['delete_indices']
    # DatasetWithWeight expects the original NewDataset attributes, not Subset.
    from utils import NewDataset
    remain = NewDataset(affine.feature[remain_indices], affine.target[remain_indices],
                        affine.target_mean, affine.target_std)
    remain.is_scale = affine.is_scale
    scores_path = ROOT/'outputs/real-conv/simulation_result/nn_conv/unchange_0.2_cost_scores.npy'
    archive_path = ROOT/'outputs/real-conv/simulation_result/nn_conv/unchange_0.2_cost.npy'
    scores = np.load(scores_path, allow_pickle=True).item()['remain']
    if scores.shape != (len(remain),) or not np.isfinite(scores).all():
        raise ValueError('Archived TAMU scores do not match the retained sample count')
    eps = cp.Variable(len(remain))
    problem = cp.Problem(cp.Minimize(scores @ eps), [
        cp.norm(eps-1, 1) <= .1*len(remain), cp.norm(eps-1, np.inf) <= 1.])
    solve_problem(problem)  # Same LP as eval_unchange; automatic compatible fallback.
    weights = eps.value.copy()
    weighted = DatasetWithWeight(remain, weights)  # Original float32 weight conversion.
    loader = DataLoader(weighted, batch_size=cfg.data.batch_size_eval, shuffle=False)
    module = return_module(cfg, {'train':'mse', 'test':'mse'}, {'train':loader, 'test':loader},
                           heads['original'], 'cg', with_weight=True)
    parameter = flatten_model(heads['original']).copy()
    update = module.stest(test_idxs=range(len(remain))).numpy()
    heads['tamu20'] = reconstruct_model(heads['original'], parameter-update).eval()
    archived = np.load(archive_path, allow_pickle=True).item()
    archive_index = list(archived['l1_constraints']).index(.1)
    full_checks = dict(metadata['full_test_validation'])
    with torch.no_grad():
        pred = heads['tamu20'](affine_test.feature).numpy()*std+mean
    actual = affine_test.target.numpy()*std+mean
    full_checks['tamu20'] = dict(mse=float(np.mean((pred-actual)**2)),
                                 mape=float(np.mean(np.abs(pred-actual)/actual)*100))
    for metric, value in full_checks['tamu20'].items():
        np.testing.assert_allclose(value, archived[f'{metric}_test'][archive_index], rtol=2e-7, atol=1e-10)
    # Full test cost checks restoration beyond MSE/MAPE. No new optimization model.
    operator = Operator(cfg.case, solver=args.solver)
    pg, _, _, c1, _ = operator.solve_one(pred)
    _, _, _, c2, _ = operator.solve_two(actual, pg)
    full_checks['tamu20']['cost'] = float(np.mean(c1+c2))
    np.testing.assert_allclose(full_checks['tamu20']['cost'], archived['cost_test'][archive_index],
                               rtol=2e-7, atol=1e-5)
    # Independent stationarity check against the exact weighted affine least squares.
    x = np.column_stack([remain.feature.numpy(), np.ones(len(remain))])
    y = remain.target.numpy()
    effective = weighted.weight.numpy().astype(np.float64)
    exact = np.linalg.solve(x.T @ (effective[:,None]*x), x.T @ (effective[:,None]*y)).T
    # Analytic rows contain [W_i, b_i]; PyTorch packs all W rows before all b.
    # Compare the named arrays directly, never the two incompatible flattenings.
    gap = float(max(np.max(np.abs(heads['tamu20'].fc.weight.detach().numpy()-exact[:,:-1])),
                    np.max(np.abs(heads['tamu20'].fc.bias.detach().numpy()-exact[:,-1]))))
    if gap > .002:
        raise AssertionError(f'Weighted CG disagrees with exact weighted affine fit: {gap}')
    snapshot.update({'tamu20.weight':heads['tamu20'].fc.weight.detach().numpy(),
                     'tamu20.bias':heads['tamu20'].fc.bias.detach().numpy(),
                     'tamu.lp_weights':weights, 'tamu.effective_weights':effective,
                     'tamu.remain_scores':scores,
                     'tamu.weighted_exact.weight':exact[:,:-1],
                     'tamu.weighted_exact.bias':exact[:,-1]})
    # Explicitly verify the earlier original/direct/retrain frozen weights.
    previous_path = ROOT/'research/cnn_sensitivity/frozen_models.npz'
    previous = np.load(previous_path, allow_pickle=False)
    for name in ('original', 'unlearned20', 'retrained'):
        for part in ('weight', 'bias'):
            np.testing.assert_array_equal(snapshot[f'{name}.{part}'], previous[f'{name}.{part}'])
    for name in core.state_dict():
        np.testing.assert_array_equal(snapshot[f'core.{name}'], previous[f'core.{name}'])
    metadata.update(full_test_validation=full_checks, tamu=dict(
        l1=.1, linf=1., lp_status=problem.status, lp_solver=problem._tamu_solver,
        lp_objective=float(problem.value), l1_actual=float(np.abs(weights-1).sum()),
        linf_actual=float(np.abs(weights-1).max()), weight_min=float(weights.min()),
        weight_max=float(weights.max()), cg_info=0, max_parameter_gap_vs_exact_weighted_fit=gap,
        scores_sha256=digest(scores_path), archived_results_sha256=digest(archive_path),
        archived_index=archive_index, archived_full_test_cost=float(archived['cost_test'][archive_index])),
        previous_snapshot_sha256=digest(previous_path), earlier_three_heads_bit_identical=True,
        retraining_scope='Refit 910 affine parameters on retained sensitive samples; shared CNN core is frozen')
    train_values = train.feature.numpy().astype(np.float64)
    sigma = train_values[:,0,args.grid_bus-1,4].std(ddof=0)
    mean_train = train_values[:,0,args.grid_bus-1,4].mean()
    files = sorted((ROOT/'data/data_case14').glob('bus_*.csv'), key=lambda p:int(p.stem.split('_')[1]))
    tx_index = int(files[args.grid_bus-1].stem.split('_')[1])
    physical, physical_hashes = read_temperature_year(tx_index)
    annual_mean, annual_std = float(physical.mean()), float(physical.std(ddof=1))
    encoded = pd.read_csv(files[args.grid_bus-1]).iloc[:,4].to_numpy()
    np.testing.assert_allclose((physical-annual_mean)/annual_std, encoded, rtol=1e-10, atol=1e-10)
    physical_train = physical[len(test):]
    np.testing.assert_allclose(sigma, physical_train.std(ddof=0)/annual_std, rtol=2e-8)
    stats = dict(grid_bus=args.grid_bus, tx_bus_index0=tx_index, tx_bus_label1=tx_index+1,
                 feature_index0=4, feature_index1=5, flattened_index0=(args.grid_bus-1)*10+4,
                 tensor_slot=f'input[0,0,{args.grid_bus-1},4]', train_rows=len(train),
                 train_annual_rows_start0=len(test), train_annual_rows_stop_exclusive=8760,
                 training_sigma_definition='population standard deviation, ddof=0, original 7008-row training pool',
                 train_mean_encoded=mean_train, train_sigma_encoded=float(sigma),
                 train_mean_kelvin=float(physical_train.mean()),
                 train_sigma_kelvin=float(physical_train.std(ddof=0)),
                 original_annual_mean_kelvin=annual_mean, original_annual_sample_std_kelvin=annual_std,
                 reference_temperature_kelvin=float(physical[args.test_index]),
                 reference_encoded_float32=float(raw[0,0,args.grid_bus-1,4]),
                 annual_normalization_verified_hours=8760, physical_source_sha256=physical_hashes)
    normalization = []
    for bus, path in enumerate(files):
        frame = pd.read_csv(path)
        for feature, column in enumerate(frame.columns[:-1]):
            normalization.append(dict(grid_bus=bus+1, source_file=str(path.relative_to(ROOT)),
                feature_index0=feature, feature=column, flattened_index0=bus*10+feature,
                train_mean_encoded=float(train_values[:,0,bus,feature].mean()),
                train_sigma_encoded_ddof0=float(train_values[:,0,bus,feature].std(ddof=0)),
                preprocessing='sin/cos' if feature<4 else 'full-year z-score, pandas ddof=1'))
    write_csv(output/'input_structure_140.csv', normalization)
    write_csv(output/'output_scaling_14.csv', [dict(grid_bus=i+1, mean_pu=float(mean[i]),
        std_pu=float(std[i]), baseMVA=100., formula='forecast_pu = output_z * std_pu + mean_pu') for i in range(14)])
    np.savez_compressed(output/'frozen_models.npz', **snapshot)
    np.savez_compressed(output/'reference_inputs.npz', raw=raw, target_pu=target,
                        mean=mean, std=std, train_sigma_encoded=np.asarray(sigma),
                        official_temperature_kelvin=physical, test_index=np.asarray(args.test_index))
    return cfg, core, heads, raw, target, mean, std, stats, metadata


def replay_restore(directory, output):
    manifest = json.loads((directory/'manifest.json').read_text())
    for filename, checksum in manifest['snapshot_sha256'].items():
        if digest(directory/filename) != checksum:
            raise ValueError(f'Checksum mismatch: {filename}')
    metadata, stats = manifest['model_restore'], manifest['temperature_statistics']
    cfg = OmegaConf.create(metadata['config'])
    core = return_nn_model(cfg, is_load=False).eval()
    snapshot = np.load(directory/'frozen_models.npz', allow_pickle=False)
    core.load_state_dict({name:torch.from_numpy(snapshot[f'core.{name}']) for name in core.state_dict()})
    heads = {name:ModelNNAffine(np.column_stack([snapshot[f'{name}.weight'], snapshot[f'{name}.bias']]).reshape(-1)).eval()
             for name in NAMES}
    data = np.load(directory/'reference_inputs.npz', allow_pickle=False)
    for filename in ('frozen_models.npz','reference_inputs.npz','input_structure_140.csv','output_scaling_14.csv'):
        (output/filename).write_bytes((directory/filename).read_bytes())
    return cfg, core, heads, data['raw'], data['target_pu'], data['mean'], data['std'], stats, metadata


def run_cases(cfg, core, heads, raw, target, mean, std, stats, output, solver, only_original=False):
    before = {name:model_digest(model) for name,model in [('core',core),*heads.items()]}
    cases = temperature_cases(raw[0], stats['train_sigma_encoded'], stats['grid_bus']-1)
    # One-sample forward calls for baseline and all interventions avoid batch-dependent roundoff.
    with torch.no_grad():
        latent = torch.cat([core(torch.from_numpy(case[None]))[0] for case in cases]).double()
        base_latent = core(torch.from_numpy(raw))[0].double()
    np.testing.assert_array_equal(cases[2], raw[0])
    np.testing.assert_array_equal(latent[2:3].numpy(), base_latent.numpy())
    operator = Operator(cfg.case, solver=solver)
    truth = target[0].astype(np.float64).copy()
    rows, buses, traces, forecasts = [], [], {}, []
    for name, head in heads.items():
        if only_original and name != 'original':
            continue
        with torch.no_grad():
            predictions = np.concatenate([head(feature[None]).numpy() for feature in latent])*std+mean
            baseline = head(base_latent).numpy()[0]*std+mean
        np.testing.assert_array_equal(predictions[2], baseline)
        baseline_cost = audited_cost(operator, baseline, truth)
        forecasts.append(predictions)
        model_trace = []
        for i,level in enumerate(LEVELS):
            prediction = predictions[i]
            cost = audited_cost(operator, prediction, truth)
            delta = (prediction-baseline)*operator.baseMVA
            mse = float(np.mean((prediction-truth)**2))  # Original evaluate: p.u.^2
            row = dict(model=name, sigma_level=float(level),
                       temperature_encoded=float(cases[i,0,stats['grid_bus']-1,4]),
                       requested_temperature_delta_kelvin=float(level*stats['train_sigma_kelvin']),
                       actual_encoded_step=float(cases[i,0,stats['grid_bus']-1,4]-raw[0,0,stats['grid_bus']-1,4]),
                       forecast_total_mw=float(prediction.sum()*operator.baseMVA),
                       forecast_total_delta_mw=float(delta.sum()), prediction_mae_delta_mw=float(np.abs(delta).mean()),
                       prediction_max_abs_delta_mw=float(np.abs(delta).max()),
                       mse_pu2=mse, mse_mw2=mse*operator.baseMVA**2,
                       mape_percent=float(np.mean(np.abs(prediction-truth)/truth)*100),
                       cost1=cost['cost1'], cost2=cost['cost2'], cost_total=cost['cost_total'],
                       cost1_delta=cost['cost1']-baseline_cost['cost1'],
                       cost2_delta=cost['cost2']-baseline_cost['cost2'],
                       cost_delta=cost['cost_total']-baseline_cost['cost_total'],
                       status1=cost['status1'],status2=cost['status2'],
                       solver1=cost['solver1'],solver2=cost['solver2'],
                       violation1=cost['violation1'],violation2=cost['violation2'],
                       objective_error=cost['objective_error'], **cost['components'])
            rows.append(row)
            for bus in range(14):
                buses.append(dict(model=name, sigma_level=float(level), grid_bus=bus+1,
                    actual_load_mw=float(truth[bus]*operator.baseMVA),
                    prediction_pu=float(prediction[bus]), prediction_mw=float(prediction[bus]*operator.baseMVA),
                    baseline_prediction_mw=float(baseline[bus]*operator.baseMVA), delta_mw=float(delta[bus])))
            model_trace.append(dict(sigma_level=float(level), forecast_pu=prediction.tolist(),
                                    target_pu=truth.tolist(), **cost))
        traces[name] = model_trace
        print(f'{name}: all five temperature cases optimal; zero forecast/cost deltas '
              f'{rows[-3]["prediction_max_abs_delta_mw"]:.3g}/{rows[-3]["cost_delta"]:.3g}', flush=True)
    after = {name:model_digest(model) for name,model in [('core',core),*heads.items()]}
    if before != after:
        raise AssertionError('Frozen model changed during interventions')
    np.testing.assert_array_equal(target[0], truth)
    for row in rows:
        if row['sigma_level']==0 and (row['prediction_max_abs_delta_mw']!=0 or abs(row['cost_delta'])>1e-6):
            raise AssertionError('Zero control failed to restore prediction or cost')
    write_csv(output/'results.csv', rows)
    write_csv(output/'predictions_14_buses.csv', buses)
    write_json(output/'cost_traces.json', traces)
    np.savez_compressed(output/'interventions.npz', raw_cases=cases, forecasts_pu=np.asarray(forecasts),
                        sigma_levels=LEVELS, target_pu=truth)
    checks = dict(models=list(traces), scenarios=len(rows), qp_solves=2*(len(rows)+len(traces)),
        unchanged_models=before, unchanged_target=True, zero_forecast_bit_identical=True,
        max_zero_cost_delta=max(abs(r['cost_delta']) for r in rows if r['sigma_level']==0),
        max_constraint_violation=max(max(r['violation1'],r['violation2']) for r in rows),
        max_objective_error=max(r['objective_error'] for r in rows),
        statuses=sorted({r['status1'] for r in rows}|{r['status2'] for r in rows}))
    write_json(output/'checks.json', checks)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'research/cnn_temperature_response')
    parser.add_argument('--test-index',type=int,default=0)
    parser.add_argument('--grid-bus',type=int,default=1)
    parser.add_argument('--solver',choices=['auto','GUROBI','ECOS','OSQP','SCS'],default='auto')
    parser.add_argument('--replay-from',type=Path)
    parser.add_argument('--only-original',action='store_true',help='Minimum five-case test before four-model study')
    args = parser.parse_args()
    if not 0 <= args.test_index < 1752 or not 1 <= args.grid_bus <= 14:
        parser.error('test-index must be 0..1751 and grid-bus 1..14')
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f'Output directory must be empty: {output}')
    output.mkdir(parents=True,exist_ok=True)
    os.chdir(ROOT)
    torch.set_num_threads(1)
    values = replay_restore(args.replay_from.resolve(),output) if args.replay_from else full_restore(args,output)
    cfg,core,heads,raw,target,mean,std,stats,metadata = values
    checks = run_cases(cfg,core,heads,raw,target,mean,std,stats,output,args.solver,args.only_original)
    write_json(output/'manifest.json',dict(source_commit=repository_commit(ROOT),
        mode='snapshot_replay' if args.replay_from else 'full_data_restoration',
        source_sha256={name:digest(ROOT/name) for name in [*MATH_FILES,'experiments/input_sensitivity.py','utils/sensitivity.py']},
        snapshot_sha256={name:digest(output/name) for name in ['frozen_models.npz','reference_inputs.npz']},
        temperature_statistics=stats, model_restore=metadata, study_checks=checks,
        perturbation='Single selected temperature coordinate += level * original-training-pool sigma (ddof=0)',
        actual_target_fixed=True, levels=LEVELS.tolist()))
    print(json.dumps(checks,indent=2),flush=True)
    print(f'Saved actual temperature responses: {output}',flush=True)


if __name__=='__main__':
    main()
