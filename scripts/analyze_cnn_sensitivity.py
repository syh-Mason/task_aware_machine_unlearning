"""Audit the existing CNN and compare frozen-model +10% interventions.

No CNN optimization or core retraining is performed. Original and 20%-unlearned
heads follow eval_unlearn.py exactly. All costs solve the original two QPs with
the same observed target. A portable snapshot supports replay without raw data.
"""
import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from func_operation import (return_core_datasets, return_dataset_for_nn_affine,
                            return_module, return_nn_model, return_trained_model,
                            return_unlearn_datasets)
from utils import flatten_model, load_influences, reconstruct_model, return_dataset
from utils.funcs import ModelNNAffine
from utils.optimization import Operator
from utils.sensitivity import audited_cost, coordinate_perturbations

FEATURE_NAMES_ZH = ['星期正弦', '星期余弦', '小时正弦', '小时余弦', '温度', '短波辐射',
                    '长波辐射', '纬向风速', '经向风速', '风速']
FEATURE_UNITS = ['无量纲'] * 4 + ['K', 'W/m²', 'W/m²', 'm/s', 'm/s', 'm/s']
HEADS = ('original', 'unlearned20')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repository_commit(directory):
    """GitHub ZIPs have no .git; source hashes still identify their exact code."""
    result = subprocess.run(['git', '-C', str(directory), 'rev-parse', '--verify', 'HEAD^{commit}'],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def model_digest(model):
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def selected_indices(size, count, reference):
    if not 0 <= reference < size or not 1 <= count <= size:
        raise ValueError('Invalid reference index or context sample count')
    pool = np.linspace(0, size-1, count, dtype=int).tolist()
    indices = [reference] + [i for i in pool if i != reference]
    return np.asarray(indices[:count])


def capture_forward(core, sample):
    records, hooks = [], []
    def hook(name):
        def record(module, inputs, output):
            records.append(dict(layer=name, module=type(module).__name__,
                                input_shape=list(inputs[0].shape), output_shape=list(output.shape)))
        return record
    for name in ('conv1', 'conv2', 'relu', 'flatten', 'lin1', 'tanh', 'lin2'):
        hooks.append(getattr(core, name).register_forward_hook(hook(name)))
    with torch.no_grad():
        core(sample)
    for handle in hooks:
        handle.remove()
    return records


def feature_catalog(raw_sample):
    files = sorted((ROOT / 'data/data_case14').glob('bus_*.csv'), key=lambda p: int(p.stem.split('_')[1]))
    if len(files) != 14:
        raise ValueError('Expected all 14 real preprocessed bus files')
    rows = []
    for bus, path in enumerate(files):
        frame = pd.read_csv(path)
        for feature, column in enumerate(frame.columns[:-1]):
            rows.append(dict(input_index0=bus*10+feature, input_index1=bus*10+feature+1,
                             grid_bus=bus+1, feature_index1=feature+1, feature=column,
                             name_zh=FEATURE_NAMES_ZH[feature], original_physical_unit=FEATURE_UNITS[feature],
                             stored_unit='无量纲；周期编码' if feature < 4 else '无量纲；按节点全年z-score',
                             encoding='sin/cos；单坐标干预不保持单位圆' if feature < 4 else '(物理值-全年均值)/全年样本标准差(ddof=1)',
                             original_tx_column0=int(path.stem.split('_')[1]), original_tx_bus_label1=int(path.stem.split('_')[1])+1,
                             source_file=f'data/data_case14/{path.name}',
                             reference_value=float(raw_sample[0, bus, feature]),
                             annual_min=float(frame[column].min()), annual_max=float(frame[column].max())))
    return rows, {p.name: digest(p) for p in files}


def rebuild_models(args, output):
    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base=None):
        cfg = compose(config_name='config', overrides=['model=conv', 'unlearn_prop=0.2'])
    influence = ROOT / 'outputs/real-conv/influence/nn_conv_cost.npy'
    cfg.model.influence_dir = str(influence)
    train, test = return_dataset(cfg)
    _, sensitive = return_core_datasets(cfg, train)
    affine_train, affine_test = return_dataset_for_nn_affine(cfg, sensitive, test)
    scores = load_influences(influence, len(affine_train))
    deleted, remain, delete_indices, remain_indices = return_unlearn_datasets(
        scores, 0.2, affine_train, cfg.unlearn_mode, cfg)
    original = return_trained_model(cfg, 'nn_conv-affine', affine_train, False)
    original_hash = model_digest(original)
    original_parameters = flatten_model(original).copy()
    loader = DataLoader(remain, batch_size=cfg.data.batch_size_eval, shuffle=False)
    module = return_module(cfg, {'train': 'mse', 'test': 'mse'}, {'train': loader, 'test': loader},
                           original, method='cg', watch_progress=False)
    ihvp = module.stest(test_idxs=range(len(remain))).numpy()
    unlearned = reconstruct_model(original, original_parameters - ihvp)
    if model_digest(original) != original_hash:
        raise AssertionError('Influence calculation mutated the original model')
    retrained = return_trained_model(cfg, 'nn_conv-affine', remain, False)
    core = return_nn_model(cfg, is_load=True, dataset='core').eval()
    heads = dict(original=original, unlearned20=unlearned)
    full_validation = {}
    archived = np.load(ROOT / 'outputs/real-conv/simulation_result/nn_conv/unlearning_0.2.npy', allow_pickle=True).item()
    mean, std = test.target_mean.numpy(), test.target_std.numpy()
    true_pu = affine_test.target.numpy() * std + mean
    for name, head in [*heads.items(), ('retrained', retrained)]:
        with torch.no_grad():
            pred = head(affine_test.feature).numpy() * std + mean
        values = dict(mse=float(np.mean((pred-true_pu)**2)), mape=float(np.mean(np.abs(pred-true_pu)/true_pu)*100))
        tag = {'original': 'ori', 'unlearned20': 'unlearn', 'retrained': 'retrain'}[name]
        for metric, value in values.items():
            np.testing.assert_allclose(value, archived[f'{metric}_{tag}']['test'], rtol=1e-8, atol=1e-11)
        full_validation[name] = values
    indices = selected_indices(len(test), args.context_samples, args.test_index)
    raw = test.feature[indices].numpy().copy()
    target = test.target[indices].numpy() * std + mean
    with torch.no_grad():
        # Use batch size one for every local intervention, including its baseline,
        # so float32 batch-kernel roundoff cannot masquerade as a small effect.
        latent = np.concatenate([core(torch.from_numpy(x[None]))[0].numpy() for x in raw]).astype(np.float64)
        np.testing.assert_allclose(latent, affine_test.feature[indices].numpy(), rtol=2e-5, atol=2e-6)
    catalog, data_hashes = feature_catalog(raw[0])
    snapshot = {f'core.{key}': value.detach().numpy() for key, value in core.state_dict().items()}
    for name, head in [*heads.items(), ('retrained', retrained)]:
        snapshot[f'{name}.weight'] = head.fc.weight.detach().numpy()
        snapshot[f'{name}.bias'] = head.fc.bias.detach().numpy()
    snapshot['delete_indices'] = np.asarray(delete_indices)
    snapshot['remain_indices'] = np.asarray(remain_indices)
    np.savez_compressed(output / 'frozen_models.npz', **snapshot)
    np.savez_compressed(output / 'reference_inputs.npz', raw=raw, latent=latent, target_pu=target,
                        mean=mean, std=std, test_indices=indices)
    metadata = dict(config=OmegaConf.to_container(cfg, resolve=True),
                    counts=dict(train=len(train), test=len(test), core=len(train)-len(sensitive), sensitive=len(sensitive),
                                deleted=len(deleted), remain=len(remain)),
                    influence_sha256=digest(influence), checkpoint_sha256=digest(ROOT / 'trained_model/nn_conv/core.pth'),
                    processed_data_sha256=data_hashes, full_test_validation=full_validation,
                    cg_info=0, max_parameter_gap_vs_retraining=float(np.max(np.abs(flatten_model(unlearned)-flatten_model(retrained)))),
                    reference_test_index=int(indices[0]), context_test_indices=indices.tolist(),
                    dataset_index_note='Full data; test indices are annual rows 0..1751; Day=row//24+1, Hour=row%24+1. No timezone inferred.')
    return cfg, core, heads, raw, latent, target, mean, std, catalog, metadata


def replay_models(directory, output):
    manifest = json.loads((directory / 'manifest.json').read_text())
    for filename in ('frozen_models.npz', 'reference_inputs.npz'):
        if digest(directory / filename) != manifest['snapshot_sha256'][filename]:
            raise ValueError(f'Snapshot checksum mismatch: {filename}')
        if (directory / filename).resolve() != (output / filename).resolve():
            (output / filename).write_bytes((directory / filename).read_bytes())
    metadata = manifest['model_validation']
    cfg = OmegaConf.create(metadata['config'])
    from func_operation import return_nn_model
    core = return_nn_model(cfg, is_load=False).eval()
    snapshot = np.load(directory / 'frozen_models.npz', allow_pickle=False)
    core.load_state_dict({key: torch.from_numpy(snapshot[f'core.{key}']) for key in core.state_dict()})
    heads = {}
    for name in HEADS:
        weight, bias = snapshot[f'{name}.weight'], snapshot[f'{name}.bias']
        heads[name] = ModelNNAffine(np.column_stack([weight, bias]).reshape(-1)).eval()
    data = np.load(directory / 'reference_inputs.npz', allow_pickle=False)
    catalog = pd.read_csv(directory / 'raw_input_features_140.csv').to_dict('records')
    return cfg, core, heads, data['raw'], data['latent'], data['target_pu'], data['mean'], data['std'], catalog, metadata


def compute_study(cfg, core, heads, raw, latent, target, mean, std, catalog, metadata, output, fraction, solver):
    immutable = {name: model_digest(model) for name, model in [('core', core), *heads.items()]}
    operator = Operator(cfg.case, solver=solver)
    base_mva = float(operator.baseMVA)
    all_rows, traces = [], {}
    all_predictions = np.empty((2, len(raw), 205, 14), dtype=np.float64)
    case_space = ['baseline'] + ['latent64'] * 64 + ['raw140'] * 140
    case_index = [-1] + list(range(64)) + list(range(140))
    analytic_errors = []
    for context in range(len(raw)):
        latent_cases = coordinate_perturbations(latent[context], fraction)
        raw_cases = coordinate_perturbations(raw[context].reshape(-1), fraction).reshape(140, 1, 14, 10)
        with torch.no_grad():
            extracted = torch.cat([core(torch.from_numpy(x[None]))[0] for x in raw_cases]).double()
        features = torch.cat([torch.from_numpy(latent[context:context+1]), torch.from_numpy(latent_cases), extracted])
        forecasts = {}
        for model_index, (name, head) in enumerate(heads.items()):
            with torch.no_grad():
                forecasts[name] = head(features).numpy() * std + mean
            all_predictions[model_index, context] = forecasts[name]
            actual_latent_change = forecasts[name][1:65] - forecasts[name][0]
            exact_latent_change = (latent_cases-latent[context]) @ head.fc.weight.detach().numpy().T * std
            np.testing.assert_allclose(actual_latent_change, exact_latent_change, rtol=1e-9, atol=1e-12)
            analytic_errors.append(float(np.max(np.abs(actual_latent_change-exact_latent_change))))
            original_latent = latent[context]
            raw_steps = raw_cases.reshape(140, -1)[np.arange(140), np.arange(140)] - raw[context].reshape(-1)
            baseline = None
            for case in range(205):
                forecast = forecasts[name][case]
                cost = audited_cost(operator, forecast, target[context])
                if case == 0:
                    baseline = cost
                space, index = case_space[case], case_index[case]
                change = forecast - forecasts[name][0]
                value = (float(original_latent[index]) if space == 'latent64'
                         else float(raw[context].reshape(-1)[index]) if space == 'raw140' else 0.)
                step = (float(latent_cases[index, index]-value) if space == 'latent64'
                        else float(raw_steps[index]) if space == 'raw140' else 0.)
                new_value = value+step
                outside = (abs(new_value)>1 if space=='latent64'
                           else abs(new_value)>1 if space=='raw140' and index%10<4 else False)
                row = dict(model=name, context=context, test_index=metadata['context_test_indices'][context],
                           annual_day=metadata['context_test_indices'][context]//24+1,
                           hour_label=metadata['context_test_indices'][context]%24+1,
                           space=space, input_index1=index+1, input_before=value, input_after=new_value, actual_step=step,
                           grid_bus=catalog[index]['grid_bus'] if space=='raw140' else '',
                           feature=catalog[index]['feature'] if space=='raw140' else f'z{index+1:02d}' if space=='latent64' else 'none',
                           encoded_value_outside_unit_bound=bool(outside),
                           prediction_mae_pu=float(np.mean(np.abs(change))),
                           prediction_mae_mw=float(np.mean(np.abs(change))*base_mva),
                           prediction_max_abs_mw=float(np.max(np.abs(change))*base_mva),
                           prediction_total_change_mw=float(change.sum()*base_mva),
                           cost_baseline=baseline['cost_total'], cost_delta=cost['cost_total']-baseline['cost_total'],
                           cost_delta_pct=(cost['cost_total']/baseline['cost_total']-1)*100,
                           **{k: cost[k] for k in ('cost1','cost2','cost_total','status1','status2','solver1','solver2',
                                                   'violation1','violation2','objective_error','ls1_total','ls2_total','storage_total','negative_forecast_count')},
                           **cost['components'])
                all_rows.append(row)
                if context == 0 and case in (0, 1, 65):
                    traces[f'{name}/{space}/dimension{index+1}'] = dict(
                        test_index=metadata['context_test_indices'][context], input_before=value,
                        actual_step=step, target_pu=target[context].tolist(),
                        prediction_baseline_pu=forecasts[name][0].tolist(), prediction_pu=forecast.tolist(),
                        clipped_forecast_pu=np.clip(forecast, 0, None).tolist(), **cost)
        # The feature change is identical across heads because the extractor is frozen.
        delta_z = features.numpy() - features.numpy()[0]
        expected_difference = delta_z @ (heads['unlearned20'].fc.weight-heads['original'].fc.weight).detach().numpy().T * std
        actual_difference = (forecasts['unlearned20']-forecasts['unlearned20'][0])-(forecasts['original']-forecasts['original'][0])
        np.testing.assert_allclose(actual_difference, expected_difference, rtol=1e-8, atol=1e-12)
        print(f'Context {context+1}/{len(raw)}: test index {metadata["context_test_indices"][context]}, both heads and all 204 interventions solved', flush=True)
    for name, model in [('core', core), *heads.items()]:
        if model_digest(model) != immutable[name]:
            raise AssertionError(f'Frozen model mutated during study: {name}')
    write_csv(output / 'all_interventions.csv', all_rows)
    write_csv(output / 'fixed_sample_interventions.csv', (r for r in all_rows if r['context']==0))
    write_json(output / 'fixed_sample_cost_trace.json', traces)
    np.savez_compressed(output / 'all_predictions.npz', forecasts_pu=all_predictions,
                        case_space=np.asarray(case_space), input_index0=np.asarray(case_index))
    with torch.no_grad():
        core_prediction = core(torch.from_numpy(raw[:1]))[1].numpy()[0]*std+mean
    write_json(output / 'fixed_sample.json', dict(test_index=metadata['reference_test_index'],
               raw_input=raw[0].tolist(), hidden64=latent[0].tolist(), target_pu=target[0].tolist(),
               target_mw=(target[0]*base_mva).tolist(), mean_pu=mean.tolist(), std_pu=std.tolist(),
               original_head_prediction_pu=all_predictions[0,0,0].tolist(),
               unlearned20_head_prediction_pu=all_predictions[1,0,0].tolist(),
               core_checkpoint_own_lin2_prediction_pu=core_prediction.tolist(), baseMVA=base_mva))
    return dict(frozen_model_hashes=immutable, exact_latent_formula_max_error=max(analytic_errors),
                baseMVA=base_mva, rows=len(all_rows), interventions=2*len(raw)*204,
                qp_solves=2*len(all_rows), statuses=sorted({r['status1'] for r in all_rows}|{r['status2'] for r in all_rows}),
                solvers=sorted({r['solver1'] for r in all_rows}|{r['solver2'] for r in all_rows}),
                max_constraint_violation=max(max(r['violation1'],r['violation2']) for r in all_rows),
                max_objective_decomposition_error=max(r['objective_error'] for r in all_rows),
                coordinate_outside_encoded_bound_count=sum(r['encoded_value_outside_unit_bound'] for r in all_rows),
                unchanged_target=True, unchanged_model_hashes=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'research/cnn_sensitivity')
    parser.add_argument('--test-index', type=int, default=0)
    parser.add_argument('--context-samples', type=int, default=24)
    parser.add_argument('--fraction', type=float, default=0.1)
    parser.add_argument('--solver', choices=['auto','GUROBI','ECOS','OSQP','SCS'], default='auto')
    parser.add_argument('--replay-from', type=Path, help='Replay the saved true inputs and frozen weights without original data or influence files.')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    replay = args.replay_from.resolve() if args.replay_from else None
    if output.exists() and any(output.iterdir()):
        parser.error(f'Preserve existing results: output directory must be empty: {output}')
    if not np.isfinite(args.fraction) or args.fraction < 0:
        parser.error('fraction must be finite and nonnegative')
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    torch.set_num_threads(1)
    values = replay_models(replay, output) if replay else rebuild_models(args, output)
    cfg, core, heads, raw, latent, target, mean, std, catalog, metadata = values
    write_csv(output / 'raw_input_features_140.csv', catalog)
    write_json(output / 'forward_trace.json', capture_forward(core, torch.from_numpy(raw[:1])))
    validation = compute_study(cfg, core, heads, raw, latent, target, mean, std, catalog, metadata,
                               output, args.fraction, args.solver)
    write_json(output / 'manifest.json', dict(
        created_at_shanghai=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        run_mode='snapshot_replay' if replay else 'original_data_reconstruction',
        replay_source=str(replay) if replay else None,
        source_commit=repository_commit(ROOT),
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in
                       [Path(__file__), ROOT/'utils/sensitivity.py', ROOT/'utils/net.py', ROOT/'utils/optimization.py',
                        ROOT/'utils/dataset.py', ROOT/'utils/funcs.py', ROOT/'func_operation.py', ROOT/'clean_data.py']},
        model_validation=metadata, study_validation=validation, fraction=args.fraction,
        perturbation='x_j -> (1+fraction)*x_j; other coordinates, models and actual target fixed',
        limitations=['64 coordinates are learned tanh features, not named physical inputs',
                     'Raw inputs are 14 buses x 10 features; numerical +10% on encoded values is not physical +10%',
                     'Negative values become more negative and zeros stay zero; interventions can be off the feature manifold',
                     '24 evenly distributed observed test samples provide descriptive context, not population inference',
                     'Original preprocessing uses full-year feature/target statistics before the train/test split'],
        snapshot_sha256={filename:digest(output/filename) for filename in ('frozen_models.npz','reference_inputs.npz')}))
    print(json.dumps(validation, indent=2), flush=True)
    print(f'Actual sensitivity results saved to {output}', flush=True)


if __name__ == '__main__':
    main()
