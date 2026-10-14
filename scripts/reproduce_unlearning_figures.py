"""Run the original baseline evaluations at the six paper proportions.

Uses full, preprocessed TX-123BT data and existing influence scores/core
checkpoints. It does not regenerate data, retrain cores, or change the model.
Plot the completed results with plot_unlearning_proportions.py.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROPORTIONS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
MODELS = {
    'linear': ('linear', 'real-full-double', 'mse', 7008),
    'conv': ('nn_conv', 'real-conv', 'cost', 4906),
    'mlpmixer': ('nn_mixer', 'real-mixer', 'cost', 4906),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_result(path):
    # These trusted files are produced by eval_unlearn.py in this checkout.
    result = np.load(path, allow_pickle=True).item()
    for metric in ('mse', 'mape', 'cost'):
        for method in ('ori', 'unlearn', 'retrain'):
            for split in ('remain', 'unlearn', 'test'):
                value = np.asarray(result[f'{metric}_{method}'][split])
                if value.shape != () or not np.isfinite(value) or value < 0:
                    raise ValueError(f'{path}: invalid {metric}_{method}/{split}: {value}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'outputs/paper-proportions')
    parser.add_argument('--jobs', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--reuse-20-percent', action='store_true',
                        help='Copy the previously verified full-data 20%% evaluations unchanged; record their hashes and source paths.')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from run_real import audit_data

    audit = audit_data()
    if not audit['processed_complete']:
        raise RuntimeError(f'Full processed real data is required: {audit}')
    result_dir = args.results_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       TAMU_SOLVER='auto', MPLBACKEND='Agg')
    inputs = {}
    for config, (model, previous, criteria, count) in MODELS.items():
        score = ROOT / 'outputs' / previous / 'influence' / f'{model}_{criteria}.npy'
        values = np.load(score, allow_pickle=False)
        if values.shape != (count,) or not np.isfinite(values).all():
            raise ValueError(f'{score}: expected {count} finite influence scores')
        inputs[config] = dict(model_type=model, influence=str(score), influence_sha256=sha256(score),
                              influence_criterion=criteria, train_samples=count)
        if config != 'linear':
            checkpoint = ROOT / 'trained_model' / model / 'core.pth'
            inputs[config].update(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
        if args.reuse_20_percent:
            validate_result(ROOT / 'outputs' / previous / 'simulation_result' / model / 'unlearning_0.2.npy')

    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    python=sys.version, proportions=PROPORTIONS,
                    settings=dict(case='case14', data_prop=1, train_prop=0.8, random_seed=42,
                                  unlearn_mode='helpful', core_prop=0.3, solver='auto'),
                    data_audit=audit,
                    processed_data_sha256={p.name: sha256(p) for p in sorted((ROOT / 'data/data_case14').glob('bus_*.csv'))},
                    inputs=inputs)
    (result_dir / 'run-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def run_model(config):
        model, previous, _, _ = MODELS[config]
        records = []
        for proportion in PROPORTIONS:
            label = f'{model}-{proportion}'
            destination = result_dir / model / f'unlearning_{proportion}.npy'
            destination.parent.mkdir(parents=True, exist_ok=True)
            if proportion == 0.2 and args.reuse_20_percent:
                source = ROOT / 'outputs' / previous / 'simulation_result' / model / destination.name
                shutil.copy2(source, destination)
                record = dict(proportion=proportion, reused=True, source=str(source), sha256=sha256(destination))
                if record['sha256'] != sha256(source):
                    raise RuntimeError(f'{label}: reused result was changed')
                print(f'{label}: reused verified result from {source}', flush=True)
            else:
                command = [sys.executable, str(ROOT / 'eval_unlearn.py'), f'model={config}',
                           'case=case14', 'data.data_prop=1', 'data.train_prop=0.8',
                           'data.random_seed=42', 'data.random_split=false', 'unlearn_mode=helpful',
                           f'unlearn_prop={proportion}', f'model.influence_dir={inputs[config]["influence"]}',
                           f'simulation_dir={result_dir}', f'hydra.run.dir={result_dir / "hydra" / label}']
                started = time.time_ns()
                log_path = result_dir / f'{label}.log'
                print(f'{label}: running full evaluation; log {log_path}', flush=True)
                with log_path.open('w') as log:
                    completed = subprocess.run(command, cwd=ROOT, env=environment,
                                               stdout=log, stderr=subprocess.STDOUT)
                if completed.returncode:
                    raise RuntimeError(f'{label}: exit {completed.returncode}; inspect {log_path}')
                if not destination.is_file() or destination.stat().st_mtime_ns < started:
                    raise RuntimeError(f'{label}: no fresh result file {destination}')
                record = dict(proportion=proportion, reused=False, command=command, exit_code=completed.returncode,
                              elapsed_seconds=(time.time_ns()-started)/1e9,
                              log=str(log_path), sha256=sha256(destination))
            result = validate_result(destination)
            record.update(result=str(destination), test_cost={method: float(result[f'cost_{method}']['test'])
                                                              for method in ('ori', 'unlearn', 'retrain')})
            records.append(record)
            (result_dir / f'{model}-runs.json').write_text(json.dumps(records, indent=2) + '\n')
            print(f'{label}: validated all 27 metrics; test cost {record["test_cost"]}', flush=True)
        return records

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        # Each worker has separate result, Hydra and log paths.
        list(pool.map(run_model, MODELS))
    print(f'Completed and validated all 18 model/proportion evaluations: {result_dir}', flush=True)


if __name__ == '__main__':
    main()
