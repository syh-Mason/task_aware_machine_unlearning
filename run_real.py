"""Validate real data, then run the documented linear cost experiment.

No synthetic-data fallback or omitted cost stage is provided. Raw-data
preprocessing follows clean_data.py unchanged. Use --check-data to audit only.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FEATURES = ['Weekday_sin', 'Weekday_cos', 'Hour_sin', 'Hour_cos',
            'Temperature (k)', 'Shortwave Radiation (w/m2)',
            'Longwave Radiation (w/m2)', 'Zonal Wind Speed (m/s)',
            'Meridional Wind Speed (m/s)', 'Wind Speed (m/s)', 'Load']


def audit_data(root=ROOT):
    raw = root / 'data' / 'Data_public'
    expected = ([raw / 'load_2019' / f'load_annual_D{day}.txt' for day in range(1, 366)]
                + [raw / 'Climate_2019' / f'climate_2019_Day{day}.csv' for day in range(1, 366)])
    missing = [str(path.relative_to(root)) for path in expected if not path.is_file() or path.stat().st_size == 0]
    processed = root / 'data' / 'data_case14'
    files = sorted(processed.glob('bus_*.csv'))
    errors = []
    if len(files) != 14:
        errors.append(f'Expected 14 processed bus CSVs; found {len(files)}')
    for path in files:
        try:
            frame = pd.read_csv(path)
            if list(frame.columns) != FEATURES or frame.shape != (8760, 11):
                errors.append(f'{path.name}: expected 8760 rows and the documented 11 columns; got {frame.shape}')
            values = frame.to_numpy(dtype=float)
            if not np.isfinite(values).all():
                errors.append(f'{path.name}: non-finite values')
            if len(values) and (np.any(values[:, -1] < 0) or np.std(values[:, -1]) == 0):
                errors.append(f'{path.name}: invalid or constant load')
        except (ValueError, OSError, pd.errors.ParserError) as error:
            errors.append(f'{path.name}: {error}')
    return dict(raw_expected_files=730, raw_present_files=730-len(missing),
                raw_complete=not missing, raw_missing_examples=missing[:6],
                processed_files=len(files), processed_complete=not errors,
                processed_errors=errors,
                provenance='Check archive against Figshare article 22144616 version 4; shape checks alone do not certify data provenance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-data', action='store_true')
    parser.add_argument('--data-prop', type=float, default=1.0,
                        help='Use 0.01 for the first real-data run; 1.0 uses all 8760 hours.')
    parser.add_argument('--unlearn-prop', type=float, default=0.2)
    parser.add_argument('--solver', choices=['auto', 'GUROBI', 'ECOS', 'OSQP', 'SCS'], default='auto')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'real')
    args = parser.parse_args()
    if not 0 < args.data_prop <= 1 or not 0 < args.unlearn_prop <= 0.3:
        parser.error('data-prop must be in (0,1]; unlearn-prop must be in (0,0.3] for the original candidate-selection algorithm')
    audit = audit_data()
    print(json.dumps(audit, indent=2), flush=True)
    if args.check_data:
        return 0 if audit['raw_complete'] or audit['processed_complete'] else 1
    if not audit['processed_complete'] and not audit['raw_complete']:
        print('Real TX-123BT data is incomplete. Supply the original archive; no experiment or cost result was generated.', file=sys.stderr)
        return 1
    if not audit['processed_complete'] and audit['processed_files']:
        print('Existing processed data is invalid; inspect and preserve it before preprocessing. No files were overwritten.', file=sys.stderr)
        return 1
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'data-audit.json').write_text(json.dumps(audit, indent=2))
    environment = os.environ.copy()
    environment['TAMU_SOLVER'] = args.solver
    def run(label, script, overrides):
        command = [sys.executable, str(ROOT / script), *overrides,
                   f'hydra.run.dir={output / "hydra" / label}']
        print('Running:', ' '.join(command), flush=True)
        with (output / f'{label}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    if not audit['processed_complete']:
        run('clean-data', 'clean_data.py', [])
        audit = audit_data()
        if not audit['processed_complete']:
            raise RuntimeError(f'Preprocessing did not produce valid data: {audit}')
        (output / 'data-audit.json').write_text(json.dumps(audit, indent=2))
    common = ['model=linear', f'data.data_prop={args.data_prop}', f'unlearn_prop={args.unlearn_prop}']
    run('check', 'check.py', common)
    influence_dir = output / 'influence'
    run('gen-index', 'gen_index.py', common + [f'influence_dir={influence_dir}'])
    evaluation = common + [f'model.influence_dir={influence_dir / "linear_mse.npy"}',
                           f'simulation_dir={output / "simulation_result"}']
    run('eval-unlearn', 'eval_unlearn.py', evaluation)
    run('eval-task-aware', 'eval_unchange.py', evaluation + ['criteria=cost'])
    print(f'Completed real-data linear baseline and task-aware cost experiments. Results: {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
