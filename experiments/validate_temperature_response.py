"""Validate saved physical input, frozen models, complete costs and optional replay."""
import argparse
import json
from pathlib import Path
import re
import sys

from PIL import Image
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.input_sensitivity import MATH_FILES, LEVELS, NAMES, temperature_cases
from scripts.analyze_cnn_sensitivity import digest, write_json


def validate(directory,replay=None):
    manifest=json.loads((directory/'manifest.json').read_text())
    for name,checksum in manifest['snapshot_sha256'].items():
        assert digest(directory/name)==checksum, name
    for name,checksum in manifest['source_sha256'].items():
        assert digest(ROOT/name)==checksum, name
    data=np.load(directory/'reference_inputs.npz',allow_pickle=False)
    interventions=np.load(directory/'interventions.npz',allow_pickle=False)
    stats=manifest['temperature_statistics']
    expected=temperature_cases(data['raw'][0],float(data['train_sigma_encoded']),stats['grid_bus']-1)
    np.testing.assert_array_equal(expected,interventions['raw_cases'])
    np.testing.assert_array_equal(interventions['target_pu'],data['target_pu'][0])
    assert interventions['forecasts_pu'].shape==(4,5,14)
    assert np.isfinite(interventions['forecasts_pu']).all()
    frame=pd.read_csv(directory/'results.csv')
    buses=pd.read_csv(directory/'predictions_14_buses.csv')
    assert len(frame)==20 and len(buses)==280
    assert set(frame.model)==set(NAMES)
    for model in NAMES:
        rows=frame[frame.model==model].sort_values('sigma_level')
        np.testing.assert_array_equal(rows.sigma_level,LEVELS)
        zero=rows[rows.sigma_level==0].iloc[0]
        assert zero.prediction_max_abs_delta_mw==0 and abs(zero.cost_delta)<1e-6
    np.testing.assert_allclose(frame.cost1+frame.cost2,frame.cost_total,rtol=0,atol=1e-10)
    assert (frame.status1=='optimal').all() and (frame.status2=='optimal').all()
    assert frame[['violation1','violation2']].to_numpy().max()<=1e-5
    components=['dispatch_generation_quadratic','dispatch_generation_linear','dispatch_shedding',
                'redispatch_shedding_linear','redispatch_shedding_quadratic',
                'redispatch_storage_linear','redispatch_storage_quadratic']
    np.testing.assert_allclose(frame[components].sum(axis=1),frame.cost_total,rtol=0,atol=1e-6)
    traces=json.loads((directory/'cost_traces.json').read_text())
    for model,model_traces in traces.items():
        for trace in model_traces:
            np.testing.assert_array_equal(trace['target_pu'],data['target_pu'][0])
    pngs=list((directory/'figures').glob('*.png'));pdfs=list((directory/'figures').glob('*.pdf'))
    assert len(pngs)==4 and len(pdfs)==4
    for path in pngs:
        with Image.open(path) as image:
            image.verify()
    for path in pdfs:
        assert path.read_bytes().startswith(b'%PDF-')
    for document in ('README.md','导师汇报.md'):
        text=(directory/document).read_text()
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if not target.startswith(('https:','http:')):
                resolved=(directory/target.split('#')[0]).resolve()
                # This validator writes its own linked output only after all checks pass.
                if resolved!=(directory/'validation.json').resolve():
                    assert resolved.exists(), target
    validation=dict(scenarios=20,forecast_values=280,frozen_input_shape=list(data['raw'].shape),
        all_14_outputs_finite=True,only_selected_temperature_changes=True,unchanged_actual_load=True,
        zero_forecast_exact=True,zero_cost_max_error=float(frame.loc[frame.sigma_level==0,'cost_delta'].abs().max()),
        original_seven_cost_components_verified=True,all_statuses_optimal=True,
        max_constraint_violation=float(frame[['violation1','violation2']].to_numpy().max()),
        snapshot_and_source_checksums_valid=True,pngs_verified=len(pngs),pdfs_verified=len(pdfs),
        relative_report_links_valid=True,replay_verified=False,
        report_generator_sha256=digest(ROOT/'experiments/report_temperature_response.py'),
        validator_sha256=digest(Path(__file__)))
    if replay:
        other=np.load(replay/'interventions.npz',allow_pickle=False)
        for name in interventions.files:
            np.testing.assert_array_equal(interventions[name],other[name])
        for name in ('results.csv','predictions_14_buses.csv','cost_traces.json'):
            assert (directory/name).read_bytes()==(replay/name).read_bytes(), name
        validation.update(replay_verified=True,replay_all_forecasts_bit_identical=True,
                           replay_results_and_cost_traces_byte_identical=True)
    logs=directory/'test-results.txt'
    if logs.exists():
        logtext=logs.read_text()
        match=re.search(r'(\d+) passed',logtext)
        # Solver fallback warnings can contain "GUROBI failed" inside a passing suite.
        assert match and not re.search(r'\b[1-9]\d* failed\b',logtext)
        validation['tests_passed']=int(match.group(1))
        skipped=re.search(r'(\d+) skipped',logtext)
        validation['tests_skipped']=int(skipped.group(1)) if skipped else 0
        validation['tests_failed']=0
    write_json(directory/'validation.json',validation)
    print(json.dumps(validation,indent=2))
    return validation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=ROOT/'research/cnn_temperature_response')
    parser.add_argument('--replay-dir',type=Path)
    args=parser.parse_args()
    validate(args.results_dir.resolve(),args.replay_dir.resolve() if args.replay_dir else None)


if __name__=='__main__':
    main()
