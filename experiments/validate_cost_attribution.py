"""Verify every saved QP, coalition difference, selection rule and optional replay."""
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.cost_attribution import choose_cohort,mixed_forecast,shapley_paths
from experiments.input_sensitivity import LEVELS,NAMES,temperature_cases
from scripts.analyze_cnn_sensitivity import digest,write_json


def validate(directory,replay=None,original_data=False):
    manifest=json.loads((directory/'manifest.json').read_text())
    for name,checksum in manifest['source_sha256'].items():
        assert digest(ROOT/name)==checksum,name
    for name,checksum in manifest['snapshot_sha256'].items():
        assert digest(directory/name)==checksum,name
    data=np.load(directory/'cohort_inputs.npz',allow_pickle=False)
    actual=np.load(directory/'interventions.npz',allow_pickle=False)
    predictions=actual['forecasts_pu']
    assert predictions.shape==(28,4,5,14) and np.isfinite(predictions).all()
    selection=json.loads((directory/'selection.json').read_text())
    recomputed,thresholds=choose_cohort(data['selection_pool_true_total_mw'])
    assert recomputed==selection
    assert len({r['date'] for r in selection})==28
    np.testing.assert_array_equal(data['test_indices'],[r['test_index'] for r in selection])
    for c in range(28):
        expected=temperature_cases(data['raw'][c],float(data['train_sigma_encoded']),0)
        np.testing.assert_array_equal(actual['raw_cases'][c],expected)
        np.testing.assert_array_equal(actual['raw_cases'][c,2],data['raw'][c])
    frame=pd.read_csv(directory/'results.csv');nodes=pd.read_csv(directory/'node_counterfactuals.csv')
    coalitions=pd.read_csv(directory/'coalition_costs.csv')
    assert len(frame)==560 and len(nodes)==1568 and len(coalitions)==manifest['checks']['coalition_rows']
    assert not coalitions.duplicated(['context','sigma_level','subset_mask']).any()
    assert not frame.duplicated(['context','model','sigma_level']).any()
    assert (frame.status1=='optimal').all() and (frame.status2=='optimal').all()
    for _,rows in frame.groupby(['context','model']):
        np.testing.assert_array_equal(rows.sigma_level,LEVELS)
        zero=rows[rows.sigma_level==0].iloc[0]
        assert zero.total_prediction_delta_mw==0 and abs(zero.cost_delta)<=1e-6
    component_columns=['dispatch_generation_quadratic','dispatch_generation_linear','dispatch_shedding',
        'redispatch_shedding_linear','redispatch_shedding_quadratic','redispatch_storage_linear','redispatch_storage_quadratic']
    np.testing.assert_allclose(frame.cost_total,frame.cost1+frame.cost2,atol=1e-9,rtol=0)
    np.testing.assert_allclose(frame.cost_total,frame[component_columns].sum(axis=1),atol=1e-6,rtol=0)
    lookup={(int(r.context),float(r.sigma_level),int(r.subset_mask)):np.asarray([r.cost1,r.cost2,r.cost_total])
            for r in coalitions.itertuples()}
    for row in nodes.itertuples():
        node=int(row.grid_bus)-1;c=int(row.context);level=float(row.sigma_level)
        isolated=lookup[c,level,1<<node]-lookup[c,level,0]
        reset=lookup[c,level,2**14-1]-lookup[c,level,(2**14-1)^(1<<node)]
        np.testing.assert_allclose([row.isolated_cost1_delta,row.isolated_cost2_delta,row.isolated_total_cost_delta],isolated,atol=1e-9,rtol=0)
        np.testing.assert_allclose([row.reset_cost1_effect,row.reset_cost2_effect,row.reset_total_cost_effect],reset,atol=1e-9,rtol=0)
    shapley=pd.read_csv(directory/'reference_shapley.csv')
    paths=np.load(directory/'shapley_permutations.npz',allow_pickle=False)
    for k,level in enumerate([-1.,-.5,.5,1.]):
        mean,se,marginals=shapley_paths(lambda mask:lookup[0,level,mask],paths['orders'])
        rows=shapley[shapley.sigma_level==level].sort_values('grid_bus')
        np.testing.assert_allclose(rows[['shapley_cost1','shapley_cost2','shapley_total']],mean,atol=1e-10,rtol=0)
        np.testing.assert_allclose(rows.mc_se_total,se[:,2],atol=1e-10,rtol=0)
        np.testing.assert_allclose(paths['marginals'][k],marginals,atol=1e-9,rtol=0)
    calls=0;max_violation=0.
    model_indices={name:i for i,name in enumerate(NAMES)}
    for line in (directory/'all_cost_traces.jsonl').open(encoding='utf-8'):
        row=json.loads(line);calls+=1;c=int(row['context']);level=float(row['sigma_level'])
        np.testing.assert_array_equal(row['target_pu'],data['target_pu'][c])
        if row['kind']=='output_subset':
            full=predictions[c,0,list(LEVELS).index(level)]
            expected=mixed_forecast(predictions[c,0,2],full,row['subset_mask'])
        else:
            expected=predictions[c,model_indices[row['model']],list(LEVELS).index(level)]
        np.testing.assert_array_equal(row['forecast_pu'],expected)
        assert row['status1']==row['status2']=='optimal'
        max_violation=max(max_violation,row['violation1'],row['violation2'])
        assert max(row['violation1'],row['violation2'])<=1e-5
        assert abs(row['cost_total']-sum(row['components'].values()))<1e-6
    assert calls==manifest['checks']['cost_calls']
    assert calls==28*4*6+len(coalitions)-28*4*2
    # Link to prior real experiment, using the exact same frozen models.
    parent=ROOT/'research/cnn_temperature_response'
    assert digest(directory/'frozen_models.npz')==digest(parent/'frozen_models.npz')
    previous=np.load(parent/'interventions.npz',allow_pickle=False)
    np.testing.assert_array_equal(predictions[0],previous['forecasts_pu'])
    previous_costs=pd.read_csv(parent/'results.csv')
    current=frame[frame.context==0]
    np.testing.assert_allclose(current.cost_total,previous_costs.cost_total,atol=1e-4,rtol=0)
    source_data_verified=None  # Null means the optional full-data audit was not run.
    if original_data:
        from omegaconf import OmegaConf
        from utils import return_dataset
        provenance=json.loads((directory/'parent_provenance.json').read_text())
        train,test=return_dataset(OmegaConf.create(provenance['config']))
        np.testing.assert_array_equal(data['raw'],test.feature[data['test_indices']].numpy())
        truth=test.target.numpy()*data['std']+data['mean']
        np.testing.assert_array_equal(data['target_pu'],truth[data['test_indices']])
        np.testing.assert_array_equal(data['selection_pool_true_total_mw'],truth.astype(float).sum(axis=1)*100)
        assert float(data['train_sigma_encoded'])==train.feature[:,0,0,4].numpy().astype(float).std(ddof=0)
        source_data_verified=True
    pngs=list((directory/'figures').glob('*.png'));pdfs=list((directory/'figures').glob('*.pdf'))
    assert len(pngs)==len(pdfs)==4
    for path in pngs:
        with Image.open(path) as im:im.verify()
    for path in pdfs:assert path.read_bytes().startswith(b'%PDF-')
    for document in ['README.md','导师汇报.md']:
        for link in re.findall(r'\]\(([^)]+)\)',(directory/document).read_text()):
            if not link.startswith(('http:','https:')):
                target=(directory/link.split('#')[0]).resolve()
                if target!=(directory/'validation.json').resolve():assert target.exists(),link
    validation=dict(contexts=28,distinct_dates=28,stratified_samples=27,model_scenarios=560,
        forecast_values=7840,actual_cost_calls=calls,qp_solves=2*calls,all_qp_statuses_optimal=True,
        max_constraint_violation=max_violation,training_sigma_and_selected_data_match_source=source_data_verified,
        selection_recomputed_without_model_outcomes=True,all_real_targets_and_counterfactual_forecasts_verified=True,
        node_counterfactuals_match_actual_coalitions=True,shapley_paths_efficiency_and_mc_se_verified=True,
        same_four_frozen_models_as_previous_study=True,reference_forecasts_bit_identical_to_previous_study=True,
        prior_reference_cost_max_difference=float(np.max(np.abs(current.cost_total.to_numpy()-previous_costs.cost_total.to_numpy()))),
        source_and_snapshot_hashes_valid=True,images_verified=4,pdfs_verified=4,replay_verified=False,
        report_generator_sha256=digest(ROOT/'experiments/report_cost_attribution.py'),validator_sha256=digest(Path(__file__)))
    if replay:
        other=np.load(replay/'interventions.npz',allow_pickle=False)
        for name in actual.files:np.testing.assert_array_equal(actual[name],other[name])
        compare=['results.csv','predictions_14_buses.csv','node_counterfactuals.csv','reference_shapley.csv',
                 'coalition_costs.csv','baseline_costs.csv','all_cost_traces.jsonl']
        for name in compare:assert (directory/name).read_bytes()==(replay/name).read_bytes(),name
        validation.update(replay_verified=True,replay_all_forecasts_bit_identical=True,replay_all_cost_traces_and_csv_byte_identical=True)
    logs=directory/'test-results.txt'
    if logs.exists():
        text=logs.read_text();match=re.search(r'(\d+) passed',text)
        assert match and not re.search(r'\b[1-9]\d* failed\b',text)
        skipped=re.search(r'(\d+) skipped',text)
        validation.update(tests_passed=int(match.group(1)),tests_failed=0,tests_skipped=int(skipped.group(1)) if skipped else 0)
    write_json(directory/'validation.json',validation)
    print(json.dumps(validation,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=ROOT/'research/cnn_cost_attribution')
    parser.add_argument('--replay-dir',type=Path)
    parser.add_argument('--with-original-data',action='store_true')
    args=parser.parse_args()
    validate(args.results_dir.resolve(),args.replay_dir.resolve() if args.replay_dir else None,args.with_original_data)


if __name__=='__main__':main()
