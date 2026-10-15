"""Controls for training-sigma perturbations and the real saved experiment."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.input_sensitivity import LEVELS, NAMES, temperature_cases

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'research/cnn_temperature_response'


def test_additive_training_sigma_preserves_zero_and_all_other_coordinates():
    # Negative z-score warms on a positive additive step; a multiplier would fail.
    sample=np.full((1,14,10),-2.,dtype=np.float32)
    cases=temperature_cases(sample,.8,0)
    assert cases.shape==(5,1,14,10) and cases.dtype==np.float32
    np.testing.assert_array_equal(sample,np.full_like(sample,-2.))
    np.testing.assert_array_equal(cases[2],sample)
    np.testing.assert_allclose(cases[:,0,0,4],[-2.8,-2.4,-2.,-1.6,-1.2],atol=2e-7)
    for case in cases:
        actual=case.copy();actual[0,0,4]=sample[0,0,4]
        np.testing.assert_array_equal(actual,sample)


@pytest.mark.parametrize('sample,sigma,bus',[
    (np.zeros((1,14,10)),0.,0),(np.zeros((1,14,10)),np.nan,0),
    (np.zeros((1,14,10)),1.,14),(np.zeros((1,14,10)),1.,-1),
    (np.zeros((14,10)),1.,0),(np.ones((1,14,10),dtype=int),1.,0),
])
def test_temperature_control_rejects_invalid_inputs(sample,sigma,bus):
    with pytest.raises(ValueError):
        temperature_cases(sample,sigma,bus)


def test_real_training_sigma_is_independent_of_test_rows_and_physical_scaling():
    manifest=json.loads((RESULTS/'manifest.json').read_text())
    s=manifest['temperature_statistics']
    data=np.load(RESULTS/'reference_inputs.npz',allow_pickle=False)
    physical=data['official_temperature_kelvin']
    assert len(physical)==8760 and np.isfinite(physical).all()
    train=physical[1752:]
    # Independent physical training segment, not full-year or the 24 context samples.
    assert s['train_sigma_kelvin']==pytest.approx(train.std(ddof=0),rel=1e-12)
    assert s['original_annual_sample_std_kelvin']==pytest.approx(physical.std(ddof=1),rel=1e-12)
    assert s['train_sigma_encoded']==pytest.approx(train.std(ddof=0)/physical.std(ddof=1),rel=2e-8)
    assert not np.isclose(s['train_sigma_encoded'],1.,rtol=1e-3)
    reconstructed=(physical[int(data['test_index'])]-physical.mean())/physical.std(ddof=1)
    assert data['raw'][0,0,s['grid_bus']-1,4]==np.float32(reconstructed)


@pytest.mark.parametrize('model',list(NAMES))
def test_saved_real_model_response_has_zero_control_and_complete_original_cost(model):
    rows=pd.read_csv(RESULTS/'results.csv')
    selected=rows[rows.model==model].sort_values('sigma_level')
    np.testing.assert_array_equal(selected.sigma_level,LEVELS)
    assert np.isfinite(selected.select_dtypes('number')).all().all()
    np.testing.assert_allclose(selected.cost_total,selected.cost1+selected.cost2,rtol=0,atol=1e-10)
    zero=selected[selected.sigma_level==0].iloc[0]
    assert zero.prediction_max_abs_delta_mw==0.
    assert abs(zero.cost_delta)<=1e-6
    assert (selected.status1=='optimal').all() and (selected.status2=='optimal').all()
    assert selected[['violation1','violation2']].to_numpy().max()<=1e-5
    traces=json.loads((RESULTS/'cost_traces.json').read_text())[model]
    reference=np.load(RESULTS/'reference_inputs.npz',allow_pickle=False)['target_pu'][0]
    for trace in traces:
        np.testing.assert_array_equal(trace['target_pu'],reference)
        assert trace['cost_total']==pytest.approx(sum(trace['components'].values()),abs=1e-6)


def test_real_tamu_lp_budget_and_reference_parameter_order():
    snapshot=np.load(RESULTS/'frozen_models.npz',allow_pickle=False)
    weights=snapshot['tamu.lp_weights']
    assert len(weights)==3925 and len(snapshot['delete_indices'])==981
    assert np.abs(weights-1).sum()<=.1*3925+1e-5
    assert np.abs(weights-1).max()<=1.+1e-5
    effective=snapshot['tamu.effective_weights']
    np.testing.assert_array_equal(effective,weights.astype(np.float32).astype(np.float64))
    # Named parameter comparison catches the incompatible [Wi,bi] / [all W, all b] ordering.
    gap=max(np.abs(snapshot['tamu20.weight']-snapshot['tamu.weighted_exact.weight']).max(),
            np.abs(snapshot['tamu20.bias']-snapshot['tamu.weighted_exact.bias']).max())
    metadata=json.loads((RESULTS/'manifest.json').read_text())['model_restore']['tamu']
    assert gap==metadata['max_parameter_gap_vs_exact_weighted_fit']
    assert gap<.002 and metadata['cg_info']==0
