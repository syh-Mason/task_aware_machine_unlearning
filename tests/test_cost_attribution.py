"""Controls for output-node diagnostics and non-additive cost attribution."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.cost_attribution import mixed_forecast, paired_permutations, shapley_paths, choose_cohort

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'research/cnn_cost_attribution'


def test_output_node_isolation_and_reset_have_exact_endpoints():
    baseline=np.arange(14,dtype=float)
    changed=baseline+np.linspace(-2,3,14)
    np.testing.assert_array_equal(mixed_forecast(baseline,changed,0),baseline)
    np.testing.assert_array_equal(mixed_forecast(baseline,changed,2**14-1),changed)
    for node in (0,4,13):
        isolated=mixed_forecast(baseline,changed,1<<node)
        reset=mixed_forecast(baseline,changed,(2**14-1)^(1<<node))
        assert isolated[node]==changed[node] and reset[node]==baseline[node]
        others=np.arange(14)!=node
        np.testing.assert_array_equal(isolated[others],baseline[others])
        np.testing.assert_array_equal(reset[others],changed[others])


def test_shapley_splits_interaction_and_forecast_magnitude_does_not_imply_cost_rank():
    # Analytic unit game, not fabricated experimental data: input changes [3,1],
    # costs [x1,10*x2] plus a 6-unit interaction. Both exact permutations are used.
    def game(mask):
        a,b=bool(mask&1),bool(mask&2)
        c1=3*a+10*b
        c2=6*a*b
        return np.array([c1,c2,c1+c2],dtype=float)
    orders=np.array([[0,1],[1,0],[0,1],[1,0]])
    mean,se,contributions=shapley_paths(game,orders,n=2)
    np.testing.assert_allclose(mean[:,2],[6,13])
    np.testing.assert_allclose(mean.sum(axis=0),[13,6,19])
    np.testing.assert_array_equal(se,np.zeros((2,3)))
    assert np.argmax([3,1])!=np.argmax(np.abs(mean[:,2]))
    # A one-at-a-time sum misses the interaction; it must not be labelled total attribution.
    assert sum(game(1<<i)[2]-game(0)[2] for i in range(2))!=game(3)[2]-game(0)[2]


def test_antithetic_permutations_are_reproducible_and_paired():
    orders=paired_permutations(42,16)
    np.testing.assert_array_equal(orders,paired_permutations(42,16))
    for first,second in orders.reshape(-1,2,14):
        np.testing.assert_array_equal(np.sort(first),np.arange(14))
        np.testing.assert_array_equal(first,second[::-1])


@pytest.mark.parametrize('bad',[-1,2**14])
def test_forecast_subset_rejects_invalid_masks(bad):
    with pytest.raises(ValueError):
        mixed_forecast(np.ones(14),np.ones(14),bad)


def test_real_cohort_selection_uses_distinct_dates_and_all_nine_strata():
    selection=json.loads((RESULTS/'selection.json').read_text())
    assert len(selection)==28 and len({r['date'] for r in selection})==28
    assert selection[0]['test_index']==0 and selection[0]['role']=='reference'
    strata=pd.DataFrame(selection[1:]).groupby(['month','load_group']).size()
    assert len(strata)==9 and (strata==3).all()
    assert all(0<=r['test_index']<1752 for r in selection)
    data=np.load(RESULTS/'cohort_inputs.npz',allow_pickle=False)
    reconstructed,_=choose_cohort(data['selection_pool_true_total_mw'])
    assert reconstructed==selection


def test_real_shapley_conserves_two_stage_cost_and_se_uses_independent_pairs():
    frame=pd.read_csv(RESULTS/'reference_shapley.csv')
    arrays=np.load(RESULTS/'shapley_permutations.npz',allow_pickle=False)
    for k,level in enumerate([-1.,-.5,.5,1.]):
        rows=frame[frame.sigma_level==level].sort_values('grid_bus')
        marginals=arrays['marginals'][k]
        pairs=marginals.reshape(-1,2,14,3).mean(axis=1)
        np.testing.assert_allclose(rows.shapley_total,pairs[:,:,2].mean(axis=0),atol=1e-10)
        np.testing.assert_allclose(rows.mc_se_total,pairs[:,:,2].std(axis=0,ddof=1)/np.sqrt(len(pairs)),atol=1e-10)
        assert rows.shapley_total.sum()==pytest.approx(rows.joint_cost_delta.iloc[0],abs=1e-8)


def test_real_cost_trace_does_not_change_true_load():
    data=np.load(RESULTS/'cohort_inputs.npz',allow_pickle=False)
    lines=(RESULTS/'all_cost_traces.jsonl').read_text().splitlines()
    for line in lines:
        row=json.loads(line)
        np.testing.assert_array_equal(row['target_pu'],data['target_pu'][row['context']])
        assert row['status1']==row['status2']=='optimal'
        assert max(row['violation1'],row['violation2'])<=1e-5
