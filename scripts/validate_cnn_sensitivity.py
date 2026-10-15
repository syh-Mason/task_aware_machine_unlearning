"""Verify measured artifacts, source hashes, replay and optional original data."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=ROOT/'research/cnn_sensitivity')
    parser.add_argument('--replay-dir',type=Path)
    parser.add_argument('--with-original-data',action='store_true')
    args=parser.parse_args();root=args.results_dir.resolve()
    meta=json.loads((root/'manifest.json').read_text())
    snapshot=np.load(root/'reference_inputs.npz',allow_pickle=False)
    catalog=pd.read_csv(root/'raw_input_features_140.csv')
    assert len(catalog)==140 and catalog.input_index1.tolist()==list(range(1,141))
    if args.with_original_data:
        for bus,rows in catalog.groupby('grid_bus'):
            frame=pd.read_csv(ROOT/rows.source_file.iloc[0])
            assert list(frame.columns[:-1])==rows.feature.tolist()
            actual=frame.iloc[snapshot['test_indices'],:-1].to_numpy().astype(np.float32)
            np.testing.assert_array_equal(actual,snapshot['raw'][:,0,int(bus)-1,:])
            np.testing.assert_allclose(frame.iloc[:,4:10].mean(),0,atol=1e-12)
            np.testing.assert_allclose(frame.iloc[:,4:10].std(ddof=1),1,atol=1e-12)
    assert len(pd.read_csv(root/'latent_features_64.csv'))==64
    rows=pd.read_csv(root/'all_interventions.csv')
    contexts=len(snapshot['raw']);expected_rows=2*contexts*205
    assert len(rows)==expected_rows and not rows.duplicated(['model','context','space','input_index1']).any()
    assert rows.groupby('space').size().to_dict()=={'baseline':2*contexts,'latent64':128*contexts,'raw140':280*contexts}
    assert np.isfinite(rows[['cost_total','cost1','cost2','prediction_mae_mw','cost_delta']]).all().all()
    assert (rows.status1=='optimal').all() and (rows.status2=='optimal').all()
    np.testing.assert_allclose(rows.cost1+rows.cost2,rows.cost_total,rtol=1e-12,atol=1e-10)
    assert max(rows.violation1.max(),rows.violation2.max())<=1e-5
    forecasts=np.load(root/'all_predictions.npz',allow_pickle=False)['forecasts_pu']
    replay_matches=None
    if args.replay_dir:
        replay=args.replay_dir.resolve()
        np.testing.assert_array_equal(forecasts,np.load(replay/'all_predictions.npz',allow_pickle=False)['forecasts_pu'])
        assert (root/'all_interventions.csv').read_bytes()==(replay/'all_interventions.csv').read_bytes()
        replay_matches=True
    trace=json.loads((root/'fixed_sample_cost_trace.json').read_text())
    for value in trace.values():
        np.testing.assert_array_equal(np.asarray(value['target_pu'],dtype=np.float32),snapshot['target_pu'][0])
    for name,wanted in meta['source_sha256'].items(): assert sha(ROOT/name)==wanted,name
    for name,wanted in meta['snapshot_sha256'].items(): assert sha(root/name)==wanted,name
    images=list((root/'figures').glob('*.png'));pdfs=list((root/'figures').glob('*.pdf'))
    assert len(images)==len(pdfs)==10
    for path in images:
        with Image.open(path) as im: im.verify()
    for path in pdfs:
        contents=path.read_bytes();assert contents.startswith(b'%PDF-') and contents.rstrip().endswith(b'%%EOF')
    for target in re.findall(r'\]\(([^)]+)\)',(root/'README.md').read_text()):
        if not target.startswith(('https://','http://')):
            assert (root/target.split('#')[0]).resolve().exists(),target
    test_log=root/'test-results.txt'
    test_count=None
    if test_log.is_file():
        log=test_log.read_text();match=re.search(r'(\d+) passed',log)
        assert match and 'failed,' not in log and not re.search(r'\d+ skipped',log)
        test_count=int(match.group(1))
    proof=dict(verified_at_shanghai=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
               tests_passed=test_count,tests_failed=0 if test_count else None,tests_skipped=0 if test_count else None,
               raw_coordinate_mapping_matches_original_csv=True if args.with_original_data else None,
               weather_annual_mean_zero_sample_std_one=True if args.with_original_data else None,
               model_and_input_snapshot_hashes_valid=True,source_hashes_valid=True,
               study_rows=expected_rows,interventions=408*contexts,main_qp_solves=2*expected_rows,
               replay_qp_solves=2*expected_rows if replay_matches else None,
               replay_forecast_values=int(forecasts.size),all_replay_forecasts_bit_identical=replay_matches,
               all_intervention_csv_replay_byte_identical=replay_matches,
               fixed_trace_targets_match_original_snapshot=True,pngs_verified=len(images),pdfs_verified=len(pdfs),
               report_links_resolve=True,report_generator_sha256=sha(ROOT/'scripts/report_cnn_sensitivity.py'))
    (root/'validation.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(proof,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
