"""Controls for genuine one-coordinate interventions and complete QP costs."""
from pathlib import Path

from hydra import compose, initialize_config_dir
import numpy as np
import pytest

from utils.optimization import Operator
from utils.sensitivity import audited_cost, coordinate_perturbations


def test_github_zip_without_git_metadata_can_record_provenance(tmp_path):
    from scripts.analyze_cnn_sensitivity import repository_commit
    assert repository_commit(tmp_path) is None
    commit = repository_commit(Path(__file__).resolve().parents[1])
    # The whole project may itself have been downloaded as a GitHub ZIP.
    assert commit is None or (len(commit) == 40 and all(c in '0123456789abcdef' for c in commit))


def test_coordinate_intervention_isolated_negative_and_zero_controls():
    values = np.array([2., -3., 0.], dtype=np.float32)
    retained = values.copy()
    actual = coordinate_perturbations(values)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(values, retained)
    # Negative +10% means greater magnitude; zero is a no-op control.
    np.testing.assert_allclose(np.diag(actual), [2.2, -3.3, 0.], atol=2e-7)
    for dimension in range(3):
        others = np.arange(3) != dimension
        np.testing.assert_array_equal(actual[dimension, others], values[others])
    np.testing.assert_array_equal(actual[2], values)
    np.testing.assert_array_equal(coordinate_perturbations(values, 0), np.tile(values, (3, 1)))


def test_raw_coordinate_labels_match_bus_feature_tensor_layout():
    # Independently pin labelled slots used in the scientific interpretation:
    # input 1 = Grid01/Weekday_sin, 35 = Grid04/Temperature, 140 = Grid14/Wind.
    sample = np.ones((1, 14, 10), dtype=np.float32)
    scenarios = coordinate_perturbations(sample.reshape(-1)).reshape(140, 1, 14, 10)
    for index, bus, feature in [(0, 0, 0), (34, 3, 4), (139, 13, 9)]:
        expected = sample.copy()
        expected[0, bus, feature] = np.float32(1.1)
        np.testing.assert_array_equal(scenarios[index], expected)


@pytest.mark.parametrize('values,fraction', [([], .1), ([[1.,2.]], .1), ([np.nan], .1), ([1.], -.1), ([1.], np.inf)])
def test_interventions_reject_invalid_or_ambiguous_inputs(values, fraction):
    with pytest.raises(ValueError):
        coordinate_perturbations(values, fraction)


def test_audited_cost_retains_first_stage_shedding_and_fixed_truth():
    # Published case14 benchmark loads are a physics regression, not study data.
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / 'conf'), version_base=None):
        cfg = compose(config_name='config')
    operator = Operator(cfg.case, solver='ECOS')
    target = operator.default_load * 0.7
    untouched = target.copy()
    forecast = operator.default_load * 3
    result = audited_cost(operator, forecast, target)
    assert result['components']['dispatch_shedding'] > 0
    assert result['status1'] == result['status2'] == 'optimal'
    assert result['violation1'] <= 1e-5 and result['violation2'] <= 1e-5
    assert result['objective_error'] < 1e-6
    assert result['cost_total'] == pytest.approx(sum(result['components'].values()), abs=1e-6)
    pg, _, _, first, _ = operator.solve_one(forecast)
    _, _, _, second, _ = operator.solve_two(target, pg)
    assert result['cost_total'] == pytest.approx(float(first[0]+second[0]), rel=1e-7)
    audited_cost(operator, forecast * .9, target)
    np.testing.assert_array_equal(target, untouched)
    np.testing.assert_array_equal(operator.load_true.value, untouched)


def test_cost_audit_rejects_batches_in_single_sample_trace():
    class VectorWidth:
        no_load = 14
    with pytest.raises(ValueError, match='one forecast and target'):
        audited_cost(VectorWidth(), np.ones((2, 14)), np.ones(14))
