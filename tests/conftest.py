"""Explicit availability gates for tests backed by non-distributed recordings."""
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
REQUIRED_DATA={
    'test_recordings_retain_motion_but_no_force_measurement':'data/processed_takes/osc_001/take.npz',
    'test_dataset_roles_preserve_an_untouched_test':'data/processed_takes/osc_001/take.npz',
    'test_real_take_build_has_new_contract_and_protects_test':'data/processed_takes/osc_001/take.npz',
    'test_real_take_build_has_new_contract_and_honors_explicit_protected_split':'data/processed_takes/osc_001/take.npz',
    'test_independent_hover_and_time_objective':'data/model_candidates/20260908-adp0-M1/model.json',
    'test_calibration_review_controls_keep_saved_candidate':'data/calibration_audits/20260905_constrained_fit',
    'test_five_pages_and_parallel_plot_viewport':'data/processed_takes/osc_001/take.npz',
    'test_fitting_emits_measured_and_predicted_validation_window':'data/force_takes/fig8_001/take.npz',
}


def pytest_collection_modifyitems(items):
    for item in items:
        name=item.originalname or item.name
        required=REQUIRED_DATA.get(name)
        if required and not (ROOT/required).exists():
            item.add_marker(pytest.mark.skip(reason=f'Separately held experimental data unavailable: {required}'))
