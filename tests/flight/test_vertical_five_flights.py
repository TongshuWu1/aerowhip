from copy import deepcopy
import pytest
from tools.evaluate_vertical_five_flights import select_flights


def fixture():
    return {f'g{g}_{i}':dict(collection='M0', keep=True, recording_group=f'g{g}')
            for g, size in enumerate((1, 3, 4)) for i in range(size)}


def test_selection_is_reproducible_balanced_unique_and_filtered():
    rows = fixture()
    rows['excluded'] = dict(collection='M0', keep=False, recording_group='g1')
    rows['other_model'] = dict(collection='M1', keep=True, recording_group='g1')
    selected = select_flights(rows, 'M0', 20260915)
    assert selected == select_flights(dict(reversed(list(rows.items()))), 'M0', 20260915)
    assert len(set(selected['selected_takes'])) == 5
    assert sorted(selected['quotas'].values()) == [1, 2, 2]
    assert 'excluded' not in selected['selected_takes'] and 'other_model' not in selected['selected_takes']


def test_selection_does_not_read_outcomes_or_initial_state_magnitudes():
    rows = fixture()
    changed = deepcopy(rows)
    for i, row in enumerate(changed.values()):
        row.update(error=1e10-i, contact_success=False, values={'vehicle_speed_m_s':float('nan')})
    assert select_flights(rows, 'M0', 123) == select_flights(changed, 'M0', 123)


def test_selection_rejects_insufficient_data():
    rows = fixture()
    with pytest.raises(ValueError):
        select_flights(rows, 'M0', 123, count=9)
    with pytest.raises(ValueError):
        select_flights(rows, 'M0', 123, count=2)
