import pytest
from experimental_data.historical_fit import select_drone_inputs


def test_preliminary_recordings_cannot_enter_scoped_drone_fit():
    data={name:object() for name in ['fig8','whip1_001','vertical','whip1_002','whip1_003']}
    settings=dict(drone_response_takes=['whip1_001','whip1_002','whip1_003'],
        folds=[['fig8','whip1_001'],['vertical','whip1_002'],['whip1_003']])
    selected,folds=select_drone_inputs(data,settings)
    assert list(selected)==settings['drone_response_takes']
    assert folds==[['whip1_001'],['whip1_002'],['whip1_003']]
    settings['drone_response_takes'].append('missing')
    with pytest.raises(ValueError,match='unique available'):select_drone_inputs(data,settings)


def test_invalid_fold_partition_is_rejected():
    with pytest.raises(ValueError,match='each selected take once'):
        select_drone_inputs({'whip':{}},dict(drone_response_takes=['whip'],folds=[['whip'],['whip']]))
