from copy import deepcopy

import pytest

from tools.continue_cable_stopping import amended_plateau


def inputs():
    return (
        dict(minimum=40, patience=3, relative=.01, check_every=5, ceiling=None),
        dict(plateau=dict(minimum=40, patience=6, relative=.005,
                          best=1.0, anchor=1.01, stale=5)),
    )


def test_resets_patience_without_discarding_best_or_mutating_source():
    settings, saved = inputs()
    original = deepcopy(saved)
    stop = amended_plateau(settings, saved, 1.001)
    assert saved == original
    assert stop.best == 1.0
    assert stop.stale == 0
    assert stop.anchor == 1.001
    assert not stop.observe(130, 1.001)[1]
    assert not stop.observe(135, 1.0005)[1]
    assert stop.observe(140, 1.0002)[1]
    assert stop.best == 1.0


def test_meaningful_progress_resets_patience_and_keeps_small_improvements():
    settings, saved = inputs()
    stop = amended_plateau(settings, saved, 1.0)
    assert stop.observe(130, .996) == (True, False)
    assert stop.stale == 1
    assert stop.observe(135, .989) == (True, False)
    assert stop.stale == 0
    assert stop.best == .989


@pytest.mark.parametrize('change', [dict(relative=.001), dict(relative=float('nan')),
    dict(relative=1.), dict(patience=0), dict(patience=7), dict(minimum=0)])
def test_rejects_unrelated_or_invalid_policy_changes(change):
    settings, saved = inputs()
    settings.update(change)
    with pytest.raises(ValueError):
        amended_plateau(settings, saved, 1.0)


def test_monitor_discloses_cable_only_override(tmp_path):
    import json
    from simulator.gui.fit_monitor import snapshot
    settings, _ = inputs()
    protocol = dict(full_update=dict(residual_stopping=dict(settings, relative=.005, patience=6),
                                     cable_residual_stopping=settings))
    (tmp_path/'protocol.json').write_text(json.dumps(protocol), encoding='utf-8')
    description = snapshot(tmp_path)['stopping']
    assert '6 checks without 0.5%' in description
    assert 'Cable-only amendment: 3 checks without 1%' in description
    assert 'vehicle rule unchanged' in description
