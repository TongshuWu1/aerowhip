import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication
from simulator.gui.recording_rounds_page import RecordingRoundsPage


def test_empty_round_library_is_safe(tmp_path):
    app=QApplication.instance() or QApplication([])
    page=RecordingRoundsPage(tmp_path)
    assert page.rounds.count()==0 and page.table.rowCount()==0
    assert page.role.currentText()=='unassigned'
    page.process_selected();page.save_selected_review()
    assert not page.job.running
    page.close();app.processEvents()


def test_archived_round_is_not_loaded_or_processed(tmp_path):
    import pytest
    from experimental_data.adaptation_rounds import process_round
    folder=tmp_path/'data/adaptation_rounds/adaptation0'
    folder.mkdir(parents=True)
    # Invalid metadata would fail if the archived recording were inspected.
    (folder/'round.json').write_text('not to be opened')
    (folder/'ARCHIVED.json').write_text('{}')
    app=QApplication.instance() or QApplication([])
    page=RecordingRoundsPage(tmp_path)
    assert page.rounds.count()==0 and page.table.rowCount()==0
    assert page.number.value()==1
    with pytest.raises(ValueError,match='archived'):
        process_round(folder)
    page.close();app.processEvents()


def test_later_preparation_authorization_shows_legacy_phases(tmp_path):
    import json
    folder=tmp_path/'data/adaptation_rounds/adaptation0'
    version=folder/'processed/20260908-legacy';version.mkdir(parents=True)
    (folder/'round.json').write_text('{}')
    (folder/'ARCHIVED.json').write_text('{"reason":"old decision"}')
    (folder/'PREPARATION_AUTHORIZATION.json').write_text(json.dumps(dict(
        schema='recording_preparation_authorization_v1',allow_preparation=True)))
    report=dict(trial_id='whip1_001',status='processed',alignment=dict(offset_s=.1,rms_m=.01),
        missing_cable_samples_per_marker=[0]*10,logged_position_update_hz=10,
        candidate_motion_intervals=[dict(start_s=18.,end_s=18.67)],
        reference_comparison=dict(prefix_matches=True,recorded_distinct_samples=20,reference_samples=25,absent_tail_times_s=[.8]),
        execution_phases=dict(csv_start_s=18.,csv_end_s=18.67))
    (version/'processing.json').write_text(json.dumps(dict(reports=[report])))
    app=QApplication.instance() or QApplication([])
    page=RecordingRoundsPage(tmp_path)
    assert page.rounds.count()==1 and page.table.rowCount()==1
    assert 'Legacy CSV-only maneuver' in page.summary.text()
    assert 'measured end position' in page.summary.text()
    assert not page.job.running
    assert (folder/'ARCHIVED.json').is_file()
    page.close();app.processEvents()
