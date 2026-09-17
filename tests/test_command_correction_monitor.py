import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import json
from pathlib import Path
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from simulator.gui import correction_monitor as monitor
from simulator.gui.command_correction_page import CommandCorrectionPage


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


def fixture(root):
    job = root/'runs/reference_tracking/M2-production'
    previous = root/'runs/rehearsals_pva/M1'
    corrected = root/'runs/rehearsals_pva/M2'
    optimization = root/'runs/reference_tracking/M2-optimizer'
    for path in (job, previous, corrected, optimization):path.mkdir(parents=True, exist_ok=True)
    time = np.array([0., .5, 1.])
    ref = np.array([[0., 0., 1.], [.5, 0., 1.], [1., 0., 1.]])
    commands = np.zeros((3, 11));commands[:, :3] = ref
    previous_commands = commands.copy();previous_commands[:, 0] += .1
    corrected_commands = commands.copy();corrected_commands[:, 0] += .02
    write(job/'settings.json', dict(correction=dict(initialization=str(previous), original_spline_duration_s=1.5,
          optimizer=dict(optimality_tolerance=1e-5)), launch=dict(origin_m=[0., 0., 1.4])))
    write(job/'reference.json', dict(interval_s=[0., 1.], planned_strike_time_s=.5,
          physical_target_m=[.5, 0., 1.], source_rehearsal='runs/rehearsals_pva/M0'))
    write(job/'status.json', dict(status='completed', stage='Corrected CSV exported'))
    write(job/'baseline.json', dict(cost_m2=.01))
    write(job/'result.json', dict(rehearsal=str(corrected), baseline_cost_m2=.01, corrected_cost_m2=.002,
          original_reference_tip_rmse_m=.1, corrected_reference_tip_rmse_m=.02,
          optimization=dict(iterations=2, stop_reason='Iteration budget reached', reused_controls_from=str(optimization))))
    write(optimization/'history.json', [dict(iteration=1, cost_m2=.004, accepted=True, command_feasible_optimality=.01),
                                      dict(iteration=2, cost_m2=.002, accepted=True, command_feasible_optimality=.005)])
    np.savez(job/'reference.npz', time_s=time, tip_position_m=ref, quadrotor_position_m=ref,
             command_time_s=time, original_command_packets=commands)
    # A previous rehearsal's OLD-model prediction must never be the baseline.
    np.savez(previous/'rehearsal.npz', command_time_s=time, commands=previous_commands,
             prediction_time_s=time, cable_positions_m=(ref+9)[:, None], origin_positions_m=ref+9)
    np.savez(job/'baseline_prediction.npz', time_s=time, cable_positions_m=(ref+[.1, 0, 0])[:, None],
             origin_positions_m=ref+[.1, 0, 0])
    np.savez(corrected/'rehearsal.npz', command_time_s=np.r_[time, 1.5],
             commands=np.r_[corrected_commands, corrected_commands[-1:]], prediction_time_s=time,
             cable_positions_m=(ref+[.02, 0, 0])[:, None], origin_positions_m=ref+[.02, 0, 0])
    (corrected/'fullstate_30hz.csv').write_text('fixture', encoding='utf-8')
    return job, previous, corrected, optimization


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def test_same_model_predictions_and_reused_convergence(tmp_path):
    job, previous, corrected, optimizer = fixture(tmp_path)
    v = monitor.snapshot(tmp_path, job)
    assert v['previous'] == previous and v['rehearsal'] == corrected
    assert v['history_path'] == optimizer and len(v['history']) == 2
    np.testing.assert_allclose(v['errors']['previous']['tip']['values'], 10.)
    np.testing.assert_allclose(v['errors']['corrected']['tip']['values'], 2.)
    assert v['strike_rows'][1][1] == pytest.approx(10.)
    assert v['strike_rows'][2][2] == pytest.approx(2.)
    assert len(v['commands']['previous']['time']) == 3
    assert len(v['commands']['corrected']['time']) == 4
    assert v['optimization']['stop_reason'] == 'Iteration budget reached'
    # Truncated predictions have no extrapolated strike score or extra errors.
    short = dict(time=np.array([0., .25]), values=np.ones((2, 3)))
    assert monitor.at_time(short, .5) is None
    assert monitor.error_curve(short, v['reference']['tip']) is None
    # Missing M1 cannot silently substitute the M0 reference command.
    (previous/'rehearsal.npz').unlink()
    v = monitor.snapshot(tmp_path, job)
    assert v['commands']['previous'] is None and v['warnings']


def test_ui_refresh_selection_views_and_no_artifact_writes(tmp_path, app):
    job, _, _, optimizer = fixture(tmp_path)
    before = {p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    page = CommandCorrectionPage(tmp_path)
    try:
        assert page.jobs.currentData() == str(job)
        assert page.open_csv.isEnabled()
        assert '80%' in page.cards[0][1].text()
        assert 'Iteration budget reached' in page.cards[3][1].text()
        for mode in range(3):
            page.mode.setCurrentIndex(mode);page.canvas.draw()
            assert page.figure.axes
        assert 'reused' in page.explanation.text()
        assert all(p.read_bytes() == data for p, data in before.items())
        write(optimizer/'history.json', [dict(iteration=3, cost_m2=.001, command_feasible_optimality=.003)])
        page.poll(force=True)
        assert page.view['history'][0]['iteration'] == 3
        # Selecting a run manually must not jump to a newly discovered active job.
        page.select_job(0)
        active = tmp_path/'runs/reference_tracking/new'
        write(active/'settings.json', dict(correction=dict(initialization='unknown')))
        write(active/'status.json', dict(status='running'))
        page.poll(force=True)
        assert page.jobs.currentData() == str(job)
        page.follow.setChecked(True)
        assert page.jobs.currentData() == str(active)
        assert not page.open_csv.isEnabled()
    finally:
        page.shutdown();page.close()


def test_live_controls_and_partial_artifacts(tmp_path, app):
    job, _, corrected, _ = fixture(tmp_path)
    (job/'result.json').unlink()
    write(job/'status.json', dict(status='running'))
    # Native 30 Hz saved control preview; no model propagation is needed.
    times = np.arange(35)/30
    np.savez(job/'reference.npz', command_time_s=times)
    controls = np.tile([0., 0., 1.4], (9, 1));controls[:, 0] = np.linspace(0, 1, 9)
    np.savez(job/'current_best.npz', position_control_points_m=controls)
    page = CommandCorrectionPage(tmp_path)
    try:
        assert page.view['provisional'] and not page.open_csv.isEnabled()
        assert page.view['commands']['corrected']['values'].shape == (35, 11)
        assert page.view['predictions']['corrected']['tip'] is None
        assert 'whip only' in page.explanation.text()
        (job/'current_best.npz').write_bytes(b'partial zip')
        (job/'history.json').write_text('[', encoding='utf-8')
        page.poll(force=True);page.canvas.draw()
        assert page.view['commands']['corrected'] is None
        assert page.view['warnings']
    finally:
        page.shutdown();page.close()
