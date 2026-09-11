"""Synthetic operator flows; never starts a planner, fit, replay or flight."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PySide6.QtCore import QProcess
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from deployment.lab_gui import LabWindow


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def fail_on_qt_callback_exception(monkeypatch):
    errors = []
    monkeypatch.setattr(sys, 'excepthook', lambda kind, error, tb: errors.append(error))
    yield
    assert not errors, f'Qt callback raised: {errors}'


def sample_overview(root):
    model = root / 'workspace/baseline/model.json'
    model.parent.mkdir(parents=True)
    model.write_text('{}', encoding='utf-8')
    rehearsal = root / 'workspace/baseline/rehearsal'
    rehearsal.mkdir()
    (rehearsal / 'rehearsal.json').write_text('{}', encoding='utf-8')
    slots = []
    for stage in ('M0', 'M1'):
        for i in range(1, 6):
            slots.append(dict(stage=stage, take=f'whip_{i:03d}', generation=stage,
                              role='adaptation' if i in (1, 2, 4) else 'validation', status='pending', review={}))
    for i in range(1, 6):
        for generation in (('M0', 'M2') if i % 2 else ('M2', 'M0')):
            slots.append(dict(stage='final', take=f'pair_{i:02d}_{generation}', generation=generation,
                              role='final', status='pending', review={}))
    return dict(studies=['lab-test'], study=dict(name='lab-test'), baseline={'m0_signature': 'synthetic'},
                models={'M0': dict(model=str(model), rehearsal=str(rehearsal))}, slots=slots)


def factory(snapshot):
    class ReadOnlyWorkspace:
        def __init__(self, root, study=None):
            self.root, self.study = root, study

        def overview(self):
            return snapshot
    return ReadOnlyWorkspace


def test_empty_startup_is_read_only_and_gates_actions(tmp_path, app):
    window = LabWindow(tmp_path, factory(dict(studies=[], study=None, baseline=None, slots=[], models={})))
    assert not list(tmp_path.iterdir())
    assert window.pages.count() == 5
    assert not window.create_button.isEnabled()
    assert not window.plan_button.isEnabled()
    assert not window.import_button.isEnabled()
    assert not window.fit_button.isEnabled()
    assert not window.evaluate_button.isEnabled()
    assert window.process is None
    window.close()


def test_retained_m0_exports_without_replanning_and_fixed_roles_are_visible(tmp_path, app):
    state = sample_overview(tmp_path)
    before = sorted(str(p) for p in tmp_path.rglob('*'))
    window = LabWindow(tmp_path, factory(state))
    assert window.export_button.isEnabled()  # The retained complete M0 rehearsal is sufficient.
    assert window.preview_button.isEnabled()
    assert [window.slots.item(i, 2).text() for i in range(5)] == ['Training', 'Training', 'Validation', 'Training', 'Validation']
    window.capture_stage.setCurrentIndex(2)
    assert window.slots.rowCount() == 10
    assert {window.slots.item(i, 2).text() for i in range(10)} == {'Final test'}
    assert sorted(str(p) for p in tmp_path.rglob('*')) == before
    window.close()


def test_import_passes_exact_selected_slot_and_literal_file_arguments(tmp_path, app, monkeypatch):
    window = LabWindow(tmp_path, factory(sample_overview(tmp_path)))
    calls = []
    monkeypatch.setattr(window, 'run_action', lambda action, arguments=None, **kwargs: calls.append((action, arguments)))
    window.import_pair()
    assert not calls  # Timing evidence and both paths are required.
    window.slots.selectRow(3)
    tracking = str(tmp_path / 'tracking with spaces;literal.csv')
    controller = str(tmp_path / 'controller.csv')
    window.tracking_file.setText(tracking)
    window.controller_file.setText(controller)
    window.offset.setValue(-.015)
    window.clock_source.setText('Shared timestamp')
    window.import_pair()
    action, args = calls[0]
    assert action == 'import'
    assert args[args.index('--take') + 1] == 'whip_004'
    assert args[args.index('--stage') + 1] == 'M0'
    assert args[args.index('--tracking') + 1] == tracking
    assert '--clock-verified' not in args
    assert '--role' not in args  # The study owns the fixed split, not the GUI.
    window.close()


def test_review_is_explicit_and_never_infers_checked_fields(tmp_path, app, monkeypatch):
    state = sample_overview(tmp_path)
    state['slots'][0]['status'] = 'imported'
    window = LabWindow(tmp_path, factory(state))
    calls = []
    monkeypatch.setattr(window, 'run_action', lambda action, arguments=None, **kwargs: calls.append((action, arguments)))
    assert window.review_button.isEnabled()
    assert not window.review_clock.isChecked() and not window.review_hardware.isChecked()
    window.reviewer.setText('Operator')
    window.review_clock.setChecked(True)
    window.save_review()
    args = calls[0][1]
    assert '--clock-reviewed' in args
    assert '--same-hardware' not in args and '--no-intervention' not in args
    window.close()


def test_alignment_correction_preserves_slot_and_exclusion_state(tmp_path, app, monkeypatch):
    state = sample_overview(tmp_path)
    state['slots'][0].update(status='reviewed', excluded=True,
                             review=dict(reviewed_by='Operator', free_motion_end_s=None, notes='Missing markers'))
    window = LabWindow(tmp_path, factory(state))
    assert window.slots.item(0, 3).text() == 'Excluded'
    assert window.exclude.isChecked()
    calls = []
    monkeypatch.setattr(window, 'run_action', lambda action, arguments=None, **kwargs: calls.append((action, arguments)))
    window.offset.setValue(.023)
    window.clock_source.setText('Corrected common timestamp')
    window.save_alignment()
    action, args = calls[0]
    assert action == 'align'
    assert args[args.index('--take') + 1] == 'whip_001'
    assert args[args.index('--offset') + 1] == '0.023'
    assert '--tracking' not in args  # Correction never replaces the original pair.
    window.close()


def test_timing_estimate_fills_only_candidate_fields_without_approving_review(tmp_path, app, monkeypatch):
    window = LabWindow(tmp_path, factory(sample_overview(tmp_path)))
    calls = []
    monkeypatch.setattr(window, 'run_action', lambda action, arguments=None, **kwargs: calls.append((action, arguments)))
    window.tracking_file.setText('tracking file.csv')
    window.controller_file.setText('controller file.csv')
    window.clock_verified.setChecked(True)
    window.review_clock.setChecked(True)
    window.estimate_timing()
    assert calls[0][0] == 'estimate-timing'
    assert calls[0][1] == ['--tracking', 'tracking file.csv', '--controller', 'controller file.csv']
    window.apply_timing_estimate(dict(offset_s=.025, rmse_m=.012, chunk_spread_s=.003,
                                     method='Shared measured motion', limitation='Estimated, not synchronized clocks'))
    assert window.offset.value() == .025
    assert not window.clock_verified.isChecked() and not window.review_clock.isChecked()
    assert '1.20 cm' in window.timing_note.text() and '3.0 ms' in window.timing_note.text()
    assert window.slot_rows[0]['status'] == 'pending'
    assert len(calls) == 1  # No import, alignment save or review action is inferred.
    with pytest.raises(ValueError, match='finite offset'):
        window.apply_timing_estimate({'offset_s': float('nan')})
    window.close()


def test_failed_fit_requires_explicit_new_preparation_and_keeps_prior_job(tmp_path, app, monkeypatch):
    state = sample_overview(tmp_path)
    for slot in state['slots'][:5]:
        slot['status'] = 'reviewed'
    job = tmp_path / 'experiments/lab-test/fits/M1-first'
    job.mkdir(parents=True)
    status = job / 'status.json'
    status.write_text('{"status":"failed","stage":"drone_nominal"}', encoding='utf-8')
    before = status.read_bytes()
    state['study']['updates'] = {'M1': {'job': str(job), 'source_stage': 'M0'}}
    window = LabWindow(tmp_path, factory(state))
    assert window.prepare_button.isEnabled()
    assert window.prepare_button.text() == 'Prepare a new attempt'
    assert not window.fit_button.isEnabled()
    calls = []
    monkeypatch.setattr(window, 'run_action', lambda action, arguments=None, **kwargs: calls.append((action, arguments)))
    window.prepare_update()
    assert calls == [('prepare', ['--generation', 'M1', '--retry'])]
    assert status.read_bytes() == before
    state['models']['M1'] = {'model': state['models']['M0']['model']}
    window.refresh_update()
    assert not window.prepare_button.isEnabled()  # A registered generation cannot be refitted in place.
    window.close()


def test_cli_runs_asynchronously_in_repository_and_surfaces_failure(tmp_path, app):
    state = sample_overview(tmp_path)
    cli = tmp_path / 'tools/lab.py'
    cli.parent.mkdir()
    cli.write_text('import json, os, sys\nprint(json.dumps({"cwd": os.getcwd(), "argv": sys.argv[1:]}), flush=True)\nsys.exit(3)\n', encoding='utf-8')
    window = LabWindow(tmp_path, factory(state))
    window.run_action('overview', ['--literal', 'a b; c'])
    assert window.process.program() == sys.executable
    assert window.process.workingDirectory() == str(tmp_path)
    for _ in range(150):
        app.processEvents()
        if window.process.state() == QProcess.ProcessState.NotRunning:
            app.processEvents()
            break
        QTest.qWait(20)
    assert not window.busy
    assert 'failed' in window.job_status.text()
    assert window.log_toggle.isChecked()
    logged = json.loads(window.log.toPlainText().strip())
    assert logged['cwd'] == str(tmp_path)
    assert logged['argv'][-1] == 'a b; c'
    assert window.log_path.is_relative_to(tmp_path)
    window.close()


def test_async_timing_result_preserves_input_paths_and_remains_unapproved(tmp_path, app):
    state = sample_overview(tmp_path)
    cli = tmp_path / 'tools/lab.py'
    cli.parent.mkdir()
    cli.write_text('import json\nprint(json.dumps(dict(offset_s=-.031, rmse_m=.02, chunk_spread_s=.004, method="Measured motion", limitation="Estimated alignment")), flush=True)\n', encoding='utf-8')
    window = LabWindow(tmp_path, factory(state))
    window.tracking_file.setText('my tracking.csv')
    window.controller_file.setText('my controller.csv')
    window.clock_verified.setChecked(True)
    window.estimate_timing()
    for _ in range(150):
        app.processEvents()
        if not window.busy:
            app.processEvents()
            break
        QTest.qWait(20)
    assert not window.busy
    assert window.offset.value() == -.031
    assert window.tracking_file.text() == 'my tracking.csv'
    assert window.controller_file.text() == 'my controller.csv'
    assert not window.clock_verified.isChecked() and not window.review_clock.isChecked()
    assert 'ready for review' in window.job_status.text()
    assert state['slots'][0]['status'] == 'pending'
    window.close()


def test_continuous_error_results_preserve_exclusions_and_negative_improvement(tmp_path, app):
    state = sample_overview(tmp_path)
    report = tmp_path / 'experiments/lab-test/results/latest/report.json'
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps(dict(
        physical=[dict(take='pair_01_M0', generation='M0', minimum_tip_target_m=.07, sample_coverage=.9, excluded=False),
                  dict(take='pair_01_M2', generation='M2', minimum_tip_target_m=.08, sample_coverage=.8, excluded=True)],
        summary=dict(M0_mean_m=.07, M2_mean_m=.08, paired_mean_improvement_m=-.01),
        prediction_summary=[dict(model='M0', tip_rmse_m=.11, drone_rmse_m=.06, takes=10)])), encoding='utf-8')
    state['study']['results'] = str(report)
    window = LabWindow(tmp_path, factory(state))
    assert window.physical_results.item(0, 2).text() == '7.00'
    assert window.physical_results.item(1, 4).text() == 'Excluded'
    assert '-1.00 cm' in window.result_summary.text()
    assert window.prediction_results.item(0, 1).text() == '11.00'
    assert 'success threshold' in window.result_note.text()
    window.close()
