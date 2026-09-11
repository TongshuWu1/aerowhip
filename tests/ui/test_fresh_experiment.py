import os
from pathlib import Path
import pytest
from experimental_data.io import atomic_json,sha256_file
from simulator.workflow import read_json,import_take


def test_fresh_bootstrap_cannot_fall_back_to_historical_batch(tmp_path,monkeypatch):
    import experimental_data.pva_bootstrap as bootstrap
    monkeypatch.setattr(bootstrap,'ROOT',tmp_path)
    atomic_json(tmp_path/'config/experiment.json',dict(schema='unseen_system_experiment_v1',preliminary_batch=None))
    with pytest.raises(ValueError,match='no historical batch'):
        bootstrap.prepare_job(tmp_path/'runs/new-fit')
    assert not (tmp_path/'runs').exists()


def test_preliminary_import_binds_raw_bytes_to_new_hardware_identity(tmp_path):
    from experimental_data import io
    source=tmp_path/'external/take_001';source.mkdir(parents=True)
    a=source/'motive.csv';b=source/'logger.csv'
    a.write_text(','.join(value for key in sorted(io.MOTIVE_METADATA_FIELDS) for value in (key,'value'))+'\noriginal tracking')
    b.write_text(','.join(sorted(io.LOGGER_REQUIRED_FIELDS))+'\noriginal commands')
    root=tmp_path/'workspace'
    atomic_json(root/'data/dataset_manifest.json',dict(takes={}))
    atomic_json(root/'config/experiment.json',dict(schema='unseen_system_experiment_v1',id='new-system'))
    atomic_json(root/'config/current_vehicle.json',dict(drone_mass_kg=.145,cable_assembly_mass_kg=.017,total_mass_kg=.162))
    assert import_take(root,source)=='take_001'
    folder=root/'data/raw_takes/take_001';meta=read_json(folder/'experiment.json')
    assert meta['experiment_id']=='new-system' and meta['hardware']['total_mass_kg']==.162
    assert meta['sources']=={p.name:sha256_file(p) for p in (a,b)}
    assert (folder/'motive.csv').read_bytes()==a.read_bytes()
    assert not meta['reviewed_for_fitting']
    with pytest.raises(ValueError,match='already exists'):import_take(root,source)


def test_fresh_recording_inbox_and_fit_page_do_not_display_old_batch(tmp_path):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from simulator.gui.preliminary_recordings_page import PreliminaryRecordingsPage
    from simulator.gui.pva_model_page import PVAModelPage
    app=QApplication.instance() or QApplication([])
    atomic_json(tmp_path/'config/experiment.json',dict(schema='unseen_system_experiment_v1',preliminary_batch=None))
    atomic_json(tmp_path/'config/current_vehicle.json',dict(drone_mass_kg=.145,cable_assembly_mass_kg=.017,total_mass_kg=.162))
    records=PreliminaryRecordingsPage(tmp_path);models=PVAModelPage(tmp_path)
    assert records.table.rowCount()==models.models.count()==models.jobs.count()==0
    assert not models.fit_start.isEnabled()
    models.fit()
    assert not models.worker.running and 'No historical batch' in models.fit_status.text()
    models.shutdown();models.close();records.close();app.processEvents()


def test_pva_log_import_accepts_explicit_pair_without_changing_legacy_contract(tmp_path):
    from experimental_data import io
    source=tmp_path/'batch';source.mkdir()
    motive=source/'veritfig8_001.csv';controller=source/'experiment_vertifig8_001.csv'
    motive.write_text(','.join(value for key in sorted(io.MOTIVE_METADATA_FIELDS) for value in (key,'value'))+'\n')
    fields=['time_s','x','y','z','cmd_age','cmd_valid',*('cmd_'+n for n in ('x','y','z','vx','vy','vz','ax','ay','az','yaw','yaw_rate'))]
    controller.write_text(','.join(fields)+'\n')
    with pytest.raises(ValueError):io.resolve_raw_pair(source)
    assert io.resolve_raw_pair(source,allow_pva=True)==(controller,motive)
    (source/'another_experiment.csv').write_bytes(controller.read_bytes())
    with pytest.raises(ValueError):io.resolve_raw_pair(source,allow_pva=True)
    selected=[controller.name,motive.name]
    assert io.resolve_raw_pair(source,allow_pva=True,filenames=selected)==(controller,motive)
    with pytest.raises(ValueError,match='directly inside'):
        io.resolve_raw_pair(source,allow_pva=True,filenames=['../outside.csv'])
    root=tmp_path/'workspace'
    atomic_json(root/'config/experiment.json',dict(schema='unseen_system_experiment_v1',id='new'))
    atomic_json(root/'config/current_vehicle.json',dict(drone_mass_kg=.145,cable_assembly_mass_kg=.017))
    atomic_json(root/'data/dataset_manifest.json',dict(takes={}))
    name=import_take(root,source,take_name='preliminary1_vertical',filenames=selected)
    meta=read_json(root/'data/raw_takes'/name/'experiment.json')
    assert meta['files']==dict(controller=controller.name,optitrack=motive.name)
    assert set(meta['sources'])==set(selected)


def test_reviewed_preliminary_job_is_monitored_without_historical_restart(tmp_path):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from simulator.gui.pva_model_page import PVAModelPage
    app=QApplication.instance() or QApplication([])
    job=tmp_path/'runs/adaptation/fresh-M0'
    atomic_json(tmp_path/'config/experiment.json',dict(schema='unseen_system_experiment_v1',preliminary_batch='new',fit_job='runs/adaptation/fresh-M0'))
    atomic_json(job/'protocol.json',dict(schema='preliminary_pva_bootstrap_v1'))
    atomic_json(job/'status.json',dict(status='running'))
    atomic_json(job/'progress.json',dict(stage='drone residual',update=35,ceiling=400))
    page=PVAModelPage(tmp_path)
    assert page.jobs.count()==1 and page.fit_progress.value()==35
    assert not page.fit_start.isEnabled() and page.fit_stop.isEnabled()
    page.fit();assert not page.worker.running
    atomic_json(job/'status.json',dict(status='completed'));page.poll()
    assert not page.fit_start.isEnabled() and not page.fit_stop.isEnabled()
    page.fit();assert not page.worker.running and 'new job' in page.fit_status.text()
    page.shutdown();page.close();app.processEvents()
