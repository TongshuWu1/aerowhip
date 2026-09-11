import json
from pathlib import Path
from PySide6.QtWidgets import QApplication
from experimental_data.model_evaluation import CATALOG,save_catalog
from experimental_data.flight_library import flight_library
from simulator.gui.adaptation_check_page import AdaptationCheckPage


def fixture(root, batch_folder='data/flight_batches'):
    models=[dict(id=n,parent='M0' if n=='M1' else None,status='Registered') for n in ('M0','M1')]
    flights=[]
    for n in ('M0','M1'):
        batch=root/batch_folder/n
        (batch/'flight_take').mkdir(parents=True);(batch/'simulation_csv').mkdir()
        (batch/'simulation_csv/fullstate_30hz.csv').write_text('identical command')
        for take in ('whip_001','whip_002'):
            for prefix in ('','experiment_'):(batch/'flight_take'/f'{prefix}{take}.csv').write_text('fixture')
        (batch/'protocol.json').write_text(json.dumps(dict(command_sha256='same-command',forecast_sha256=n+'-ghost',
            planned_roles={'whip_001':'adaptation','whip_002':'validation'})))
        flights.append(dict(model=n,command_sha256='same-command',forecast_sha256=n+'-ghost'))
    save_catalog(root,dict(schema=CATALOG,models=models,flights=flights))


def test_generation_filter_keeps_identical_commands_different_forecasts_separate(tmp_path):
    fixture(tmp_path)
    lib=flight_library(tmp_path)
    assert [s['generation'] for s in lib['sessions']]==['M0','M1']
    # A renamed folder/bare generation must not make unregistered data M1.
    p=tmp_path/'data/flight_batches/M1/protocol.json'
    p.write_text(json.dumps(dict(command_sha256='same-command',forecast_sha256='unknown',parent_generation=1)))
    assert flight_library(tmp_path)['sessions'][1]['generation']=='unassigned'


def test_select_model_clears_old_replay_and_future_is_empty(tmp_path):
    fixture(tmp_path);app=QApplication.instance() or QApplication([])
    page=AdaptationCheckPage(tmp_path);selector=page.library_selector
    assert selector.generations.currentData()=='M0'
    assert page.takes.currentData()=='whip_001' and 'Adaptation' in page.takes.currentText()
    assert page.batches.count()==1
    page.data={'sentinel':'old M0 ghost'}
    selector.generations.setCurrentIndex(selector.generations.findData('M1'))
    assert page.data is None and not page.play.isEnabled()
    assert Path(page.batches.currentData()).name=='M1' and page.takes.count()==2
    selector.generations.setCurrentIndex(selector.generations.findData('M2'))
    assert page.batches.count()==0 and page.takes.count()==0 and not page.load.isEnabled()
    assert 'has not been created' in page.status.text()
    assert page.select_flight(tmp_path/'data/flight_batches/M0','whip_002')
    assert selector.generations.currentData()=='M0' and page.takes.currentData()=='whip_002'
    assert 'Validation' in page.takes.currentText()
    assert not (tmp_path/'runs').exists()
    page.shutdown();page.close()


def test_future_generation_inbox_groups_before_flight_report_registration(tmp_path):
    from experimental_data.model_evaluation import model_identity
    from experimental_data.io import sha256_file
    fixture(tmp_path)
    batch=tmp_path/'data/flight_batches/M1'
    rehearsal=tmp_path/'saved-M1';rehearsal.mkdir()
    (rehearsal/'model.json').write_text(json.dumps({'cable':{'external_drag_s_inv':.7}}))
    (rehearsal/'rehearsal.npz').write_bytes(b'new frozen prediction')
    cat=json.loads((tmp_path/'config/evaluation/campaign.json').read_text())
    cat['flights']=[f for f in cat['flights'] if f['model']=='M0']
    cat['models'][1]['signature']=model_identity(rehearsal/'model.json')[0]
    save_catalog(tmp_path,cat)
    (batch/'protocol.json').write_text(json.dumps(dict(rehearsal=str(rehearsal),
        command_sha256=sha256_file(batch/'simulation_csv/fullstate_30hz.csv'),
        forecast_sha256=sha256_file(rehearsal/'rehearsal.npz'))))
    assert flight_library(tmp_path)['sessions'][1]['generation']=='M1'
    (rehearsal/'model.json').write_text('{}')
    assert flight_library(tmp_path)['sessions'][1]['generation']=='unassigned'


def test_legacy_batch_folders_remain_discoverable(tmp_path):
    fixture(tmp_path, 'rehearsal_csv_and_result_in_real_flight')
    assert [s['generation'] for s in flight_library(tmp_path)['sessions']] == ['M0', 'M1']
