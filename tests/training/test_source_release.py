from pathlib import Path
import hashlib
import json
import zipfile
import os
import subprocess
import sys
import pytest
from tools.build_source_release import build,portable_configs

ROOT=Path(__file__).resolve().parents[2]


def test_portable_configuration_retains_physics_and_pva_without_local_selection():
    configurations,_=portable_configs(ROOT)
    for name in ('ppo','sac'):
        original=json.loads((ROOT/f'config/{name}.json').read_text(encoding='utf-8'))
        assert configurations[name]['bootstrap']=={'enabled':False}
    original=json.loads((ROOT/'config/ppo.json').read_text(encoding='utf-8'))
    assert configurations['ppo']['reward']==original['reward']
    assert configurations['ppo']['action']==original['action']
    original_model=json.loads((ROOT/'config/model.json').read_text(encoding='utf-8'))
    for key,value in original_model['cable'].items():
        if key not in ('parameter_source','previous_parameter_source'):
            assert configurations['model']['cable'][key]==value
    assert configurations['task']==json.loads((ROOT/'config/task.json').read_text(encoding='utf-8'))
    for method in ('mppi','ppo'):
        original=json.loads((ROOT/f'config/pva/{method}.json').read_text(encoding='utf-8'))
        original['device']='auto'
        original['model_path']=''
        original.pop('development_model_review',None)
        if original.get('ppo_objective'):
            original['ppo_objective']['reference_source']='config/pva/wave_reference.npz'
            original['ppo_objective']['source_mppi_run']='separately held selected MPPI run; see source settings checksum'
        assert configurations['pva/'+method]==original
    assert configurations['experiment']['fit_job'] is None
    assert configurations['evaluation/campaign']['models']==[]
    assert not configurations['research_30hz/model']['fullstate_execution']['enabled']


def test_release_excludes_local_artifacts_and_has_verifiable_hashes(tmp_path):
    result=build(ROOT,tmp_path/'source')
    output=Path(result['directory']);manifest=json.loads((output/'RELEASE_MANIFEST.json').read_text())
    assert not (output/'.git').exists()
    assert (output/'planning/mppi_trajectory.py').is_file()
    assert (output/'tools/adapt_whip.py').is_file()
    assert (output/'simulator/gui/model_evolution_page.py').is_file()
    assert not (output/'config/pva/flight_selection.json').exists()
    assert not (output/'config/pva/replay.json').exists()
    assert json.loads((output/'data/dataset_manifest.json').read_text())['takes']=={}
    with zipfile.ZipFile(result['archive']) as archive:
        for name in archive.namelist():
            assert not name.startswith(('runs/','results/','archive/','.git/','data/raw_takes/'))
            assert not name.endswith(('.pt','.npz','.pdf','.csv'))
        for name,record in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==record['sha256']
    assert json.loads((output/'PUBLICATION_METADATA.json').read_text())['license'] is None
    # Run outside the live checkout: catch omitted packages, incompatible catalog
    # schemas, missing assets and hidden dependencies on selected local models.
    env={**os.environ,'QT_QPA_PLATFORM':'offscreen'}
    env.pop('PYTHONPATH',None)
    script='''from pathlib import Path
from PySide6.QtWidgets import QApplication
from simulator.gui.pva_main_window import PVAResearchWindow
from experimental_data.model_evaluation import load_catalog
import planning.mppi_trajectory as planner
root=Path.cwd();assert Path(planner.__file__).is_relative_to(root)
app=QApplication([]);window=PVAResearchWindow(root)
assert window.main_tabs.count()==6
assert load_catalog(root)['models']==[]
assert not window.mppi_page.run.isEnabled()
assert not window.ppo_page.run.isEnabled()
assert not window.model_page.worker.running
window.close();app.processEvents()
'''
    check=subprocess.run([sys.executable,'-c',script],cwd=output,env=env,capture_output=True,text=True,timeout=60)
    assert check.returncode==0,check.stdout+check.stderr
    with pytest.raises(ValueError,match='new output'):build(ROOT,output)
