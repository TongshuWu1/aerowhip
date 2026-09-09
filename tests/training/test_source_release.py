from pathlib import Path
import hashlib
import json
import zipfile
import pytest
from tools.build_source_release import build,portable_configs

ROOT=Path(__file__).resolve().parents[2]


def test_portable_configuration_retains_physics_rewards_and_embedded_prior():
    configurations,_=portable_configs(ROOT)
    for name in ('ppo','sac'):
        original=json.loads((ROOT/f'config/{name}.json').read_text(encoding='utf-8'))
        for key in ('actions','phase_control_steps','total_control_steps'):
            assert configurations[name]['bootstrap'][key]==original['bootstrap'][key]
    original=json.loads((ROOT/'config/ppo.json').read_text(encoding='utf-8'))
    assert configurations['ppo']['reward']==original['reward']
    assert configurations['ppo']['action']==original['action']
    original_model=json.loads((ROOT/'config/model.json').read_text(encoding='utf-8'))
    for key,value in original_model['cable'].items():
        if key not in ('parameter_source','previous_parameter_source'):
            assert configurations['model']['cable'][key]==value
    assert configurations['task']==json.loads((ROOT/'config/task.json').read_text(encoding='utf-8'))


def test_release_excludes_local_artifacts_and_has_verifiable_hashes(tmp_path):
    result=build(ROOT,tmp_path/'source')
    output=Path(result['directory']);manifest=json.loads((output/'RELEASE_MANIFEST.json').read_text())
    assert not (output/'.git').exists()
    assert json.loads((output/'data/dataset_manifest.json').read_text())['takes']=={}
    with zipfile.ZipFile(result['archive']) as archive:
        for name in archive.namelist():
            assert not name.startswith(('runs/','results/','archive/','.git/','data/raw_takes/'))
            assert not name.endswith(('.pt','.npz','.pdf','.csv'))
        for name,record in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==record['sha256']
    assert json.loads((output/'PUBLICATION_METADATA.json').read_text())['license'] is None
    with pytest.raises(ValueError,match='new output'):build(ROOT,output)
