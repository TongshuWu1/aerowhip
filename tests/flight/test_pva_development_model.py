"""Development planning preserves incomplete-fit provenance and optional assets."""
import pytest
from experimental_data.io import atomic_json, sha256_file
from simulator.workflow import read_json
from learning.pva_env import defaults
from planning.pva_job import prepare


def fixture_model(root):
    weight=root/'drone_residual.pt';weight.write_bytes(b'frozen drone weights')
    drone=root/'drone_model.json'
    atomic_json(drone,{'residual':{'checkpoint':weight.name,'sha256':sha256_file(weight)}})
    model=root/'model.json'
    atomic_json(model,{'provenance':{'fit_complete':False,'flight_ready':False},
        'motion_residual':{'enabled':False},
        'cable':{'external_drag_s_inv':.4,'curvature_frame_regularization':2e-5},
        'fullstate_execution':{'enabled':True,'checkpoint':drone.name,'sha256':sha256_file(drone)}})
    return model


def test_unreviewed_candidate_stays_blocked(tmp_path):
    cfg=defaults('mppi');cfg['model_path']=str(fixture_model(tmp_path))
    with pytest.raises(ValueError,match='fit checks'):
        prepare(tmp_path,cfg,'unreviewed')
    assert not (tmp_path/'runs').exists()


def test_reviewed_physical_candidate_freezes_without_fake_nn(tmp_path):
    source=fixture_model(tmp_path);digest=sha256_file(source)
    cfg=defaults('mppi');cfg['model_path']=str(source)
    job,_=prepare(tmp_path,cfg,'M0 development',development_review='Reviewed scalar damping; inherited drone fit; simulation only')
    saved=read_json(job/'model.json')
    assert saved['provenance']=={'fit_complete':False,'flight_ready':False}
    assert saved['motion_residual']=={'enabled':False}
    assert saved['cable']['external_drag_s_inv']==.4
    assert not (job/'assets/cable_residual.pt').exists()
    assert (job/'assets/drone_residual.pt').read_bytes()==b'frozen drone weights'
    assert read_json(job/'identity.json')['development_review']
    assert sha256_file(source)==digest


def test_explicit_development_review_enables_simulation_ppo(tmp_path):
    cfg=defaults('ppo');cfg['model_path']=str(fixture_model(tmp_path))
    with pytest.raises(ValueError,match='fit checks'):
        prepare(tmp_path,cfg,'unreviewed PPO')
    job,_=prepare(tmp_path,cfg,'PPO',development_review='User-authorized same M0 PPO simulation; not flight validation')
    assert read_json(job/'model.json')['provenance']['fit_complete'] is False
    assert read_json(job/'identity.json')['evidence']=='simulation only'
