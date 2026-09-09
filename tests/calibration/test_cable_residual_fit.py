from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from experimental_data.cable_residual_fit import (permitted_takes, prepare_residual_job,
    prepare_takes, acceptance, apply_candidate, training_objective,
    residual_base_model, make_residual)
from experimental_data.io import atomic_json, canonical_json_hash, sha256_file
from experimental_data.differentiable_fit import save_weights
from simulator.cable import CableConfiguration
from simulator.cable.residual import MotionResidual
from simulator.workflow import read_json

ROOT = Path(__file__).resolve().parents[2]


def test_nn_only_ignores_calibrated_drag_and_starts_at_zero():
    payload = read_json(ROOT/'config/model.json')
    settings = read_json(ROOT/'config/cable_residual.json')
    assert settings['drag_mode'] == 'nn_only'
    cable = CableConfiguration.from_mapping(payload['cable'])
    networks = []
    for previous_drag in (.3, 2.0):
        original = deepcopy(payload)
        original['cable']['external_drag_s_inv'] = previous_drag
        base = residual_base_model(original, settings)
        assert base['cable']['external_drag_s_inv'] == 0
        assert original['cable']['external_drag_s_inv'] == previous_drag
        torch.manual_seed(123)
        network = make_residual(cable, original, settings).double()
        assert not network.learn_drag and not hasattr(network, 'raw_drag')
        assert network.initial_drag_s_inv == 0
        assert sum(p.numel() for p in network.parameters()) == 4577
        q = torch.randn(2, cable.node_count, 3, dtype=torch.float64)
        v = torch.randn_like(q)
        assert torch.count_nonzero(network(q, v)) == 0
        assert torch.count_nonzero(network.drag_rates(q)) == 0
        networks.append(network)
    for name, value in networks[0].state_dict().items():
        torch.testing.assert_close(value, networks[1].state_dict()[name], rtol=0, atol=0)


@pytest.fixture
def residual_workspace(tmp_path):
    (tmp_path/'config').mkdir()
    for name in ('model', 'cable_residual'):
        shutil.copy2(ROOT/f'config/{name}.json', tmp_path/f'config/{name}.json')
    model = read_json(tmp_path/'config/model.json')
    count = 650
    positions = np.zeros((count,3)); positions[:,2] = 1.5
    offset = np.asarray(model['recorded_data']['optitrack_to_attachment_offset_body_m'])
    arc = np.cumsum(model['cable']['marker_interval_lengths_m'])
    markers = np.repeat((positions+offset)[:,None],10,axis=1)
    markers[:,:,2] -= arc
    arrays = dict(time_s=np.arange(count)*.01, uav_position_m=positions,
        uav_orientation_xyzw=np.tile([0.,0.,0.,1.],(count,1)), uav_valid=np.ones(count,bool),
        cable_marker_positions_m=markers, cable_marker_valid=np.ones((count,10),bool),
        auto_frame_valid=np.ones(count,bool))
    for name in ('fig8_001','fig8_003'):
        folder=tmp_path/'data/processed_takes'/name; folder.mkdir(parents=True)
        np.savez(folder/'take.npz', **arrays)
    manifest=dict(takes={'fig8_001':dict(role='training'), 'fig8_003':dict(role='validation'),
        'fig8vertical_002':dict(role='training')})  # Hard block even if role is wrong.
    atomic_json(tmp_path/'data/dataset_manifest.json', manifest)
    return tmp_path


def test_snapshot_excludes_protected_and_whip_data_and_keeps_physics(residual_workspace):
    root=residual_workspace
    original=(root/'config/model.json').read_bytes()
    job, command=prepare_residual_job(root, updates=2)
    assert '--job' in command
    assert set(read_json(job/'dataset_manifest.json')['takes']) == {'fig8_001','fig8_003'}
    assert not (job/'processed_takes/fig8vertical_002').exists()
    assert read_json(job/'settings.json')['updates'] == 2
    assert (root/'config/model.json').read_bytes() == original
    manifest=read_json(root/'data/dataset_manifest.json')
    manifest['takes']['whip1_001']=dict(role='training')
    with pytest.raises(ValueError,match='not an authorized preliminary'):
        permitted_takes(manifest)


def test_preparation_uses_past_only_state_and_never_commands(residual_workspace):
    job,_=prepare_residual_job(residual_workspace)
    payload=read_json(job/'model.json')
    cable,model,takes,_=prepare_takes(job,payload,.05,1,'cpu',training_only=True)
    assert len(takes)==1 and takes[0].role=='training'
    first=takes[0]
    path=job/'processed_takes/fig8_001/take.npz'
    with np.load(path) as data:
        arrays={k:data[k] for k in data.files}
    start=first.starts[0]
    arrays['cable_marker_positions_m'][start+1:] += .001
    arrays['command_position_m']=np.full((650,3),999.)
    np.savez(path,**arrays)
    _,_,changed,_=prepare_takes(job,payload,.05,1,'cpu',training_only=True)
    torch.testing.assert_close(first.initial_positions_m,changed[0].initial_positions_m)
    torch.testing.assert_close(first.initial_velocities_m_s,changed[0].initial_velocities_m_s)
    assert not torch.equal(first.measured_marker_positions_m,changed[0].measured_marker_positions_m)


def test_validation_gate_requires_both_horizons_and_limits_take_regression():
    baseline=dict(equal_take_marker_rmse_m=.1,equal_take_tip_rmse_m=.2,
                  per_take={'a':dict(marker_rmse_m=.1)})
    better=dict(equal_take_marker_rmse_m=.09,equal_take_tip_rmse_m=.19,
                per_take={'a':dict(marker_rmse_m=.099)})
    evaluation={h:dict(physics=dict(validation=deepcopy(baseline)),residual=dict(validation=deepcopy(better)))
                for h in ('2.0','5.0')}
    assert acceptance(evaluation)[0]
    evaluation['5.0']['residual']['validation']['per_take']['a']['marker_rmse_m']=.11
    assert not acceptance(evaluation)[0]


def test_checkpoint_selection_rejects_validation_and_matches_training_metric(residual_workspace):
    from experimental_data.constrained_identification import evaluate_together
    job,_=prepare_residual_job(residual_workspace)
    payload=read_json(job/'model.json')
    cable,model,takes,_=prepare_takes(job,payload,.01,1,'cpu')
    parameters=takes[0].initial_positions_m.new_tensor([payload['cable'][k]
        for k in ('EI_n_m2','Cb_n_m2_s','external_drag_s_inv')])
    with pytest.raises(ValueError,match='training takes only'):
        training_objective(takes,model,cable,parameters)
    expected,_,_=evaluate_together(takes,model,cable,parameters)
    actual=training_objective([t for t in takes if t.role=='training'],model,cable,parameters)
    assert actual==pytest.approx(expected['training']['objective'])


@pytest.mark.parametrize('learn_drag', [False, True])
def test_application_checks_validation_and_exact_calibration(residual_workspace, learn_drag):
    root=residual_workspace
    job,_=prepare_residual_job(root)
    payload=read_json(root/'config/model.json')
    network=MotionResidual(CableConfiguration.from_mapping(payload['cable']).node_count,hidden=32,acceleration_limit=.5,
        learn_drag=learn_drag, initial_drag_s_inv=payload['cable']['external_drag_s_inv']).double()
    save_weights(job/'residual_candidate.pt',network)
    candidate=deepcopy(payload)
    if learn_drag:
        candidate['cable']['external_drag_s_inv']=0
    candidate['motion_residual']=dict(enabled=True,checkpoint=str(job/'residual_candidate.pt'),
                                      sha256=sha256_file(job/'residual_candidate.pt'))
    atomic_json(job/'candidate_model.json',candidate)
    review=dict(accepted=False,model_sha256=canonical_json_hash(payload),candidate_sha256=canonical_json_hash(candidate))
    atomic_json(job/'review.json',review)
    atomic_json(job/'evaluation.json',{})
    with pytest.raises(ValueError,match='did not pass'):
        apply_candidate(root,job)
    assert read_json(root/'config/model.json')==payload
    review['accepted']=True; atomic_json(job/'review.json',review)
    changed=deepcopy(payload);changed['cable']['external_drag_s_inv']+=.01
    atomic_json(root/'config/model.json',changed)
    with pytest.raises(ValueError,match='calibration changed'):
        apply_candidate(root,job)
    atomic_json(root/'config/model.json',payload)
    version=apply_candidate(root,job)
    actual=read_json(root/'config/model.json')
    assert actual['cable']==candidate['cable'] and actual['point_mass']==payload['point_mass']
    assert actual['motion_residual']['enabled']
    assert (root/actual['motion_residual']['checkpoint']).is_file()
    assert (root/'data/baselines'/version/'model.json').is_file()
