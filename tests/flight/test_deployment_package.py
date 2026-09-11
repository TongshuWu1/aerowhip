import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from deployment.planner import controller_acceleration, force_at_elapsed, validate_initial_state, verify_package
from simulator.point_mass import ForceControlledPointCable

ROOT = Path(__file__).resolve().parents[2]


def test_force_mapping_accounts_for_gravity_and_cable_mass():
    mass, cable, g = .159, .01609091, 9.80665
    np.testing.assert_allclose(controller_acceleration([0, 0, mass*g], mass, g), [0, 0, 0], atol=1e-14)
    np.testing.assert_allclose(controller_acceleration([.2, -.1, (mass+cable)*g], mass, g),
                               [.2/mass, -.1/mass, cable*g/mass], atol=1e-14)
    with pytest.raises(ValueError):
        controller_acceleration([0, 0, 1], 0, g)


def test_time_selection_holds_twenty_hz_and_never_repeats():
    forces = np.array([[1., 0., 1.]]*5 + [[2., 0., 1.]]*3)
    np.testing.assert_array_equal(force_at_elapsed(forces, .01, .049), forces[0])
    np.testing.assert_array_equal(force_at_elapsed(forces, .01, .05), forces[5])
    np.testing.assert_array_equal(force_at_elapsed(forces, .01, .079999), forces[-1])
    assert force_at_elapsed(forces, .01, .08) is None
    assert force_at_elapsed(forces, .01, 99.) is None
    with pytest.raises(ValueError):
        force_at_elapsed(forces, .01, -1.)


def test_initialization_requires_full_projected_state():
    config = json.loads((ROOT/'config/model.json').read_text())
    model = ForceControlledPointCable.from_mapping(config)
    state = model.hanging_state(torch.tensor([0.,0.,1.5], dtype=torch.float64))
    q, v = state.positions_m[0].numpy(), state.velocities_m_s[0].numpy()
    validated = validate_initial_state(config, q, v)
    assert validated.positions_m.shape == (1,12,3)
    with pytest.raises(ValueError):
        validate_initial_state(config, q[:10], v[:10])
    wrong = q.copy(); wrong[-1, 2] -= .02
    with pytest.raises(ValueError):
        validate_initial_state(config, wrong, v)


def test_manifest_rejects_mutation_and_missing_critical_file(tmp_path):
    files = {}
    for relative in ('model.json','task.json','ppo.json','checkpoints/policy.pt'):
        path = tmp_path/relative; path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'original')
        files[relative] = hashlib.sha256(b'original').hexdigest()
    manifest = {'schema':'selected_ppo_package_v1','files':files}
    path = tmp_path/'policy_manifest.json'
    path.write_text(json.dumps(manifest))
    verify_package(tmp_path)
    (tmp_path/'task.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='failed verification'):
        verify_package(tmp_path)
    del files['task.json']; path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Incomplete'):
        verify_package(tmp_path)


@pytest.mark.parametrize('learn_drag', [False, True])
def test_export_includes_applied_cable_residual(tmp_path, learn_drag):
    import shutil
    from deployment.package import export_policy, digest
    from simulator.cable import CableConfiguration
    from simulator.cable.residual import MotionResidual
    from experimental_data.differentiable_fit import save_weights
    root=tmp_path/'project'
    run=root/'runs/ppo/test'
    (run/'checkpoints').mkdir(parents=True)
    model=json.loads((ROOT/'config/model.json').read_text())
    relative=Path('data/baselines/example/motion_residual.pt')
    (root/relative).parent.mkdir(parents=True)
    network=MotionResidual(CableConfiguration.from_mapping(model['cable']).node_count,
        learn_drag=learn_drag, initial_drag_s_inv=model['cable']['external_drag_s_inv']).double()
    if learn_drag:
        model['cable']['external_drag_s_inv'] = 0
    save_weights(root/relative,network)
    model['motion_residual']=dict(enabled=True,checkpoint=relative.as_posix(),sha256=digest(root/relative))
    (run/'model.json').write_text(json.dumps(model))
    for name in ('task','ppo'):
        shutil.copy2(ROOT/f'config/{name}.json',run/f'{name}.json')
    (root/'requirements').mkdir();(root/'requirements/headless.txt').write_text('')
    (root/'deployment').mkdir();(root/'deployment/TESTING_README.md').write_text('test')
    torch.save(dict(schema='force_ppo_checkpoint_v1',episodes=0),run/'checkpoints/latest.pt')
    destination=tmp_path/'export'
    export_policy(root,run/'checkpoints/latest.pt',destination)
    assert digest(destination/relative)==digest(root/relative)
    ForceControlledPointCable.from_mapping(json.loads((destination/'policy/model.json').read_text()),root=destination)
    manifest=json.loads((destination/'TRANSFER_MANIFEST.json').read_text())
    assert relative.as_posix() in manifest['files']
