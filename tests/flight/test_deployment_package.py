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
