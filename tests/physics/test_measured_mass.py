import json
from pathlib import Path
import numpy as np
import torch

from deployment.planner import controller_force, controller_acceleration
from simulator.point_mass import ForceControlledPointCable

ROOT = Path(__file__).resolve().parents[2]


def test_measured_mass_is_counted_once_in_hover_and_controller_export():
    config = json.loads((ROOT/'config/model.json').read_text())
    model = ForceControlledPointCable.from_mapping(config)
    assert abs(model.point_mass_kg - .145) < 1e-12
    assert abs(model.system_mass_kg - .162) < 1e-12
    assembly = config['cable']['bare_cable_mass_kg'] + sum(config['cable']['moving_marker_masses_kg'])
    assert abs(assembly - .017) < 1e-12
    total = model.hover_force_world_n(dtype=torch.float64, device='cpu').numpy()
    residual = controller_force(total, .145, 9.80665)
    np.testing.assert_allclose(residual, [0, 0, .017*9.80665], atol=1e-12)
    acceleration = controller_acceleration(total, .145, 9.80665)
    np.testing.assert_allclose(.145*(acceleration+[0, 0, 9.80665]), total, atol=1e-12)
    # Alternative total-weight controller convention: zero feedforward supports
    # the hanging assembly, while arbitrary strike thrust still reconstructs exactly.
    np.testing.assert_allclose(controller_force(total, .162, 9.80665), [0,0,0], atol=1e-12)
    strike = np.array([1.2, -.3, .9])
    np.testing.assert_allclose(.162*(controller_acceleration(strike, .162, 9.80665)+[0,0,9.80665]), strike, atol=1e-12)
