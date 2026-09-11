import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from deployment.package import export_policy, digest
from deployment.planner import create_plan, force_at_elapsed, verify_package, write_plan
from deployment.rehearsal import RehearsalFlight, assumed_hanging_state
from simulator.rollout import _build_policy_agent
from simulator.strike_plan import StrikePlan

ROOT = Path(__file__).resolve().parents[2]


def test_controller_csv_reconstructs_total_force_without_changing_plan(configs, tmp_path):
    mass, g, cable = .159, 9.81, .01609091
    initial = assumed_hanging_state(configs[0], [0, 0, 1.5], [0, 0, 0])
    forces = torch.tensor([[.2, -.1, (mass+cable)*g]]*5 + [[0., 0., .5]]*3, dtype=torch.float64)
    original = forces.clone()
    plan = StrikePlan(forces, .01, initial, torch.zeros(1,79))
    metadata = write_plan(plan, tmp_path/'converted', task=configs[1], metadata={},
                          controller_mass_kg=mass, controller_gravity_m_s2=g)
    def read(name):
        return np.loadtxt(tmp_path/'converted'/name, delimiter=',', skiprows=1)
    total, residual, acceleration = [read(name) for name in
        ('commands.csv', 'controller_force.csv', 'controller_acceleration.csv')]
    np.testing.assert_array_equal(total[:, :2], residual[:, :2])
    np.testing.assert_array_equal(total[:, :2], acceleration[:, :2])
    np.testing.assert_allclose(residual[:, 2:]+[0, 0, mass*g], total[:, 2:])
    np.testing.assert_allclose(mass*(acceleration[:, 2:]+[0, 0, g]), total[:, 2:])
    assert residual[0, 4] == pytest.approx(cable*g)
    assert residual[1, 4] < 0  # Valid gravity-subtracted command; never clamp to zero.
    assert total[-1, 1] == metadata['cutoff_s'] == .08
    assert metadata['controller_export']['mass_kg'] == mass
    torch.testing.assert_close(forces, original, rtol=0, atol=0)
    with pytest.raises(ValueError, match='both'):
        write_plan(plan, tmp_path/'invalid', task=configs[1], metadata={}, controller_mass_kg=mass)
    assert not (tmp_path/'invalid').exists()


@pytest.fixture
def configs():
    torch.set_num_threads(1)
    return [json.loads((ROOT/'config'/f'{name}.json').read_text(encoding='utf-8')) for name in ('model','task','ppo')]


@pytest.fixture
def checkpoint(tmp_path, configs):
    configs[1]['episode_duration_s'] = .1
    run = tmp_path/'saved_run'
    (run/'checkpoints').mkdir(parents=True)
    for name, config in zip(('model','task','ppo'), configs):
        (run/f'{name}.json').write_text(json.dumps(config))
    agent = _build_policy_agent(configs[2], torch.device('cpu'))
    path = run/'checkpoints/latest.pt'
    torch.save(dict(schema='force_ppo_checkpoint_v1', episodes=123,
                    policy=agent.policy.state_dict()), path)
    return path


def test_assumption_ignores_actual_cable_and_requires_ten_seconds(configs):
    flight = RehearsalFlight(*configs, policy=lambda _: torch.zeros(1,3))
    expected = flight.launch_state()
    flight.state.positions_m[:, 2:, 0] += .1
    flight.state.velocities_m_s[:, 2:, 1] += .3
    actual = flight.launch_state()
    torch.testing.assert_close(actual.positions_m, expected.positions_m)
    torch.testing.assert_close(actual.velocities_m_s, expected.velocities_m_s)
    assert flight.is_settled()
    flight.settled_s = 9.99
    assert not flight.ready
    flight.settled_s = 10.
    assert flight.ready
    flight.state.velocities_m_s[:, 0, 0] = .1
    assert not flight.is_settled()
    with pytest.raises(ValueError):
        assumed_hanging_state(configs[0], [0,0,float('nan')], [0,0,0])


def test_launch_checks_drone_drift_only_and_execution_never_calls_actor(configs, tmp_path):
    def forbidden(_):
        pytest.fail('Actor called during execution')
    flight = RehearsalFlight(*configs, policy=forbidden)
    flight.settled_s = 10.
    initial = flight.launch_state()
    forces = flight.last_command.repeat(8,1)
    forces[:5,0] = .1
    forces[5:,0] = -.2
    plan = StrikePlan(forces, .01, initial, torch.zeros(1,79))
    metadata = write_plan(plan, tmp_path/'plan', task=configs[1], metadata={})
    rows = list(csv.DictReader((tmp_path/'plan/commands.csv').open()))
    assert len(rows) == 2
    assert float(rows[0]['until_s']) == .05
    assert float(rows[-1]['until_s']) == metadata['cutoff_s'] == .08
    assert force_at_elapsed(forces.numpy(), .01, .08) is None
    # Physical cable differs, but preparation cannot see it.
    flight.state.positions_m[:,2:,0] += .005
    physical_before = flight.state.positions_m.clone()
    flight.start_strike(plan)
    torch.testing.assert_close(flight.state.positions_m, physical_before)
    for expected in forces:
        frame = flight.step()
        np.testing.assert_array_equal(frame['command'], expected.numpy())
    assert flight.phase == flight.RECOVER
    assert flight.force_sequence is None
    flight.phase = flight.HOVER
    flight.settled_s = 10.
    flight.state.positions_m[:,0,0] += 1.
    with pytest.raises(ValueError, match='changed'):
        flight.start_strike(plan)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='Requires CUDA'))])
def test_exported_package_runs_independently_with_drone_state_and_target(checkpoint, configs, tmp_path, device):
    setup = dict(initial_attachment_position_m=[0,0,1.51], target_position_m=[1.02,0,1.4])
    package = export_policy(ROOT, checkpoint, tmp_path/'portable', experiment_setup=setup)
    assert json.loads((package/'experiment_setup.json').read_text(encoding='utf-8')) == setup
    manifest = verify_package(package/'policy')
    assert manifest['episodes'] == 123
    assert digest(checkpoint) == digest(package/'policy/checkpoints/policy.pt')
    assert package.with_suffix('.zip').is_file()
    assert not (package/'simulator/gui').exists()
    initial = tmp_path/'drone.npz'
    np.savez(initial, attachment_position_m=[0.,0.,1.5],
             attachment_velocity_m_s=[.001,0.,0.], state_time_s=12.)
    target = [1.02,.01,1.39]
    output = tmp_path/'plan'
    command = [sys.executable,'-m','deployment.planner','--package','policy','--initial',str(initial),
               '--target',*map(str,target),'--output',str(output),'--device',device]
    result = subprocess.run(command, cwd=package, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout+result.stderr
    metadata = json.loads((output/'plan.json').read_text(encoding='utf-8'))
    assert metadata['target_position_m'] == target
    assert metadata['initialization'] == 'assumed_vertical_cable'
    assert metadata['state_time_s'] == 12.
    assert metadata['cutoff_s'] <= .1
    assert metadata['policy_dt_s'] == .05
    verify_package(package/'policy')
    local = tmp_path/'local_plan'
    create_plan(package/'policy', initial, local, target_position_m=target, device=device)
    with np.load(output/'plan.npz') as exported, np.load(local/'plan.npz') as current:
        np.testing.assert_array_equal(exported['forces_world_n'], current['forces_world_n'])
    assert json.loads((package/'policy/task.json').read_text(encoding='utf-8'))['target_position_m'] == configs[1]['target_position_m']
