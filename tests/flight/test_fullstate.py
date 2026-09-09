import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deployment.fullstate import sample_fullstate, simulate_trajectory, write_fullstate
from simulator.cable import DderState


def test_analytic_trajectory_derivatives_and_cutoff():
    t = np.arange(82)*.01
    p = np.column_stack((t**3, 2*t**2, 1.5+t))
    v = np.column_stack((3*t**2, 4*t, np.ones_like(t)))
    ts, ps, vs, acc = sample_fullstate(t, p, v)
    assert ts[-1] == pytest.approx(.81)
    np.testing.assert_allclose(np.diff(ts[:-1]), 1/30, atol=1e-14)
    np.testing.assert_allclose(ps, np.column_stack((ts**3, 2*ts**2, 1.5+ts)), atol=1e-12)
    np.testing.assert_allclose(vs, np.column_stack((3*ts**2, 4*ts, np.ones_like(ts))), atol=1e-11)
    np.testing.assert_allclose(acc, np.column_stack((6*ts, 4*np.ones_like(ts), np.zeros_like(ts))), atol=1e-9)


def test_rollout_includes_initial_and_terminal_and_preserves_plan(tmp_path):
    state = DderState(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3))
    plan = SimpleNamespace(initial_state=state, forces_world_n=torch.ones(3, 3), dt_s=.01)
    def physics(q, v, force):
        return q + v*.01 + force[:, None]*.00005, v + force[:, None]*.01
    trajectory = simulate_trajectory(plan, physics)
    assert len(trajectory['time_s']) == 4
    assert torch.count_nonzero(state.positions_m) == 0
    (tmp_path/'plan.npz').write_bytes(b'source')
    meta = write_fullstate(trajectory, tmp_path, dict(cutoff_s=.03))
    assert meta['flight_ready'] is False
    assert meta['sample_rate_hz'] == 30
    data = np.loadtxt(tmp_path/'fullstate_30hz.csv', delimiter=',', skiprows=1)
    np.testing.assert_allclose(data[:, 7:10], 1, atol=1e-5)
    assert data[-1, 0] == pytest.approx(.03)
    assert json.loads((tmp_path/'fullstate.json').read_text())['source_plan_sha256']
    with pytest.raises(ValueError):
        write_fullstate(trajectory, tmp_path, {})
    with pytest.raises(InterruptedError):
        simulate_trajectory(plan, physics, lambda: True)


def test_invalid_trajectory_rejected():
    with pytest.raises(ValueError):
        sample_fullstate([0, 0], np.zeros((2, 3)), np.zeros((2, 3)))


def test_fullstate_planning_does_not_add_recovery_simulation(tmp_path, monkeypatch):
    from simulator.gui import rehearsal_worker
    from deployment import fullstate_recovery
    sentinel = object()
    monkeypatch.setattr(rehearsal_worker,'compile_strike_plan',lambda *a,**kw:sentinel)
    def forbidden(*args,**kwargs):
        raise AssertionError('Full-state export must not add a second recovery simulation')
    monkeypatch.setattr(fullstate_recovery,'prepare_complete_trajectory',forbidden)
    worker = rehearsal_worker.RehearsalWorker([],tmp_path/'policy.pt',tmp_path,fullstate_mode=True)
    plan,trajectory = worker.prepare_plan()
    assert plan is sentinel and trajectory is None
