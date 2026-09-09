from concurrent.futures import ThreadPoolExecutor
import ctypes
import ctypes.util
import json
import os
from pathlib import Path

import pytest
import torch

from simulator.cable import DderState
from simulator.cable.cuda_rehearsal_solvers import RehearsalSolvers
from simulator.cuda_graph_physics import CudaGraphPhysics
from simulator.gpu_rehearsal import GpuRehearsalPhysics
from simulator.point_mass import ForceControlledPointCable

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires CUDA')
ROOT = Path(__file__).resolve().parents[2]


def test_rehearsal_can_restart_on_fresh_worker_threads():
    # Each UI run uses a new worker. A cached allocation alone does not make
    # the CUDA context current on that new OS thread.
    def run_once():
        torch.empty(1, device='cuda')
        # Make the missing thread-local context deterministic, even when a
        # driver/runtime happens to bind one during the allocation above.
        loader = ctypes.WinDLL if os.name == 'nt' else ctypes.CDLL
        driver = loader('nvcuda.dll' if os.name == 'nt' else
                        (ctypes.util.find_library('cuda') or 'libcuda.so.1'))
        driver.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        driver.cuCtxSetCurrent.restype = ctypes.c_int
        assert driver.cuCtxSetCurrent(None) == 0
        solvers = RehearsalSolvers()
        model = ForceControlledPointCable.from_mapping(json.loads((ROOT/'config/model.json').read_text()))
        state = model.hanging_state(torch.tensor([0., 0., 1.5], dtype=torch.float64))
        physics = GpuRehearsalPhysics(model, state, .01, solvers)
        result = physics(state.positions_m, state.velocities_m_s,
                         torch.tensor([[0., 0., 1.8]], dtype=torch.float64))
        assert all(torch.isfinite(value).all() for value in result)
        return result

    results = []
    for _ in range(3):
        with ThreadPoolExecutor(max_workers=1) as executor:
            results.append(executor.submit(run_once).result())
    for result in results[1:]:
        for actual, expected in zip(result, results[0]):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_fused_free_cable_matches_reference_transition_and_preserves_history():
    torch.set_num_threads(1)
    model = ForceControlledPointCable.from_mapping(json.loads((ROOT/'config/model.json').read_text()))
    state = model.hanging_state(torch.tensor([0.,0.,1.5], dtype=torch.float64,device='cuda'))
    reference = CudaGraphPhysics(model,state,.01)
    fast = CudaGraphPhysics(model,state,.01,linear_solvers=RehearsalSolvers())
    # Same-input comparisons isolate equation correctness from accumulated
    # roundoff at the model's straight/bent damping branch.
    for i in range(60):
        force = state.positions_m.new_tensor([[1.5 if i<25 else -.5,.1,2.1]])
        expected = reference(state,force)
        actual = fast(state,force)
        torch.testing.assert_close(actual.positions_m,expected.positions_m,atol=1e-9,rtol=1e-9)
        torch.testing.assert_close(actual.velocities_m_s,expected.velocities_m_s,atol=1e-7,rtol=1e-7)
        if i == 0:
            saved, copied = actual.positions_m, actual.positions_m.clone()
        state = expected
    assert torch.equal(saved,copied)


def test_live_and_planner_gpu_buffers_are_independent_under_concurrency():
    torch.set_num_threads(1)
    model = ForceControlledPointCable.from_mapping(json.loads((ROOT/'config/model.json').read_text()))
    state = model.hanging_state(torch.tensor([0.,0.,1.5],dtype=torch.float64))
    torch.empty(1,device='cuda')
    solvers = RehearsalSolvers()
    live = GpuRehearsalPhysics(model,state,.01,solvers,priority=-1)
    planner = GpuRehearsalPhysics(model,state,.01,solvers)
    positive = torch.tensor([[1.,.1,2.]],dtype=torch.float64)
    negative = torch.tensor([[-1.,-.1,2.]],dtype=torch.float64)
    args = state.positions_m,state.velocities_m_s
    expected_positive, expected_negative = live(*args,positive), planner(*args,negative)
    with ThreadPoolExecutor(2) as executor:
        for _ in range(5):
            left = executor.submit(live,*args,positive)
            right = executor.submit(planner,*args,negative)
            for actual,expected in zip(left.result(),expected_positive):
                torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            for actual,expected in zip(right.result(),expected_negative):
                torch.testing.assert_close(actual,expected,atol=0,rtol=0)
