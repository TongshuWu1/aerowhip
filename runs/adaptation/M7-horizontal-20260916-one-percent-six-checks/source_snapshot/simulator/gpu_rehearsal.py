"""Independent GPU contexts for live physics and offline strike planning."""
import copy
import torch

from simulator.cable import DderState
from simulator.cuda_graph_physics import CudaGraphPhysics
from simulator.cable.cuda_rehearsal_solvers import RehearsalSolvers


class GpuRehearsalPhysics:
    def __init__(self, model, initial, dt_s, solvers, *, priority=0):
        if not torch.cuda.is_available():
            raise RuntimeError('Testing requires a CUDA GPU and a CUDA-enabled PyTorch installation.')
        self.stream = torch.cuda.Stream(priority=priority)
        with torch.cuda.stream(self.stream):
            state = DderState(initial.positions_m.cuda(), initial.velocities_m_s.cuda())
            self.graph = CudaGraphPhysics(model, state, dt_s, linear_solvers=solvers)
        self.stream.synchronize()

    @torch.no_grad()
    def __call__(self, positions, velocities, force):
        with torch.cuda.stream(self.stream):
            result = self.graph(DderState(positions,velocities), force)
            return result.positions_m.cpu(), result.velocities_m_s.cpu()


def prepare_gpu_rehearsal(flight):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Install CUDA-enabled PyTorch for GPU Testing.')
    if flight.state.positions_m.dtype != torch.float64 or flight.state.positions_m.shape != (1,12,3):
        raise ValueError('GPU Testing currently supports the saved float64, 12-node cable model.')
    # Capture both graphs before either starts executing. Buffers and streams
    # must never be shared between the moving plant and its private prediction.
    # Warm the allocator. RehearsalSolvers explicitly binds the CUDA device
    # on this worker thread before NVRTC's PTX is loaded by the driver API.
    torch.empty(1, device='cuda')
    solvers = RehearsalSolvers()
    live = GpuRehearsalPhysics(flight.model, flight.state, flight.dt_s, solvers, priority=-1)
    planner = GpuRehearsalPhysics(flight.model, flight.state, flight.dt_s, solvers)
    actor = copy.deepcopy(flight.policy.__self__.policy).to('cuda').eval()

    @torch.no_grad()
    def policy(observation):
        with torch.cuda.stream(planner.stream):
            return actor.deterministic(observation.to('cuda')).cpu()

    # Warm actor allocations before the 100 Hz clock starts.
    policy(torch.zeros(1,79))
    return live, planner, policy
