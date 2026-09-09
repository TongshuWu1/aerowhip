"""Replay an unchanged fixed-shape DDER transition with one CUDA graph launch."""
import torch
import weakref
from .cable import DderState, FREE_ENDPOINTS


class CudaGraphPhysics:
    def __init__(self, model, state, dt, constants=None, *, linear_solvers=None):
        if not state.positions_m.is_cuda:
            raise ValueError('CUDA graph physics requires CUDA tensors.')
        if state.endpoint_orientations is not None or state.endpoint_twist_rad is not None:
            raise ValueError('Point-force graph physics requires free, position-only nodes.')
        self.dder=model.dder
        self.linear_solvers=linear_solvers
        self.controller=model.controller
        self.q=state.positions_m.detach().clone()
        self.v=state.velocities_m_s.detach().clone()
        self.force=self.q.new_zeros(self.q.shape[0],3)
        self.dt=self.q.new_full((self.q.shape[0],),float(dt))
        self.constants=constants or model.dder.runtime_constants(self.q)
        stream=torch.cuda.Stream(device=self.q.device)
        stream.wait_stream(torch.cuda.current_stream(self.q.device))
        with torch.cuda.stream(stream):
            for _ in range(2):self._transition()
        torch.cuda.current_stream(self.q.device).wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output=self._transition()

    def _transition(self):
        return self.dder.step_runtime(
            DderState(self.q,self.v),self.q[:,:0],self.dt,self.constants,
            external_force_world_n=self.controller.node_forces(self.force,validate=False),
            iterative_damping=True,damping_backend='pcg32_experimental',
            linear_solvers=self.linear_solvers,
            analytic_bending=self.linear_solvers is not None,
            pinned_endpoints=FREE_ENDPOINTS,create_graph=False)

    @torch.no_grad()
    def __call__(self,state,force):
        if state.positions_m.shape!=self.q.shape or state.positions_m.dtype!=self.q.dtype:
            raise ValueError('CUDA graph state shape or precision changed.')
        self.q.copy_(state.positions_m);self.v.copy_(state.velocities_m_s)
        self.force.copy_(force);self.graph.replay()
        # Callers retain previous states and recordings across later replays.
        return DderState(self.output.positions_m.clone(),self.output.velocities_m_s.clone())


def install_graph_runtime(model):
    """Lazy capture avoids allocating graphs for scoring-only environments."""
    eager=weakref.WeakMethod(model.step_runtime)
    model_ref=weakref.ref(model)
    captured=None
    captured_dt=None
    @torch.no_grad()
    def step(state,force,dt):
        nonlocal captured,captured_dt
        model=model_ref()
        if not state.positions_m.is_cuda:return eager()(state,force,dt)
        if captured is None:
            captured_dt=float(dt)
            captured=CudaGraphPhysics(model,state,captured_dt)
        if float(dt)!=captured_dt:raise ValueError('CUDA graph timestep changed.')
        command=torch.as_tensor(force,dtype=state.positions_m.dtype,device=state.positions_m.device)
        if command.ndim==1:command=command[None]
        result=captured(state,command)
        return model._result(state,result,command,dt)
    model.step_runtime=step
