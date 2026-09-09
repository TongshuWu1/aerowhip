"""Shared direct-solve cable integrator for new planning and execution paths."""
import weakref
import torch
from .cable import DderState,FREE_ENDPOINTS,START_PINNED_FREE_END


def research_physics_step(physics,state,command,dt,constants,*,controller=None,create_graph=False):
    """Functional equations shared by captured inference and differentiable rollouts."""
    force=controller.node_forces(command,validate=False) if controller is not None else None
    boundary=state.positions_m[:,:0] if controller is not None else command[:,None]
    return physics.step_runtime(state,boundary,dt,constants,
        external_force_world_n=force,pinned_endpoints=FREE_ENDPOINTS if controller is not None else START_PINNED_FREE_END,
        create_graph=create_graph,iterative_damping=False,analytic_bending=True,dense_constraint_solve=True)


class ResearchPhysics:
    def __init__(self,physics,state,dt,*,controller=None,constants=None,graph=True):
        self.physics=physics;self.controller=controller
        self.q=state.positions_m.detach().clone();self.v=state.velocities_m_s.detach().clone()
        self.input=self.q.new_zeros(len(self.q),3) if controller is not None else self.q[:,0].clone()
        self.dt=self.q.new_full((len(self.q),),dt);self.constants=constants or physics.runtime_constants(self.q)
        self.graph=None
        if graph and self.q.is_cuda:
            # Windows' default MAGMA Cholesky path allocates during capture.
            # cuSOLVER implements the same direct system solve with graph support.
            torch.backends.cuda.preferred_linalg_library('cusolver')
            stream=torch.cuda.Stream(device=self.q.device);stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):self._step()
            torch.cuda.current_stream().wait_stream(stream)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self.output=self._step()

    def _step(self):
        return research_physics_step(self.physics,DderState(self.q,self.v),self.input,self.dt,self.constants,
            controller=self.controller,create_graph=False)

    @torch.no_grad()
    def __call__(self,state,command):
        self.q.copy_(state.positions_m);self.v.copy_(state.velocities_m_s);self.input.copy_(command)
        if self.graph is not None:self.graph.replay();out=self.output
        else:out=self._step()
        return DderState(out.positions_m.clone(),out.velocities_m_s.clone())


def install_runtime(model,*,graph=True):
    ref=weakref.ref(model);captured=None;captured_dt=None
    @torch.no_grad()
    def step(state,force,dt):
        nonlocal captured,captured_dt
        current=ref()
        if captured is None:
            captured_dt=float(dt);captured=ResearchPhysics(current.dder,state,dt,controller=current.controller,graph=graph)
        if float(dt)!=captured_dt:raise ValueError('Research physics timestep changed')
        command=torch.as_tensor(force,dtype=state.positions_m.dtype,device=state.positions_m.device)
        if command.ndim==1:command=command[None].expand(len(state.positions_m),-1)
        return current._result(state,captured(state,command),command,dt)
    model.step_runtime=step
