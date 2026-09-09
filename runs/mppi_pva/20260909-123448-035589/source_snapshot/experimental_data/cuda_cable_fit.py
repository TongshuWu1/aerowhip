"""Full temporal cable fitting with reusable, recomputed CUDA VJP blocks.

Only execution changes: same float64 solver, substeps, residual, and full
trajectory loss. No short windows, detached recurrent states or extra flights.
Nominal parameters are fixed in this NN-only accelerator.
"""
from dataclasses import replace
import torch
from simulator.cable import DderState
from simulator.cuda_autograd import CudaAutogradBlock
from simulator.research_physics import research_physics_step


class CudaCableFit:
    def __init__(self, engine, data, parameters, block_steps=3,fast_solve=True,fast_geometry=True):
        if block_steps < 1:
            raise ValueError('Positive graph block size required')
        if parameters.requires_grad:
            raise ValueError('This accelerator fits residual weights; nominal parameters must be fixed')
        self.block_steps=block_steps
        q=data['q'];b=len(q)
        physics=engine.physics
        constants=replace(physics.runtime_constants(q),
            bending_stiffness_n_m2=parameters[0].expand(b),
            bending_damping_n_m2_s=parameters[1].expand(b))
        dt=q.new_full((b,),engine.dt_s)
        def advance(q,v,roots):
            qs,vs=[],[]
            for root in roots.unbind(1):
                state=research_physics_step(physics,DderState(q,v),root,dt,constants,create_graph=True,fast_solve=fast_solve,fast_geometry=fast_geometry)
                q,v=state.positions_m,state.velocities_m_s
                qs.append(q);vs.append(v)
            return torch.stack(qs,1),torch.stack(vs,1)
        sizes={min(block_steps,data['roots'].shape[1]-1)}
        remainder=(data['roots'].shape[1]-1)%block_steps
        if remainder:sizes.add(remainder)
        self.blocks={size:CudaAutogradBlock(advance,
            (q,data['v'],data['roots'][:,1:size+1]),tuple(physics.motion_residual.parameters())) for size in sorted(sizes)}

    def __call__(self,data,*,gradients):
        q,v=data['q'],data['v'];qs=[q[:,None]];vs=[v[:,None]]
        with torch.enable_grad() if gradients else torch.no_grad():
            for i in range(1,data['roots'].shape[1],self.block_steps):
                roots=data['roots'][:,i:i+self.block_steps]
                qp,vp=self.blocks[roots.shape[1]](q,v,roots)
                q,v=qp[:,-1],vp[:,-1]
                qs.append(qp);vs.append(vp)
            q,v=torch.cat(qs,1),torch.cat(vs,1)
            if not bool(torch.isfinite(q).all()&torch.isfinite(v).all()):
                raise FloatingPointError('Nonfinite cable prediction')
        return q,v
