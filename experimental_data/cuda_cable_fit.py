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

    def verify_eager(self,data,path,*,atol=1e-7,rtol=1e-4):
        """Compare two recurrent graph blocks against identical eager equations.

        This diagnoses capture/VJP execution separately from the whole-window
        finite-difference check. It does not test different physics equations.
        """
        from .io import atomic_json
        size=max(self.blocks);block=self.blocks[size]
        if data['roots'].shape[1]-1<2*size:
            raise ValueError('Eager comparison requires two complete graph blocks')
        parameters=block.parameters
        def run(fn):
            q,v=data['q'],data['v'];outputs=[]
            for i in (1,1+size):
                qp,vp=fn(q,v,data['roots'][:,i:i+size]);q,v=qp[:,-1],vp[:,-1]
                outputs.extend((qp,vp))
            loss=sum(x.square().mean() for x in outputs)
            return outputs,torch.autograd.grad(loss,parameters)
        with torch.enable_grad():
            expected,eg=run(block.function);actual,ag=run(block)
        rows=[]
        for label,observed,wanted in [('output',actual,expected),('gradient',ag,eg)]:
            for index,(a,e) in enumerate(zip(observed,wanted)):
                rows.append(dict(kind=label,index=index,maximum_absolute_difference=float((a-e).detach().abs().max()),
                    passed=bool(torch.allclose(a,e,atol=atol,rtol=rtol))))
        result=dict(passed=all(r['passed'] for r in rows),steps=2*size,atol=atol,rtol=rtol,checks=rows)
        atomic_json(path,result)
        if not result['passed']:raise ValueError('Captured/eager cable forward or gradient mismatch')
        return result

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
