"""All reviewed cable coefficients, then an explicit zero-extension residual."""
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from .whip_full_optim import trust_fit,progress,train_residual
from .cuda_cable_fit import CudaCableFit
from .io import atomic_json
from simulator.cable import DderState
from simulator.cable.residual import MotionResidual
from simulator.research_physics import ResearchPhysics


def residual_vector(q,data,marker_ids,scale):
    """Candidate x take x time x observed marker x xyz, fixed missingness."""
    delta=(q[:,:,:,marker_ids]-data['truth'][None])/scale
    factor=torch.sqrt(2/(torch.sqrt(1+delta.square().sum(-1))+1))
    mask=data['mask'].to(q.dtype);allw=.5*mask/mask.sum((1,2))[:,None,None]
    tip=mask[:,:,-1];tipw=torch.zeros_like(mask);tipw[:,:,-1]=.5*tip/tip.sum(1)[:,None]
    weights=(allw+tipw)*data['weights'][:,None,None]
    return delta*factor[...,None]*weights.sqrt()[None,:,:,:,None]


class FullCableForward:
    def __init__(self,engine,data,contract,count=7):
        self.data=data;self.engine=engine;self.contract=contract;self.count=count;self.b=len(data['q'])
        q=data['q'].repeat_interleave(count,0);v=data['v'].repeat_interleave(count,0)
        self.initial=DderState(q,v);self.roots=data['roots'].repeat_interleave(count,0)
        original=engine.physics.runtime_constants(q)
        self.values=torch.stack([q.new_full((len(q),),float(x)) for x in (
            original.bending_stiffness_n_m2.flatten()[0],original.bending_damping_n_m2_s.flatten()[0],original.external_drag_s_inv.flatten()[0])])
        self.constants=replace(original,bending_stiffness_n_m2=self.values[0],
            bending_damping_n_m2_s=self.values[1],external_drag_s_inv=self.values[2])
        self.step=ResearchPhysics(engine.physics,self.initial,engine.dt_s,constants=self.constants,graph=True,fast_solve=True,fast_geometry=True)
        self.ids=list(engine.cable.marker_node_indices[1:]);self.cached=None

    @torch.no_grad()
    def predict(self,values):
        values=self.values.new_tensor(values);self.values.copy_(values.T.repeat(1,self.b))
        state=self.initial;states=[state.positions_m]
        for root in self.roots[:,1:].unbind(1):state=self.step(state,root);states.append(state.positions_m)
        q=torch.stack(states,1).reshape(self.b,self.count,-1,self.data['q'].shape[1],3).permute(1,0,2,3,4)
        if not bool(torch.isfinite(q).all()):raise FloatingPointError('Nonfinite cable physical candidate')
        return q

    def evaluate(self,x):
        if self.cached is None or not np.array_equal(x,self.cached):
            eps=self.contract['cable_log_difference_step'];values=np.r_[x[None],x[None]+eps*np.eye(3),x[None]-eps*np.eye(3)]
            q=self.predict(np.exp(values));r=residual_vector(q,self.data,self.ids,self.contract['cable_scale_m']).flatten(1)
            prior=r.new_tensor(np.sqrt(self.contract['nominal_prior'])*(values-self.log_prior))
            r=torch.cat([r,prior],1).cpu().numpy()
            self.value=r[0],((r[1:4]-r[4:7])/(2*eps)).T;self.cached=x.copy()
        return self.value


def fit_physics(engine,data,model,contract,folder,job):
    progress(job,'cable_physics',windows=len(data['q']),parameters=['EI','Cb','external damping'])
    x0=np.log([model['cable'][k] for k in ('EI_n_m2','Cb_n_m2_s','external_drag_s_inv')])
    batch=FullCableForward(engine,data,contract);batch.log_prior=x0.copy()
    # A second perturbation size checks a complete temporal Jacobian direction.
    value,jac=batch.evaluate(x0);eps=contract['cable_log_difference_step']/2
    plus=batch.evaluate(x0+eps*np.array([0.,1.,0.]))[0];minus=batch.evaluate(x0-eps*np.array([0.,1.,0.]))[0]
    fd=(plus-minus)/(2*eps);relative=float(np.linalg.norm(fd-jac[:,1])/max(np.linalg.norm(fd),1e-12))
    if relative>.02:raise ValueError(f'Cable physical sensitivity check failed: {relative}')
    atomic_json(Path(job)/'cable_physical_gradient_check.json',dict(relative_error=relative,parameter='log Cb',full_window=True))
    bounds=np.log(contract['cable_bounds']);x,result=trust_fit(batch.evaluate,x0,(bounds[0],bounds[1]),folder,job,'cable_physics',contract['nominal_stopping'])
    return np.exp(x),result


def make_residual(engine,contract):
    spec=contract['cable_residual'];torch.manual_seed(contract['seed'])
    existing=engine.physics.motion_residual
    if existing is None:
        net=MotionResidual(engine.cable.node_count,**spec).to(device='cuda',dtype=torch.float64)
    else:
        reference=next(engine.drone.residual.parameters())
        net=deepcopy(existing._network(reference) if hasattr(existing,'_network') else existing)
        inherited=net.specification();inherited.setdefault('mode','acceleration')
        if any(inherited.get(k)!=v for k,v in spec.items()):
            raise ValueError('Warm-start cable residual architecture differs from contract')
    net.requires_grad_(True)
    engine.physics.motion_residual=net
    return net


def residual_objective(engine,data,parameters,contract):
    params=data['q'].new_tensor(parameters[:2]);accelerator=CudaCableFit(engine,data,params,block_steps=3)
    net=engine.physics.motion_residual;ids=list(engine.cable.marker_node_indices[1:])
    reference=deepcopy(net).requires_grad_(False)
    def objective():
        q,v=accelerator(data,gradients=torch.is_grad_enabled())
        r=residual_vector(q[None],data,ids,contract['cable_scale_m'])
        extra=net(q.flatten(0,1),v.flatten(0,1)).reshape_as(q)
        mask=data['mask'].any(-1).to(q.dtype)
        penalty=((extra[:,:,1:].square().mean((2,3))*mask).sum(1)/mask.sum(1)*data['weights']).sum()
        old=reference(q.flatten(0,1),v.flatten(0,1)).reshape_as(q)
        change=((((extra-old)[:,:,1:].square().mean((2,3))*mask).sum(1)/mask.sum(1))*data['weights']).sum()
        return r.square().sum()+(contract['residual_magnitude']*penalty+contract['residual_change']*change)/net.acceleration_limit**2
    return objective,accelerator


def gradient_check(net,objective,path,parameter=None,*,atol=1e-6,rtol=.02):
    """Check every trainable tensor plus two seeded joint directions.

    Requires agreement at two adjacent step sizes. Near-zero derivatives use
    an absolute tolerance; stationarity alone is neither failure nor proof.
    No optimizer step or global random-number state change is made.
    """
    named=[(n,p) for n,p in net.named_parameters() if p.requires_grad]
    if parameter is not None:named=[(n,p) for n,p in named if p is parameter]
    if not named:raise ValueError('No trainable parameters to check')
    originals=[p.detach().clone() for _,p in named];directions=[];results=[]
    try:
        net.zero_grad();loss=objective()
        if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite gradient-check loss')
        loss.backward()
        for i,(name,p) in enumerate(named):
            if p.grad is None or not bool(torch.isfinite(p.grad).all()):
                raise ValueError('Missing/invalid residual gradient: '+name)
            index=int(p.grad.abs().flatten().argmax())
            direction=[torch.zeros_like(v) for v in originals]
            direction[i].reshape(-1)[index]=1
            directions.append((name,index,direction))
        generator=torch.Generator().manual_seed(20260914)
        for j in range(2):
            direction=[torch.randn(v.shape,generator=generator,dtype=torch.float64).to(v) for v in originals]
            norm=torch.sqrt(sum(v.square().sum() for v in direction))
            directions.append((f'joint_random_{j}',None,[v/norm for v in direction]))
        gradients=[p.grad.detach().clone() for _,p in named]
        for name,index,direction in directions:
            analytic=float(sum((g*d).sum() for g,d in zip(gradients,direction)))
            checks=[];passed=False
            for eps in (1e-4,5e-5,1e-5,5e-6,1e-6,5e-7,1e-7,5e-8):
                with torch.no_grad():
                    for (_,p),v,d in zip(named,originals,direction):p.copy_(v+eps*d)
                    plus=float(objective())
                    for (_,p),v,d in zip(named,originals,direction):p.copy_(v-eps*d)
                    minus=float(objective())
                    for (_,p),v in zip(named,originals):p.copy_(v)
                finite=(plus-minus)/(2*eps);error=abs(finite-analytic)
                agrees=np.isfinite(finite) and error<=atol+rtol*max(abs(finite),abs(analytic))
                stable=bool(checks) and abs(finite-checks[-1]['central_difference'])<=atol+rtol*max(abs(finite),abs(checks[-1]['central_difference']))
                checks.append(dict(epsilon=eps,central_difference=finite,absolute_error=error,
                    relative_error=error/max(abs(finite),abs(analytic),atol),agrees=bool(agrees)))
                if agrees and stable and checks[-2]['agrees']:passed=True;break
            results.append(dict(direction=name,index=index,analytic=analytic,checks=checks,
                **{k:v for k,v in checks[-1].items() if k!='agrees'},passed=passed))
    finally:
        with torch.no_grad():
            for (_,p),v in zip(named,originals):p.copy_(v)
        net.zero_grad()
    result=dict(results[-1],loss=float(loss.detach()),directions=results,
        absolute_tolerance=atol,relative_tolerance=rtol,passed=all(r['passed'] for r in results))
    atomic_json(path,result)
    if not result['passed']:
        raise ValueError('Full residual gradient mismatch: '+', '.join(r['direction'] for r in results if not r['passed']))
    return result
