"""CUDA-batched analytic translation sensitivities and attitude replay."""
from dataclasses import replace,asdict
import time,gc
import numpy as np
import torch
from scipy.optimize import least_squares
from .bootstrap_drone import TranslationBatch
from .nominal_pose_fit import GAIN_NAMES,parameters
from .current_adaptation_fit import note
from .current_adaptation import save
from simulator.drone_pose_response import rotation_exp


def capture(function):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        function();function()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):outputs=function()
    torch.cuda.current_stream().wait_stream(stream)
    return graph,outputs


class CudaNominal:
    def __init__(self,trials,prior,delay):
        params=replace(prior,delay_s=delay);batch=TranslationBatch(trials,params)
        self.g=torch.tensor([getattr(params,k) for k in GAIN_NAMES],device='cuda',dtype=torch.float64)
        self.prior=self.g.clone();p=batch.p;v=batch.v
        base=p.new_tensor(np.stack([t.b0 for t in trials]));basis=p.new_tensor(np.stack([t.basis_for(delay) for t in trials]))
        indices=torch.tensor(np.stack([x[2] for x in batch.plans]),device='cuda');bi=torch.arange(len(trials),device='cuda')[:,None]
        truth=torch.stack(batch.truth);weights=p.new_tensor(np.stack([t.weights for t in trials])).sqrt()[:,:,None]/np.sqrt(len(trials))/.05
        pick=torch.tensor([0,0,1],device='cuda');dkp=p.new_zeros((3,6));dkd=dkp.clone();dff=dkp.clone()
        for j,k in enumerate([0,0,1]):dkp[j,k]=1;dkd[j,k+2]=1;dff[j,k+4]=1
        def forward():
            kp=self.g[:2][pick];kd=self.g[2:4][pick];ff=self.g[4:][pick]
            bias=base+(basis*self.g).sum(-1);pp=p;vv=v;sp=p.new_zeros((len(p),3,6));sv=sp.clone();ps=[pp];js=[sp]
            for h,c in zip(batch.dt,batch.cmd):
                ep=c[:,:3]-pp;ev=c[:,3:6]-vv;a=kp*ep+kd*ev+ff*c[:,6:9]+bias
                sa=dkp*ep[:,:,None]-kp[None,:,None]*sp+dkd*ev[:,:,None]-kd[None,:,None]*sv+dff*c[:,6:9,None]+basis
                pm=pp+.5*h*vv;vm=vv+.5*h*a;smp=sp+.5*h[:,:,None]*sv;smv=sv+.5*h[:,:,None]*sa
                ep=c[:,:3]-pm;ev=c[:,3:6]-vm;am=kp*ep+kd*ev+ff*c[:,6:9]+bias
                sam=dkp*ep[:,:,None]-kp[None,:,None]*smp+dkd*ev[:,:,None]-kd[None,:,None]*smv+dff*c[:,6:9,None]+basis
                pp=pp+h*vm;vv=vv+h*am;sp=sp+h[:,:,None]*smv;sv=sv+h[:,:,None]*sam;ps.append(pp);js.append(sp)
            pred=torch.stack(ps)[indices,bi];jac=torch.stack(js)[indices,bi]
            residual=torch.cat([((pred-truth)*weights).flatten(),np.sqrt(.03)*(self.g-self.prior)/self.prior])
            derivative=torch.cat([(jac*weights[:,:,:,None]).reshape(-1,6),torch.diag(np.sqrt(.03)/self.prior)])
            return residual,derivative,pred
        self.forward=forward  # Keep every captured input allocation alive.
        self.graph,self.outputs=capture(forward)
        self.cached=None

    def evaluate(self,g):
        if self.cached is None or not np.array_equal(self.cached,g):
            self.g.copy_(self.g.new_tensor(g));self.graph.replay()
            self.value=tuple(x.cpu().numpy().copy() for x in self.outputs[:2]);self.cached=np.array(g).copy()
        return self.value


def frame(a,yaw):
    z=torch.stack([a[...,0],a[...,1],a[...,2]+9.80665],-1);z=z/z.norm(dim=-1,keepdim=True).clamp_min(1e-9)
    heading=torch.stack([yaw.cos(),yaw.sin(),torch.zeros_like(yaw)],-1)
    y=torch.linalg.cross(z,heading,dim=-1);y=y/y.norm(dim=-1,keepdim=True).clamp_min(1e-9)
    return torch.stack([torch.linalg.cross(y,z,dim=-1),y,z],-1)


def log_rotation(r):
    v=torch.stack([r[...,2,1]-r[...,1,2],r[...,0,2]-r[...,2,0],r[...,1,0]-r[...,0,1]],-1)*.5
    sine=v.norm(dim=-1);cosine=(r.diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5
    factor=torch.where((sine<1e-6)&(cosine>0),1+sine.square()/6,torch.atan2(sine,cosine)/sine.clamp_min(1e-12))
    return v*factor[...,None],cosine


class CudaAttitude:
    def __init__(self,trials,params,network=None):
        batch=TranslationBatch(trials,params);p=batch.p;v=batch.v;bias=batch.bias;kp,kd,ff,_=params.tensors(p)
        aa=[];am=[]
        with torch.no_grad():
            for h,c in zip(batch.dt,batch.cmd):
                a=kp*(c[:,:3]-p)+kd*(c[:,3:6]-v)+ff*c[:,6:9]+bias
                extra=network(p,v,bias,c) if network is not None else torch.zeros_like(p)
                pm=p+.5*h*v;vm=v+.5*h*(a+extra);mid=kp*(c[:,:3]-pm)+kd*(c[:,3:6]-vm)+ff*c[:,6:9]+bias
                p=p+h*vm;v=v+h*(mid+(network(pm,vm,bias,c) if network is not None else 0.));aa.append(a);am.append(mid)
        initial=[t.initial_pose(params) for t in trials];r0=torch.cat([s.rotation for s in initial]);w0=torch.cat([s.omega_tracking for s in initial])
        mean_a=p.new_tensor(np.stack([t.b0 for t in trials]));mean_r=p.new_tensor(np.stack([t.mean_rotation for t in trials]));yaw=p.new_tensor([t.hover[2][0,0,9] for t in trials])
        truth=p.new_tensor(np.stack([t.truth['rotation_tracking_to_world'] for t in trials]));weights=p.new_tensor(np.stack([t.weights for t in trials])).sqrt()[:,:,None]/np.sqrt(len(trials))/.15
        indices=torch.tensor(np.stack([x[2] for x in batch.plans]),device='cuda');bi=torch.arange(len(trials),device='cuda')[:,None]
        self.values=p.new_tensor([params.attitude_acceleration_scale_xy,params.attitude_acceleration_scale_z,params.attitude_time_constant_s]);self.prior=self.values.clone()
        a=torch.stack(aa);a_mid=torch.stack(am)
        def forward():
            scale=torch.stack([self.values[0],self.values[0],self.values[1]]);tau=self.values[2]
            align=frame(mean_a*scale,yaw).transpose(-1,-2)@mean_r
            desired=frame(a*scale,batch.cmd[:,:,9])@align;desired_mid=frame(a_mid*scale,batch.cmd[:,:,9])@align
            r=r0;w=w0;rs=[r];minimum=r.new_ones(())
            for h,rd,rmid in zip(batch.dt,desired,desired_mid):
                error,cos=log_rotation(r.transpose(-1,-2)@rd);minimum=torch.minimum(minimum,cos.min())
                alpha=error/tau.square()-2*w/tau;rm=r@rotation_exp(.5*h*w);wm=w+.5*h*alpha
                error,cos=log_rotation(rm.transpose(-1,-2)@rmid);minimum=torch.minimum(minimum,cos.min())
                alpham=error/tau.square()-2*wm/tau;r=r@rotation_exp(h*wm);w=w+h*alpham;rs.append(r)
            prediction=torch.stack(rs)[indices,bi];error,_=log_rotation(prediction.transpose(-1,-2)@truth)
            residual=torch.cat([(error*weights).flatten(),np.sqrt(.01)*(self.values.log()-self.prior.log())])
            return residual,minimum,prediction
        self.forward=forward
        self.graph,self.outputs=capture(forward)

    def evaluate(self,x):
        self.values.copy_(self.values.new_tensor(np.exp(x)));self.graph.replay()
        if float(self.outputs[1])<-.9999:return np.full(self.outputs[0].numel(),100.)
        return self.outputs[0].cpu().numpy().copy()


def fit_nominal_gpu(trials,prior,folder,job,*,cold_start=True):
    gains0=np.array([getattr(prior,k) for k in GAIN_NAMES]);low=np.array([.1,.1,.1,.1,.05,.05]);high=np.array([80.,80.,20.,20.,3.,3.]);choices=[]
    for delay in [0.,.02,.04,.06,.08,.10,.12]:
        note(job,'CUDA nominal translation',delay_s=delay,windows=len(trials));start=time.perf_counter();batch=CudaNominal(trials,prior,delay)
        # Check an analytic Jacobian column against finite differences before optimization.
        value,jac=batch.evaluate(gains0);epsilon=1e-5;g=gains0.copy();g[0]+=epsilon
        fd=(batch.evaluate(g)[0]-value)/epsilon
        if np.max(abs(fd-jac[:,0]))>2e-4:raise AssertionError('CUDA nominal sensitivity mismatch')
        fit=least_squares(lambda g:batch.evaluate(g)[0],gains0,jac=lambda g:batch.evaluate(g)[1],bounds=(low,high),max_nfev=100,ftol=5e-5,xtol=1e-6,gtol=1e-6,x_scale=gains0)
        row=dict(delay_s=delay,gains=fit.x.tolist(),objective=float(2*fit.cost),nfev=fit.nfev,success=bool(fit.success),stop_reason=str(fit.message),seconds=time.perf_counter()-start)
        choices.append(row);save(folder/'nominal_progress.json',choices);del batch;gc.collect()
    best=min(choices,key=lambda r:r['objective']);params=parameters(best['gains'],.08,best['delay_s'])
    for t in trials:t.basis=t.basis_for(params.delay_s)
    params=fit_attitude_gpu(trials,params,None,folder/'nominal_attitude.json',job)
    save(folder/'nominal_fit.json',dict(parameters=asdict(params),translation_profiles=choices,training=[t.name for t in trials],method='CUDA batched analytic sensitivity replay; bounded CPU trust-region optimizer',gain_bounds=[low,high]))
    return params


def fit_attitude_gpu(trials,params,network,path,job):
    note(job,'CUDA attitude response',windows=len(trials));batch=CudaAttitude(trials,params,network)
    initial=np.log([params.attitude_acceleration_scale_xy,params.attitude_acceleration_scale_z,params.attitude_time_constant_s])
    fit=least_squares(batch.evaluate,initial,bounds=(np.log([.1,.05,.02]),np.log([3.,1.5,.3])),diff_step=1e-4,max_nfev=75,ftol=5e-5,xtol=1e-6,gtol=1e-6)
    selected=fit.x if np.linalg.norm(batch.evaluate(fit.x))<np.linalg.norm(batch.evaluate(initial)) else initial
    sx,sz,tau=np.exp(selected);save(path,dict(values=[sx,sz,tau],success=bool(fit.success),stop_reason=str(fit.message),nfev=fit.nfev,method='CUDA batched SO(3) midpoint replay',active_bounds=fit.active_mask))
    return replace(params,attitude_acceleration_scale_xy=sx,attitude_acceleration_scale_z=sz,attitude_time_constant_s=tau)
