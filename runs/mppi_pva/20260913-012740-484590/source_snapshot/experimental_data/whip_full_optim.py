"""GPU batched, event-exact full response fitting and durable stopping."""
from pathlib import Path
from dataclasses import replace,asdict
from copy import deepcopy
import time,gc
import numpy as np
import torch
from scipy.optimize import least_squares
from .io import atomic_json
from .plateau import Plateau
from .nominal_pose_fit import GAIN_NAMES,integration_steps
from .preliminary_gpu_drone import capture,frame,log_rotation
from .bootstrap_drone import TranslationBatch
from simulator.drone_pose_response import rotation_exp
from simulator.cuda_autograd import CudaAutogradBlock


def progress(job,stage,**values):
    job=Path(job)
    if (job/'STOP').exists():raise InterruptedError('Stopped; completed stages and optimizer state retained')
    row=dict(stage=stage,**values);atomic_json(job/'progress.json',row)
    atomic_json(job/'status.json',dict(status='running',model_selected=False,**row))
    print(stage,values,flush=True)


def trust_fit(evaluate,initial,bounds,folder,job,label,settings):
    """Best iterate is retained even after callback plateau or solver termination."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    initial=np.asarray(initial,float);baseline=float(evaluate(initial)[0]@evaluate(initial)[0])
    best=initial.copy();best_loss=baseline;history=[];started=time.perf_counter()
    stop=Plateau(settings['minimum'],settings['patience'],settings['relative']);stop.observe(0,baseline)
    def callback(x):
        nonlocal best,best_loss
        value=evaluate(x)[0];loss=float(value@value)
        if loss<best_loss:best=x.copy();best_loss=loss
        _,done=stop.observe(len(history)+1,best_loss)
        row=dict(update=len(history)+1,loss=loss,best_loss=best_loss,baseline_loss=baseline,values=np.exp(x).tolist(),elapsed_s=time.perf_counter()-started)
        history.append(row);atomic_json(folder/'history.json',history)
        atomic_json(folder/'search_state.json',dict(best=best.tolist(),best_loss=best_loss,plateau=asdict(stop),resume_automatic=False))
        progress(job,label,update=len(history),loss=loss,best_loss=best_loss)
        if done:raise StopIteration
    fit=least_squares(lambda x:evaluate(x)[0],initial,jac=lambda x:evaluate(x)[1],bounds=bounds,
        max_nfev=settings['ceiling'],ftol=1e-6,xtol=1e-6,gtol=1e-6,callback=callback)
    value=evaluate(fit.x)[0];loss=float(value@value)
    if loss<best_loss:best=fit.x.copy();best_loss=loss
    reason='practical_plateau' if fit.status==-2 else ('safety_ceiling' if fit.status==0 else 'solver_termination')
    result=dict(best=best.tolist(),best_loss=best_loss,baseline_loss=baseline,updates=len(history),nfev=fit.nfev,
        stop_reason=reason,solver_message=str(fit.message),active_bounds=fit.active_mask.tolist(),elapsed_s=time.perf_counter()-started)
    atomic_json(folder/'result.json',result)
    return best,result


class FullTranslation:
    """All six gains, inherited nonlinear residual and parameter-dependent past state."""
    def __init__(self,trials,params,network,contract,count=13):
        self.trials=trials;self.params=params;self.network=network;self.contract=contract;self.count=count
        self.plans=[integration_steps(t.schedule,t.time,params.delay_s) for t in trials]
        length=max(len(p[0]) for p in self.plans);nt=max(len(t.time) for t in trials);b=len(trials)
        dt=np.zeros((length,b,1));cmd=np.zeros((length,b,11));indices=np.zeros((b,nt),int)
        truth=np.zeros((b,nt,3));weights=np.zeros((b,nt))
        for i,(t,(h,c,ix)) in enumerate(zip(trials,self.plans)):
            dt[:len(h),i,0]=h;cmd[:len(h),i]=c;cmd[len(h):,i]=c[-1]
            indices[i,:len(ix)]=ix;indices[i,len(ix):]=ix[-1]
            truth[i,:len(t.time)]=np.nan_to_num(t.truth['position_origin_m'])
            weights[i,:len(t.time)]=t.weights/len(trials)
        tensor=lambda x:torch.as_tensor(np.asarray(x),device='cuda',dtype=torch.float64)
        self.dt=tensor(dt).repeat_interleave(count,1);self.cmd=tensor(cmd).repeat_interleave(count,1)
        self.p=tensor([t.p0 for t in trials]).repeat_interleave(count,0);self.v=tensor([t.v0 for t in trials]).repeat_interleave(count,0)
        self.b0=tensor([t.b0 for t in trials]).repeat_interleave(count,0)
        self.basis=tensor(np.stack([t.basis_for(params.delay_s) for t in trials])).repeat_interleave(count,0)
        self.indices=torch.as_tensor(indices,device='cuda');self.truth=tensor(truth);self.weights=tensor(weights)
        self.window_weights=self.weights.sum(1);self.duration=self.dt[:,:,0].sum(0)
        self.gains=tensor([[getattr(params,k) for k in GAIN_NAMES]]).repeat(count,1)
        self.pick=torch.tensor([0,0,1],device='cuda');self.bi=torch.arange(b,device='cuda')[:,None,None]
        self.ci=torch.arange(count,device='cuda')[None,None,:]
        self.index= self.indices[:,:,None].expand(b,nt,count)
        self.prior=self.gains[0].clone();self.cached=None
        self.reference=deepcopy(network).requires_grad_(False)
        self.b=b

    def rollout(self,*,regularization=False,initial=None):
        g=self.gains.repeat(self.b,1);kp=g[:,:2][:,self.pick];kd=g[:,2:4][:,self.pick];ff=g[:,4:][:,self.pick]
        bias=self.b0+(self.basis*g[:,None]).sum(-1)
        p,v=(self.p,self.v) if initial is None else initial
        ps=[p];magnitude=p.new_zeros(len(p));change=magnitude.clone()
        for h,c in zip(self.dt,self.cmd):
            a=kp*(c[:,:3]-p)+kd*(c[:,3:6]-v)+ff*c[:,6:9]+bias+self.network(p,v,bias,c)
            pm=p+.5*h*v;vm=v+.5*h*a;extra=self.network(pm,vm,bias,c)
            am=kp*(c[:,:3]-pm)+kd*(c[:,3:6]-vm)+ff*c[:,6:9]+bias+extra
            if regularization:
                magnitude=magnitude+h[:,0]*extra.square().mean(-1)
                old=self.reference(pm,vm,bias,c)
                change=change+h[:,0]*(extra-old).square().mean(-1)
            p=p+h*vm;v=v+h*am;ps.append(p)
        allp=torch.stack(ps).reshape(-1,self.b,self.count,3)
        pred=allp[self.index,self.bi,self.ci].permute(2,0,1,3)
        delta=(pred-self.truth[None])/self.contract['position_scale_m'];sq=delta.square().sum(-1)
        factor=torch.sqrt(2/(torch.sqrt(1+sq)+1))
        residual=delta*factor[...,None]*self.weights.sqrt()[None,:,:,None]
        penalty=((magnitude/self.duration).reshape(self.b,self.count)*self.window_weights[:,None]).sum(0)
        difference=((change/self.duration).reshape(self.b,self.count)*self.window_weights[:,None]).sum(0)
        return pred,residual,penalty,difference

    def capture_nominal(self):
        def forward():
            pred,r,_,_=self.rollout()
            reg=np.sqrt(self.contract['nominal_prior'])*(self.gains/self.prior).log()
            return torch.cat([r.flatten(1),reg],1),pred
        self.forward=forward;self.graph,self.outputs=capture(forward)

    def evaluate(self,x):
        if self.cached is None or not np.array_equal(x,self.cached):
            epsilon=1e-4;values=np.r_[x[None],x[None]+epsilon*np.eye(6),x[None]-epsilon*np.eye(6)]
            self.gains.copy_(self.gains.new_tensor(np.exp(values)));self.graph.replay()
            r=self.outputs[0].cpu().numpy().copy()
            if not np.isfinite(r).all():raise FloatingPointError('Nonfinite drone candidate')
            self.value=r[0],((r[1:7]-r[7:13])/(2*epsilon)).T;self.cached=x.copy()
        return self.value

    def residual_block(self):
        def objective(p,v):
            # Inputs are the same fixed causal state; parameters contain the trainable NN.
            pred,r,mag,change=self.rollout(regularization=True,initial=(p,v))
            value=r.square().sum()+self.contract['residual_magnitude']*mag[0]/.5**2+self.contract['residual_change']*change[0]/.5**2
            return (value,)
        self.objective=objective
        self.block=CudaAutogradBlock(objective,(self.p,self.v),tuple(self.network.parameters()))
        return lambda:self.block(self.p,self.v)[0]


def fit_nominal(trials,engine,contract,folder,job):
    folder=Path(folder);folder.mkdir();prior=engine.drone.parameters;profiles=[];best=None;selected=float('inf')
    original=np.log([getattr(prior,k) for k in GAIN_NAMES]);bounds=np.log(contract['drone_gain_bounds'])
    for delay in contract['delay_candidates_s']:
        progress(job,'drone_nominal',delay_s=delay,windows=len(trials))
        params=replace(prior,delay_s=delay);batch=FullTranslation(trials,params,engine.drone.residual,contract)
        batch.capture_nominal()
        x,res=trust_fit(batch.evaluate,original,(bounds[0],bounds[1]),folder/f'delay-{delay:.3f}',job,'drone_nominal',contract['nominal_stopping'])
        delay_penalty=contract['nominal_prior']*((delay-prior.delay_s)/.04)**2
        score=res['best_loss']+delay_penalty
        profiles.append(dict(delay_s=delay,score=score,delay_prior=delay_penalty,**res))
        if score<selected:selected=score;best=replace(prior,delay_s=delay,**dict(zip(GAIN_NAMES,np.exp(x))))
        atomic_json(folder/'profiles.json',profiles);del batch;gc.collect();torch.cuda.empty_cache()
    # Nominal response includes attitude, fitted separately to preserve its objective.
    best=fit_attitude(trials,best,engine.drone.residual,contract,folder/'attitude',job)
    atomic_json(folder/'parameters.json',asdict(best));return best


class FullAttitude:
    """Production SO(3) midpoint response with unequal, masked native timelines."""
    def __init__(self,trials,params,network,contract):
        batch=TranslationBatch(trials,params);p=batch.p;v=batch.v;bias=batch.bias;kp,kd,ff,_=params.tensors(p)
        aa=[];am=[]
        with torch.no_grad():
            for h,c in zip(batch.dt,batch.cmd):
                a=kp*(c[:,:3]-p)+kd*(c[:,3:6]-v)+ff*c[:,6:9]+bias
                pm=p+.5*h*v;vm=v+.5*h*(a+network(p,v,bias,c))
                mid=kp*(c[:,:3]-pm)+kd*(c[:,3:6]-vm)+ff*c[:,6:9]+bias
                p=p+h*vm;v=v+h*(mid+network(pm,vm,bias,c));aa.append(a);am.append(mid)
        initial=batch.states;r0=torch.cat([s.rotation for s in initial]);w0=torch.cat([s.omega_tracking for s in initial])
        mean_a=p.new_tensor(np.stack([t.b0 for t in trials]));mean_r=p.new_tensor(np.stack([t.mean_rotation for t in trials]))
        yaw=p.new_tensor([t.hover[2][0,0,9] for t in trials]);nt=max(len(t.time) for t in trials)
        truth=p.new_tensor(np.broadcast_to(np.eye(3),(len(trials),nt,3,3)).copy());weights=p.new_zeros(len(trials),nt)
        indices=torch.zeros(len(trials),nt,device=p.device,dtype=torch.long)
        for i,t in enumerate(trials):
            n=len(t.time);truth[i,:n]=p.new_tensor(t.truth['rotation_tracking_to_world'])
            weights[i,:n]=p.new_tensor(t.weights)/len(trials);indices[i,:n]=torch.as_tensor(batch.plans[i][2],device=p.device)
            indices[i,n:]=indices[i,n-1]
        bi=torch.arange(len(trials),device=p.device)[:,None]
        self.values=p.new_tensor([params.attitude_acceleration_scale_xy,params.attitude_acceleration_scale_z,params.attitude_time_constant_s])
        self.prior=self.values.clone();a=torch.stack(aa);a_mid=torch.stack(am)
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
            delta=error/contract['orientation_scale_rad'];factor=torch.sqrt(2/(torch.sqrt(1+delta.square().sum(-1))+1))
            residual=delta*factor[...,None]*weights.sqrt()[...,None]
            return torch.cat([residual.flatten(),np.sqrt(contract['nominal_prior'])*(self.values.log()-self.prior.log())]),minimum,prediction
        self.forward=forward;self.graph,self.outputs=capture(forward)

    def evaluate(self,x):
        self.values.copy_(self.values.new_tensor(np.exp(x)));self.graph.replay()
        if float(self.outputs[1])<-.9999:raise FloatingPointError('Attitude candidate crosses rotation-log singularity')
        r=self.outputs[0].cpu().numpy().copy()
        if not np.isfinite(r).all():raise FloatingPointError('Nonfinite attitude candidate')
        return r


def fit_attitude(trials,params,network,contract,folder,job):
    progress(job,'attitude response',windows=len(trials));batch=FullAttitude(trials,params,network,contract)
    x0=np.log([params.attitude_acceleration_scale_xy,params.attitude_acceleration_scale_z,params.attitude_time_constant_s]);cache={}
    def evaluate(x):
        if 'x' not in cache or not np.array_equal(x,cache['x']):
            r=batch.evaluate(x);eps=1e-4
            jac=np.stack([(batch.evaluate(x+eps*e)-batch.evaluate(x-eps*e))/(2*eps) for e in np.eye(3)],1)
            cache.update(x=x.copy(),value=(r,jac))
        return cache['value']
    b=np.log(contract['attitude_bounds']);x,res=trust_fit(evaluate,x0,(b[0],b[1]),folder,job,'attitude response',contract['nominal_stopping'])
    sx,sz,tau=np.exp(x);del batch;gc.collect()
    return replace(params,attitude_acceleration_scale_xy=sx,attitude_acceleration_scale_z=sz,attitude_time_constant_s=tau)


def train_residual(net,objective,contract,folder,job,label):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);settings=contract['residual_stopping']
    optimizer=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=1e-4)
    score=float(objective().detach());baseline=score;best=deepcopy(net.state_dict());best_update=0
    stop=Plateau(settings['minimum'],settings['patience'],settings['relative']);stop.observe(0,score)
    history=[];reason='safety_ceiling';started=time.perf_counter()
    for update in range(1,settings['ceiling']+1):
        progress(job,label,update=update,best_loss=stop.best)
        optimizer.zero_grad();loss=objective()
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite '+label)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if update%settings['check_every']==0 or update==settings['ceiling']:
            with torch.no_grad():score=float(objective())
            improved,done=stop.observe(update,score)
            if improved:best=deepcopy(net.state_dict());best_update=update
            history.append(dict(update=update,loss=float(loss.detach()),best_loss=stop.best,selection_loss=score,
                baseline_loss=baseline,gradient_norm=float(norm),elapsed_s=time.perf_counter()-started))
            atomic_json(folder/'history.json',history)
            torch.save(dict(update=update,current=net.state_dict(),best=best,optimizer=optimizer.state_dict(),
                best_update=best_update,plateau=asdict(stop)),folder/'state.tmp')
            (folder/'state.tmp').replace(folder/'state.pt')
            if done:reason='practical_plateau';break
    net.load_state_dict(best);net.requires_grad_(False)
    result=dict(updates=update,selected_update=best_update,best_loss=stop.best,baseline_loss=baseline,stop_reason=reason,elapsed_s=time.perf_counter()-started)
    atomic_json(folder/'result.json',result);return result
