"""Training-only selection and whole-flight validation for current adp0."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import itertools
import time
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from torch.utils.checkpoint import checkpoint

from .current_adaptation import *
from .attitude_identification import AttitudeTrial, DOMAIN_ERROR
from .bootstrap_drone import TranslationBatch
from .differentiable_fit import save_weights
from simulator.drone_pose_response import PoseResponseParameters
from simulator.drone_pose_residual import save_residual
from simulator.research_execution import ResearchExecutionModel
from simulator.research_physics import ResearchPhysics, research_physics_step


def note(job,stage,**info):
    row=dict(stage=stage,**info)
    save(Path(job)/'progress.json',row);print(json.dumps(finite_json(row)),flush=True)


def load_engine(job,*,trainable=False):
    path=Path(job)/'source_candidate/model.json'
    return ResearchExecutionModel.from_mapping(read(path),root=path.parent,device='cuda',trainable_residuals=trainable)


def folds(names):
    return [(f'leave_out_{n}',[x for x in names if x!=n],[n]) for n in names]+[('all_five',names,[])]


def fit_nominal(trials,prior,folder,job):
    if (folder/'nominal_fit.json').exists():
        saved=read(folder/'nominal_fit.json')
        if saved['training']!=[t.name for t in trials]:raise ValueError('Nominal fold membership changed')
        return PoseResponseParameters(**saved['parameters'])
    gains0=np.array([getattr(prior,k) for k in GAIN_NAMES])
    low=np.maximum(gains0*.5,[.1,.1,.1,.1,.01,.01]);high=np.minimum(gains0*2,[80,80,20,20,3,3])
    choices=[]
    for delay in sorted(set([.02,.04,.06,.08,.10,prior.delay_s])):
        cache={}
        def objective(g):
            if 'g' not in cache or not np.array_equal(g,cache['g']):
                errors=[];jacs=[]
                for t in trials:
                    p,_,jac=t.linear(g,delay);w=np.sqrt(t.weights/len(trials))/.05
                    errors.append(((p-t.truth['position_origin_m'])*w[:,None]).ravel())
                    jacs.append((jac*w[:,None,None]).reshape(-1,6))
                errors.append(np.sqrt(.03)*(g-gains0)/gains0)
                jacs.append(np.diag(np.sqrt(.03)/gains0))
                cache.update(g=g.copy(),value=(np.concatenate(errors),np.vstack(jacs)))
            return cache['value']
        fit=least_squares(lambda x:objective(x)[0],gains0,jac=lambda x:objective(x)[1],bounds=(low,high),
            max_nfev=45,ftol=1e-7,xtol=1e-7,gtol=1e-7,x_scale=gains0)
        choices.append(dict(delay_s=delay,gains=fit.x.tolist(),objective=float(2*fit.cost),nfev=fit.nfev,success=bool(fit.success)))
        note(job,folder.name+' nominal translation',**choices[-1])
    best=min(choices,key=lambda x:x['objective']);g=np.array(best['gains']);delay=best['delay_s']
    cached=[AttitudeTrial(t,g,delay) for t in trials]
    initial=np.array([prior.attitude_acceleration_scale_xy,prior.attitude_acceleration_scale_z,prior.attitude_time_constant_s])
    lower=np.maximum(initial*.5,[.1,.05,.02]);upper=np.minimum(initial*2,[3.,1.5,.30])
    traces=[]
    def attitude(x):
        values=np.exp(x);errors=[]
        try:
            for tr,c in zip(trials,cached):
                p=c.predict(values);err=Rotation.from_matrix(p.transpose(0,2,1)@c.truth).as_rotvec()
                errors.append((err*np.sqrt(tr.weights[:,None]/len(trials))/.15).ravel())
        except ValueError as exc:
            if str(exc) not in [DOMAIN_ERROR,'Desired specific-force direction is undefined','Desired heading and specific-force direction are singular']:raise
            return np.full(sum(len(t.time)*3 for t in trials)+3,100.)
        errors.append(np.sqrt(.03)*(x-np.log(initial)))
        out=np.concatenate(errors);traces.append(dict(values=values.tolist(),objective=float(out@out)))
        return out
    a=least_squares(attitude,np.log(initial),bounds=(np.log(lower),np.log(upper)),max_nfev=30,diff_step=1e-4,
        ftol=1e-6,xtol=1e-6,gtol=1e-6)
    sx,sz,tau=np.exp(a.x)
    params=parameters(g,tau,delay,(sx,sz))
    save(folder/'nominal_fit.json',dict(parameters=asdict(params),translation_profiles=choices,
        gain_bounds=[low,high],attitude_bounds=[lower,upper],attitude_trace=traces,
        training=[t.name for t in trials],geometry_frozen=True,prior='M0 nominal; old NN omitted for nominal fit'))
    return params


class WeightedTranslationBatch(TranslationBatch):
    def loss(self,network,regularization=.01):
        predictions,penalty=self.predict(network)
        losses=[]
        for p,y,t in zip(predictions,self.truth,self.trials):
            w=p.new_tensor(t.weights[:len(p)])
            # Smooth robust vector-position loss, 5 cm transition.
            sq=(p-y).square().sum(-1)/.05**2
            losses.append((2*(torch.sqrt(1+sq)-1)*w).sum())
        return torch.stack(losses).mean()+regularization*penalty/.5**2


def pose_prediction(trial,drone,times):
    d=trial.data;p=trial.initial_pose(drone.parameters)
    return drone.predict(p,torch.tensor(d['packets'][None],device=trial.device,dtype=torch.float64),
        d['packet_time'],times,trial.offset,graph=True,
        hover_command=torch.tensor(d['hover_commands'][-1:],device=trial.device,dtype=torch.float64))


def metric(pred,truth,times):
    out={}
    for name,lo,hi in [('whip',0,1),('early_recovery',1,3),('late_recovery_hold',3,11.2),('complete',0,11.2)]:
        mask=(times>=lo)&(times<=hi)
        error=np.linalg.norm(pred-truth,axis=-1)
        valid=mask.reshape((-1,)+(1,)*(error.ndim-1))&np.isfinite(error)
        values=error[valid]
        out[name]=dict(rmse_m=float(np.sqrt(np.mean(values**2))) if len(values) else None,
            mean_m=float(np.mean(values)) if len(values) else None,samples=int(len(values)),
            maximum_m=float(np.max(values)) if len(values) else None)
    return out


def drone_run(job=JOB):
    job=Path(job);model=read(job/'source_candidate/model.json');names=sorted(p.name for p in (job/'inputs').iterdir())
    trials={n:Trial(job,n,model) for n in names};settings=read(job/'protocol.json')
    for label,training,heldout in folds(names):
        folder=job/'drone'/label
        if (folder/'result.json').exists():continue
        folder.mkdir(parents=True,exist_ok=True);engine=load_engine(job,trainable=True)
        prior=deepcopy(engine.drone.parameters);params=fit_nominal([trials[n] for n in training],prior,folder,job)
        # Tiny per-flight MLP recurrences are faster on CPU than dispatching
        # thousands of small CUDA kernels. Execution validation remains CUDA.
        net=engine.drone.residual.cpu();start=deepcopy(net.state_dict());batch=WeightedTranslationBatch([trials[n] for n in training],params,device='cpu')
        optimizer=torch.optim.Adam(net.parameters(),lr=.001)
        best=deepcopy(net.state_dict());best_score=float(batch.loss(net).detach());best_update=0;history=[]
        for update in range(1,settings['drone_updates']+1):
            optimizer.zero_grad();loss=batch.loss(net)
            prior_penalty=torch.stack([(p-start[n]).square().mean() for n,p in net.named_parameters()]).mean()
            (loss+.01*prior_penalty).backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            if update%10==0:
                with torch.no_grad():score=float(batch.loss(net))
                if score<best_score:best_score=score;best=deepcopy(net.state_dict());best_update=update
                history.append(dict(update=update,loss=score,gradient_norm=float(norm)))
                save(folder/'history.json',history);note(job,label+' drone residual',**history[-1])
        net.load_state_dict(best);net.requires_grad_(False);net.to('cuda');save_residual(folder/'drone_residual.pt',net)
        payload=dict(nominal=dict(schema='nominal_loaded_drone_pose_candidate_v3',parameters=asdict(params),
            training_takes=training,alignment='prehover_effective_alignment',geometry=trials[names[0]].offset,
            adaptation_job=str(job),selected=False),residual=dict(schema='drone_pose_residual_v1',
                checkpoint='drone_residual.pt',sha256=sha256_file(folder/'drone_residual.pt'),specification=net.specification(),
                target='Additional realized origin acceleration; no direct injection into attitude drive'))
        save(folder/'drone_model.json',payload)
        engine.drone.parameters=params
        results={}
        for n in names:
            t=trials[n];pred=pose_prediction(t,engine.drone,t.time);p=pred['position_origin_m'][0].cpu().numpy()
            r=pred['rotation_tracking_to_world'][0].cpu().numpy()
            angular=np.linalg.norm(Rotation.from_matrix(r.transpose(0,2,1)@t.truth['rotation_tracking_to_world']).as_rotvec(),axis=1)
            check=WeightedTranslationBatch([t],params).predict(net)[0][0].detach().cpu().numpy()
            difference=float(np.max(np.abs(check-p)))
            # Shared nominal midpoint event partitions agree within numerical tolerance.
            if difference>2e-5:raise AssertionError(f'Drone training/execution mismatch: {difference}')
            np.savez_compressed(folder/(n+'.npz'),time=t.time,position=p,rotation=r)
            results[n]=dict(position=metric(p,t.truth['position_origin_m'],t.time),
                whip_attitude_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(angular[(t.time>=0)&(t.time<=1)]**2)))),
                training_runtime_max_abs_difference_m=difference)
        save(folder/'result.json',dict(training=training,heldout=heldout,selected_update=best_update,results=results))
        note(job,label+' drone complete',heldout=heldout,selected_update=best_update)


def cable_windows(trials,physics,starts,horizon):
    """Past-only state estimates; measured boundary is explicit for cable fitting."""
    records=[];rejected=[]
    for t in trials:
        for cutoff in starts:
            try:
                state,start,projection=t.cable_state(physics,cutoff=cutoff if cutoff else None)
                times=start+np.arange(round(horizon/t.model['simulation']['dt_s'])+1)*t.model['simulation']['dt_s']
                _,_,sites=t.measured(times)
                if not np.isfinite(sites).all():raise ValueError('Missing/jump-masked observations or boundary inside window')
                records.append(dict(name=t.name,cutoff=cutoff,time=times,q=state.positions_m,v=state.velocities_m_s,
                    roots=torch.tensor(sites[:,0][None],device=t.device,dtype=torch.float64),
                    truth=torch.tensor(sites[:,1:][None],device=t.device,dtype=torch.float64),projection_m=projection))
            except ValueError as exc:
                if str(exc) not in ['Missing causal cable history','Missing/jump-masked observations or boundary inside window']:raise
                rejected.append(dict(name=t.name,cutoff=cutoff,reason=str(exc)))
    return records,rejected


def join_windows(records):
    return {key:torch.cat([r[key] for r in records]) for key in ['q','v','roots','truth']}


def cable_forward(engine,data,parameters,*,gradients=False,checkpoint_steps=10):
    q,v=data['q'],data['v'];dt=q.new_full((len(q),),engine.dt_s)
    params=q.new_tensor(parameters) if not isinstance(parameters,torch.Tensor) else parameters
    if params.ndim==1:params=params[:,None].expand(-1,len(q))
    constants=replace(engine.physics.runtime_constants(q),bending_stiffness_n_m2=params[0],bending_damping_n_m2_s=params[1])
    fast=None if gradients else ResearchPhysics(engine.physics,DderState(q,v),engine.dt_s,constants=constants,graph=True)
    def advance(q,v,roots):
        out=[];vel=[]
        for root in roots.unbind(1):
            s=(research_physics_step(engine.physics,DderState(q,v),root,dt,constants,create_graph=True)
                if gradients else fast(DderState(q,v),root))
            q,v=s.positions_m,s.velocities_m_s;out.append(q);vel.append(v)
        return q,v,torch.stack(out,1),torch.stack(vel,1)
    positions=[q[:,None]];velocities=[v[:,None]]
    with torch.enable_grad() if gradients else torch.no_grad():
        for i in range(1,data['roots'].shape[1],checkpoint_steps):
            roots=data['roots'][:,i:i+checkpoint_steps]
            if gradients:q,v,qs,vs=checkpoint(advance,q,v,roots,use_reentrant=False)
            else:q,v,qs,vs=advance(q,v,roots)
            positions.append(qs);velocities.append(vs)
        q=torch.cat(positions,1);v=torch.cat(velocities,1)
        if not bool(torch.isfinite(q).all()&torch.isfinite(v).all()):raise FloatingPointError('Nonfinite cable prediction')
    return q,v


def cable_objectives(q,truth,marker_indices):
    err=q[:,1:,marker_indices]-truth[:,1:]
    sq=err.square().sum(-1)/.05**2
    robust=2*(torch.sqrt(1+sq)-1)
    return .7*robust.mean((1,2))+.3*robust[:,:,-1].mean(1)


def record_weights(records,training):
    weights=np.zeros(len(records))
    for name in training:
        for whip,amount in [(True,.8),(False,.2)]:
            ids=[i for i,r in enumerate(records) if r['name']==name and (r['cutoff']<1)==whip]
            if ids:weights[ids]=amount/len(ids)/len(training)
    if weights.sum()==0:raise ValueError('No training cable windows')
    return weights/weights.sum()


def cable_run(job=JOB):
    job=Path(job);settings=read(job/'protocol.json');model=read(job/'source_candidate/model.json')
    names=sorted(p.name for p in (job/'inputs').iterdir());trials=[Trial(job,n,model) for n in names]
    engine=load_engine(job,trainable=True)
    # Full whip plus early recovery windows select parameters/checkpoints. Missing
    # recovery windows are documented; no selected window resets during rollout.
    long,rejected=cable_windows(trials,engine.physics,[0.,1.2,2.0],1.02)
    short,short_rejected=cable_windows(trials,engine.physics,settings['cable_window_starts_s'],settings['cable_window_s'])
    for name in names:
        if not any(r['name']==name and r['cutoff']==0 for r in long):raise ValueError('Every flight needs a complete-whip validation window')
    out=job/'cable';out.mkdir(exist_ok=True)
    save(out/'windows.json',dict(long=[{k:r[k] for k in ['name','cutoff','time','projection_m']} for r in long],
        short=[{k:r[k] for k in ['name','cutoff','time','projection_m']} for r in short],
        rejected=rejected+short_rejected))
    ld=join_windows(long);sd=join_windows(short);marker_indices=list(engine.cable.marker_node_indices[1:])
    prior=np.array([model['cable']['EI_n_m2'],model['cable']['Cb_n_m2_s']])
    candidates=np.array([prior*np.array(f) for f in itertools.product([.25,1.,4.],repeat=2)])
    if not (out/'physical_grid.npz').exists():
        expanded={k:v.repeat_interleave(len(candidates),0) for k,v in ld.items()}
        params=torch.tensor(np.tile(candidates,(len(long),1)).T,device='cuda',dtype=torch.float64)
        note(job,'cable physical grid',candidates=len(candidates),windows=len(long))
        with torch.no_grad():
            q,_=cable_forward(engine,expanded,params)
            losses=cable_objectives(q,expanded['truth'],marker_indices).reshape(len(long),len(candidates)).cpu().numpy()
        np.savez_compressed(out/'physical_grid.npz',candidates=candidates,losses=losses)
    with np.load(out/'physical_grid.npz') as z:losses=z['losses'];candidates=z['candidates']
    for label,training,heldout in folds(names):
        folder=out/label
        if (folder/'result.json').exists():continue
        folder.mkdir(exist_ok=True);engine=load_engine(job,trainable=True);net=engine.physics.motion_residual
        lw=record_weights(long,training);scores=lw@losses+.01*(np.log(candidates/prior)**2).mean(1)
        chosen=candidates[scores.argmin()];save(folder/'physics.json',dict(candidates=candidates,scores=scores,selected=chosen,training=training,heldout=heldout))
        note(job,label+' cable residual starting',physics=chosen,training=training)
        # Old feature/damping weights may adapt gently; the bounded new head starts at zero.
        original=deepcopy(net.state_dict());optimizer=torch.optim.Adam([
            {'params':list(net.net.parameters()),'lr':settings['cable_learning_rate']*.25},
            {'params':list(net.correction_head.parameters()),'lr':settings['cable_learning_rate']}])
        chosen_tensor=torch.tensor(chosen,device='cuda',dtype=torch.float64)
        selected=[i for i,r in enumerate(short) if r['name'] in training]
        train_sd={k:v[selected] for k,v in sd.items()};sw=torch.tensor(record_weights([short[i] for i in selected],training),device='cuda',dtype=torch.float64)
        long_ids=[i for i,r in enumerate(long) if r['name'] in training]
        train_ld={k:v[long_ids] for k,v in ld.items()};tw=torch.tensor(record_weights([long[i] for i in long_ids],training),device='cuda',dtype=torch.float64)
        def select_score():
            with torch.no_grad():
                q,v=cable_forward(engine,train_ld,chosen_tensor)
                _,extra=net.components(q.reshape(-1,q.shape[2],3),v.reshape(-1,v.shape[2],3))
                value=(cable_objectives(q,train_ld['truth'],marker_indices)*tw).sum()+.01*extra.square().mean()/.5**2
                return float(value)
        best_score=select_score();best=deepcopy(net.state_dict());best_update=0;history=[]
        for update in range(1,settings['cable_updates']+1):
            started=time.perf_counter();optimizer.zero_grad()
            q,v=cable_forward(engine,train_sd,chosen_tensor,gradients=True)
            loss=(cable_objectives(q,train_sd['truth'],marker_indices)*sw).sum()
            _,extra=net.components(q.reshape(-1,q.shape[2],3),v.reshape(-1,v.shape[2],3))
            prior_penalty=torch.stack([(p-original[n]).square().mean() for n,p in net.named_parameters()]).mean()
            loss=loss+.01*extra.square().mean()/.5**2+.01*prior_penalty
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite residual loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            row=dict(update=update,training_loss=float(loss.detach()),gradient_norm=float(norm),elapsed_s=time.perf_counter()-started)
            if update%6==0:
                score=select_score();row['complete_window_selection_score']=score
                if score<best_score:best_score=score;best=deepcopy(net.state_dict());best_update=update
            history.append(row);save(folder/'history.json',history);note(job,label+' cable residual',**row)
        net.load_state_dict(best);net.requires_grad_(False);save_weights(folder/'cable_residual.pt',net)
        with torch.no_grad():q,v=cable_forward(engine,ld,chosen_tensor)
        np.savez_compressed(folder/'predictions.npz',positions=q.cpu().numpy(),velocities=v.cpu().numpy())
        save(folder/'result.json',dict(training=training,heldout=heldout,selected_physics=chosen,
            selected_update=best_update,selection_score=best_score,
            per_window_objectives=cable_objectives(q,ld['truth'],marker_indices).cpu().numpy(),
            specification=net.specification()))
        note(job,label+' cable complete',selected_update=best_update)
