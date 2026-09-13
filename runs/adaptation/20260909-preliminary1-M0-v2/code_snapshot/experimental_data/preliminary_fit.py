"""One fresh GPU M0 fit, with whole-take validation and immutable inputs."""
from pathlib import Path
from dataclasses import replace,asdict
from copy import deepcopy
import itertools,os,time,shutil,gc
import numpy as np
import torch
from scipy.optimize import least_squares
from .current_adaptation import read,save
from .current_adaptation_fit import load_engine,note,cable_windows,join_windows,cable_objectives,pose_prediction
from .preliminary_prepare import PreliminaryTrial
from .preliminary_gpu_drone import fit_nominal_gpu,fit_attitude_gpu
from .pva_bootstrap import selected_training
from .cuda_drone_fit import CudaDroneFit
from .cuda_cable_fit import CudaCableFit
from .differentiable_fit import save_weights
from .io import sha256_file
from simulator.cable import DderState
from simulator.research_physics import ResearchPhysics
from simulator.research_execution import ResearchExecutionModel
from simulator.drone_pose_residual import save_residual


class CableForward:
    def __init__(self,engine,data):
        self.data=data;self.parameters=data['q'].new_ones((2,len(data['q'])))*1e-6
        constants=replace(engine.physics.runtime_constants(data['q']),bending_stiffness_n_m2=self.parameters[0],bending_damping_n_m2_s=self.parameters[1])
        self.step=ResearchPhysics(engine.physics,DderState(data['q'],data['v']),engine.dt_s,constants=constants,graph=True,fast_solve=True,fast_geometry=True)

    @torch.no_grad()
    def __call__(self,parameters):
        values=self.parameters.new_tensor(parameters) if not isinstance(parameters,torch.Tensor) else parameters
        self.parameters.copy_(values[:,None] if values.ndim==1 else values)
        state=DderState(self.data['q'],self.data['v']);q=[state.positions_m];v=[state.velocities_m_s]
        for root in self.data['roots'][:,1:].unbind(1):
            state=self.step(state,root);q.append(state.positions_m);v.append(state.velocities_m_s)
        q,v=torch.stack(q,1),torch.stack(v,1)
        if not bool(torch.isfinite(q).all()&torch.isfinite(v).all()):raise FloatingPointError('Nonfinite cable trajectory')
        return q,v


def equal_take_weights(records):
    names=sorted({r['take'] for r in records})
    return np.array([1/(len(names)*sum(s['take']==r['take'] for s in records)) for r in records])


def fit_drone(job,trials,engine,settings):
    folder=job/'drone';folder.mkdir();params=fit_nominal_gpu(trials,engine.drone.parameters,folder,job)
    gc.collect();net=engine.drone.residual;batch=CudaDroneFit(trials,params,net);note(job,'CUDA drone residual',windows=len(trials))
    def select():
        with torch.no_grad():return float(batch())
    result=selected_training(net,torch.optim.Adam(net.parameters(),lr=.001,weight_decay=1e-4),batch,select,settings['drone'],folder,job,'drone residual')
    net.requires_grad_(False);del batch;gc.collect()
    params=fit_attitude_gpu(trials,params,net,folder/'attitude_refinement.json',job)
    engine.drone.parameters=params;save_residual(folder/'drone_residual.pt',net)
    save(folder/'drone_model.json',dict(nominal=dict(schema='nominal_loaded_drone_pose_candidate_v3',parameters=asdict(params),training_takes=sorted({t.take for t in trials}),initialization='causal moving-history state, not assumed rest'),
        residual=dict(checkpoint='drone_residual.pt',sha256=sha256_file(folder/'drone_residual.pt'),specification=net.specification())))
    save(folder/'result.json',dict(stopping=result,parameters=asdict(params)))


def fit_cable(job,records,engine,settings):
    folder=job/'cable';folder.mkdir();data=join_windows(records);weights=data['q'].new_tensor(equal_take_weights(records));indices=list(engine.cable.marker_node_indices[1:])
    candidates=np.array(list(itertools.product(np.logspace(-8,-5,4),np.logspace(-8,-4,5))))
    expanded={k:v.repeat_interleave(len(candidates),0) for k,v in data.items()}
    note(job,'CUDA cable physical grid',windows=len(records),candidates=len(candidates))
    replay=CableForward(engine,expanded);q,_=replay(np.tile(candidates,(len(records),1)).T)
    values=(cable_objectives(q,expanded['truth'],indices).reshape(len(records),-1)*weights[:,None]).sum(0).cpu().numpy()
    chosen=candidates[np.argmin(values)];save(folder/'grid.json',dict(candidates=candidates,scores=values,selected=chosen));del replay,expanded,q
    physics=folder/'physics';physics.mkdir();history=[]
    expanded={k:v.repeat_interleave(3,0) for k,v in data.items()};replay=CableForward(engine,expanded)
    marker=data['q'].new_full((len(indices),),.7/len(indices));marker[-1]+=.3
    scale=(weights[:,None,None]*marker[None,None,:]/(data['truth'].shape[1]-1)).sqrt()
    cache={};epsilon=.001
    def evaluate(x):
        if 'x' not in cache or not np.array_equal(x,cache['x']):
            note(job,'cable physics',evaluation=len(history)+1);start=time.perf_counter()
            proposals=np.stack([np.exp(x),np.exp(x+[epsilon,0]),np.exp(x+[0,epsilon])]);q,_=replay(np.tile(proposals,(len(records),1)).T)
            err=(q[:,1:,indices]-expanded['truth'][:,1:])/.05;sq=err.square().sum(-1)
            factor=torch.sqrt(2/(torch.sqrt(1+sq)+1));res=err*factor[...,None]
            res=res.reshape(len(records),3,*res.shape[1:])*scale[:,None,:,:,None]
            columns=res.permute(1,0,2,3,4).flatten(1).cpu().numpy();base=np.r_[columns[0],.01*(x-np.log(1e-6))]
            jac=np.vstack([((columns[1:]-columns[:1])/epsilon).T,.01*np.eye(2)])
            history.append(dict(update=len(history)+1,loss=float(base@base),parameters=np.exp(x),seconds=time.perf_counter()-start));save(physics/'history.json',history)
            cache.update(x=x.copy(),value=(base,jac))
        return cache['value']
    fit=least_squares(lambda x:evaluate(x)[0],np.log(chosen),jac=lambda x:evaluate(x)[1],bounds=(np.log([1e-9,1e-9]),np.log([1e-4,1e-3])),max_nfev=70,ftol=5e-4,xtol=1e-5,gtol=1e-5)
    chosen=data['q'].new_tensor(np.exp(fit.x));save(physics/'parameters.json',dict(EI_n_m2=float(chosen[0]),Cb_n_m2_s=float(chosen[1]),active_bounds=fit.active_mask,success=bool(fit.success),stop_reason=str(fit.message),nfev=fit.nfev,objective=float(2*fit.cost)))
    save(physics/'stopping.json',dict(converged=bool(fit.success),stop_reason=str(fit.message)));del replay,expanded
    net=engine.physics.motion_residual;net.requires_grad_(True);residual=folder/'residual';residual.mkdir()
    note(job,'CUDA cable residual graph preparation',windows=len(records));accelerator=CudaCableFit(engine,data,chosen,block_steps=3)
    def objective(gradients):
        q,v=accelerator(data,gradients=gradients)
        _,extra=net.components(q[:,1:].flatten(0,1),v[:,1:].flatten(0,1))
        penalty=extra.square().reshape(len(q),-1).mean(-1)/.5**2
        return ((cable_objectives(q,data['truth'],indices)+.01*penalty)*weights).sum()
    def select():
        with torch.no_grad():return float(objective(False))
    result=selected_training(net,torch.optim.Adam(net.parameters(),lr=.001,weight_decay=1e-4),lambda:objective(True),select,settings['cable'],residual,job,'cable residual')
    net.requires_grad_(False);save_weights(folder/'cable_residual.pt',net)
    save(folder/'result.json',dict(stopping=result,selected_physics=chosen.cpu().numpy(),specification=net.specification(),windows=len(records),objective='Equal-take weighted recursive one-second windows across full preliminary motion; measured attachment'))


def rms(pred,truth):
    valid=np.isfinite(pred).all(-1)&np.isfinite(truth).all(-1);error=np.linalg.norm(pred-truth,axis=-1)[valid]
    return dict(rmse_m=float(np.sqrt(np.mean(error**2))) if len(error) else None,samples=len(error))


def publish(job,trials,records):
    out=job/'candidate';out.mkdir();model=read(job/'source_candidate/model.json')
    for n in ('drone_model.json','drone_residual.pt'):shutil.copy2(job/'drone'/n,out/n)
    shutil.copy2(job/'cable/cable_residual.pt',out/'cable_residual.pt');result=read(job/'cable/result.json');ei,cb=result['selected_physics']
    model['cable'].update(EI_n_m2=ei,Cb_n_m2_s=cb)
    model['fullstate_execution'].update(checkpoint='drone_model.json',sha256=sha256_file(out/'drone_model.json'),source_job=str(job))
    model['motion_residual'].update(checkpoint='cable_residual.pt',sha256=sha256_file(out/'cable_residual.pt'),specification=result['specification'])
    protocol=read(job/'protocol.json');model['provenance'].update(label='M0 - preliminary1 - 145 g drone / 17 g cable',fit_job=str(job),fit_complete=False,training_takes=protocol['training_takes'],validation_takes=protocol['validation_takes'],prospective_flight_evidence=False)
    save(out/'model.json',model);engines={'cold':load_engine(job),'fitted':ResearchExecutionModel.from_mapping(model,root=out,device='cuda')};results={}
    # Assess all prepared drone windows and all valid cable windows without using
    # validation results to select/restart weights. No fold/ablation campaign.
    for label,engine in engines.items():
        note(job,'checking '+label+' model')
        from .bootstrap_drone import TranslationBatch
        batch=TranslationBatch(trials,engine.drone.parameters)
        with torch.no_grad():prediction,_=batch.predict(engine.drone.residual)
        for t,p in zip(trials,prediction):
            entry=results.setdefault(t.take,dict(role=t.role,drone={},cable_tip={}))
            entry['drone'].setdefault(label,[]).append(rms(p.cpu().numpy()[1:],t.truth['position_origin_m'][1:]))
        data=join_windows(records);replay=CableForward(engine,data);physical=[engine.physics.parameters.bending_stiffness_n_m2,engine.physics.parameters.bending_damping_n_m2_s]
        q,_=replay(physical)
        for record,p,y in zip(records,q.cpu().numpy(),data['truth'].cpu().numpy()):results[record['take']]['cable_tip'].setdefault(label,[]).append(rms(p[1:,-1],y[1:,-1]))
        del replay,batch,prediction
    summary={}
    for take,row in results.items():
        summary[take]=dict(role=row['role'])
        for component in ('drone','cable_tip'):
            summary[take][component]={label:dict(rmse_m=float(np.sqrt(sum(x['rmse_m']**2*x['samples'] for x in values)/sum(x['samples'] for x in values))),samples=sum(x['samples'] for x in values)) for label,values in row[component].items()}
    save(out/'window_diagnostics.json',dict(takes=summary,evidence='Retrospective causal 2 s drone / 1 s cable windows; validation take excluded from fitting and selection'))
    # One coupled two-second prediction per take, chosen by midpoint index before
    # seeing prediction error. Both use the same measured initial pose/cable history.
    coupled={}
    consistency={}
    for take in sorted(results):
        eligible={r['name'] for r in records if r['cutoff']==0.}
        choices=[t for t in trials if t.take==take and t.name in eligible];t=choices[len(choices)//2];times=t.grid(end=2.)
        _,_,truth=t.measured(times);coupled[take]={}
        for label,engine in engines.items():
            note(job,'coupled '+label+' '+take)
            state,start,projection=t.cable_state(engine.physics)
            pred=engine.predict(t.initial_pose(engine.drone.parameters),state,torch.tensor(t.data['packets'][None],device='cuda',dtype=torch.float64),t.data['packet_time'],times,graph=True,hover_command=torch.tensor(t.data['hover_commands'][-1:],device='cuda',dtype=torch.float64))
            q=pred.get('cable_positions_m',pred.get('positions_m'))[0].cpu().numpy();p=pred['position_origin_m'][0].cpu().numpy()
            measured=t.measured(times)[0];coupled[take][label]=dict(drone=rms(p,measured),tip=rms(q[:,-1],truth[:,-1]),projection_m=projection,window=t.name)
            np.savez_compressed(out/(take+'-'+label+'-coupled.npz'),time_s=times,predicted_origin=p,predicted_cable=q,measured_origin=measured,measured_sites=truth)
            if label=='fitted':
                from .bootstrap_drone import TranslationBatch
                with torch.no_grad():reference,_=TranslationBatch([t],engine.drone.parameters).predict(engine.drone.residual)
                runtime=pose_prediction(t,engine.drone,t.time)
                error=float((reference[0]-runtime['position_origin_m'][0]).abs().max())
                if error>1e-9 or not bool(runtime['valid'].all()):raise AssertionError('Training/runtime drone mismatch')
                consistency[take]=dict(max_translation_difference_m=error)
    save(out/'coupled_diagnostics.json',dict(takes=coupled,evidence='Two-second representative motion, not full-flight/strike validation'))
    save(out/'execution_consistency.json',consistency)
    for path,digest in read(job/'protected_before.json').items():
        if sha256_file(path)!=digest:raise AssertionError('Original source bytes changed during fitting')
    model['provenance']['fit_complete']=True;save(out/'model.json',model)
    save(job/'status.json',dict(status='completed',candidate=str(out/'model.json'),evidence='Local preliminary identification; no prospective flight claim'))


def run(job):
    job=Path(job).resolve();lock=job/'worker.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    try:
        if read(job/'status.json')['status']!='prepared':raise ValueError('Only a freshly prepared job may start')
        torch.set_num_threads(4);torch.manual_seed(20260909)
        save(job/'status.json',dict(status='running',pid=os.getpid(),gpu=torch.cuda.get_device_name()))
        model=read(job/'source_candidate/model.json');windows=read(job/'windows.json');trials=[PreliminaryTrial(job,w,model) for w in windows]
        training=[t for t in trials if t.role=='training'];takes=sorted({t.take for t in training})
        for t in training:t.weights*=len(training)/(len(takes)*sum(x.take==t.take for x in training))
        engine=load_engine(job,trainable=True);settings=read(job/'protocol.json')['stopping']
        note(job,'preparing cable windows',drone_windows=len(trials))
        records,rejected=cable_windows(trials,engine.physics,[0.,1.],1.)
        mapping={t.name:t for t in trials}
        for r in records:r['take']=mapping[r['name']].take;r['role']=mapping[r['name']].role
        save(job/'cable_windows.json',dict(accepted=[{k:v for k,v in r.items() if k in ('name','take','role','cutoff','projection_m')} for r in records],rejected=rejected))
        fit_drone(job,training,engine,settings)
        fit_cable(job,[r for r in records if r['role']=='training'],engine,settings)
        publish(job,trials,records)
    except BaseException as exc:
        save(job/'status.json',dict(status='stopped' if isinstance(exc,InterruptedError) else 'failed',error=str(exc)));raise
    finally:lock.unlink(missing_ok=True)
