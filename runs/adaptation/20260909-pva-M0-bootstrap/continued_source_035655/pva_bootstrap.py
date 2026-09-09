"""Cold preliminary model from normalized adp0; no historical learned prior.

This is an in-sample local bootstrap, not prospective flight validation. All
losses use measured state from OptiTrack and actual held command receipts.
"""
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
import itertools
import json
import shutil
import time
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .current_adaptation import ROOT, BATCH, prepare, read, save, Trial
from .current_adaptation_fit import (load_engine, fit_nominal, WeightedTranslationBatch,
    pose_prediction, metric, cable_windows, join_windows, cable_forward, cable_objectives, note)
from .adaptation_attitude import ResidualAttitudeTrial, DOMAIN_ERROR
from .differentiable_fit import save_weights
from .io import sha256_file
from .plateau import Plateau
from simulator.cable import CableConfiguration
from simulator.cable.residual import MotionResidual
from simulator.drone_pose_residual import DronePoseResidual, save_residual
from simulator.drone_pose_response import PoseResponseParameters
from simulator.research_execution import ResearchExecutionModel


def cold_seed(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    # Structural measurements are retained, fitted coefficients and assets are not.
    original=read(ROOT/'config/research_30hz/model.json')
    cable=deepcopy(original['cable'])
    for key in ('parameter_source','previous_parameter_source'):cable.pop(key,None)
    cable.update(EI_n_m2=1e-6,Cb_n_m2_s=1e-6,external_drag_s_inv=0.,curvature_frame_regularization=2e-7)
    torch.manual_seed(20260909)
    drone=DronePoseResidual(hidden=16,acceleration_limit=.5,hover_gate=True).double()
    save_residual(folder/'drone_residual.pt',drone)
    parameters=PoseResponseParameters(10.,10.,5.,5.,1.,1.,.08,.02,
        attitude_drive_model='independent_scale_v3')
    save(folder/'drone_model.json',dict(nominal=dict(parameters=asdict(parameters),
        provenance='Cold engineering estimates, no learned parameters'),residual=dict(
        checkpoint='drone_residual.pt',sha256=sha256_file(folder/'drone_residual.pt'),specification=drone.specification())))
    residual=MotionResidual(CableConfiguration.from_mapping(cable).node_count,hidden=32,
        acceleration_limit=.5,mode='dissipative_plus_acceleration',damping_limit_s_inv=2.).double()
    save_weights(folder/'cable_residual.pt',residual)
    execution=deepcopy(original['fullstate_execution'])
    execution.update(checkpoint='drone_model.json',sha256=sha256_file(folder/'drone_model.json'),
        reference='bounded_jerk_pva_30hz_v1',source_job=None)
    model=dict(schema='loaded_drone_cable_pva_model_v1',cable=cable,
        fullstate_execution=execution,recorded_data=original['recorded_data'],
        simulation=original['simulation'],mass_measurement=original['mass_measurement'],
        motion_residual=dict(enabled=True,checkpoint='cable_residual.pt',sha256=sha256_file(folder/'cable_residual.pt'),
            specification=residual.specification(),drag_mode='nn_only'),
        provenance=dict(learned_prior=False,source_drone='cf_7',fit_state='hover_normalized_optitrack',
            geometry='Existing measured lengths and tracking-to-attachment convention preserved',
            mass_distribution='Provisional structural distribution retained; total 18 g measured',
            nominal_estimates='Kp=10, Kd=5, feedforward=1, attitude tau=.08 s, delay=.02 s, EI=Cb=1e-6',
            cable_boundary='Predicted rotated attachment; loaded drone response includes effective cable loading'))
    save(folder/'model.json',model)
    return folder/'model.json'


def prepare_job(job,batch=BATCH):
    job=Path(job).resolve()
    seed=cold_seed(job.parent/(job.name+'-cold-seed'))
    prepare(job,Path(batch),seed)
    protocol=read(job/'protocol.json')
    protocol.update(schema='normalized_adp0_cold_pva_bootstrap_v1',seed=20260909,
        training='All five current normalized cf7/adp0 flights; no old data or learned model prior',
        heldout='None. Training diagnostics only; next new flights provide prospective evaluation.',
        phases={'whip':[0,1]},fit_phase_weights={'whip':1.},
        stopping={'drone':dict(minimum=80,check_every=10,patience=5,relative=.005,ceiling=2000),
                  'physics':dict(minimum=24,check_every=6,patience=5,relative=.005,ceiling=180),
                  'cable':dict(minimum=24,check_every=6,patience=5,relative=.005,ceiling=240)},
        cable_window_starts_s=[0,.15,.30,.45,.60,.75,.85],cable_window_s=.12,
        nominal_prior='Cold explicit estimates; weak regularization, broad bounds',
        initialization_note='Drone NN has a smooth zero-at-rest gate; nominal measured-hover compensation retains the equilibrium',
        evidence='Local retrospective normalized training replay; not battery identification or new vehicle validation')
    save(job/'protocol.json',protocol)
    source=job/'code_snapshot';source.mkdir()
    for name in ('experimental_data','simulator'):
        shutil.copytree(ROOT/name,source/name,ignore=shutil.ignore_patterns('__pycache__'))
    save(job/'status.json',dict(status='prepared',stage='inputs frozen',created=datetime.now().isoformat()))
    return job


def selected_training(net,optimizer,loss_fn,selection,settings,folder,job,label,extra_state=None):
    stop=Plateau(settings['minimum'],settings['patience'],settings['relative'])
    score=selection();stop.observe(0,score)
    best=deepcopy(net.state_dict());best_update=0;history=[];reason='safety_ceiling'
    for update in range(1,settings['ceiling']+1):
        note(job,label,update=update,ceiling=settings['ceiling'])
        start=time.perf_counter();optimizer.zero_grad();loss=loss_fn()
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite '+label+' loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();row=dict(update=update,loss=float(loss.detach()),gradient_norm=float(norm),seconds=time.perf_counter()-start)
        converged=False
        if update%settings['check_every']==0 or update==settings['ceiling']:
            score=selection();improved,converged=stop.observe(update,score)
            if improved:best=deepcopy(net.state_dict());best_update=update
            row.update(selection_loss=score,best_loss=stop.best,plateau_checks=stop.stale)
            # Durable best and optimizer state; a crash is never silently resumed.
            temp=folder/'progress.tmp'
            torch.save(dict(update=update,current=net.state_dict(),optimizer=optimizer.state_dict(),best=best,
                best_update=best_update,stopping=asdict(stop),extra=extra_state),temp)
            temp.replace(folder/'progress.pt')
        history.append(row);save(folder/'history.json',history)
        if converged:reason='plateau';break
    net.load_state_dict(best)
    result=dict(updates=update,selected_update=best_update,selection_loss=stop.best,stop_reason=reason,
        converged=reason=='plateau',evidence='training-only selection')
    save(folder/'stopping.json',result);return result


def fit_drone(job,trials,engine,settings):
    folder=job/'drone';folder.mkdir()
    params=fit_nominal(trials,engine.drone.parameters,folder,job,cold_start=True)
    net=engine.drone.residual.cpu();batch=WeightedTranslationBatch(trials,params,device='cpu')
    optimizer=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=1e-4)
    def select():
        with torch.no_grad():return float(batch.loss(net))
    result=selected_training(net,optimizer,lambda:batch.loss(net),select,settings['drone'],folder,job,'drone residual')
    net.requires_grad_(False)
    # Refit attitude only after translation and its residual are frozen.
    cached=[ResidualAttitudeTrial(t,params,net) for t in trials]
    initial=np.log([params.attitude_acceleration_scale_xy,params.attitude_acceleration_scale_z,params.attitude_time_constant_s])
    def attitude(x):
        rows=[]
        try:
            for t,c in zip(trials,cached):
                p=c.predict(np.exp(x));valid=t.weights>0
                error=Rotation.from_matrix(p[valid].transpose(0,2,1)@c.truth[valid]).as_rotvec()
                rows.append((error*np.sqrt(t.weights[valid,None]/len(trials))/.15).ravel())
        except ValueError as exc:
            if str(exc) not in (DOMAIN_ERROR,'Desired specific-force direction is undefined','Desired heading and specific-force direction are singular'):raise
            return np.full(sum(int((t.weights>0).sum())*3 for t in trials)+3,100.)
        rows.append(np.sqrt(.01)*(x-np.log([1.,1.,.08])))
        return np.concatenate(rows)
    fit=least_squares(attitude,initial,bounds=(np.log([.1,.05,.02]),np.log([3.,1.5,.3])),
        max_nfev=100,diff_step=1e-4,ftol=1e-6,xtol=1e-6,gtol=1e-6)
    chosen=fit.x if np.linalg.norm(attitude(fit.x))<np.linalg.norm(attitude(initial)) else initial
    sx,sz,tau=np.exp(chosen);params=replace(params,attitude_acceleration_scale_xy=sx,attitude_acceleration_scale_z=sz,attitude_time_constant_s=tau)
    save(folder/'attitude_refinement.json',dict(initial=np.exp(initial),selected=np.exp(chosen),success=bool(fit.success),nfev=fit.nfev))
    save_residual(folder/'drone_residual.pt',net)
    save(folder/'drone_model.json',dict(nominal=dict(schema='nominal_loaded_drone_pose_candidate_v3',parameters=asdict(params),
        training_takes=[t.name for t in trials],geometry=trials[0].offset,adaptation_job=str(job),cold_start=True),
        residual=dict(schema='drone_pose_residual_v1',checkpoint='drone_residual.pt',sha256=sha256_file(folder/'drone_residual.pt'),
            specification=net.specification(),target='Bounded motion-dependent origin acceleration; nominal hover equilibrium preserved')))
    engine.drone.parameters=params;net.to('cuda')
    runtime={}
    for t in trials:
        p=pose_prediction(t,engine.drone,t.time)['position_origin_m'][0].cpu().numpy()
        reference=WeightedTranslationBatch([t],params).predict(net)[0][0].detach().cpu().numpy()
        difference=float(np.max(np.abs(p-reference)))
        if difference>2e-5:raise AssertionError(f'Drone fit/execution mismatch {difference}')
        runtime[t.name]=dict(position=metric(p,t.truth['position_origin_m'],t.time),max_runtime_difference_m=difference)
    save(folder/'result.json',dict(stopping=result,training=runtime))


class PhysicalParameters(torch.nn.Module):
    def __init__(self,value):
        super().__init__();self.raw=torch.nn.Parameter(torch.as_tensor(np.log(value),device='cuda',dtype=torch.float64))
    def forward(self):
        return self.raw.clamp(self.raw.new_tensor(np.log([1e-9,1e-9])),self.raw.new_tensor(np.log([1e-4,1e-3]))).exp()


def fit_cable(job,trials,engine,settings,*,short_residual=False):
    folder=job/'cable';folder.mkdir(exist_ok=True)
    short,rejected=cable_windows(trials,engine.physics,[0,.15,.30,.45,.60,.75,.85],.12)
    long,missing=cable_windows(trials,engine.physics,[0.],1.02)
    if len(long)!=len(trials) or not short:raise ValueError('Every flight must cover the whip')
    save(folder/'windows.json',dict(short=[{k:r[k] for k in ('name','cutoff','projection_m')} for r in short],rejected=rejected+missing))
    sd,ld=join_windows(short),join_windows(long);indices=list(engine.cable.marker_node_indices[1:])
    # Explicit broad physical search, not scaled around an old fitted model.
    candidates=np.array(list(itertools.product(np.logspace(-8,-5,4),np.logspace(-8,-4,5))))
    expanded={k:v.repeat_interleave(len(candidates),0) for k,v in ld.items()}
    parameters=torch.tensor(np.tile(candidates,(len(long),1)).T,device='cuda',dtype=torch.float64)
    note(job,'cold cable physical grid',candidates=len(candidates))
    q,_=cable_forward(engine,expanded,parameters)
    values=cable_objectives(q,expanded['truth'],indices).reshape(len(long),-1).mean(0).detach().cpu().numpy()
    chosen=candidates[np.argmin(values)];save(folder/'grid.json',dict(candidates=candidates,scores=values,selected=chosen))
    phys_dir=folder/'physics'
    if phys_dir.exists():
        archived=folder/('physics-autograd-stopped-'+datetime.now().strftime('%H%M%S'))
        if not phys_dir.resolve().is_relative_to(job.resolve()) or not archived.resolve().is_relative_to(job.resolve()):raise ValueError('Fit archive must stay inside its job')
        phys_dir.rename(archived)
    phys_dir.mkdir();history=[];engine.physics.motion_residual.requires_grad_(False)
    # With only TWO unknown physical coefficients, finite-difference least
    # squares on the identical full-whip forward model is substantially cheaper
    # than retaining the whole DDER autograd graph. NN fitting below stays BPTT.
    marker_weights=np.full(len(indices),.7/len(indices));marker_weights[-1]+=.3
    scale=torch.tensor(marker_weights/(len(long)*(ld['truth'].shape[1]-1)),device='cuda',dtype=torch.float64).sqrt()
    def physical_residual(log_values):
        note(job,'cable physics least squares',evaluation=len(history)+1)
        start=time.perf_counter()
        q,_=cable_forward(engine,ld,np.exp(log_values))
        error=q[:,1:,indices]-ld['truth'][:,1:]
        squared=error.square().sum(-1)/.05**2
        robust=2*(torch.sqrt(1+squared)-1)
        result=np.r_[(robust.clamp_min(0).sqrt()*scale).detach().cpu().numpy().ravel(),.01*(log_values-np.log(1e-6))]
        if not np.isfinite(result).all():raise FloatingPointError('Nonfinite cable physics residual')
        history.append(dict(update=len(history)+1,loss=float(result@result),parameters=np.exp(log_values),seconds=time.perf_counter()-start))
        save(phys_dir/'history.json',history);return result
    fit=least_squares(physical_residual,np.log(chosen),bounds=(np.log([1e-9,1e-9]),np.log([1e-4,1e-3])),
        diff_step=.002,max_nfev=100,ftol=5e-4,xtol=1e-5,gtol=1e-5)
    chosen=torch.tensor(np.exp(fit.x),device='cuda',dtype=torch.float64)
    save(phys_dir/'parameters.json',dict(EI_n_m2=float(chosen[0]),Cb_n_m2_s=float(chosen[1]),
        bounds=[[1e-9,1e-9],[1e-4,1e-3]],active_bounds=fit.active_mask.tolist(),success=bool(fit.success),
        stop_reason=str(fit.message),nfev=fit.nfev,objective=float(2*fit.cost),method='bounded finite-difference least squares on full-whip DDER rollout'))
    save(phys_dir/'stopping.json',dict(converged=bool(fit.success),stop_reason=str(fit.message),evaluations=len(history)))
    engine.physics.motion_residual.requires_grad_(True)
    if not short_residual:
        return fit_cable_full_whip(job,trials,engine)
    net=engine.physics.motion_residual;res_dir=folder/'residual';res_dir.mkdir()
    # Equal weight per flight, even when some short windows were excluded.
    sw=np.array([1/sum(s['name']==r['name'] for s in short)/len(trials) for r in short]);sw=torch.tensor(sw,device='cuda')
    def objective(data,weights,gradients):
        q,v=cable_forward(engine,data,chosen,gradients=gradients)
        _,extra=net.components(q.flatten(0,1),v.flatten(0,1))
        per_extra=extra.square().reshape(len(q),-1).mean(-1)/.5**2
        return (cable_objectives(q,data['truth'],indices)*weights).sum()+.01*(per_extra*weights).sum()
    lw=sw.new_full((len(long),),1/len(long))
    def select():
        with torch.no_grad():return float(objective(ld,lw,False))
    result=selected_training(net,torch.optim.Adam(net.parameters(),lr=.002,weight_decay=1e-4),
        lambda:objective(sd,sw,True),select,settings['cable'],res_dir,job,'cable residual')
    save_weights(folder/'cable_residual.pt',net);net.requires_grad_(False)
    save(folder/'result.json',dict(stopping=result,selected_physics=chosen.cpu().numpy(),specification=net.specification()))


def publish_and_check(job):
    out=job/'candidate';out.mkdir()
    model=read(job/'source_candidate/model.json')
    for name in ('drone_model.json','drone_residual.pt'):shutil.copy2(job/'drone'/name,out/name)
    shutil.copy2(job/'cable/cable_residual.pt',out/'cable_residual.pt')
    result=read(job/'cable/result.json');ei,cb=result['selected_physics']
    model['cable'].update(EI_n_m2=ei,Cb_n_m2_s=cb)
    model['motion_residual'].update(checkpoint='cable_residual.pt',sha256=sha256_file(out/'cable_residual.pt'),specification=result['specification'])
    model['fullstate_execution'].update(checkpoint='drone_model.json',sha256=sha256_file(out/'drone_model.json'),source_job=str(job))
    model['provenance'].update(label='M0 · normalized adp0 bootstrap',fit_job=str(job),fit_complete=False,
        prospective_flight_evidence=False,training_takes=sorted(p.name for p in (job/'inputs').iterdir()))
    save(out/'model.json',model)
    engine=ResearchExecutionModel.from_mapping(model,root=out,device='cuda')
    results={}
    for name in model['provenance']['training_takes']:
        t=Trial(job,name,model,end=1.);times=t.grid();state,start,projection=t.cable_state(engine.physics)
        prediction=engine.predict(t.initial_pose(engine.drone.parameters),state,
            torch.tensor(t.data['packets'][None],device='cuda',dtype=torch.float64),t.data['packet_time'],times,
            graph=True,hover_command=torch.tensor(t.data['hover_commands'][-1:],device='cuda',dtype=torch.float64))
        p,_,sites=t.measured(times)
        q=prediction['cable_positions_m'][0].cpu().numpy() if 'cable_positions_m' in prediction else prediction['positions_m'][0].cpu().numpy()
        origin=prediction['position_origin_m'][0].cpu().numpy()
        results[name]=dict(drone=metric(origin,p,times)['whip'],tip=metric(q[:,-1],sites[:,-1],times)['whip'],projection_m=projection)
        np.savez_compressed(out/(name+'.npz'),time_s=times,predicted_origin=origin,predicted_cable=q,measured_origin=p,measured_sites=sites)
    save(out/'training_diagnostics.json',dict(evidence='In-sample normalized whip replay. No prospective claim.',takes=results))
    # Only source dataset is protected during this job: concurrent authorized UI/config work is independent.
    changed=[]
    for path,digest in read(job/'protected_before.json').items():
        if str(BATCH.resolve()).lower() in path.lower() and (not Path(path).exists() or sha256_file(path)!=digest):changed.append(path)
    if changed:raise AssertionError('Source flight data changed during fit: '+str(changed))
    model['provenance']['fit_complete']=True;save(out/'model.json',model)
    save(job/'status.json',dict(status='completed',candidate=str(out/'model.json'),completed=datetime.now().isoformat(),
        evidence='training diagnostics only'))
    return out/'model.json'


def fit_cable_full_whip(job,trials,engine):
    """Use the long rollout as the actual NN loss, not just checkpoint selection.

    This explicitly replaces a short-window descent that can accumulate worse
    trajectory error. The short-window attempt is preserved for the audit.
    """
    folder=job/'cable';output=folder/'residual_full_whip';output.mkdir()
    physical=read(folder/'physics/parameters.json')
    chosen=torch.tensor([physical['EI_n_m2'],physical['Cb_n_m2_s']],device='cuda',dtype=torch.float64)
    long,rejected=cable_windows(trials,engine.physics,[0.],1.02)
    if len(long)!=len(trials):raise ValueError('All training takes must cover the complete whip')
    data=join_windows(long);net=engine.physics.motion_residual
    prior_file=folder/'residual/progress.pt'
    if prior_file.exists():net.load_state_dict(torch.load(prior_file,map_location='cuda',weights_only=False)['best'])
    indices=list(engine.cable.marker_node_indices[1:])
    # Exact whip scoring; the final uniform integration samples only pad the
    # grid. Samples after 1 s and before onset receive zero loss.
    time_mask=torch.tensor(np.stack([(r['time'][1:]>=0)&(r['time'][1:]<=1.) for r in long]),device='cuda',dtype=torch.float64)
    time_weight=time_mask/time_mask.sum(1,keepdim=True)
    def objective(gradients):
        q,v=cable_forward(engine,data,chosen,gradients=gradients)
        error=q[:,1:,indices]-data['truth'][:,1:]
        robust=2*(torch.sqrt(1+error.square().sum(-1)/.05**2)-1)
        per_time=.7*robust.mean(-1)+.3*robust[:,:,-1]
        _,extra=net.components(q[:,1:].flatten(0,1),v[:,1:].flatten(0,1))
        penalty=extra.square().reshape(len(q),len(time_mask[0]),-1).mean(-1)/.5**2
        return ((per_time+.01*penalty)*time_weight).sum(1).mean()
    def selection():
        with torch.no_grad():return float(objective(False))
    settings=dict(minimum=12,check_every=3,patience=4,relative=.005,ceiling=120)
    protocol=read(job/'protocol.json');protocol.update(cable_fit='Measured attachment; full-whip BPTT and same full-whip selection loss, exactly 0–1 s; short-window attempt retained but superseded')
    protocol['stopping']['cable_full_whip']=settings;save(job/'protocol.json',protocol)
    selected=selected_training(net,torch.optim.Adam(net.parameters(),lr=.001,weight_decay=1e-4),
        lambda:objective(True),selection,settings,output,job,'full-whip cable residual')
    save_weights(folder/'cable_residual.pt',net);net.requires_grad_(False)
    save(folder/'result.json',dict(stopping=selected,selected_physics=chosen.cpu().numpy(),specification=net.specification(),
        objective='Full recursive 0–1 s whip, measured attachment; fresh Adam, best earlier short-window candidate initialization'))


def run(job,*,continue_after_drone=False,continue_cable_full_whip=False):
    job=Path(job);lock=job/'worker.lock'
    with lock.open('x') as stream:stream.write(str(__import__('os').getpid()))
    try:
        prior_status=read(job/'status.json')['status']
        if continue_after_drone or continue_cable_full_whip:
            if prior_status not in ('stopped','failed') or not (job/'drone/result.json').exists():raise ValueError('Explicit continuation requires a stopped job and completed drone fit')
            if (job/'STOP').exists():(job/'STOP').unlink()
            snapshot=job/('continued_source_'+datetime.now().strftime('%H%M%S'));snapshot.mkdir()
            for name in ('pva_bootstrap.py','current_adaptation_fit.py'):shutil.copy2(ROOT/'experimental_data'/name,snapshot/name)
            protocol=read(job/'protocol.json');protocol['cable_physics_optimizer']='Bounded finite-difference least squares, full-whip objective, ftol .0005, xtol/gtol .00001, safety nfev100. Replaces slow full-whip autograd; stopped attempts preserved.';save(job/'protocol.json',protocol)
        elif prior_status!='prepared':raise ValueError('Only prepared jobs can start; never silently resume')
        save(job/'status.json',dict(status='running',pid=__import__('os').getpid()))
        torch.set_num_threads(4);torch.manual_seed(20260909)
        model=read(job/'source_candidate/model.json');settings=read(job/'protocol.json')['stopping']
        trials=[Trial(job,p.name,model,end=1.) for p in sorted((job/'inputs').iterdir())]
        for trial in trials:
            valid=trial.masks['fit_position']&(trial.time<=1.)
            trial.weights=valid.astype(float)/valid.sum()
        engine=load_engine(job,trainable=True)
        if not (continue_after_drone or continue_cable_full_whip):fit_drone(job,trials,engine,settings)
        if continue_cable_full_whip:fit_cable_full_whip(job,trials,engine)
        else:fit_cable(job,trials,engine,settings)
        publish_and_check(job)
    except BaseException as exc:
        save(job/'status.json',dict(status='stopped' if isinstance(exc,InterruptedError) else 'failed',error=str(exc)))
        raise
    finally:lock.unlink(missing_ok=True)
