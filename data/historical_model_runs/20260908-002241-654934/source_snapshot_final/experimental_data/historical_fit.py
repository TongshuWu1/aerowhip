"""Staged all-historical identification, with whole-take development checks.

Run --prepare, then --job PATH --stage cable or drone. Never starts PPO or
changes the active model. Outputs an explicit reviewable versioned candidate.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import time
import shutil
import numpy as np
import torch
from scipy.optimize import least_squares

from .historical_dataset import prepare, read_inputs
from .io import atomic_json, sha256_file
from .force_dataset import _normalized_rotations, reconstruct_dder_nodes
from .state_initialization import causal_state
from .cable_fit import PreparedTake, contiguous_window_starts
from .constrained_geometry import fit_offset
from .constrained_identification import combine, population
from .differentiable_fit import rollout, prediction_loss, subset, save_weights
from .cable_residual_fit import gradient_check
from simulator.cable import CableConfiguration, DderModel
from simulator.cable.residual import MotionResidual
from simulator.drone_tracking import DroneTrackingResidual, predict_trajectory


def read(path):
    return json.loads(Path(path).read_text())


def progress(folder,label,**values):
    record=dict(label=label,**values)
    atomic_json(folder/'progress.json',record)
    print(json.dumps(record),flush=True)


def check_stop(job):
    if (job/'STOP_REQUESTED').exists():
        raise InterruptedError('Stopped by request. Active model and PPO unchanged.')


def choose_starts(a, valid, horizon, maximum, history=20):
    """Cover low/medium/high recorded motion; no candidate-error-based selection."""
    after=round(horizon/.01)
    starts=np.array(contiguous_window_starts(valid,horizon_steps=after+history,stride_steps=5),dtype=int)+history
    if not len(starts):return starts
    # Use the maximum observed displacement over the window, not initial speed:
    # this retains whip windows that start from a stationary settled cable.
    reference=a['markers'][:,-1]
    intensity=np.array([np.linalg.norm(reference[s:s+after+1]-reference[s],axis=1).max() for s in starts])
    order=np.argsort(intensity,kind='stable')
    groups=np.array_split(order,3)
    selected=[]
    for group in groups:
        if not len(group):continue
        ix=group[np.argsort(starts[group])]
        selected.extend(starts[ix[np.linspace(0,len(ix)-1,min(maximum//3,len(ix)),dtype=int)]])
    return np.array(sorted(set(selected)),dtype=int)


def fit_geometry(data, names, payload):
    groups={}
    for name in names:
        a=data[name];rotation,rv=_normalized_rotations(a['quaternion'])
        delta=a['markers'][:,0]-a['position']
        tangent=a['markers'][:,1]-a['markers'][:,0]
        norm=np.linalg.norm(tangent,axis=1)
        tangent=tangent/np.maximum(norm[:,None],1e-12)
        valid=a['cable_fit_valid']&rv&(norm>1e-5)
        root_speed=np.linalg.norm(np.gradient(a['position'],.01,axis=0),axis=1)
        relative_speed=np.linalg.norm(np.gradient(a['markers'][:,-1]-a['position'],.01,axis=0),axis=1)
        quiet=valid&(root_speed<.05)&(relative_speed<.1)
        body=np.einsum('tji,tj->ti',rotation,delta)
        bt=np.einsum('tji,tj->ti',rotation,tangent)
        groups[name]=dict(role='training',c1=body[valid],quiet_c1=body[quiet],quiet_tangent=bt[quiet])
    result=fit_offset(groups,height=-payload['recorded_data']['optitrack_to_attachment_offset_body_m'][2],
                      length=payload['cable']['marker_interval_lengths_m'][0])
    return result


def cable_takes(data,payload,horizon,maximum,device):
    cable=CableConfiguration.from_mapping(payload['cable'])
    model=DderModel(cable.dder_parameters(EI=payload['cable']['EI_n_m2'],Cb=payload['cable']['Cb_n_m2_s']))
    takes=[]
    for name,a in data.items():
        rotation,rv=_normalized_rotations(a['quaternion'])
        root=a['position']+np.einsum('tij,j->ti',rotation,payload['recorded_data']['optitrack_to_attachment_offset_body_m'])
        valid=a['cable_fit_valid']&rv
        nodes,node_valid=reconstruct_dder_nodes(root,valid,a['markers'],a['marker_valid']&valid[:,None],cable)
        starts=choose_starts(a,valid&node_valid.all(1),horizon,maximum)
        if not len(starts):continue
        hist=torch.tensor(np.stack([nodes[s-20:s+1] for s in starts]),dtype=torch.float64)
        with torch.no_grad():state=causal_state(hist,.01,model)
        steps=round(horizon/.01)
        truth=np.stack([a['markers'][s:s+steps+1] for s in starts])
        initial=float(np.sqrt(np.mean(np.sum((state.positions_m[:,cable.marker_node_indices[1:]].numpy()-truth[:,0])**2,axis=-1))))
        takes.append(PreparedTake(name,'training',state.positions_m.to(device),state.velocities_m_s.to(device),
            torch.tensor(np.stack([root[s:s+steps+1] for s in starts]),dtype=torch.float64,device=device),
            torch.tensor(truth,dtype=torch.float64,device=device),tuple(int(s) for s in starts),initial,.01))
    return cable,model,takes


@torch.no_grad()
def cable_metrics(takes,model,cable,params,path=None):
    joined=combine(takes)
    prediction=rollout(joined,model,cable,params)
    error=prediction[:,1:]-joined.measured_marker_positions_m[:,1:]
    if not torch.isfinite(error).all():raise ValueError('Nonfinite recursive cable evaluation')
    sq=error.square().sum(-1)
    result={};offset=0
    for take in takes:
        batch=sq[offset:offset+take.window_count]
        result[take.take_id]=dict(marker_rmse_m=float(batch.mean().sqrt()),tip_rmse_m=float(batch[:,:,-1].mean().sqrt()),
            objective=float((.002*(torch.sqrt(1+batch/.002**2)-1)).mean()),windows=take.window_count,
            initial_marker_rmse_m=take.initialization_marker_rmse_m,starts=list(take.starts))
        offset+=take.window_count
    if path is not None:
        np.savez_compressed(path,prediction=prediction.cpu().numpy(),measured=joined.measured_marker_positions_m.cpu().numpy(),
            take=np.array([t.take_id for t in takes for _ in t.starts]),start_frame=np.array(joined.starts))
    return result


def mean_metric(metrics,names,key):
    values=[metrics[n][key] for n in names if n in metrics]
    if not values:raise ValueError('No usable data for requested evaluation group')
    return float(np.mean(values))


class BiasPopulation:
    def __init__(self,network,shifts,windows):
        self.network=network
        self.shifts=next(network.parameters()).new_tensor(shifts).repeat(windows)[:,None]
    def __call__(self,q,v):return self.network(q,v,output_bias_shift=self.shifts)
    def drag_rates(self,q):return q.new_zeros(q.shape[-2])


@torch.no_grad()
def calibrate_output_bias(training,model,cable,params,network,folder):
    """Tune NN output biases on full predictions, without unstable long BPTT.

This adjusts network weights, not an external damping parameter. All candidate
selection is fit-take-only. Raw bias offsets are dimensionless, not drag rates.
"""
    if network.mode!='dissipative':raise ValueError('Long output calibration expects the dissipative NN')
    records=[];center=0.
    labels={t.take_id:['balanced']*t.window_count for t in training}
    for shifts in ([-.75,-.25,0.,.25,.75,1.5],None):
        if shifts is None:shifts=[center-.125,center,center+.125]
        model.motion_residual=BiasPopulation(network,shifts,sum(t.window_count for t in training))
        scores,_=population(training,labels,model,cable,[params.tolist()]*len(shifts))
        center=float(shifts[int(np.argmin(scores))])
        records.append(dict(dimensionless_output_bias_shifts=shifts,training_objective=scores.tolist(),selected=center))
        atomic_json(folder/'output_bias_calibration.json',dict(passes=records,selection='Training-only two-second rollout',
            initialized_from_fixed_drag=False,external_drag_s_inv=0.))
    network.net[-1].bias.add_(center)
    model.motion_residual=network
    return center


def local_gradient_check(probe,model,cable,params,network):
    checks=[]
    for epsilon in (1e-3,3e-4,1e-4,3e-5,1e-5,3e-6,1e-6):
        check=gradient_check(probe,model,cable,params,network,epsilon=epsilon)
        checks.append(check)
        if len(checks)>=2 and all(c['passed'] for c in checks[-2:]):
            a,b=[c['finite_difference'] for c in checks[-2:]]
            if abs(a-b)/max(abs(a),abs(b),1e-10)<.05:
                return dict(passed=True,steps=checks,criterion='Two consecutive steps agree with autodiff and each other within 5%')
    return dict(passed=False,steps=checks,criterion='No converged pair within original 5% derivative tolerance')


def fit_cable(job, output_name='cable', reuse_directory=None, *, settings_override=None, folds=None, physics_only_reuse=False):
    settings=read(job/'protocol.json');data=read_inputs(job);original=read(job/'original_model.json')
    if settings_override:settings.update(settings_override)
    output=job/output_name;output.mkdir(exist_ok=False)
    atomic_json(output/'settings.json',settings)
    names=list(data);reviews=[]
    for fold,excluded in enumerate([*settings['folds'],[]]):
        check_stop(job)
        label=f'fold_{fold+1}' if excluded else 'final'
        if folds and label not in folds:continue
        folder=output/label
        reusable=job/reuse_directory/label if reuse_directory else None
        if reusable and (reusable/'review.json').exists() and not physics_only_reuse:
            shutil.copytree(reusable,folder)
            reviews.append(read(folder/'review.json'))
            atomic_json(folder/'reused_from.json',dict(source=str(reusable),reason='Completed fit unchanged; finite-difference checking amendment only'))
            progress(output,f'{label}: reusing completed unchanged fit')
            continue
        folder.mkdir()
        train=[n for n in names if n not in excluded]
        torch.manual_seed(settings['seed']);rng=np.random.default_rng(settings['seed'])
        candidate=deepcopy(original);candidate.pop('motion_residual',None)
        candidate['cable']['external_drag_s_inv']=0.
        geometry=fit_geometry(data,train,candidate)
        candidate['recorded_data']['optitrack_to_attachment_offset_body_m']=geometry['offset_body_m']
        atomic_json(folder/'geometry.json',geometry)
        progress(output,f'{label}: physical cable grid (zero external drag)',training=train,held_out=excluded)
        if reusable and (reusable/'physical_model.json').exists():
            saved=read(reusable/'physical_model.json')
            np.testing.assert_allclose(saved['recorded_data']['optitrack_to_attachment_offset_body_m'],geometry['offset_body_m'],atol=1e-12)
            candidate=saved
            chosen=np.array([candidate['cable'][k] for k in ('EI_n_m2','Cb_n_m2_s','external_drag_s_inv')])
            for file in ('physical_grid.json','physical_refinement.json'):
                shutil.copy2(reusable/file,folder/file)
        else:
            cable,model,takes=cable_takes(data,candidate,settings['cable_horizon_s'],settings['cable_windows_per_take'],'cuda')
            training=[t for t in takes if t.take_id in train]
            if len(training)!=len(train):raise ValueError('A take has no usable physical fitting windows; inspect audit')
            ei=sorted(set(settings['physical_EI_grid']+[candidate['cable']['EI_n_m2']]))
            cb=sorted(set(settings['physical_Cb_grid']+[candidate['cable']['Cb_n_m2_s']]))
            candidates=[[e,c,0.] for e in ei for c in cb]
            intensities={t.take_id:['balanced']*t.window_count for t in training}
            scores,_=population(training,intensities,model,cable,candidates)
            chosen=np.asarray(candidates[int(np.argmin(scores))])
            atomic_json(folder/'physical_grid.json',dict(candidates=candidates,training_objective=scores.tolist()))
            # One fixed local grid pass, chosen using only fit-take data.
            low=np.array([1e-9,1e-8]);high=np.array([2.8e-4,.02])
            refined=np.unique(np.array([[*np.clip(chosen[:2]*[e,c],low,high),0.] for e in (.25,1,4) for c in (.25,1,4)]),axis=0)
            scores,_=population(training,intensities,model,cable,refined.tolist())
            chosen=refined[int(np.argmin(scores))]
            candidate['cable'].update(EI_n_m2=float(chosen[0]),Cb_n_m2_s=float(chosen[1]))
            atomic_json(folder/'physical_refinement.json',dict(candidates=refined.tolist(),training_objective=scores.tolist(),selected=chosen.tolist(),
                interpretation='Effective constrained parameters; stiffness/internal damping may remain weakly identifiable'))
        atomic_json(folder/'physical_model.json',candidate)
        cable,model,short=cable_takes(data,candidate,.05,settings['cable_training_windows_per_take'],'cuda')
        _,_,long=cable_takes(data,candidate,settings['cable_selection_horizon_s'],settings['cable_windows_per_take'],'cuda')
        train_short=[t for t in short if t.take_id in train]
        train_long=[t for t in long if t.take_id in train]
        params=torch.tensor(chosen,dtype=torch.float64,device='cuda')
        if reusable and (reusable/'physical_evaluation.json').exists():
            initial_metrics=read(reusable/'physical_evaluation.json')
            shutil.copy2(reusable/'physical_predictions.npz',folder/'physical_predictions.npz')
        else:
            initial_metrics=cable_metrics(long,model,cable,params,folder/'physical_predictions.npz')
        atomic_json(folder/'physical_evaluation.json',initial_metrics)
        network=MotionResidual(cable.node_count,hidden=settings['cable_hidden'],acceleration_limit=settings['cable_acceleration_limit'],
            mode=settings.get('cable_mode','acceleration'),damping_limit_s_inv=settings.get('cable_damping_limit_s_inv',2.)).to(device='cuda',dtype=torch.float64)
        model.motion_residual=network
        probe=combine([subset(t,[0]) for t in train_short])
        check=local_gradient_check(probe,model,cable,params,network)
        atomic_json(folder/'gradient_check.json',check)
        if not check['passed']:raise ValueError(f'{label}: residual finite difference check failed')
        best=deepcopy(network.state_dict());best_update=0
        best_score=mean_metric(initial_metrics,train,'objective')
        optimizer=torch.optim.Adam(network.parameters(),lr=settings['cable_learning_rate'])
        history=[]
        for update in range(1,settings['cable_updates']+1):
            check_stop(job)
            batch=combine([subset(t,rng.integers(0,t.window_count,size=2)) for t in train_short])
            optimizer.zero_grad()
            pred=rollout(batch,model,cable,params,gradients=True)
            penalty=network(batch.initial_positions_m,batch.initial_velocities_m_s).square().mean()
            loss=prediction_loss(pred,batch.measured_marker_positions_m)+settings['cable_regularization']*penalty
            if not torch.isfinite(loss):raise ValueError('Nonfinite cable NN loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(network.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            row=dict(update=update,loss=float(loss.detach()),gradient_norm=float(norm))
            if update%12==0 or update==settings['cable_updates']:
                metrics=cable_metrics(train_long,model,cable,params)
                score=mean_metric(metrics,train,'objective');row['selection_objective']=score
                if score<best_score:best,best_update,best_score=deepcopy(network.state_dict()),update,score
            history.append(row);atomic_json(folder/'history.json',history)
            progress(output,f'{label}: cable NN {update}/{settings["cable_updates"]}',**row)
        network.load_state_dict(best)
        if settings.get('calibrate_output_bias'):
            progress(output,f'{label}: calibrating NN output layer on two-second predictions')
            calibrate_output_bias(train_long,model,cable,params,network,folder)
        save_weights(folder/'cable_residual.pt',network)
        result=cable_metrics(long,model,cable,params,folder/'residual_predictions.npz')
        atomic_json(folder/'residual_evaluation.json',result)
        candidate['motion_residual']=dict(enabled=True,checkpoint=str((folder/'cable_residual.pt').resolve()),
            sha256=sha256_file(folder/'cable_residual.pt'),specification=network.specification(),drag_mode='nn_only')
        candidate['force_accounting']['aerodynamic_drag']='bounded_NN_acceleration_on_free_cable_nodes; no fixed external damping'
        atomic_json(folder/'candidate_model.json',candidate)
        review=dict(label=label,training=train,held_out=excluded,selected_update=best_update,
            trainable_parameters=sum(p.numel() for p in network.parameters()),physical_parameters=chosen.tolist(),
            all_data_final_fit=not bool(excluded),independent_test=False)
        if excluded:
            review['held_out']={key:dict(physics=mean_metric(initial_metrics,excluded,key),residual=mean_metric(result,excluded,key))
                               for key in ('marker_rmse_m','tip_rmse_m')}
        atomic_json(folder/'review.json',review);reviews.append(review)
    atomic_json(output/'review.json',dict(status='FITTED_REQUIRES_COMBINED_REVIEW',folds=reviews,
        no_independent_test=True,active_model_changed=False,ppo_started=False))
    if not folds:atomic_json(job/'cable_result.json',dict(directory=output_name))
    progress(output,'Cable physics and NN fits complete; candidates preserved')


def drone_windows(data,settings):
    rows=[];count=round(settings['drone_horizon_s']/.01)
    for name,a in data.items():
        starts=choose_starts(a,a['drone_fit_valid'],settings['drone_horizon_s'],settings['drone_windows_per_take'],history=25)
        for s in starts:
            # Initial velocity depends on measured past only.
            times=np.arange(-10,1)*.01
            velocity=np.linalg.lstsq(np.column_stack((np.ones(11),times,times**2)),a['position'][s-10:s+1],rcond=None)[0][1]
            commands=a['commands'][s-20:s+count].copy()
            if 'world_attachment_offset' in a:
                # The offset at initialization is known. Future measured
                # orientation contributes only to the target attachment path.
                commands[:,:3]+=a['world_attachment_offset'][s]
            rows.append(dict(name=name,start=int(s),position=a['position'][s:s+count+1],velocity=velocity,
                commands=commands,time=a['time'][s:s+count+1]))
    return rows


def drone_arrays(rows,delay,device=None):
    lag=round(delay/.01);count=len(rows[0]['position'])-1
    args=(np.stack([r['position'][0] for r in rows]),np.stack([r['velocity'] for r in rows]),
          np.stack([r['commands'][20-lag:20-lag+count] for r in rows]),
          np.stack([r['commands'][15-lag:15-lag+count] for r in rows]),np.full((len(rows),count),.01))
    truth=np.stack([r['position'] for r in rows])
    if device is not None:
        args=tuple(torch.tensor(x,dtype=torch.float64,device=device) for x in args)
        truth=torch.tensor(truth,dtype=torch.float64,device=device)
    return args,truth


def numpy_drone(args,gains):
    p,v,commands,_,dt=args;p=p.copy();v=v.copy();result=[p.copy()]
    for i in range(commands.shape[1]):
        cmd=commands[:,i]
        acc=gains[:3]*(cmd[:,:3]-p)+gains[3:6]*(cmd[:,3:6]-v)+gains[6:9]*cmd[:,6:9]
        p=p+.01*v+.00005*acc;v=v+.01*acc;result.append(p.copy())
    return np.stack(result,axis=1)


def drone_metrics(pred,truth,rows):
    result={}
    for name in sorted({r['name'] for r in rows}):
        ix=[i for i,r in enumerate(rows) if r['name']==name]
        d=pred[ix,1:]-truth[ix,1:]
        result[name]=dict(position_rmse_m=float(np.sqrt(np.mean(np.sum(d*d,axis=-1)))),
            maximum_position_error_m=float(np.linalg.norm(d,axis=-1).max()),windows=len(ix))
    return result


def fit_drone(job, *, attachment=False):
    settings=read(job/'protocol.json');data=read_inputs(job);rows=drone_windows(data,settings)
    output=job/('drone_attachment' if attachment else 'drone');output.mkdir(exist_ok=False)
    atomic_json(output/'windows.json',[dict(take=r['name'],start=r['start']) for r in rows])
    names=list(data);reviews=[]
    if {r['name'] for r in rows}!=set(names):raise ValueError('Some takes lack a valid full-state/drone fitting window')
    for fold,excluded in enumerate([*settings['folds'],[]]):
        check_stop(job);label=f'fold_{fold+1}' if excluded else 'final'
        folder=output/label;folder.mkdir()
        attachment_geometry=None
        if attachment:
            attachment_geometry=fit_geometry(data,[n for n in names if n not in excluded],read(job/'original_model.json'))
            attachment_data={}
            for name,a in data.items():
                rotation,rv=_normalized_rotations(a['quaternion'])
                offset=np.einsum('tij,j->ti',rotation,attachment_geometry['offset_body_m'])
                attachment_data[name]=dict(a,position=a['position']+offset,world_attachment_offset=offset,
                    drone_fit_valid=a['drone_fit_valid']&rv)
            rows=drone_windows(attachment_data,settings)
            atomic_json(folder/'attachment_geometry.json',attachment_geometry)
            atomic_json(folder/'windows.json',[dict(take=r['name'],start=r['start']) for r in rows])
        train=[r for r in rows if r['name'] not in excluded]
        # Equal take contribution even if quality masks yield fewer windows.
        weights=np.array([1/sum(x['name']==r['name'] for x in train) for r in train])
        weights=weights/weights.sum()
        candidates=[]
        progress(output,f'{label}: fitting effective full-state response and delay')
        for delay in settings['drone_delay_candidates_s']:
            args,truth=drone_arrays(train,delay)
            def objective(gains):
                return ((numpy_drone(args,gains)[:,1:]-truth[:,1:])*np.sqrt(weights[:,None,None])).ravel()
            result=least_squares(objective,[4.]*3+[3.]*3+[1.]*3,bounds=settings['nominal_gain_bounds'],
                                 loss='soft_l1',f_scale=.02/np.sqrt(len(train)),max_nfev=100)
            candidates.append(dict(delay_s=delay,gains=result.x.tolist(),training_mse=float(np.mean(objective(result.x)**2))))
        nominal=min(candidates,key=lambda x:x['training_mse'])
        atomic_json(folder/'nominal.json',dict(selected=nominal,candidates=candidates,interpretation='Effective full-state response; not motor or firmware parameters'))
        torch.manual_seed(settings['seed'])
        net=DroneTrackingResidual(settings['drone_hidden'],settings['drone_acceleration_limit']).to(device='cuda',dtype=torch.float64)
        gains=torch.tensor(nominal['gains'],device='cuda',dtype=torch.float64)
        args,truth=drone_arrays(train,nominal['delay_s'],'cuda')
        w=truth.new_tensor(weights)
        def objective_torch(p):return (((p[:,1:]-truth[:,1:]).square().mean((1,2)))*w).sum()
        with torch.no_grad():initial=predict_trajectory(*args,gains)[0];best_loss=float(objective_torch(initial))
        best=deepcopy(net.state_dict());best_update=0
        optimizer=torch.optim.Adam(net.parameters(),lr=settings['drone_learning_rate'])
        history=[]
        for update in range(1,settings['drone_updates']+1):
            check_stop(job);optimizer.zero_grad()
            pred,v,_=predict_trajectory(*args,gains,net)
            correction=net(pred[:,:-1],v[:,:-1],args[2],args[3])
            loss=objective_torch(pred)+settings['drone_regularization']*(correction.square().mean((1,2))*w).sum()
            if not torch.isfinite(loss):raise ValueError('Nonfinite drone NN loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            if update%20==0 or update==settings['drone_updates']:
                with torch.no_grad():score=float(objective_torch(predict_trajectory(*args,gains,net)[0]))
                if score<best_loss:best_loss,best_update,best=score,update,deepcopy(net.state_dict())
                history.append(dict(update=update,training_mse=score));atomic_json(folder/'history.json',history)
                progress(output,f'{label}: drone NN {update}/{settings["drone_updates"]}',training_mse=score)
        net.load_state_dict(best)
        checkpoint=dict(schema='effective_fullstate_drone_residual_v1',specification=net.specification(),
            state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},nominal=nominal,history_s=.05,
            reference_point='cable_attachment' if attachment else 'OptiTrack cf_7 origin',flight_ready=False)
        if attachment:
            checkpoint.update(attachment_offset_body_m=attachment_geometry['offset_body_m'],
                command_position_transform='logged_cf7_command_plus_initial_world_attachment_offset',
                interpretation='Effective attachment tracking, includes loaded drone response and changing attachment orientation; no second cable reaction')
        torch.save(checkpoint,folder/'drone_residual.pt')
        args,truth=drone_arrays(rows,nominal['delay_s'],'cuda')
        with torch.no_grad():
            baseline=predict_trajectory(*args,gains)[0].cpu().numpy()
            predicted=predict_trajectory(*args,gains,net)[0].cpu().numpy()
        truth=truth.cpu().numpy()
        metrics=dict(nominal=drone_metrics(baseline,truth,rows),residual=drone_metrics(predicted,truth,rows))
        atomic_json(folder/'evaluation.json',metrics)
        np.savez_compressed(folder/'predictions.npz',nominal=baseline,residual=predicted,measured=truth,
            take=np.array([r['name'] for r in rows]),start=np.array([r['start'] for r in rows]))
        review=dict(label=label,held_out=excluded,selected_update=best_update,delay_s=nominal['delay_s'],
            checksum=sha256_file(folder/'drone_residual.pt'),all_data_final_fit=not bool(excluded),independent_test=False)
        atomic_json(folder/'review.json',review);reviews.append(review)
    atomic_json(output/'review.json',dict(status='FITTED_REQUIRES_COMBINED_REVIEW',folds=reviews,
        no_independent_test=True,active_model_changed=False,ppo_started=False))
    progress(output,'Drone response and NN fits complete; candidates preserved')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--job',type=Path);parser.add_argument('--stage',choices=('cable','drone','attachment'))
    parser.add_argument('--output-name',default='cable')
    parser.add_argument('--reuse-directory')
    parser.add_argument('--settings-override',type=Path)
    parser.add_argument('--fold',action='append')
    parser.add_argument('--physics-only-reuse',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        print(prepare(Path(__file__).resolve().parents[1]),flush=True);return
    if args.job is None or args.stage is None:parser.error('--job and --stage are required')
    torch.set_num_threads(1)
    if not torch.cuda.is_available():raise ValueError('CUDA required for this explicitly configured fit')
    status=args.job/f'{args.stage}_status.json'
    atomic_json(status,dict(status='RUNNING',hardware=torch.cuda.get_device_name(),started=time.time()))
    try:
        if args.stage=='cable':fit_cable(args.job,args.output_name,args.reuse_directory,
            settings_override=read(args.settings_override) if args.settings_override else None,
            folds=args.fold,physics_only_reuse=args.physics_only_reuse)
        else:fit_drone(args.job,attachment=args.stage=='attachment')
    except BaseException as error:
        atomic_json(status,dict(status='STOPPED' if isinstance(error,InterruptedError) else 'FAILED',error=str(error)))
        raise
    atomic_json(status,dict(status='COMPLETED',hardware=torch.cuda.get_device_name(),finished=time.time()))


if __name__=='__main__':main()
