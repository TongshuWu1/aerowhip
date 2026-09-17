"""Initial drone tracking fit using explicitly authorized historical whip trials.

Uses received/logged full-state commands, including the early measured-position
hold. Does not substitute the intended CSV tail, infer thrust, train a cable
residual, or automatically install the result in the simulator.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.optimize import least_squares
import torch

from simulator.drone_tracking import DroneTrackingResidual, predict_trajectory
from simulator.workflow import read_json, stamp
from .io import atomic_json, canonical_json_hash, sha256_file

TRIALS = {'whip1_001':'training', 'whip1_002':'training', 'whip1_003':'validation'}
SETTINGS = dict(seed=1729, hidden=32, acceleration_limit=2., updates=160,
                learning_rate=.001, penalty=.001, history_s=.05,
                delay_candidates_s=[0.,.02,.04,.06,.08,.10],
                interval_s=[-.2,1.2], device='cuda')


def held_commands(t, native_time, native_commands, native_valid, query):
    indices = np.searchsorted(native_time, query, side='right')-1
    if (indices<0).any() or (query>native_time[-1]).any():
        raise ValueError('Command history does not cover the requested interval.')
    if not native_valid[indices].all() or not np.isfinite(native_commands[indices,:9]).all():
        raise ValueError('Missing logged commands inside the fitted interval.')
    if (query-native_time[indices]>.10).any():
        raise ValueError('Gap in the native command log.')
    return native_commands[indices,:9]


def load_trial(path, name, role, settings):
    quality = read_json(path/'quality.json')
    if quality.get('output_files',{}).get('dataset.npz') != sha256_file(path/'dataset.npz'):
        raise ValueError('Processed recording changed since preparation.')
    with np.load(path/'dataset.npz',allow_pickle=False) as data:
        t=data['controller_time_s']; position=data['drone_position_m']
        valid=data['drone_position_valid']; ct=data['controller_native_time_s']
        commands=data['controller_native_fullstate']; command_valid=data['controller_native_valid']
        ref_valid=data['reference_valid']
    active=np.flatnonzero(command_valid & (np.linalg.norm(np.nan_to_num(commands[:,3:9]),axis=1)>1e-3))
    if not len(active):
        raise ValueError('No maneuver command onset.')
    onset=ct[active[0]]
    indices=np.flatnonzero((t>=onset+settings['interval_s'][0]) & (t<=onset+settings['interval_s'][1]))
    if len(indices)<100 or not valid[indices].all() or not ref_valid[indices].all():
        raise ValueError('Need continuous drone motion and command coverage.')
    start=indices[0]
    if start<10 or not valid[start-10:start+1].all():
        raise ValueError('Need pre-window position history for causal initialization.')
    if not np.isfinite(position[indices]).all() or not np.allclose(np.diff(t[indices]),.01,atol=1e-6):
        raise ValueError('Expected finite 100 Hz OptiTrack trajectory.')
    history_time=t[start-10:start+1]-t[start]
    velocity=np.linalg.lstsq(np.column_stack((np.ones(11),history_time,history_time**2)),
                            position[start-10:start+1],rcond=None)[0][1]
    return dict(name=name, role=role, time=t[indices]-onset, absolute_time=t[indices],
        measured_position=position[indices], initial_velocity=velocity,
        native_time=ct,native_commands=commands,native_valid=command_valid,
        alignment=quality['alignment'], source_sha256=sha256_file(path/'dataset.npz'))


def prepare_job(root):
    root=Path(root).resolve()
    source=root/'data/adaptation_rounds/adaptation0/processed/20260907-194802-922485'
    trials=[load_trial(source/name,name,role,SETTINGS) for name,role in TRIALS.items()]
    job=root/'data/drone_residual_runs'/stamp()
    job.mkdir(parents=True,exist_ok=False)
    atomic_json(job/'settings.json',SETTINGS)
    metadata=[]
    for trial in trials:
        arrays={k:v for k,v in trial.items() if isinstance(v,np.ndarray)}
        np.savez_compressed(job/f'{trial["name"]}.npz',**arrays)
        metadata.append({k:v for k,v in trial.items() if not isinstance(v,np.ndarray)})
    atomic_json(job/'dataset.json',dict(trials=metadata,
        source=str(source), authorization='User explicitly reauthorized yesterday’s whip trials for initial drone residual only',
        command_source='Logged full-state commands including early hold; not the intended CSV tail',
        reference_point='OptiTrack cf_7 origin; COM/body/attachment equivalence not assumed',
        limitations=['Approximate clock alignment includes logging latency',
            'Only repeated executions of one maneuver; validation is development-only',
            'Effective closed-loop response includes fixed cable loading and logging/interface effects',
            'No verified contact/intervention labels; provisional descriptive fit only']))
    hashes={}
    project=Path(__file__).resolve().parents[1]
    for name in ('simulator/drone_tracking.py','experimental_data/drone_residual_fit.py'):
        destination=job/'source_snapshot'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(project/name,destination)
        hashes[name]=sha256_file(destination)
    atomic_json(job/'provenance.json',dict(source_sha256=hashes,
        dataset_sha256=sha256_file(job/'dataset.json'),settings_sha256=sha256_file(job/'settings.json'),
        trial_sha256={t['name']:sha256_file(job/f'{t["name"]}.npz') for t in trials},
        active_model_sha256=canonical_json_hash(read_json(root/'config/model.json')),
        cable_residual_trained=False, active_model_changed=False))
    return job,[sys.executable,'-u','-m','experimental_data.drone_residual_fit','--job',str(job)]


def prepare_inputs(trial, delay, history):
    t=trial['absolute_time']
    values=(trial['native_time'],trial['native_commands'],trial['native_valid'])
    command=held_commands(t,*values,t[:-1]-delay)
    past=held_commands(t,*values,t[:-1]-delay-history)
    return command,past,np.diff(t)


def nominal_prediction(trial, gains, delay, history):
    commands,_,dt=prepare_inputs(trial,delay,history)
    position=trial['measured_position'][0].copy();velocity=trial['initial_velocity'].copy()
    values=[position.copy()]
    for command,step in zip(commands,dt):
        acceleration=gains[:3]*(command[:3]-position)+gains[3:6]*(command[3:6]-velocity)+gains[6:9]*command[6:9]
        position=position+step*velocity+.5*step*step*acceleration
        velocity=velocity+step*acceleration
        values.append(position.copy())
    return np.asarray(values)


def batch(trials,delay,settings,device):
    # Trim the final sample only when onset phase gives different row counts.
    count=min(len(t['absolute_time']) for t in trials)
    commands=[];past=[];dt=[]
    for trial in trials:
        u,h,d=prepare_inputs(trial,delay,settings['history_s'])
        commands.append(u[:count-1]);past.append(h[:count-1]);dt.append(d[:count-1])
    tensor=lambda v:torch.tensor(np.asarray(v),dtype=torch.float64,device=device)
    args=(tensor([t['measured_position'][0] for t in trials]),tensor([t['initial_velocity'] for t in trials]),
          tensor(commands),tensor(past),tensor(dt))
    return args,tensor([t['measured_position'][:count] for t in trials])


def position_metrics(predicted,truth):
    delta=predicted-truth
    return dict(position_rmse_m=float(np.sqrt(np.mean(np.sum(delta[1:]**2,axis=-1)))),
        maximum_position_error_m=float(np.linalg.norm(delta[1:],axis=-1).max()),
        endpoint_error_m=float(np.linalg.norm(delta[-1])),
        axis_rmse_m=np.sqrt(np.mean(delta[1:]**2,axis=0)).tolist())


def fit(job):
    job=Path(job).resolve();settings=read_json(job/'settings.json')
    meta=read_json(job/'dataset.json');provenance=read_json(job/'provenance.json')
    for filename,key in (('settings.json','settings_sha256'),('dataset.json','dataset_sha256')):
        if sha256_file(job/filename)!=provenance[key]:raise ValueError('Saved fitting inputs changed.')
    trials=[]
    for row in meta['trials']:
        path=job/f'{row["name"]}.npz'
        if sha256_file(path)!=provenance['trial_sha256'][row['name']]:raise ValueError('Trial snapshot changed.')
        with np.load(path,allow_pickle=False) as data:trial={k:data[k] for k in data.files}
        trials.append(dict(**trial,**row))
    training=[t for t in trials if t['role']=='training']
    if len(training)!=2 or len(trials)!=3:raise ValueError('Initial fit expects two training trials and one validation trial.')
    started=time.perf_counter()
    def progress(label,**extra):
        value=dict(label=label,elapsed_s=time.perf_counter()-started,**extra)
        atomic_json(job/'progress.json',value);print(json.dumps(value),flush=True)
    progress('Fitting effective tracking response on whip trials 1 and 2')
    initial=np.asarray([4.]*3+[3.]*3+[1.]*3)
    def objective(gains,delay):
        return np.concatenate([(nominal_prediction(t,gains,delay,settings['history_s'])[1:]-t['measured_position'][1:]).ravel()
                               for t in training])
    candidates=[]
    for delay in settings['delay_candidates_s']:
        fit_result=least_squares(objective,initial,args=(delay,),bounds=([.1]*6+[0.]*3,[80.]*3+[20.]*3+[2.]*3),
                                 loss='soft_l1',f_scale=.02,max_nfev=100)
        score=float(np.mean(objective(fit_result.x,delay)**2))
        candidates.append(dict(delay_s=delay,gains=fit_result.x.tolist(),training_mse=score))
    nominal=min(candidates,key=lambda c:c['training_mse'])
    atomic_json(job/'nominal.json',dict(selected=nominal,candidates=candidates,
        gain_order=['kp_x','kp_y','kp_z','kd_x','kd_y','kd_z','accel_gain_x','accel_gain_y','accel_gain_z'],
        interpretation='Effective response and delay; not firmware gains or identified actuator physics'))
    torch.set_num_threads(1);torch.manual_seed(settings['seed'])
    device=settings['device']
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA is unavailable.')
    network=DroneTrackingResidual(settings['hidden'],settings['acceleration_limit']).to(device=device,dtype=torch.float64)
    gains=torch.tensor(nominal['gains'],dtype=torch.float64,device=device)
    args,truth=batch(training,nominal['delay_s'],settings,device)
    with torch.no_grad():
        baseline=predict_trajectory(*args,gains)[0]
        best_loss=float((baseline[:,1:]-truth[:,1:]).square().mean())
    initial_loss=best_loss;best=deepcopy(network.state_dict());best_update=0
    optimizer=torch.optim.Adam(network.parameters(),lr=settings['learning_rate'])
    history=[]
    for update in range(1,settings['updates']+1):
        if (job/'STOP_REQUESTED').exists():raise InterruptedError('Drone residual stopped; simulator unchanged.')
        optimizer.zero_grad()
        predicted,velocity,_=predict_trajectory(*args,gains,network)
        residual=network(predicted[:,:-1],velocity[:,:-1],args[2],args[3])
        loss=(predicted[:,1:]-truth[:,1:]).square().mean()+settings['penalty']*residual.square().mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite training objective.')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(network.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if update%10==0 or update==settings['updates']:
            with torch.no_grad():
                p=predict_trajectory(*args,gains,network)[0]
                score=float((p[:,1:]-truth[:,1:]).square().mean())
            if score<best_loss:best_loss=score;best=deepcopy(network.state_dict());best_update=update
            history.append(dict(update=update,training_mse=score,regularized_loss=float(loss.detach()),gradient_norm=float(norm)))
            atomic_json(job/'history.json',history)
            progress(f'Drone residual: {update}/{settings["updates"]}',training_mse=score)
    network.load_state_dict(best)
    checkpoint=dict(schema='effective_fullstate_drone_residual_v1',specification=network.specification(),
        state_dict={k:v.detach().cpu() for k,v in network.state_dict().items()},nominal=nominal,
        history_s=settings['history_s'],reference_point=meta['reference_point'],flight_ready=False)
    torch.save(checkpoint,job/'drone_residual.pt')
    evaluation={}
    for trial in trials:
        args,truth=batch([trial],nominal['delay_s'],settings,device)
        with torch.no_grad():
            nominal_p=predict_trajectory(*args,gains)[0][0].cpu().numpy()
            residual_p=predict_trajectory(*args,gains,network)[0][0].cpu().numpy()
        actual=truth[0].cpu().numpy()
        evaluation[trial['name']]=dict(role=trial['role'],nominal=position_metrics(nominal_p,actual),
                                     residual=position_metrics(residual_p,actual))
        np.savez_compressed(job/f'{trial["name"]}_prediction.npz',time_s=trial['time'][:len(actual)],
            measured=actual,nominal=nominal_p,residual=residual_p)
    atomic_json(job/'evaluation.json',evaluation)
    validation=evaluation['whip1_003']
    improves=(validation['residual']['position_rmse_m']<validation['nominal']['position_rmse_m']
        and validation['residual']['maximum_position_error_m']<=validation['nominal']['maximum_position_error_m'])
    atomic_json(job/'review.json',dict(validation_improved=improves,selected_update=best_update,
        initial_training_mse=initial_loss,selected_training_mse=best_loss,
        checkpoint_sha256=sha256_file(job/'drone_residual.pt'),automatically_applied=False,
        status='PROVISIONAL_INITIAL_MODEL',validation='Trial 3 excluded from gain fitting and NN training; one repeated maneuver only',
        limitations=meta['limitations'],hardware=torch.cuda.get_device_name() if device=='cuda' else 'CPU'))
    plot(job,trials)
    progress('Complete: initial drone model saved; active simulation unchanged',validation_improved=improves)


def plot(job,trials):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure=Figure(figsize=(12,8),layout='constrained');FigureCanvasAgg(figure)
    axes=figure.subplots(3,3)
    for row,trial in enumerate(trials):
        with np.load(job/f'{trial["name"]}_prediction.npz') as data:
            for axis,label in enumerate('XYZ'):
                ax=axes[row,axis]
                for key,style in [('measured','-'),('nominal','--'),('residual',':')]:
                    ax.plot(data['time_s'],data[key][:,axis],style,label=key)
                ax.set(title=f'{trial["name"]} / {trial["role"]}: {label}',xlabel='Time from first maneuver command (s)',ylabel='Position (m)')
                ax.grid(alpha=.2)
    axes[0,0].legend()
    figure.savefig(job/'tracking_comparison.png',dpi=160)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True);args=parser.parse_args()
    atomic_json(args.job/'status.json',dict(status='RUNNING'))
    try:fit(args.job)
    except BaseException as error:
        atomic_json(args.job/'status.json',dict(status='FAILED',error=str(error)));raise
    atomic_json(args.job/'status.json',dict(status='COMPLETED'))


if __name__=='__main__':main()
