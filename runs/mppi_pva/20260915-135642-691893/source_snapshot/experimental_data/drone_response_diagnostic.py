"""Reviewed PVA-to-motion diagnostics, not actuator identification or a fit.

Native-pose derivatives are retrospective diagnostics only. Model probes use
the production recursive pose solver, fixed neural weights and past-only state.
No candidate is published, promoted or used to replace an original forecast.
"""
from copy import copy
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .io import atomic_json, sha256_file
from .whip_adaptation import WhipTrial, verify_hashes, rms_summary
from . import whip_adaptation_fit as fitting
from simulator.workflow import read_json


def local_kinematics(t, position, valid, *, half_width_s=.06, max_gap_s=.025):
    """Local cubic least squares on each contiguous valid native-pose segment.

    Symmetric windows require support on both sides; gaps and segment edges
    remain NaN. Time centering/scaling makes units explicit and improves conditioning.
    No missing value is filled and no contact-excluded sample should be supplied.
    """
    t=np.asarray(t,float); p=np.asarray(position,float); valid=np.asarray(valid,bool)
    if (t.ndim!=1 or len(t)<2 or p.shape!=(len(t),3) or valid.shape!=t.shape or
        not np.isfinite(t).all() or np.any(np.diff(t)<=0)):
        raise ValueError('Expected increasing native times, Nx3 positions and N validity flags')
    if not np.isfinite([half_width_s,max_gap_s]).all() or min(half_width_s,max_gap_s)<=0:
        raise ValueError('Positive derivative window and gap threshold required')
    valid=valid & np.isfinite(p).all(-1)
    v=np.full_like(p,np.nan); a=np.full_like(p,np.nan)
    indices=np.flatnonzero(valid)
    segments=np.split(indices,np.flatnonzero((np.diff(indices)!=1)|(np.diff(t[indices])>max_gap_s))+1)
    for segment in segments:
        if len(segment)<7: continue
        cadence=float(np.median(np.diff(t[segment])))
        # At most half a native interval of endpoint tolerance, never across gaps.
        tolerance=min(cadence*.51,half_width_s*.1)
        for i in segment:
            if t[i]-t[segment[0]]<half_width_s-tolerance or t[segment[-1]]-t[i]<half_width_s-tolerance: continue
            ids=segment[np.abs(t[segment]-t[i])<=half_width_s+1e-10]
            if len(ids)<7: continue
            x=(t[ids]-t[i])/half_width_s
            design=np.stack([np.ones_like(x),x,x*x,x*x*x],-1)
            weights=np.exp(-.5*(x/.6)**2)
            coefficients,_,rank,_=np.linalg.lstsq(design*weights[:,None]**.5,p[ids]*weights[:,None]**.5,rcond=None)
            if rank!=4: continue
            v[i]=coefficients[1]/half_width_s
            a[i]=2*coefficients[2]/half_width_s**2
    return v,a


def check_command_coverage(trial, times, delay):
    """Check endpoints and every held-command interval, including invalid gaps."""
    d=trial.data
    edges=np.unique(np.r_[times,d['packet_time']+delay,d['packet_valid_until']+delay])
    edges=edges[(edges>=times[0])&(edges<=times[-1])]
    for t in np.r_[edges,(edges[:-1]+edges[1:])*.5]: trial.schedule.sample(float(t-delay))


def sensitivity_summary(columns):
    """Dimensionless local finite-difference sensitivity; not a confidence bound."""
    j=np.asarray(columns,float)
    if j.ndim!=2 or min(j.shape)<1 or not np.isfinite(j).all():
        raise ValueError('Finite nonempty sensitivity matrix required')
    norms=np.linalg.norm(j,axis=0)
    unit=np.divide(j,norms[None],out=np.zeros_like(j),where=norms[None]>1e-12)
    singular=np.linalg.svd(unit,compute_uv=False)
    return dict(column_norms=norms.tolist(),cosine_similarity=(unit.T@unit).tolist(),
        normalized_singular_values=singular.tolist(),
        numerical_rank=int((singular>max(float(singular[0]),1e-12)*1e-3).sum()),
        note='Local numerical dependence at M0; not global identifiability, parameter uncertainty or measured capability.')


def _probe_parameters(p):
    # Perturb one quantity at a time, not a fitting grid or a parameter selection.
    delta=min(.01,p.delay_s) if p.delay_s>0 else .01
    return [('feedforward_xy',max(float(p.feedforward_xy)*.1,.01)),
            ('delay_s',delta),('attitude_time_constant_s',float(p.attitude_time_constant_s)*.1)]


def _plot(folder, name, t, command, measured, predicted, velocity, acceleration, predicted_acceleration):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(13,8),sharex=True)
    for axis,label in enumerate('XYZ'):
        for row,unit in enumerate(('Position (m)','Velocity (m/s)','Acceleration (m/s²)')):
            ax=axes[row,axis]; ax.plot(t,command[:,row*3+axis],label='Command',color='#64748b',ls='--')
            obs=(measured,velocity,acceleration)[row]
            pred=(predicted['position_origin_m'],predicted['velocity_origin_m_s'],predicted_acceleration)[row]
            ax.plot(t,obs[:,axis],label='Measured' if row==0 else 'Estimated from native pose',color='#0891b2')
            ax.plot(t,pred[:,axis],label='M0 recursive prediction',color='#e87924')
            ax.grid(alpha=.2)
            if row==0:ax.set_title(label)
            if axis==0:ax.set_ylabel(unit)
            if row==2:ax.set_xlabel('Time from command onset (s)')
    axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=7)
    fig.suptitle(f'{name}: reinitialized drone diagnostic — derivatives are retrospective, gaps unfilled')
    fig.tight_layout();fig.savefig(folder/(name+'.png'),dpi=140);plt.close(fig)


@torch.no_grad()
def diagnose_drone(job, output, device='cuda'):
    job=Path(job).resolve(); output=Path(output).resolve()
    model,protocol,engine=fitting.load(job,device)
    names=[n for n,row in protocol['takes'].items() if row['role']=='adaptation']
    if not names:raise ValueError('Drone diagnosis requires adaptation takes')
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'status.json',dict(status='running',model_selected=False))
    started=time.perf_counter();parameters=engine.drone.parameters
    probes=_probe_parameters(parameters); matrices=[]; reports={}
    try:
        for number,name in enumerate(names):
            if (output/'STOP').exists():raise InterruptedError('Stopped between takes')
            print(f'DRONE diagnostic {number+1}/{len(names)}: {name}',flush=True)
            trial=WhipTrial(job,name,model,device);d=trial.data
            # Exclude samples at/after reviewed end BEFORE derivative estimation.
            keep=d['time']<trial.end-1e-9
            td=d['time'][keep];pd=d['position'][keep];vd=d['pose_valid'][keep]
            velocity,acceleration=local_kinematics(td,pd,vd)
            _,wide_acceleration=local_kinematics(td,pd,vd,half_width_s=.09)
            ids=np.flatnonzero(td>=trial.hover_time[-1]-1e-10);t=td[ids]
            if len(t)<8:raise ValueError(name+': insufficient reviewed native pose samples')
            valid=vd[ids]&np.isfinite(pd[ids]).all(-1)
            score=(t>=0)&valid
            if not score.any() or valid[t>=0].mean()<protocol['minimum_observation_fraction']:
                raise ValueError(name+': insufficient reviewed drone-pose coverage')
            command=np.stack([trial.schedule.sample(float(x))[0].cpu().numpy() for x in t])
            packets=torch.as_tensor(d['packets'][None],device=device,dtype=torch.float64)
            hold=torch.as_tensor(d['hover_commands'][-1:],device=device,dtype=torch.float64)
            def predict(p):
                check_command_coverage(trial,t,p.delay_s)
                drone=copy(engine.drone);drone.parameters=p
                result=drone.predict(trial.initial_pose(p),packets,d['packet_time'],t,engine.offset,graph=True,hover_command=hold)
                if not bool(result['valid'].all()):raise ValueError(name+': invalid pose-domain probe')
                return {k:v[0].cpu().numpy() for k,v in result.items()}
            baseline=predict(parameters)
            truth_rotation=d['rotation'][keep][ids]
            def residual_vector(pred):
                pos=(pred['position_origin_m'][score]-pd[ids][score])/.02
                relative=truth_rotation[score].transpose(0,2,1)@pred['rotation_tracking_to_world'][score]
                angle=Rotation.from_matrix(relative).as_rotvec()/.05
                return np.c_[pos,angle].reshape(-1)/np.sqrt(score.sum())
            residual=residual_vector(baseline);columns=[];probe_metrics={}
            for parameter,step in probes:
                if (output/'STOP').exists():raise InterruptedError('Stopped between probes')
                center=float(getattr(parameters,parameter));low=max(0.,center-step);high=center+step
                minus=predict(replace(parameters,**{parameter:low}))
                plus=predict(replace(parameters,**{parameter:high}))
                # Derivative with respect to dimensionless perturbation (step units).
                columns.append((residual_vector(plus)-residual_vector(minus))*step/(high-low))
                probe_metrics[parameter]=dict(baseline=center,low=low,high=high,scale=step,
                    low_loss=float(np.dot(residual_vector(minus),residual_vector(minus))),
                    high_loss=float(np.dot(residual_vector(plus),residual_vector(plus))))
            matrix=np.stack(columns,-1);matrices.append(matrix)
            _,predicted_acceleration=local_kinematics(t,baseline['position_origin_m'],np.ones(len(t),bool))
            native_acc=acceleration[ids];wide_acc=wide_acceleration[ids]
            acc_mask=score&np.isfinite(native_acc).all(-1)&np.isfinite(wide_acc).all(-1)
            considered=t>=0
            observed=np.where(valid[:,None],pd[ids],np.nan)
            disagreement=rms_summary(np.linalg.norm(native_acc[considered]-wide_acc[considered],axis=-1))
            disagreement['rmse_m_s2']=disagreement.pop('rmse_m')
            reports[name]=dict(role='adaptation',scored_frames=int(score.sum()),
                tracking_position=rms_summary(np.linalg.norm(observed[considered]-command[considered,:3],axis=-1)),
                model_position=rms_summary(np.linalg.norm(observed[considered]-baseline['position_origin_m'][considered],axis=-1)),
                derivative_comparison_frames=int(acc_mask.sum()),
                acceleration_window_disagreement=disagreement,
                baseline_scaled_pose_loss=float(np.dot(residual,residual)),probes=probe_metrics,
                local_sensitivity=sensitivity_summary(matrix),maximum_acceleration_identified=False)
            np.savez_compressed(output/(name+'.npz'),time_s=t,command=command,measured_origin=pd[ids],
                measured_rotation=truth_rotation,score_mask=score,velocity_estimate=velocity[ids],
                acceleration_estimate=native_acc,acceleration_wide_estimate=wide_acc,
                predicted_acceleration_estimate=predicted_acceleration,sensitivity=matrix,**baseline)
            _plot(output,name,t,command,np.where(valid[:,None],pd[ids],np.nan),baseline,velocity[ids],native_acc,predicted_acceleration)
        report=dict(schema='drone_response_diagnostic_v1',job=str(job),takes=reports,
            input_manifests={str(job/n):sha256_file(job/n) for n in ('prepared_hashes.json','source_hashes.json','code_hashes.json')},
            parameter_order=[n for n,_ in probes],pooled_sensitivity=sensitivity_summary(np.concatenate(matrices)/np.sqrt(len(names))),
            derivative_half_width_s=.06,derivative_check_half_width_s=.09,derivative_max_gap_s=.025,
            pose_scales=dict(position_m=.02,orientation_rad=.05),device=device,dtype='float64',
            elapsed_s=time.perf_counter()-started,model_selected=False,drone_fitted=False,
            saturation_fitted=False,maximum_acceleration_identified=False,
            evidence='Adaptation-only recursive diagnostics. Finite probes are not selected models. Derivatives use future neighbors only inside the reviewed free-motion interval. No actuator-limit claim.')
        verify_hashes(read_json(job/'prepared_hashes.json'));verify_hashes(protocol['frozen_hashes'])
        atomic_json(output/'report.json',report)
        atomic_json(output/'evidence_hashes.json',{str(p):sha256_file(p) for p in output.iterdir() if p.is_file() and p.name!='status.json'})
        atomic_json(output/'status.json',dict(status='completed',model_selected=False,drone_fitted=False))
        return report
    except BaseException as exc:
        atomic_json(output/'status.json',dict(status='stopped' if isinstance(exc,InterruptedError) else 'failed',error=str(exc),model_selected=False))
        raise
