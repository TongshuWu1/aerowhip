"""Maneuver-anchored held-out command -> attachment -> cable validation."""
import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from .historical_fit import read,read_inputs,progress
from .historical_combined_review import MaskedResidual
from .force_dataset import _normalized_rotations,reconstruct_dder_nodes
from .state_initialization import causal_state
from .cable_fit import PreparedTake
from .constrained_identification import combine
from .differentiable_fit import rollout
from .io import atomic_json,sha256_file,canonical_json_hash
from simulator.cable import CableConfiguration,DderModel
from simulator.cable.residual import FrozenMotionResidual
from simulator.fullstate_execution import FullStateAttachmentModel


def verify_split(held,cable_training,drone_training):
    if held in cable_training or held in drone_training:
        raise ValueError('Held-out whip entered model fitting')


def command_inputs(a,start,count,lag,offset):
    cmd=a['commands'][start-lag:start-lag+count].copy()
    past=a['commands'][start-lag-5:start-lag-5+count].copy()
    cmd[:,:3]+=offset;past[:,:3]+=offset
    return cmd,past


def errors(pred,truth,valid):
    err=np.linalg.norm(pred-truth,axis=-1)
    mask=np.broadcast_to(valid.reshape((-1,)+(1,)*(err.ndim-1)),err.shape)&np.isfinite(err)
    return dict(rmse_m=float(np.sqrt(np.mean(err[mask]**2))),maximum_m=float(err[mask].max()),
        valid_frames=int(valid.sum())) if mask.any() else None


@torch.no_grad()
def validate(job):
    torch.set_num_threads(1)
    settings=read(job/'protocol.json');data=read_inputs(job)
    out=job/'validation';out.mkdir(exist_ok=False)
    results={};count=round(settings['validation_horizon_s']/.01)
    for i,names in enumerate(settings['folds'],1):
        assert len(names)==1
        name=names[0];label=f'fold_{i}';a=data[name];folder=out/name;folder.mkdir()
        cable_dir=job/'cable'/label;payload=read(cable_dir/'candidate_model.json')
        training=read(cable_dir/'review.json')['training']
        drone_training=[n for n in settings['takes'] if n!=name]
        verify_split(name,training,drone_training)
        checkpoint=job/'drone_attachment'/label/'drone_residual.pt'
        assert read(job/'drone_attachment'/label/'review.json')['held_out']==[name]
        drone=FullStateAttachmentModel(checkpoint,sha256_file(checkpoint),device='cuda')
        np.testing.assert_allclose(drone.offset_body_m,payload['recorded_data']['optitrack_to_attachment_offset_body_m'],atol=1e-10)
        active=a['command_valid']&(np.linalg.norm(a['commands'][:,3:9],axis=1)>1e-6)
        start=int(np.flatnonzero(active)[0]);stop=start+count+1
        if start<25 or stop>len(a['time']):raise ValueError('Incomplete maneuver/history recording')
        if not a['drone_fit_valid'][start-25:stop].all():raise ValueError('Missing command/drone inputs in maneuver')
        if not a['cable_fit_valid'][start-20:start+1].all():raise ValueError('Invalid initial cable history')
        rotation,_=_normalized_rotations(a['quaternion'])
        offsets=np.einsum('tij,j->ti',rotation,drone.offset_body_m)
        attachment=a['position']+offsets
        t=np.arange(-10,1)*.01
        velocity=np.linalg.lstsq(np.column_stack((np.ones(11),t,t*t)),attachment[start-10:start+1],rcond=None)[0][1]
        lag=round(drone.delay_s/.01)
        commands,past=command_inputs(a,start,count,lag,offsets[start])
        tensor=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float64,device='cuda')
        args=(tensor(attachment[start][None]),tensor(velocity[None]),tensor(commands[None]),tensor(past[None]),tensor(np.full((1,count),.01)))
        nominal=drone.predict(*args,residual=False)[0];corrected=drone.predict(*args,residual=True)[0]
        cable=CableConfiguration.from_mapping(payload['cable'])
        model=DderModel(cable.dder_parameters(EI=payload['cable']['EI_n_m2'],Cb=payload['cable']['Cb_n_m2_s']))
        hist=slice(start-20,start+1)
        nodes,valid=reconstruct_dder_nodes(attachment[hist],a['cable_fit_valid'][hist],a['markers'][hist],a['marker_valid'][hist],cable)
        if not valid.all():raise ValueError('Invalid initial reconstructed state')
        state=causal_state(tensor(nodes[None]),.01,model)
        v=state.velocities_m_s.clone();v[:,0]=args[1]
        truth=tensor(a['markers'][start:stop][None]);root_truth=tensor(attachment[start:stop][None])
        take=PreparedTake(name,'validation',state.positions_m,v,root_truth,truth,(start,),0.,.01)
        labels=['physics_only','cable_nn_only','drone_nn_only','both_residuals','measured_attachment_diagnostic']
        roots=[nominal,nominal,corrected,corrected,root_truth]
        variants=[replace(take,take_id=n,root_positions_m=r) for n,r in zip(labels,roots)]
        model.motion_residual=MaskedResidual(FrozenMotionResidual(payload['motion_residual']['checkpoint'],payload['motion_residual']['sha256']),tensor([0,1,0,1,1]))
        params=tensor([payload['cable']['EI_n_m2'],payload['cable']['Cb_n_m2_s'],0.])
        progress(out,f'{name}: complete 1.5-second prediction from maneuver onset')
        predicted=rollout(combine(variants),model,cable,params).cpu().numpy()
        if not np.isfinite(predicted).all():raise ValueError('Nonfinite held-out prediction')
        roots=np.concatenate([r.cpu().numpy() for r in roots])
        measured=a['markers'][start:stop];measured_root=attachment[start:stop]
        valid=a['cable_fit_valid'][start:stop];times=np.arange(count+1)*.01
        hit=round(settings['validation_hit_time_s']/.01);target=np.array(settings['validation_target_m'])
        def target_metrics(tip,mask):
            distance=np.linalg.norm(tip-target,axis=1);ix=np.flatnonzero(mask&np.isfinite(distance))
            closest=ix[np.argmin(distance[ix])]
            return dict(at_planned_time_m=float(distance[hit]) if mask[hit] else None,
                closest_m=float(distance[closest]),closest_time_s=float(times[closest]))
        metrics={}
        for k,n in enumerate(labels):
            metrics[n]=dict(attachment=errors(roots[k],measured_root,np.ones(len(times),bool)),
                cable=errors(predicted[k],measured,valid),tip=errors(predicted[k,:,-1],measured[:,-1],valid),
                maneuver_tip=errors(predicted[k,:,-1],measured[:,-1],valid&(times<=.67)),
                tip_error_at_planned_time_m=float(np.linalg.norm(predicted[k,hit,-1]-measured[hit,-1])) if valid[hit] else None,
                target=target_metrics(predicted[k,:,-1],np.ones(len(times),bool)))
        result=dict(metrics=metrics,observed_target=target_metrics(measured[:,-1],valid),start_frame=start,
            excluded_cable_frames=int((~valid).sum()),held_out=name,cable_training=training,drone_training=drone_training,
            independent_test=False,measured_attachment_diagnostic_is_not_open_loop_validation=True,
            model_sha256=canonical_json_hash(payload),drone_sha256=sha256_file(checkpoint))
        atomic_json(folder/'metrics.json',result);results[name]=result
        np.savez_compressed(folder/'trajectories.npz',time_s=times,labels=np.array(labels),prediction=predicted,
            attachment_prediction=roots,measured_markers=measured,measured_attachment=measured_root,cable_valid=valid,
            reference=a['commands'][start:stop])
    atomic_json(out/'review.json',dict(takes=results,horizon_s=count*.01,
        independent_test=False,method='Leave-one-whip-out weights and geometry; causal initial state, recorded commands only thereafter',
        target_status=settings['target_status'],active_model_changed=False,ppo_started=False))
    progress(out,'Complete held-out maneuver validation finished')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True)
    validate(parser.parse_args().job)
