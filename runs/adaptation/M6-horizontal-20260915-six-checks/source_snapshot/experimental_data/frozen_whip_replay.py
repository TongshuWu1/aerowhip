"""Auditable historical-force counterfactual, restricted to recorded CSV execution."""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from deployment.fullstate import sample_fullstate
from simulator.point_mass import ForceControlledPointCable
from simulator.cable import DderState, CableConfiguration, DderModel
from simulator.cable.residual import FrozenMotionResidual
from simulator.gpu_rehearsal import GpuRehearsalPhysics
from simulator.cable.cuda_rehearsal_solvers import RehearsalSolvers
from simulator.fullstate_execution import FullStateAttachmentModel
from .historical_fit import read, read_inputs
from .force_dataset import _normalized_rotations, reconstruct_dder_nodes
from .state_initialization import causal_state
from .cable_fit import PreparedTake
from .constrained_identification import combine
from .differentiable_fit import rollout
from .whip_validation import errors, command_inputs
from .io import atomic_json, sha256_file


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1]
    source=root/'runs/rehearsals/20260906-222405-804438'
    plan_dir=source/'plan_001'
    fit=root/'data/historical_model_runs/20260908-013552-736857-whip-only-drone'
    out=root/'runs/audits'/datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-frozen-whip-replay')
    out.mkdir(parents=True,exist_ok=False)
    print(str(out),flush=True)
    plan=np.load(plan_dir/'plan.npz');archived=np.load(plan_dir/'fullstate_source.npz')
    metadata=read(plan_dir/'fullstate.json')
    assert sha256_file(plan_dir/'plan.npz')==metadata['source_plan_sha256']
    assert sha256_file(plan_dir/'fullstate_30hz.csv')==metadata['csv_sha256']
    csv=np.loadtxt(plan_dir/'fullstate_30hz.csv',delimiter=',',skiprows=1)
    old_packets=np.column_stack(sample_fullstate(archived['time_s'],archived['positions_m'][:,0],archived['velocities_m_s'][:,0])[1:])
    np.testing.assert_allclose(old_packets,csv[:,1:10],rtol=0,atol=1e-10)
    old=read(source/'package/policy/model.json');new=read(fit/'candidate_bundle/model.json')
    active_hash=sha256_file(root/'config/model.json')
    tensor=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float64,device='cuda')
    initial=DderState(torch.tensor(plan['positions_m'][None]),torch.tensor(plan['velocities_m_s'][None]))
    torch.empty(1,device='cuda');solvers=RehearsalSolvers()
    trajectories={}
    for label,payload in [('archived_model_replay',old),('current_model_replay',new)]:
        print(label,flush=True)
        physics=GpuRehearsalPhysics(ForceControlledPointCable.from_mapping(payload),initial,.01,solvers)
        q,v=initial.positions_m.clone(),initial.velocities_m_s.clone()
        qs,vs=[q[0].numpy().copy()],[v[0].numpy().copy()]
        for force in plan['forces_world_n']:
            q,v=physics(q,v,torch.tensor(force[None]))
            qs.append(q[0].numpy().copy());vs.append(v[0].numpy().copy())
        trajectories[label]=(np.array(qs),np.array(vs))
        assert np.isfinite(qs).all() and np.isfinite(vs).all()
    reproduction=float(np.max(np.abs(trajectories['archived_model_replay'][0]-archived['positions_m'])))
    if reproduction>1e-5:raise ValueError(f'Archived physics replay mismatch: {reproduction} m')
    qnew,vnew=trajectories['current_model_replay']
    times=np.arange(len(qnew))*.01
    new_packets=np.column_stack(sample_fullstate(times,qnew[:,0],vnew[:,0])[1:])
    np.savez_compressed(out/'frozen_plan_replay.npz',forces_world_n=plan['forces_world_n'],
        initial_positions_m=plan['positions_m'],initial_velocities_m_s=plan['velocities_m_s'],time_s=times,
        archived_positions_m=archived['positions_m'],new_virtual_positions_m=qnew,
        new_virtual_velocities_m_s=vnew,old_packets=old_packets,new_attachment_packets=new_packets)
    atomic_json(out/'candidate_model.json',new)
    drone=FullStateAttachmentModel(root/new['fullstate_execution']['checkpoint'],new['fullstate_execution']['sha256'],device='cuda')
    cable=CableConfiguration.from_mapping(new['cable'])
    model=DderModel(cable.dder_parameters(EI=new['cable']['EI_n_m2'],Cb=new['cable']['Cb_n_m2_s']))
    model.motion_residual=FrozenMotionResidual(root/new['motion_residual']['checkpoint'],new['motion_residual']['sha256'])
    data=read_inputs(fit);results={}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(16,12))
    for row,(name,a) in enumerate(data.items()):
        print(name,flush=True)
        active=a['command_valid']&(np.linalg.norm(a['commands'][:,3:9],axis=1)>1e-6)
        start=int(np.flatnonzero(active)[0]);end=start
        while end<len(active) and active[end]:end+=1
        count=end-start-1  # Last state strictly before controller switched to hold.
        assert 60<=count<=68 and a['drone_fit_valid'][start-25:end].all()
        assert a['cable_fit_valid'][start-20:start+1].all()
        rotation,_=_normalized_rotations(a['quaternion'])
        offsets=np.einsum('tij,j->ti',rotation,drone.offset_body_m)
        attachment=a['position']+offsets
        t=np.arange(-10,1)*.01
        vel=np.linalg.lstsq(np.column_stack((np.ones(11),t,t*t)),attachment[start-10:start+1],rcond=None)[0][1]
        hist=slice(start-20,start+1)
        nodes,valid=reconstruct_dder_nodes(attachment[hist],a['cable_fit_valid'][hist],a['markers'][hist],a['marker_valid'][hist],cable)
        assert valid.all()
        state=causal_state(tensor(nodes[None]),.01,model)
        v=state.velocities_m_s.clone();v[:,0]=tensor(vel)
        observed=a['commands'][start:end]
        distances=np.max(np.abs(observed[:,None]-old_packets[None]),axis=-1)
        packet_ids=distances.argmin(1)
        packet_error=float(distances.min(1).max())
        assert packet_error<1e-6 and np.array_equal(np.unique(packet_ids),np.arange(20))
        # Preserve actual packet timing and pre-maneuver command history.
        replacement=a['commands'].copy()
        replacement[start:end]=new_packets[packet_ids]
        replacement[start:end,:3]-=offsets[start]
        variants=[];roots=[]
        labels=['actual_logged_csv_both_residuals','same_force_new_csv_both_residuals']
        for label,commands in zip(labels,[a['commands'],replacement]):
            cmd,past=command_inputs(dict(commands=commands),start,count,round(drone.delay_s/.01),offsets[start])
            predicted=drone.predict(tensor(attachment[start][None]),tensor(vel[None]),tensor(cmd[None]),tensor(past[None]),tensor(np.full((1,count),.01)))[0]
            roots.append(predicted.cpu().numpy()[0])
            variants.append(PreparedTake(label,'development',state.positions_m,v,predicted,
                tensor(a['markers'][start:end][None]),(start,),0.,.01))
        predicted=rollout(combine(variants),model,cable,tensor([new['cable']['EI_n_m2'],new['cable']['Cb_n_m2_s'],0.])).cpu().numpy()
        assert np.isfinite(predicted).all()
        truth=a['markers'][start:end];valid=a['cable_fit_valid'][start:end];tt=np.arange(count+1)*.01
        metrics={label:dict(tip=errors(predicted[k,:,-1],truth[:,-1],valid),
            attachment=errors(roots[k],attachment[start:end],np.ones(count+1,bool))) for k,label in enumerate(labels)}
        metrics['old_virtual_plan']=dict(tip=errors(archived['positions_m'][:count+1,cable.marker_node_indices[-1]],truth[:,-1],valid))
        metrics['new_virtual_plan']=dict(tip=errors(qnew[:count+1,cable.marker_node_indices[-1]],truth[:,-1],valid))
        results[name]=dict(start_frame=start,last_csv_state_s=count*.01,packet_count=20,
            historical_packet_match_max_abs=packet_error,metrics=metrics)
        np.savez_compressed(out/f'{name}.npz',time_s=tt,labels=np.array(labels),predicted_markers=predicted,
            predicted_attachment=np.array(roots),measured_markers=truth,measured_attachment=attachment[start:end],
            cable_valid=valid,packet_ids=packet_ids,logged_commands=observed,
            counterfactual_vehicle_commands=replacement[start:end])
        ax=fig.add_subplot(3,2,2*row+1,projection='3d')
        measured=truth[:,-1].copy();measured[~valid]=np.nan
        paths=[measured,archived['positions_m'][:count+1,-1],qnew[:count+1,-1],predicted[0,:,-1],predicted[1,:,-1]]
        titles=['OptiTrack cable tip','Old virtual plan','New virtual plan / same force','Both residuals / actual CSV','Both residuals / new CSV']
        colors=['black','tab:blue','tab:orange','tab:green','tab:red']
        for p,label,color in zip(paths,titles,colors):ax.plot(*p.T,label=label,color=color)
        allp=np.concatenate(paths);middle=(np.nanmax(allp,axis=0)+np.nanmin(allp,axis=0))/2
        radius=np.nanmax(np.nanmax(allp,axis=0)-np.nanmin(allp,axis=0))/2
        ax.set(xlim=(middle[0]-radius,middle[0]+radius),ylim=(middle[1]-radius,middle[1]+radius),zlim=(middle[2]-radius,middle[2]+radius),xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title=f'{name}: CSV maneuver only')
        ax.set_box_aspect((1,1,1));ax.legend(fontsize=7,loc='upper left')
        ax=fig.add_subplot(3,2,2*row+2)
        for p,label,color in zip(paths[1:],titles[1:],colors[1:]):ax.plot(tt,np.linalg.norm(p-measured,axis=1)*100,label=label,color=color)
        ax.set(xlabel='Time from first CSV packet (s)',ylabel='Cable-tip discrepancy (cm)');ax.legend(fontsize=8);ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
    summary=dict(source_plan=str(plan_dir),source_plan_sha256=sha256_file(plan_dir/'plan.npz'),
        source_csv_sha256=metadata['csv_sha256'],candidate=str(fit/'candidate_bundle/model.json'),
        archived_reproduction_max_position_difference_m=reproduction,
        executed_prefix_new_vs_old_packet_position_rmse_m=float(np.sqrt(np.mean(np.sum((new_packets[:20,:3]-old_packets[:20,:3])**2,axis=1)))),
        hardware=torch.cuda.get_device_name(),takes=results,independent_test=False,
        interpretation='Current all-data candidate: historical commands are in-sample validation. New CSV is a counterfactual, not the flown input.',
        initialization='Virtual: unchanged archived plan state. Execution: causal measured cable and attachment state; no subsequent measurement feedback.',
        timing='Use original observed packet timing. Exclude every state at or after first post-maneuver hold.',
        export_mapping='New virtual attachment positions minus initial world attachment offset; old CSV used exactly as logged.',
        active_unchanged=active_hash==sha256_file(root/'config/model.json'),ppo_started=False)
    assert summary['active_unchanged']
    atomic_json(out/'summary.json',summary)
    lines=['# Frozen historical force replay','',summary['interpretation'],'',summary['initialization'],'',summary['timing'],'',
        f'Archived physics reproduction maximum position difference: {reproduction:.3g} m.',
        f'New versus old virtual command position RMSE over first 20 packets: {summary["executed_prefix_new_vs_old_packet_position_rmse_m"]*100:.2f} cm.','',
        '| Take | Last CSV state (s) | Old virtual tip (cm) | New virtual tip (cm) | Both NN, actual CSV tip (cm) | Both NN, new CSV tip (cm) |',
        '|---|---:|---:|---:|---:|---:|']
    for name,r in results.items():
        m=r['metrics'];vals=[m[k]['tip']['rmse_m']*100 for k in ['old_virtual_plan','new_virtual_plan',*labels]]
        lines.append(f'| {name} | {r["last_csv_state_s"]:.2f} | '+ ' | '.join(f'{x:.2f}' for x in vals)+' |')
    lines+=['','Errors are time-aligned trajectory discrepancies, not hitting errors. Intended hit time 0.78 s is outside the executed CSV segment. No recovery/hold is scored.','',
        'Both residuals remain enabled in the complete response. Cable NN participates in virtual force dynamics and in the separate predicted execution, not twice in one cable step. Drone response consumes PVA, not force directly.','',
        'Clock alignment and tracked-point conventions retain the existing dataset assumptions. No new trajectory was flown. Raw data, active configuration and selected PPO are unchanged.','',
        '![Comparison](comparison.png)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
