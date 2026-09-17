"""Compare frozen strike speeds and replay the CSVs under one common model."""
from pathlib import Path
import sys,json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experimental_data.io import atomic_json,sha256_file
from experimental_data.model_evaluation import model_identity
from experimental_data.whip_adaptation import verify_hashes
from simulator.workflow import read_json
from simulator.research_execution import ResearchExecutionModel
from planning.reference_correction import CoupledRollout


def at(t,x,event):return np.array([np.interp(event,t,x[:,j]) for j in range(x.shape[1])])


def metric(t,p,v,event,target,direction):
    pos=at(t,p,event);vel=at(t,v,event);speed=float(np.linalg.norm(vel))
    return dict(time_s=float(event),speed_m_s=speed,forward_velocity_m_s=float(vel@direction),
        velocity_xyz_m_s=vel.tolist(),angle_deg=float(np.degrees(np.arccos(np.clip(vel@direction/max(speed,1e-12),-1,1)))),
        target_distance_m=float(np.linalg.norm(pos-target)))


def summarize(t,p,v,origin_v,event,target,direction):
    mask=(t>=0)&(t<=1.5+1e-9);idx=np.flatnonzero(mask)[np.argmin(np.linalg.norm(p[mask]-target,axis=-1))]
    return dict(fixed_time=metric(t,p,v,event,target,direction),
        closest_in_0_to_1p5s=metric(t,p,v,t[idx],target,direction),
        quadrotor_speed_at_fixed_time_m_s=float(np.linalg.norm(at(t,origin_v,event))),
        peak_tip_speed_in_0_to_1p5s_m_s=float(np.linalg.norm(v[mask],axis=-1).max()))


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    out=ROOT/'runs/audits/current-strike-speed-20260913';out.mkdir(parents=True,exist_ok=False)
    current=read_json(ROOT/'exports/CURRENT_FLIGHT.json')
    assert current['model_id']=='M2-selected'
    cfg=read_json(ROOT/current['rehearsal']/'settings.json')
    meta=read_json(ROOT/'runs/reference_tracking/M0-paper-fixed-reference/reference.json')
    event=meta['planned_strike_time_s'];target=np.asarray(meta['physical_target_m'])
    direction=np.asarray(cfg['task']['strike_direction'],float);direction/=np.linalg.norm(direction)
    paths={'M0':'M0_Bspline_slower_brake_1s','M1':'M1_local_fixed_tip_reference','M2':'M2_selected_fixed_tip_reference'}
    packets=[];saved={};command={};hashes={}
    for name,folder in paths.items():
        export=ROOT/'exports'/folder;metadata=read_json(export/('manifest.json' if name=='M0' else 'export.json'))
        csv=export/'fullstate_30hz.csv';assert sha256_file(csv)==metadata['csv_sha256']
        hashes[str(csv)]=sha256_file(csv)
        arr=np.genfromtxt(csv,delimiter=',',skip_header=1)
        assert np.allclose(arr[:46,0],np.arange(46)/30,atol=1e-12)
        packets.append(arr[:46,1:])
        command[name]=dict(quadrotor_command_speed_at_strike_m_s=float(np.linalg.norm(at(arr[:,0],arr[:,4:7],event))),
            peak_command_speed_in_strike_prefix_m_s=float(np.linalg.norm(arr[arr[:,0]<=meta['interval_s'][1],4:7],axis=-1).max()))
        run=ROOT/metadata['rehearsal'];hashes[str(run/'rehearsal.npz')]=sha256_file(run/'rehearsal.npz')
        with np.load(run/'rehearsal.npz') as f:
            saved[name]=summarize(f['prediction_time_s'],f['cable_positions_m'][:,-1],f['cable_velocities_m_s'][:,-1],
                f['origin_velocities_m_s'],event,target,direction)
    model_path=ROOT/current['rehearsal']/'model.json';hashes.update(model_identity(model_path)[1])
    engine=ResearchExecutionModel.from_mapping(read_json(model_path),root=model_path.parent,device='cuda')
    refpath=ROOT/'runs/reference_tracking/M0-paper-fixed-reference/reference.npz'
    assert sha256_file(refpath)==meta['reference_sha256'];hashes[str(refpath)]=sha256_file(refpath)
    with np.load(refpath) as f:initial=f['cable_position_m'][0]
    rollout=CoupledRollout(engine,3,cfg['launch']['origin_m'],initial,cfg['limits'])
    grid=np.arange(226)*engine.dt_s
    result=rollout(torch.as_tensor(np.stack(packets),device='cuda',dtype=torch.float64),np.arange(46)/30,grid)
    assert bool(result['complete_valid'].all())
    common={}
    for i,name in enumerate(paths):
        common[name]=summarize(grid,result['cable_positions_m'][i,:,-1].cpu().numpy(),
            result['cable_velocities_m_s'][i,:,-1].cpu().numpy(),result['velocity_origin_m_s'][i].cpu().numpy(),event,target,direction)
    with np.load(ROOT/current['rehearsal']/'rehearsal.npz') as f:
        delta=float(np.max(np.abs(result['cable_positions_m'][2].cpu().numpy()-f['cable_positions_m'][:len(grid)])))
        velocity_delta=float(np.max(np.abs(result['cable_velocities_m_s'][2].cpu().numpy()-f['cable_velocities_m_s'][:len(grid)])))
    assert delta<1e-5 and velocity_delta<1e-3
    measured={}
    for name in ('M0','M1'):
        path=ROOT/f'runs/data_review/{name}-paper-20260913/paper_metrics.json'
        report=read_json(path);hashes[str(path)]=sha256_file(path)
        rows=report['takes'];rows=list(rows.values()) if isinstance(rows,dict) else rows
        speeds=[r['fixed_time_tip_speed_m_s'] for r in rows]
        measured[name]=dict(mean_m_s=float(np.mean(speeds)),sample_sd_m_s=float(np.std(speeds,ddof=1)),
            per_take_m_s=speeds,definition='Previously computed local derivative of observed tip positions at original planned strike time')
    verify_hashes(hashes)
    report=dict(planned_strike_time_s=event,target_m=target.tolist(),strike_direction=direction.tolist(),
        saved_forecasts_each_original_model=saved,common_current_M2_model=common,command_speeds=command,
        measured_M0_M1_fixed_time_speeds=measured,M2_physical_speed_available=False,
        common_replay_saved_M2_max_position_difference_m=delta,common_replay_saved_M2_max_velocity_difference_m_s=velocity_delta,
        limitations='Simulation velocities are not measured contact speed or impact force. Closest-approach diagnostic uses the saved 150 Hz grid. Fixed-time comparison is primary.',
        source_hashes=hashes)
    atomic_json(out/'report.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
