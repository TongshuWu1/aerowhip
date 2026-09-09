"""Read-only model/training/export audit; writes evidence in its own audit folder."""
from pathlib import Path
import ast
import json
import numpy as np
import torch
from deployment.fullstate import sample_fullstate
from deployment.package import export_policy
from experimental_data.historical_fit import read_inputs,read
from experimental_data.io import atomic_json,sha256_file
from learning.point_force_env import PointForceWhipEnvironment
from learning.deployment_rollout import sample_batch,plan_batch,execute_batch
from learning.fullstate_rollout import frozen_reference
from run_ppo import build_agent
from simulator.point_mass import ForceControlledPointCable
from simulator.cable import DderState
from simulator.gpu_rehearsal import GpuRehearsalPhysics
from simulator.cable.cuda_rehearsal_solvers import RehearsalSolvers
from simulator.strike_plan import compile_strike_plan


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1]
    out=Path((root/'runs/audits/current_theory_audit.txt').read_text())
    run=Path((root/'runs/ppo/adaptation1_launch_path.txt').read_text())
    job=root/'data/historical_model_runs/20260908-013552-736857-whip-only-drone'
    evidence={}
    data=read_inputs(job);windows=read(job/'drone_attachment/final/windows.json')
    coverage={}
    for name,a in data.items():
        active=a['command_valid']&(np.linalg.norm(a['commands'][:,3:9],axis=1)>1e-6)
        s=int(np.flatnonzero(active)[0]);e=s
        while active[e]:e+=1
        starts=[r['start'] for r in windows if r['take']==name]
        weights=np.zeros(len(active),int)
        for start in starts:weights[start:start+100]+=1
        coverage[name]=dict(onset=s,end=e,starts_relative_s=[(x-s)*.01 for x in starts],
            total_fitted_intervals=len(starts)*100,maneuver_fitted_intervals=int(weights[s:e].sum()),
            unique_maneuver_frames=int((weights[s:e]>0).sum()),maneuver_frames=e-s,
            unrepresented_relative_times_s=(np.flatnonzero(weights[s:e]==0)*.01).tolist())
    evidence['drone_fit_coverage']=coverage
    controller=root.parent  # Supplied file remains external and unchanged.
    path=Path('C:/Users/wts28/Downloads/full_state_pva.py')
    tree=ast.parse(path.read_text())
    seq=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='PVA_SEQUENCE' for t in n.targets))
    historical=root/'runs/rehearsals/20260906-222405-804438/plan_001'
    csv=np.loadtxt(historical/'fullstate_30hz.csv',delimiter=',',skiprows=1)
    controller_pva=np.array([np.r_[r[1],r[2],r[3]] for r in seq])
    evidence['supplied_controller']=dict(path=str(path),sha256=sha256_file(path),samples=len(seq),
        last_time_s=seq[-1][0],prefix_pva_max_abs=float(abs(controller_pva-csv[:len(seq),1:10]).max()),
        execution_duration_including_last_hold_s=seq[-1][0]+1/30,
        current_colleague_file_verified=False)
    try:
        export_policy(root,run/'checkpoints/best_validation.pt',out/'rejected_export')
    except ValueError as error:
        evidence['export_probe']=dict(rejected=True,reason=str(error),output_created=(out/'rejected_export').exists())
    else:
        evidence['export_probe']=dict(rejected=False)
    # A constant-acceleration analytic trajectory must survive the interpolator.
    dt=.01;t=np.arange(101)*dt;acc=np.array([1.,0.,0.])
    p=.5*t[:,None]**2*acc;v=t[:,None]*acc
    _,_,_,sampled=sample_fullstate(t,p,v)
    evidence['analytic_pva_max_acceleration_error']=float(abs(sampled-acc).max())
    # Symplectic-Euler positions with 12 substeps have O(dt) position/velocity
    # inconsistency which Hermite exactly preserves and turns into acceleration ripple.
    p_discrete=p+t[:,None]*(dt/12)*acc/2
    _,_,_,sampled=sample_fullstate(t,p_discrete,v)
    evidence['semiimplicit_12_substep_pva']=dict(first_acceleration=sampled[0].tolist(),
        minimum_x=float(sampled[:,0].min()),maximum_x=float(sampled[:,0].max()),
        mean_x_excluding_terminal=float(sampled[:-1,0].mean()),physical_acceleration_x=1.)
    old=np.load(historical/'fullstate_source.npz');tt=csv[:20,0]
    ix=np.searchsorted(old['time_s'],tt,side='right')-1
    dv=np.diff(old['velocities_m_s'][:,0],axis=0)/dt
    errors=csv[:20,7:10]-dv[ix]
    evidence['historical_hermite_vs_interval_delta_v']=dict(
        vector_rmse_m_s2=float(np.sqrt(np.mean(np.sum(errors**2,axis=1)))),
        maximum_m_s2=float(np.linalg.norm(errors,axis=1).max()),
        first_exported=csv[0,7:10].tolist(),first_delta_v=dv[0].tolist(),
        interpretation='Endpoint derivative versus interval-average acceleration; discrepancy is not itself a ground-truth acceleration error.')
    atomic_json(out/'probe_evidence.json',evidence)
    print('Static/data/kinematic checks complete',flush=True)
    model,task,ppo=[read(out/f'{key}.json') for key in ['model','task','ppo']]
    assert model['motion_residual']['enabled'] and model['fullstate_execution']['enabled']
    agent=build_agent(ppo,torch.device('cuda'));checkpoint=torch.load(out/'policy_snapshot.pt',map_location='cuda',weights_only=False)
    agent.policy.set_action_prior(checkpoint.get('action_prior'));agent.policy.load_state_dict(checkpoint['policy'])
    env=PointForceWhipEnvironment(model,task,ppo,batch_size=1,device=torch.device('cuda'))
    batch=sample_batch(env,{**ppo['deployment'],'nominal_fraction':1.},torch.Generator(device='cuda').manual_seed(1729))
    forces,cutoffs=plan_batch(env,agent,batch)
    print('Training planner complete',flush=True)
    training_nominal=dict(success=bool(env.episode_success[0]),cutoff_s=float(cutoffs[0])*.01)
    state=DderState(batch.estimate.positions_m.cpu(),batch.estimate.velocities_m_s.cpu())
    physics=GpuRehearsalPhysics(ForceControlledPointCable.from_mapping(model),state,.01,RehearsalSolvers())
    ui_plan=compile_strike_plan(model,task,ppo,state,lambda o:agent.deterministic_action(o.cuda()).cpu(),physics=physics)
    reference=frozen_reference(env,batch,forces,cutoffs)
    evidence['training_vs_ui_plan']=dict(training_steps=int(cutoffs[0]),ui_steps=len(ui_plan.forces_world_n),
        force_max_abs_difference_n=float(abs(forces[:len(ui_plan.forces_world_n),0].cpu()-ui_plan.forces_world_n).max()),
        training_nominal=training_nominal,checkpoint_episodes=int(checkpoint['episodes']))
    trace=[]
    score=execute_batch(env,batch,forces,cutoffs,ppo['deployment'],trace=lambda i,f,s,active,hit:trace.append(s.positions_m.clone()))
    evidence['complete_prediction']=dict(success=bool(score.episode_success[0]),failed=bool(score.failed[0]),
        hit_time_s=float(score.episode_hit_time_s[0]) if bool(score.episode_success[0]) else None,
        no_recovery_objective=float(score.episode_component_sums['recovery'][0])==0)
    # Direct proof that the deployment initializer differs from randomized training estimates.
    varied=sample_batch(env,{**ppo['deployment'],'nominal_fraction':0.},torch.Generator(device='cuda').manual_seed(1729))
    straight=env.model.hanging_state(varied.estimate.positions_m[:,0],varied.estimate.velocities_m_s[:,0])
    evidence['initial_cable_information']=dict(training_estimate_vs_assumed_hanging_tip_m=float((varied.estimate.positions_m[:,-1]-straight.positions_m[:,-1]).norm()),
        training_estimate_vs_assumed_hanging_tip_velocity_m_s=float((varied.estimate.velocities_m_s[:,-1]-straight.velocities_m_s[:,-1]).norm()))
    packets=np.column_stack(sample_fullstate(np.arange(reference[0].shape[1])*.01,reference[0][0].cpu().numpy(),reference[1][0].cpu().numpy())[1:])
    z=packets[:,8]+9.80665;tilt=np.degrees(np.arctan2(np.linalg.norm(packets[:,6:8],axis=1),z))
    evidence['snapshot_reference_feasibility_surrogates']=dict(min_a_plus_g_z=float(z.min()),max_tilt_deg=float(tilt.max()),
        max_acceleration_m_s2=float(np.linalg.norm(packets[:,6:9],axis=1).max()),
        includes_cable_reaction_or_motor_constraints=False)
    np.savez_compressed(out/'snapshot_prediction.npz',forces=forces.cpu().numpy(),
        virtual_root=reference[0].cpu().numpy(),predicted_execution=np.stack([batch.truth.positions_m.cpu().numpy(),*[x.cpu().numpy() for x in trace]]),
        packets=packets)
    evidence['hardware']=dict(os='Windows',gpu=torch.cuda.get_device_name(),training_left_running=True)
    atomic_json(out/'probe_evidence.json',evidence)
    print(json.dumps(evidence,indent=2),flush=True)


if __name__=='__main__':main()
