"""Standard-physics verification and early braking for the separate new strike."""
from pathlib import Path
from copy import deepcopy
from numpy.polynomial import polynomial as poly
import sys,csv,json,shutil,math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from simulator.workflow import read_json
from experimental_data.io import atomic_json,sha256_file
from experimental_data.model_evaluation import model_identity,load_catalog
from experimental_data.whip_adaptation import verify_hashes
from planning.pva_job import freeze_model_assets
from planning.strike_objective import StrikeCapture,strike_angle_deg
from planning.reference_correction import CoupledRollout,command_valid
from learning.pva_env import PVAEnvironment
from deployment.braking_recovery import complete_packets
from deployment.research_rehearsal import FIELDS

torch.set_num_threads(4)
record=read_json(Path(sys.argv[1]));source=Path(record['job'])
job=ROOT/'runs/recovery/Right-to-Left-Target-Quick-Brake'
output=Path(record['output']);export=Path(record['export'])
assert not any(p.exists() for p in (job,output,export))
assert read_json(source/'status.json')['status']=='completed'
old_csv=ROOT/'exports/M7_Horizontal_Command_Correction/fullstate_30hz.csv'
original_m7_hash=sha256_file(old_csv)
hashes={str(source/n):sha256_file(source/n) for n in ('model.json','plan.npz','settings.json','result.json')}
cfg=read_json(source/'settings.json');cfg['performance']=dict(fused_ticks=False,fast_solve=False,fast_geometry=False)
model=read_json(source/'model.json');job.mkdir(parents=True)
frozen=freeze_model_assets(model,job,source_root=source,portable=True)
atomic_json(job/'model.json',frozen);atomic_json(job/'source_settings.json',cfg);atomic_json(job/'source_hashes.json',hashes)
shutil.copy2(__file__,job/'export_right_to_left_strike.py')
atomic_json(job/'status.json',dict(status='running',stage='Standard replay and quick braking'))
with np.load(source/'plan.npz') as z:actions=z['normalized_jerk'].copy()
env=PVAEnvironment(frozen,cfg,root=job,device='cuda')
with torch.no_grad():
    result=env.rollout(actions=env.tensor(actions)[None],trace=True)
    score,terms=StrikeCapture().score(env,result,env.tensor(actions)[None],cfg['trajectory_objective'])
assert np.isfinite(float(score[0]))
strike=float(env.strike_time[0]);strike_distance=float(env.strike_distance[0]);angle=float(strike_angle_deg(env.strike_velocity[0],env.direction))
assert strike_distance<=.02,('Target miss',strike_distance)
assert angle<=15. and float(env.strike_velocity[0,0])<0
cut=int(math.ceil(strike*30-1e-9));handover=cut/30
assert cut<len(result['packets'][0])
whip=result['packets'][0,:cut+1].cpu().numpy()
settings=deepcopy(cfg['recovery']);settings.update(minimum_brake_s=1/30,hold_s=6.)
t,packets,phases,recovery=complete_packets(whip,cfg['launch']['origin_m'],cfg['limits'],cfg['action']['jerk_limit_m_s3'],settings)
assert bool(command_valid(torch.tensor(packets)[None],cfg['limits']).all())
assert recovery['brake_end_s']<=1.,('Brake longer than one second',recovery['brake_end_s'])
for d in range(3):
    bc=poly.polyder(np.array(recovery['brake_coefficients_normalized']),m=d,axis=0)/recovery['brake_end_s']**d
    rc=poly.polyder(np.array(recovery['return_coefficients_normalized']),m=d,axis=0)/recovery['return_s']**d
    np.testing.assert_allclose(poly.polyval(0.,bc),whip[-1,3*d:3*d+3],atol=1e-9)
    np.testing.assert_allclose(poly.polyval(1.,bc),poly.polyval(0.,rc),atol=1e-9)
grid=np.arange(round(t[-1]/env.dt)+1)*env.dt
rollout=CoupledRollout(env.engine,1,cfg['launch']['origin_m'],env.initial_state.positions_m[0],cfg['limits'],production=True)
print('Strike',strike,'braking starts',handover,'brake duration',recovery['brake_end_s'],flush=True)
with torch.no_grad():prediction=rollout(env.tensor(packets)[None],t,grid)
q=prediction['cable_positions_m'][0].cpu().numpy();v=prediction['cable_velocities_m_s'][0].cpu().numpy()
origin=prediction['position_origin_m'][0].cpu().numpy();ov=prediction['velocity_origin_m_s'][0].cpu().numpy()
np.savez_compressed(job/'full_replay.npz',time_s=grid,cable_positions_m=q,origin_positions_m=origin,origin_velocities_m_s=ov)
start=np.array(cfg['launch']['origin_m']);target=np.array(cfg['launch']['target_m'])
radius=np.linalg.norm(origin[:,:2]-start[:2],axis=1)
prefix=round(handover/env.dt)+1
oldq=np.stack([env.initial_state.positions_m[0].cpu().numpy()]+[x['cable'][0].cpu().numpy() for x in env.frames])
np.testing.assert_allclose(q[:prefix],oldq[:prefix],atol=1e-8,rtol=0)
np.testing.assert_array_equal(packets[:cut+1],whip)
brake_start=round(handover/env.dt);brake_end=handover+recovery['brake_end_s']
brakemask=(grid>=handover-1e-10)&(grid<=brake_end+1.)
final_error=float(np.linalg.norm(origin[-1]-start))
review=dict(model_id='M7',target_m=target.tolist(),starting_hover_m=start.tolist(),
    desired_tip_direction=[-1.,0.,0.],strike_time_s=strike,strike_distance_m=strike_distance,
    tip_velocity_m_s=env.strike_velocity[0].cpu().tolist(),tip_speed_m_s=float(env.strike_velocity[0].norm()),strike_angle_deg=angle,
    braking_start_s=handover,braking_duration_s=recovery['brake_end_s'],
    maximum_horizontal_distance_from_hover_m=float(radius.max()),
    predicted_vehicle_xyz_min_m=origin.min(0).tolist(),predicted_vehicle_xyz_max_m=origin.max(0).tolist(),
    maximum_distance_from_braking_start_m=float(np.linalg.norm(origin[brakemask,:2]-origin[brake_start,:2],axis=1).max()),
    minimum_cable_height_m=float(q[:,:,2].min()),final_origin_error_m=final_error,
    full_replay_valid=bool(prediction['complete_valid'].all()),original_M7_csv_sha256=original_m7_hash)
atomic_json(job/'motion_review.json',review);print(json.dumps(review,indent=2),flush=True)
assert review['full_replay_valid'],'Full predicted recovery violates saved limits'
assert final_error<.03,('Final hover not settled',final_error)
assert radius.max()<2.,('Excursion still too large for the requested compact motion',float(radius.max()))
np.testing.assert_allclose(packets[-1,:3],start,atol=1e-10,rtol=0);np.testing.assert_allclose(packets[-1,3:],0,atol=1e-10,rtol=0)
cfg['method']='mppi_with_post_strike_braking';cfg['task']['duration_s']=handover;cfg['recovery']=settings
cfg['post_strike_braking']=dict(source_job=str(source),full_search_horizon_s=len(actions)/30,braking_start_s=handover,
    strike_preserved=True,post_strike_command_replaced=True)
atomic_json(job/'settings.json',cfg)
output.mkdir(parents=True);outmodel=freeze_model_assets(frozen,output,source_root=job,portable=True)
atomic_json(output/'model.json',outmodel);atomic_json(output/'settings.json',cfg)
assert model_identity(output/'model.json')[0]==model_identity(source/'model.json')[0]
arrays=dict(command_time_s=t,commands=packets,command_phase=phases,prediction_time_s=grid,
    cable_positions_m=q,cable_velocities_m_s=v,origin_positions_m=origin,origin_velocities_m_s=ov,
    origin_rotations=prediction['rotation_tracking_to_world'][0].cpu().numpy(),target_position_m=target,
    jerk_time_s=np.arange(cut)/30,jerk_m_s3=actions[:cut]*np.array(cfg['action']['jerk_limit_m_s3']))
np.savez_compressed(output/'rehearsal.npz',**arrays)
with (output/'fullstate_30hz.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.writer(f);writer.writerow(FIELDS);writer.writerows(np.c_[t,packets])
np.savez_compressed(output/'plan.npz',normalized_jerk=actions[:cut],command_packets=packets,command_time_s=t,
    source_whip_command_packets=whip,braking_start_s=handover,plan_complete=True,planner_mode='mppi_with_post_strike_braking',command_contract=cfg['command_contract'])
summary=dict(schema='pva_fullstate_30hz_v1',planner='Right-to-left target strike with quick brake (M7 predictor)',
    planner_mode='open_loop',command_contract=cfg['command_contract'],objective_schema='targeted_fold_strike_v1',job=str(job),checkpoint=None,
    initial_tracking_origin_m=start.tolist(),target_position_m=target.tolist(),whip_end_s=handover,total_duration_s=float(t[-1]),
    strike_time_s=strike,strike_distance_m=strike_distance,minimum_tip_distance_m=float(np.linalg.norm(q[:prefix,-1]-target,axis=-1).min()),
    predicted_valid_hit=False,predicted_hit_time_s=None,predicted_fold_valid=bool(result['fold_valid'][0]),fold_requirement='diagnostic_only',
    reference_feasible=True,predicted_attitude_checked=True,recovery_prediction_complete=True,prediction_valid_through_s=float(grid[-1]),recovery=recovery,
    maximum_predicted_tilt_limit_deg=cfg['limits']['maximum_tilt_deg'],minimum_predicted_cable_height_m=review['minimum_cable_height_m'],
    command_semantics='Desired tracked-origin P/V/A, 30 Hz zero-order hold, kinematic acceleration, zero yaw',
    evidence='Standard-physics simulation only; no physical flight or impact-force simulation',
    source_planning_job=str(source),active_flight_selection_changed=False,csv_sha256=sha256_file(output/'fullstate_30hz.csv'),
    strike_angle_deg=angle,maximum_strike_angle_deg=15.,model_id='M7')
atomic_json(output/'rehearsal.json',summary);atomic_json(output/'motion_review.json',review)
atomic_json(output/'task.json',dict(desired_strike_direction_world=[-1.,0.,0.],target_position_m=target.tolist(),target_marker_radius_m=.02,
    acceptance='Predicted tip passage within 2 cm, within 15 degrees of -X; no physical contact-force model'))
verify_hashes(hashes);assert sha256_file(old_csv)==original_m7_hash
atomic_json(output/'verification.json',dict(original_M7_unchanged=True,command_prefix_unchanged=True,
    strike_prediction_preserved=True,full_standard_replay_passed=True,pva_continuity_verified=True,rate_hz=30,rows=len(packets),physical_flight_tested=False))
def at(a,moment):return np.array([np.interp(moment,grid,a.reshape(len(grid),-1)[:,k]) for k in range(a[0].size)]).reshape(a.shape[1:])
fig,axes=plt.subplots(1,2,figsize=(10,4.5));fig.subplots_adjust(bottom=.23,top=.8,wspace=.3)
end=grid<=handover
for ax,(a,b),title in zip(axes,[(0,1),(0,2)],['Top view: right to left (−X)','Side view: height']):
    ax.plot(origin[:,a],origin[:,b],color='#327ca6',lw=1.5,label='Predicted vehicle, full motion')
    ax.plot(q[end,-1,a],q[end,-1,b],'--',color='#d55e00',label='Predicted tip, strike phase')
    qs=at(q,strike);ax.plot(qs[:,a],qs[:,b],'-o',color='#d55e00',ms=3,lw=2,label='Cable at strike')
    ax.scatter(target[a],target[b],c='#198657',marker='x',s=80,label='Target',zorder=5)
    ax.set(xlabel='XYZ'[a]+' (m)',ylabel='XYZ'[b]+' (m)',title=title);ax.set_aspect('equal');ax.grid(alpha=.2)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=8)
fig.suptitle(f'New right-to-left strike — simulation\nTarget (0, 1.25, 1.3) m · Start (0, 0, 1.5) m · Brake {recovery["brake_end_s"]:.2f} s')
fig.savefig(output/'strike-preview.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(9,4.5),layout='constrained');curves=[];traces=[];markers=[]
indices=np.flatnonzero(grid<=brake_end+1.)[::5];points=q[indices].reshape(-1,3)
for ax,(a,b),title in zip(axes,[(0,1),(0,2)],['Top view: right to left (−X)','Side view']):
    low=np.minimum(points.min(0),target)-.12;high=np.maximum(points.max(0),target)+.12
    ax.scatter(target[a],target[b],c='#198657',marker='x',s=65)
    curves.append(ax.plot([],[],'o-',c='#d55e00',lw=2,ms=3)[0]);traces.append(ax.plot([],[],'--',c='#d55e00',alpha=.4)[0])
    markers.append(ax.plot([],[],'o',c='#327ca6',ms=7)[0])
    ax.set(xlim=(low[a],high[a]),ylim=(low[b],high[b]),xlabel='XYZ'[a]+' (m)',ylabel='XYZ'[b]+' (m)',title=title);ax.set_aspect('equal');ax.grid(alpha=.2)
title=fig.suptitle('')
def frame(i):
    for (a,b),line,trace,marker in zip([(0,1),(0,2)],curves,traces,markers):
        line.set_data(q[i,:,a],q[i,:,b]);trace.set_data(q[:i+1,-1,a],q[:i+1,-1,b]);marker.set_data([origin[i,a]],[origin[i,b]])
    phase='Strike' if grid[i]<=handover else 'Braking' if grid[i]<=brake_end else 'Return'
    title.set_text(f'New right-to-left strike — simulation · {grid[i]:.2f} s · {phase}')
    return curves+traces+markers+[title]
animation=FuncAnimation(fig,frame,frames=indices,interval=50,blit=False)
animation.save(output/'strike-preview.gif',writer=PillowWriter(fps=20),dpi=100);plt.close(fig)
shutil.copytree(output,export);(export/'flight_take').mkdir()
atomic_json(export/'export.json',dict(summary,rehearsal=str(output),operation='New target; original M7 command unchanged',motion_review=review))
(export/'README.md').write_text('# New right-to-left strike with quick braking\n\n'
    'Uses the M7 predictor; the original M7 horizontal command is unchanged. '
    'Starting tracked-origin hover: (0, 0, 1.5) m. Target: (0, 1.25, 1.3) m. Desired tip direction: -X.\n\n'
    f'Predicted miss {1000*strike_distance:.2f} mm; tip speed {review["tip_speed_m_s"]:.2f} m/s; direction error {angle:.2f} degrees. '
    f'Strike at {strike:.4f} s, braking begins {handover:.4f} s, commanded deceleration lasts {recovery["brake_end_s"]:.3f} s.\n\n'
    f'Maximum predicted vehicle distance from starting hover: {radius.max():.3f} m horizontally. '
    f'Maximum predicted distance from brake-start position during braking and the following second: {review["maximum_distance_from_braking_start_m"]:.3f} m.\n\n'
    f'Complete CSV: {len(packets)} rows at 30 Hz, {t[-1]:.3f} s, including braking, return, and final hold. '
    'This is not a takeoff command.\n\n'
    'Simulation checks passed; real-flight performance is untested. Store new-task recordings under flight_take. '
    'Active flight selection has not been changed.\n',encoding='utf-8')
atomic_json(job/'result.json',summary);atomic_json(job/'status.json',dict(status='completed',rehearsal=str(output),export=str(export)))
print('EXPORTED',export/'fullstate_30hz.csv',flush=True)
