"""Separate fixed-target right-to-left strike using frozen M7; no flight activation."""
from pathlib import Path
from copy import deepcopy
import sys,itertools,json,shutil
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from simulator.workflow import read_json
from experimental_data.io import atomic_json,sha256_file
from experimental_data.model_evaluation import load_catalog,model_identity
from experimental_data.whip_adaptation import verify_hashes
from planning.pva_job import prepare,run
from planning.strike_objective import StrikeCapture,OBJECTIVE
from planning.strike_mppi import check_recovery_reference
from learning.pva_env import PVAEnvironment

torch.set_num_threads(4)
out=ROOT/'tmp/right_to_left_seed_screen';out.mkdir(exist_ok=False)
entry=next(m for m in load_catalog(ROOT)['models'] if m['id']=='M7')
verify_hashes(entry['hashes']);assert model_identity(entry['model'])[0]==entry['signature']
cfg=read_json(ROOT/'runs/rehearsals_pva/M5-Curved-Side-Whip-1p3m/settings.json')
cfg['model_path']=entry['model']
cfg['launch']=dict(origin_m=[0.,0.,1.5],target_m=[0.,1.25,1.3],start_radius_m=0.,target_radius_m=0.)
cfg['task'].update(duration_s=52/30,strike_direction=[-1.,0.,0.])
cfg['trajectory_objective']=dict(OBJECTIVE,distance_scale_m=.05,maximum_strike_angle_deg=15.,
    prefer_aligned_strike=True,speed_metric='tip_gain_over_root',minimum_tip_speed_gain_m_s=2.)
cfg['mppi'].update(iterations=40,seed=718)
cfg['recovery'].update(minimum_brake_s=1/30,hold_s=6.)
cfg['proposal_templates_path']=str(out/'proposal_baselines.npz')
paths=[ROOT/'runs/rehearsals_pva/M5-Forward-Left-Horizontal-Whip/plan.npz',
       ROOT/'runs/rehearsals_pva/M7-Horizontal-Command-Correction/plan.npz']
with np.load(paths[0]) as z:a=z['normalized_jerk'].copy()
with np.load(paths[1]) as z:b=np.pad(z['normalized_jerk'],((0,6),(0,0)));b[:,0]*=-1
time=(np.arange(52)+.5)/30
def translate_jerk(displacement):
    duration=1.5;u=time/duration
    return np.where(u<=1,displacement*(60-360*u+360*u*u)/duration**3/60,0.)
actions=[];labels=[]
for family,base in [('rotated_forward_left',a),('mirrored_curved',b)]:
    for sx,sy,sz,offset in itertools.product([.4,.6,.8],[.5,.75,1.],[.3,.7],[.3,.7,1.]):
        value=base.copy()*[sx,sy,sz]
        value[:,:2]=np.stack([-value[:,1],value[:,0]],axis=1)
        value[:,0]+=translate_jerk(offset)
        if abs(value).max()>1:continue
        actions.append(value);labels.append(dict(family=family,source_axis_scales=[sx,sy,sz],rightward_preparation_m=offset))
model=read_json(entry['model']);root=Path(entry['model']).parent
env=PVAEnvironment(model,cfg,root=root,batch_size=len(actions),device='cuda')
actual=PVAEnvironment(model,cfg,root=root,device='cuda')
print('Screening',len(actions),'initial command shapes',flush=True)
with torch.no_grad():
    commands=env.tensor(np.stack(actions));result=env.rollout(actions=commands)
    scores,_=StrikeCapture().score(env,result,commands,cfg['trajectory_objective'])
    rows=[];selected=[]
    for i in scores.argsort(descending=True).tolist():
        if not np.isfinite(float(scores[i])):break
        failure=None
        try:check_recovery_reference(result['packets'][i],cfg,actual)
        except ValueError as exc:failure=str(exc)
        rows.append(dict(index=i,score=float(scores[i]),distance_m=float(env.strike_distance[i]),
            strike_time_s=float(env.strike_time[i]),velocity_m_s=env.strike_velocity[i].cpu().tolist(),
            parameters=labels[i],recovery_failure=failure))
        if failure is None:selected.append(i)
        if len(selected)==2:break
    report=dict(candidates=len(actions),physics_valid=int((~result['failed']).sum()),
        qualified=int(torch.isfinite(scores).sum()),selected=selected,review=rows,
        sources={str(p):sha256_file(p) for p in paths})
    atomic_json(out/'review.json',report);atomic_json(out/'settings.json',cfg)
    if not selected:raise ValueError('No eligible seed; retain recorded screen for review')
    if len(selected)==1:selected*=2
np.savez_compressed(out/'proposal_baselines.npz',normalized_jerk=np.stack(actions)[selected])
print(json.dumps(report,indent=2),flush=True)
del env,actual,result,commands;torch.cuda.empty_cache()
job,command=prepare(ROOT,cfg,'New right-to-left strike: target (0,1.25,1.3), hover 1.5 m')
shutil.copy2(out/'proposal_baselines.npz',job/'proposal_baselines.npz')
shutil.copy2(__file__,job/'planning_runner.py')
atomic_json(job/'task_intent.json',dict(model_id='M7',origin_m=cfg['launch']['origin_m'],target_m=cfg['launch']['target_m'],
    desired_tip_direction=[-1.,0.,0.],simulation_target_review_tolerance_m=.02,
    recovery='Final export truncates the command at the first 30 Hz update after the selected strike, then appends the shortest admissible smooth deceleration and independently checks the complete vehicle/cable motion',
    initial_screen=str(out),original_M7_command_unchanged=True,active_flight_selection_changed=False))
atomic_json(ROOT/'tmp/right_to_left_search.json',dict(job=str(job),command=command,
    output=str(ROOT/'runs/rehearsals_pva/Right-to-Left-Target-Y1p25-Z1p3-Quick-Brake'),
    export=str(ROOT/'exports/Right_to_Left_Target_Y1p25_Z1p3_Quick_Brake')))
print('JOB',job,flush=True);run(job)
