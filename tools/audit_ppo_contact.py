"""Central-scenario first-contact gates at the same interpolated entry as PVA."""
from pathlib import Path
import argparse
import shutil
import sys
import math
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from simulator.workflow import read_json
from simulator.pva_commands import sphere_entry
from learning.pva_env import PVAEnvironment
from planning.pva_job import load_policy
from experimental_data.io import atomic_json,sha256_file


@torch.no_grad()
def audit(job,out):
    out.mkdir(parents=True,exist_ok=False);checkpoint=out/'policy.pt'
    shutil.copy2(job/'checkpoints/best.pt',checkpoint)
    cfg=read_json(job/'settings.json');task=cfg['task']
    env=PVAEnvironment(read_json(job/'model.json'),cfg,root=job,batch_size=1)
    agent=load_policy(checkpoint,env,cfg);original=env._tick;contacts=[];closest=None
    def tick(reward,left,right,clock_left,clock_cutoff,**kwargs):
        nonlocal closest
        q=env.state.positions_m.clone();v=env.state.velocities_m_s.clone()
        p=env.pose.position.clone();pv=env.pose.velocity.clone()
        was_active=bool(env.active[0]);was_contact=bool(env.contact[0])
        pulled=bool(env.pull_ready[0]);waved=bool(env.wave_stage[0]>=3)
        result=original(reward,left,right,clock_left,clock_cutoff,**kwargs)
        if not was_active or env.failed[0]:return result
        newq=env.state.positions_m;newv=env.state.velocities_m_s
        delta=newq[0,-1]-q[0,-1]
        alpha=float((((env.target[0]-q[0,-1])*delta).sum()/delta.square().sum().clamp_min(1e-20)).clamp(0,1))
        entry=sphere_entry(q,newq,env.target,task['target_radius_m'])[0]
        def describe(fraction):
            position=p[0]+fraction*(env.pose.position[0]-p[0])
            drone=pv[0]+fraction*(env.pose.velocity[0]-pv[0])
            tipv=v[0,-1]+fraction*(newv[0,-1]-v[0,-1])
            tip=q[0,-1]+fraction*delta
            backward=float(env.pull_peak[0]-((position-env.origin0[0])*env.direction).sum())
            speed=float((tipv*env.direction).sum());drone_speed=float((drone*env.direction).sum())
            angle=math.degrees(math.acos(max(-1,min(1,speed/max(float(tipv.norm()),1e-12)))))
            return dict(time_s=left+fraction*env.dt,distance_m=float((tip-env.target[0]).norm()),
                tip=tip.tolist(),tip_velocity=tipv.tolist(),tip_forward_speed=speed,tip_angle_deg=angle,
                drone_forward_speed=drone_speed,backward_distance_m=backward,pull_ready=pulled,wave_ready=waved,
                speed_ok=speed>=task['minimum_directed_speed_m_s'],angle_ok=angle<=task['maximum_angle_deg'],
                reverse_speed_ok=drone_speed<=-task['minimum_backward_speed_m_s'],
                reverse_distance_ok=backward>=task['minimum_backward_distance_m'],first_contact_ok=not was_contact)
        near=describe(alpha)
        if closest is None or near['distance_m']<closest['distance_m']:closest=near
        if torch.isfinite(entry[-1]):
            event=describe(float(entry[-1]));event['tip_before_other_nodes']=bool(entry[-1]<entry[:-1].min())
            event['accepted_hit']=bool(env.success[0]);contacts.append(event)
        return result
    env._tick=tick
    for _ in range(env.steps):
        env.step(agent.deterministic_action(env.observation()),trace=True);env.frames=[]
        if not env.active.any():break
    result=dict(checkpoint_sha256=sha256_file(checkpoint),closest=closest,contacts=contacts,
                success=bool(env.success[0]),failed=bool(env.failed[0]),evidence='Central simulation scenario only')
    atomic_json(out/'result.json',result);print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    audit(args.job,args.output)
