"""Read-only recovery boundary analysis for a saved MPPI plan."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
from simulator.workflow import read_json
from learning.pva_env import PVAEnvironment
from deployment.pva_rehearsal import complete_pva_packets
from deployment.curved_recovery import plan_curved_recovery
from experimental_data.io import atomic_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    cfg=read_json(args.job/'settings.json');env=PVAEnvironment(read_json(args.job/'model.json'),cfg,root=args.job)
    with np.load(args.job/'plan.npz') as plan:result=env.rollout(actions=env.tensor(plan['normalized_jerk'])[None])
    whip=result['packets'][0,:int(result['cutoffs'][0])+1].cpu().numpy();exit=whip[-1];hover=np.array(cfg['launch']['origin_m'])
    limits=cfg['limits'];options=dict(maximum_downward_acceleration_m_s2=9.80665-limits['minimum_specific_vertical_m_s2'],
        maximum_horizontal_acceleration_m_s2=limits['maximum_specific_force_m_s2'],maximum_descent_speed_m_s=limits['maximum_speed_m_s'],
        maximum_tilt_deg=limits['maximum_tilt_deg'],maximum_speed_m_s=limits['maximum_speed_m_s'],
        maximum_specific_force_m_s2=limits['maximum_specific_force_m_s2'],minimum_specific_vertical_m_s2=limits['minimum_specific_vertical_m_s2'])
    report=dict(exit=exit.tolist(),cutoff_s=(len(whip)-1)/30,options=options)
    try:
        sample,info=plan_curved_recovery(exit[:3],exit[3:6],exit[6:9],hover,options);report['recovery']=info
    except ValueError as e:report['error']=str(e)
    atomic_json(args.output,report);print(report)
