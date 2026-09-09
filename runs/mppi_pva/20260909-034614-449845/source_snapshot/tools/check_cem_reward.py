"""Verify configurable CEM rewards and hit criteria with a saved spline on CUDA."""
from pathlib import Path
import argparse
import copy
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from experimental_data.io import atomic_json
from simulator.workflow import read_json
from planning.cem_execution import SplineEvaluator
from planning.cem_objective import task_with_settings

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--result',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();folder=Path(args.result).resolve()
    model,task,ppo,settings=[read_json(folder/f'{n}.json') for n in ('model','task','ppo','cem')]
    for name,file in [('motion_residual','cable_residual.pt'),('fullstate_execution','drone_model.json')]:model[name]['checkpoint']=str(folder/'assets'/file)
    with np.load(folder/'spline.npz') as data:vector=data['vector'][None]
    evaluator=SplineEvaluator(model,task,ppo,settings,'cuda')
    baseline,base_info=evaluator(vector)
    evaluator.settings['reward']={'time_weight_per_s':0}
    no_time,no_time_info=evaluator(vector)
    expected=25*read_json(folder/'rehearsal.json')['predicted_hit_time_s']
    np.testing.assert_allclose(no_time-baseline,[expected],atol=1e-6)
    strict=task_with_settings(task,{'success':{'minimum_directed_tip_speed_m_s':100.}})
    strict_evaluator=SplineEvaluator(model,strict,ppo,copy.deepcopy(settings),'cuda')
    _,strict_info=strict_evaluator(vector)
    assert base_info['success_fraction']==1 and strict_info['success_fraction']==0
    report=dict(baseline_score=float(baseline[0]),no_time_cost_score=float(no_time[0]),
                measured_score_change=float(no_time[0]-baseline[0]),expected_score_change=expected,
                baseline_hit_fraction=base_info['success_fraction'],strict_speed_hit_fraction=strict_info['success_fraction'],
                device='Windows / RTX 4080 / CUDA',source=str(folder))
    atomic_json(Path(args.output),report);print(report)
