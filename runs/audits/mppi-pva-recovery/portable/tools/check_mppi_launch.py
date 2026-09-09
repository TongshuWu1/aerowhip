"""Compare old/new seed initialization through M1; no optimization or export."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from simulator.workflow import read_json
from experimental_data.io import atomic_json,sha256_file
from planning.cem_run import MPPI_DEFAULTS
from planning.cem_objective import task_with_settings
from planning.cem_execution import SplineEvaluator
from planning.spline import fit_seed,initialize_at_origin

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    seed=root/'runs/rehearsals/20260908-203914-039721'
    output=root/'runs/audits/20260908-mppi-launch-consistency';output.mkdir(parents=True,exist_ok=True)
    settings=dict(MPPI_DEFAULTS,**read_json(root/'config/mppi.json'))
    model_path=root/settings['model_path'];model=read_json(model_path)
    task=task_with_settings(read_json(seed/'task.json'),settings)
    origin=np.asarray(settings['launch_setup']['initial_tracking_origin_m'])
    task['initial_root_position_m']=(origin+model['recorded_data']['optitrack_to_attachment_offset_body_m']).tolist()
    task['target_position_m']=settings['launch_setup']['target_position_m']
    config=read_json(seed/'ppo.json');meta=read_json(seed/'rehearsal.json')
    duration=meta['whip_end_s']
    with np.load(seed/'rehearsal.npz') as data:packets=data['commands'][:int(round(duration*30))+1].copy()
    before=sha256_file(seed/'rehearsal.npz')
    old=fit_seed(packets,duration,settings['control_points']);old[:3]=origin
    new=initialize_at_origin(packets,duration,settings['control_points'],origin)
    evaluator=SplineEvaluator(model,task,config,settings,'cuda')
    scores,diagnostics=evaluator(np.array([np.r_[old[3:].ravel(),duration],np.r_[new[3:].ravel(),duration]]))
    result=dict(old_initialization_feasible=bool(np.isfinite(scores[0])),
        corrected_initialization_feasible=bool(np.isfinite(scores[1])),corrected_score=float(scores[1]) if np.isfinite(scores[1]) else None,
        launch=settings['launch_setup'],model_sha256=sha256_file(model_path),diagnostics=diagnostics,
        seed_unchanged=sha256_file(seed/'rehearsal.npz')==before,optimized=False,exported=False)
    atomic_json(output/'verification.json',result);print(result)
    assert result['corrected_initialization_feasible'] and result['seed_unchanged']
