"""Review saved timing-trial forecast; never run fitting or optimization."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from experimental_data.io import atomic_json
from simulator.workflow import read_json
from tools.prepare_mppi_timed_baselines import AUDIT
from tools.check_preferred_fold_objective import load_case,encounter
from tools import review_preferred_fold_run as review

if __name__=='__main__':
    review.AUDIT=AUDIT
    review.main(trial_title='Timing and strength trial: same M0, unchanged preferred-fold objective\nOld forecast is a style reference; every new candidate was simulated under development M0.')
    status=read_json(AUDIT/'status.json');job=Path(status['job'])
    event=encounter(load_case(Path(status['rehearsal'])))
    report={k:(v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else v)
        for k,v in event.items() if k!='q'}
    report.update(tip_forward_m_s=float(event['tip_v']@event['direction']),
        drone_forward_m_s=float(event['drone_v']@event['direction']),
        angle_deg=float(np.degrees(np.arccos(event['tip_v']@event['direction']/np.linalg.norm(event['tip_v'])))))
    atomic_json(AUDIT/'contact_review.json',report)
    result=read_json(job/'result.json');baselines=read_json(job/'baseline_evaluation.json')
    assert result['best_reward']+1e-7>=max(baselines['scores'])
    atomic_json(AUDIT/'baseline_comparison.json',dict(baseline_scores=baselines['scores'],
        result_score=result['best_reward'],improvement=result['best_reward']-max(baselines['scores']),
        reward_changed=False,model_changed=False,scope='Simulation objective, not physical validation'))
