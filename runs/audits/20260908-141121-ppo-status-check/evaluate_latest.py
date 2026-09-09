from pathlib import Path
import sys,json,time
root=Path(__file__).resolve().parents[3]
run=root/'runs/ppo/20260908-080833-956336-seed655'
sys.path.insert(0,str(run/'source_snapshot'))
import torch
import run_ppo
from experimental_data.io import atomic_json,sha256_file
from learning.experiment_records import finite_json
model,task,config=run_ppo.load_configs(run)
device=torch.device('cuda')
run_ppo.configure_accelerator(device)
agent=run_ppo.build_agent(config,device)
checkpoint=run/'checkpoints/latest.pt'
run_ppo._load_checkpoint(agent,checkpoint,load_optimizer=False)
started=time.perf_counter()
result=run_ppo.evaluate(model,task,config,agent,episodes=256,batch_size=256,device=device)
result=finite_json(dict(result,checkpoint=str(checkpoint),checkpoint_sha256=sha256_file(checkpoint),training_episodes=167936,assessment='Read-only re-evaluation on the same fixed 256 development scenarios after writer failure; no training update',runtime_s=time.perf_counter()-started))
atomic_json(Path(__file__).with_name('latest-checkpoint-validation.json'),result)
print(json.dumps({k:result[k] for k in ['training_episodes','success_rate','nominal_success_rate','median_minimum_tip_distance_m','nonfinite_rate','reference_infeasible_rate','pose_domain_failure_rate','runtime_s']}),flush=True)
