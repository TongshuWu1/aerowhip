"""Bounded four-attempt integration audit; never registers as an active PPO run."""
import argparse
from pathlib import Path
import sys
import time
import json
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_ppo
from simulator.research_config import workspace_configs
from experimental_data.io import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args()
    folder=Path(args.output).resolve();folder.mkdir(parents=True,exist_ok=True)
    model,task,config=deepcopy(workspace_configs(Path(__file__).resolve().parents[1]))
    task['episode_duration_s']=.1
    config['seed']=989;config['bootstrap']=None
    config['validation']['enabled']=False;config['update_guard']['enabled']=False
    config['ppo']['update_epochs']=1;config['early_stopping']={'enabled':False}
    config['live_scene']={'enabled':True}
    for key,value in [('model',model),('task',task),('ppo',config)]:atomic_json(folder/'config'/f'{key}.json',value)
    run_ppo.write_active_run=lambda *args:None
    original=run_ppo.collect_rollout
    calls=0
    def collect(*args,**kwargs):
        nonlocal calls
        result=original(*args,**kwargs);calls+=1
        if calls==1:
            started=time.monotonic()
            while not (folder/'run/live_scene/viewer_status.json').exists():
                if time.monotonic()-started>90:raise TimeoutError('Start the Isaac live smoke viewer for this audit run.')
                time.sleep(.1)
        return result
    run_ppo.collect_rollout=collect
    run=run_ppo.train(device_name='cuda',requested_episodes=4,batch_size=2,artifact=folder/'run',
        resume_checkpoint=None,config_directory=folder/'config')
    history=[json.loads(line) for line in (run/'live_scene/history.jsonl').read_text().splitlines()]
    assert len(history)==2 and [x['batch_end_attempt'] for x in history]==[2,4]
    assert history[0]['checkpoint_sha256']!=history[1]['checkpoint_sha256']
    atomic_json(folder/'integration.json',dict(attempts=4,updates=2,active_run_pointer_untouched=True,
        batches=[{k:x[k] for k in ('generation','batch_end_attempt','mean_episode_reward','checkpoint_sha256')} for x in history]))
    print('LIVE_TRAINING_AUDIT_OK',flush=True)


if __name__=='__main__':main()
