"""Atomic per-attempt batches; row order within a parallel batch is arbitrary."""
import os
from pathlib import Path
import numpy as np


def save_attempts(directory, score, completed, elapsed_s):
    folder=Path(directory)/'attempts';folder.mkdir(exist_ok=True)
    def array(value):return value.detach().cpu().numpy().reshape(-1)
    reward=array(score.episode_reward);count=len(reward)
    data=dict(episode=np.arange(completed-count+1,completed+1,dtype=np.int64),
        batch_end=np.full(count,completed,dtype=np.int64),
        elapsed_s=np.full(count,elapsed_s),reward=reward,
        success=array(score.episode_success),
        impact_speed_m_s=array(score.episode_impact_speed),
        hit_time_s=array(score.episode_hit_time_s),
        numerical_failure=array(score.failed))
    deployment=getattr(score,'deployment',{})
    for name in ('joint_success','recovered','maximum_execution_drone_displacement_m','nominal'):
        if name in deployment:data[name]=array(deployment[name])
    path=folder/f'{completed:010d}.npz';temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream,**data)
    os.replace(temporary,path)
