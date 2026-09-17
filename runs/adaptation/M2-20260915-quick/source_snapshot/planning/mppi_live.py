"""Read-only 30 Hz snapshots from already evaluated MPPI batches.

No extra rollout, RNG draw, action change, or trace=True slow path. Latest live
snapshots are replaceable display telemetry; history/committed plans are durable.
"""
from pathlib import Path
import time
import numpy as np
import torch
from simulator.artifact_io import replace_with_retry


class RolloutCapture:
    def __init__(self):self.frames=[]
    def __call__(self,env):
        self.frames.append((env.pose.position.clone(),env.pose.rotation.clone(),env.state.positions_m.clone(),env.command[:,:3].clone()))

    def select(self,ids):
        ids=torch.as_tensor(ids,device=self.frames[0][0].device,dtype=torch.long)
        if not hasattr(self,'stacked'):self.stacked=[torch.stack([frame[j] for frame in self.frames],1) for j in range(4)]
        return {key:self.stacked[j][ids].cpu().numpy()
            for j,key in enumerate(('origin_positions_m','origin_rotations','cable_positions_m','command_positions_m'))}


def series(capture,index,iteration,score,result,label):
    arrays={key:value[0] for key,value in capture.select([index]).items()}
    extra={}
    if 'target_hit_times_s' in result:extra['target_hit_times_s']=result['target_hit_times_s'][index].cpu().numpy()
    return dict(arrays,**extra,iteration=iteration,candidate_id=index,score=float(score[index]),
        failed=bool(result['failed'][index]),success=bool(result['success'][index]),
        termination_time_s=float(result['duration_s'][index]),label=label)


def publish(job,actual,capture,score,result,best,iteration,*,labels=None):
    started=time.perf_counter();count=len(score)
    ids=list(dict.fromkeys([*torch.topk(score,min(3,count)).indices.cpu().tolist(),*np.linspace(0,count-1,min(4,count),dtype=int).tolist(),count-1]))
    rows=[best]+[series(capture,i,iteration,score,result,(labels or {}).get(i,'Evaluated mean' if i==count-1 else f'Sample {i}')) for i in ids
        if best['iteration']!=iteration or i!=best['candidate_id']]
    length=max(len(r['origin_positions_m']) for r in rows)
    payload={key:np.stack([np.concatenate([r[key],np.repeat(r[key][-1:],length-len(r[key]),axis=0)],axis=0) for r in rows])
        for key in ('origin_positions_m','origin_rotations','cable_positions_m','command_positions_m')}
    payload.update(schema=np.array('mppi_live_v1'),iteration=iteration,command_step=actual.index,created_unix_s=time.time(),
        time_s=actual.index/30+np.arange(length)/30,frame_counts=np.array([len(r['origin_positions_m']) for r in rows]),
        labels=np.array([r['label'] for r in rows]),series_iterations=np.array([r['iteration'] for r in rows]),
        candidate_ids=np.array([r['candidate_id'] for r in rows]),scores=np.array([r['score'] for r in rows]),
        failed=np.array([r['failed'] for r in rows]),success=np.array([r['success'] for r in rows]),
        termination_time_s=np.array([r['termination_time_s'] for r in rows]),target_position_m=actual.target[0].cpu().numpy(),
        committed_origin_m=np.stack([actual.initial_pose.position[0].cpu().numpy()]+[f['origin'][0].cpu().numpy() for f in actual.frames]),
        committed_tip_m=np.stack([actual.initial_state.positions_m[0,-1].cpu().numpy()]+[f['cable'][0,-1].cpu().numpy() for f in actual.frames]))
    if actual.settings['task'].get('target_sequence_m'):
        payload.update(target_positions_m=np.asarray(actual.settings['task']['target_sequence_m']),
            target_hit_times_s=np.stack([r['target_hit_times_s'] for r in rows]))
    temporary=Path(job)/'live.tmp.npz';np.savez_compressed(temporary,**payload);replace_with_retry(temporary,Path(job)/'live.npz')
    return time.perf_counter()-started
