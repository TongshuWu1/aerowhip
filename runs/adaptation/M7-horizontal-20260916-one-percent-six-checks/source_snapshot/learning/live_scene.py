"""Observe actual PPO collections without changing their actions or rewards.

The last two completed batches form a bounded visualization cache. Small history
records are retained; normal PPO attempt/checkpoint artifacts remain authoritative.
"""
from pathlib import Path
import uuid
import numpy as np
import torch
from experimental_data.io import atomic_json, sha256_file, canonical_json_hash
from simulator.artifact_io import replace_with_retry
from simulator.research_pose import settled_initial


class TrainingSceneRecorder:
    def __init__(self, context):
        self.context = context

    def begin(self, env, batch, cutoffs):
        self.env, self.batch, self.cutoffs = env, batch, cutoffs
        self.positions = [batch.truth.positions_m.clone()]
        self.hits = [torch.zeros(env.batch_size, device=env.device, dtype=torch.bool)]

    def trace(self, index, command, state, striking, hit):
        self.positions.append(state.positions_m.clone())
        self.hits.append(hit.clone())

    def publish(self, agent, score, rollout, forces):
        env, batch = self.env, self.batch
        context = self.context
        folder = Path(context['artifact']) / 'live_scene'
        folder.mkdir(exist_ok=True)
        slot = folder / f'slot_{context["batch_index"] % 2}'
        slot.mkdir(exist_ok=True)
        q = torch.stack(self.positions).cpu().numpy()
        offset = np.asarray(env.model_config['recorded_data']['optitrack_to_attachment_offset_body_m'])
        initial = settled_initial(batch.truth.positions_m[:,0],batch.truth.velocities_m_s[:,0],offset)
        origin = np.repeat(initial.position.cpu().numpy()[None],len(q),axis=0)
        rotation = np.repeat(initial.rotation.cpu().numpy()[None],len(q),axis=0)
        valid = np.ones(q.shape[:2],dtype=bool)
        if hasattr(score,'predicted_pose'):
            pose=score.predicted_pose
            origin=pose['position_origin_m'].cpu().numpy().transpose(1,0,2).copy()
            rotation=pose['rotation_tracking_to_world'].cpu().numpy().transpose(1,0,2,3).copy()
            valid=pose['valid'].cpu().numpy().T.copy()
        cutoff=getattr(score,'terminal_steps',self.cutoffs).cpu().numpy()
        index=np.minimum(np.arange(len(q))[:,None],cutoff[None]);columns=np.arange(env.batch_size)[None]
        origin,rotation,valid=origin[index,columns],rotation[index,columns],valid[index,columns]
        error=np.linalg.norm(origin+np.einsum('tbij,j->tbi',rotation,offset)-q[:,:,0],axis=-1)
        valid &= np.isfinite(q).all(axis=(2,3)) & np.isfinite(error) & (error<1e-7)
        valid[0]=True
        valid=np.logical_and.accumulate(valid,axis=0)
        last=np.maximum.accumulate(np.where(valid,np.arange(len(q))[:,None],0),axis=0)
        q,origin,rotation=q[last,columns],origin[last,columns],rotation[last,columns]
        reward=score.episode_reward.cpu().numpy()
        success=score.episode_success.cpu().numpy()
        arrays=dict(cable=q,origin=origin,rotation=rotation,valid=valid,
            hit=torch.stack(self.hits).cpu().numpy() & success[None],
            target=batch.target_position_m.cpu().numpy(),cutoff=cutoff,success=success,
            failed=score.failed.cpu().numpy(),reward=reward,attachment_offset_m=offset,
            time_s=np.arange(len(q))*env.physics_dt_s,virtual_force_n=forces.cpu().numpy(),
            policy_actions=rollout.actions.cpu().numpy(),ppo_rewards=rollout.rewards.cpu().numpy())
        checkpoint=agent.checkpoint()
        checkpoint['episodes']=context['episodes_before']
        temporary=slot/'policy.tmp'
        torch.save(checkpoint,temporary);replace_with_retry(temporary,slot/'policy.pt')
        temporary=slot/'replay.tmp'
        with temporary.open('wb') as stream:np.savez_compressed(stream,**arrays)
        replace_with_retry(temporary,slot/'replay.npz')
        for name,value in [('model',env.model_config),('task',env.task_config),('ppo',env.ppo_config)]:
            atomic_json(slot/f'{name}.json',value)
        generation=uuid.uuid4().hex
        metadata=dict(schema='multidrone_replay_v1',generation=generation,live_training=True,
            run=str(Path(context['artifact']).resolve()),num_envs=env.batch_size,
            training_attempts=context['episodes_before'],batch_index=context['batch_index'],
            batch_start_attempt=context['episodes_before']+1,batch_end_attempt=context['episodes_before']+env.batch_size,
            collector_gradient_updates=int(agent.gradient_updates),checkpoint_sha256=sha256_file(slot/'policy.pt'),
            replay_sha256=sha256_file(slot/'replay.npz'),target_radius_m=float(env.target_radius_m),
            success_count=int(success.sum()),failure_count=int(arrays['failed'].sum()),
            mean_episode_reward=float(reward.mean()),reward_component_means={k:float(v.mean()) for k,v in score.episode_component_sums.items()},
            ppo_reward_cast_max_error=float(np.max(np.abs(arrays['ppo_rewards'].sum(axis=0)[:,0]-reward))),
            config_sha256={k:canonical_json_hash(v) for k,v in [('model',env.model_config),('task',env.task_config),('ppo',env.ppo_config)]},
            recording='Actual exploratory PPO training collection; latest completed batch, not a second rollout',
            physics='Current calibrated drone + cable + both residuals; shared PPO motion and reward source',
            update_state='Collected before PPO update; see run status for optimization progress',
            frame='World XYZ metres; tracked origin O; cable attachment A=O+R*offset')
        atomic_json(slot/'replay.json',metadata)
        # Pointer is the commit: a renderer never sees a partially published batch.
        atomic_json(folder/'latest.json',dict(generation=generation,slot=slot.name,batch_index=context['batch_index']))
        with (folder/'history.jsonl').open('a',encoding='utf-8') as stream:
            import json
            stream.write(json.dumps(metadata)+'\n')
        return metadata
