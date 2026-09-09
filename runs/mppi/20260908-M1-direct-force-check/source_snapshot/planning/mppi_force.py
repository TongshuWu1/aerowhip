"""Direct 30 Hz PPO action sequences, evaluated by the unmodified PPO pipeline."""
import numpy as np
import torch
from learning.point_force_env import PointForceWhipEnvironment
from learning.deployment_rollout import DeploymentBatch,plan_batch
from learning.research_rollout import execute_research_batch


class SequenceAgent:
    """Supplies one independently optimized XYZ action at each control tick."""
    def __init__(self,actions):self.actions=actions;self.index=0
    def deterministic_action(self,observation):
        action=self.actions[:,self.index];self.index+=1
        return action


class ForceEvaluator:
    def __init__(self,model,task,config,settings,device='cuda',cancelled=None):
        self.model,self.task,self.config,self.settings=model,task,config,settings
        self.device=torch.device(device);self.cancelled=cancelled;self.contexts={}
        self.steps=round(task['episode_duration_s']/task['control_dt_s'])

    def check_cancelled(self,*_):
        if self.cancelled and self.cancelled():
            from .cem_execution import Cancelled
            raise Cancelled('Stopped by user; candidate history is preserved.')

    def context(self,batch_size):
        if batch_size not in self.contexts:
            self.contexts[batch_size]=PointForceWhipEnvironment(self.model,self.task,self.config,
                batch_size=batch_size,device=self.device)
        return self.contexts[batch_size]

    @torch.no_grad()
    def evaluate(self,vectors,record=False):
        self.check_cancelled();vectors=np.asarray(vectors,float)
        if vectors.ndim!=2 or vectors.shape[1]!=self.steps*3 or not np.isfinite(vectors).all():
            raise ValueError('Expected a finite batch of 30 Hz XYZ action sequences.')
        env=self.context(len(vectors));env.reset();state=env.state
        one=state.positions_m.new_ones(len(vectors))
        batch=DeploymentBatch(state,state,one,one,one[:,None].expand(-1,3),one[:,None]*0,
            torch.ones(len(vectors),dtype=torch.bool,device=env.device),env.target.clone())
        # Same clipping, gravity compensation, force norm and vertical limits
        # as PPO's physical_force; do not substitute an acceleration controller.
        actions=state.positions_m.new_tensor(vectors.reshape(len(vectors),self.steps,3)).clamp(-1,1)
        frames=[state.positions_m[0].cpu().numpy().copy()] if record else None
        def trace(i,force,current,*_):frames.append(current.positions_m[0].cpu().numpy().copy())
        forces,cutoffs=plan_batch(env,SequenceAgent(actions),batch,progress=self.check_cancelled)
        score=execute_research_batch(env,batch,forces,cutoffs,self.config['deployment'],
            trace=trace if record else None,progress=self.check_cancelled)
        # Objective is exactly the PPO execution return, including failure costs.
        # This keeps a learning signal when an entire proposal batch is infeasible.
        values=score.episode_reward.cpu().numpy().copy()
        values[~np.isfinite(values)]=-np.inf
        valid=(~score.failed).cpu().numpy()
        diagnostics=dict(success_fraction=float(score.episode_success.double().mean()),
            model_feasible_fraction=float(np.mean(valid)),
            closest_tip_m=float(score.episode_minimum_tip_distance.min()),
            reward_components={k:float(v.mean()) for k,v in score.episode_component_sums.items()})
        return values,diagnostics,dict(env=env,batch=batch,forces=forces,cutoffs=cutoffs,score=score,frames=frames,actions=actions)

    def __call__(self,vectors):
        values,diagnostics,_=self.evaluate(vectors)
        return values,diagnostics

    def seed_actions(self,forces,times):
        env=self.context(1)
        query=np.arange(self.steps)/30
        indices=np.searchsorted(times,query+1e-9,side='right')-1
        seed=np.asarray(forces)[indices.clip(0,len(forces)-1)].copy()
        hover=env.hover_force_world_n[0].cpu().numpy()
        # Beyond the available seed, initialize with hover; never stretch time.
        seed[query>float(times[-1])+1/30-1e-9]=hover
        scale=env.action_delta_scale_n[0].cpu().numpy()
        return np.clip((seed-hover)/scale,-1,1).ravel()
