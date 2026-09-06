"""Soft Actor-Critic for the shared 3D point-force task."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn

def _mlp(n:int, out:int, h:int):
    return nn.Sequential(nn.Linear(n,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,out))

class SACActor(nn.Module):
    def __init__(self, obs:int, actions:int, hidden:int, indices=(0,2), *, action_prior=None, initial_log_std=None):
        super().__init__(); self.action_dim=actions; self.indices=tuple(indices); self.network=_mlp(obs,2*len(self.indices),hidden)
        if not self.indices or len(set(self.indices)) != len(self.indices) or any(i < 0 or i >= actions for i in self.indices):
            raise ValueError('SAC action indices must be unique and within the action vector.')
        from .action_prior import validate_action_prior
        self.action_prior=validate_action_prior(action_prior,actions)
        if initial_log_std is not None:
            initial=torch.as_tensor(initial_log_std,dtype=torch.float32)
            if initial.ndim==0:initial=initial.expand(len(self.indices))
            if initial.shape!=(len(self.indices),) or not bool(torch.isfinite(initial).all() & (initial>-5).all() & (initial<2).all()):
                raise ValueError('Initial SAC log standard deviations must match active axes and lie inside (-5, 2).')
            with torch.no_grad():
                self.network[-1].weight.zero_();self.network[-1].bias.zero_()
                self.network[-1].bias[len(self.indices):].copy_(torch.atanh((initial+5)/3.5-1))
        elif self.action_prior is not None:
            with torch.no_grad():
                self.network[-1].weight[:len(self.indices)].zero_()
                self.network[-1].bias[:len(self.indices)].zero_()
    def _distribution(self, observation):
        from .action_prior import prior_latent
        mean, raw=self.network(observation).chunk(2,-1)
        mean=mean+prior_latent(observation,self.action_prior,self.action_dim)[...,self.indices]
        return mean, -5.0+3.5*(torch.tanh(raw)+1.0)
    def sample(self, observation):
        mean, log_std=self._distribution(observation); normal=torch.distributions.Normal(mean,log_std.exp()); raw=normal.rsample(); bounded=torch.tanh(raw)
        correction = 2 * (math.log(2) - raw - torch.nn.functional.softplus(-2 * raw))
        logp=(normal.log_prob(raw)-correction).sum(-1,keepdim=True)
        action=torch.zeros((observation.shape[0],self.action_dim),device=observation.device); action[:,self.indices]=bounded; return action,logp
    def deterministic(self, observation):
        mean,_=self._distribution(observation); action=torch.zeros((observation.shape[0],self.action_dim),device=observation.device); action[:,self.indices]=torch.tanh(mean); return action

class SACCritic(nn.Module):
    def __init__(self, obs:int, actions:int, hidden:int): super().__init__(); self.network=_mlp(obs+actions,1,hidden)
    def forward(self, observation, action): return self.network(torch.cat((observation,action),-1))

class ReplayBuffer:
    def __init__(self, capacity:int, observation_dim:int, action_dim:int, device:torch.device=torch.device('cpu'), *, critic_context_dim:int=0):
        if capacity < 1: raise ValueError('Replay capacity must be positive.')
        self.capacity=int(capacity); self.size=0; self.cursor=0; self.device=device
        self.critic_context_dim = critic_context_dim
        self.tensors=[torch.empty((capacity,n),dtype=torch.float32,device=device) for n in (observation_dim,action_dim,1,observation_dim,1)]
        if critic_context_dim:
            self.tensors.extend(torch.empty((capacity,critic_context_dim),dtype=torch.float32,device=device) for _ in range(2))
    def add(self, observation, action, reward, next_observation, done, mask, critic_context=None, next_critic_context=None):
        sources = [observation, action, reward, next_observation, done]
        if self.critic_context_dim:
            if critic_context is None or next_critic_context is None:
                raise ValueError('This replay buffer requires both critic contexts.')
            sources.extend([critic_context, next_critic_context])
        valid=mask.reshape(-1).bool(); values=[x[valid].detach().to(self.device) for x in sources]; count=values[0].shape[0]
        if count>self.capacity: values=[x[-self.capacity:] for x in values]; count=self.capacity
        indices=(torch.arange(count,device=self.device)+self.cursor)%self.capacity
        for destination,source in zip(self.tensors,values): destination[indices]=source
        self.cursor=(self.cursor+count)%self.capacity; self.size=min(self.capacity,self.size+count)
    def sample(self,count:int,device:torch.device,generator:torch.Generator):
        if not self.size: raise ValueError('Cannot sample an empty replay buffer.')
        indices=torch.randint(self.size,(count,),generator=generator,device=self.device); return tuple(x[indices].to(device) for x in self.tensors)

@dataclass(frozen=True)
class SACMetrics: actor_loss:float; critic_loss:float; temperature:float

class SimpleSACAgent:
    def __init__(self,observation_dim:int,action_dim:int=3,*,device:torch.device,hidden_dim:int=256,actor_learning_rate:float=3e-4,critic_learning_rate:float=3e-4,temperature_learning_rate:float=3e-4,gamma:float=.99,target_smoothing_tau:float=.005,initial_temperature:float=.2,stochastic_action_indices=(0,2),critic_context_dim:int=0,maximum_gradient_norm:float=10.,action_prior=None,initial_log_std=None,target_entropy=None):
        if initial_temperature <= 0 or not 0 <= gamma <= 1 or not 0 < target_smoothing_tau <= 1:
            raise ValueError('Invalid SAC temperature, discount, or target smoothing.')
        self.critic_context_dim=critic_context_dim; self.maximum_gradient_norm=maximum_gradient_norm
        self.agent_config = dict(hidden_dim=hidden_dim,actor_learning_rate=actor_learning_rate,critic_learning_rate=critic_learning_rate,temperature_learning_rate=temperature_learning_rate,gamma=gamma,target_smoothing_tau=target_smoothing_tau,initial_temperature=initial_temperature,stochastic_action_indices=list(stochastic_action_indices),critic_context_dim=critic_context_dim,maximum_gradient_norm=maximum_gradient_norm)
        self.device=device; self.gamma=float(gamma); self.tau=float(target_smoothing_tau); self.actor=SACActor(observation_dim,action_dim,hidden_dim,stochastic_action_indices,action_prior=action_prior,initial_log_std=initial_log_std).to(device)
        if self.actor.action_prior is not None:self.agent_config['action_prior']=self.actor.action_prior
        if initial_log_std is not None:self.agent_config['initial_log_std']=initial_log_std
        if target_entropy is not None:
            if not math.isfinite(target_entropy):raise ValueError('SAC target entropy must be finite.')
            self.agent_config['target_entropy']=float(target_entropy)
        observation_dim += critic_context_dim
        self.critic1=SACCritic(observation_dim,action_dim,hidden_dim).to(device); self.critic2=SACCritic(observation_dim,action_dim,hidden_dim).to(device); self.target1=SACCritic(observation_dim,action_dim,hidden_dim).to(device); self.target2=SACCritic(observation_dim,action_dim,hidden_dim).to(device)
        self.target1.load_state_dict(self.critic1.state_dict()); self.target2.load_state_dict(self.critic2.state_dict())
        self.target1.requires_grad_(False); self.target2.requires_grad_(False)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=actor_learning_rate); self.critic_optimizer=torch.optim.Adam(list(self.critic1.parameters())+list(self.critic2.parameters()),lr=critic_learning_rate)
        self.log_temperature=torch.tensor(math.log(initial_temperature),device=device,requires_grad=True); self.temperature_optimizer=torch.optim.Adam([self.log_temperature],lr=temperature_learning_rate); self.target_entropy=-float(len(stochastic_action_indices)) if target_entropy is None else float(target_entropy); self.gradient_updates=0
    @torch.no_grad()
    def act(self,observation): return self.actor.sample(observation)[0]
    @torch.no_grad()
    def deterministic_action(self,observation): return self.actor.deterministic(observation)
    def update(self,batch, *, update_actor=True):
        from .training_control import check_training_stop
        check_training_stop()
        observation,action,reward,next_observation,done=batch[:5]; temperature=self.log_temperature.exp().detach()
        if self.critic_context_dim:
            context,next_context=batch[5:]
            critic_observation=torch.cat((observation,context),-1)
            next_critic_observation=torch.cat((next_observation,next_context),-1)
        else:
            critic_observation,next_critic_observation=observation,next_observation
        with torch.no_grad():
            next_action,next_logp=self.actor.sample(next_observation); target=reward+self.gamma*(1-done)*(torch.minimum(self.target1(next_critic_observation,next_action),self.target2(next_critic_observation,next_action))-temperature*next_logp)
        critic_loss=(self.critic1(critic_observation,action)-target).square().mean()+(self.critic2(critic_observation,action)-target).square().mean()
        if not bool(torch.isfinite(critic_loss)): raise FloatingPointError('SAC critic loss became non-finite.')
        self.critic_optimizer.zero_grad(set_to_none=True); critic_loss.backward()
        nn.utils.clip_grad_norm_(list(self.critic1.parameters())+list(self.critic2.parameters()),self.maximum_gradient_norm,error_if_nonfinite=True)
        self.critic_optimizer.step()
        if not update_actor:
            with torch.no_grad():
                for target_parameter,parameter in zip(self.target1.parameters(),self.critic1.parameters()): target_parameter.lerp_(parameter,self.tau)
                for target_parameter,parameter in zip(self.target2.parameters(),self.critic2.parameters()): target_parameter.lerp_(parameter,self.tau)
            self.gradient_updates+=1
            return SACMetrics(0.,float(critic_loss.detach()),float(temperature))
        self.critic1.requires_grad_(False); self.critic2.requires_grad_(False)
        try:
            sampled,logp=self.actor.sample(observation); actor_loss=(temperature*logp-torch.minimum(self.critic1(critic_observation,sampled),self.critic2(critic_observation,sampled))).mean()
            self.actor_optimizer.zero_grad(set_to_none=True); actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(),self.maximum_gradient_norm,error_if_nonfinite=True)
            self.actor_optimizer.step()
        finally:
            self.critic1.requires_grad_(True); self.critic2.requires_grad_(True)
        temperature_loss=-(self.log_temperature*(logp.detach()+self.target_entropy)).mean(); self.temperature_optimizer.zero_grad(set_to_none=True); temperature_loss.backward(); self.temperature_optimizer.step()
        with torch.no_grad():
            self.log_temperature.clamp_(-16.,3.)
            for target_parameter,parameter in zip(self.target1.parameters(),self.critic1.parameters()): target_parameter.lerp_(parameter,self.tau)
            for target_parameter,parameter in zip(self.target2.parameters(),self.critic2.parameters()): target_parameter.lerp_(parameter,self.tau)
        self.gradient_updates+=1; return SACMetrics(float(actor_loss.detach()),float(critic_loss.detach()),float(self.log_temperature.exp().detach()))
