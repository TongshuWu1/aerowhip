"""Allocate the complete SAC replay and rollout footprint before a large study."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from run_sac import configs,build_agent
from learning.simple_sac import ReplayBuffer
from learning.simple_ppo import PPORollout
from learning.point_force_env import PointForceWhipEnvironment
from simulator.workflow import atomic_json

torch.set_num_threads(1)
model,task,shared,sac=configs();device=torch.device('cuda');batch=32768
env=PointForceWhipEnvironment(model,task,shared,batch_size=batch,device=device)
agent=build_agent(sac,device,task)
rollout=PPORollout.allocate(env.control_step_count,batch,79,3,device=device)
capacity=4*env.control_step_count*batch
replay=ReplayBuffer(capacity,79,3,device,critic_context_dim=agent.critic_context_dim)
for tensor in replay.tensors:tensor.zero_()
obs=env.observation() if hasattr(env,'observation') else torch.zeros(1024,79,device=device)
obs=obs[:1024].float()
context=torch.zeros(len(obs),agent.critic_context_dim,device=device)
agent.update((obs,agent.act(obs),torch.ones(len(obs),1,device=device),obs,torch.ones(len(obs),1,device=device),context,context))
torch.cuda.synchronize()
result=dict(batch=batch,replay_capacity=capacity,critic_context_dim=agent.critic_context_dim,
    peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
    free_gib=torch.cuda.mem_get_info()[0]/1024**3,
    scope='Replay, planning rollout, environment and SAC optimizer allocation; graph physics benchmark recorded separately')
atomic_json(Path('data/comparison_memory_20hz.json'),result);print(result)
