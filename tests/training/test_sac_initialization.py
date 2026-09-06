import math
import torch
from learning.simple_sac import SACActor,SimpleSACAgent
from learning.simple_ppo import BoundedGaussianPolicy
from learning.sac_deployment import _SamplingPlanner


PRIOR=dict(enabled=True,total_control_steps=4,phase_control_steps=[1,2],
           actions=[[.7,0.,.2],[-.5,.1,-.3]],source='test')


def test_sac_prior_matches_ppo_and_survives_checkpoint_reconstruction():
    from run_sac import checkpoint
    from simulator.rollout import _build_sac_agent
    from pathlib import Path
    sac=SimpleSACAgent(79,device=torch.device('cpu'),hidden_dim=16,
        stochastic_action_indices=[0,1,2],action_prior=PRIOR,
        initial_log_std=[math.log(.05),math.log(.02),math.log(.05)],target_entropy=-7.)
    ppo=BoundedGaussianPolicy(79,3,16,action_prior=PRIOR)
    obs=torch.zeros(5,79);obs[:,-1]=torch.tensor([1.,.75,.5,.25,0.])
    torch.testing.assert_close(sac.deterministic_action(obs),ppo.deterministic(obs))
    fresh=_build_sac_agent(Path('.'),torch.device('cpu'),checkpoint=checkpoint(sac,0))
    fresh.actor.load_state_dict(sac.actor.state_dict())
    torch.testing.assert_close(fresh.deterministic_action(obs),sac.deterministic_action(obs))
    assert fresh.target_entropy==-7.


def test_critic_warmup_learns_without_moving_actor_or_temperature():
    torch.set_num_threads(1);torch.manual_seed(5)
    agent=SimpleSACAgent(4,device=torch.device('cpu'),hidden_dim=16,stochastic_action_indices=[0,1,2])
    before={k:v.clone() for k,v in agent.actor.state_dict().items()}
    critic={k:v.clone() for k,v in agent.critic1.state_dict().items()}
    alpha=agent.log_temperature.clone()
    obs=torch.randn(16,4);action=agent.act(obs)
    agent.update((obs,action,torch.ones(16,1)*200,obs,torch.ones(16,1)),update_actor=False)
    for key,value in agent.actor.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)
    torch.testing.assert_close(alpha,agent.log_temperature,rtol=0,atol=0)
    assert any(not torch.equal(v,critic[k]) for k,v in agent.critic1.state_dict().items())
    assert agent.gradient_updates==1


def test_warmup_samples_near_configured_prior():
    torch.manual_seed(6)
    agent=SimpleSACAgent(4,device=torch.device('cpu'),hidden_dim=16,
        stochastic_action_indices=[0,1,2],action_prior=PRIOR,initial_log_std=math.log(.02))
    obs=torch.zeros(1000,4);obs[:,-1]=1
    action,_,_=_SamplingPlanner(agent,True,torch.Generator().manual_seed(2)).act(obs)
    torch.testing.assert_close(action.mean(0),torch.tensor(PRIOR['actions'][0]),rtol=0,atol=.005)
