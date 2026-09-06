"""30 Hz commands retain exact duration and the searched force waveform."""
from copy import deepcopy
import torch
from run_sac import configs,build_agent
from learning.point_force_env import PointForceWhipEnvironment
from learning.action_prior import prior_latent
from simulator.strike_sequence import freeze_followthrough


def test_30hz_grid_prior_and_followthrough():
    model,task,shared,sac=deepcopy(configs())
    task['control_dt_s']=1/30;task['episode_duration_s']=1.
    model['simulation']['dt_s']=1/150;model['cable']['substeps']=8
    env=PointForceWhipEnvironment(model,task,shared,batch_size=1,device=torch.device('cpu'))
    assert env.control_step_count==30
    assert round(env.control_dt_s/env.physics_dt_s)==5
    assert abs(env.physics_dt_s/model['cable']['substeps']-1/1200)<1e-15
    prior=dict(enabled=True,total_control_steps=30,phase_control_steps=[3]*10,
        actions=[[i*.05,0.,-.2] for i in range(10)])
    sac['bootstrap']=prior
    agent=build_agent(sac,torch.device('cpu'),task)
    assert agent.critic_context_dim==79+90
    obs=torch.zeros(30,79);obs[:,-1]=1-torch.arange(30)/30
    actions=prior_latent(obs,prior,3).tanh()
    torch.testing.assert_close(actions,torch.tensor(prior['actions']).repeat_interleave(3,dim=0))
    forces=torch.ones(150,1,3);cutoff=torch.tensor([117])
    _,extended=freeze_followthrough(forces,cutoff,dt_s=env.physics_dt_s,maximum_steps=150,duration_s=.02)
    assert extended.item()==120
    assert abs((extended-cutoff).item()*env.physics_dt_s-.02)<1e-15
