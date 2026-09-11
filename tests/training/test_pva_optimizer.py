import torch
from planning.pva_job import whiten,validate_settings
from learning.pva_env import defaults
import pytest


def test_correlated_mppi_change_of_measure_matches_gaussian_density():
    torch.manual_seed(2);rho=.6;n=4
    covariance=torch.tensor([[rho**abs(i-j) for j in range(n)] for i in range(n)],dtype=torch.float64)
    mean=torch.tensor([.1,-.2,.4,.1],dtype=torch.float64)
    distribution=torch.distributions.MultivariateNormal(mean,covariance)
    latent=distribution.sample((5,));noise=latent-mean
    calculated=-.5*(whiten(latent[:,:,None],rho).square()-whiten(noise[:,:,None],rho).square()).sum((1,2))
    prior=torch.distributions.MultivariateNormal(torch.zeros_like(mean),covariance)
    torch.testing.assert_close(calculated,prior.log_prob(latent)-distribution.log_prob(latent),atol=1e-12,rtol=0)


def test_settings_reject_force_semantics_and_invalid_noise():
    cfg=defaults();validate_settings(cfg)
    cfg['command_contract']='force_ppo_checkpoint_v1'
    with pytest.raises(ValueError):validate_settings(cfg)
    cfg=defaults();cfg['mppi']['noise_correlation']=1.
    with pytest.raises(ValueError):validate_settings(cfg)


@pytest.mark.parametrize('key,value',[('iterations',-1),('iterations',2.5),('minimum_iterations',101),('patience',0),('samples',2.5)])
def test_settings_reject_invalid_mppi_counts(key,value):
    cfg=defaults('mppi');cfg['mppi']['iterations']=100;cfg['mppi'][key]=value
    with pytest.raises(ValueError):validate_settings(cfg)


def test_mppi_saved_best_metrics_and_actions_stay_together(tmp_path,monkeypatch):
    import numpy as np
    import planning.pva_job as jobs
    from simulator.workflow import read_json
    class Environment:
        steps=3;device=torch.device('cpu')
        def __init__(self,*args,**kwargs):self.calls=0;self.samples=[]
        def reset(self):pass
        def rollout(self,actions):
            self.samples.append(actions.clone());self.calls+=1
            return dict(reward=torch.tensor([10.,2.,-1.] if self.calls==1 else [3.,4.,-1.]),
                success=torch.tensor([True,False,False] if self.calls==1 else [False,False,False]),
                failed=torch.tensor([False,True,False]),minimum_tip_distance_m=torch.tensor([.01,.8,1.] if self.calls==1 else [.7,.6,1.]))
    env=Environment();monkeypatch.setattr(jobs,'PVAEnvironment',lambda *args,**kwargs:env)
    cfg=defaults('mppi');cfg['mppi'].update(mode='open_loop',samples=2,iterations=2,minimum_iterations=2)
    result=jobs.mppi(tmp_path,{},cfg)
    rows=read_json(tmp_path/'history.json')
    assert result==read_json(tmp_path/'result.json')
    assert result['best_success'] and not result['best_failed'] and result['best_iteration']==1
    assert rows[0]['best_minimum_tip_distance_m']==rows[1]['best_minimum_tip_distance_m']
    with np.load(tmp_path/'plan.npz') as plan:np.testing.assert_array_equal(plan['normalized_jerk'],env.samples[0][0].numpy())
    assert not (tmp_path/'plan.tmp.npz').exists()


def test_mppi_stop_preserves_saved_plan(tmp_path):
    from planning.pva_job import progress
    (tmp_path/'plan.npz').write_bytes(b'saved plan')
    (tmp_path/'STOP').touch()
    with pytest.raises(InterruptedError):progress(tmp_path,iteration=2)
    assert (tmp_path/'plan.npz').read_bytes()==b'saved plan'


def test_mean_candidate_can_win_without_biasing_importance_weights(tmp_path,monkeypatch):
    import numpy as np
    import planning.pva_job as jobs
    class Environment:
        steps=3;device=torch.device('cpu')
        def reset(self):pass
        def rollout(self,actions):
            self.actions=actions
            return dict(reward=torch.tensor([0.,0.,1000.]),success=torch.tensor([False,False,True]),
                failed=torch.zeros(3,dtype=torch.bool),minimum_tip_distance_m=torch.tensor([1.,1.,.01]))
    env=Environment();monkeypatch.setattr(jobs,'PVAEnvironment',lambda *args,**kwargs:env)
    cfg=defaults('mppi');cfg['mppi'].update(mode='open_loop',samples=2,iterations=1,minimum_iterations=1)
    result=jobs.mppi(tmp_path,{},cfg)
    with np.load(tmp_path/'plan.npz') as plan:
        np.testing.assert_array_equal(plan['normalized_jerk'],np.zeros((3,3)))
        np.testing.assert_allclose(plan['proposal_mean'],torch.atanh(env.actions[:-1]).mean(0).numpy(),atol=1e-15)
    assert result['best_success']


def test_duplicate_launch_does_not_rewrite_completed_status(tmp_path):
    from planning.pva_job import run
    from experimental_data.io import atomic_json
    from simulator.workflow import read_json
    status=dict(status='completed',best_success=True,iterations=27)
    atomic_json(tmp_path/'status.json',status)
    with pytest.raises(ValueError,match='already started'):run(tmp_path)
    assert read_json(tmp_path/'status.json')==status and not (tmp_path/'worker.lock').exists()


@pytest.mark.parametrize('manual_stop',[False,True])
def test_uncapped_mppi_passes_old_ceiling_and_stops_on_plateau_or_request(tmp_path,monkeypatch,manual_stop):
    import planning.pva_job as jobs
    class Environment:
        steps=1;device=torch.device('cpu');calls=0
        def reset(self):pass
        def rollout(self,actions):
            self.calls+=1
            if manual_stop and self.calls==103:(tmp_path/'STOP').touch()
            return dict(reward=torch.full((3,),float(min(self.calls,105))),success=torch.zeros(3,dtype=torch.bool),
                failed=torch.zeros(3,dtype=torch.bool),minimum_tip_distance_m=torch.ones(3))
    env=Environment();monkeypatch.setattr(jobs,'PVAEnvironment',lambda *args,**kwargs:env)
    cfg=defaults('mppi');cfg['mppi'].update(mode='open_loop',samples=2,iterations=0,minimum_iterations=20,patience=3)
    validate_settings(cfg)
    if manual_stop:
        with pytest.raises(InterruptedError):jobs.mppi(tmp_path,{},cfg)
        assert env.calls==103 and (tmp_path/'plan.npz').exists()
    else:
        result=jobs.mppi(tmp_path,{},cfg)
        assert result['iterations']==108 and result['stop_reason']=='reward_plateau'
