from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from learning.pva_env import PVAEnvironment, defaults
from learning.pva_ppo_rollout import decision_steps, decision_rollout, collect
from learning.simple_ppo import PPORollout, generalized_advantage_estimate


def settings(repeat=3):
    cfg = defaults('ppo')
    cfg['policy_timing'] = dict(schema='fixed_jerk_hold_v1', control_steps=repeat)
    cfg['training'].update(gamma=1., gae_lambda=1., reward_scale=1.)
    return cfg


@pytest.mark.parametrize('repeat', [0, -1, True, 1.5])
def test_bad_timing_is_rejected(repeat):
    with pytest.raises(ValueError):
        decision_steps(settings(repeat))


def test_legacy_timing_and_unsupported_discount():
    assert decision_steps(defaults('ppo')) == 1
    cfg = settings(); cfg['training']['gae_lambda'] = .95
    with pytest.raises(ValueError, match='lambda'):
        decision_steps(cfg)
    cfg = settings(); cfg['method'] = 'mppi'
    with pytest.raises(ValueError, match='timing'):
        decision_steps(cfg)


def test_partial_blocks_and_early_terminal_reward_are_counted_once():
    data = PPORollout.allocate(7, 3, 2, 3, device=torch.device('cpu'))
    for name in data.__dataclass_fields__:
        getattr(data, name).zero_()
    # Termination inside the first block, at a boundary, and in final short block.
    for row, terminal in enumerate((1, 2, 6)):
        data.masks[:terminal+1,row] = 1
        data.dones[terminal:,row] = 1
        data.rewards[terminal,row] = 10 + row
    small = decision_rollout(data, 7, 3)
    assert small.rewards.shape == (3, 3, 1)
    torch.testing.assert_close(small.rewards.sum(0), data.rewards.sum(0))
    torch.testing.assert_close(small.masks[:,:,0], torch.tensor([[1.,1.,1.],[0.,0.,1.],[0.,0.,1.]]))
    _, returns = generalized_advantage_estimate(small.rewards, small.dones, small.masks, small.values, gamma=1., gae_lambda=1.)
    torch.testing.assert_close(returns[0,:,0], torch.tensor([10.,11.,12.]))
    assert ((small.dones * small.masks).sum(0) == 1).all()


class TickEnvironment:
    """Different row termination times exercise control ticks within a hold."""
    rollout = PVAEnvironment.rollout

    def __init__(self, repeat):
        self.settings = settings(repeat)
        self.steps = 7; self.batch_size = 3; self.observation_dim = 2
        self.reset()

    def reset(self, **kwargs):
        self.index = 0; self.active = torch.ones(3, dtype=torch.bool)
        self.actions = []
        return self.observation()

    def observation(self):
        return torch.full((3,2), float(self.index))

    def step(self, action, **kwargs):
        mask = self.active.clone()
        self.actions.append(action.clone()); self.index += 1
        self.active &= self.index < torch.tensor([2,3,7])
        return self.observation(), mask.float()[:,None], (~self.active).float()[:,None], mask.float()[:,None]

    def result(self):
        return dict(actions=torch.stack(self.actions))


def test_training_and_inference_use_same_decisions_and_likelihoods():
    env = TickEnvironment(3); calls = []
    def action(obs):
        calls.append(float(obs[0,0]))
        return obs[:,:1].expand(-1,3)/10
    agent = SimpleNamespace(act=lambda obs: (action(obs), obs[:,:1]+20, obs[:,:1]+30))
    buffer = PPORollout.allocate(7,3,2,3,device=torch.device('cpu'))
    compact, result = collect(env,agent,buffer,generator=None)
    assert calls == [0,3,6]
    torch.testing.assert_close(compact.observations[:,0,0], torch.tensor([0.,3.,6.]))
    torch.testing.assert_close(compact.log_probabilities[:,0,0], torch.tensor([20.,23.,26.]))
    torch.testing.assert_close(compact.rewards.sum(0)[:,0], torch.tensor([2.,3.,7.]))
    calls.clear();env.reset()
    replay = env.rollout(policy=action)
    assert calls == [0,3,6]
    torch.testing.assert_close(replay['actions'], result['actions'], atol=0, rtol=0)
    # Explicit 30 Hz plans must never be held again, even with PPO settings.
    env.reset(); explicit = torch.randn(3,7,3)
    replay = env.rollout(actions=explicit)
    torch.testing.assert_close(replay['actions'], explicit.transpose(0,1))


def test_legacy_collect_has_one_sample_per_tick():
    env = TickEnvironment(1)
    agent = SimpleNamespace(act=lambda obs: (obs[:,:1].expand(-1,3), obs[:,:1], obs[:,:1]))
    buffer = PPORollout.allocate(7,3,2,3,device=torch.device('cpu'))
    result, _ = collect(env,agent,buffer,generator=None)
    assert len(result.actions) == 7
    torch.testing.assert_close(result.observations, buffer.observations, atol=0, rtol=0)


def test_checkpoint_timing_mismatch_is_rejected(tmp_path):
    from planning.pva_job import make_agent, save_checkpoint, load_policy
    cfg = settings()
    env = SimpleNamespace(settings=cfg, observation_dim=4, device=torch.device('cpu'))
    agent = make_agent(env,cfg); path = tmp_path/'policy.pt'
    save_checkpoint(path,agent,env,42)
    loaded = load_policy(path,env,cfg)
    obs = torch.randn(2,4)
    torch.testing.assert_close(agent.deterministic_action(obs), loaded.deterministic_action(obs), atol=0, rtol=0)
    old = deepcopy(cfg);old.pop('policy_timing')
    with pytest.raises(ValueError, match='timing'):
        load_policy(path,env,old)
