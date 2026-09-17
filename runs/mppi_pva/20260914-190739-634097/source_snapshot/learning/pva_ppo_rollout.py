"""PPO decisions with fixed-duration jerk actions and unchanged 30 Hz physics input.

Each block contributes one likelihood ratio. Intermediate command steps are
environment transitions, not additional policy samples. Gamma/lambda are one
for repeated actions, so summing their rewards preserves the episodic objective.
"""
import torch
from learning.simple_ppo import PPORollout


def decision_steps(settings):
    spec = settings.get('policy_timing')
    if spec is None:
        return 1
    if settings.get('method') != 'ppo' or spec.get('schema') != 'fixed_jerk_hold_v1':
        raise ValueError('Unknown PPO policy timing contract')
    count = spec.get('control_steps')
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('Policy control_steps must be a positive integer')
    if count > round(settings['task']['duration_s'] * 30):
        raise ValueError('Policy hold exceeds the episode')
    if count > 1 and any(settings['training'].get(key, default) != 1.
                         for key, default in (('gamma', 1.), ('gae_lambda', .95))):
        raise ValueError('Repeated PPO actions currently require gamma and GAE lambda 1')
    return count


def decision_rollout(rollout, used, repeat):
    """Collapse complete control-step data, retaining terminal partial blocks."""
    fields = {name: getattr(rollout, name)[:used]
              for name in rollout.__dataclass_fields__}
    if repeat == 1:
        return PPORollout(**fields)
    starts = range(0, used, repeat)
    compact = {name: value[::repeat] for name, value in fields.items()}
    compact['rewards'] = torch.stack([
        fields['rewards'][start:start + repeat].sum(0) for start in starts])
    compact['dones'] = torch.stack([
        fields['dones'][min(start + repeat, used) - 1] for start in starts])
    return PPORollout(**compact)


@torch.no_grad()
def collect(env, agent, buffer, *, generator):
    from learning.ppo_trajectory_reward import enabled, TrajectoryReward
    repeat = decision_steps(env.settings)
    obs = env.reset(randomize=True, generator=generator)
    objective = TrajectoryReward(env) if enabled(env.settings) else None
    scale = env.settings['training'].get('reward_scale', 1.)
    for k in range(env.steps):
        if k % repeat == 0:
            action, lp, value = agent.act(obs)
        following, reward, done, mask = env.step(action)
        if objective is not None:
            objective.observe(env)
        buffer.observations[k] = obs
        buffer.actions[k] = action
        buffer.log_probabilities[k] = lp
        buffer.values[k] = value
        buffer.rewards[k] = reward * scale
        buffer.dones[k] = done
        buffer.masks[k] = mask
        obs = following
        if not bool(env.active.any()):
            break
    used = k + 1
    result = env.result()
    if objective is not None:
        result = objective.finish(env, result)
        buffer.rewards[:used] = objective.training_rewards(
            result, buffer.masks[:used], buffer.dones[:used]) * scale
    return decision_rollout(buffer, used, repeat), result
