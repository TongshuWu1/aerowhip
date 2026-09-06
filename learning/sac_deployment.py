"""Off-policy SAC experience from complete initial-state-only strike plans."""
import torch

from .deployment_rollout import collect_deployment_rollout
from .point_force_env import POINT_FORCE_OBSERVATION_DIM
from .simple_ppo import PPORollout


def critic_context_dimension(task):
    # Initial estimate plus zero-padded actions already chosen. The actor still
    # sees only its usual 79-dimensional nominal-model observation.
    return POINT_FORCE_OBSERVATION_DIM + 3 * round(task['episode_duration_s'] / task['control_dt_s'])


class _SamplingPlanner:
    def __init__(self, agent, warmup, generator):
        self.agent, self.warmup, self.generator = agent, warmup, generator

    def act(self, observation):
        if self.warmup and getattr(self.agent.actor,'action_prior',None) is None:
            action = observation.new_zeros((len(observation), 3))
            indices = self.agent.actor.indices
            action[:, indices] = 2 * torch.rand((len(observation), len(indices)),
                                                device=observation.device, generator=self.generator) - 1
        else:
            action = self.agent.act(observation)
        unused = observation.new_zeros((len(observation), 1))
        return action, unused, unused


@torch.no_grad()
def add_plan_to_replay(rollout, replay):
    """Store planning transitions with one terminal reward for execution/recovery.

    The critics need the plan prefix to value the final physical outcome.
    Their history contains no actual feedback or future commands.
    """
    steps, batch, _ = rollout.observations.shape
    initial = rollout.observations[0]
    prefix = initial.new_zeros((batch, steps * 3))
    count = 0
    for index in range(steps):
        mask = rollout.masks[index]
        valid_count = int(mask.sum())
        if not valid_count:
            break
        context = torch.cat((initial, prefix), -1)
        prefix = prefix.clone()
        prefix[:, index * 3:(index + 1) * 3] = rollout.actions[index]
        next_context = torch.cat((initial, prefix), -1)
        next_observation = (rollout.observations[index + 1] if index + 1 < steps
                            else torch.zeros_like(initial))
        replay.add(rollout.observations[index], rollout.actions[index], rollout.rewards[index],
                   next_observation, rollout.dones[index], mask, context, next_context)
        count += valid_count
    return count


@torch.no_grad()
def collect_sac(environment, agent, replay, *, warmup=False, generator=None, progress=None):
    rollout = PPORollout.allocate(environment.control_step_count, environment.batch_size,
                                  POINT_FORCE_OBSERVATION_DIM, 3, device=environment.device)
    score = collect_deployment_rollout(environment, _SamplingPlanner(agent, warmup, generator), rollout,
                                       progress=progress)
    if progress is not None:
        progress('Storing SAC experience',0,0)
    return score, add_plan_to_replay(rollout, replay)
