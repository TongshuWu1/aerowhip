"""Compile a feedback policy into one finite, open-loop strike."""

from dataclasses import dataclass
from typing import Callable

import torch

from learning.point_force_env import PointForceWhipEnvironment
from .cable import DderState


@dataclass(frozen=True)
class StrikePlan:
    forces_world_n: torch.Tensor
    dt_s: float
    initial_state: DderState
    initial_observation: torch.Tensor

    @property
    def duration_s(self) -> float:
        return len(self.forces_world_n) * self.dt_s


@torch.no_grad()
def compile_strike_plan(model_config, task_config, ppo_config, initial_state,
                        policy, *, physics=None, cancel_requested: Callable[[], bool] | None = None) -> StrikePlan:
    """Predict from the initial state and execute one attempt, hit or miss.

    All policy queries use this private model rollout. The resulting force
    sequence has no state inputs, hit inputs, looping, or repeating tail.
    """
    env = PointForceWhipEnvironment(model_config, task_config, ppo_config,
                                    batch_size=1, device=torch.device("cpu"))
    observation = env.reset(initial_state)
    initial_observation = observation.clone()
    initial_state = env.state
    if physics is not None:
        def fast_step(state, force, dt):
            q, v = physics(state.positions_m, state.velocities_m_s, force)
            return env.model._result(state, DderState(q, v), force, dt)
        env.model.step_runtime = fast_step
    forces = []

    def record(_state, force, _reaction):
        if cancel_requested is not None and cancel_requested():
            raise InterruptedError("Strike preparation cancelled.")
        forces.append(force[0].detach().clone())

    for _ in range(env.control_step_count):
        if cancel_requested is not None and cancel_requested():
            raise InterruptedError("Strike preparation cancelled.")
        action = policy(observation)
        if not bool(torch.isfinite(action).all()):
            raise ValueError("Policy produced an invalid force sequence; staying in hover.")
        result = env.step(action, stop_when_all_done=True, physics_trace_callback=record)
        observation = result.next_observation
        if not bool(env.active.any()):
            break
    if bool(env.failed[0]):
        raise ValueError("Predicted motion exceeds simulation limits; staying in hover.")
    from .strike_sequence import freeze_followthrough
    frozen,cutoffs=freeze_followthrough(torch.stack(forces)[:,None],torch.tensor([len(forces)]),
        dt_s=env.physics_dt_s,maximum_steps=round(task_config['episode_duration_s']/env.physics_dt_s),
        duration_s=ppo_config.get('deployment',{}).get('strike_followthrough_s',0.))
    return StrikePlan(frozen[:int(cutoffs[0]),0], env.physics_dt_s, initial_state, initial_observation)
