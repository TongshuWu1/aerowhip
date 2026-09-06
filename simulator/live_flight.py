"""Continuous point-mass flight: position PID, policy strike, and PID recovery."""

from __future__ import annotations

from collections import deque
from functools import lru_cache
import json
from pathlib import Path
import warnings

import numpy as np
import torch

from learning.point_force_env import (
    POINT_FORCE_OBSERVATION_DIM, PointForceWhipEnvironment,
    _segment_enters_sphere, tip_velocity_strike_gate,
)
from .cable import DderState, FREE_ENDPOINTS
from .point_mass import ForceControlledPointCable
from .rollout import _build_policy_agent, _build_sac_agent
from .strike_plan import StrikePlan, compile_strike_plan


@lru_cache(maxsize=2)
def prepare_live_physics(model_json: str):
    """Compile the existing CPU equations, keeping every 10 ms physics step.

    FX expands the energy gradient before tracing; a direct JIT trace of
    autograd would incorrectly capture parts of the initial geometry.
    """
    from torch.fx.experimental.proxy_tensor import make_fx

    model_config = json.loads(model_json)
    model = ForceControlledPointCable.from_mapping(model_config)
    state = model.hanging_state(torch.tensor([0., 0., 1.5], dtype=torch.float64))
    dt = torch.tensor([model_config["simulation"]["dt_s"]], dtype=torch.float64)
    constants = model.dder.runtime_constants(state.positions_m)
    force = model.hover_force_world_n(dtype=torch.float64, device="cpu")[None]

    def transition(q, v, command):
        result = model.dder.step_runtime(
            DderState(q, v), q[:, :0], dt, constants,
            external_force_world_n=model.controller.node_forces(command, validate=False),
            pinned_endpoints=FREE_ENDPOINTS, functional_force_autograd=True,
        )
        return result.positions_m, result.velocities_m_s

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        graph = make_fx(transition)(state.positions_m, state.velocities_m_s, force)
        compiled = torch.jit.trace(
            graph, (state.positions_m, state.velocities_m_s, force), check_trace=False,
        )
    # Materialize JIT optimizations before the wall-clock simulation starts.
    for _ in range(3):
        compiled(state.positions_m, state.velocities_m_s, force)
    return compiled


class HoverPID:
    """Position PID with weight feedforward and bounded integral/force."""

    def __init__(self, model, target, maximum_force_n: float) -> None:
        self.target = target.clone()
        self.mass = model.system_mass_kg
        self.feedforward = model.hover_force_world_n(dtype=target.dtype, device=target.device)[None]
        self.maximum_force_n = maximum_force_n
        self.kp = target.new_tensor([4., 4., 6.])
        self.kd = target.new_tensor([3.5, 3.5, 4.])
        self.ki = target.new_tensor([.15, .15, .4])
        self.integral = torch.zeros_like(target)

    def reset(self) -> None:
        self.integral.zero_()

    def command(self, state: DderState, dt_s: float) -> torch.Tensor:
        error = self.target - state.positions_m[:, 0]
        candidate = (self.integral + error * dt_s).clamp(-.5, .5)
        force = self.feedforward + self.mass * (
            self.kp * error - self.kd * state.velocities_m_s[:, 0] + self.ki * candidate
        )
        # Conditional integration prevents windup during the recovery transient.
        norm = force.norm(dim=-1, keepdim=True)
        unsaturated = (norm <= self.maximum_force_n) & (force[:, 2:3] >= 0)
        self.integral.copy_(torch.where(unsaturated, candidate, self.integral))
        force[:, 2].clamp_(min=0.)
        return force * torch.clamp(
            self.maximum_force_n / force.norm(dim=-1, keepdim=True).clamp_min(1e-12), max=1.)


class LiveFlight:
    """One persistent physical state; only the active controller is switched."""

    HOVER, POLICY, RECOVER = 0, 1, 2

    def __init__(self, model_config, task_config, ppo_config, *, policy=None,
                 checkpoint_path: Path | None = None, physics=None) -> None:
        self.environment = PointForceWhipEnvironment(
            model_config, task_config, ppo_config, batch_size=1, device=torch.device("cpu"),
        )
        self.model = self.environment.model
        self.state = self.environment.state
        self.dt_s = self.environment.physics_dt_s
        self.physics = physics
        self.policy = policy
        self.checkpoint_path = checkpoint_path
        self.hover_position = self.state.positions_m[:, 0].clone()
        self.hanging_offsets = self.state.positions_m - self.hover_position[:, None]
        self.pid = HoverPID(self.model, self.hover_position, self.environment.maximum_force_norm_n)
        self.phase = self.HOVER
        self.time_s = 0.
        self.steps = 0
        self.strike_steps = 0
        self.force_sequence = None
        self.sequence_duration_s = 0.
        self.plans = []
        self.attempt = 0
        self.success = False
        self.non_tip_first = False
        self.tip_contact = False
        self.hit_now = False
        self.settled_s = 0.
        self.message = "PID hover — settling cable"
        self.last_command = self.pid.feedforward.clone()
        self.last_reaction = torch.zeros_like(self.last_command)
        self.events = []
        # Keep the latest ten minutes; flight itself has no duration limit.
        self.history = deque(maxlen=int(round(600 / self.dt_s)) + 1)
        self.history.append(self.snapshot())

    @classmethod
    def from_checkpoint(cls, model_config, task_config, ppo_config, checkpoint_path,
                        *, physics=None):
        path = Path(checkpoint_path).resolve()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        schema = checkpoint.get("schema")
        if schema not in {"force_ppo_checkpoint_v1", "force_sac_checkpoint_v1"}:
            raise ValueError("Live strikes require a point-force PPO or SAC checkpoint.")
        if (int(checkpoint.get("observation_dim", POINT_FORCE_OBSERVATION_DIM)) != POINT_FORCE_OBSERVATION_DIM
                or int(checkpoint.get("action_dim", 3)) != 3):
            raise ValueError("Checkpoint does not match the cable state and 3D force inputs.")
        saved_config = path.parent.parent / "ppo.json"
        if saved_config.is_file():
            ppo_config = json.loads(saved_config.read_text(encoding="utf-8"))
        saved_model = path.parent.parent / "model.json"
        if saved_model.is_file():
            trained_config = json.loads(saved_model.read_text(encoding="utf-8"))
            trained_model = ForceControlledPointCable.from_mapping(trained_config)
            current_model = ForceControlledPointCable.from_mapping(model_config)
            if (float(trained_config["simulation"]["dt_s"]) != float(model_config["simulation"]["dt_s"])
                    or trained_model.point_mass_kg != current_model.point_mass_kg
                    or trained_model.dder.parameters != current_model.dder.parameters):
                raise ValueError("Checkpoint cable/drone parameters differ from the live model.")
        saved_task = path.parent.parent / "task.json"
        if saved_task.is_file():
            trained = json.loads(saved_task.read_text(encoding="utf-8"))
            for key in ("control_dt_s", "episode_duration_s"):
                if float(trained[key]) != float(task_config[key]):
                    raise ValueError(f"Checkpoint and live task disagree on {key}.")
        if schema == "force_sac_checkpoint_v1":
            agent = _build_sac_agent(Path(__file__).resolve().parents[1], torch.device("cpu"), checkpoint=checkpoint)
            agent.actor.load_state_dict(checkpoint["actor"])
            agent.actor.eval()
        else:
            agent = _build_policy_agent(ppo_config, torch.device("cpu"))
            agent.policy.set_action_prior(checkpoint.get("action_prior"))
            agent.policy.load_state_dict(checkpoint["policy"])
            agent.policy.eval()
        flight = cls(model_config, task_config, ppo_config, policy=agent.deterministic_action,
                     checkpoint_path=path, physics=physics)
        flight.checkpoint_episodes = int(checkpoint.get("episodes", 0))
        flight.algorithm = "SAC" if schema == "force_sac_checkpoint_v1" else "PPO"
        return flight

    @property
    def ready(self) -> bool:
        return self.phase == self.HOVER and self.settled_s >= .5 and self.policy is not None

    def plan_strike(self, initial_state: DderState, cancel_requested=None) -> StrikePlan:
        env = self.environment
        return compile_strike_plan(env.model_config, env.task_config, env.ppo_config,
                                   initial_state, self.policy, physics=self.physics,
                                   cancel_requested=cancel_requested)

    def start_strike(self, plan: StrikePlan) -> None:
        if not self.ready:
            raise ValueError("Wait for PID hover and the cable to settle before striking.")
        if (plan.dt_s != self.dt_s or plan.forces_world_n.ndim != 2
                or plan.forces_world_n.shape[1] != 3 or len(plan.forces_world_n) == 0
                or not bool(torch.isfinite(plan.forces_world_n).all())):
            raise ValueError("Invalid one-shot force sequence.")
        # This check happens before force control, never during the strike.
        deployment = self.environment.ppo_config.get('deployment', {})
        if (float((self.state.positions_m - plan.initial_state.positions_m).norm(dim=-1).max()) > float(deployment.get('launch_position_drift_m', .04))
                or float((self.state.velocities_m_s - plan.initial_state.velocities_m_s).norm(dim=-1).max()) > float(deployment.get('launch_velocity_drift_m_s', .15))):
            raise ValueError("Initial state changed while preparing the strike; settle and try again.")
        self.initial_observation = plan.initial_observation.clone()
        self.force_sequence = plan.forces_world_n.detach().clone()
        self.sequence_duration_s = plan.duration_s
        self.phase = self.POLICY
        self.strike_steps = 0
        self.attempt += 1
        self.success = self.non_tip_first = self.tip_contact = self.hit_now = False
        self.settled_s = 0.
        self.message = f"Executing one strike — {plan.duration_s:.2f} s"
        self.events.append({"time_s": self.time_s, "event": "strike", "attempt": self.attempt,
                            "planned_duration_s": plan.duration_s})
        self.plans.append({"attempt": self.attempt, "dt_s": plan.dt_s, "duration_s": plan.duration_s,
                           "forces_world_n": plan.forces_world_n.tolist(),
                           "initial_positions_m": plan.initial_state.positions_m[0].tolist(),
                           "initial_velocities_m_s": plan.initial_state.velocities_m_s[0].tolist()})

    def return_to_hover(self, reason="Manual return") -> None:
        self.phase = self.RECOVER
        self.force_sequence = None
        self.pid.reset()
        self.settled_s = 0.
        self.message = reason + " — PID returning to hover"
        self.events.append({"time_s": self.time_s, "event": reason, "attempt": self.attempt})

    def _check_contact(self, previous: DderState) -> None:
        env = self.environment
        crossing = _segment_enters_sphere(
            previous.positions_m[:, env.marker_indices], self.state.positions_m[:, env.marker_indices],
            env.target, env.target_radius_m,
        )
        tip, other = bool(crossing[0, -1]), bool(crossing[0, :-1].any())
        contacted_before = self.tip_contact or self.non_tip_first
        self.tip_contact |= tip
        self.non_tip_first |= other and (not tip or env.tip_must_enter_first)
        self.hit_now = not self.success and not (env.first_contact_only and contacted_before) and tip and not other and not self.non_tip_first and bool(tip_velocity_strike_gate(
            self.state.velocities_m_s[:, -1], env.desired_direction,
            minimum_directed_speed_m_s=env.minimum_directed_speed_m_s,
            maximum_direction_error_deg=env.maximum_tip_velocity_direction_error_deg,
        )[0])
        if self.hit_now:
            self.success = True

    @torch.no_grad()
    def step(self) -> dict:
        env = self.environment
        was_policy = self.phase == self.POLICY
        self.hit_now = False
        if was_policy:
            # Consume each precomputed command exactly once. No observation,
            # policy inference, or hit result feeds back into this sequence.
            self.last_command = self.force_sequence[self.strike_steps:self.strike_steps + 1]
        else:
            self.last_command = self.pid.command(self.state, self.dt_s)
        previous = self.state
        if self.physics is None:
            transition = self.model.step_runtime(previous, self.last_command, self.dt_s)
            candidate = transition.state
        else:
            q, v = self.physics(previous.positions_m, previous.velocities_m_s, self.last_command)
            candidate = DderState(q, v)
        if (not bool(torch.isfinite(candidate.positions_m).all() & torch.isfinite(candidate.velocities_m_s).all())
                or float(candidate.positions_m.abs().max()) > env.numerical_position_limit_m
                or float(candidate.velocities_m_s.norm(dim=-1).max()) > env.numerical_speed_limit_m_s):
            raise RuntimeError("Live physics exceeded its numerical limits; simulation stopped.")
        self.last_reaction = (
            self.model.point_mass_kg * (candidate.velocities_m_s[:, 0] - previous.velocities_m_s[:, 0]) / self.dt_s
            - self.last_command - self.model.point_mass_kg * self.model.gravity_world_m_s2[None]
        )
        self.state = candidate
        self.steps += 1
        self.time_s = self.steps * self.dt_s
        if was_policy:
            self.strike_steps += 1
            self._check_contact(previous)  # Simulation diagnostics only; never a stop signal.
            if self.strike_steps == len(self.force_sequence):
                self.return_to_hover("Sequence complete")
                self.message = f"Single strike finished at {self.sequence_duration_s:.2f} s — PID recovery"
        else:
            root_error = (self.state.positions_m[:, 0] - self.hover_position).norm()
            cable_error = (self.state.positions_m - self.state.positions_m[:, :1] - self.hanging_offsets).norm(dim=-1).max()
            settled = float(root_error) < .03 and float(cable_error) < .04 and float(self.state.velocities_m_s.norm(dim=-1).max()) < .15
            self.settled_s = self.settled_s + self.dt_s if settled else 0.
            if self.settled_s >= .5:
                self.phase = self.HOVER
                self.message = "PID hover — ready to strike"
        frame = self.snapshot()
        self.history.append(frame)
        return frame

    def snapshot(self) -> dict:
        array = lambda t: t[0].detach().cpu().numpy().copy()
        return {"time_s": self.time_s, "positions": array(self.state.positions_m),
                "velocities": array(self.state.velocities_m_s), "command": array(self.last_command),
                "reaction": array(self.last_reaction), "phase": self.phase, "ready": self.ready,
                "policy_elapsed_s": self.strike_steps * self.dt_s,
                "sequence_duration_s": self.sequence_duration_s,
                "attempt": self.attempt, "success": self.success, "hit": self.hit_now,
                "tip_contact": self.tip_contact, "disqualified": self.non_tip_first, "message": self.message}

    def recording(self) -> tuple[dict, dict]:
        frames = list(self.history)
        values = lambda key: np.asarray([frame[key] for frame in frames])
        arrays = {"time_s": values("time_s"), "cable_node_position_world_m": values("positions"),
                  "cable_node_velocity_world_m_s": values("velocities"), "commanded_force_world_n": values("command")[1:],
                  "effective_cable_reaction_on_point_world_n": values("reaction")[1:],
                  "controller_phase": values("phase"), "strike_attempt": values("attempt"),
                  "policy_elapsed_s": values("policy_elapsed_s"),
                  "strike_success": values("success"), "valid_hit": values("hit"),
                  "tip_contact": values("tip_contact"), "strike_disqualified": values("disqualified")}
        summary = {"schema": "point_force_simulation_summary_v1", "source": "live_pid_policy_flight",
                   "algorithm": getattr(self, "algorithm", "PPO"),
                   "device": "cpu", "dt_s": self.dt_s, "steps": len(frames) - 1,
                   "duration_s": frames[-1]["time_s"] - frames[0]["time_s"],
                   "checkpoint_path": str(self.checkpoint_path), "checkpoint_episodes": getattr(self, "checkpoint_episodes", 0),
                   "target_position_m": self.environment.target[0].tolist(),
                   "desired_strike_direction_world": self.environment.desired_direction[0].tolist(),
                   "success_condition": dict(self.environment.task_config["success"]),
                   "strike_execution": "initial_state_only_open_loop_once", "strike_plans": self.plans,
                   "success": self.success, "events": self.events, "hover_position_m": self.hover_position[0].tolist(),
                   "controller_phases": {"0": "PID hover", "1": "Policy force", "2": "PID recovery"}}
        return arrays, summary
