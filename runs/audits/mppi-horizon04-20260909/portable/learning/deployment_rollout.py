"""Train a model-based strike planner against independent, open-loop execution.

The actor only sees a nominal model initialized from one state estimate.
Candidate commands are generated before the separate execution prediction.
Legacy modes use a frozen virtual-model cutoff; native execution-success mode
ends its scored episode at the predicted hit and exports that prefix offline.
No measured execution feedback enters action generation.
"""

from dataclasses import dataclass, replace
import math

import torch

from simulator.cable import DderState, FREE_ENDPOINTS
from simulator.live_flight import HoverPID
from .point_force_env import PointForceWhipEnvironment, _replace_state_rows


@dataclass
class DeploymentBatch:
    estimate: DderState
    truth: DderState
    stiffness_scale: torch.Tensor
    damping_scale: torch.Tensor
    force_gain: torch.Tensor
    force_lag_s: torch.Tensor
    nominal: torch.Tensor
    target_position_m: torch.Tensor | None = None


def sample_batch(env, settings, generator) -> DeploymentBatch:
    """Provisional uniform tolerances; straight cable rotations preserve lengths."""
    size = env.batch_size
    varied = (torch.rand((size, 1), device=env.device, generator=generator)
              >= float(settings["nominal_fraction"])).to(env.dtype)

    def noise(width, columns=3):
        return (2 * torch.rand((size, columns), device=env.device, dtype=env.dtype,
                               generator=generator) - 1) * float(width) * varied

    def ball(radius):
        radius = float(radius)
        if not math.isfinite(radius) or radius < 0:
            raise ValueError('Position randomization radius must be finite and nonnegative')
        if radius == 0:
            return torch.zeros((size, 3), device=env.device, dtype=env.dtype)
        direction = torch.randn((size, 3), device=env.device, dtype=env.dtype, generator=generator)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        distance = torch.rand((size, 1), device=env.device, dtype=env.dtype, generator=generator).pow(1/3)
        return direction * distance * radius * varied

    root = env._batch_vector(env.task_config["initial_root_position_m"])
    spherical_start = 'initial_position_radius_m' in settings
    root = root + (ball(settings['initial_position_radius_m']) if spherical_start
                   else noise(settings["initial_position_m"]))
    velocity = env._batch_vector(env.task_config["initial_root_velocity_m_s"])
    velocity = velocity + noise(settings["initial_velocity_m_s"])
    tilt = noise(math.radians(settings["initial_cable_tilt_deg"]), 2)
    omega = noise(settings["initial_angular_velocity_rad_s"])
    rest = root.new_tensor(env.model.cable_configuration.rest_lengths_m)

    def shape_noise(key):
        if not settings.get(key, 0.):return root.new_zeros(size, len(rest), 2)
        return noise(math.radians(settings[key]), 2 * len(rest)).reshape(size, len(rest), 2)
    bend = shape_noise('initial_cable_bend_deg')
    bend_error = shape_noise('state_cable_bend_error_deg')

    def state(position, speed, angle, angular_speed, shape):
        angles = angle[:, None] + shape
        direction = torch.cat((angles, -torch.ones_like(angles[..., :1])), dim=-1)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        segments = rest[None, :, None] * direction
        offsets = torch.cat((segments.new_zeros(size, 1, 3), segments.cumsum(1)), dim=1)
        velocities = speed[:, None] + torch.linalg.cross(
            angular_speed[:, None].expand_as(offsets), offsets)
        return DderState(position[:, None] + offsets, velocities)

    position_error = noise(settings['state_position_error_m'])
    # New spherical experiments bound the physical start; estimation error is
    # applied to the estimate. Legacy cube-based experiments retain their law.
    estimate = state(root + position_error if spherical_start else root,
                     velocity, tilt, omega, bend)
    truth = state(root if spherical_start else root + position_error,
                  velocity + noise(settings["state_velocity_error_m_s"]),
                  tilt + noise(math.radians(settings["state_cable_tilt_error_deg"]), 2),
                  omega, bend + bend_error)
    if settings.get('planning_cable_state')=='hanging':
        estimate=state(root+position_error if spherical_start else root,velocity,
            torch.zeros_like(tilt),torch.zeros_like(omega),torch.zeros_like(bend))
    lag = torch.rand((size, 1), device=env.device, dtype=env.dtype, generator=generator)
    # No extra RNG draws for legacy, fixed-target configurations.
    target = env._batch_vector(env.task_config['target_position_m']) + ball(settings.get('target_position_radius_m', 0.))
    return DeploymentBatch(estimate, truth,
                           1 + noise(settings["stiffness_fraction"], 1)[:, 0],
                           1 + noise(settings["damping_fraction"], 1)[:, 0],
                           1 + noise(settings["force_gain_fraction"]),
                           lag * float(settings["force_lag_max_s"]) * varied,
                           varied[:, 0] == 0, target)


@torch.no_grad()
def plan_batch(env, agent, batch, rollout=None, *, progress=None):
    terminal_execution = env.ppo_config['deployment'].get('termination') == 'execution_success_or_timeout'
    env.ignore_contact_termination = terminal_execution
    target = batch.target_position_m if batch.target_position_m is not None else env._batch_vector(env.task_config['target_position_m'])
    env.target = target.detach().clone()
    observation = env.reset(batch.estimate)
    forces = []
    if terminal_execution:
        env._virtual_reference_positions = [batch.estimate.positions_m[:,0].clone()]
        env._virtual_reference_velocities = [batch.estimate.velocities_m_s[:,0].clone()]
        env._planning_failure_steps = torch.full((env.batch_size,), env.control_step_count*env.physics_steps_per_control+1,
            device=env.device,dtype=torch.long)
    attempt_cutoffs=torch.zeros(env.batch_size,device=env.device,dtype=torch.long)
    if rollout is not None:
        for name in ("observations", "actions", "rewards", "dones", "masks",
                     "log_probabilities", "values"):
            getattr(rollout, name).zero_()

    def record(_state, force, _reaction):
        nonlocal attempt_cutoffs
        forces.append(force.clone())
        if terminal_execution:
            env._virtual_reference_positions.append(_state.positions_m[:,0].clone())
            env._virtual_reference_velocities.append(_state.velocities_m_s[:,0].clone())
            env._planning_failure_steps=torch.where(env.failed & (env._planning_failure_steps>len(forces)),
                torch.full_like(env._planning_failure_steps,len(forces)),env._planning_failure_steps)
        attempt_cutoffs=torch.where((attempt_cutoffs==0)&~env.active,
            torch.full_like(attempt_cutoffs,len(forces)),attempt_cutoffs)

    for index in range(env.control_step_count):
        if rollout is None:
            action = agent.deterministic_action(observation)
        else:
            action, log_probability, value = agent.act(observation)
        result = env.step(action, stop_when_all_done=True, physics_trace_callback=record)
        if rollout is not None:
            rollout.observations[index].copy_(observation)
            rollout.actions[index].copy_(action)
            rollout.log_probabilities[index].copy_(log_probability)
            rollout.values[index].copy_(value)
            rollout.dones[index].copy_(result.done)
            rollout.masks[index].copy_(result.include_transition[:, None])
        observation = result.next_observation
        if progress is not None:
            progress('Planning force sequences', index + 1, env.control_step_count)
        if not bool(env.active.any()):
            break
    # Legacy snapshots require a predicted hit. New experiments can execute
    # every finite attempt, using the same frozen first-contact/horizon cutoff.
    cutoffs = torch.where(env.episode_success,
                          torch.nan_to_num(env.episode_hit_time_s, nan=0.) / env.physics_dt_s,
                          0.).round().long()
    if not env.ppo_config['deployment'].get('require_predicted_success',True):
        attempt_cutoffs=torch.where(attempt_cutoffs>0,attempt_cutoffs,
            torch.full_like(attempt_cutoffs,len(forces)))
        cutoffs=attempt_cutoffs if terminal_execution else torch.where(env.failed,torch.zeros_like(attempt_cutoffs),attempt_cutoffs)
    from simulator.strike_sequence import freeze_followthrough
    frozen,cutoffs=freeze_followthrough(torch.stack(forces),cutoffs,dt_s=env.physics_dt_s,
        maximum_steps=round(env.task_config['episode_duration_s']/env.physics_dt_s),
        duration_s=0. if terminal_execution else env.ppo_config['deployment'].get('strike_followthrough_s',0.))
    if env.model_config.get('fullstate_execution',{}).get('schema')=='tracked_pose_execution_v1':
        from simulator.research_reference import packet_cutoffs
        frozen,cutoffs=packet_cutoffs(frozen,cutoffs,env.physics_steps_per_control,
            round(env.task_config['episode_duration_s']/env.physics_dt_s))
    return frozen,cutoffs


@torch.no_grad()
def execute_batch(nominal, batch, forces, cutoffs, settings, *, trace=None, progress=None):
    """Execute immutable commands, then the same PID used by live flight.

    The ordinary task environment is only a first-hit reward accumulator. Its
    frozen terminal state never feeds the independently evolving physical state.
    """
    if nominal.model_config.get('fullstate_execution', {}).get('enabled'):
        from .fullstate_rollout import execute_fullstate_batch
        return execute_fullstate_batch(nominal,batch,forces,cutoffs,settings,trace=trace,progress=progress)
    score = PointForceWhipEnvironment(nominal.model_config, nominal.task_config,
                                     nominal.ppo_config, batch_size=nominal.batch_size,
                                     device=nominal.device)
    score.target = (batch.target_position_m if batch.target_position_m is not None else nominal.target).detach().clone()
    score.reset(batch.truth)
    score.set_success_condition(target_radius_m=nominal.target_radius_m,
                                minimum_directed_speed_m_s=nominal.minimum_directed_speed_m_s,
                                maximum_tip_velocity_direction_error_deg=nominal.maximum_tip_velocity_direction_error_deg)
    score.physics_steps_per_control = 1
    maximum_steps = int(cutoffs.max())
    score.control_step_count = max(maximum_steps + 1, 1)
    dt = nominal.physics_dt_s
    recovery_steps = round(float(settings["recovery_duration_s"]) / dt)
    model = score.model
    state = batch.truth
    constants = model.dder.runtime_constants(state.positions_m)
    constants = replace(constants,
                        bending_stiffness_n_m2=constants.bending_stiffness_n_m2 * batch.stiffness_scale,
                        bending_damping_n_m2_s=constants.bending_damping_n_m2_s * batch.damping_scale)
    dt_tensor = state.positions_m.new_full((nominal.batch_size,), dt)
    pid = HoverPID(model, nominal._batch_vector(nominal.task_config["initial_root_position_m"]),
                   nominal.maximum_force_norm_n)
    hanging = model.hanging_state(pid.target).positions_m - pid.target[:, None]
    actual_force = nominal.hover_force_world_n.expand(nominal.batch_size, -1).clone()
    # A zero time constant is instantaneous; otherwise exact first-order hold.
    alpha = torch.where(batch.force_lag_s > 0,
                        1 - torch.exp(-dt / batch.force_lag_s.clamp_min(1e-12)), 1.)
    deployed = cutoffs > 0
    failed = torch.zeros_like(deployed)
    recovered = torch.zeros_like(deployed)
    settled = torch.zeros_like(cutoffs)
    recovery_error = torch.zeros_like(score.episode_reward)
    execution_maximum_displacement=torch.zeros_like(score.episode_reward)
    dummy_action = torch.zeros((nominal.batch_size, 3), device=nominal.device)
    initial_failure = nominal.failed.clone()
    graph_physics=None
    physics_rows=None
    if maximum_steps and state.positions_m.is_cuda and nominal.ppo_config.get('cuda_graph_physics', False):
        from simulator.cuda_graph_physics import CudaGraphPhysics
        # Refused plans never evolve in the plant. Gather once for the fixed
        # CUDA graph, then scatter into the original batch for scoring/records.
        if nominal.ppo_config.get('compact_deployment_physics', True):
            physics_rows=torch.where(deployed)[0]
        if physics_rows is not None and len(physics_rows)<nominal.batch_size:
            graph_state=DderState(state.positions_m[physics_rows],state.velocities_m_s[physics_rows])
            graph_constants=model.dder.runtime_constants(graph_state.positions_m)
            graph_constants=replace(graph_constants,
                bending_stiffness_n_m2=graph_constants.bending_stiffness_n_m2*batch.stiffness_scale[physics_rows],
                bending_damping_n_m2_s=graph_constants.bending_damping_n_m2_s*batch.damping_scale[physics_rows])
            graph_physics=CudaGraphPhysics(model,graph_state,dt,graph_constants)
        else:
            physics_rows=None
            graph_physics=CudaGraphPhysics(model,state,dt,constants)
    for index in range(maximum_steps + recovery_steps if bool(deployed.any()) else 0):
        from .training_control import check_training_stop
        check_training_stop()
        striking = deployed & (index < cutoffs)
        recovering = deployed & (index >= cutoffs) & (index < cutoffs + recovery_steps)
        evolving = (striking | recovering) & ~failed
        # PID is permitted after the frozen cutoff; it has no strike authority.
        saved_integral = pid.integral.clone()
        hover = pid.command(state, dt)
        pid.integral.copy_(torch.where(recovering[:, None], pid.integral, saved_integral))
        planned = forces[index] if index < len(forces) else hover
        command = torch.where(striking[:, None], planned, hover)
        actual_force = actual_force + alpha * (batch.force_gain * command - actual_force)
        actual_force[:, 2].clamp_(min=nominal.minimum_vertical_force_n)
        actual_force *= (nominal.maximum_force_norm_n /
                         actual_force.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1.)
        previous = state
        if physics_rows is not None:
            subset=graph_physics(DderState(previous.positions_m[physics_rows],previous.velocities_m_s[physics_rows]),actual_force[physics_rows])
            candidate=DderState(previous.positions_m.index_copy(0,physics_rows,subset.positions_m),
                                previous.velocities_m_s.index_copy(0,physics_rows,subset.velocities_m_s))
        else:
            candidate = graph_physics(previous,actual_force) if graph_physics is not None else model.dder.step_runtime(
                previous, previous.positions_m[:, :0], dt_tensor, constants,
                external_force_world_n=model.controller.node_forces(actual_force, validate=False),
                iterative_damping=state.positions_m.is_cuda,
                damping_backend="pcg32_experimental" if state.positions_m.is_cuda else "pcg60_reference",
                pinned_endpoints=FREE_ENDPOINTS, create_graph=False)
        finite = (torch.isfinite(candidate.positions_m).flatten(1).all(-1)
                  & torch.isfinite(candidate.velocities_m_s).flatten(1).all(-1))
        bounded = ((candidate.positions_m.abs().flatten(1).amax(-1) <= nominal.numerical_position_limit_m)
                   & (candidate.velocities_m_s.norm(dim=-1).amax(-1) <= nominal.numerical_speed_limit_m_s))
        failed |= evolving & ~(finite & bounded)
        score.episode_nonfinite |= evolving & ~finite
        score.episode_position_limit |= evolving & finite & (candidate.positions_m.abs().flatten(1).amax(-1) > nominal.numerical_position_limit_m)
        score.episode_speed_limit |= evolving & finite & (candidate.velocities_m_s.norm(dim=-1).amax(-1) > nominal.numerical_speed_limit_m_s)
        state = _replace_state_rows(candidate, previous, evolving & finite & bounded)
        if index < maximum_steps:
            score.active &= striking
            transition = model._result(previous, candidate, actual_force, dt)
            # Reuse the task's gate and reward equations without a second physics
            # integration or allowing the scorer to control the physical plant.
            score.model.step_runtime = lambda _state, _force, _dt: transition
            score.step(dummy_action)
        root_error = (state.positions_m[:, 0] - pid.target).norm(dim=-1)
        execution_maximum_displacement=torch.maximum(execution_maximum_displacement,
            (state.positions_m[:,0]-batch.truth.positions_m[:,0]).norm(dim=-1)*deployed.to(score.dtype))
        cable_error = (state.positions_m - state.positions_m[:, :1] - hanging).norm(dim=-1).amax(-1)
        speed = state.velocities_m_s.norm(dim=-1).amax(-1)
        stable = recovering & ~failed & (root_error < .03) & (cable_error < .04) & (speed < .15)
        settled = torch.where(stable, settled + 1, 0)
        recovered |= settled >= round(.5 / dt)
        at_end = deployed & (index + 1 == cutoffs + recovery_steps)
        recovery_error = torch.where(at_end, root_error, recovery_error)
        if trace is not None:
            trace(index, command.clone(), state, striking.clone(), score.episode_success.clone())
        if progress is not None:
            progress('Executing force sequences and PID recovery', index + 1, maximum_steps + recovery_steps)

    # Charge the planned duration, never a feedback-dependent stopping time.
    scored_seconds = torch.where(score.episode_success, score.episode_hit_time_s,
                                  cutoffs * dt)
    score.episode_reward -= score.reward_weights.time_per_s * (cutoffs * dt - scored_seconds)
    score.episode_component_sums['time'] -= score.reward_weights.time_per_s * (cutoffs * dt - scored_seconds)
    recovery_cost = float(settings["recovery_failure_penalty"]) * (deployed & ~recovered).to(score.dtype)
    score.episode_reward -= recovery_cost
    score.episode_reward -= score.reward_weights.numerical_failure * (failed & ~score.failed).to(score.dtype)
    score.episode_component_sums['numerical_failure'] -= score.reward_weights.numerical_failure * (failed & ~score.failed).to(score.dtype)
    score.failed |= failed | initial_failure
    for name in ('episode_nonfinite','episode_position_limit','episode_speed_limit'):
        setattr(score,name,getattr(score,name) | getattr(nominal,name))
    score.episode_timed_out = deployed & ~score.episode_success & ~score.failed
    # Failed planning still receives its nominal progress signal, but cannot
    # count as an executed hit. This allows PPO to learn to produce valid plans.
    score.episode_reward = torch.where(deployed, score.episode_reward, nominal.episode_reward)
    for name in score.episode_component_sums:
        score.episode_component_sums[name]=torch.where(deployed,
            score.episode_component_sums[name],nominal.episode_component_sums[name])
    score.episode_component_sums['recovery']=-recovery_cost
    score.deployment = {
        "planned": deployed,
        "predicted_valid_hit": nominal.episode_success.clone(),
        "recovered": recovered & ~failed,
        "joint_success": score.episode_success & recovered & ~failed,
        "duration_s": cutoffs * dt,
        "recovery_position_error_m": recovery_error,
        "maximum_execution_drone_displacement_m": execution_maximum_displacement,
        "nominal": batch.nominal,
        "planning_minimum_tip_distance_m": nominal.episode_minimum_tip_distance.clone(),
        "planning_maximum_drone_displacement_m": nominal.episode_maximum_point_displacement.clone(),
        "planning_failure": nominal.failed.clone(),
        "planning_non_tip_first": nominal.episode_non_tip_first.clone(),
        "planning_invalid_tip_entry": nominal.episode_invalid_tip_entry.clone(),
        "planning_tip_entry_speed": nominal.episode_first_tip_directed_speed.clone(),
        "planning_tip_entry_angle": nominal.episode_first_tip_angle.clone(),
        **{f'target_{axis}_m': score.target[:, i].clone() for i, axis in enumerate('xyz')},
        **{f'initial_attachment_{axis}_m': batch.truth.positions_m[:, 0, i].clone() for i, axis in enumerate('xyz')},
    }
    score.execution_state = state
    return score


@torch.no_grad()
def collect_deployment_rollout(environment, agent, rollout=None, *, generator=None, recorder=None, progress=None):
    settings = environment.ppo_config["deployment"]
    batch = sample_batch(environment, settings, generator)
    forces, cutoffs = plan_batch(environment, agent, batch, rollout, progress=progress)
    live_recorder = None
    if rollout is not None and getattr(environment, '_live_scene_context', None):
        from .live_scene import TrainingSceneRecorder
        live_recorder = TrainingSceneRecorder(environment._live_scene_context)
        if recorder is not None:
            raise ValueError('Training scene and evaluation recorders must not share a collection.')
        recorder = live_recorder
    if recorder is not None:
        recorder.begin(environment, batch, cutoffs)
    score = execute_batch(environment, batch, forces, cutoffs, settings,
                          trace=recorder.trace if recorder is not None else None, progress=progress)
    if hasattr(score, 'terminal_steps') and recorder is not None:
        recorder.cutoff = float(score.terminal_steps[0]) * environment.physics_dt_s
    if rollout is not None:
        if hasattr(score, 'terminal_steps'):
            trim_terminal_rollout(rollout, score.terminal_steps, environment.physics_steps_per_control)
        # Monte Carlo credit for the configured execution objective. Native
        # hit-termination mode credits only its terminal prefix; recovery is excluded.
        # gamma=gae_lambda=1 avoids adding an implicit preference for fast hits.
        final_steps = rollout.masks[:, :, 0].sum(0).long() - 1
        rollout.rewards[final_steps, torch.arange(environment.batch_size, device=environment.device), 0] = score.episode_reward.float()
    if live_recorder is not None:
        live_recorder.publish(agent, score, rollout, forces)
    for name, value in vars(score).items():
        if name.startswith("episode_") or name in ("failed", "deployment", "displacement_allowance_m"):
            setattr(environment, name, value)
    return score


def trim_terminal_rollout(rollout, terminal_steps, steps_per_control):
    """Only commands generated before the execution terminal event receive credit."""
    lengths=((terminal_steps+steps_per_control-1)//steps_per_control).clamp_min(1)
    lengths=torch.minimum(lengths,rollout.masks[:,:,0].sum(0).long())
    live=torch.arange(len(rollout.masks),device=lengths.device)[:,None]<lengths[None]
    rollout.masks.mul_(live[:,:,None]);rollout.rewards.zero_();rollout.dones.zero_()
    rollout.dones[lengths-1,torch.arange(len(lengths),device=lengths.device),0]=1.


@torch.no_grad()
def evaluate_deployment(model, task, config, agent, *, episodes, batch_size, device):
    import uuid
    import numpy as np
    from .experiment_records import EvaluationResult
    class Recorder:
        def begin(self, env, batch, cutoffs):
            self.dt = env.physics_dt_s
            self.positions = [batch.truth.positions_m[:1].clone()]
            self.velocities = [batch.truth.velocities_m_s[:1].clone()]
            self.commands = [env.hover_force_world_n.reshape(1, 3).clone()]
            self.hits = [cutoffs[:1].new_zeros(1, dtype=torch.bool)]
            self.striking = [cutoffs[:1] > 0]
            self.cutoff = float(cutoffs[0]) * self.dt
            self.scenarios = {
                'estimate_positions_m': batch.estimate.positions_m,
                'estimate_velocities_m_s': batch.estimate.velocities_m_s,
                'truth_positions_m': batch.truth.positions_m,
                'truth_velocities_m_s': batch.truth.velocities_m_s,
                'target_position_m': env.target.clone(),
                **{key: getattr(batch, key) for key in ('stiffness_scale', 'damping_scale',
                                                       'force_gain', 'force_lag_s', 'nominal')}}

        def trace(self, index, command, state, striking, hit):
            self.positions.append(state.positions_m[:1].clone())
            self.velocities.append(state.velocities_m_s[:1].clone())
            self.commands.append(command[:1].clone())
            self.hits.append(hit[:1].clone())
            self.striking.append(striking[:1].clone())

    # Private RNG makes comparisons repeatable and does not consume exploration RNG.
    generator = torch.Generator(device=device).manual_seed(int(config["deployment"]["validation_seed"]))
    rows, recorders = [], []
    for completed in range(0, episodes, batch_size):
        env = PointForceWhipEnvironment(model, task, config,
                                        batch_size=min(batch_size, episodes - completed), device=device)
        recorder = Recorder()
        score = collect_deployment_rollout(env, agent, generator=generator, recorder=recorder)
        recorders.append(recorder)
        rows.append({
            "success": score.episode_success, "non_tip": score.episode_non_tip_first,
            "invalid_tip": score.episode_invalid_tip_entry, "timeout": score.episode_timed_out,
            "failure": score.failed, "reward": score.episode_reward,
            "impact_speed": score.episode_impact_speed,
            "hit_tip_directed_speed": score.episode_hit_tip_directed_speed,
            "hit_relative_tip_directed_speed": score.episode_hit_relative_tip_directed_speed,
            "hit_attachment_directed_speed": score.episode_hit_attachment_directed_speed,
            "displacement_allowance_m": score.displacement_allowance_m,
            **{'reward_'+key:value for key,value in score.episode_component_sums.items()},
            "nonfinite": score.episode_nonfinite, "position_limit": score.episode_position_limit,
            "speed_limit": score.episode_speed_limit,
            "distance": score.episode_minimum_tip_distance,
            "displacement": score.episode_maximum_point_displacement,
            "integral": score.episode_point_displacement_integral_m_s,
            "cost_integral": score.episode_point_displacement_cost_integral_s,
            "hit_time": score.episode_hit_time_s, **score.deployment})
    data = {key: torch.cat([row[key].detach().cpu() for row in rows]) for key in rows[0]}
    def mean(key):
        return float(data[key].double().mean())
    def finite_mean(key):
        values=data[key][torch.isfinite(data[key])]
        return float(values.mean()) if len(values) else None
    hits = data["hit_time"][torch.isfinite(data["hit_time"])]
    nominal_count = int(data["nominal"].sum())
    result = EvaluationResult({
        "evaluation_mode": "initial_state_only_open_loop_once_with_pid_recovery",
        "uncertainty_source": config["deployment"]["uncertainty_source"],
        "validation_seed": config["deployment"]["validation_seed"],
        "episodes": episodes, "successes": int(data["success"].sum()),
        "success_rate": mean("success"), "plan_success_rate": mean("predicted_valid_hit"),
        "execution_rate": mean('planned'),
        "nominal_episodes": nominal_count,
        "nominal_success_rate": float(data["success"][data["nominal"]].double().mean()) if nominal_count else None,
        "nominal_hit_and_recovery_rate": float(data["joint_success"][data["nominal"]].double().mean()) if nominal_count else None,
        "recovery_rate": mean("recovered"), "hit_and_recovery_rate": mean("joint_success"),
        "mean_planned_duration_s": mean("duration_s"),
        "mean_recovery_position_error_m": mean("recovery_position_error_m"),
        "mean_maximum_execution_drone_displacement_m": mean('maximum_execution_drone_displacement_m'),
        "non_tip_first_rate": mean("non_tip"), "invalid_tip_entry_rate": mean("invalid_tip"),
        "timeout_rate": mean("timeout"), "numerical_failure_rate": mean("failure"),
        "nonfinite_rate": mean("nonfinite"), "position_limit_rate": mean("position_limit"),
        "speed_limit_rate": mean("speed_limit"),
        "mean_episode_reward": mean("reward"),
        "median_minimum_tip_distance_m": float(data["distance"].median()),
        "mean_maximum_point_displacement_m": mean("displacement"),
        "mean_point_displacement_integral_m_s": mean("integral"),
        "mean_point_displacement_cost_integral_s": mean("cost_integral"),
        "mean_hit_time_s": float(hits.mean()) if len(hits) else None,
        "mean_impact_speed_m_s": float(data['impact_speed'][data['success']].mean()) if bool(data['success'].any()) else None,
        **{f'mean_{key}_m_s': float(data[key][data['success']].mean()) if bool(data['success'].any()) else None
           for key in ('hit_tip_directed_speed', 'hit_relative_tip_directed_speed', 'hit_attachment_directed_speed')},
        "median_planning_minimum_tip_distance_m": float(data['planning_minimum_tip_distance_m'].median()),
        "mean_planning_maximum_drone_displacement_m": mean('planning_maximum_drone_displacement_m'),
        "planning_non_tip_first_rate": mean('planning_non_tip_first'),
        "planning_invalid_tip_entry_rate": mean('planning_invalid_tip_entry'),
        "mean_planning_tip_entry_directed_speed_m_s": finite_mean('planning_tip_entry_speed'),
        "mean_planning_tip_entry_angle_deg": finite_mean('planning_tip_entry_angle'),
        "reward_components": {key.removeprefix('reward_'):mean(key) for key in data if key.startswith('reward_')},
    })
    first = recorders[0]
    if model.get('fullstate_execution',{}).get('schema')=='tracked_pose_execution_v1':
        result['evaluation_mode']='30hz_frozen_reference_tracked_pose_and_cable_whip_only'
        result['reference_infeasible_rate']=mean('reference_infeasible')
        result['pose_domain_failure_rate']=mean('pose_domain_failure')
        result['mean_maximum_reference_tilt_deg']=mean('maximum_reference_tilt_deg')
        result['mean_maximum_reference_speed_m_s']=mean('maximum_reference_speed_m_s')
        result['mean_reference_position_correction_m']=mean('reference_position_correction_m')
        if 'termination_time_s' in data:
            result['mean_termination_time_s']=mean('termination_time_s')
            result['termination_mode']='execution_success_or_timeout'
    arrays = {key: torch.stack(values).cpu().numpy() for key, values in (
        ('positions_m', first.positions), ('velocities_m_s', first.velocities), ('commanded_force_world_n', first.commands),
        ('hit', first.hits), ('striking', first.striking))}
    if config['deployment'].get('termination')=='execution_success_or_timeout':
        end=round(first.cutoff/first.dt)+1
        arrays={key:value[:end] for key,value in arrays.items()}
    arrays['time_s'] = np.arange(len(arrays['positions_m'])) * first.dt
    result.recording = (arrays, dict(target_position_m=first.scenarios['target_position_m'][0].cpu().tolist(),
        desired_strike_direction_world=task['desired_strike_direction_world'],
        target_radius_m=task['success']['tip_target_distance_m'], cutoff_s=first.cutoff,
        first_trial_success=bool(data['success'][0]),
        first_trial_recovered=bool(data['recovered'][0]), first_trial_planned=bool(data['planned'][0])))
    result.scenarios = {key: torch.cat([rec.scenarios[key].detach().cpu() for rec in recorders]).numpy()
                        for key in first.scenarios}
    result.trials = [{key: values[index].item() for key, values in data.items()}
                     for index in range(episodes)]
    result.evaluation_id = uuid.uuid4().hex
    return result
