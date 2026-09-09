"""Score spline references through the accepted drone and cable models, on GPU.

The PPO environment supplies the existing first-contact strike gate only. Its
force action, actor and reward are not the CEM dynamics or objective.
"""
import copy
import numpy as np
import torch
from learning.point_force_env import PointForceWhipEnvironment, _replace_state_rows
from simulator.cable import CableConfiguration, DderModel
from simulator.cable.residual import FrozenMotionResidual
from simulator.research_physics import ResearchPhysics
from simulator.research_pose import ResearchPoseModel, settled_initial
from simulator.research_reference import reference_packet_validity
from .spline import decode, sample


class Cancelled(Exception):
    pass


class SplineEvaluator:
    def __init__(self, model, task, config, settings, device='cuda', cancelled=None):
        self.model, self.task, self.config = model, task, config
        self.settings, self.device, self.cancelled = settings, torch.device(device), cancelled
        self.dt = float(model['simulation']['dt_s'])
        if not np.isclose(self.dt, 1/150):
            raise ValueError('This planner requires the native 150 Hz physics / 30 Hz command model.')
        self.offset = np.asarray(model['recorded_data']['optitrack_to_attachment_offset_body_m'])
        self.origin = np.asarray(task['initial_root_position_m'])-self.offset
        spec = model['fullstate_execution']
        self.tracker = ResearchPoseModel(spec['checkpoint'], spec['sha256'], self.device)
        self.contexts = {}

    def check_cancelled(self):
        if self.cancelled and self.cancelled():
            raise Cancelled('Stopped by user; completed optimization candidates remain saved.')

    def context(self, batch):
        if batch not in self.contexts:
            task = copy.deepcopy(self.task)
            task['episode_duration_s'] = np.floor(self.settings['maximum_duration_s']*30+1e-9)/30
            env = PointForceWhipEnvironment(self.model, task, self.config, batch_size=batch, device=self.device)
            env.reset()
            cable = CableConfiguration.from_mapping(self.model['cable'])
            physics = DderModel(cable.dder_parameters(EI=self.model['cable']['EI_n_m2'], Cb=self.model['cable']['Cb_n_m2_s']))
            residual = self.model['motion_residual']
            physics.motion_residual = FrozenMotionResidual(residual['checkpoint'], residual['sha256'])
            integrator = ResearchPhysics(physics, env.state, self.dt, graph=self.device.type == 'cuda')
            self.contexts[batch] = env, integrator
        return self.contexts[batch]

    def splines(self, vectors):
        return [decode(v, self.origin, self.settings['control_points'],
                       (self.settings['minimum_duration_s'], self.settings['maximum_duration_s'])) for v in vectors]

    @torch.no_grad()
    def rollout(self, commands, durations, *, record=False):
        """Padded packets are never scored beyond each row's true duration."""
        self.check_cancelled()
        batch, packets = len(commands), np.asarray(commands)
        env, cable = self.context(batch)
        env.reset(); state = env.state
        env.physics_steps_per_control = 1
        count = (packets.shape[1]-1)*5
        env.control_step_count = count+1
        env.terminate_on_invalid_contact = False
        tensor = state.positions_m.new_tensor(packets)
        cutoff = torch.as_tensor(np.rint(np.asarray(durations)/self.dt), device=self.device, dtype=torch.long)
        clock = torch.arange(count+1, device=self.device)[None]
        within = clock <= cutoff[:, None]
        valid_packets, _ = reference_packet_validity(tensor, self.model['fullstate_execution']['feasibility'])
        command_within = torch.arange(packets.shape[1], device=self.device)[None]*5 <= cutoff[:, None]
        failed = ((~valid_packets) & command_within).any(1)
        hover = tensor[:, 0].clone(); hover[:, 3:9] = 0
        safe = torch.where(failed[:, None, None], hover[:, None], tensor)
        pose = self.tracker.predict(settled_initial(state.positions_m[:, 0], state.velocities_m_s[:, 0], self.offset),
            safe, np.arange(packets.shape[1])/30, np.arange(count+1)*self.dt, self.offset,
            graph=self.device.type == 'cuda', hover_command=hover)
        failed |= ((~pose['valid']) & within).any(1)
        z = pose['position_origin_m'][:, :, 2]
        failed |= (((z > self.settings['maximum_height_m']) | (z < self.settings['minimum_height_m'])) & within).any(1)
        peak = torch.where(within, z, -torch.inf).amax(1)
        dummy = state.positions_m.new_zeros(batch, 3)
        frames = [state.positions_m[0].cpu().numpy().copy()] if record else None
        closest_speed = state.positions_m.new_zeros(batch)
        closest_distance = env.initial_tip_distance.clone()
        for i in range(count):
            if i % 30 == 0:
                self.check_cancelled()
            previous = state
            candidate = cable(previous, pose['position_attachment_m'][:, i+1])
            q, v = candidate.positions_m, candidate.velocities_m_s
            finite = torch.isfinite(q).flatten(1).all(1) & torch.isfinite(v).flatten(1).all(1)
            bounded = (q.abs().flatten(1).amax(1) <= env.numerical_position_limit_m) & (v.norm(dim=-1).amax(1) <= env.numerical_speed_limit_m_s)
            height = (q[:, :, 2].amax(1) <= self.settings['maximum_height_m']) & (q[:, :, 2].amin(1) >= self.settings['minimum_height_m'])
            used = i < cutoff
            failed |= used & (~finite | ~bounded | ~height)
            state = _replace_state_rows(candidate, previous, used & ~failed)
            peak = torch.maximum(peak, torch.where(used & finite, q[:, :, 2].amax(1), peak))
            distance = (q[:, -1]-env.target).norm(dim=-1)
            improve = env.active & used & ~failed & (distance < closest_distance)
            closest_distance = torch.where(improve, distance, closest_distance)
            directed = (v[:, -1]*env.desired_direction).sum(1)
            closest_speed = torch.where(improve, directed, closest_speed)
            env.active &= used & ~failed
            transition = env.model._result(previous, state, dummy, self.dt)
            env.model.step_runtime = lambda *_: transition
            env.step(dummy)
            if record:
                frames.append(state.positions_m[0].cpu().numpy().copy())
        success = env.episode_success & ~failed
        result = dict(success=success.cpu().numpy(), failed=failed.cpu().numpy(),
                      distance=env.episode_minimum_tip_distance.cpu().numpy(),
                      closest_speed=closest_speed.cpu().numpy(), peak=peak.cpu().numpy(),
                      hit_time=env.episode_hit_time_s.cpu().numpy(),
                      invalid_contact=(env.episode_non_tip_first | env.episode_invalid_tip_entry).cpu().numpy())
        if record:
            result.update(cable=np.asarray(frames), origin=pose['position_origin_m'][0].cpu().numpy(),
                          rotations=pose['rotation_tracking_to_world'][0].cpu().numpy())
        return result

    @torch.no_grad()
    def __call__(self, vectors):
        splines = self.splines(vectors)
        maximum = max(int(round(d*30)) for _, d in splines)
        commands, dense_valid, jerk = [], [], []
        for c, duration in splines:
            times = np.arange(int(round(duration*150))+1)/150
            dense = sample(c, duration, times)
            valid, _ = reference_packet_validity(torch.as_tensor(dense)[None], self.model['fullstate_execution']['feasibility'])
            dense_valid.append(bool(valid.all()) and dense[:, 2].max() <= self.settings['maximum_height_m'] and dense[:, 2].min() >= self.settings['minimum_height_m'])
            jerk.append(float(np.mean(np.diff(dense[:, 6:9], axis=0)**2)*150**2))
            commands.append(sample(c, duration, np.minimum(np.arange(maximum+1)/30, duration)))
        result = self.rollout(commands, [d for _, d in splines])
        valid = np.asarray(dense_valid) & ~result['failed']
        terminal = np.where(result['success'], result['hit_time'], [d for _, d in splines])
        # Dedicated CEM objective, independent of the saved PPO reward.
        scores = (1000*result['success'] - 100*np.clip(result['distance'], 0, 10)
                  + 10*np.clip(result['closest_speed']/max(self.task['success']['minimum_directed_tip_speed_m_s'], .01), 0, 1)
                  - 25*terminal - .001*np.minimum(jerk, 10000)
                  - 100*result['invalid_contact'] - 20*np.maximum(result['peak']-2.6, 0))
        scores[~valid] = -np.inf
        return scores, dict(success_fraction=float(np.mean(result['success'] & valid)),
                            closest_tip_m=float(np.min(result['distance'][valid])) if valid.any() else None)
