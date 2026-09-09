"""Paced simulation with a separate planner; all Qt rendering stays on the UI thread."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
from queue import Empty, SimpleQueue
import threading
import time

import numpy as np
import torch
from PySide6.QtCore import QObject, Signal, Slot

from deployment.package import digest
from deployment.planner import write_plan
from deployment.rehearsal import save_rehearsal
from deployment.tracking_rehearsal import TrackingRehearsalFlight as RehearsalFlight
from simulator.live_flight import prepare_live_physics
from simulator.gpu_rehearsal import prepare_gpu_rehearsal
from simulator.strike_plan import compile_strike_plan


class RehearsalWorker(QObject):
    status = Signal(str)
    plan_ready = Signal(str, object)
    plan_invalidated = Signal()
    failed = Signal(str)
    finished = Signal()

    def __init__(self, configs, checkpoint, directory, *, device='cuda', controller_export=None, fullstate_mode=False):
        super().__init__()
        self.configs = copy.deepcopy(configs)
        self.checkpoint, self.directory = Path(checkpoint), Path(directory)
        self.fullstate_mode = fullstate_mode
        self.device = device
        self.controller_export = dict(controller_export or {})
        self.stop_requested = threading.Event()
        self.commands = SimpleQueue()
        self.frame_lock = threading.Lock()
        self.latest_frame = None

    def prepare_plan(self, *args, **kwargs):
        plan = compile_strike_plan(*args, **kwargs)
        # Full-state export samples the existing live rehearsal after recovery.
        # Do not simulate a second strike or introduce another recovery controller.
        return plan, None

    def command(self, name, value=None):
        if name == 'stop':
            self.stop_requested.set()
        else:
            self.commands.put((name, value))

    def take_frame(self):
        with self.frame_lock:
            frame, self.latest_frame = self.latest_frame, None
        return frame

    @Slot()
    def run(self):
        flight = executor = planning = timer = None
        cancel = threading.Event()
        plan = None
        plan_number = revision = 0
        auto_plan = True
        notice = 'Approaching hover position'
        outcome = 'Stopped by user'
        events = []
        wall_samples, physics_times, lateness = [], [], []
        hardware = self.device
        try:
            torch.set_num_threads(1)
            self.status.emit('Loading PPO and warming up GPU physics…' if self.device == 'cuda' else 'Preparing CPU test physics…')
            flight = RehearsalFlight.from_checkpoint(*self.configs, self.checkpoint)
            if self.device == 'cuda':
                flight.physics, planning_physics, flight.policy = prepare_gpu_rehearsal(flight)
                hardware = torch.cuda.get_device_name()
            elif self.device == 'cpu':
                flight.physics = prepare_live_physics(json.dumps(self.configs[0], sort_keys=True))
                planning_physics = flight.physics
            else:
                raise ValueError('Unknown rehearsal device')
            policy_hash = digest(self.checkpoint)
            flight.begin_approach()
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rehearsal_planner')
            if __import__('os').name == 'nt':
                import ctypes
                winmm = ctypes.WinDLL('winmm')
                if winmm.timeBeginPeriod(1) == 0:
                    timer = winmm
            deadline = window_start = time.perf_counter()
            wall_start = deadline
            window_sim, rate = 0., 1.
            while not self.stop_requested.is_set():
                iteration_started = time.perf_counter()
                wall_samples.append(iteration_started-wall_start)
                lateness.append(max(0., iteration_started-deadline))
                request_plan = False
                try:
                    while True:
                        name, value = self.commands.get_nowait()
                        if name == 'target' and flight.attempt == 0:
                            flight.set_target(value)
                            revision += 1
                            plan = None
                            self.plan_invalidated.emit()
                            notice = 'Target changed; generate a new plan when hover is settled'
                        elif name == 'plan' and flight.attempt == 0:
                            request_plan = True
                        elif name == 'execute' and plan is not None and flight.attempt == 0:
                            try:
                                flight.start_strike(plan)
                                events.append(dict(event='execute', time_s=flight.time_s,
                                                   plan_directory=str(plan_directory)))
                                notice = 'Executing frozen force sequence'
                            except ValueError as error:
                                notice = str(error) + ' Use Generate again.'
                                events.append(dict(event='launch_refused', time_s=flight.time_s, reason=str(error)))
                            plan = None
                            self.plan_invalidated.emit()
                except Empty:
                    pass
                if auto_plan and flight.ready:
                    request_plan, auto_plan = True, False
                if request_plan and planning is None and flight.ready:
                    plan = None
                    self.plan_invalidated.emit()
                    plan_number += 1
                    plan_directory = self.directory / f'plan_{plan_number:03d}'
                    planning_task = copy.deepcopy(flight.environment.task_config)
                    state_time, planning_revision = flight.time_s, revision
                    initial = flight.launch_state()  # No true cable node is read by this adapter.
                    planning_task['target_position_m'] = flight.tracking.packet.target_m.tolist()
                    state_time = flight.tracking.packet.time_s
                    started = time.perf_counter()
                    planning = executor.submit(self.prepare_plan, self.configs[0], planning_task,
                        self.configs[2], initial, flight.policy, physics=planning_physics,
                        cancel_requested=cancel.is_set)
                    notice = 'Hovering while generating the force sequence…'
                elif request_plan:
                    notice = 'Wait for settled hover and the current planning job'
                if planning is not None and planning.done():
                    try:
                        candidate, trajectory = planning.result()
                        metadata = write_plan(candidate, plan_directory, task=planning_task,
                            **self.controller_export,
                            metadata=dict(initialization='assumed_vertical_cable', simulation_only=True,
                                state_time_s=state_time, planning_wall_seconds=time.perf_counter()-started,
                                policy_sha256=policy_hash, planning_device=self.device,
                                target_changed_during_planning=planning_revision != revision))
                        if planning_revision == revision:
                            plan = candidate
                            self.plan_ready.emit(str(plan_directory / 'commands.csv'), metadata)
                            notice = ('Force plan ready — execute the rehearsal to record whip + original PID recovery'
                                      if self.fullstate_mode else 'CSV ready — inspect it, then Execute sequence')
                        else:
                            notice = 'Target changed during planning; CSV saved but launch disabled. Generate again.'
                    except Exception as error:
                        notice = f'Plan refused: {error}'
                        events.append(dict(event='planning_failed', reason=str(error), time_s=flight.time_s))
                        self.status.emit(notice)
                    planning = None
                physics_started = time.perf_counter()
                applied_phase = flight.phase
                frame = flight.step()
                frame['applied_controller_phase'] = applied_phase
                physics_times.append(time.perf_counter()-physics_started)
                frame['tracking_wall_time_s'] = iteration_started-wall_start
                if flight.attempt:
                    notice = ('Executing frozen force sequence' if flight.phase == flight.POLICY
                              else 'Normal controller recovering to hover')
                elif planning is None and plan is None and auto_plan:
                    notice = f'Hover settling: {min(flight.settled_s, 10.):.1f} / 10.0 s'
                now = time.perf_counter()
                if now - window_start >= 1.:
                    rate = (flight.time_s - window_sim) / (now - window_start)
                    window_start, window_sim = now, flight.time_s
                    (self.directory/'status.json').write_text(json.dumps(dict(
                        message=notice, simulation_time_s=flight.time_s, settled_s=flight.settled_s,
                        realtime_rate=rate, root_position_m=frame['positions'][0].tolist(),
                        root_velocity_m_s=frame['velocities'][0].tolist(), device=hardware,
                        tracking_hz=100*rate, control_hz=20*rate), indent=2)+'\n', encoding='utf-8')
                frame.update(message=notice, realtime_rate=rate, preparing=planning is not None,
                             target=flight.environment.target[0].tolist(), settled_s=flight.settled_s,
                             has_plan=plan is not None, runtime_device=hardware,
                             tracking_hz=100*rate, control_hz=20*rate)
                with self.frame_lock:
                    self.latest_frame = frame
                if flight.attempt and flight.ready:
                    if self.fullstate_mode:
                        from deployment.fullstate import write_rehearsal_fullstate
                        self.status.emit('Exporting unchanged whip with smooth braking and gentle return…')
                        arrays, summary = flight.recording()
                        reference_metadata = write_rehearsal_fullstate(arrays, summary, plan_directory, metadata,
                            gentle_recovery=True, attachment_offset_tracking_m=(
                                self.configs[0]['recorded_data']['optitrack_to_attachment_offset_body_m']
                                if self.configs[0].get('fullstate_execution',{}).get('enabled') else None),
                            initial_tracking_orientation_xyzw=(
                                [0.,0.,0.,1.] if self.configs[0].get('fullstate_execution',{}).get('enabled') else None),
                            initial_pose_source='nominal level simulation with tracking axes aligned to world; not a measured flight pose')
                        self.plan_ready.emit(str(plan_directory/'fullstate_30hz.csv'), reference_metadata)
                    outcome = 'Completed — ' + ('valid simulated hit' if flight.success else 'no valid simulated hit')
                    break
                deadline += flight.dt_s
                if now - deadline > flight.dt_s:
                    deadline = now
                self.stop_requested.wait(max(0., deadline - time.perf_counter()))
        except Exception as error:
            outcome = f'{type(error).__name__}: {error}'
            self.failed.emit(outcome)
        finally:
            cancel.set()
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            if timer is not None:
                timer.timeEndPeriod(1)
            try:
                if flight is not None:
                    elapsed = wall_samples[-1]-wall_samples[0] if len(wall_samples)>1 else 0.
                    indices = [round(t/flight.dt_s) for t in getattr(flight,'controller_updates',[])]
                    control_wall = np.asarray([wall_samples[i] for i in indices if i < len(wall_samples)])
                    timing = dict(device=hardware,
                        measured_tracking_hz=(len(wall_samples)-1)/elapsed if elapsed else 0.,
                        measured_control_hz=(len(control_wall)-1)/(control_wall[-1]-control_wall[0]) if len(control_wall)>1 else 0.,
                        physics_step_mean_ms=float(np.mean(physics_times))*1000 if physics_times else 0.,
                        physics_step_p95_ms=float(np.percentile(physics_times,95))*1000 if physics_times else 0.,
                        tick_interval_p95_ms=float(np.percentile(np.diff(wall_samples),95))*1000 if elapsed else 0.,
                        late_over_10ms=sum(value>.01 for value in lateness))
                    np.savez_compressed(self.directory/'timing.npz', tracking_wall_time_s=wall_samples,
                        controller_wall_time_s=control_wall, physics_step_wall_s=physics_times, tick_lateness_s=lateness)
                    save_rehearsal(self.directory, flight, outcome=outcome, preparation_events=events,
                                   runtime_device=self.device, timing=timing)
            except Exception as error:
                self.failed.emit(f'Could not save rehearsal: {error}')
            self.status.emit(outcome)
            self.finished.emit()
