"""Wall-clock paced flight, independent of the GUI/rendering thread."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, SimpleQueue
import threading
import time

from PySide6.QtCore import QObject, Signal, Slot
import torch

from simulator.live_flight import LiveFlight, prepare_live_physics
from simulator.cable import DderState


class LiveFlightWorker(QObject):
    finished = Signal(object, object)
    failed = Signal(str)
    status = Signal(str)

    def __init__(self, model, task, ppo, checkpoint):
        super().__init__()
        self.configs = model, task, ppo
        self.checkpoint = checkpoint
        self.commands = SimpleQueue()
        self.stop_requested = threading.Event()
        self.strike_pending = threading.Event()
        self.frame_lock = threading.Lock()
        self.latest_frame = None

    def command(self, name):
        if name == "stop":
            self.stop_requested.set()
        else:
            if name == 'strike':
                if self.strike_pending.is_set():
                    return
                self.strike_pending.set()
            self.commands.put(name)

    def take_frame(self):
        with self.frame_lock:
            frame, self.latest_frame = self.latest_frame, None
        return frame

    @Slot()
    def run(self):
        flight = None
        timer = None
        executor = None
        planning = None
        plan_cancel = threading.Event()
        notice = None
        try:
            # Training runs in its own process. One CPU thread is faster for
            # the small dense matrices in this single-cable simulation.
            torch.set_num_threads(1)
            flight = LiveFlight.from_checkpoint(*self.configs, self.checkpoint)
            self.status.emit("Preparing live physics…")
            flight.physics = prepare_live_physics(json.dumps(self.configs[0], sort_keys=True))
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="strike_planner")
            if os.name == "nt":
                import ctypes
                winmm = ctypes.WinDLL("winmm")
                # Windows otherwise rounds short Event waits to about 15.6 ms,
                # longer than one complete physics step.
                if winmm.timeBeginPeriod(1) == 0:
                    timer = winmm
            deadline = window_start = time.perf_counter()
            window_sim = flight.time_s
            rate = 1.
            while not self.stop_requested.is_set():
                try:
                    while True:
                        command = self.commands.get_nowait()
                        if command == "strike":
                            if planning is not None or not flight.ready:
                                raise ValueError("Wait for the cable to settle before preparing a strike.")
                            initial = DderState(flight.state.positions_m.clone(), flight.state.velocities_m_s.clone())
                            plan_cancel.clear()
                            notice = None
                            planning = executor.submit(flight.plan_strike, initial, plan_cancel.is_set)
                        elif command == "hover":
                            plan_cancel.set()
                            notice = None
                            flight.return_to_hover()
                except Empty:
                    pass
                except ValueError as error:
                    self.strike_pending.clear()
                    notice = str(error)
                if planning is not None and planning.done():
                    try:
                        plan = planning.result()
                        if not plan_cancel.is_set() and not self.stop_requested.is_set():
                            flight.start_strike(plan)
                            notice = None
                    except InterruptedError:
                        pass
                    except ValueError as error:
                        notice = str(error)
                    planning = None
                    self.strike_pending.clear()
                frame = flight.step()
                now = time.perf_counter()
                if now - window_start >= 1.:
                    rate = (flight.time_s - window_sim) / (now - window_start)
                    window_start, window_sim = now, flight.time_s
                frame = {**frame, "realtime_rate": rate, "preparing": planning is not None}
                if planning is not None:
                    frame.update(ready=False, message="PID hover — preparing one strike from the initial state")
                elif notice is not None:
                    frame["message"] = notice
                with self.frame_lock:
                    self.latest_frame = frame
                deadline += flight.dt_s
                # Never skip physics or accumulate a long catch-up burst.
                if now - deadline > flight.dt_s:
                    deadline = now
                self.stop_requested.wait(max(0., deadline - time.perf_counter()))
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")
        finally:
            plan_cancel.set()
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            if timer is not None:
                timer.timeEndPeriod(1)
            if flight is not None and len(flight.history) > 1:
                self.finished.emit(*flight.recording())
            else:
                self.finished.emit(None, None)
