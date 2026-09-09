"""Per-maneuver calibration and deadline scheduling; no ROS or battery dependency."""
from dataclasses import asdict, dataclass
from collections import deque
import csv
import json
from pathlib import Path
import time

import numpy as np


@dataclass(frozen=True)
class FrozenCalibration:
    scale: float
    samples: int
    duration_s: float
    mean_output: float
    output_std_fraction: float
    mass_kg: float
    gravity_m_s2: float
    mass_thrust: float

    def acceleration(self, residual_force_n):
        """Raw residual CSV -> corrected total thrust -> firmware acceleration.

        Firmware massThrust is left unchanged. Do not apply this correction
        again in firmware or scale only residual Fz.
        """
        f = np.asarray(residual_force_n, dtype=float)
        if f.shape != (3,) or not np.isfinite(f).all():
            raise ValueError('Expected finite residual XYZ force in N')
        gravity = np.array([0., 0., self.gravity_m_s2])
        return self.scale*(f/self.mass_kg + gravity)-gravity


class HoverCalibrator:
    def __init__(self, mass_kg, gravity, mass_thrust, *, window_s=3., min_samples=50,
                 max_gap_s=.15, position_tolerance_m=.02, speed_limit_m_s=.03,
                 minimum_cos_tilt=.996194698, max_variation=.03,
                 scale_bounds=(.8, 1.2), output_bounds=(0., 65535.)):
        if not all(np.isfinite(v) and v > 0 for v in (mass_kg, gravity, mass_thrust, window_s, max_gap_s)):
            raise ValueError('Positive finite mass, gravity, mapping and timing required')
        self.mass, self.gravity, self.mass_thrust = mass_kg, gravity, mass_thrust
        self.window_s, self.min_samples, self.max_gap = window_s, min_samples, max_gap_s
        self.position_tolerance, self.speed_limit = position_tolerance_m, speed_limit_m_s
        self.minimum_cos, self.max_variation = minimum_cos_tilt, max_variation
        self.scale_bounds, self.output_bounds = scale_bounds, output_bounds
        self.samples = deque()
        self.raw = []
        self.last_stamp = None
        self.frozen = None

    def add(self, source_time_s, received_time_s, output, position, reference, cos_tilt, battery_voltage=None):
        if self.frozen is not None:
            raise RuntimeError('Frozen calibration cannot accept strike measurements')
        position, reference = np.asarray(position, float), np.asarray(reference, float)
        valid = (position.shape == reference.shape == (3,) and
                 np.isfinite([source_time_s, received_time_s, output, cos_tilt]).all() and
                 np.isfinite(position).all() and np.isfinite(reference).all())
        if not valid:
            self.samples.clear()
            raise ValueError('Invalid calibration telemetry')
        if self.last_stamp is not None and source_time_s <= self.last_stamp:
            self.samples.clear()
            raise ValueError('Duplicate or backward telemetry timestamps')
        self.last_stamp = source_time_s
        self.raw.append(dict(source_time_s=source_time_s, received_time_s=received_time_s,
                             pwm=float(output), battery_voltage=battery_voltage,
                             position_m=position.tolist(), cos_tilt=float(cos_tilt)))
        stable = (self.output_bounds[0] < output < self.output_bounds[1] and
                  self.minimum_cos <= cos_tilt <= 1.000001 and
                  np.linalg.norm(position-reference) <= self.position_tolerance)
        if self.samples:
            prev = self.samples[-1]
            dt = source_time_s-prev[0]
            stable &= (dt <= self.max_gap and 0 < received_time_s-prev[1] <= self.max_gap and
                       np.linalg.norm(position-prev[3])/dt <= self.speed_limit)
        if not stable:
            self.samples.clear()
            return
        self.samples.append((source_time_s, received_time_s, float(output*cos_tilt), position.copy(), float(output)))
        # Keep one point just before the window boundary for integration.
        while len(self.samples)>2 and self.samples[1][0] <= source_time_s-self.window_s:
            self.samples.popleft()

    def freeze(self, now):
        if self.frozen is not None:
            return self.frozen
        a=list(self.samples)
        if len(a)<self.min_samples or a[-1][0]-a[0][0] < self.window_s-1e-9:
            raise ValueError('Need a complete stable hover window')
        if not 0 <= now-a[-1][1] <= self.max_gap:
            raise ValueError('Hover thrust telemetry is stale')
        ts=np.array([s[0] for s in a]); values=np.array([s[2] for s in a])
        mean=float(np.trapezoid(values, ts)/(ts[-1]-ts[0]))
        variation=float(np.std(values)/mean)
        scale=mean/(self.mass_thrust*self.mass*self.gravity)
        if variation>self.max_variation or not self.scale_bounds[0] <= scale <= self.scale_bounds[1]:
            raise ValueError('Hover correction is unstable or outside configured bounds')
        self.frozen=FrozenCalibration(scale,len(a),float(ts[-1]-ts[0]),
                                      mean,variation,self.mass,self.gravity,self.mass_thrust)
        return self.frozen


def load_force_csv(path, mass_kg, gravity):
    path=Path(path)
    metadata=json.loads((path.parent/'plan.json').read_text())
    config=metadata['controller_export']
    if abs(config['mass_kg']-mass_kg)>1e-9 or abs(config['gravity_m_s2']-gravity)>1e-9:
        raise ValueError('CSV controller mass/gravity do not match configuration')
    if path.name != config['force_file']:
        raise ValueError('Use the raw controller force CSV in N, not acceleration or total thrust')
    with path.open(newline='') as stream:
        reader=csv.reader(stream)
        if next(reader) != ['time_s','until_s','fx_ff_n','fy_ff_n','fz_ff_n']:
            raise ValueError('Unexpected CSV units/header')
        rows=np.array(list(reader),dtype=float)
    if rows.ndim!=2 or rows.shape[1]!=5 or not len(rows) or not np.isfinite(rows).all():
        raise ValueError('Invalid force sequence')
    if (rows[0,0]!=0 or np.any(rows[:,1]<=rows[:,0]) or
        not np.allclose(rows[1:,0],rows[:-1,1],rtol=0,atol=1e-10) or
        abs(rows[-1,1]-metadata['cutoff_s'])>1e-10 or abs(metadata['policy_dt_s']-.05)>1e-10):
        raise ValueError('Sequence must be contiguous with the saved 20 Hz cutoff')
    for value in rows[:-1,1]:
        if abs(value/.05-round(value/.05))>1e-7:
            raise ValueError('Unexpected internal control boundary')
    return rows


def execute_frozen(rows, calibration, send, pump, *, clock=time.monotonic, maximum_lateness_s=.01):
    """Absolute deadlines; no interpolation, repeated tail or accumulated hold drift."""
    start=clock()
    log=[]
    for row in rows:
        deadline=start+row[0]
        while clock()<deadline:
            pump(min(.002, deadline-clock()))
        late=clock()-deadline
        if late>maximum_lateness_s:
            raise RuntimeError(f'Strike command deadline missed by {late:.4f} s')
        acceleration=calibration.acceleration(row[2:])
        sent=clock()
        send(acceleration)
        log.append(dict(planned_time_s=float(row[0]),actual_time_s=sent-start,
                        residual_force_n=row[2:].tolist(),acceleration_m_s2=acceleration.tolist()))
    while clock()<start+rows[-1,1]:
        pump(min(.002,start+rows[-1,1]-clock()))
    return log
