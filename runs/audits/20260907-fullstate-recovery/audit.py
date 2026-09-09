"""Read-only audit of the user-supplied reference, not measured flight state."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[3]
output = Path(__file__).resolve().parent
source = Path('C:/Users/wts28/Downloads/fullstate_30hz.csv')
digest = hashlib.sha256(source.read_bytes()).hexdigest()
assert digest == '0f33bc1522658ba961e9ec905ac3f9f5e7bc9da0201732120497c428dad267ef'
with source.open(newline='') as stream:
    rows = list(csv.DictReader(stream))
t = np.array([float(r['time_s']) for r in rows])
a = np.array([[float(r[k]) for k in ('ax_m_s2','ay_m_s2','az_m_s2')] for r in rows])
p = np.array([[float(r[k]) for k in ('px_m','py_m','pz_m')] for r in rows])
phase = np.array([r['phase'] for r in rows])
specific = a + [0, 0, 9.80665]
direction = specific/np.linalg.norm(specific, axis=1)[:, None]
jumps = np.rad2deg(np.arccos(np.clip((direction[1:]*direction[:-1]).sum(1), -1, 1)))
index = int(jumps.argmax())
tilt = np.rad2deg(np.arctan2(np.linalg.norm(specific[:, :2], axis=1), specific[:, 2]))
signed_tilt = np.rad2deg(np.arctan2(specific[:, 0], specific[:, 2]))
groups = {}
for name in np.unique(phase):
    use = phase == name
    groups[name] = dict(min_az_m_s2=float(a[use, 2].min()), max_tilt_surrogate_deg=float(tilt[use].max()),
                       nonpositive_vertical_specific_force_rows=int((specific[use, 2] <= 0).sum()))
with np.load(root/'runs/rehearsals/20260907-221528-727626/flight.npz') as flight:
    force = flight['commanded_force_world_n']
    interval_end = flight['time_s'][1:] - 30.63
    last = np.flatnonzero(np.isclose(interval_end, .8, atol=1e-9))[0]
    source_jump = float(np.rad2deg(np.arccos(np.clip(
        np.dot(force[last], force[last+1])/np.linalg.norm(force[last])/np.linalg.norm(force[last+1]), -1, 1))))
    force_tilt = np.rad2deg(np.arctan2(force[:, 0], force[:, 2]))
result = dict(source=str(source), csv_sha256=digest, rows=len(t), by_phase=groups,
    reference_direction_jump_deg=float(jumps[index]), transition_time_s=t[index:index+2].tolist(),
    reference_direction_average_rate_deg_s=float(jumps[index]/(t[index+1]-t[index])),
    next_specific_force_projected_on_previous_reference_axis_m_s2=float(specific[index+1]@direction[index]),
    source_simulation_force_jump_deg=source_jump,
    source_force_before_n=force[last].tolist(), source_force_after_n=force[last+1].tolist(),
    assumptions='World Z up, a+[0,0,g] feedforward only; no cable load, feedback or actual attitude included.',
    interpretation='Strong transition-feasibility concern; actual failure mechanism not established from reference alone.',
    controller_source='https://github.com/bitcraze/crazyflie-firmware/blob/master/src/modules/src/controller/controller_mellinger.c',
    installed_firmware_verified=False, vehicle_limits_available=False, flight_files_changed=False)
(output/'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
use = (t >= .6) & (t <= 1.15)
axes[0].plot(t[use], a[use,0], '.-', label='CSV ax')
axes[0].plot(t[use], a[use,2], '.-', label='CSV az')
axes[0].set(ylabel='Acceleration (m/s²)', title='Whip → recovery: abrupt reversal in the supplied reference')
axes[0].legend()
axes[1].plot(t[use], signed_tilt[use], '.-', label='CSV feedforward direction (XZ)')
source_use = (interval_end >= .6) & (interval_end <= 1.15)
axes[1].step(interval_end[source_use], force_tilt[source_use], where='pre', label='Source simulator force direction (XZ)')
axes[1].set(ylabel='Signed angle from +Z (deg)', xlabel='CSV time (s)')
axes[1].legend()
for ax in axes[:2]:
    ax.axvline(.8, color='black', linestyle='--', alpha=.5)
    ax.grid(alpha=.25)
axes[2].plot(t, p[:,2], label='Commanded height, not measured flight')
axes[2].axvline(.8, color='black', linestyle='--', alpha=.5)
axes[2].set(xlabel='CSV time (s)', ylabel='Reference height (m)')
axes[2].grid(alpha=.25)
axes[2].legend()
fig.savefig(output/'recovery_diagnostic.png', dpi=150)
print(json.dumps(result, indent=2))
