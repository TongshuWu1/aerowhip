"""Read-only alignment of the six authorized whip1 recordings."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REFERENCE = ROOT/'runs/rehearsals/20260906-222405-804438/plan_001/fullstate_30hz.csv'
ref = np.genfromtxt(REFERENCE, delimiter=',', names=True)
reference_values = np.column_stack([ref[n] for n in ref.dtype.names[1:10]])
results = []
fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
for k in range(1, 4):
    cp = HERE/'raw'/f'experiment_whip1_{k:03}.csv'
    mp = HERE/'raw'/f'whip1_{k:03}.csv'
    c = np.genfromtxt(cp, delimiter=',', names=True)
    with mp.open(newline='') as stream:
        rows = list(csv.reader(stream))
    assert rows[5][6:9] == ['Position']*3 and rows[3][6:9] == ['cf_7']*3
    m = np.array([[float(x) if x else np.nan for x in row[:10]] for row in rows[7:]])
    t, pos, ct = m[:, 1], m[:, 6:9], c['time_s']
    logged = np.column_stack([c[n] for n in ['x','y','z']])
    def cost(offset, selection=slice(None)):
        query = t[selection]+offset
        predicted = np.column_stack([np.interp(query, ct, logged[:, j]) for j in range(3)])
        return np.nanmean(np.sum((predicted-pos[selection])**2, axis=1))
    offsets = np.arange(ct[0]-t[0], ct[-1]-t[-1], .01)
    best = offsets[np.argmin([cost(o) for o in offsets])]
    optimum = minimize_scalar(cost, bounds=(best-.02, best+.02), method='bounded')
    active = (c['cmd_valid']==1) & (c['cmd_age']<.1) & (np.nan_to_num(abs(c['cmd_ax']))>.001)
    ix = np.flatnonzero(active)
    first, stop = ix[0], ix[-1]+1
    cmd_names = ['cmd_'+n for n in ['x','y','z','vx','vy','vz','ax','ay','az']]
    values = np.column_stack([c[n][active] for n in cmd_names])
    unique = values[np.r_[True,np.any(np.diff(values,axis=0)!=0,axis=1)]]
    changed = np.r_[False, np.any(np.diff(logged,axis=0)!=0,axis=1)]
    updates = np.flatnonzero(changed & (ct>10) & (ct<20))
    # Command age is treated as reception-age, not a source/sensor timestamp.
    start_received = ct[first]-c['cmd_age'][first]
    stop_received = ct[stop]-c['cmd_age'][stop]
    halves = [minimize_scalar(lambda o: cost(o,s), bounds=(optimum.x-.15,optimum.x+.15),
                            method='bounded').x for s in [slice(0,len(t)//2),slice(len(t)//2,None)]]
    item = dict(trial=f'{k:03}', controller_rows=len(c), optitrack_rows=len(m),
        optitrack_time_range_s=[float(t[0]),float(t[-1])],
        optitrack_missing_rigid_body_rows=int((~np.isfinite(pos).all(axis=1)).sum()),
        offset_controller_equals_optitrack_plus_s=float(optimum.x),
        aligned_position_3d_rms_m=float(np.sqrt(optimum.fun)), half_window_offsets_s=halves,
        controller_logger_hz=float(1/np.median(np.diff(ct))),
        logged_position_update_hz=float(1/np.median(np.diff(ct[updates]))),
        reference_unique_commands=len(unique),
        exact_reference_prefix_max_difference=float(abs(unique-reference_values[:len(unique)]).max()),
        observed_reference_duration_s=float(stop_received-start_received),
        observed_start_controller_s=float(start_received), observed_stop_controller_s=float(stop_received),
        missing_reference_times_s=ref['time_s'][len(unique):].tolist(),
        post_sequence_command_position_m=[float(c['cmd_'+n][stop]) for n in ['x','y','z']],
        peak_optitrack_z_m=float(np.nanmax(pos[:,2])),
        note='Offsets match measured positions without rotation, translation or time warping; include logging latency. Not clock synchronization or tracking-error estimates.')
    results.append(item)
    for j,n in enumerate(['x','y','z']):
        ax=axes[k-1,j]
        ax.plot(t+optimum.x-start_received,pos[:,j],label='OptiTrack (aligned)',color='#2563eb')
        ax.step(ct-start_received,logged[:,j],where='post',label='Logged measured position',color='#d97706',alpha=.7)
        valid=(c['cmd_valid']==1)&(c['cmd_age']<.1)
        ax.step(ct[valid]-start_received,c['cmd_'+n][valid],where='post',label='Recorded desired position',color='#16a34a')
        ax.axvline(stop_received-start_received,color='#dc2626',ls=':',label='Switch to fixed position')
        ax.set_xlim(-.3,1.8);ax.set_ylabel(f'Trial {k:03}: {n} (m)');ax.grid(alpha=.2)
        if k==3: ax.set_xlabel('Seconds from first maneuver command')
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.975),ncol=2)
fig.suptitle('Whip1: measured-log alignment and recorded references',y=.995)
fig.tight_layout(rect=(0,0,1,.93));fig.savefig(HERE/'alignment.png',dpi=150,bbox_inches='tight')
manifest=[dict(file=f.name,size=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
          for f in sorted((HERE/'raw').glob('*.csv'))]
payload=dict(method='Read-only constant-offset measured-position alignment; no model adaptation',
    reference=str(REFERENCE),reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
    sources=manifest,results=results)
(HERE/'results.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
print(json.dumps(results,indent=2))
