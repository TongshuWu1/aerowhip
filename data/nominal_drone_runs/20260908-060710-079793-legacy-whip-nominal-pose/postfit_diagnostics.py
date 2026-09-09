"""Additional sampled-command diagnostics for this immutable fit job."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.spatial.transform import Rotation
import torch
JOB=Path(__file__).resolve().parent
ROOT=JOB.parents[2]
sys.path.insert(0,str(ROOT))
from experimental_data.nominal_pose_fit import PreparedTrial,GAIN_NAMES
from experimental_data.io import sha256_file
from simulator.drone_pose_response import command_attitude

protocol=json.loads((JOB/'protocol.json').read_text())
results=json.loads((JOB/'results.json').read_text())
out={}
for label,item in results.items():
    g=np.array([item['parameters'][k] for k in GAIN_NAMES]);delay=item['parameters']['delay_s']
    kp=g[:2][[0,0,1]];kd=g[2:4][[0,0,1]];ff=g[4:][[0,0,1]]
    out[label]={}
    for name in protocol['takes']:
        trial=PreparedTrial(JOB/'inputs'/protocol['source_version']/name,review=protocol['reviews'][name])
        p,v,_=trial.linear(g,delay)
        cmd=np.array([trial.schedule.sample(t-delay)[0].numpy() for t in trial.time])
        a=kp*(cmd[:,:3]-p)+kd*(cmd[:,3:6]-v)+ff*cmd[:,6:9]+trial.b0+trial.basis@g
        specific=a+np.array([0,0,9.80665])
        desired=command_attitude(torch.as_tensor(a),torch.as_tensor(cmd[:,9]),9.80665).numpy()
        step=np.r_[0,np.degrees(Rotation.from_matrix(desired[:-1].transpose(0,2,1)@desired[1:]).magnitude())]
        tilt=np.degrees(np.arccos(np.clip(specific[:,2]/np.linalg.norm(specific,axis=1),-1,1)))
        phases={}
        for phase in ['csv_maneuver','post_maneuver_fullstate_hold']:
            m=trial.truth['execution_phase']==phase
            phases[phase]=dict(sample_count=int(m.sum()),command_az_min_m_s2=float(cmd[m,8].min()),
                modeled_origin_az_min_m_s2=float(a[m,2].min()),samples_specific_z_negative=int((specific[m,2]<0).sum()),
                effective_command_frame_tilt_max_deg=float(tilt[m].max()),effective_command_frame_step_max_deg=float(step[m].max()))
        out[label][name]=phases
        np.savez_compressed(JOB/label/(name+'_acceleration_diagnostic.npz'),time_s=trial.time,
            sampled_delayed_command=cmd,modeled_origin_acceleration_m_s2=a,
            effective_command_frame=desired,execution_phase=trial.truth['execution_phase'])
with (JOB/'acceleration_diagnostics.json').open('x') as f:
    json.dump(dict(source_sha256=sha256_file(__file__),scope='Sampled at measurement times; not event/substep extrema or measured physical thrust',
        interpretation='a_O is at a non-COM tracking origin. a_O+g is an approximate attitude construction, not identified motor thrust. A negative Z here does not establish measured vehicle inversion or negative motor thrust.',
        results=out),f,indent=2,allow_nan=False)
print(json.dumps(out,indent=2))
