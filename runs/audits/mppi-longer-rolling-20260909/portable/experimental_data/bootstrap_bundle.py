"""Versioned bootstrap inputs for the corrected drone/cable model family."""
from pathlib import Path
import json,shutil
import numpy as np
from .io import atomic_json,sha256_file
from .historical_dataset import TAKES,FOLDS,load_source,quality_masks
from simulator.workflow import stamp

NOMINAL='20260908-062555-496491-legacy-whip-nominal-pose'
VERSION='20260908-041031-260837'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def prepare(root):
    root=Path(root).resolve();job=root/'data/bootstrap_model_runs'/stamp();job.mkdir(parents=True)
    source=root/'data/nominal_drone_runs'/NOMINAL;model=read(root/'config/model.json')
    protected=read(source/'protected-before.json')
    for name,digest in protected.items():assert sha256_file(name)==digest,name
    atomic_json(job/'protected-before.json',protected);atomic_json(job/'original_model.json',model)
    manifest=read(root/'data/dataset_manifest.json');atomic_json(job/'dataset_manifest.json',manifest)
    dest=job/'nominal';dest.mkdir();nominal_hashes={}
    for label in ['final_all_three']+['leave_out_whip1_00'+str(i) for i in (1,2,3)]:
        target=dest/(label+'.json');shutil.copy2(source/label/'nominal_model.json',target)
        nominal_hashes[label]=sha256_file(target)
    shutil.copytree(source/'inputs'/VERSION,job/'pose_inputs'/VERSION)
    shutil.copy2(source/'protocol.json',job/'nominal_source_protocol.json')
    inputs={};audit={};(job/'inputs').mkdir()
    for name in TAKES:
        if name.startswith('whip'):
            path=root/'data/adaptation_rounds/adaptation0/processed'/VERSION/name/'dataset.npz'
            with np.load(path) as d:
                a=dict(time=d['optitrack_time_s'],position=d['drone_position_m'],quaternion=d['drone_quaternion_xyzw'],
                    position_valid=d['drone_position_valid'],markers=d['cable_position_m'],marker_valid=d['cable_valid'],
                    commands=d['reference_fullstate'][:,:9],command_valid=d['reference_valid'],command_age=d['reference_age_s'])
                phases=read(path.parent/'execution_phases.json');ct=d['controller_time_s']
                # Keep past hover plus the complete observed whip for cable fit.
                scope=(ct>=phases['csv_start_s']-1.3)&(ct<phases['csv_end_s'])
                details=dict(source_version=VERSION,fit_scope='1.3s pre-hover plus observed CSV; no post-hold or landing loss')
                a['controller_time']=ct.copy()
        else:
            if not manifest['takes'][name]['enabled']:raise ValueError(f'Expected enabled bootstrap take: {name}')
            if manifest['takes'][name]['segments']:raise ValueError('Explicit preliminary manual segments need review before this bootstrap fit')
            path,a,details=load_source(root,name)
            with np.load(path) as d:scope=d['auto_frame_valid'].copy()
        a['time']=a['time']-a['time'][0]
        if not np.allclose(np.diff(a['time']),.01,atol=1e-7):raise ValueError('Expected 100Hz source times')
        cable,drone,reasons=quality_masks(a,model)
        cable&=scope;drone&=scope;reasons['outside_bootstrap_scope']=~scope
        a.update(cable_fit_valid=cable,drone_fit_valid=drone)
        target=job/'inputs'/(name+'.npz');np.savez_compressed(target,**a)
        maskfile=job/'inputs'/(name+'_masks.npz');np.savez_compressed(maskfile,time_s=a['time'],**reasons)
        inputs[name]=dict(source=str(path),source_sha256=sha256_file(path),snapshot_sha256=sha256_file(target),mask_sha256=sha256_file(maskfile))
        audit[name]=dict(frames=len(a['time']),kept_cable_frames=int(cable.sum()),exclusion_counts={k:int(v.sum()) for k,v in reasons.items()},**details)
    protocol=dict(schema='corrected_bootstrap_bundle_v1',takes=list(TAKES),folds=[list(x) for x in FOLDS],
        nominal_source=NOMINAL,nominal_hashes=nominal_hashes,provenance=inputs,
        geometry='Frozen active tracking-frame offset, span lengths and measured masses; no new geometry fit',
        cable_horizon_s=.65,cable_selection_horizon_s=.65,cable_windows_per_take=6,cable_training_windows_per_take=18,
        cable_updates=24,cable_hidden=32,cable_learning_rate=.0005,cable_regularization=.001,
        cable_mode='dissipative',cable_damping_limit_s_inv=2.,external_drag_s_inv=0.,
        cable_bias_grid=[-.5,0,.25,.5,1.,1.5],
        physical_EI_grid=[1e-9,1e-7,1e-5,2.8e-4],physical_Cb_grid=[1e-8,1e-6,1e-4,.002,.02],
        drone_response_takes=['whip1_001','whip1_002','whip1_003'],drone_updates=100,drone_hidden=16,
        drone_acceleration_limit_m_s2=.5,drone_regularization=.01,drone_learning_rate=.003,seed=1731,
        assessment='Whole observed whip recursive predictions; development folds, not independent evidence',
        activation='Separate candidate only; no PPO, no active model changes, no flight')
    atomic_json(job/'protocol.json',protocol);atomic_json(job/'audit.json',audit)
    atomic_json(job/'status.json',dict(status='PREPARED',active_model_changed=False,ppo_started=False))
    print(job,flush=True);return job

def snapshot_stage(job,stage,files):
    root=Path(__file__).resolve().parents[1];hashes={}
    for name in files:
        target=job/'source_snapshot'/stage/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/name,target);hashes[name]=sha256_file(target)
    atomic_json(job/(stage+'_source_hashes.json'),hashes)

def verify(job):
    protocol=read(job/'protocol.json')
    for path,digest in read(job/'protected-before.json').items():assert sha256_file(path)==digest,path
    for name,row in protocol['provenance'].items():
        assert sha256_file(row['source'])==row['source_sha256']
        assert sha256_file(job/'inputs'/(name+'.npz'))==row['snapshot_sha256']
    return dict(protected_files_verified=len(read(job/'protected-before.json')),originals_preserved=True)
