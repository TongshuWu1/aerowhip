"""Translate a completed native rehearsal without re-evaluating its policy.

This preserves motion under the current translation-invariant free-flight
model. It does not model floor contact, ground effect or camera visibility.
"""
from copy import deepcopy
from pathlib import Path
import csv
import json
import shutil
import numpy as np
from experimental_data.io import atomic_json,sha256_file
from .research_rehearsal import FIELDS


def translate_arrays(arrays,offset):
    shift=np.asarray(offset,float)
    if shift.shape!=(3,) or not np.isfinite(shift).all():raise ValueError('A finite XYZ translation is required.')
    result={k:v.copy() for k,v in arrays.items()}
    result['commands'][:,:3]+=shift
    for key in ('origin_positions_m','cable_positions_m','target_position_m'):result[key]+=shift
    return result


def translate_metadata(metadata,offset):
    m=deepcopy(metadata);d=np.asarray(offset,float)
    for key in ('initial_tracking_origin_m','initial_attachment_m','target_position_m'):
        m[key]=(np.asarray(m[key])+d).tolist()
    recovery=m['recovery']
    if recovery.get('schema') not in (None,'independent_vertical_recovery_v1','curved_moving_recovery_v1'):
        raise ValueError('Unsupported recovery metadata; translation refused.')
    for key in ('hover_position_m','brake_position_m','approach_position_m','position_min_m','position_max_m'):
        if key in recovery:recovery[key]=(np.asarray(recovery[key])+d).tolist()
    for key in ('turn_coefficients_normalized','approach_coefficients_normalized'):
        if recovery.get(key) is not None:
            c=np.asarray(recovery[key]);c[0]+=d;recovery[key]=c.tolist()
    for key in ('peak_height_m','minimum_height_m'):
        if key in recovery.get('turn_metrics',{}):recovery['turn_metrics'][key]+=float(d[2])
    return m


def translate_rehearsal(source,destination,offset=(0.,0.,-.3)):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if destination==source or destination.is_relative_to(source) or destination.exists():
        raise ValueError('Choose a new output folder outside the source rehearsal.')
    shift=np.asarray(offset,float)
    if shift.shape!=(3,) or not np.isfinite(shift).all():raise ValueError('A finite XYZ translation is required.')
    metadata=json.loads((source/'rehearsal.json').read_text(encoding='utf-8'))
    if metadata.get('schema')!='research_fullstate_30hz_v1':raise ValueError('Select a completed native 30 Hz rehearsal.')
    with np.load(source/'rehearsal.npz',allow_pickle=False) as stream:arrays={k:stream[k].copy() for k in stream.files}
    new=translate_arrays(arrays,shift);m=translate_metadata(metadata,shift)
    # Copy only the current native bundle contract, not old diagnostic reports.
    required=['rehearsal.json','rehearsal.npz','fullstate_30hz.csv','virtual_force_30hz.csv','model.json','task.json','ppo.json']
    inputs=[source/n for n in required]+[p for folder in ('assets','checkpoints') for p in (source/folder).rglob('*') if p.is_file()]
    hashes={p.relative_to(source).as_posix():sha256_file(p) for p in inputs}
    # Preserve string values of every non-position CSV field, including PVA
    # derivatives and timing. CSV and NPZ must agree before any transformation.
    with (source/'fullstate_30hz.csv').open(newline='',encoding='utf-8') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
    if fields!=FIELDS:raise ValueError('Unexpected FullState CSV schema.')
    original=np.array([[float(row[k]) for k in fields] for row in rows])
    np.testing.assert_array_equal(original,np.c_[arrays['command_time_s'],arrays['commands']])
    destination.mkdir(parents=True)
    for p in inputs:
        out=destination/p.relative_to(source);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
    for row in rows:
        for key,delta in zip(('px_m','py_m','pz_m'),shift):
            if delta:row[key]=repr(float(row[key])+float(delta))
    with (destination/'fullstate_30hz.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    np.savez_compressed(destination/'rehearsal.npz',**new)
    task=json.loads((source/'task.json').read_text(encoding='utf-8'))
    task['initial_root_position_m']=m['initial_attachment_m'];task['target_position_m']=m['target_position_m']
    atomic_json(destination/'task.json',task)
    model=json.loads((source/'model.json').read_text(encoding='utf-8'))
    for key,name in [('motion_residual','cable_residual.pt'),('fullstate_execution','drone_model.json')]:
        model[key]['checkpoint']=str(destination/'assets'/name)
    atomic_json(destination/'model.json',model)
    previous=metadata.get('translation',{})
    cumulative=np.asarray(previous.get('cumulative_offset_world_m',[0,0,0]))+shift
    m['translation']=dict(schema='rigid_rehearsal_translation_v1',offset_world_m=shift.tolist(),
        cumulative_offset_world_m=cumulative.tolist(),source_rehearsal=str(source),source_files=hashes,
        original_planning_origin_m=previous.get('original_planning_origin_m',metadata['initial_tracking_origin_m']),
        original_planning_target_m=previous.get('original_planning_target_m',metadata['target_position_m']),
        prediction_method='Rigid translation of saved prediction under translation-invariant free-flight model; policy not re-run',
        policy_weights_unchanged=True,velocity_acceleration_force_timing_unchanged=True,
        absolute_height_effects_modeled=False)
    m['height_bounds_m']={name:dict(minimum=float(z.min()),maximum=float(z.max())) for name,z in [
        ('commanded_tracking_origin',new['commands'][:,2]),('predicted_tracking_origin',new['origin_positions_m'][:,2]),
        ('predicted_cable',new['cable_positions_m'][...,2])]}
    m['csv_sha256']=sha256_file(destination/'fullstate_30hz.csv')
    atomic_json(destination/'rehearsal.json',m)
    (destination/'TRANSLATION.txt').write_text(
        'The complete command, start, target and predicted scene were translated together.\n'
        f'Incremental XYZ shift [m]: {shift.tolist()}\n'
        f'Use actual tracked-origin hover [m]: {m["initial_tracking_origin_m"]}\n'
        f'Use actual target [m]: {m["target_position_m"]}\n'
        'PPO weights, forces, velocity, acceleration, yaw and timing are unchanged.\n'
        'Prediction is translated using model symmetry, not a new measured flight.\n'
        'Re-running PPO at the lower coordinates is a different operation and may change the maneuver.\n',encoding='utf-8')
    assert all(sha256_file(source/name)==digest for name,digest in hashes.items())
    return m
