"""Batch vertical hover normalization; distinct from clock synchronization.

This is a nuisance-bias correction, not proof of improved physical tracking.
Only observed stationary holds contribute. Raw files/commands stay unchanged.
"""
from pathlib import Path
import json
import numpy as np
from .adaptation_check import flight_names,find_rehearsal,command_onset,recorded_alignment,sha256
from .adaptation_rounds import read_optitrack,read_controller,COMMAND_COLUMNS
from .io import atomic_json


def summarize_hold(time, measured_z, commanded_z, eligible, start, end):
    mask=eligible&(time>=start)&(time<=end)
    ids=np.flatnonzero(mask)
    if len(ids)<10 or time[ids[-1]]-time[ids[0]]<.09:
        raise ValueError('Insufficient recorded stationary hover samples for vertical calibration.')
    error=measured_z[ids]-commanded_z[ids]
    span=float(time[ids[-1]]-time[ids[0]])
    result=dict(samples=len(ids),start_s=float(time[ids[0]]),end_s=float(time[ids[-1]]),
        covered_s=span,median_bias_m=float(np.median(error)),
        spread_p90_m=float(np.quantile(error,.95)-np.quantile(error,.05)),
        drift_m_s=float(np.polyfit(time[ids]-time[ids[0]],error,1)[0]),
        short_trim=span<1.)
    return result


def combine_holds(rows):
    """Equal pre/post weight per take, then median across takes."""
    if not rows:raise ValueError('No calibrated takes.')
    biases=[];warnings=[]
    for row in rows:
        pre,post=row['pre'],row['post']
        if abs(pre['median_bias_m']-post['median_bias_m'])>.02:
            warnings.append(row['take']+': pre/post biases differ by over 2 cm')
        for name,hold in [('pre',pre),('post',post)]:
            if hold['spread_p90_m']>.02 or abs(hold['drift_m_s'])>.01:
                warnings.append(row['take']+': '+name+' hover is not sufficiently stationary')
        biases.append((pre['median_bias_m']+post['median_bias_m'])/2)
    bias=float(np.median(biases))
    if np.max(np.abs(np.asarray(biases)-bias))>.02:
        warnings.append('Between-take biases differ from batch median by over 2 cm')
    return dict(bias_z_m=bias,per_take_bias_m=biases,checks_passed=not warnings,warnings=warnings)


def calibrate_batch(root,batch):
    root,batch=Path(root),Path(batch)
    csv_path=batch/'simulation_csv/fullstate_30hz.csv'
    rehearsal=find_rehearsal(root,csv_path)
    ref=np.genfromtxt(csv_path,delimiter=',',names=True)
    values=np.column_stack([ref[n] for n in ref.dtype.names[1:]])
    # Locate the final stationary commanded hold, not the moving recovery.
    static=(np.linalg.norm(values[:,3:9],axis=1)<1e-8)&(np.max(np.abs(values[:,:3]-values[-1,:3]),axis=1)<1e-8)
    first=len(static)-1
    while first>0 and static[first-1]:first-=1
    if not static[-1] or first==len(static)-1:raise ValueError('No final stationary CSV hold.')
    post_start=float(ref['time_s'][first])+1.0
    post_end=float(ref['time_s'][-1])-.2
    hashes={str(csv_path.resolve()):sha256(csv_path)};rows=[];labels=set()
    for name in flight_names(batch):
        mp=batch/'flight_take'/f'{name}.csv';cp=batch/'flight_take'/f'experiment_{name}.csv'
        m=read_optitrack(mp);c=read_controller(cp,commands_only=True);labels.add(m['drone_label'])
        onset,_,_=command_onset(c,ref);alignment=recorded_alignment(root,batch,name,mp,cp)
        t=m['time']+alignment['offset_s']-onset
        idx=np.searchsorted(c['time_s'],t+onset,side='right')-1
        safe=np.clip(idx,0,len(c)-1);commands=np.column_stack([c[n][safe] for n in COMMAND_COLUMNS])
        age=t+onset-c['time_s'][safe]+c['cmd_age'][safe]
        valid=(idx>=0)&(t+onset<=c['time_s'][-1])&(c['cmd_valid'][safe]>.5)&(age>=0)&(age<=.1)
        valid &= np.isfinite(m['drone']).all(1)&np.isfinite(commands).all(1)&(np.linalg.norm(commands[:,3:9],axis=1)<1e-8)
        pre_valid=valid&(np.max(np.abs(commands[:,:3]-values[0,:3]),axis=1)<1e-8)
        post_valid=valid&(np.max(np.abs(commands[:,:3]-values[-1,:3]),axis=1)<1e-8)
        # Discard first two seconds of nominal 10 s prehold and first second of posthold.
        pre=summarize_hold(t,m['drone'][:,2],commands[:,2],pre_valid,-8.,-.02)
        post=summarize_hold(t,m['drone'][:,2],commands[:,2],post_valid,post_start,post_end)
        rows.append(dict(take=name,drone=m['drone_label'],pre=pre,post=post,clock_alignment=alignment))
        for p in [mp,cp,mp.with_suffix('.tracking.json'),batch/'time_alignment.json']:
            if p.exists():hashes[str(p.resolve())]=sha256(p)
    if len(labels)!=1:raise ValueError('A calibration batch must contain a single drone identity; split cf7 and cf3.')
    result=dict(schema='batch_hover_z_calibration_v1',**combine_holds(rows),takes=rows,source_hashes=hashes,
        method='median across takes of equal-weight pre/post median measured-minus-commanded Z',
        operation='corrected_z = raw_z - bias_z_m; same translation for drone and all cable markers',
        purpose='Retrospective hover-normalized fitting; raw real-flight target error is unchanged',
        post_hover_uses_future_data=True,source_drone=next(iter(labels)),
        thresholds=dict(pre_post_difference_m=.02,batch_deviation_m=.02,hover_spread_p90_m=.02,hover_drift_m_s=.01),
        short_hover_warning=any(r[h]['short_trim'] for r in rows for h in ['pre','post']))
    if any(sha256(p)!=digest for p,digest in hashes.items()):raise ValueError('Input changed during calibration.')
    atomic_json(batch/'height_calibration.json',result)
    return result


def load_calibration(batch):
    path=Path(batch)/'height_calibration.json'
    if not path.exists():return None
    result=json.loads(path.read_text())
    if result.get('schema')!='batch_hover_z_calibration_v1':raise ValueError('Unknown vertical calibration schema.')
    if not result['checks_passed']:raise ValueError('Batch height calibration needs review: '+'; '.join(result['warnings']))
    expected={r['take'] for r in result['takes']}
    if expected!=set(flight_names(batch)):raise ValueError('Batch membership changed; recalibrate vertical bias.')
    if not np.isfinite(result['bias_z_m']):raise ValueError('Invalid vertical bias.')
    for p,digest in result['source_hashes'].items():
        if not Path(p).is_file() or sha256(p)!=digest:raise ValueError('Vertical calibration source changed; recalibrate this batch.')
    return result


def corrected_tracking(measured,bias):
    corrected=dict(measured)
    for key in ['drone','cable']:
        corrected[key]=np.array(measured[key],copy=True)
        corrected[key][...,2]-=bias
    return corrected


def normalized_evaluation(data):
    """One authoritative evaluation view; no raw fallback or double shift."""
    if 'hover_normalized' not in data or 'height_calibration' not in data:
        raise ValueError('Valid batch height calibration required for normalized evaluation. '+data.get('height_calibration_error',''))
    calibration=data['height_calibration']
    if not calibration.get('checks_passed',False) or not np.isfinite(calibration['bias_z_m']):
        raise ValueError('Valid batch height calibration required for normalized evaluation.')
    return dict(data,**data['hover_normalized'],evaluation_frame='hover_normalized_z')
