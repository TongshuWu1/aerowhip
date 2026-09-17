"""Read saved model-adaptation evidence without fitting or changing flight ghosts."""
from pathlib import Path
import numpy as np
from simulator.workflow import read_json
from .io import sha256_file

METRICS = [('drone_whip_rmse_m','Drone whip RMS'),('marker_whip_rmse_m','Cable markers whip RMS'),
           ('tip_whip_rmse_m','Tip whip RMS'),('tip_strike_error_m','Tip prediction error at strike')]


def discover_studies(root):
    return sorted(p.parent.parent for p in (Path(root)/'runs/adaptation').glob('*/validation/results.json'))


def load_study(root,folder):
    root,folder=Path(root).resolve(),Path(folder).resolve()
    results=read_json(folder/'validation/results.json');rows=results['heldout_rows']
    if not rows or len({r['take'] for r in rows})!=len(rows):raise ValueError('Expected distinct held-out flight rows.')
    means={}
    for metric,_ in METRICS:
        for prefix in ('baseline','adapted'):
            key=prefix+'_'+metric;values=np.array([r[key] for r in rows],float)
            if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid saved adaptation metric: '+key)
            means[key]=float(values.mean())
            if not np.isclose(means[key],results['equal_flight_mean'][key],atol=1e-10):
                raise ValueError('Saved aggregate does not match equal-flight means: '+key)
    models=[]
    for path in (root/'data/model_candidates').glob('*/model.json'):
        model=read_json(path)
        report=model.get('adaptation',{}).get('report')
        if report and Path(report).resolve().parent.parent==folder:models.append(path)
    return dict(folder=folder,rows=rows,means=means,results=results,models=models,
        protocol=read_json(folder/'protocol.json',{}),assessment=read_json(folder/'assessment.json',{}))


def comparison_arrays(study,take,mode='heldout'):
    if take not in {r['take'] for r in study['rows']}:raise ValueError('Unknown flight.')
    if mode not in ('heldout','all_five'):raise ValueError('Choose held-out or all-data comparison.')
    base=study['folder']/'validation'
    paths=[base/'M0_initialized'/take/'prediction.npz',
           base/(('leave_out_'+take) if mode=='heldout' else 'all_five')/take/'both/prediction.npz']
    arrays=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as data:arrays.append({k:data[k].copy() for k in data.files})
    first,second=arrays
    if not np.array_equal(first['time'],second['time']):raise ValueError('Model predictions use different clocks.')
    for key in ('measured_position','measured_sites'):
        if not np.array_equal(first[key],second[key],equal_nan=True):raise ValueError('Comparisons have different measurements.')
    return first,second


def policies_for_model(root,model_path):
    digest=sha256_file(model_path);rows=[]
    for path in sorted((Path(root)/'runs/ppo').glob('*/run.json'),reverse=True):
        run=read_json(path,{})
        amendment=run.get('model_amendment') or {}
        if amendment.get('source_sha256')==digest:
            rows.append((path.parent,run,read_json(path.parent/'status.json',{})))
    return rows


FLIGHT_METRICS=[('drone_rms_m','Drone prediction RMS'),('tip_rms_m','Tip prediction RMS'),
                ('target_at_strike_m','Measured tip → target at planned strike'),
                ('closest_target_m','Measured closest tip → target')]


def measured_flight_metrics(data):
    """Native measured samples against the prediction saved BEFORE this flight.

    Fit/held-out model predictions are never an input to the flight ledger.
    Missing measurements remain missing; no nearest-finite strike substitution.
    """
    from .adaptation_check import interpolate_positions
    from .hover_calibration import normalized_evaluation
    data=normalized_evaluation(data)
    t=np.asarray(data['time']);end=data['metadata']['whip_end_s'];whip=(t>=0)&(t<=end+1e-9)
    def rms(key):
        x=np.asarray(data[key])[whip];x=x[np.isfinite(x)]
        return float(np.sqrt(np.mean(x*x))) if len(x) else None
    distance=np.asarray(data['target_error'])[whip];distance=distance[np.isfinite(distance)]
    strike=data['metadata'].get('predicted_hit_time_s')
    strike_error=None
    if strike is not None and 0<=strike<=end:
        position=interpolate_positions(t,data['measured_cable'][:,-1],np.array([strike]))[0]
        if np.isfinite(position).all():strike_error=float(np.linalg.norm(position-data['target']))
    return dict(take=data['take'],evaluation_frame='hover_normalized_z',
        bias_z_m=data['height_calibration']['bias_z_m'],
        drone_rms_m=rms('drone_error'),tip_rms_m=rms('tip_error'),
        target_at_strike_m=strike_error,closest_target_m=float(distance.min()) if len(distance) else None,
        complete_whip=data['tracking_span'][0]<=0 and data['tracking_span'][1]>=end,
        drone_samples=int(np.isfinite(np.asarray(data['drone_error'])[whip]).sum()),
        tip_samples=int(np.isfinite(np.asarray(data['tip_error'])[whip]).sum()),
        planned_strike_s=strike,whip_end_s=end,source_hashes=data['hashes'])


def recorded_flight_progress(root):
    from .adaptation_check import discover_batches,flight_names,find_rehearsal,load_comparison
    root=Path(root);rows=[];errors=[]
    for batch in discover_batches(root):
        try:
            rehearsal=find_rehearsal(root,batch/'simulation_csv/fullstate_30hz.csv')
            model=read_json(rehearsal/'model.json')
            # M1 is the first fitted candidate; keep the saved source identity too.
            tag='M1' if model.get('adaptation',{}).get('round')=='current_adp0' else 'M0' if not model.get('adaptation',{}).get('training') else 'Adapted model'
            tag=model.get('provenance',{}).get('label',tag)
            flights=[]
            for take in flight_names(batch):
                try:flights.append(measured_flight_metrics(load_comparison(root,batch,take,rehearsal)))
                except (OSError,ValueError,KeyError) as error:errors.append(f'{batch.name}/{take}: {error}')
            means={}
            for key,_ in FLIGHT_METRICS:
                valid=[r[key] for r in flights if r['complete_whip'] and r[key] is not None]
                means[key]=float(np.mean(valid)) if valid else None
                means[key+'_flights']=len(valid)
            if flights:rows.append(dict(batch=str(batch),round=batch.name,model=tag,rehearsal=str(rehearsal),
                model_source=model['fullstate_execution']['source_job'],flights=flights,means=means,
                complete_flights=sum(r['complete_whip'] for r in flights),actual_flights=len(flights)))
        except (OSError,ValueError,KeyError) as error:errors.append(f'{batch.name}: {error}')
    if not any(r['model']=='M1' for r in rows):
        rows.append(dict(round='adp1',model='M1',actual_flights=0,complete_flights=0,flights=[],
            means={key:None for key,_ in FLIGHT_METRICS},status='Awaiting adp1 flights'))
    return dict(rows=rows,errors=errors,evaluation_frame='hover_normalized_z',source='hover-normalized recorded motion vs exact CSV-matched pre-flight forecast; post-hover batch calibration; no fitted/held-out substitutes')
