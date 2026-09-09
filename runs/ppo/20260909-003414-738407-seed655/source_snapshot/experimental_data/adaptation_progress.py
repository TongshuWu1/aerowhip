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
