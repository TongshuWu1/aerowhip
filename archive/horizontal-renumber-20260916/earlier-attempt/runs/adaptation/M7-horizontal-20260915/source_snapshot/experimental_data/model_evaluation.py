"""Explicit model lineage and comparable, immutable sim-real evaluation evidence.

Reading this catalog never runs physics, fits a model, or changes flight selection.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import numpy as np
from .io import atomic_json, sha256_file
from .whip_adaptation import verify_hashes, rms_summary, SCHEMA
from simulator.workflow import read_json

CATALOG = 'model_evolution_v1'
REPORT = 'same_flight_evaluation_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def model_identity(path):
    """Ignore packaging paths/provenance, bind all referenced component bytes."""
    path = Path(path).resolve()
    hashes = {str(path): sha256_file(path)}

    def walk(value, base):
        if isinstance(value, list):
            return [walk(v, base) for v in value]
        if not isinstance(value, dict):
            return value
        result = {}
        for key, item in value.items():
            if key == 'provenance':
                continue
            if key == 'checkpoint' and isinstance(item, str) and item:
                asset = Path(item)
                asset = (asset if asset.is_absolute() else base / asset).resolve()
                hashes[str(asset)] = sha256_file(asset)
                result[key] = walk(read_json(asset), asset.parent) if asset.suffix == '.json' else hashes[str(asset)]
            elif key == 'sha256':
                continue  # Component content is bound above, independent of packaging.
            else:
                result[key] = walk(item, base)
        return result

    identity = digest(walk(read_json(path), path.parent))
    return identity, hashes


def catalog_path(root):
    return Path(root) / 'config/evaluation/campaign.json'


def comparison_identity(path,response_update=None):
    """Bind fixed physical fields and nested assets, excluding only reviewed updates."""
    if response_update is not None:
        from .response_update_contract import validate
        validate(response_update)
    def walk(value,base,route=()):
        if isinstance(value,list):return [walk(v,base,route) for v in value]
        if not isinstance(value,dict):return value
        result={}
        for key,item in value.items():
            if key in ('provenance','sha256'):continue
            if route==('fullstate_execution','checkpoint','nominal') and key=='training_takes':continue
            if response_update is None and route==('cable',) and key=='external_drag_s_inv':continue
            if response_update is not None and route==('fullstate_execution','checkpoint','nominal','parameters') and key=='feedforward_xy':continue
            if key=='checkpoint' and isinstance(item,str) and item:
                asset=Path(item);asset=asset if asset.is_absolute() else base/asset
                result[key]=walk(read_json(asset),asset.parent,route+(key,)) if asset.suffix=='.json' else sha256_file(asset)
            else:result[key]=walk(item,base,route+(key,))
        return result
    path=Path(path);return digest(walk(read_json(path),path.parent))


def load_catalog(root):
    path = catalog_path(root)
    if not path.exists():
        return dict(schema=CATALOG, models=[], flights=[])
    value = read_json(path)
    if value.get('schema') != CATALOG:
        raise ValueError('Unsupported evaluation catalog')
    return value


def save_catalog(root, value):
    # Preserve every prior catalog revision; registration is not model selection.
    path = catalog_path(root)
    if path.exists():
        history = path.parent / 'history' / (sha256_file(path) + '.json')
        history.parent.mkdir(parents=True, exist_ok=True)
        if not history.exists():
            history.write_bytes(path.read_bytes())
    atomic_json(path, value)


def seed_catalog(root):
    if catalog_path(root).exists():
        raise ValueError('A campaign already exists; it will not be overwritten')
    selection = read_json(Path(root) / 'config/pva/flight_selection.json')
    rehearsal = Path(selection['rehearsal'])
    model = rehearsal / 'model.json'
    signature, hashes = model_identity(model)
    csv = Path(selection['command_csv']); ghost = rehearsal / 'rehearsal.npz'
    verify_hashes({str(csv): selection['command_sha256'], str(ghost): selection['forecast_sha256']})
    catalog = dict(schema=CATALOG, experiment=read_json(Path(root) / 'config/experiment.json')['id'],
        models=[dict(id='M0', parent=None, model=str(model), signature=signature, hashes=hashes,
            job=None, training_sources=[], status='Development baseline', created_at=datetime.now(timezone.utc).isoformat())],
        flights=[dict(model='M0', rehearsal=str(rehearsal), command_csv=str(csv),
            command_sha256=selection['command_sha256'], forecast_sha256=selection['forecast_sha256'],
            comparison=None, status='Awaiting recorded flight')])
    save_catalog(root, catalog)
    return catalog


def add_candidate(root, job):
    job = Path(job).resolve()
    # Audits/synthetic fixtures are never discovered or registered as flight evidence.
    if job.parent != (Path(root) / 'runs/adaptation').resolve():
        raise ValueError('Register a real prepared job directly under runs/adaptation; audit fixtures are excluded')
    p = read_json(job / 'protocol.json'); result = read_json(job / 'fit/result.json')
    if p.get('schema') != SCHEMA or result.get('status') != 'completed':
        raise ValueError('A completed reviewed whip fit is required')
    candidate_hashes=result.get('candidate_hashes',{})
    if str(job/'candidate/model.json') not in candidate_hashes:
        raise ValueError('Fit report must bind the completed candidate and diagnostic artifacts')
    verify_hashes(candidate_hashes)
    verify_hashes(read_json(job / 'prepared_hashes.json'))
    verify_hashes(read_json(job / 'source_hashes.json'))
    catalog = load_catalog(root); generation = int(p.get('parent_generation', 0)) + 1
    parent = next((m for m in catalog['models'] if m['id'] == f'M{generation-1}'), None)
    if parent is None:
        raise ValueError('Register the parent model first')
    verify_hashes(parent['hashes'])
    parent_identity, _ = model_identity(job / 'source_candidate/model.json')
    if parent_identity != parent['signature']:
        raise ValueError('Prepared parent does not match the registered model')
    name = f'M{generation}'
    if any(m['id'] == name for m in catalog['models']):
        raise ValueError(name + ' already exists; prior model versions cannot be replaced')
    model = job / 'candidate/model.json'; value = read_json(model)
    provenance = value.get('provenance', {})
    if provenance.get('parent_model_sha256') != sha256_file(job / 'source_candidate/model.json'):
        raise ValueError('Candidate parent provenance mismatch')
    if provenance.get('generation_index', generation) != generation:
        raise ValueError('Candidate generation mismatch')
    signature, hashes = model_identity(model)
    hashes.update(candidate_hashes)
    hashes.update({str(job / n): sha256_file(job / n) for n in ('fit/result.json', 'protocol.json')})
    sources = set(parent['training_sources'])
    original = read_json(job / 'source_hashes.json')
    for take in result['training_takes']:
        pair={Path(path).name:h for path,h in original.items() if Path(path).name in (take+'.csv', 'experiment_'+take+'.csv')}
        if len(pair)!=2:raise ValueError('Training take lacks both source identities: '+take)
        # Repeated executions can have byte-identical command logs. Training
        # ancestry concerns measured observations, not reuse of a command file.
        sources.add(pair[take+'.csv'])
    catalog['models'].append(dict(id=name, parent=parent['id'], model=str(model), signature=signature,
        hashes=hashes, job=str(job), training_sources=sorted(sources), status='Candidate · prospective flight pending',
        created_at=datetime.now(timezone.utc).isoformat()))
    save_catalog(root, catalog)
    return name


def add_flight_report(root, comparison):
    folder = Path(comparison).resolve(); report = read_json(folder / 'report.json')
    if report.get('schema') != SCHEMA:
        raise ValueError('Choose an original frozen-forecast comparison report')
    from .whip_adaptation import protocol
    p = protocol(report['batch']); hashes = read_json(folder / 'source_hashes.json'); verify_hashes(hashes)
    if 'audits' in Path(report['batch']).relative_to(Path(root).resolve()).parts:
        raise ValueError('Synthetic/audit comparisons cannot become real flight evidence')
    catalog = load_catalog(root)
    signature, _ = model_identity(Path(p['rehearsal']) / 'model.json')
    matches = [m for m in catalog['models'] if m['signature'] == signature]
    if p.get('model_id'):
        matches = [m for m in matches if m['id'] == p['model_id']]
    if len(matches) != 1:
        raise ValueError('Register this forecast model before its flight report')
    model=matches[0];name=model['id']
    if any(f.get('comparison') == str(folder) for f in catalog['flights']):
        raise ValueError('This comparison is already registered')
    if any(f.get('batch') == report['batch'] for f in catalog['flights']):
        raise ValueError('This batch is already registered; do not count repeat analyses as new flights')
    row = dict(model=name, rehearsal=p['rehearsal'], batch=report['batch'],
        command_csv=str(Path(report['batch']) / 'simulation_csv/fullstate_30hz.csv'),
        command_sha256=p['command_sha256'], forecast_sha256=p['forecast_sha256'],
        comparison=str(folder), hashes={**hashes, str(folder/'report.json'):sha256_file(folder/'report.json')},
        status='Recorded · contact outcome needs review')
    pending = next((f for f in catalog['flights'] if not f.get('comparison') and f['forecast_sha256']==p['forecast_sha256']), None)
    if pending is not None:
        pending.update(row)
    else:
        catalog['flights'].append(row)
    save_catalog(root, catalog)


def metric_arrays(path, marker_ids, end_s):
    """One shared scored interval; NaNs remain gaps rather than good predictions."""
    with np.load(path) as z:
        a = {k:z[k].copy() for k in z.files}
    score = (a['time_s'] >= 0) & (a['time_s'] < end_s - 1e-9)
    mask = a['mask'] & score[:, None]
    errors = {'drone': np.linalg.norm(a['predicted_origin']-a['measured_origin'],axis=-1)}
    errors['drone'][~score] = np.nan
    for mode, key in [('command_driven','coupled_cable'), ('conditional_cable','conditional_cable')]:
        error = np.linalg.norm(a[key][:,marker_ids]-a['measured_sites'][:,1:],axis=-1)
        error[~mask] = np.nan
        errors[mode+'_tip'] = error[:,-1]
        errors[mode+'_markers'] = error
    return a, score, errors


def summarize_diagnostics(folder, protocol, marker_ids):
    result = {}
    for name, review in protocol['takes'].items():
        a, score, errors = metric_arrays(Path(folder)/(name+'.npz'),marker_ids,review['end_s'])
        result[name] = dict(role=review['role'], end_s=review['end_s'],
            metrics={key:rms_summary(v[score]) for key,v in errors.items()},
            horizons={str(h):{key:rms_summary(v[score & (a['time_s']<=h+1e-9)]) for key,v in errors.items()}
                for h in (.25,.5,1.) if review['end_s'] >= h})
    return result


def evaluate_models(root, job, model_ids, output, device='cuda'):
    """Explicit diagnostic replay. No planning/parameter update/promotion."""
    import torch
    from . import whip_adaptation_fit as fitting
    from simulator.research_execution import ResearchExecutionModel
    root = Path(root).resolve(); job = Path(job).resolve(); output = Path(output).resolve()
    catalog = load_catalog(root)
    selected = [m for m in catalog['models'] if m['id'] in model_ids]
    if len(selected) != len(set(model_ids)) or len(selected) < 2:
        raise ValueError('Choose at least two registered models')
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'status.json', dict(status='running'))
    try:
        baseline, p, engine = fitting.load(job, device)
        rows = fitting.records(job, list(p['takes']), baseline, engine, device)
        ids = list(engine.cable.marker_node_indices[1:])
        report = dict(schema=REPORT, job=str(job), protocol=p, models={}, device=device,
            experiment=catalog.get('experiment'), evidence='Reinitialized same-flight diagnostics; never an original prospective forecast',
            aggregation='Equal take weight; per-take values retained; no frame-level confidence intervals')
        # Exact same measured state, grid, masks, packets and boundary inputs for all models.
        contract = {str(job/n):sha256_file(job/n) for n in ('prepared_hashes.json','source_hashes.json','code_hashes.json')}
        contract.update(read_json(job/'prepared_hashes.json')); contract.update(read_json(job/'source_hashes.json'))
        report['comparison_key'] = digest(contract)
        hashes = dict(contract)
        scope=p.get('response_update')
        base_identity=comparison_identity(job/'source_candidate/model.json',scope)
        for index, model in enumerate(selected):
            if (output/'STOP').exists():
                raise InterruptedError('Stopped between models; completed diagnostics preserved')
            verify_hashes(model['hashes']); hashes.update(model['hashes'])
            if model_identity(model['model'])[0] != model['signature']:
                raise ValueError('Registered model identity changed')
            value = read_json(model['model'])
            candidate_engine = ResearchExecutionModel.from_mapping(value,root=Path(model['model']).parent,device=device)
            if comparison_identity(model['model'],scope)!=base_identity or candidate_engine.dt_s!=engine.dt_s:
                raise ValueError(model['id']+': model changes exceed reviewed comparison scope')
            label=f'Evaluating {model["id"]} · {index+1}/{len(selected)} models'
            atomic_json(output/'progress.json',dict(label=label,completed=index,total=len(selected)))
            print(label,flush=True)
            fitting.evaluate(rows,candidate_engine,value['cable']['external_drag_s_inv'],output/model['id'])
            metrics = summarize_diagnostics(output/model['id'],p,ids)
            source_hashes = read_json(job/'source_hashes.json')
            for name, take in metrics.items():
                raw = {h for path,h in source_hashes.items() if Path(path).name == name+'.csv'}
                take['data_use'] = 'In training ancestry' if raw & set(model['training_sources']) else (
                    'Held out of this update' if take['role']=='validation' else 'Development comparison')
            report['models'][model['id']] = dict(model=model, takes=metrics, marker_ids=ids)
        verify_hashes(hashes)
        for path in output.glob('*/*.npz'):
            hashes[str(path)] = sha256_file(path)
        atomic_json(output/'report.json',report)
        hashes[str(output/'report.json')] = sha256_file(output/'report.json')
        atomic_json(output/'evidence_hashes.json',hashes)
        atomic_json(output/'status.json',dict(status='completed',models=list(report['models'])))
        atomic_json(output/'progress.json',dict(label='Comparison complete',completed=len(selected),total=len(selected)))
        return report
    except BaseException as exc:
        atomic_json(output/'status.json',dict(status='stopped' if isinstance(exc,InterruptedError) else 'failed',error=str(exc)))
        raise


def load_evaluation(folder):
    folder = Path(folder)
    if read_json(folder/'status.json').get('status') != 'completed':
        raise ValueError('Evaluation is incomplete; inspect the preserved log')
    verify_hashes(read_json(folder/'evidence_hashes.json'))
    result = read_json(folder/'report.json')
    if result.get('schema') != REPORT:
        raise ValueError('Unsupported comparison report')
    return result


def paired_summary(report, metric, role='validation'):
    models = list(report['models'])
    if not models:
        return [], []
    names = [name for name,row in report['models'][models[0]]['takes'].items() if role=='all' or row['role']==role]
    # Never silently compare different masks, durations, or surviving subsets.
    for name in names:
        reference = report['models'][models[0]]['takes'][name]
        for model in models[1:]:
            other = report['models'][model]['takes'][name]
            for field in ('valid_count','total_count'):
                if other['metrics'][metric][field] != reference['metrics'][metric][field]:
                    raise ValueError('Unequal observation coverage; paired aggregation blocked')
            if other['end_s'] != reference['end_s']:
                raise ValueError('Unequal reviewed intervals; paired aggregation blocked')
    values = [[report['models'][model]['takes'][n]['metrics'][metric]['rmse_m'] for n in names] for model in models]
    return names, values


def review_outcome(root, comparison, take, outcome, execution, reviewer, evidence):
    """Append a source-bound human review; never infer contact from a missing marker."""
    if outcome not in ('hit','miss','unknown') or execution not in ('feasible','infeasible','unknown'):
        raise ValueError('Unknown outcome or execution classification')
    if not reviewer.strip() or not evidence.strip():
        raise ValueError('Record the reviewer and supporting observation or uncertainty')
    catalog=load_catalog(root)
    flight=next((f for f in catalog['flights'] if f.get('comparison')==str(Path(comparison).resolve())),None)
    if flight is None:raise ValueError('Register this flight comparison first')
    verify_hashes(flight['hashes'])
    report_path=Path(comparison)/'report.json'
    if take not in read_json(report_path)['takes']:raise ValueError('Take is not in this flight comparison')
    review=dict(take=take,outcome=outcome,execution=execution,reviewer=reviewer.strip(),evidence=evidence.strip(),
        comparison_sha256=sha256_file(report_path),created_at=datetime.now(timezone.utc).isoformat(),
        interpretation='Reviewed tip-target task outcome, separate from modeled contact and physical target collision')
    folder=Path(root)/'runs/evaluation/outcomes';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(digest(review)+'.json');atomic_json(path,review)
    flight.setdefault('outcomes',{})[take]=dict(path=str(path),sha256=sha256_file(path))
    save_catalog(root,catalog)


def flight_outcomes(flight):
    result={}
    for take,reference in flight.get('outcomes',{}).items():
        verify_hashes({reference['path']:reference['sha256']})
        row=read_json(reference['path'])
        if row['comparison_sha256']!=sha256_file(Path(flight['comparison'])/'report.json'):
            raise ValueError('Outcome review belongs to another comparison')
        result[take]=row
    return result


def success_counts(rows):
    """Report unresolved attempts explicitly; no invented zero for pending flights."""
    hits=sum(r['outcome']=='hit' and r['execution']=='feasible' for r in rows)
    failures=sum(r['outcome']=='miss' or r['execution']=='infeasible' for r in rows)
    unknown=len(rows)-hits-failures
    return dict(attempts=len(rows),hits=hits,failures=failures,unknown=unknown,
        assessed_rate=hits/(hits+failures) if hits+failures else None)
