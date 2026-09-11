"""Explicit development-study audit. Frozen inference only; never fit or plan.

Run once per output directory. Original reports and prediction ghosts are inputs,
never destinations. The GUI only reads the resulting checksummed review.
"""
from pathlib import Path
import argparse
import sys
import time
import platform
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.workflow import read_json
from experimental_data.io import atomic_json, sha256_file
from experimental_data.model_evaluation import model_identity, metric_arrays
from experimental_data.whip_adaptation import verify_hashes, rms_summary
from experimental_data.whip_adaptation_fit import records, evaluate
from experimental_data.adaptation_check import load_comparison, flight_names
from experimental_data.flight_performance import local_velocity, encounter
from experimental_data.adaptation_rounds import read_optitrack, read_controller
from experimental_data.preliminary_prepare import recorded_packets
from experimental_data.current_adaptation import causal_history_indices
from simulator.geometry import attachment_positions, normalized_rotations_xyzw
from simulator.drone_pose_response import CommandSchedule
from simulator.research_execution import ResearchExecutionModel

IDS = ['M0', 'M1-full', 'M2-frozen-refit-v1']
JOBS = ['M1-full-whip-v2', 'M2-frozen-refit-v1']
METRICS = ['drone', 'conditional_cable_tip', 'command_driven_tip', 'command_driven_markers']


def mean(values):
    a = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(a)) if a else None


def prepare_latest(out, flight):
    """Same reviewed geometry masks/history contract as the preceding full fits."""
    batch = Path(flight['batch']); p = read_json(batch/'protocol.json')
    intake = read_json(ROOT/'runs/data_review/M2-whip-intake-20260910/intake.json')
    model = read_json(Path(flight['rehearsal'])/'model.json')
    prepared = {}; readiness = {}
    for name in flight_names(batch):
        d = load_comparison(ROOT, batch, name); end = d['metadata']['whip_end_s']
        m = read_optitrack(batch/'flight_take'/f'{name}.csv')
        c = read_controller(batch/'flight_take'/f'experiment_{name}.csv')
        onset = intake[name]['onset_s']
        t = m['time'] + intake[name]['alignment']['offset_s'] - onset
        rotation, rv = normalized_rotations_xyzw(m['quaternion'])
        anchor, av = attachment_positions(m['drone'], m['quaternion'], model['recorded_data']['optitrack_to_attachment_offset_body_m'])
        sites = np.concatenate([anchor[:, None], m['cable']], 1)
        pv = np.isfinite(m['drone']).all(1) & rv & av
        mv = np.isfinite(m['cable']).all(-1)
        pj = np.linalg.norm(np.diff(m['drone'], axis=0), axis=-1) > .1
        mj = np.linalg.norm(np.diff(m['cable'], axis=0), axis=-1) > .15
        pv[:-1] &= ~pj; pv[1:] &= ~pj
        mv[:-1] &= ~mj; mv[1:] &= ~mj
        long = np.linalg.norm(np.diff(sites, axis=1), axis=-1) > np.asarray(model['cable']['marker_interval_lengths_m']) + .015
        mv &= ~long; mv[:, :-1] &= ~long[:, 1:]
        pre = causal_history_indices(t, 0., .4); cpre = causal_history_indices(t, 0., 1.)
        if not (pv[cpre].all() and mv[cpre].all()):
            readiness[name] = dict(usable=False, reason='Strict causal history has missing/rejected observations')
            continue
        pt, packets, until, coverage_end = recorded_packets(c, invalid_rows_break_coverage=True)
        pt -= onset; until -= onset; coverage_end -= onset
        schedule = CommandSchedule(pt, torch.tensor(packets[None], dtype=torch.float64), coverage_end_s=coverage_end, valid_until_s=until)
        hold = np.stack([schedule.sample(x)[0].numpy() for x in t[pre]])
        if not np.allclose(hold, hold[:1], atol=1e-8, rtol=0) or np.max(abs(hold[:, 3:9])) > 1e-8:
            raise ValueError('Nonstationary preflight command: ' + name)
        for x in np.r_[t[pre], t[(t >= 0) & (t <= end)]]:
            schedule.sample(x); schedule.sample(x-.12)
        folder = out/'inputs'/name; folder.mkdir(parents=True)
        np.savez_compressed(folder/'data.npz', time=t, position=m['drone'], quaternion=m['quaternion'], rotation=rotation,
            sites=sites, pose_valid=pv, marker_valid=mv, pre_indices=pre, hover_commands=hold,
            packet_time=pt, packets=packets, packet_valid_until=until, command_coverage_end=coverage_end)
        prepared[name] = dict(role=p['planned_roles'][name], end_s=end)
        readiness[name] = dict(usable=True, last_history_time_s=float(t[cpre[-1]]))
    p.update(takes=prepared, evidence='Diagnostic inputs only; all three models frozen before M2 flight. No fitting or role reassignment.')
    atomic_json(out/'protocol.json', p); atomic_json(out/'readiness.json', readiness)
    return out


def flight_outcomes(flight):
    """Recompute identical geometric measurements for all sessions, without simulation."""
    batch = Path(flight['batch']); protocol = read_json(batch/'protocol.json')
    roles = protocol['planned_roles']
    if flight['model'] == 'M1-full':
        # M1's original pre-collection protocol left roles empty. The reviewed
        # full M2 preparation is the authoritative later assignment.
        reviewed = read_json(ROOT/'runs/adaptation/M2-frozen-refit-v1/protocol.json')
        roles = {n:r['role'] for n,r in reviewed['takes'].items()}
    original = read_json(Path(flight['comparison'])/'report.json')
    model = read_json(Path(flight['rehearsal'])/'model.json'); rows = {}
    for name in flight_names(batch):
        d = load_comparison(ROOT, batch, name); t = d['time']; q = d['measured_cable'].copy()
        valid = np.isfinite(q).all(-1)
        jump = np.linalg.norm(np.diff(q, axis=0), axis=-1) > .15
        valid[:-1] &= ~jump; valid[1:] &= ~jump
        long = np.linalg.norm(np.diff(q, axis=1), axis=-1) > np.asarray(model['cable']['marker_interval_lengths_m']) + .015
        valid[:, 1:] &= ~long; valid[:, :-1] &= ~long
        q[~valid] = np.nan
        end = d['metadata']['whip_end_s']; w = (t >= 0) & (t <= end+.5)
        direction = np.asarray(d['task']['desired_strike_direction_world'], float); direction /= np.linalg.norm(direction)
        events = {str(width): encounter(t[w], q[w, -1], d['target'], .05, local_velocity(t, q[:, -1], width)[w], direction) for width in (3, 5, 7, 9)}
        raw = original['takes'][name]
        rows[name] = dict(role=roles[name], drone=raw['drone'], tip=raw['tip'],
            encounter=events['5'], speed_sensitivity={key:e['nearest'].get('outward_speed_m_s') for key,e in events.items()},
            whip_end_s=end, outcome_end_s=end+.5, batch=str(batch), forecast_sha256=flight['forecast_sha256'])
    return rows


def main():
    args = argparse.ArgumentParser(); args.add_argument('--output', required=True)
    out = Path(args.parse_args().output).resolve(); out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter(); torch.set_num_threads(4)
    catalog = read_json(ROOT/'config/evaluation/campaign.json')
    models = {m['id']:m for m in catalog['models'] if m['id'] in IDS}
    flights = [next(f for f in catalog['flights'] if f['model'] == name) for name in IDS]
    protected = {}
    for m in models.values():
        verify_hashes(m['hashes']); protected.update(m['hashes'])
    for f in flights:
        for base in (Path(f['batch']), Path(f['rehearsal']), Path(f['comparison'])):
            protected.update({str(p):sha256_file(p) for p in base.rglob('*') if p.is_file()})
    for job in JOBS:
        for filename in ('prepared_hashes.json', 'source_hashes.json'):
            h = read_json(ROOT/'runs/adaptation'/job/filename); verify_hashes(h); protected.update(h)
    for p in (ROOT/'config/pva').glob('*.json'): protected[str(p)] = sha256_file(p)
    atomic_json(out/'protected_before.json', protected)
    result = dict(schema='system_comparison_v1', evidence='Development study; three flown models, two adaptation updates. No parameter selection in this audit.',
        models=[], flights={}, datasets={}, metrics=METRICS, device=torch.cuda.get_device_name(), os=platform.platform(),
        aggregation='Equal take means of Euclidean RMS, not pooled frames. No confidence interval from correlated frames.',
        source_hashes={}, report_document='docs/development/M0_M1_M2_SYSTEM_COMPARISON.md')
    engines = {}
    for i,name in enumerate(IDS):
        m = models[name]; path = Path(m['model']); value = read_json(path)
        signature, hashes = model_identity(path); result['source_hashes'].update(hashes)
        e = ResearchExecutionModel.from_mapping(value, root=path.parent, device='cuda'); engines[name] = (e, value)
        drone = read_json(value['fullstate_execution']['checkpoint'])
        result['models'].append(dict(label=f'M{i}', id=name, signature=signature, parent=m.get('parent'), model=str(path),
            cable={k:value['cable'][k] for k in ('EI_n_m2','Cb_n_m2_s','external_drag_s_inv')},
            cable_residual=bool(value.get('motion_residual',{}).get('enabled')), drone=drone,
            interpretation=['Preliminary initialization plus reviewed cable development', 'Full staged adaptation from M0 flight and preliminary replay', 'Full staged adaptation from M1 flight plus prior training replay'][i]))
        f=flights[i]; rows=flight_outcomes(f)
        result['flights'][name]=dict(takes=rows, command_sha256=f['command_sha256'], forecast_sha256=f['forecast_sha256'],
            rehearsal=f['rehearsal'], batch=f['batch'], mean_drone_rms_m=mean(r['drone']['rmse_m'] for r in rows.values()),
            mean_tip_rms_m=mean(r['tip']['rmse_m'] for r in rows.values()),
            observed_entries=sum(r['encounter']['observed_sphere_entry'] for r in rows.values()),
            mean_nearest_m=mean(r['encounter']['nearest']['distance_m'] for r in rows.values()),
            mean_speed_m_s=mean(r['encounter']['nearest'].get('outward_speed_m_s') for r in rows.values()))
    latest = prepare_latest(out/'M2_inputs', flights[2])
    jobs = [ROOT/'runs/adaptation'/JOBS[0], ROOT/'runs/adaptation'/JOBS[1], latest]
    for i,job in enumerate(jobs):
        label=f'M{i} flights'; p=read_json(job/'protocol.json'); names=list(p['takes'])
        e, value=engines['M0']; data=records(job, names, value, e, 'cuda')
        ds=dict(job=str(job), roles={n:p['takes'][n]['role'] for n in names}, models={},
            excluded=['whip_003: missing cable history; observed flight outcome still retained'] if i==0 else [],
            interpretation='Same recorded commands, causal observations, geometry, grid and masks; each model reconstructs its hidden hover response state. No observation resets after onset.')
        for name in IDS:
            print('Frozen evaluation:', label, name, flush=True)
            engine,value=engines[name]; folder=out/f'M{i}_diagnostic'/name
            evaluate(data, engine, value['cable']['external_drag_s_inv'], folder)
            ds['models'][name]={}
            for row in data:
                take=row['name']; path=folder/(take+'.npz'); end=p['takes'][take]['end_s']
                _,scored,errors=metric_arrays(path, list(engine.cable.marker_node_indices[1:]), end)
                scores={key:rms_summary(values[scored]) for key,values in errors.items()}
                # Ancestor training use, not merely the batch's next-round role.
                trained=(p['takes'][take]['role']=='adaptation' and IDS.index(name)>i)
                ds['models'][name][take]=dict(metrics=scores, end_s=end, array=str(path),
                    data_use='Training or training replay' if trained else 'Excluded from this model fitting',
                    prospective_for_model=IDS.index(name)<=i)
        result['datasets'][label]=ds
        atomic_json(out/'progress.json',dict(stage=label, status='evaluated',elapsed_s=time.perf_counter()-start))
    # Validate paired inputs rather than merely claiming comparable curves.
    for ds in result['datasets'].values():
        for take in ds['roles']:
            paths=[ds['models'][n][take]['array'] for n in IDS]
            with np.load(paths[0]) as z: baseline={k:z[k] for k in ('time_s','mask','measured_origin','measured_sites')}
            for path in paths[1:]:
                with np.load(path) as z:
                    for k,v in baseline.items():np.testing.assert_array_equal(v,z[k])
    verify_hashes(protected)
    # Bound UI report to original evidence, prepared arrays and all diagnostic results.
    # Current UI/planner preferences were protected during the audit, but changing
    # a future preference does not invalidate an immutable historical report.
    config_prefix=str(ROOT/'config/pva')
    result['source_hashes'].update({p:h for p,h in protected.items() if not p.startswith(config_prefix)})
    result['source_hashes'].update({str(p):sha256_file(p) for p in out.rglob('*') if p.is_file() and p.name not in ('report.json','progress.json')})
    result['elapsed_s']=time.perf_counter()-start; result['protected_files_verified']=len(protected)
    atomic_json(out/'report.json', result)
    print('Completed:', result['elapsed_s'], 'seconds;',len(protected),'originals unchanged',flush=True)


if __name__ == '__main__':main()
