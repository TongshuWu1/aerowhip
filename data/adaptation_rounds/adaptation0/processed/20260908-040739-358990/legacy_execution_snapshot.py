"""Describe legacy CSV-only maneuvers without inventing unlogged commands.

The supplied flight script is parsed as data, never imported or executed.
Phase boundaries use first logger observations, not inferred vehicle actuation.
"""
import ast
import json
from pathlib import Path
import shutil

import numpy as np

from .io import sha256_file, utc_now
from simulator.workflow import stamp

UNKNOWN = 'unobserved_command'
CSV = 'csv_maneuver'
PRE = 'pre_maneuver_fullstate_hold'
POST = 'post_maneuver_fullstate_hold'
OTHER = 'other_fullstate'


def read_sequence(path):
    names = {'PVA_SEQUENCE', 'NOMINAL_SAMPLE_RATE', 'PRE_TRAJECTORY_HOVER_TIME', 'RECOVERY_TIME'}
    values = {}
    for node in ast.parse(Path(path).read_text(encoding='utf-8-sig')).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    values[target.id] = ast.literal_eval(node.value)
    if set(values) != names:
        raise ValueError('Legacy source must contain literal sequence and timing constants')
    rows = values.pop('PVA_SEQUENCE')
    times = np.array([r[0] for r in rows], dtype=float)
    commands = np.array([list(p)+list(v)+list(a)+[yaw, rate]
                         for _, p, v, a, yaw, rate in rows], dtype=float)
    if (len(times) < 2 or commands.shape != (len(times), 11)
            or not np.isfinite(commands).all() or not np.isfinite(times).all()
            or times[0] != 0 or np.any(np.diff(times) <= 0)):
        raise ValueError('Invalid literal PVA sequence')
    if not all(np.isfinite(v) and v > 0 for v in values.values()):
        raise ValueError('Invalid legacy timing constants')
    # Identical consecutive PVA samples cannot be counted from value changes.
    # Refuse an ambiguous classification instead of guessing packet counts.
    if len(np.unique(commands, axis=0)) != len(commands):
        raise ValueError('Repeated CSV values need packet-level review; cannot infer row identity')
    return times, commands, values


def create_profile(directory, controller, logger, geometry_model):
    """Archive reviewed sources and explicit geometry; never replace a profile."""
    directory = Path(directory)
    target = directory/'execution_profile.json'
    if target.exists():
        raise ValueError('Execution profile already exists; omit source options to reuse it')
    times, commands, constants = read_sequence(controller)
    model = json.loads(Path(geometry_model).read_text())
    offset = np.asarray(model['recorded_data']['optitrack_to_attachment_offset_body_m'], dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError('Invalid tracking-to-attachment geometry')
    folder = directory/'execution_context'/stamp()
    folder.mkdir(parents=True)
    sources = {}
    for key, source in [('controller', controller), ('logger', logger), ('geometry_model', geometry_model)]:
        dest = folder/(key+'.py' if key != 'geometry_model' else key+'.json')
        shutil.copy2(source, dest)
        sources[key] = dict(path=dest.relative_to(directory).as_posix(), sha256=sha256_file(dest),
                            supplied_path=str(Path(source).resolve()))
    profile = dict(schema='legacy_whip_execution_profile_v1', created_utc=utc_now(), sources=sources,
        sequence_time_s=times.tolist(), sequence_fullstate=commands.tolist(), source_constants=constants,
        geometry=dict(offset_tracking_m=offset.tolist(), measurement='55 mm vertical; lateral terms from preserved active calibration',
                      convention='p_attachment = p_cf7 + R_tracking_to_world @ offset_tracking_m'),
        scope='CSV maneuver only; other FullState commands are separate; installed script revision unverified',
        post_sequence_source_behavior='Hold cf.get_position() sampled at sequence end; no return-to-start path',
        command_reference='Unmodified logged cf_7 top-origin commands; user confirms no flight input offset',
        freshness_limit_s=.1, model_fit_performed=False)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(profile, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return target


def load_profile(directory, output):
    directory, output = Path(directory), Path(output)
    path = directory/'execution_profile.json'
    if not path.is_file():
        return None
    profile = json.loads(path.read_text())
    if profile.get('schema') != 'legacy_whip_execution_profile_v1':
        raise ValueError('Unsupported execution profile')
    for key, source in profile['sources'].items():
        original = (directory/source['path']).resolve()
        if not original.is_relative_to(directory.resolve()) or sha256_file(original) != source['sha256']:
            raise ValueError('Execution source path or checksum mismatch')
        shutil.copy2(original, output/original.name)
    times, commands, constants = read_sequence(output/'controller.py')
    if (times.tolist() != profile['sequence_time_s'] or commands.tolist() != profile['sequence_fullstate']
            or constants != profile['source_constants']):
        raise ValueError('Execution profile differs from archived controller literals')
    model = json.loads((output/'geometry_model.json').read_text())
    if model['recorded_data']['optitrack_to_attachment_offset_body_m'] != profile['geometry']['offset_tracking_m']:
        raise ValueError('Geometry differs from archived model')
    shutil.copy2(path, output/'execution_profile.json')
    return profile


def classify_execution(ct, values, fresh, age, profile):
    """Match one complete source sequence in order; retain every logged row.

    Repeated cached observations are not new CSV samples. Missing/reordered/
    ambiguous sequences fail for review; zero V/A is not a maneuver detector.
    """
    expected = np.asarray(profile['sequence_fullstate'])
    changed = np.r_[True, np.any(values[1:] != values[:-1], axis=1) | ~fresh[:-1]]
    starts = np.flatnonzero(fresh & changed)
    block_values = values[starts]
    hits = np.all(np.isclose(block_values[:, None], expected[None], rtol=0, atol=1e-10), axis=2)
    if not np.all(hits.sum(axis=0) == 1):
        raise ValueError('CSV values are missing or ambiguous across command gaps/repetitions; review required')
    candidates = []
    for k in np.flatnonzero(hits[:, 0]):
        if k+len(expected) <= len(starts) and all(hits[k+j, j] for j in range(len(expected))):
            candidates.append(int(k))
    if len(candidates) != 1:
        raise ValueError('CSV sequence missing, reordered or ambiguous; manual execution review required')
    k = candidates[0]
    first = int(starts[k])
    next_block = k+len(expected)
    if next_block == len(starts):
        raise ValueError('CSV termination not observed; cannot assert maneuver end')
    end = int(starts[next_block])
    if not fresh[first:end].all() or np.any(np.diff(ct[first:end+1]) > .05):
        raise ValueError('Command gap inside CSV sequence; manual review required')
    phase = np.full(len(ct), UNKNOWN, dtype='<U40')
    phase[fresh] = OTHER
    holding = fresh & (np.max(np.abs(values[:, 3:9]), axis=1) <= 1e-10) & (np.abs(values[:, 10]) <= 1e-10)
    phase[holding & (np.arange(len(ct)) < first)] = PRE
    phase[holding & (np.arange(len(ct)) >= end)] = POST
    phase[first:end] = CSV
    sample_index = np.full(len(ct), -1, dtype=int)
    events = []
    for j in range(len(expected)):
        a, b = int(starts[k+j]), int(starts[k+j+1])
        sample_index[a:b] = j
        events.append(dict(csv_row=j, csv_time_s=profile['sequence_time_s'][j],
            first_observed_s=float(ct[a]), previous_logger_row_s=float(ct[max(0,a-1)]),
            callback_time_estimate_s=float(ct[a]-age[a]), next_value_observed_s=float(ct[b])))
    bounds = np.r_[0, np.flatnonzero(phase[1:] != phase[:-1])+1, len(ct)]
    intervals = [dict(phase=str(phase[a]), start_s=float(ct[a]),
                      end_s=float(ct[b] if b < len(ct) else ct[-1]), samples=int(b-a),
                      end_is_log_boundary=bool(b == len(ct))) for a,b in zip(bounds[:-1], bounds[1:])]
    post = np.flatnonzero(phase == POST)
    post_positions = np.unique(values[post, :3], axis=0).tolist() if len(post) else []
    # The cache remains finite briefly after publishing stops. This is an upper
    # bound on observed callback coverage, not a controller mode transition.
    last_callback = float(np.max(ct[post]-age[post])) if len(post) else None
    pre = np.flatnonzero(phase == PRE)
    report = dict(schema='legacy_whip_phases_v1', csv_start_s=float(ct[first]), csv_end_s=float(ct[end]),
        csv_duration_observed_s=float(ct[end]-ct[first]), csv_samples_observed=len(events),
        intervals=intervals, csv_events=events,
        pre_hold_target_m=values[pre[-1], :3].tolist() if len(pre) else None,
        post_hold_targets_m=post_positions, post_last_callback_estimate_s=last_callback,
        post_callback_coverage_s=last_callback-float(ct[end]) if last_callback is not None else None,
        source_post_hold_duration_s=profile['source_constants']['RECOVERY_TIME'],
        source_behavior=profile['post_sequence_source_behavior'],
        timing='First observed logger row; callback estimate = row time - cmd_age is diagnostic only. Not radio/onboard actuation time.',
        unknown='No fresh FullState observation; may include takeoff, landing, intervention or cache expiry. No command imputed.',
        phase_meaning='Command source/target, not proof that the drone is stationary or settled')
    return phase, sample_index, report
