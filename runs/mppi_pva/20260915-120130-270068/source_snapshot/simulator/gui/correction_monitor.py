"""Read saved command corrections without running a model or changing artifacts."""
from pathlib import Path
from zipfile import BadZipFile
import numpy as np
from .fit_monitor import read, modified, number


def resolve(root, value):
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else Path(root)/path


def discover(root):
    jobs = []
    for settings in (Path(root)/'runs/reference_tracking').glob('*/settings.json'):
        path = settings.parent
        if (path/'ARCHIVED').exists() or not read(settings, {}).get('correction'):
            continue
        status = read(path/'status.json', {})
        jobs.append(dict(path=path, status=status,
                         modified=max(modified(path/'status.json'), modified(path/'history.json')),
                         label=f"{path.name} · {status.get('status', 'unavailable')}"))
    return sorted(jobs, key=lambda j:j['modified'], reverse=True)


def locations(root, path):
    path = Path(path)
    settings = read(path/'settings.json', {})
    result = read(path/'result.json', {})
    optimization = result.get('optimization') or read(path/'optimization.json', {})
    initialization = settings.get('correction', {}).get('initialization')
    previous = resolve(root, initialization)
    if initialization in (None, 'Original executed reference controls'):
        previous = resolve(root, read(path/'reference.json', {}).get('source_rehearsal'))
    rehearsal = resolve(root, result.get('rehearsal'))
    history = resolve(root, optimization.get('reused_controls_from')) or path
    return settings, result, optimization, previous, rehearsal, history


def fingerprint(root, path):
    path = Path(path)
    _, _, _, previous, rehearsal, history = locations(root, path)
    files = [path/name for name in ('settings.json', 'status.json', 'result.json', 'reference.json',
             'optimization.json', 'reference.npz', 'baseline.json', 'baseline_prediction.npz',
             'current_best.npz', 'plan.npz', 'replay_provenance.json')]
    files.append(history/'history.json')
    files.extend(p/'rehearsal.npz' for p in (previous, rehearsal) if p is not None)
    return tuple((str(p), modified(p)) for p in files)


def arrays(path, warnings):
    if path is None or not path.exists():
        return {}
    try:
        with np.load(path, allow_pickle=False) as saved:
            # These exports may also contain text metadata. Only read numbers.
            return {k:saved[k].copy() for k in saved.files if saved[k].dtype.kind in 'fiu'}
    except (OSError, ValueError, EOFError, BadZipFile) as exc:
        warnings.append(f'{path.name} is not readable yet: {exc}')
        return {}


def series(time, values, columns=3):
    if time is None or values is None:
        return None
    t, v = np.asarray(time), np.asarray(values)
    if (t.ndim != 1 or v.ndim != 2 or len(t) != len(v) or len(t) < 2 or
            v.shape[1] < columns or not np.isfinite(t).all() or
            not np.isfinite(v).all() or not (np.diff(t) > 0).all()):
        return None
    return dict(time=t, values=v)


def motion(saved, baseline=False):
    time = saved.get('time_s' if baseline else 'prediction_time_s')
    cable = saved.get('cable_positions_m')
    tip = cable[:, -1] if cable is not None and cable.ndim == 3 else None
    return dict(tip=series(time, tip), vehicle=series(time, saved.get('origin_positions_m')))


def error_curve(prediction, reference):
    """Compare only overlapping saved times; never extrapolate a prediction."""
    if prediction is None or reference is None:
        return None
    t, ref = reference['time'], reference['values']
    mask = (t >= prediction['time'][0]) & (t <= prediction['time'][-1])
    if mask.sum() < 2:
        return None
    values = np.column_stack([np.interp(t[mask], prediction['time'], prediction['values'][:, j])
                              for j in range(3)])
    return dict(time=t[mask], values=np.linalg.norm(values-ref[mask], axis=1)*100)


def at_time(curve, time):
    if curve is None or time is None or not curve['time'][0] <= time <= curve['time'][-1]:
        return None
    return np.array([np.interp(time, curve['time'], curve['values'][:, j]) for j in range(3)])


def snapshot(root, path):
    path = Path(path)
    settings, result, optimization, previous, rehearsal, history_path = locations(root, path)
    status = read(path/'status.json', {})
    warnings = []
    ref = arrays(path/'reference.npz', warnings)
    before = arrays(previous/'rehearsal.npz' if previous else None, warnings)
    after = arrays(rehearsal/'rehearsal.npz' if rehearsal else None, warnings)
    commands = dict(previous=series(before.get('command_time_s'), before.get('commands'), 9),
                    corrected=series(after.get('command_time_s'), after.get('commands'), 9),
                    reference=series(ref.get('command_time_s'), ref.get('original_command_packets'), 9))
    if commands['previous'] is None and previous is not None:
        warnings.append('Previous full command is unavailable; it has not been replaced with the M0 command.')
    provisional = commands['corrected'] is None
    if provisional:
        saved = arrays(path/'current_best.npz', warnings)
        if not saved:
            saved = arrays(path/'plan.npz', warnings)
        if 'position_control_points_m' in saved:
            try:
                # Decode the saved command spline on CPU only. Do not invent a
                # live recovery trajectory or launch a prediction from this UI.
                import torch
                from planning.position_spline import PositionSpline
                duration = settings['correction']['original_spline_duration_s']
                spline = PositionSpline(duration)
                packets, _ = spline.decode(torch.as_tensor(saved['position_control_points_m'], dtype=torch.float64),
                                           settings['launch']['origin_m'])
                times = ref.get('command_time_s')
                if times is not None:
                    commands['corrected'] = series(times, packets[:len(times)].numpy(), 9)
            except (KeyError, ValueError, RuntimeError, TypeError) as exc:
                warnings.append(f'Saved command preview is unavailable: {exc}')
    reference = dict(tip=series(ref.get('time_s'), ref.get('tip_position_m')),
                     vehicle=series(ref.get('time_s'), ref.get('quadrotor_position_m')))
    predictions = dict(previous=motion(arrays(path/'baseline_prediction.npz', warnings), True),
                       corrected=motion(after))
    errors = {name:{part:error_curve(pred[part], reference[part]) for part in ('tip', 'vehicle')}
              for name, pred in predictions.items()}
    meta = read(path/'reference.json', {})
    strike = number(meta.get('planned_strike_time_s'))
    target = np.asarray(meta.get('physical_target_m', []))
    strike_rows = []
    for name, curve in [('M0 reference', reference['tip']),
                        ('Previous command', predictions['previous']['tip']),
                        ('Corrected command', predictions['corrected']['tip'])]:
        point, anchor = at_time(curve, strike), at_time(reference['tip'], strike)
        strike_rows.append((name,
            float(np.linalg.norm(point-anchor)*100) if point is not None and anchor is not None else None,
            float(np.linalg.norm(point-target)*100) if point is not None and target.shape == (3,) else None))
    history = [r for r in read(history_path/'history.json', []) if isinstance(r, dict)]
    baseline = read(path/'baseline.json', {})
    initial = number(result.get('baseline_cost_m2', baseline.get('cost_m2')))
    final = number(result.get('corrected_cost_m2'))
    if final is None and history:
        final = number(history[-1].get('cost_m2'))
    return dict(path=path, settings=settings, result=result, status=status, optimization=optimization,
                previous=previous, rehearsal=rehearsal, history_path=history_path, history=history,
                commands=commands, provisional=provisional, reference=reference,
                predictions=predictions, errors=errors, strike=strike, strike_rows=strike_rows,
                initial=initial, final=final, warnings=warnings, target=target,
                provenance=read(path/'replay_provenance.json', {}),
                whip_end=number(meta.get('interval_s', [None, None])[-1]))
