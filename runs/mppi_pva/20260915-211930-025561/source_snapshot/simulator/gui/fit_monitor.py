"""Read-only views of saved fitting progress, including externally started jobs."""
import json
import math
from pathlib import Path

LEGACY_SCHEMAS = {'normalized_adp0_cold_pva_bootstrap_v1', 'preliminary_pva_bootstrap_v1'}


def read(path, default):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        return value if isinstance(value, type(default)) else default
    except (OSError, ValueError):
        return default


def modified(path):
    try:
        return Path(path).stat().st_mtime_ns
    except OSError:
        return 0


def discover(root):
    jobs = []
    for path in (Path(root)/'runs/adaptation').glob('*/protocol.json'):
        if (path.parent/'ARCHIVED').exists():
            continue
        protocol = read(path, {})
        full = protocol.get('full_update', {})
        if not full and protocol.get('schema') not in LEGACY_SCHEMAS:
            continue
        status = read(path.parent/'status.json', {})
        label = (f"{full.get('parent_id', '?')} → {full.get('candidate_id', '?')}"
                 if full else 'Preliminary M0')
        jobs.append(dict(path=path.parent, protocol=protocol, status=status,
                         modified=modified(path.parent/'status.json'),
                         label=f"{label} · {status.get('status', 'prepared')} · {path.parent.name}"))
    return sorted(jobs, key=lambda j:j['modified'], reverse=True)


def number(value):
    return float(value) if isinstance(value, (float, int)) and math.isfinite(value) else None


def stage_view(path, title, folder, active, progress):
    folder = Path(path)/folder
    rows = [r for r in read(folder/'history.json', []) if isinstance(r, dict)]
    result = read(folder/'result.json', {})
    baseline = number(result.get('baseline_loss', rows[0].get('baseline_loss') if rows else None))
    points = []
    if baseline is not None:
        points.append((0, baseline, baseline))
    for r in rows:
        x = number(r.get('update'))
        y = number(r.get('selection_loss', r.get('loss')))
        if x is not None and y is not None:
            points.append((x, y, number(r.get('best_loss'))))
    best = number(result.get('best_loss', rows[-1].get('best_loss') if rows else None))
    if best is None and active:
        best = number(progress.get('best_loss'))
    selected = number(result.get('selected_update'))
    update = progress.get('update') if active else result.get('updates', rows[-1].get('update') if rows else None)
    reason = result.get('stop_reason', '')
    verified = result.get('numerically_verified')
    state = ('Finalizing' if active and reason and verified is not True else 'Training' if active else
             reason.replace('_', ' ').capitalize() if reason else 'Recorded' if rows else 'Waiting')
    return dict(title=title, folder=folder, points=points, baseline=baseline, best=best,
                reduction=100*(1-best/baseline) if baseline is not None and baseline>0 and best is not None else None,
                update=update, selected_update=selected, state=state, active=active, result=result,
                fingerprint=(modified(folder/'history.json'), modified(folder/'result.json')))


def snapshot(path):
    path = Path(path);protocol = read(path/'protocol.json', {});status = read(path/'status.json', {})
    full = protocol.get('full_update', {})
    progress = ({**read(path/'progress.json', {}), **status} if full
                else {**status, **read(path/'progress.json', {})})
    stage = progress.get('stage', '')
    running = status.get('status') == 'running'
    if full:
        parameters = read(path/'drone_nominal/parameters.json', {})
        delay = number(parameters.get('delay_s'))
        # Nominal search profiles have different delays; show only the chosen
        # profile, rather than joining unrelated losses into one curve.
        nominal = f'drone_nominal/delay-{delay:.3f}' if delay is not None else 'drone_nominal/pending'
        specs = [('Vehicle parameters', nominal, 'drone_nominal'),
                 ('Vehicle residual', 'drone_residual', 'drone_residual'),
                 ('Cable parameters', 'cable_physics', 'cable_physics'),
                 ('Cable residual', 'cable_residual', 'cable_residual')]
    else:
        residual = 'cable/residual_full_whip' if (path/'cable/residual_full_whip/history.json').exists() else 'cable/residual'
        specs = [('Vehicle parameters', 'drone/nominal', 'drone_nominal'),
                 ('Vehicle residual', 'drone', 'drone'),
                 ('Cable parameters', 'cable/physics', 'cable_physics'),
                 ('Cable residual', residual, 'cable_residual')]
    stages = [stage_view(path, title, folder, running and key in stage, progress) for title,folder,key in specs]
    settings = full.get('residual_stopping', {})
    if settings:
        checks = settings.get('patience', '?');every = settings.get('check_every', '?')
        percent = 100*settings.get('relative', 0)
        stopping = (f"Residual stopping: at least {settings.get('minimum', '?')} updates; "
                    f"{checks} checks without {percent:g}% meaningful improvement, checked every {every} updates.")
        budgets = full.get('stage_budgets', {})
        if budgets:
            stopping += ' Caps: '+ '; '.join(f"{k.replace('_', ' ')} {b.get('maximum_updates')} updates / {b.get('maximum_seconds')} s" for k,b in budgets.items())+'.'
        elif settings.get('ceiling') is None:
            stopping += ' No residual update or time cap.'
        else:
            stopping += f" Residual update cap: {settings['ceiling']}."
    else:
        stopping = 'Stopping follows this job’s saved fit settings.'
    cable_settings = full.get('cable_residual_stopping')
    if cable_settings:
        stopping += (f" Cable-only amendment: {cable_settings.get('patience')} checks without "
                     f"{100*cable_settings.get('relative', 0):g}% meaningful improvement, "
                     f"checked every {cable_settings.get('check_every')} updates; "
                     "vehicle rule unchanged. Patience restarts at continuation.")
    return dict(path=path, protocol=protocol, status=status, progress=progress, stages=stages,
                stopping=stopping, stop_supported=not full and protocol.get('schema') in LEGACY_SCHEMAS,
                fingerprint=(str(path), tuple(s['fingerprint'] for s in stages)))
