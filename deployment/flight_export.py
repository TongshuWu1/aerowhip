"""Export selected completed rehearsals without changing flight selection."""
from pathlib import Path
import csv
import hashlib
import io
import json
import math
import re
import tempfile
import zipfile


def export_flights(rehearsals, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Choose a new export folder; existing recordings are never overwritten.')
    sources = list(dict.fromkeys(Path(p).resolve() for p in rehearsals))
    if not sources:
        raise ValueError('Select at least one rehearsal.')
    prepared = []
    names = set()
    for source in sources:
        meta = json.loads((source/'rehearsal.json').read_text(encoding='utf-8'))
        if (meta.get('schema') != 'pva_fullstate_30hz_v1' or meta.get('preview_only')
                or meta.get('recovery_prediction_complete') is not True):
            raise ValueError(f'{source.name}: a complete PVA rehearsal with recovery is required.')
        # Capture bytes once so later source changes cannot affect this export.
        payload = (source/'fullstate_30hz.csv').read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != meta.get('csv_sha256'):
            raise ValueError(f'{source.name}: CSV differs from the saved rehearsal.')
        reader = csv.reader(io.StringIO(payload.decode('utf-8-sig')))
        header = next(reader, [])
        expected = ['time_s','px_m','py_m','pz_m','vx_m_s','vy_m_s','vz_m_s',
                    'ax_m_s2','ay_m_s2','az_m_s2','yaw_rad','yaw_rate_rad_s']
        rows = [[float(v) for v in row] for row in reader]
        if header != expected or len(rows) < 2 or any(len(r) != 12 or not all(map(math.isfinite, r)) for r in rows):
            raise ValueError(f'{source.name}: invalid complete 30 Hz CSV.')
        if (abs(rows[0][0]) > 1e-8 or abs(rows[-1][0]-meta['total_duration_s']) > 1e-8
                or any(abs(b[0]-a[0]-1/30) > 1e-8 for a,b in zip(rows,rows[1:]))
                or any(abs(a-b)>1e-8 for a,b in zip(rows[0][1:4],meta['initial_tracking_origin_m']))):
            raise ValueError(f'{source.name}: CSV timing or starting position differs from its rehearsal.')
        name = re.sub(r'[^A-Za-z0-9._-]+', '_', source.name).strip('._') or 'Rehearsal'
        base = name; suffix = 2
        while name.casefold() in names:
            name = f'{base}_{suffix}'; suffix += 1
        names.add(name.casefold())
        model = json.loads((source/'model.json').read_text(encoding='utf-8'))
        generation = model.get('provenance',{}).get('generation_index')
        recording = 'flight_take'+(f'/M{generation}' if type(generation) is int and generation >= 0 else '')
        prepared.append((source,name,meta,payload,recording))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.flight-export-', dir=destination.parent) as temp:
        stage = Path(temp)/'bundle'; stage.mkdir()
        summary = []
        for source,name,meta,payload,recording in prepared:
            folder = stage/name; (folder/recording).mkdir(parents=True)
            (folder/'fullstate_30hz.csv').write_bytes(payload)
            (folder/'export.json').write_text(json.dumps(dict(meta,rehearsal=str(source)),indent=2),encoding='utf-8')
            start = ', '.join(f'{v:.4f}' for v in meta['initial_tracking_origin_m'])
            (folder/'README.md').write_text(
                f'# {name}\n\nUse fullstate_30hz.csv with the existing controller.\n'
                f'Starting tracked-origin position: ({start}) m.\n'
                f'Complete duration: {meta["total_duration_s"]:.3f} s, including braking, return and final hold.\n'
                'The first CSV row is the starting hover, not a takeoff command. Use the starting height for this plan.\n\n'
                f'Save recordings in {recording}/. The saved model and prediction remain in the source rehearsal listed in export.json.\n'
                'Complete simulated recovery was checked; physical performance remains to be measured.\n',encoding='utf-8')
            summary.append(dict(name=name,rehearsal=str(source),csv_sha256=meta['csv_sha256'],
                                initial_tracking_origin_m=meta['initial_tracking_origin_m'],duration_s=meta['total_duration_s']))
        (stage/'exports.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        archive = stage/'flight_commands.zip'
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for _,name,_,_,recording in prepared:
                for path in (stage/name).rglob('*'):
                    if path.is_file(): z.write(path,path.relative_to(stage).as_posix())
                z.writestr(f'{name}/{recording}/','')
            z.write(stage/'exports.json','exports.json')
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None: raise ValueError('Export ZIP verification failed.')
            for _,name,meta,_,_ in prepared:
                if hashlib.sha256(z.read(f'{name}/fullstate_30hz.csv')).hexdigest() != meta['csv_sha256']:
                    raise ValueError('Exported CSV verification failed.')
        # Publish only a complete bundle, never replace an existing directory.
        stage.rename(destination)
    return destination
