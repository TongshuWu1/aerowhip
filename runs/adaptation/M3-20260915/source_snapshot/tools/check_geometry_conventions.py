"""Read-only geometry audit of existing data; no optimization, fitting or training.

Writes a new report and descriptive measurements, never replaces processed data.
Run from the repository root with: python -m tools.check_geometry_conventions
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
from experimental_data.historical_dataset import TAKES, load_source
from experimental_data.io import sha256_file
from simulator.geometry import normalized_rotations_xyzw, attachment_positions
from simulator.workflow import stamp

ROOT=Path(__file__).resolve().parents[1]


def raw_rigid_markers(path):
    with path.open(newline='',encoding='utf-8-sig') as stream:
        reader=csv.reader(stream);header=[next(reader) for _ in range(7)]
        def columns(name,kind,axes):
            return [next(i for i,n in enumerate(header[3]) if n==name and
                header[5][i]==kind and header[6][i]==axis) for axis in axes]
        names=sorted({n for n in header[3] if n.startswith('cf_7:Marker')})
        ids=columns('cf_7','Position','XYZ')+columns('cf_7','Rotation','XYZW')
        ids += [i for name in names for i in columns(name,'Position','XYZ')]
        a=np.asarray([[float(row[i]) if i<len(row) and row[i].strip() else np.nan for i in ids]
                      for row in reader if row])
    R,valid=normalized_rotations_xyzw(a[:,3:7])
    delta=a[:,7:].reshape(-1,len(names),3)-a[:,:3,None].transpose(0,2,1)
    local=np.einsum('tji,tmj->tmi',R,delta)
    wrong=np.einsum('tij,tmj->tmi',R,delta)
    center=np.nanmedian(local,axis=0)
    return dict(metadata=dict(zip(header[0][::2],header[0][1::2])),names=names,
        quaternion_valid_frames=int(valid.sum()),
        marker_local_median_m=center.tolist(),
        local_marker_spread_rms_m=float(np.sqrt(np.nanmean(np.sum((local-center)**2,axis=-1)))),
        inverse_convention_spread_rms_m=float(np.sqrt(np.nanmean(np.sum((wrong-np.nanmedian(wrong,axis=0))**2,axis=-1)))),
        interpretation='Descriptive rigidity check; does not identify the firmware body axes, COM or attachment location')


def audit(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'geometry.json').exists():raise ValueError('Use a new audit output directory')
    candidate=ROOT/'data/historical_model_runs/20260908-013552-736857-whip-only-drone/candidate_bundle/model.json'
    paths=dict(active=ROOT/'config/model.json',historical_candidate=candidate)
    models={k:json.loads(p.read_text()) for k,p in paths.items()}
    protected=[*paths.values(),ROOT/'config/task.json',ROOT/'config/ppo.json',ROOT/'config/baseline.json',
        candidate.with_name('motion_residual.pt'),candidate.with_name('drone_attachment.pt'),
        ROOT/'runs/ppo/20260906-201957-294110-measured-mass-seed653/checkpoints/best_validation.pt']
    for name in TAKES:
        folder=(ROOT/'data/adaptation_rounds/adaptation0/raw'/name if name.startswith('whip') else ROOT/'data/raw_takes'/name)
        protected.extend(folder.glob('*.csv'))
    before={str(p.relative_to(ROOT)):sha256_file(p) for p in protected}
    rows={};geometry={}
    for label,model in models.items():
        geometry[label]=dict(offset_tracking_m=model['recorded_data']['optitrack_to_attachment_offset_body_m'],
            attachment_to_c1_arc_m=model['cable']['marker_interval_lengths_m'][0],
            cable_arc_lengths_m=model['cable']['marker_interval_lengths_m'],
            interval_subdivisions=model['cable']['segments_per_marker_interval'])
    for name in TAKES:
        path,a,details=load_source(ROOT,name)
        before[str(path.relative_to(ROOT))]=sha256_file(path)
        folder=(ROOT/'data/adaptation_rounds/adaptation0/raw'/name if name.startswith('whip') else ROOT/'data/raw_takes'/name)
        record=dict(source=str(path.relative_to(ROOT)),frames=len(a['time']),
            rigid_markers=raw_rigid_markers(folder/(name+'.csv')),geometry={})
        for label,model in models.items():
            attachment,valid=attachment_positions(a['position'],a['quaternion'],geometry[label]['offset_tracking_m'])
            ok=valid&a['position_valid']&a['marker_valid'][:,0]&np.isfinite(a['markers'][:,0]).all(-1)
            chord=np.linalg.norm(a['markers'][:,0]-attachment,axis=-1)
            arc=geometry[label]['attachment_to_c1_arc_m']
            offset=attachment-a['position'];first=np.flatnonzero(ok)[0]
            effect=np.linalg.norm(offset-offset[first],axis=-1)
            record['geometry'][label]=dict(valid_first_span_frames=int(ok.sum()),
                first_span_chord_quantiles_m=np.quantile(chord[ok],[0,.5,.95,1]).tolist(),
                first_span_excess_over_2mm_frames=int((ok&(chord>arc+.002)).sum()),
                first_span_excess_over_25mm_frames=int((ok&(chord>arc+.025)).sum()),
                rotation_vs_fixed_initial_offset_max_m=float(effect[ok].max()))
        rows[name]=record
    after={p:sha256_file(ROOT/p) for p in before}
    if before!=after:raise RuntimeError('An input changed while auditing; do not use this report')
    result=dict(schema='geometry_convention_audit_v1',scope='descriptive geometry only; no fitting',
        protected_files_sha256=before,inputs_unchanged=True,geometry=geometry,takes=rows,
        remaining=['flight computer mocap configuration and installed firmware revision',
            'tracking rigid-body to firmware body-frame transform',
            'independent metrology of lateral offset and marker-center versus cable-centerline placement'])
    (output/'geometry.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    report=['# Geometry audit — existing recordings only','',
        'No fit, optimization, data rejection or training. All measurements and configurations were preserved.',
        'Statistics cover each entire recording, not only its maneuver. A 2 mm excess is a diagnostic threshold, not a newly applied mask.',
        '', '| Take | Rigid marker spread using R^T (mm) | Using R instead (mm) | Candidate first-span median (mm) | Candidate excess >2 mm / valid |',
        '|---|---:|---:|---:|---:|']
    for name,r in rows.items():
        m=r['rigid_markers'];g=r['geometry']['historical_candidate']
        report.append(f"| {name} | {m['local_marker_spread_rms_m']*1000:.3f} | {m['inverse_convention_spread_rms_m']*1000:.3f} | {g['first_span_chord_quantiles_m'][1]*1000:.2f} | {g['first_span_excess_over_2mm_frames']}/{g['valid_first_span_frames']} |")
    report += ['', 'Lower marker spread after R^T maps world observations into a rigid local constellation and supports the recorded quaternion direction. It does not establish the aircraft-frame alignment or independently measure the attachment.',
        '', 'Keep 55 mm rigid vertical offset and 63 mm flexible first-span arc length as recorded approximate measurements. Existing fitted lateral offsets remain separate from those measurements.',
        '', 'See geometry.json for both preserved geometry versions, complete descriptive statistics and input hashes.']
    (output/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    print(audit(args.output or ROOT/'runs/audits'/(stamp()+'-geometry-conventions')))
