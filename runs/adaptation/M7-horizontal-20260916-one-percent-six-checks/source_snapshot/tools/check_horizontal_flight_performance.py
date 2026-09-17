"""Compare measured M5/M6/M7 horizontal executions against one desired motion.

Read frozen, event-log-verified segments. No model fitting or command changes.
"""
from pathlib import Path
import sys, json, csv, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experimental_data.adaptation_rounds import read_optitrack
from experimental_data.io import sha256_file, atomic_json
from tools.process_repeated_pva_recordings import _marker_quality
from tools.evaluate_vertical_generations import closest_approach

OUT = ROOT/'output/horizontal_M5_M6_M7_performance_20260915'
COLLECTIONS = {'M5':'M5-curved-side','M6':'M6-horizontal','M7':'M7-horizontal'}
COLORS = {'M5':'#d78323','M6':'#2879b8','M7':'#249773'}

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def interpolate_valid(t, q, valid, query):
    """Only interpolate adjacent valid samples, with at most 30 ms separation."""
    query = np.atleast_1d(query)
    right = np.searchsorted(t, query).clip(1,len(t)-1); left = right-1
    ok = (query>=t[0]) & (query<=t[-1]) & valid[left] & valid[right]
    ok &= (t[right]-t[left])<=.03
    f = (query-t[left])/(t[right]-t[left])
    result = q[left]+f[:,None]*(q[right]-q[left])
    result[~ok] = np.nan
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    refdir = ROOT/'runs/reference_tracking/M5-horizontal-fixed-reference'
    meta = read(refdir/'reference.json')
    assert sha256_file(refdir/'reference.npz') == meta['reference_sha256']
    with np.load(refdir/'reference.npz') as z:
        rt=z['time_s']; rq=z['tip_position_m']
    strike=meta['planned_strike_time_s']; end=meta['interval_s'][1]
    target=np.asarray(meta['physical_target_m'])
    grid=np.linspace(0,end,308)
    refgrid=np.stack([np.interp(grid,rt,rq[:,j]) for j in range(3)],axis=1)
    rows=[]; curves={}; hashes={str(refdir/f):sha256_file(refdir/f) for f in ('reference.json','reference.npz')}
    for model,label in COLLECTIONS.items():
        collection_path=ROOT/f'runs/data_review/{label}-complete-collection/report.json'
        collection=read(collection_path)
        hashes[str(collection_path)]=sha256_file(collection_path)
        for path,digest in collection['source_hashes'].items():
            assert sha256_file(path)==digest, path
            hashes[path]=digest
        curves[model]=[]
        for batch in sorted((ROOT/'data/flight_batches').glob(f'{label}-session*-four-whips')):
            manifest=read(batch/'split_manifest.json'); protocol=read(batch/'protocol.json')
            alignment=read(batch/'time_alignment.json')
            rehearsal=Path(protocol['rehearsal']); physics=read(rehearsal/'model.json')
            assert sha256_file(rehearsal/'rehearsal.npz') == protocol['forecast_sha256']
            with np.load(rehearsal/'rehearsal.npz') as z:
                pt=z['prediction_time_s']; pq=z['cable_positions_m'][:,-1]
            for row in manifest['repetitions']:
                name=row['name']; path=batch/'flight_take'/f'{name}.csv'
                assert sha256_file(path)==alignment[name]['optitrack_sha256']
                hashes[str(path)]=sha256_file(path)
                m=read_optitrack(path); _, mv=_marker_quality(m,physics)
                t=np.asarray(m['time'])+alignment[name]['offset_s']-row['onset_s']
                q=np.asarray(m['cable'])[:,-1]; valid=mv[:,-1]&np.isfinite(q).all(-1)
                ref=np.stack([np.interp(t,rt,rq[:,j]) for j in range(3)],axis=1)
                predicted=np.stack([np.interp(t,pt,pq[:,j]) for j in range(3)],axis=1)
                pre=(t>=0)&(t<=strike); whole=(t>=0)&(t<=end)
                assert np.max(np.diff(t[whole]))<=.03
                error=np.linalg.norm(q-ref,axis=-1)
                def rmse(mask,delta):
                    keep=mask&valid
                    return float(np.sqrt(np.mean(np.sum(delta[keep]**2,axis=-1))))
                event=closest_approach(t[whole],q[whole],valid[whole],target)
                qs=interpolate_valid(t,q,valid,[strike])[0]
                curve=interpolate_valid(t,q,valid,grid)
                curves[model].append(curve)
                record=dict(model=model,source_model=collection['model'],take=name,recording_group=row['recording_group'],
                    reference_rmse_prestrike_m=rmse(pre,q-ref),
                    reference_rmse_full_whip_m=rmse(whole,q-ref),
                    saved_prediction_rmse_prestrike_m=rmse(pre,q-predicted),
                    planned_time_target_distance_m=float(np.linalg.norm(qs-target)) if np.isfinite(qs).all() else None,
                    minimum_target_distance_m=event['distance_m'],closest_approach_time_s=event['time_s'],
                    closest_approach_at_boundary=event['at_observed_window_boundary'],
                    valid_tip_fraction=float(valid[whole].mean()),
                    prestrike_samples=int(np.count_nonzero(pre&valid)),
                    full_whip_samples=int(np.count_nonzero(whole&valid)),
                    alignment_rms_m=row['alignment_rms_m'])
                rows.append(record)
            for file in ('split_manifest.json','protocol.json','time_alignment.json'):
                hashes[str(batch/file)]=sha256_file(batch/file)
        assert len(curves[model])==collection['processed_whips']
        print(model, 'processed', len(curves[model]),flush=True)
    keys=['reference_rmse_prestrike_m','reference_rmse_full_whip_m','saved_prediction_rmse_prestrike_m',
          'planned_time_target_distance_m','minimum_target_distance_m']
    def aggregate(items):
        return {key:dict(n=len(a),mean_cm=float(np.mean(a)*100),sd_cm=float(np.std(a,ddof=1)*100))
                for key in keys if len(a:=np.array([r[key] for r in items if r[key] is not None]))}
    summary={model:dict(whips=len(items),recordings=len(set(r['recording_group'] for r in items)),
        metrics=aggregate(items),recording_means={g:aggregate([r for r in items if r['recording_group']==g])
        for g in sorted(set(r['recording_group'] for r in items))},
        minimum_tip_valid_fraction=min(r['valid_tip_fraction'] for r in items),
        boundary_minima=sum(r['closest_approach_at_boundary'] for r in items))
        for model in COLLECTIONS if (items:=[r for r in rows if r['model']==model])}
    contract=dict(evidence='Actual OptiTrack tip measurements; desired trajectory is the saved M5 simulation.',
        reference=str(refdir),prestrike_interval_s=[0,strike],full_whip_interval_s=[0,end],target_m=target.tolist(),
        alignment='Saved measured-vehicle-stream clock alignment and event-log onset. No target fitting, spatial registration, or time warping. Clock offsets include unknown transport latency.',
        validity='Existing marker-quality masks; 30 ms maximum interpolation gap. Native samples used for RMSE; interpolated common grid only for visualization.',
        aggregation='Equal-whip mean and sample SD; recording means also retained. Repetitions within sessions are not independent sessions.',
        scope='Different executed commands; execution performance, not an isolated same-flight predictor comparison. All complete repetitions in the selected collections are included.',
        display_to_source_model={m:read(ROOT/f'runs/data_review/{label}-complete-collection/report.json')['model'] for m,label in COLLECTIONS.items()},
        label_policy='Display labels only; original recording names and model ancestry are preserved. When source M6 is omitted, displayed M5 to M6 does not imply a single model-update round.',
        target_interpretation='Distance to a virtual target, not physical contact or hit rate.',
        operator_conditions='User previously stated there is no contact or intervention.',
        recovery_note='M7 session 1 whip 4 has a partial recovery tail; its complete scored maneuver is included.')
    atomic_json(OUT/'metrics.json',dict(protocol=contract,summary=summary,flights=rows))
    hashes[str(Path(__file__))]=sha256_file(__file__)
    atomic_json(OUT/'source_hashes.json',hashes)
    with (OUT/'per_flight_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(2,3,figsize=(13.2,7.3),layout='constrained')
    for model in COLLECTIONS:
        a=np.asarray(curves[model]); color=COLORS[model]; n=len(a)
        mean=np.nanmean(a,axis=0)
        for curve in a: axs[0,0].plot(curve[:,0],curve[:,1],color=color,alpha=.14,lw=.7)
        axs[0,0].plot(mean[:,0],mean[:,1],color=color,lw=2,label=f'{model} measured (n={n})')
        errors=np.linalg.norm(a-refgrid,axis=-1)*100
        e=np.nanmean(errors,axis=0); sd=np.nanstd(errors,axis=0,ddof=1)
        axs[0,1].plot(grid,e,color=color,lw=1.8,label=model)
        axs[0,1].fill_between(grid,np.maximum(0,e-sd),e+sd,color=color,alpha=.12)
    axs[0,0].plot(refgrid[:,0],refgrid[:,1],'k--',lw=1.7,label='Desired trajectory')
    axs[0,0].scatter(*target[:2],marker='x',s=70,color='black',zorder=5,label='Target')
    axs[0,0].set(xlabel='X (m)',ylabel='Y (m)',title='(a) Measured tip paths — top view')
    axs[0,0].set_aspect('equal',adjustable='datalim');axs[0,0].legend(fontsize=8,loc='best')
    axs[0,1].axvline(strike,color='.35',ls=':',lw=1)
    axs[0,1].text(strike-.025,.97,'Planned strike',rotation=90,ha='right',va='top',transform=axs[0,1].get_xaxis_transform(),fontsize=8)
    axs[0,1].set(xlabel='Time from command onset (s)',ylabel='3D tip error (cm)',title='(b) Error to desired trajectory',xlim=(0,end),ylim=(0,None))
    for ax,key,title in [(axs[0,2],'reference_rmse_prestrike_m','(c) Tracking RMSE: 0–1.293 s'),
                          (axs[1,0],'reference_rmse_full_whip_m','(d) Tracking RMSE: 0–1.533 s'),
                          (axs[1,1],'planned_time_target_distance_m','(e) Target distance at planned strike'),
                          (axs[1,2],'minimum_target_distance_m','(f) Closest target distance: full whip')]:
        for i,model in enumerate(COLLECTIONS):
            selected=[r for r in rows if r['model']==model]; values=np.array([r[key]*100 for r in selected])
            ax.scatter(i+np.linspace(-.12,.12,len(values)),values,color=COLORS[model],alpha=.6,s=22)
            ax.errorbar(i,values.mean(),yerr=values.std(ddof=1),fmt='s',color='black',ms=5,capsize=5,lw=1.2)
        ax.set(xticks=range(len(COLLECTIONS)),xticklabels=[f'{m}\nn={summary[m]["whips"]}' for m in COLLECTIONS],ylabel='Distance (cm)',title=title,ylim=(0,None),xlim=(-.5,len(COLLECTIONS)-.5))
    for ax in axs.flat: ax.grid(alpha=.18)
    fig.suptitle('Horizontal maneuver: '+' → '.join(COLLECTIONS),fontsize=15)
    fig.supxlabel('Actual flight measurements · bands and error bars: ±1 SD across whips',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(OUT/f'horizontal_comparison.{ext}',dpi=200)
    plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(8,7),sharex=True,layout='constrained')
    for j,ax in enumerate(axs):
        for model in COLLECTIONS:
            a=np.asarray(curves[model])[:,:,j];mean=np.nanmean(a,axis=0);sd=np.nanstd(a,axis=0,ddof=1)
            ax.plot(grid,mean,color=COLORS[model],label=f'{model} measured')
            ax.fill_between(grid,mean-sd,mean+sd,color=COLORS[model],alpha=.13)
        ax.plot(grid,refgrid[:,j],'k--',label='Desired trajectory')
        ax.axvline(strike,color='.4',ls=':');ax.set_ylabel(f'Tip {"XYZ"[j]} (m)');ax.grid(alpha=.18)
    axs[0].legend(ncol=2,fontsize=9);axs[0].set_title('Horizontal maneuver — measured tip coordinates (mean ±1 SD)')
    axs[-1].set(xlabel='Time from command onset (s)',xlim=(0,end))
    for ext in ('png','pdf'):fig.savefig(OUT/f'tip_coordinates.{ext}',dpi=200)
    plt.close(fig)
    lines=['# Horizontal flight performance: '+', '.join(COLLECTIONS),'',contract['evidence'],'',
           '| Command | Whips / recordings | Tip RMSE to desired (cm) | Planned-time target distance (cm) | Closest target distance (cm) |',
           '|---|---:|---:|---:|---:|']
    for m,s in summary.items():
        vals=[s['metrics'][k] for k in (keys[0],keys[3],keys[4])]
        lines.append(f'| {m} | {s["whips"]} / {s["recordings"]} | '+' | '.join(f'{v["mean_cm"]:.2f} ± {v["sd_cm"]:.2f}' for v in vals)+' |')
    lines+=['','Mean ± sample SD across whips. RMSE uses 0 to 1.29333 s; closest approach uses 0 to 1.53333 s.','',
        *[f'- **{k}:** {v}' for k,v in contract.items() if k not in ('reference','evidence')], '',
        'Saved-prediction RMSE is included in metrics.json as a diagnostic; each predictor saw different executed commands, so it is not a controlled predictor comparison.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    assert all(sha256_file(path)==digest for path,digest in hashes.items())
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m5-m7-as-m5-m6',action='store_true',help='Omit source M6; display source M7 as M6, preserving provenance.')
    args=parser.parse_args()
    if args.m5_m7_as_m5_m6:
        OUT=ROOT/'output/horizontal_M5_M6_comparison_20260915'
        COLLECTIONS={'M5':'M5-curved-side','M6':'M7-horizontal'}
        COLORS={'M5':'#d78323','M6':'#249773'}
    main()
