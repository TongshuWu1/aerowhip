"""Read original preferred forecasts and compare one fixed-command M0 replay.

No search/fitting, archive restoration, or mutation of original predictions.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import shutil
import numpy as np
import torch
from matplotlib.figure import Figure
from experimental_data.io import atomic_json,sha256_file
from simulator.workflow import read_json
from simulator.pva_commands import sphere_entry
from learning.pva_env import PVAEnvironment
from tools.audit_whip_wave import diagnostics

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT.parent/(ROOT.name+'_archive_20260909_unseen_145g_17g')
OUT=ROOT/'runs/audits/preferred-whip-review-20260910'


def evaluate(arrays,meta,cfg):
    t=arrays['prediction_time_s'];end=meta['whip_end_s'];use=t<=end+1e-10
    t=t[use];q=arrays['cable_positions_m'][use];v=arrays['cable_velocities_m_s'][use]
    target=np.asarray(cfg['launch']['target_m'])
    entry=sphere_entry(torch.tensor(q[:-1]),torch.tensor(q[1:]),
        torch.tensor(target)[None].expand(len(q)-1,-1),cfg['task']['target_radius_m']).numpy()
    ids=np.where(np.isfinite(entry[:,-1]))[0]
    contact=None if not len(ids) else float(t[ids[0]]+entry[ids[0],-1]*(t[ids[0]+1]-t[ids[0]]))
    stop=contact if contact is not None else end
    # This is a retrospective geometry/contact diagnostic, not a new success label.
    diagnostic_meta=dict(meta,predicted_hit_time_s=contact)
    metrics,values=diagnostics(arrays,diagnostic_meta,cfg)
    tt,qq,vv,angles,peak_location,material,drone_v,axis=values
    edges=np.diff(qq,axis=1);lengths=np.linalg.norm(edges[0],axis=-1)
    dual=(lengths[:-1]+lengths[1:])/2
    curvature=angles/dual[None]
    tangent=edges/np.linalg.norm(edges,axis=-1,keepdims=True)
    # Total turning and tangent opposition describe folding without tracking one argmax.
    dot=np.einsum('tij,tkj->tik',tangent,tangent)
    total_turn=angles.sum(1);opposition=np.maximum(0,-dot.min((1,2)))
    band=[material<.45,(material>=.45)&(material<.75),material>=.75]
    origins=arrays['origin_positions_m'][:len(tt)]
    displacement=(origins-np.asarray(cfg['launch']['origin_m']))@axis
    pull=np.maximum.accumulate((displacement>=cfg['task']['minimum_pull_distance_m'])&(drone_v>=cfg['task']['minimum_pull_speed_m_s']))
    eligible=np.where(pull)[0]
    turn_mass=(curvature**2)*dual
    centroid=(turn_mass*material).sum(1)/turn_mass.sum(1).clip(1e-12)
    metrics.update(first_tip_contact_s=contact,geometric_contact=contact is not None,
        peak_total_turn_rad=float(total_turn.max()),peak_tangent_opposition=float(opposition.max()),
        peak_band_local_turn_rad=[float(angles[:,b].max()) for b in band],
        maximum_band_integrated_turn_rad=[float(angles[:,b].sum(1).max()) for b in band],
        pull_ready_time_s=float(tt[eligible[0]]) if len(eligible) else None,
        shape_scope='Through first geometric tip contact, or command end if no contact; no changed historical hit label')
    return metrics,dict(t=tt,q=qq,v=vv,origin=origins,angles=angles,curvature=curvature,
        material=material,centroid=centroid,total_turn=total_turn,opposition=opposition,
        peak_location=peak_location,drone_v=drone_v,target=target,
        origin0=np.asarray(cfg['launch']['origin_m']))


def main():
    OUT.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,OUT/'source.py')
    manifest={r['path']:r['sha256'] for r in read_json(ARCHIVE/'manifest.json')['files']}
    sources={
        'preferred_far':ARCHIVE/'payload/runs/rehearsals_pva/20260909-173305-488091-mppi-wave',
        'preferred_near':ARCHIVE/'payload/runs/rehearsals_pva/20260909-172103-268459-mppi-wave',
        'current':ROOT/'runs/rehearsals_pva/20260910-011618-458510-M0-development-whip'}
    protected={};saved={};views={};reports={}
    for label,path in sources.items():
        for name in ('rehearsal.npz','rehearsal.json','settings.json','model.json','plan.npz','fullstate_30hz.csv'):
            file=path/name;digest=sha256_file(file);protected[str(file)]=digest
            if file.is_relative_to(ARCHIVE):assert manifest[file.relative_to(ARCHIVE/'payload').as_posix()]==digest
        with np.load(path/'rehearsal.npz') as z:arrays={k:z[k].copy() for k in z.files}
        meta=read_json(path/'rehearsal.json');cfg=read_json(path/'settings.json')
        saved[label]=(arrays,meta,cfg)
        reports[label],views[label]=evaluate(arrays,meta,cfg)
    # One fixed old action sequence under the current development model.
    cfg=read_json(ROOT/'runs/mppi_pva/20260910-011618-458510/settings.json')
    cfg['task']['duration_s']=1.3
    model=read_json(ROOT/'runs/mppi_pva/20260910-011618-458510/model.json')
    with np.load(sources['preferred_far']/'plan.npz') as z:actions=z['normalized_jerk'].copy()
    assert actions.shape==(39,3)
    with torch.no_grad():
        env=PVAEnvironment(model,cfg,root=ROOT/'runs/mppi_pva/20260910-011618-458510',device='cuda')
        result=env.rollout(actions=env.tensor(actions)[None],max_steps=39,trace=True)
    stack=lambda key,initial:np.concatenate((initial[None],torch.stack([f[key][0] for f in env.frames]).cpu().numpy()))
    arrays=dict(prediction_time_s=np.r_[0.,[f['time_s'] for f in env.frames]],
        cable_positions_m=stack('cable',env.initial_state.positions_m[0].cpu().numpy()),
        cable_velocities_m_s=stack('cable_velocity',env.initial_state.velocities_m_s[0].cpu().numpy()),
        origin_positions_m=stack('origin',env.initial_pose.position[0].cpu().numpy()),
        origin_velocities_m_s=stack('origin_velocity',env.initial_pose.velocity[0].cpu().numpy()))
    meta=dict(whip_end_s=env.index/30,predicted_valid_hit=bool(result['success'][0]),
        failed=bool(result['failed'][0]),predicted_hit_time_s=float(result['duration_s'][0]) if bool(result['success'][0]) else None,
        label='NEW diagnostic: archived accepted commands under development M0; not the original forecast',
        source_commands=str(sources['preferred_far']/'plan.npz'),source_sha256=sha256_file(sources['preferred_far']/'plan.npz'))
    np.savez_compressed(OUT/'old_commands_new_M0.npz',**arrays,normalized_jerk=actions)
    atomic_json(OUT/'old_commands_new_M0.json',meta)
    reports['old_commands_new_M0'],views['old_commands_new_M0']=evaluate(arrays,meta,cfg)
    reports['old_commands_new_M0']['model_failed']=meta['failed']
    reports['old_commands_new_M0']['minimum_sampled_tip_distance_m']=float(result['minimum_tip_distance_m'][0])
    atomic_json(OUT/'comparison.json',reports)
    atomic_json(OUT/'source_hashes.json',protected)
    labels=['preferred_far','current','old_commands_new_M0']
    titles=['Preferred old forecast','Current full-whip forecast','Old commands / new M0']
    fig=Figure(figsize=(15,9),layout='constrained');ax=fig.subplots(3,3)
    vmax=max(views[k]['curvature'].max() for k in labels)
    for col,(key,title) in enumerate(zip(labels,titles)):
        a=views[key];t=a['t'];q=a['q'];o=a['origin'];x0=a['origin0'][0]
        # Translate old X only for visual comparison. Source arrays remain untouched.
        for fraction,color in zip((.4,.6,.8,1.),('#cbd5e1','#94a3b8','#f59e0b','#dc2626')):
            i=np.argmin(abs(t-fraction*t[-1]));ax[0,col].plot(q[i,:,0]-x0,q[i,:,2],'-o',ms=3,color=color,label=f'{t[i]:.2f}s')
            ax[0,col].plot(o[i,0]-x0,o[i,2],'s',color=color,ms=5)
        ax[0,col].plot(a['target'][0]-x0,a['target'][2],'bx',ms=10)
        ax[0,col].set(title=title,xlabel='X relative to initial drone [m]',ylabel='Z [m]',xlim=(-.2,1.5),ylim=(.1,2.25))
        ax[0,col].set_aspect('equal');ax[0,col].legend(fontsize=8);ax[0,col].grid(alpha=.2)
        mesh=ax[1,col].pcolormesh(t,a['material'],a['curvature'].T,shading='nearest',vmin=0,vmax=vmax,cmap='magma')
        ax[1,col].plot(t,a['peak_location'],'c.',ms=2,label='Strongest bend')
        ax[1,col].set(xlabel='Time [s]',ylabel='Material position: root 0 / tip 1',ylim=(0,1));ax[1,col].legend(fontsize=8)
        ax[2,col].plot(t,a['total_turn'],label='Total cable turning [rad]')
        ax[2,col].plot(t,a['v'][:,-1,0],label='Tip forward speed [m/s]')
        ax[2,col].plot(t,a['drone_v'],label='Drone forward speed [m/s]')
        ax[2,col].set(xlabel='Time [s]',ylim=(-3,7));ax[2,col].grid(alpha=.2);ax[2,col].legend(fontsize=8)
    fig.colorbar(mesh,ax=list(ax[1]),label='Local curvature [rad/m]',shrink=.75)
    fig.suptitle('Saved motion review: strong travelling fold versus target contact\nDifferent models; historical X shifted for display only. No optimization or new fit.',fontsize=14)
    fig.savefig(OUT/'comparison.png',dpi=160)
    assert all(sha256_file(Path(p))==digest for p,digest in protected.items())
    atomic_json(OUT/'status.json',dict(status='completed',protected_files_unchanged=len(protected),
        optimization_started=False,fitting_started=False,old_archive_restored=False,
        diagnostic_replay='One 39-command rollout with development M0'))
    print(json.dumps(reports,indent=2),flush=True)

if __name__=='__main__':main()
