"""Fixed preference examples and counterexamples before any new MPPI search."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from experimental_data.io import atomic_json,sha256_file
from simulator.workflow import read_json
from simulator.pva_commands import sphere_entry
from planning.whip_objective import fold_quality,material_resample,encounter_quality,WAVE_OBJECTIVE
from tools.review_preferred_whip import ROOT,ARCHIVE

AUDIT=ROOT/'runs/audits/mppi-preferred-fold-20260910'
PREFERRED=ARCHIVE/'payload/runs/rehearsals_pva/20260909-173305-488091-mppi-wave'
CURRENT=ROOT/'runs/rehearsals_pva/20260910-011618-458510-M0-development-whip'
NEAR=ARCHIVE/'payload/runs/rehearsals_pva/20260909-172103-268459-mppi-wave'
tensor=lambda x:torch.as_tensor(x,dtype=torch.float64)


def load_case(path,cfg=None,meta=None):
    file=path/'rehearsal.npz' if path.is_dir() else path
    with np.load(file) as z:a={k:z[k].copy() for k in z.files}
    cfg=cfg or read_json(path/'settings.json');meta=meta or read_json(path/'rehearsal.json')
    t=a['prediction_time_s'];use=t<=meta['whip_end_s']+1e-9
    q=a['cable_positions_m'][use];v=a['cable_velocities_m_s'][use];t=t[use]
    p=a['origin_positions_m'][use];pv=a['origin_velocities_m_s'][use]
    length=np.linalg.norm(np.diff(q[0],axis=0),axis=-1);material=np.r_[0,np.cumsum(length)]/length.sum()
    return dict(q=q,v=v,p=p,pv=pv,t=t,material=material,length=length.sum(),cfg=cfg)


def encounter(case):
    q=case['q'];cfg=case['cfg'];target=np.array(cfg['launch']['target_m'])
    entry=sphere_entry(tensor(q[:-1]),tensor(q[1:]),tensor(target)[None].expand(len(q)-1,-1),cfg['task']['target_radius_m']).numpy()
    contacts=np.flatnonzero(np.isfinite(entry).any(1));touch=len(contacts)>0
    if touch:i=contacts[0];f=entry[i].min();tip_first=entry[i,-1]<entry[i,:-1].min()
    else:
        delta=np.diff(q[:,-1],axis=0)
        frac=np.clip(((target-q[:-1,-1])*delta).sum(-1)/np.maximum((delta**2).sum(-1),1e-20),0,1)
        distance=np.linalg.norm(q[:-1,-1]+frac[:,None]*delta-target,axis=-1)
        i=distance.argmin();f=frac[i];tip_first=False
    interp=lambda k:case[k][i]+f*(case[k][i+1]-case[k][i])
    end=interp('t');eq=interp('q');direction=np.array(cfg['task']['strike_direction'])
    forward=(case['p'][:i+1]-cfg['launch']['origin_m'])@direction
    # Include only the same interpolated contact position in the pull peak.
    pos=(interp('p')-cfg['launch']['origin_m'])@direction
    backward=max(forward.max(),pos)-pos
    return dict(end=end,q=eq,distance=np.linalg.norm(eq[-1]-target),tip_v=interp('v')[-1],
        drone_v=interp('pv'),backward=backward,reach=(eq[-1]-eq[0])@direction/case['length'],
        direction=direction,tip_first=tip_first,contact=touch)


def reference_from(case,event):
    times=np.linspace(0,event['end'],21);flat=case['q'].reshape(len(case['t']),-1)
    q=np.stack([np.interp(times,case['t'],column) for column in flat.T],-1).reshape(21,-1,3)
    return material_resample(tensor((q-q[:,:1])/case['length']),tensor(case['material']))


def evaluate(case,reference):
    event=encounter(case);w=WAVE_OBJECTIVE
    fold=fold_quality(tensor(case['q'])[None],tensor(case['t']),tensor([event['end']]),tensor(event['q'])[None],
        tensor(case['material']),case['length'],reference,w['shape_sigma'],w['tangent_sigma'])[0]
    keys=('distance','tip_v','drone_v','backward','reach','direction','tip_first','contact')
    values=[torch.as_tensor(event[k])[None] if k in ('tip_first','contact') else tensor(event[k])[None] for k in keys]
    contact,cast,proximity=encounter_quality(*values,case['cfg']['task'],w['proximity_scale_m'])
    terms=dict(contact=float(w['contact']*contact[0]),fold=float(w['fold']*fold*proximity[0]),
        cast=float(w['cast']*cast[0]),miss=float(-w['miss']*(1-proximity[0])))
    return dict(fold_quality=float(fold),task_score=sum(terms.values()),terms=terms,
        encounter_s=float(event['end']),tip_distance_m=float(event['distance']),contact=bool(event['contact']))


def main():
    AUDIT.mkdir(parents=True,exist_ok=False)
    cases={k:load_case(p) for k,p in [('preferred_far',PREFERRED),('preferred_near',NEAR),('current_sweep',CURRENT)]}
    diagnostic=ROOT/'runs/audits/preferred-whip-review-20260910/old_commands_new_M0'
    cases['old_commands_new_M0']=load_case(diagnostic.with_suffix('.npz'),read_json(CURRENT/'settings.json'),read_json(diagnostic.with_suffix('.json')))
    reference=reference_from(cases['preferred_far'],encounter(cases['preferred_far']))
    np.savez_compressed(AUDIT/'wave_reference.npz',shape=reference.numpy())
    base=cases['preferred_far'];q=base['q'];relative=q-q[:,:1]
    from copy import deepcopy
    def add(name,shape):
        c=deepcopy(base);c['q']=q[:,:1]+shape;c['v']=np.gradient(c['q'],c['t'],axis=0);cases[name]=c
    angles=np.diff(relative,axis=1);unit=angles/np.linalg.norm(angles,axis=-1,keepdims=True)
    peak=np.arccos(np.clip((unit[:,:-1]*unit[:,1:]).sum(-1),-1,1)).sum(-1).argmax()
    add('static_fold',np.repeat(relative[peak:peak+1],len(q),axis=0))
    add('reflected_time',relative[::-1].copy())
    shuffle=np.random.default_rng(20260910).permutation(len(q));shuffle[0]=0
    add('shuffled_frames',relative[shuffle])
    add('weak_bend',relative*.25+relative[:1]*.75)
    direction=relative[:,-1]/np.linalg.norm(relative[:,-1],axis=-1,keepdims=True)
    add('rigid_straight_swing',direction[:,None]*base['material'][None,:,None]*base['length'])
    cases['off_target_fold']=deepcopy(base);cases['off_target_fold']['cfg']['launch']['target_m'][1]+=2.
    rows={name:evaluate(case,reference) for name,case in cases.items()}
    atomic_json(AUDIT/'ranking.json',rows)
    # Invariances compare the same underlying polyline and physical timeline.
    translated=deepcopy(base)
    shift=np.array([3.,-2.,.7])
    for key in ('q','p'):translated[key]+=shift
    for key in ('origin_m','target_m'):translated['cfg']['launch'][key]=(np.array(translated['cfg']['launch'][key])+shift).tolist()
    shifted=evaluate(translated,reference)
    refined=deepcopy(base)
    for key in ('q','v'):
        data=base[key];refined[key]=np.stack([data[:,:-1],(data[:,:-1]+data[:,1:])/2],2).reshape(len(data),-1,3)
        refined[key]=np.concatenate([refined[key],data[:,-1:]],1)
    material=base['material'];refined['material']=np.r_[np.stack([material[:-1],(material[:-1]+material[1:])/2],1).ravel(),1.]
    refined_result=evaluate(refined,reference)
    stretched=deepcopy(base);stretched['t']*=1.12;stretched['v']/=1.12;stretched['pv']/=1.12
    timed=evaluate(stretched,reference)
    invariance=dict(translation_score_error=abs(shifted['task_score']-rows['preferred_far']['task_score']),
        subdivision_fold_error=abs(refined_result['fold_quality']-rows['preferred_far']['fold_quality']),
        time_scaling_fold_error=abs(timed['fold_quality']-rows['preferred_far']['fold_quality']))
    atomic_json(AUDIT/'invariances.json',invariance)
    negatives=('static_fold','reflected_time','shuffled_frames','weak_bend','rigid_straight_swing')
    passed=(all(rows[k]['fold_quality']>rows['current_sweep']['fold_quality'] for k in ('preferred_far','preferred_near'))
        and all(rows['preferred_near']['fold_quality']>rows[k]['fold_quality'] for k in negatives)
        and rows['preferred_near']['task_score']>rows['current_sweep']['task_score']
        and rows['current_sweep']['task_score']>rows['off_target_fold']['task_score']
        and max(invariance.values())<1e-8)
    atomic_json(AUDIT/'ranking_check.json',dict(status='passed' if passed else 'failed',
        scope='Fixed simulated preference examples, not physical validation; task terms shown without command/recovery costs',
        objective=WAVE_OBJECTIVE,reference_source=str(PREFERRED/'rehearsal.npz'),
        reference_source_sha256=sha256_file(PREFERRED/'rehearsal.npz'),reference_sha256=sha256_file(AUDIT/'wave_reference.npz')))
    print(rows);print(invariance);assert passed,'Fixed objective preference checks failed; no optimization authorized by preflight'

if __name__=='__main__':main()
