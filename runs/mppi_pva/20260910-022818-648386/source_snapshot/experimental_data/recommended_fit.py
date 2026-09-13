"""UI calibration: fixed material parameters, constrained geometry, training-only drag search."""
from copy import deepcopy
import csv
from pathlib import Path
import shutil
import numpy as np
import torch
from .constrained_geometry import load_geometry_data,fit_offset,consistency
from .constrained_identification import load_takes,population,evaluate_together,read
from .cable_fit import contiguous_window_starts
from .io import atomic_json,sha256_file,canonical_json_hash


def select_windows(job, maximum=12):
    manifest=read(job/'dataset_manifest.json')['takes'];candidates={};training_speeds=[]
    for name,row in manifest.items():
        if not row.get('enabled',True) or row['role'] not in ('training','validation'):continue
        with np.load(job/'force_takes'/name/'take.npz') as data:
            valid=data['state_valid'];velocity=data['cable_node_velocity_world_m_s']
            starts=[s+20 for s in contiguous_window_starts(valid,horizon_steps=220,stride_steps=100)]
            if not starts:continue
            speed=np.linalg.norm(velocity[starts,-1]-velocity[starts,0],axis=-1)
            candidates[name]=(row['role'],starts,speed)
            if row['role']=='training':training_speeds.extend(speed)
    if not training_speeds or not any(v[0]=='validation' for v in candidates.values()):
        raise ValueError('Enable at least one fit take and one validation take with 2.2 seconds of valid motion.')
    cutoffs=np.quantile(training_speeds,[1/3,2/3]);rows=[]
    for name,(role,starts,speed) in candidates.items():
        bins=np.searchsorted(cutoffs,speed)
        for b in range(3):
            ix=np.flatnonzero(bins==b)
            if not len(ix):continue
            chosen=ix[np.linspace(0,len(ix)-1,min(maximum//3,len(ix)),dtype=int)]
            rows.extend(dict(take=name,role=role,start_frame=starts[i],intensity=['low','medium','high'][b],
                             method='causal_polynomial') for i in chosen)
    benchmark=job/'benchmark';benchmark.mkdir()
    with (benchmark/'windows.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_json(benchmark/'protocol.json',dict(horizon_s=2.,speed_cutoffs_m_s=cutoffs.tolist(),
        selection='Training speed terciles, evenly spaced starts within each take and bin',maximum_windows_per_take=maximum))
    return benchmark


def fit_recommended(job,root):
    job,root=Path(job),Path(root);torch.set_num_threads(1)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    def progress(stage,label,current=0,total=0):
        atomic_json(job/'progress.json',dict(stage=stage,label=label,update=current,total=total))
        print(label,flush=True)
    progress('geometry','Calibrating attachment geometry')
    for name,row in read(job/'dataset_manifest.json')['takes'].items():
        if row.get('enabled',True) and row['role'] in ('training','validation'):
            destination=job/'processed_takes'/name/'take.npz';destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(root/'data/processed_takes'/name/'take.npz',destination)
    data,hashes=load_geometry_data(job,root);starting=read(job/'model.json');candidate=deepcopy(starting)
    candidate.pop('motion_residual',None)
    height=-starting['recorded_data']['optitrack_to_attachment_offset_body_m'][2]
    length=starting['cable']['marker_interval_lengths_m'][0]
    if height<=0:raise ValueError('Attachment Z must be below the tracked drone origin (negative).')
    fitted=fit_offset(data,height=height,length=length)
    candidate['recorded_data']['optitrack_to_attachment_offset_body_m']=fitted['offset_body_m']
    candidate['cable']['substeps']=12
    atomic_json(job/'geometry_fit.json',dict(selected=fitted,
        before=consistency(data,starting['recorded_data']['optitrack_to_attachment_offset_body_m'],length),
        after=consistency(data,fitted['offset_body_m'],length)))
    benchmark=select_windows(job)
    cable,model,takes,intensities=load_takes(job,benchmark,candidate,device=device)
    training=[t for t in takes if t.role=='training']
    rates=sorted(set([0.,.1,.2,.3,.4,.5,.75,1.,starting['cable'].get('external_drag_s_inv',0.)]))
    params=[[candidate['cable']['EI_n_m2'],candidate['cable']['Cb_n_m2_s'],rate] for rate in rates]
    progress('drag','Fitting cable drag on fit takes',0,len(rates))
    scores,_=population(training,intensities,model,cable,params)
    chosen=int(scores.argmin());candidate['cable']['external_drag_s_inv']=rates[chosen]
    atomic_json(job/'drag_search.json',dict(rates_s_inv=rates,training_loss_m=scores.tolist(),selected=chosen))
    progress('validation','Checking 2-second and 5-second predictions')
    evaluation={}
    for horizon in (2.,5.):
        evaluation[str(horizon)]={}
        for label,payload in [('starting_inputs',starting),('recommended',candidate)]:
            c,m,ts,_=load_takes(job,benchmark,payload,horizon=horizon,device=device)
            if not all(any(t.role==role for t in ts) for role in ('training','validation')):
                if horizon==2.:raise ValueError('No usable validation windows.')
                evaluation.pop(str(horizon));break
            if label=='starting_inputs' and payload.get('motion_residual',{}).get('enabled'):
                from simulator.point_mass import ForceControlledPointCable
                m.motion_residual=ForceControlledPointCable.from_mapping(payload,root=root).dder.motion_residual
            p=torch.tensor([payload['cable']['EI_n_m2'],payload['cable']['Cb_n_m2_s'],
                            payload['cable'].get('external_drag_s_inv',0.)],dtype=torch.float64,device=device)
            metrics,prediction,truth=evaluate_together(ts,m,c,p)
            evaluation[str(horizon)][label]=metrics
            if horizon==2.:
                np.savez_compressed(job/f'{label}_2s_predictions.npz',prediction=prediction,measured=truth,
                    take=np.array([t.take_id for t in ts for _ in t.starts]),start_frame=np.array([s for t in ts for s in t.starts]))
        atomic_json(job/'evaluation.json',evaluation)
    candidate['force_accounting']['aerodynamic_drag']='effective_world_velocity_decay_on_free_cable_vertices'
    atomic_json(job/'recommended_model.json',candidate)
    source_hashes={}
    for name in ['experimental_data/'+p for p in ['recommended_fit.py','constrained_geometry.py','constrained_identification.py','state_initialization.py','differentiable_fit.py','comprehensive_fit_audit.py','force_dataset.py','cable_fit.py']]+['simulator/cable/dder.py','simulator/cable/config.py']:
        source=Path(__file__).resolve().parents[1]/name;dest=job/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest);source_hashes[name]=sha256_file(dest)
    atomic_json(job/'candidate_review.json',dict(schema='recommended_calibration_v1',recommended='recommended',
        model_sha256=canonical_json_hash(candidate),input_hashes=hashes,source_hashes=source_hashes,
        status='PROVISIONAL_RESEARCH_CANDIDATE',protected_test_used=False,
        method='Fixed EI/Cb; constrained lateral attachment offset; training-only external drag search',
        limitations=['Measured attachment input, not command-driven flight validation.',
                     'Height and first span fixed to fit inputs; quiet tangent extrapolation is approximate.',
                     'Short causal initialization; neural residual and long-horizon gradient fitting disabled.']))
    progress('complete','Fit complete — review validation before applying')
