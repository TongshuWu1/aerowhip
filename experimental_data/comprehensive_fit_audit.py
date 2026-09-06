"""One-factor fitting diagnostics on existing development recordings.

These are sensitivity experiments, not automatic model calibration. Candidate
geometry is estimated from training recordings only. Protected test data is never
opened. One oracle boundary explicitly uses a future cable marker for diagnosis.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from simulator.cable import CableConfiguration, DderModel, DderState
from experimental_data.force_dataset import _normalized_rotations
from experimental_data.state_initialization import endpoint_velocity
from experimental_data.io import atomic_json, sha256_file


def geometry_summary(data, model_payload):
    rest=np.array(model_payload['cable']['marker_interval_lengths_m'])
    rows={}; offset_estimates=[]; length_estimates=[]
    for name,d in data.items():
        a=d['force'];p=d['processed'];q=a['cable_node_position_world_m'][:,[0,*range(2,12)]]
        v=a['cable_node_velocity_world_m_s'];valid=a['state_valid']
        chords=np.linalg.norm(np.diff(q,axis=1),axis=-1)
        quiet=valid & (np.linalg.norm(v[:,0],axis=1)<.05) & (np.linalg.norm(v[:,-1]-v[:,0],axis=1)<.1)
        c1=q[:,1]; tangent=q[:,2]-c1;tangent/=np.linalg.norm(tangent,axis=1)[:,None]
        inferred=c1-rest[0]*tangent-p['uav_position_m']
        body_offset=np.einsum('tji,tj->ti',d['rotation'],inferred)
        median_offset=np.median(body_offset[quiet],axis=0) if quiet.any() else None
        median_lengths=np.median(chords[quiet],axis=0) if quiet.any() else None
        rows[name]=dict(role=d['role'],valid_frames=int(valid.sum()),quiet_frames=int(quiet.sum()),
            chord_median_m=np.median(chords[valid],axis=0).tolist(),
            chord_p05_m=np.percentile(chords[valid],5,axis=0).tolist(),
            chord_p95_m=np.percentile(chords[valid],95,axis=0).tolist(),
            chord_exceeds_arc_plus_2mm_percent=(100*(chords[valid]>rest+.002).mean(0)).tolist(),
            quiet_chord_median_m=median_lengths.tolist() if median_lengths is not None else None,
            quiet_straight_first_span_inferred_body_offset_m=median_offset.tolist() if median_offset is not None else None)
        if d['role']=='training' and quiet.sum()>=20:
            offset_estimates.append(median_offset);length_estimates.append(median_lengths)
    # Equal-take median, not a pooled-frame estimate dominated by long quiet takes.
    offset=np.median(offset_estimates,axis=0)
    lengths=np.median(length_estimates,axis=0);lengths[0]=rest[0]
    return dict(configured_arc_lengths_m=rest.tolist(),per_take=rows,
        candidate_offset_body_m=offset.tolist(),candidate_effective_lengths_m=lengths.tolist(),
        caveats=['Offset estimate extrapolates the c1-to-c2 tangent over a straight first span during quiet motion.',
                 'This is a conditional geometric hypothesis, not independent attachment metrology.',
                 'Measured chords are not generally arc lengths; effective-length variants are diagnostic only.'])


def reconstruct(sites,cable,kind='linear'):
    material=np.r_[0,np.cumsum(cable.marker_interval_lengths_m)]
    nodes=np.r_[0,np.cumsum(cable.rest_lengths_m)]
    if kind=='cubic':
        h=np.diff(material);matrix=np.eye(len(material))
        rhs=np.zeros_like(sites)
        for i in range(1,len(material)-1):
            matrix[i]=0;matrix[i,i-1]=h[i-1];matrix[i,i]=2*(h[i-1]+h[i]);matrix[i,i+1]=h[i]
            rhs[:,i]=6*((sites[:,i+1]-sites[:,i])/h[i]-(sites[:,i]-sites[:,i-1])/h[i-1])
        second=np.linalg.solve(matrix,rhs.transpose(1,0,2).reshape(len(material),-1))
        second=second.reshape(len(material),len(sites),3).transpose(1,0,2)
        i=np.minimum(np.searchsorted(material,nodes,side='right')-1,len(h)-1)
        left=material[i+1]-nodes;right=nodes-material[i];width=h[i]
        return (second[:,i]*((left**3)/(6*width))[None,:,None]
            +second[:,i+1]*((right**3)/(6*width))[None,:,None]
            +(sites[:,i]-second[:,i]*(width**2/6)[None,:,None])*(left/width)[None,:,None]
            +(sites[:,i+1]-second[:,i+1]*(width**2/6)[None,:,None])*(right/width)[None,:,None])
    result=np.empty((len(sites),cable.node_count,3))
    result[:,0]=sites[:,0];idx=0
    for i,n in enumerate(cable.interval_subdivisions):
        for j in range(1,n+1):
            idx+=1;alpha=j/n
            result[:,idx]=(1-alpha)*sites[:,i]+alpha*sites[:,i+1]
    return result


def variants(geometry):
    offset=geometry['candidate_offset_body_m']
    return [
        dict(name='reference'),
        dict(name='substeps_6',substeps=6),dict(name='substeps_12',substeps=12),
        dict(name='constraints_12',constraint_iterations=12),
        dict(name='projection_1',projection_passes=1),dict(name='projection_24',projection_passes=24),
        dict(name='velocity_7',velocity_samples=7),dict(name='velocity_21',velocity_samples=21),
        dict(name='shape_endpoint_polynomial',smooth_shape=True),
        dict(name='offset_45mm',offset=[0,0,-.045]),dict(name='offset_65mm',offset=[0,0,-.065]),
        dict(name='offset_xy_training',offset=[offset[0],offset[1],-.055]),
        dict(name='offset_xyz_training',offset=offset),
        dict(name='two_interval_lengths',interval_corrections={1:.085,7:.100}),
        dict(name='effective_lengths_training',lengths=geometry['candidate_effective_lengths_m']),
        dict(name='offset_and_effective_lengths',offset=offset,lengths=geometry['candidate_effective_lengths_m']),
        dict(name='body_tangent_clamp',boundary='body_clamp'),
        dict(name='offset_body_tangent_clamp',boundary='body_clamp',offset=offset),
        dict(name='oracle_c1_tangent',boundary='oracle_tangent'),
        dict(name='mesh_21_linear',subdivisions=2,substeps=6),
        dict(name='mesh_21_cubic',subdivisions=2,substeps=6,interpolation='cubic'),
        dict(name='mesh_41_cubic',subdivisions=4,substeps=12,interpolation='cubic'),
        dict(name='drag_0p1',drag=.1),dict(name='drag_0p5',drag=.5),dict(name='drag_1',drag=1.),
        dict(name='marker_mass_half',marker_mass_scale=.5),dict(name='marker_mass_1p5',marker_mass_scale=1.5),
        dict(name='bare_mass_half',bare_mass_scale=.5),dict(name='bare_mass_1p5',bare_mass_scale=1.5),
        dict(name='previous_active_parameters',active_parameters=True),
        dict(name='substeps_24',substeps=24),
        dict(name='offset_clamp_substeps12',boundary='body_clamp',offset=offset,substeps=12),
        dict(name='offset_clamp_substeps24',boundary='body_clamp',offset=offset,substeps=24),
        dict(name='offset_clamp_first4',boundary='body_clamp',offset=offset,substeps=12,subdivisions=[4]+[1]*9),
        dict(name='offset_clamp_first8',boundary='body_clamp',offset=offset,substeps=12,subdivisions=[8]+[1]*9),
        dict(name='offset_pivot_first8',offset=offset,substeps=12,subdivisions=[8]+[1]*9),
        dict(name='weak_damping',Cb=1e-8),dict(name='strong_bending',EI=.00028),
        dict(name='previous_neural_residual',residual=True),
        dict(name='offset_clamp_drag0p1',boundary='body_clamp',offset=offset,substeps=12,drag=.1),
        dict(name='offset_clamp_drag0p3',boundary='body_clamp',offset=offset,substeps=12,drag=.3),
        dict(name='offset_clamp_drag0p5',boundary='body_clamp',offset=offset,substeps=12,drag=.5),
        dict(name='offset_pivot_drag0p1',offset=offset,substeps=12,drag=.1),
        dict(name='offset_pivot_drag0p3',offset=offset,substeps=12,drag=.3),
        dict(name='offset_pivot_drag0p5',offset=offset,substeps=12,drag=.5),
        dict(name='offset_short_clamp_drag0p1',boundary='body_clamp',offset=offset,substeps=12,subdivisions=[8]+[1]*9,drag=.1),
        dict(name='offset_short_clamp_drag0p3',boundary='body_clamp',offset=offset,substeps=12,subdivisions=[8]+[1]*9,drag=.3),
        dict(name='strong_EI_0p001_fine',EI=.001,substeps=12),
        dict(name='strong_EI_0p004_fine',EI=.004,substeps=12),
        dict(name='offset_pivot_drag0p3_substeps24',offset=offset,substeps=24,drag=.3),
        dict(name='offset_clamp_drag0p3_substeps24',boundary='body_clamp',offset=offset,substeps=24,drag=.3),
    ]


def assemble(data,settings,model_payload,fit):
    payload=deepcopy(model_payload['cable'])
    for key in ['substeps','constraint_iterations']:
        if key in settings:payload[key]=settings[key]
    if 'subdivisions' in settings:payload['segments_per_marker_interval']=settings['subdivisions']
    if 'lengths' in settings:payload['marker_interval_lengths_m']=settings['lengths']
    for i,v in settings.get('interval_corrections',{}).items():payload['marker_interval_lengths_m'][i]=v
    payload['moving_marker_masses_kg']=[m*settings.get('marker_mass_scale',1) for m in payload['moving_marker_masses_kg']]
    payload['bare_cable_mass_kg']*=settings.get('bare_mass_scale',1)
    cable=CableConfiguration.from_mapping(payload)
    ei,cb=(payload['EI_n_m2'],payload['Cb_n_m2_s']) if settings.get('active_parameters') else (fit['EI_n_m2'],fit['Cb_n_m2_s'])
    ei=settings.get('EI',ei);cb=settings.get('Cb',cb)
    params=replace(cable.dder_parameters(EI=ei,Cb=cb),external_drag_s_inv=settings.get('drag',0.))
    model=DderModel(params)
    if settings.get('residual'):
        from simulator.cable.residual import FrozenMotionResidual
        checkpoint=Path(__file__).resolve().parents[1]/'data/workflow_jobs/20260905-061423-970034-fit/residual_candidate.pt'
        model.motion_residual=FrozenMotionResidual(str(checkpoint),sha256_file(checkpoint))
    offset=np.array(settings.get('offset',model_payload['recorded_data']['optitrack_to_attachment_offset_body_m']))
    histories=[];boundaries=[];truth=[];records=[]
    clamp=settings.get('boundary','pivot')!='pivot';mask=(2,0) if clamp else (1,0)
    for name,d in data.items():
        p=d['processed'];r=d['rotation'];root=p['uav_position_m']+np.einsum('tij,j->ti',r,offset)
        sites=np.concatenate((root[:,None],p['cable_marker_positions_m']),axis=1)
        # Reconstruct per selected window only: invalid frames elsewhere cannot
        # break spline reconstruction for an otherwise valid evaluation window.
        for start,original in d['windows']:
            after=d.get('horizon_steps',200)
            nodes=reconstruct(sites[start-20:start+after+1],cable,settings.get('interpolation','linear'))
            if clamp:
                if settings['boundary']=='body_clamp':direction=-r[start-20:start+after+1,:,2]
                else:
                    direction=sites[start-20:start+after+1,1]-root[start-20:start+after+1]
                    direction/=np.linalg.norm(direction,axis=1)[:,None]
                nodes[:,1]=root[start-20:start+after+1]+cable.rest_lengths_m[0]*direction
            histories.append(nodes[:21]);boundaries.append(nodes[20:,:mask[0]])
            truth.append(p['cable_marker_positions_m'][start:start+after+1])
            records.append(dict(take=name,role=d['role'],start_frame=start,intensity=original['intensity']))
    history=torch.tensor(np.stack(histories),dtype=torch.float64)
    q=history[:,-1].clone()
    if settings.get('smooth_shape'):
        x=torch.linspace(-1,0,11,dtype=q.dtype);design=torch.stack((x*0+1,x,x*x),-1)
        smooth=torch.einsum('t,btnc->bnc',torch.linalg.pinv(design)[0],history[:,-11:])
        smooth[:,:mask[0]]=q[:,:mask[0]];q=smooth
    v=endpoint_velocity(history,.01,settings.get('velocity_samples',11))
    pinned=q[:,:mask[0]].clone();pv=v[:,:mask[0]].clone()
    with torch.no_grad():
        for _ in range(settings.get('projection_passes',8)):
            q=model.project_lengths(q,pinned,pinned_endpoints=mask)
        v=model.project_velocities(q,v,pv,pinned_endpoints=mask)
    return cable,model,mask,q,v,torch.tensor(np.stack(boundaries),dtype=torch.float64),np.stack(truth),records


@torch.no_grad()
def evaluate_variant(data,settings,model_payload,fit,device):
    begin=time.perf_counter()
    cable,model,mask,q,v,boundaries,truth,records=assemble(data,settings,model_payload,fit)
    q=q.to(device);v=v.to(device);boundaries=boundaries.to(device)
    constants=model.runtime_constants(q);dt=q.new_full((len(q),),.01)
    state=DderState(q,v);nodes=list(cable.marker_node_indices[1:]);predicted=[q[:,nodes].cpu().numpy()]
    for step in range(1,boundaries.shape[1]):
        state=model.step_runtime(state,boundaries[:,step],dt,constants,
            pinned_endpoints=mask,iterative_damping=False,dense_constraint_solve=True,analytic_bending=True)
        predicted.append(state.positions_m[:,nodes].cpu().numpy())
    predicted=np.stack(predicted,axis=1);delta=predicted-truth;sq=np.sum(delta**2,axis=-1)
    finite=np.isfinite(sq).all((1,2));results=[]
    if not finite.all():
        raise ValueError(f'Nonfinite prediction in {int((~finite).sum())} windows.')
    for i,record in enumerate(records):
        results.append(dict(**record,variant=settings['name'],finite=bool(finite[i]),
            initialization_rmse_m=float(np.sqrt(sq[i,0].mean())),
            marker_rmse_m=float(np.sqrt(sq[i,1:].mean())),tip_rmse_m=float(np.sqrt(sq[i,1:,-1].mean())),
            c2_to_c10_rmse_m=float(np.sqrt(sq[i,1:,1:].mean())),
            loss_m=float((.002*(np.sqrt(1+sq[i,1:]/.002**2)-1)).mean()),
            per_marker_rmse_m=np.sqrt(sq[i,1:].mean(0)).tolist(),
            mean_tip_bias_xyz_m=delta[i,1:,-1].mean(0).tolist(),
            tip_error_at_0p1s_m=float(np.sqrt(sq[i,10,-1])),tip_error_at_1s_m=float(np.sqrt(sq[i,100,-1])),
            tip_error_at_2s_m=float(np.sqrt(sq[i,200,-1]))))
    return results,dict(runtime_s=time.perf_counter()-begin,node_count=cable.node_count,
        substeps=cable.substeps,constraint_iterations=cable.constraint_iterations,
        maximum_length_error_m=float((torch.linalg.vector_norm(torch.diff(state.positions_m,dim=1),dim=-1)-q.new_tensor(cable.rest_lengths_m)).abs().max())),{
        'prediction':predicted,'measured':truth}


def aggregate(rows):
    groups=[]
    for variant in sorted({r['variant'] for r in rows}):
        for role in ('training','validation'):
            records=[r for r in rows if r['variant']==variant and r['role']==role]
            per_take={}
            for name in sorted({r['take'] for r in records}):
                sub=[r for r in records if r['take']==name]
                metrics={k:float(np.sqrt(np.mean([r[k]**2 for r in sub]))) for k in
                    ['marker_rmse_m','tip_rmse_m','c2_to_c10_rmse_m','initialization_rmse_m']}
                per_take[name]=dict(**metrics,loss_m=float(np.mean([r['loss_m'] for r in sub])),windows=len(sub))
            groups.append(dict(variant=variant,role=role,per_take=per_take,
                **{key:float(np.mean([r[key] for r in per_take.values()])) for key in
                   ['marker_rmse_m','tip_rmse_m','c2_to_c10_rmse_m','initialization_rmse_m','loss_m']}))
    return groups


def run(source,benchmark,output,only=None,horizon=2.):
    output.mkdir(parents=True,exist_ok=False);atomic_json(output/'status.json',dict(status='RUNNING'))
    torch.set_num_threads(1);root=Path(__file__).resolve().parents[1]
    p=json.loads((source/'model.json').read_text());fit=json.loads((benchmark/'protocol.json').read_text())['physical_parameters']
    if json.loads((benchmark/'protocol.json').read_text())['horizon_s'] != 2.:
        raise ValueError('This audit requires the saved two-second benchmark.')
    roles=json.loads((source/'dataset_manifest.json').read_text())['takes']
    with (benchmark/'windows.csv').open() as f:windows=[r for r in csv.DictReader(f) if r['method']=='causal_polynomial']
    data={};hashes={}
    for name in sorted({r['take'] for r in windows}):
        if roles[name]['role'] not in ('training','validation'):raise ValueError('Protected take requested.')
        arrays={}
        for key,path in [('force',source/'force_takes'/name/'take.npz'),('processed',root/'data/processed_takes'/name/'take.npz')]:
            with np.load(path) as a:arrays[key]={k:a[k] for k in a.files}
            hashes[str(path)]=sha256_file(path)
        np.testing.assert_allclose(arrays['force']['time_s'],arrays['processed']['time_s'])
        np.testing.assert_allclose(np.diff(arrays['force']['time_s']),.01,atol=1e-10,rtol=1e-8)
        rotation,valid=_normalized_rotations(arrays['processed']['uav_orientation_xyzw'])
        after=round(horizon/.01)
        selected=[(int(r['start_frame']),r) for r in windows if r['take']==name]
        selected=[(s,r) for s,r in selected if s+after<len(arrays['force']['state_valid']) and arrays['force']['state_valid'][s:s+after+1].all()]
        data[name]=dict(**arrays,rotation=rotation,role=roles[name]['role'],windows=selected,horizon_steps=after)
    geometry=geometry_summary(data,p);atomic_json(output/'geometry.json',geometry)
    planned=variants(geometry)
    if only:planned=[v for v in planned if v['name'] in only]
    if any(v.get('residual') for v in planned):
        checkpoint=root/'data/workflow_jobs/20260905-061423-970034-fit/residual_candidate.pt'
        hashes[str(checkpoint)]=sha256_file(checkpoint)
    files=['experimental_data/comprehensive_fit_audit.py','experimental_data/state_initialization.py','simulator/cable/dder.py','simulator/cable/config.py','simulator/cable/residual.py']
    for name in files:
        target=output/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/name,target);hashes[name]=sha256_file(target)
    atomic_json(output/'protocol.json',dict(variants=planned,physical_parameters=fit,horizon_s=horizon,
        windows_per_take={name:len(d['windows']) for name,d in data.items()},
        source_job=str(source),benchmark=str(benchmark),input_hashes=hashes,
        limitations=['All seven recordings are development data; protected test is excluded.',
            'Most variants change one factor; combined variants are explicitly named.',
            'Geometry estimates use training quiet frames only; diagnostic effective lengths are not material metrology.',
            'Oracle c1 tangent receives future c1 direction; it is not a deployment-valid comparison.',
            'Offsets, masses, drag and clamps are hypotheses, not established corrections.',
            'Uses saved two-second benchmark starts, retaining only those with complete requested prediction duration.',
            'All variants within a run share windows; no material parameters or residual weights are optimized.',
            'The named previous_neural_residual variant loads the prior frozen residual with its original physical parameters.',
            'Repeated validation inspection makes this model development, not final test evidence.']))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');rows=[];timing={};failed={}
    for variant in planned:
        print(f"Testing {variant['name']}",flush=True)
        try:
            result,info,traces=evaluate_variant(data,variant,p,fit,device)
            rows.extend(result);timing[variant['name']]=info
            if variant['name'] in ('reference','offset_xyz_training','body_tangent_clamp','mesh_21_cubic',
                                  'offset_clamp_substeps12','offset_clamp_first8','strong_bending','weak_damping'):
                np.savez_compressed(output/f"{variant['name']}_traces.npz",**traces)
            atomic_json(output/'windows.json',rows)
            atomic_json(output/'result.json',dict(groups=aggregate(rows),timing=timing,failed=failed))
            print(f"Completed {variant['name']} in {info['runtime_s']:.1f}s",flush=True)
        except Exception as e:
            failed[variant['name']]=str(e);print(f"Failed {variant['name']}: {e}",flush=True)
        atomic_json(output/'progress.json',dict(completed=list(timing),failed=failed))
    atomic_json(output/'result.json',dict(groups=aggregate(rows),timing=timing,failed=failed))
    flat=aggregate(rows)
    with (output/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=[k for k in flat[0] if k!='per_take']);w.writeheader()
        w.writerows([{k:v for k,v in r.items() if k!='per_take'} for r in flat])
    atomic_json(output/'status.json',dict(status='COMPLETED_WITH_FAILURES' if failed else 'COMPLETED'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--benchmark',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--only',nargs='+')
    parser.add_argument('--horizon',type=float,default=2.)
    a=parser.parse_args();run(a.source.resolve(),a.benchmark.resolve(),a.output.resolve(),a.only,a.horizon)
