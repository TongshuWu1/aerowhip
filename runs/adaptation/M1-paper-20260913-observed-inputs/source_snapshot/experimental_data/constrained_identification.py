"""Reproducible constrained-geometry, three-parameter, then residual calibration.

All selection uses fit takes. The protected recording is excluded. Output is a
reviewable candidate; this runner never replaces the active model.
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

from simulator.cable import CableConfiguration, DderModel
from simulator.cable.residual import MotionResidual, FrozenMotionResidual
from .cable_fit import PreparedTake
from .comprehensive_fit_audit import assemble
from .differentiable_fit import rollout, evaluate, save_weights, subset
from .force_dataset import _normalized_rotations
from .io import atomic_json, sha256_file
from .state_initialization import causal_state

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def load_takes(source, benchmark, payload, horizon=2., device='cpu'):
    with (benchmark/'windows.csv').open() as f:
        rows=[r for r in csv.DictReader(f) if r['method']=='causal_polynomial']
    roles=read(source/'dataset_manifest.json')['takes']
    data={}
    for name in sorted({r['take'] for r in rows}):
        if roles[name]['role'] not in ('training','validation'):
            raise ValueError('Protected data cannot be used for candidate calibration.')
        processed=source/'processed_takes' if (source/'processed_takes').exists() else ROOT/'data/processed_takes'
        with np.load(processed/name/'take.npz') as p, np.load(source/'force_takes'/name/'take.npz') as f:
            arrays={k:p[k] for k in p.files}
            np.testing.assert_allclose(p['time_s'],f['time_s'])
            rotation,valid_r=_normalized_rotations(p['uav_orientation_xyzw'])
            valid=f['state_valid'] & valid_r
            after=round(horizon/.01)
            selected=[(int(r['start_frame']),r) for r in rows if r['take']==name]
            selected=[(s,r) for s,r in selected if s>=20 and s+after<len(valid) and valid[s-20:s+after+1].all()]
            if not selected:
                continue
            data[name]=dict(processed=arrays,rotation=rotation,role=roles[name]['role'],
                            windows=selected,horizon_steps=after)
    fit=dict(EI_n_m2=payload['cable']['EI_n_m2'],Cb_n_m2_s=payload['cable']['Cb_n_m2_s'])
    if not data:
        cable=CableConfiguration.from_mapping(payload['cable'])
        return cable,DderModel(cable.dder_parameters()),[],{}
    cable,model,mask,q,v,b,truth,records=assemble(data,{'name':'calibration'},payload,fit)
    takes=[];intensities={}
    for name in data:
        indices=[i for i,r in enumerate(records) if r['take']==name]
        initial_rmse=float(np.sqrt(np.mean(np.sum((q.numpy()[indices][:,cable.marker_node_indices[1:]]-truth[indices,0])**2,axis=-1))))
        takes.append(PreparedTake(name,data[name]['role'],q[indices].to(device),v[indices].to(device),
            b[indices,:,0].to(device),torch.tensor(truth[indices],dtype=torch.float64,device=device),
            tuple(records[i]['start_frame'] for i in indices),initial_rmse,.01))
        intensities[name]=[records[i]['intensity'] for i in indices]
    return cable,model,takes,intensities


def combine(takes):
    return replace(takes[0],take_id='combined',
        initial_positions_m=torch.cat([t.initial_positions_m for t in takes]),
        initial_velocities_m_s=torch.cat([t.initial_velocities_m_s for t in takes]),
        root_positions_m=torch.cat([t.root_positions_m for t in takes]),
        measured_marker_positions_m=torch.cat([t.measured_marker_positions_m for t in takes]),
        starts=tuple(s for t in takes for s in t.starts))


def weights(takes,intensities):
    values=[]
    for t in takes:
        labels=intensities[t.take_id];bins=set(labels)
        values.extend(1/(len(takes)*len(bins)*labels.count(label)) for label in labels)
    return takes[0].initial_positions_m.new_tensor(values)


@torch.no_grad()
def population(takes,intensities,model,cable,candidates):
    take=combine(takes);count=len(candidates);n=take.window_count
    def repeat(x):return x.repeat_interleave(count,dim=0)
    expanded=replace(take,initial_positions_m=repeat(take.initial_positions_m),
        initial_velocities_m_s=repeat(take.initial_velocities_m_s),root_positions_m=repeat(take.root_positions_m),
        measured_marker_positions_m=repeat(take.measured_marker_positions_m))
    parameters=take.initial_positions_m.new_tensor(candidates).repeat(n,1).T
    predicted=rollout(expanded,model,cable,parameters)
    sq=(predicted[:,1:]-expanded.measured_marker_positions_m[:,1:]).square().sum(-1)
    loss=(.002*(torch.sqrt(1+sq/.002**2)-1)).mean((1,2)).reshape(n,count)
    if not torch.isfinite(loss).all():raise ValueError('Nonfinite population loss.')
    return (weights(takes,intensities)[:,None]*loss).sum(0).cpu().numpy(),loss.cpu().numpy()


def initialize(output,source,benchmark,geometry):
    output.mkdir(parents=True,exist_ok=False)
    payload=read(source/'model.json');g=read(geometry/'result.json')
    payload['recorded_data']['optitrack_to_attachment_offset_body_m']=g['selected']['offset_body_m']
    payload['cable']['marker_interval_lengths_m'][0]=g['selected']['first_span_m']
    payload['cable']['substeps']=12
    previous=read(benchmark/'protocol.json')['physical_parameters']
    payload['cable'].update(previous,external_drag_s_inv=.3)
    atomic_json(output/'geometry_model.json',payload)
    shutil.copy2(geometry/'result.json',output/'geometry_fit.json')
    files=['experimental_data/constrained_geometry.py','experimental_data/constrained_identification.py',
           'experimental_data/differentiable_fit.py','experimental_data/comprehensive_fit_audit.py',
           'experimental_data/state_initialization.py','simulator/cable/dder.py',
           'simulator/cable/config.py','simulator/cable/residual.py']
    hashes={}
    for name in files:
        target=output/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target);hashes[name]=sha256_file(target)
    atomic_json(output/'protocol.json',dict(source=str(source),benchmark=str(benchmark),geometry=str(geometry),
        source_hashes=hashes,input_hashes=g['protocol']['input_hashes'],seed=1729,
        selection='Training-only robust trajectory loss; equal take and equal nonempty initial-intensity bin weighting.',
        initialization='11-frame past-only quadratic velocity, measured positions with constraint projection',
        physics_grid=dict(EI=[1e-8,1e-6,1e-4,.001],Cb=[1e-6,1e-4,.01],drag=[0.,.1,.3,.5,.8]),
        parameter_bounds=[[1e-9,.001],[1e-8,.02],[1e-4,1.5]],
        refinement=dict(updates=12,horizons_s=[.5,1.,2.],updates_per_horizon=4,learning_rate=.05),
        residual=dict(updates=12,hidden=32,acceleration_limit_m_s2=.5,learning_rate=.0005,penalty=.001),
        evaluation_horizons_s=[2.,5.],substeps=12,dtype='float64',
        caveats=['Repeatedly inspected validation is development evidence.',
                 'Measured root trajectory is provided; this is conditional cable identification.',
                 'Geometry sensitivity ranges are assumptions; selected height and first span stay fixed.',
                 'A finite optimization budget does not establish convergence or material identifiability.']))


def grid(output,source,benchmark,payload,device):
    cable,model,takes,intensities=load_takes(source,benchmark,payload,device=device)
    train=[t for t in takes if t.role=='training']
    config=read(output/'protocol.json')['physics_grid']
    candidates=np.array([(a,b,c) for a in config['EI'] for b in config['Cb'] for c in config['drag']])
    candidates=np.vstack((candidates,[payload['cable']['EI_n_m2'],payload['cable']['Cb_n_m2_s'],.3]))
    losses=[];window_losses=[];started=time.perf_counter()
    for start in range(0,len(candidates),8):
        loss,individual=population(train,intensities,model,cable,candidates[start:start+8])
        losses.extend(loss.tolist());window_losses.append(individual)
        atomic_json(output/'grid_progress.json',dict(completed=len(losses),total=len(candidates),elapsed_s=time.perf_counter()-started))
        print(f'Grid {len(losses)}/{len(candidates)}; best loss {min(losses)*1000:.3f} mm',flush=True)
    chosen=int(np.argmin(losses))
    atomic_json(output/'grid.json',dict(candidates=candidates.tolist(),loss_m=losses,selected_index=chosen,
        selected_parameters=candidates[chosen].tolist(),windows={t.take_id:t.window_count for t in train}))
    np.savez_compressed(output/'grid_windows.npz',loss_m=np.concatenate(window_losses,axis=1),
        weights=weights(train,intensities).cpu().numpy(),take=np.array([t.take_id for t in train for _ in t.starts]))


def direct_refine(output,source,benchmark,payload,device):
    """Bounded pattern search on actual two-second rollouts; no unstable adjoints."""
    cable,model,takes,intensities=load_takes(source,benchmark,payload,device=device)
    train=[t for t in takes if t.role=='training']
    initial=read(output/'grid.json');best=np.array(initial['selected_parameters']);best_loss=min(initial['loss_m'])
    history=[];all_candidates=[];all_losses=[]
    # Explicit stability margin: EI <= .004 is below the audited .004865 limit
    # at 12 substeps for this fixed geometry. Cb and drag remain nonnegative.
    for iteration,(factor,drag_step) in enumerate([(3.,.1),(2.,.05),(1.4,.025)],1):
        candidates=np.unique(np.array([
            [np.clip(best[0]*a,1e-9,.004),np.clip(best[1]*b,1e-8,.02),np.clip(best[2]+c,0.,1.5)]
            for a in [1/factor,1.,factor] for b in [1/factor,1.,factor] for c in [-drag_step,0.,drag_step]]),axis=0)
        scores=[]
        for start in range(0,len(candidates),9):
            score,_=population(train,intensities,model,cable,candidates[start:start+9]);scores.extend(score.tolist())
        chosen=int(np.argmin(scores))
        if scores[chosen]<best_loss:best=candidates[chosen];best_loss=scores[chosen]
        history.append(dict(iteration=iteration,selected_parameters=best.tolist(),training_loss_m=best_loss,
                            candidates=candidates.tolist(),loss_m=scores))
        all_candidates.extend(candidates.tolist());all_losses.extend(scores)
        atomic_json(output/'direct_physics_history.json',history)
        atomic_json(output/'physics.json',dict(selected_parameters=best.tolist(),training_loss_m=best_loss,
            improved=best_loss<min(initial['loss_m']),applied=False,
            method='bounded three-dimensional pattern search on full two-second training rollouts',
            gradient_refinement_rejected='Half-second real-window adjoint disagreed with finite differences.',
            EI_upper_bound_n_m2=.004))
        print(f'Direct refinement {iteration}/3: {best.tolist()}, loss {best_loss*1000:.3f} mm',flush=True)


def refine(output,source,benchmark,payload,device,residual=False):
    cable,model,takes,intensities=load_takes(source,benchmark,payload,device=device)
    train=[t for t in takes if t.role=='training']
    settings=read(output/'protocol.json')
    stage='residual' if residual else 'physics'
    initial=read(output/('physics.json' if residual else 'grid.json'))['selected_parameters']
    parameters=torch.tensor(initial,dtype=torch.float64,device=device)
    torch.manual_seed(settings['seed']);rng=np.random.default_rng(settings['seed'])
    network=None
    if residual:
        checks=read(output/'short_gradient_checks.json')['rows']
        accepted=[h for h in sorted({r['horizon_s'] for r in checks})
                  if all(r['passed'] for r in checks if r['horizon_s']==h)]
        if not accepted:
            raise ValueError('No tested short horizon has usable real-window gradients; residual training is deferred.')
        residual_horizon=min(.1,max(accepted))
        atomic_json(output/'residual_protocol.json',dict(horizon_s=residual_horizon,updates=24,
            gradient_check='short_gradient_checks.json',selection_horizon_s=2.,
            sampling='Random causal short segments inside stratified two-second training windows.',
            optimizer='Adam, learning rate 0.0005; regularization 0.001 on sampled initial-state correction',
            reason='Half-second real-trajectory gradients disagree with finite differences; use locally checked short dynamics gradients.'))
        network=MotionResidual(cable.node_count,hidden=32,acceleration_limit=.5).to(device=device,dtype=torch.float64)
        model.motion_residual=network;variables=list(network.parameters());raw=None
    else:
        bounds=parameters.new_tensor(settings['parameter_bounds']).T
        raw=torch.nn.Parameter(parameters.clamp(min=bounds[0],max=bounds[1]).log())
        variables=[raw]
    optimizer=torch.optim.Adam(variables,lr=.0005 if residual else .05)
    def get_parameters():return parameters if residual else raw.exp()
    best=float(population(train,intensities,model,cable,[initial])[0][0]);best_parameters=initial
    best_weights=deepcopy(network.state_dict()) if network else None
    history=[dict(update=0,training_loss_m=best,parameters=initial)]
    start=time.perf_counter()
    updates=24 if residual else 12
    for update in range(1,updates+1):
        horizon=residual_horizon if residual else [.5,1.,2.][min((update-1)//4,2)]
        items=[];batch_labels={}
        for take in train:
            labels=np.array(intensities[take.take_id]);indices=[]
            for label in sorted(set(labels)):
                indices.append(int(rng.choice(np.flatnonzero(labels==label))))
            item=subset(take,indices);frames=round(horizon/.01)+1
            if residual:
                if cable.node_count != 12 or cable.interval_subdivisions != (2,)+(1,)*9:
                    raise ValueError('Short residual sampler requires the measured 12-node topology.')
                anchors=rng.integers(10,take.horizon_steps-frames+2,size=len(indices))
                history_sites=[];roots=[];targets=[]
                for j,anchor in enumerate(anchors):
                    history_sites.append(torch.cat((item.root_positions_m[j,anchor-10:anchor+1,None],
                        item.measured_marker_positions_m[j,anchor-10:anchor+1]),dim=1))
                    roots.append(item.root_positions_m[j,anchor:anchor+frames])
                    targets.append(item.measured_marker_positions_m[j,anchor:anchor+frames])
                sites=torch.stack(history_sites)
                nodes=torch.cat((sites[:,:,:1],.5*(sites[:,:,:1]+sites[:,:,1:2]),sites[:,:,1:]),dim=2)
                with torch.no_grad():state=causal_state(nodes,.01,model)
                items.append(replace(item,initial_positions_m=state.positions_m,initial_velocities_m_s=state.velocities_m_s,
                    root_positions_m=torch.stack(roots),measured_marker_positions_m=torch.stack(targets)))
            else:
                items.append(replace(item,root_positions_m=item.root_positions_m[:,:frames],
                                     measured_marker_positions_m=item.measured_marker_positions_m[:,:frames]))
            batch_labels[take.take_id]=labels[indices].tolist()
        item=combine(items);optimizer.zero_grad()
        prediction=rollout(item,model,cable,get_parameters(),gradients=True)
        sq=(prediction[:,1:]-item.measured_marker_positions_m[:,1:]).square().sum(-1)
        per_window=(.002*(torch.sqrt(1+sq/.002**2)-1)).mean((1,2))
        loss=(weights(items,batch_labels)*per_window).sum()
        if residual:
            correction=network(item.initial_positions_m,item.initial_velocities_m_s)
            loss=loss+.001*correction.square().mean()
        if not torch.isfinite(loss):raise ValueError(f'Nonfinite {stage} objective.')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(variables,1.)
        if not torch.isfinite(norm):raise ValueError(f'Nonfinite {stage} gradient.')
        if residual and float(norm)>1.:
            history.append(dict(update=update,horizon_s=horizon,skipped='excessive short-rollout gradient',
                                gradient_norm=float(norm)))
            atomic_json(output/f'{stage}_history.json',history)
            print(f'Skipped residual update {update}: gradient norm {float(norm):.3g}',flush=True)
            continue
        optimizer.step()
        if not residual:
            with torch.no_grad():raw.clamp_(bounds[0].log(),bounds[1].log())
        row=dict(update=update,horizon_s=horizon,batch_loss_m=float(loss.detach()),gradient_norm=float(norm),
                 parameters=get_parameters().detach().cpu().tolist(),elapsed_s=time.perf_counter()-start)
        if update%(8 if residual else 4)==0:
            score=float(population(train,intensities,model,cable,[row['parameters']])[0][0])
            row['training_loss_m']=score
            if score<best:
                best=score;best_parameters=row['parameters']
                if network:best_weights=deepcopy(network.state_dict())
        history.append(row);atomic_json(output/f'{stage}_history.json',history)
        print(f'{stage} update {update}/{updates}, horizon {horizon}s, elapsed {row["elapsed_s"]:.0f}s',flush=True)
    if network:
        network.load_state_dict(best_weights);save_weights(output/'residual_candidate.pt',network)
    atomic_json(output/f'{stage}.json',dict(selected_parameters=best_parameters,training_loss_m=best,
        improved=best<history[0]['training_loss_m'],applied=False))


@torch.no_grad()
def evaluate_together(takes,model,cable,parameters):
    """Batch independent takes once; retain the same per-take reporting metric."""
    item=combine(takes);prediction=rollout(item,model,cable,parameters)
    sq=(prediction[:,1:]-item.measured_marker_positions_m[:,1:]).square().sum(-1)
    if not torch.isfinite(sq).all():raise ValueError('Nonfinite candidate evaluation.')
    per_role={'training':{},'validation':{}};start=0
    for t in takes:
        end=start+t.window_count;err=sq[start:end];start=end
        lead={}
        for seconds in (.1,.25,.5,.7,1.,2.,3.,5.):
            frame=round(seconds/t.dt_s)
            if frame<=t.horizon_steps:
                lead[str(seconds)]=dict(marker_rmse_m=float(err[:,frame-1].mean().sqrt()),
                    tip_rmse_m=float(err[:,frame-1,-1].mean().sqrt()))
        per_role[t.role][t.take_id]=dict(objective=float((.002*(torch.sqrt(1+err/.002**2)-1)).mean()),
            marker_rmse_m=float(err.mean().sqrt()),tip_rmse_m=float(err[:,:,-1].mean().sqrt()),
            windows=t.window_count,lead_times=lead,initialization_marker_rmse_m=t.initialization_marker_rmse_m)
    result={}
    for role,rows in per_role.items():
        result[role]=dict(aggregation='arithmetic_mean_of_per_take_RMSE',per_take=rows,
            objective=float(np.mean([r['objective'] for r in rows.values()])),
            equal_take_marker_rmse_m=float(np.mean([r['marker_rmse_m'] for r in rows.values()])),
            equal_take_tip_rmse_m=float(np.mean([r['tip_rmse_m'] for r in rows.values()])))
    return result,prediction.cpu().numpy(),item.measured_marker_positions_m.cpu().numpy()


def evaluate_candidate(output,source,benchmark,payload,device):
    selected=read(output/'physics.json')['selected_parameters'];candidate=deepcopy(payload)
    candidate['cable'].update(EI_n_m2=selected[0],Cb_n_m2_s=selected[1],external_drag_s_inv=selected[2])
    candidate['force_accounting']['aerodynamic_drag']='effective_world_velocity_decay_on_cable_nodes; rate is not a measured aerodynamic coefficient'
    candidate['cable']['parameter_source']=str(output/'physics.json')
    atomic_json(output/'candidate_model.json',candidate)
    residual_candidate=deepcopy(candidate)
    residual_candidate['motion_residual']=dict(enabled=True,checkpoint=str(output/'residual_candidate.pt'),
        sha256=sha256_file(output/'residual_candidate.pt'),kind='bounded_motion_discrepancy')
    atomic_json(output/'candidate_with_residual.json',residual_candidate)
    rows={}
    for horizon in (2.,5.):
        rows[str(horizon)]={}
        for label,p in [('original_fitted',read(source/'model.json')),('geometry_drag_seed',payload),
                        ('constrained_physics',candidate),('constrained_plus_residual',candidate)]:
            p=deepcopy(p)
            if label=='original_fitted':p['cable'].update(read(benchmark/'protocol.json')['physical_parameters'])
            cable,model,takes,_=load_takes(source,benchmark,p,horizon=horizon,device=device)
            if label=='constrained_plus_residual':
                model.motion_residual=FrozenMotionResidual(output/'residual_candidate.pt',sha256_file(output/'residual_candidate.pt'))
            parameters=torch.tensor([p['cable']['EI_n_m2'],p['cable']['Cb_n_m2_s'],p['cable'].get('external_drag_s_inv',0.)],dtype=torch.float64,device=device)
            scores,prediction,truth=evaluate_together(takes,model,cable,parameters)
            rows[str(horizon)][label]=scores
            if horizon==2.:
                np.savez_compressed(output/f'{label}_2s_predictions.npz',prediction=prediction,measured=truth,
                    take=np.array([t.take_id for t in takes for _ in t.starts]),
                    start_frame=np.array([s for t in takes for s in t.starts]))
            atomic_json(output/'evaluation.json',rows)
            print(f'Evaluated {label} at {horizon}s',flush=True)
    atomic_json(output/'status.json',dict(status='COMPLETED',active_model_changed=False,protected_test_used=False))


@torch.no_grad()
def resolution_check(output,source,benchmark,payload,device):
    selected=read(output/'physics.json')['selected_parameters']
    predictions={};records=None;truth=None
    for substeps in (12,24):
        p=deepcopy(payload);p['cable']['substeps']=substeps
        cable,model,takes,_=load_takes(source,benchmark,p,device=device)
        item=combine(takes);parameters=item.initial_positions_m.new_tensor(selected)
        prediction=rollout(item,model,cable,parameters).cpu().numpy()
        predictions[str(substeps)]=prediction
        truth=item.measured_marker_positions_m.cpu().numpy()
        records=[dict(take=t.take_id,role=t.role,start=s) for t in takes for s in t.starts]
        print(f'Resolution check: {substeps} substeps',flush=True)
    change=np.sum((predictions['12'][:,1:]-predictions['24'][:,1:])**2,axis=-1)
    rows={}
    for name in sorted({r['take'] for r in records}):
        ix=[i for i,r in enumerate(records) if r['take']==name]
        rows[name]=dict(marker_prediction_difference_rmse_m=float(np.sqrt(change[ix].mean())),
            tip_prediction_difference_rmse_m=float(np.sqrt(change[ix,:,-1].mean())),
            **{f'marker_rmse_{s}_m':float(np.sqrt(np.sum((predictions[s][ix,1:]-truth[ix,1:])**2,axis=-1).mean()))
               for s in ('12','24')})
    atomic_json(output/'resolution_check.json',dict(per_take=rows,records=records,
        interpretation='12-to-24-substep prediction difference at frozen parameters, not continuum error.'))
    np.savez_compressed(output/'resolution_predictions.npz',prediction_12=predictions['12'],prediction_24=predictions['24'],measured=truth)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--benchmark',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['grid','physics','direct','residual','evaluate','resolution'],required=True)
    a=p.parse_args();torch.set_num_threads(1)
    a.output=a.output.resolve();a.source=a.source.resolve();a.benchmark=a.benchmark.resolve()
    if a.stage=='grid':initialize(a.output,a.source,a.benchmark,a.geometry.resolve())
    stage_hashes={}
    for name in [*read(a.output/'protocol.json')['source_hashes'], 'simulator/point_mass.py']:
        target=a.output/'stage_sources'/a.stage/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target);stage_hashes[name]=sha256_file(target)
    atomic_json(a.output/f'{a.stage}_source_hashes.json',stage_hashes)
    atomic_json(a.output/'status.json',dict(status='RUNNING',stage=a.stage))
    payload=read(a.output/'geometry_model.json');device='cuda' if torch.cuda.is_available() else 'cpu'
    if a.stage=='grid':grid(a.output,a.source,a.benchmark,payload,device)
    elif a.stage=='direct':direct_refine(a.output,a.source,a.benchmark,payload,device)
    elif a.stage=='resolution':resolution_check(a.output,a.source,a.benchmark,payload,device)
    elif a.stage in ('physics','residual'):refine(a.output,a.source,a.benchmark,payload,device,a.stage=='residual')
    else:evaluate_candidate(a.output,a.source,a.benchmark,payload,device)
