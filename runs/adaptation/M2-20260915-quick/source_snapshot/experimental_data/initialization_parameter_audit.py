"""Coarse physical loss surfaces with fixed causal initialization; no NN training."""
import argparse
from dataclasses import replace
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from simulator.cable import CableConfiguration, DderModel
from .cable_fit import PreparedTake
from .differentiable_fit import rollout, evaluate
from .state_initialization import causal_state
from .io import atomic_json, sha256_file


def load_takes(benchmark,source,model,cable):
    with (benchmark/'windows.csv').open() as f:
        rows=[r for r in csv.DictReader(f) if r['method']=='causal_polynomial']
    protocol=json.loads((benchmark/'protocol.json').read_text())
    takes=[]
    for name in sorted({r['take'] for r in rows}):
        selected=[r for r in rows if r['take']==name]
        starts=tuple(int(r['start_frame']) for r in selected)
        with np.load(source/'force_takes'/name/'take.npz') as a:
            q=a['cable_node_position_world_m'];root=a['root_position_world_m']
            dt=float(np.median(np.diff(a['time_s'])))
        steps=round(protocol['horizon_s']/dt)
        history=torch.tensor(np.stack([q[s-10:s+1] for s in starts]),dtype=torch.float64)
        with torch.no_grad():state=causal_state(history,dt,model)
        takes.append(PreparedTake(name,selected[0]['role'],state.positions_m,state.velocities_m_s,
            torch.tensor(np.stack([root[s:s+steps+1] for s in starts]),dtype=torch.float64),
            torch.tensor(np.stack([q[s:s+steps+1,cable.marker_node_indices[1:]] for s in starts]),dtype=torch.float64),
            starts,0.,dt))
    return takes


@torch.no_grad()
def population(take,model,cable,candidates,device):
    """Exact dense float64 trajectory loss; same transition as the state benchmark."""
    count=len(candidates);n=take.window_count
    def repeat(t):return t.to(device).repeat_interleave(count,dim=0)
    expanded=replace(take,initial_positions_m=repeat(take.initial_positions_m),
        initial_velocities_m_s=repeat(take.initial_velocities_m_s),root_positions_m=repeat(take.root_positions_m),
        measured_marker_positions_m=repeat(take.measured_marker_positions_m))
    parameters=torch.tensor(candidates,dtype=torch.float64,device=device).repeat(n,1).T
    prediction=rollout(expanded,model,cable,parameters)
    sq=(prediction[:,1:]-expanded.measured_marker_positions_m[:,1:]).square().sum(-1)
    sq=sq.reshape(n,count,take.horizon_steps,cable.moving_marker_count)
    distance=sq.sqrt();scale=.002
    objective=(scale*(torch.sqrt(1+(distance/scale).square())-1)).mean((0,2,3))
    return dict(objective=objective.cpu().numpy(),marker_rmse_m=sq.mean((0,2,3)).sqrt().cpu().numpy(),
                tip_rmse_m=sq[:,:,:,-1].mean((0,2)).sqrt().cpu().numpy())


def run(benchmark,source,output,size=7):
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'status.json',dict(status='RUNNING'))
    torch.set_num_threads(1)
    p=json.loads((benchmark/'protocol.json').read_text())
    payload=json.loads((source/'model.json').read_text())
    cable=CableConfiguration.from_mapping(payload['cable'])
    previous=p['physical_parameters']
    model=DderModel(cable.dder_parameters(EI=previous['EI_n_m2'],Cb=previous['Cb_n_m2_s']))
    takes=load_takes(benchmark,source,model,cable)
    training=[t for t in takes if t.role=='training'];validation=[t for t in takes if t.role=='validation']
    ei=np.geomspace(1e-9,2.8e-4,size);cb=np.geomspace(1e-8,.02,size)
    grid=np.array([(a,b) for a in ei for b in cb])
    previous_pair=np.array([previous['EI_n_m2'],previous['Cb_n_m2_s']])
    candidates=np.vstack([grid,previous_pair])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checks={}
    # Verify a short sample agrees across devices before using GPU loss surfaces.
    sample=replace(training[0],initial_positions_m=training[0].initial_positions_m[:1],
        initial_velocities_m_s=training[0].initial_velocities_m_s[:1],root_positions_m=training[0].root_positions_m[:1,:11],
        measured_marker_positions_m=training[0].measured_marker_positions_m[:1,:11],starts=training[0].starts[:1])
    cpu=population(sample,model,cable,previous_pair[None],torch.device('cpu'))
    actual=population(sample,model,cable,previous_pair[None],device)
    for key in cpu:
        np.testing.assert_allclose(actual[key],cpu[key],rtol=1e-6,atol=1e-9)
        checks[key]=float(np.max(np.abs(actual[key]-cpu[key])))
    snapshot={}
    for name in ['experimental_data/initialization_parameter_audit.py','experimental_data/state_initialization.py',
                 'experimental_data/differentiable_fit.py','simulator/cable/dder.py']:
        f=Path(__file__).resolve().parents[1]/name;target=output/'source_snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target);snapshot[name]=sha256_file(f)
    atomic_json(output/'protocol.json',dict(source_job=str(source),benchmark=str(benchmark),
        initializer='fixed past-only 11-sample quadratic velocity and projected measured positions',
        grid_size=size,parameter_bounds=[[float(ei[0]),float(ei[-1])],[float(cb[0]),float(cb[-1])]],
        selection='equal-take mean robust trajectory loss on training only, including previous parameters',
        device=str(device),dtype='float64',cross_device_short_rollout_absolute_difference=checks,
        benchmark_protocol_sha256=sha256_file(benchmark/'protocol.json'),windows_sha256=sha256_file(benchmark/'windows.csv'),
        source_sha256=snapshot,interpretation='coarse discrete profile, not a confidence interval or convergence claim'))
    rows=[];results={};started=time.perf_counter()
    # Chunk candidate populations to bound memory and keep exact dense solves.
    for take in training:
        pieces=[]
        for begin in range(0,len(candidates),10):
            pieces.append(population(take,model,cable,candidates[begin:begin+10],device))
        result={k:np.concatenate([item[k] for item in pieces]) for k in pieces[0]}
        if not all(np.isfinite(v).all() for v in result.values()):raise ValueError(f'Nonfinite grid at {take.take_id}')
        results[take.take_id]=result
        for i,(a,b) in enumerate(candidates):
            rows.append(dict(take=take.take_id,candidate=i,EI_n_m2=a,Cb_n_m2_s=b,**{k:float(v[i]) for k,v in result.items()}))
        print(f'Grid complete: {take.take_id}, elapsed {time.perf_counter()-started:.1f}s',flush=True)
    losses=np.stack([r['objective'] for r in results.values()])
    aggregate=losses.mean(0);chosen=int(aggregate.argmin())
    best=candidates[chosen]
    per_take={name:dict(candidate=int(r['objective'].argmin()),parameters=candidates[r['objective'].argmin()].tolist()) for name,r in results.items()}
    leave_one_out={name:candidates[np.delete(losses,i,axis=0).mean(0).argmin()].tolist() for i,name in enumerate(results)}
    # Exactly one training-selected candidate is checked on validation, not the full grid.
    report=dict(selected_parameters=dict(EI_n_m2=float(best[0]),Cb_n_m2_s=float(best[1])),
        selected_candidate=chosen,previous_candidate=len(candidates)-1,
        training_objective_previous=float(aggregate[-1]),training_objective_selected=float(aggregate[chosen]),
        per_take_grid_minima=per_take,leave_one_training_take_out_minima=leave_one_out,
        near_best_within_one_percent=candidates[aggregate<=aggregate.min()*1.01].tolist(),
        training_previous=evaluate(tuple(training),model,cable,torch.tensor(previous_pair)),
        training_selected=evaluate(tuple(training),model,cable,torch.tensor(best)),
        validation_previous=evaluate(tuple(validation),model,cable,torch.tensor(previous_pair)),
        validation_selected=evaluate(tuple(validation),model,cable,torch.tensor(best)),
        elapsed_s=time.perf_counter()-started,applied=False,protected_test_used=False)
    atomic_json(output/'result.json',report)
    with (output/'loss_surface.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(7,5),layout='constrained')
    mesh=ax.pcolormesh(ei,cb,aggregate[:-1].reshape(size,size).T*1000,shading='nearest',cmap='viridis')
    ax.scatter(previous_pair[0],previous_pair[1],marker='x',color='white',s=70,label='Previous fit')
    ax.scatter(best[0],best[1],marker='*',color='#D55E00',s=100,label='Training-selected candidate')
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('EI [N m²]');ax.set_ylabel('Cb [N m² s]')
    ax.legend(fontsize=8);fig.colorbar(mesh,ax=ax,label='Mean take robust trajectory loss [mm]')
    ax.set_title('Causal initialization: coarse physical loss surface')
    for ext in ['png','pdf','svg']:fig.savefig(output/f'loss_surface.{ext}',dpi=170)
    plt.close(fig)
    atomic_json(output/'status.json',dict(status='COMPLETED'))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--size',type=int,default=7)
    a=parser.parse_args()
    run(a.benchmark.resolve(),a.source.resolve(),a.output.resolve(),a.size)
