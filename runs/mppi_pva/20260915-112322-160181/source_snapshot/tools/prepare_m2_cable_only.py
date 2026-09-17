"""Disable the current M2 quadrotor correction without refitting any component."""
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experimental_data import whip_full_data as data
from experimental_data.whip_adaptation_fit import records,evaluate
from experimental_data.whip_full_fit import immutable_identity,validate_candidate
from experimental_data.whip_adaptation import verify_hashes
from experimental_data.model_evaluation import model_identity
from experimental_data.io import atomic_json,sha256_file
from simulator.workflow import read_json
from planning.pva_job import freeze_model_assets


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    source=ROOT/'runs/adaptation/M2-selected-20260913'
    original_fit=ROOT/'runs/adaptation/M2-paper-20260913'
    out=ROOT/'runs/adaptation/M2-cable-only-20260913'
    out.mkdir(parents=True,exist_ok=False)
    verify_hashes(read_json(source/'fit/result.json')['candidate_hashes'])
    parent,protocol,parent_engine=data.load(original_fit)
    names=['M1_003','M1_005']
    rows=records(original_fit,names,parent,parent_engine,'cuda')
    folder=out/'candidate';folder.mkdir()
    m=freeze_model_assets(read_json(source/'candidate/model.json'),folder,
        source_root=source/'candidate',portable=True)
    dp=folder/m['fullstate_execution']['checkpoint'];d=read_json(dp)
    cp=dp.parent/d['residual']['checkpoint']
    payload=torch.load(cp,map_location='cpu',weights_only=True)
    # The frozen loader requires a checkpoint. An exact-zero network preserves
    # its schema and is verified against executing the nominal equations with None.
    payload['state_dict']={k:torch.zeros_like(v) for k,v in payload['state_dict'].items()}
    torch.save(payload,cp)
    d['residual']['sha256']=sha256_file(cp);atomic_json(dp,d)
    m['fullstate_execution']['sha256']=sha256_file(dp)
    m['provenance'].update(label='M2-cable-only',source_job=str(out),
        source_model=str(source/'candidate/model.json'),candidate_variant='cable_residual_only',
        drone_residual_enabled=False,drone_residual_implementation='Exact-zero compatibility checkpoint; nominal-only equivalence checked',
        nominal_parameters_refitted=False,training_restarted=False,
        model_choice='User-requested ablation; not automatically selected by prediction score',
        prospective_flight_evidence=False)
    atomic_json(folder/'model.json',m)
    assert immutable_identity(folder/'model.json')==immutable_identity(source/'candidate/model.json')
    source_model=read_json(source/'candidate/model.json')
    assert m['cable']==source_model['cable']
    source_dp=source/'candidate'/source_model['fullstate_execution']['checkpoint']
    assert d['nominal']==read_json(source_dp)['nominal']
    source_cp=source/'candidate'/source_model['motion_residual']['checkpoint']
    assert sha256_file(folder/m['motion_residual']['checkpoint'])==sha256_file(source_cp)
    engine=validate_candidate(folder/'model.json',protocol['full_update'])
    assert all(torch.count_nonzero(v)==0 for v in engine.drone.residual.state_dict().values())
    differences={}
    for r in rows:
        t=r['trial'];dat=t.data
        args=(t.initial_pose(engine.drone.parameters),torch.as_tensor(dat['packets'][None],device='cuda',dtype=torch.float64),
              dat['packet_time'],r['grid'],engine.offset)
        kwargs=dict(graph=False,hover_command=torch.as_tensor(dat['hover_commands'][-1:],device='cuda',dtype=torch.float64))
        zero=engine.drone.predict(*args,**kwargs)
        residual=engine.drone.residual;engine.drone.residual=None
        try:nominal=engine.drone.predict(*args,**kwargs)
        finally:engine.drone.residual=residual
        differences[r['name']]={k:float((zero[k]-nominal[k]).abs().max()) for k in
            ('position_origin_m','velocity_origin_m_s','rotation_tracking_to_world','position_attachment_m')}
        assert all(v==0. for v in differences[r['name']].values())
    print('Exact nominal-only equivalence verified; evaluating existing recordings',flush=True)
    metrics=evaluate(rows,engine,m['cable']['external_drag_s_inv'],out/'evaluation')
    means=dict(tip_rmse_mean_m=float(np.mean([r['command_driven']['tip']['rmse_m'] for r in metrics.values()])),
        quadrotor_rmse_mean_m=float(np.mean([r['drone']['rmse_m'] for r in metrics.values()])))
    prior=read_json(source/'comparison.json')
    comparison=dict(schema='fixed_parameter_residual_ablation_v1',data_use=prior['data_use'],
        selection_takes=names,training_restarted=False,nominal_refitted=False,
        source_model=str(source/'candidate/model.json'),selected='cable_residual_only',
        original_M1=prior['metrics']['M1_baseline'],
        retained_drone_residual=prior['metrics']['retained_drone_M2_cable'],
        cable_only=dict(**means,per_take=metrics),nominal_only_exact_differences=differences,
        cable_parameters_and_weights_unchanged=True,quadrotor_parameters_unchanged=True,
        choice='User-requested fixed-parameter ablation, regardless of retrospective score; prospective physical test pending')
    atomic_json(out/'comparison.json',comparison)
    atomic_json(out/'fit/selection_frozen.json',dict(selected='cable_residual_only',
        comparison_sha256=sha256_file(out/'comparison.json'),script_sha256=sha256_file(__file__),
        signature=model_identity(folder/'model.json')[0],physical_performance_pending=True))
    hashes=model_identity(folder/'model.json')[1]
    for p in (out/'comparison.json',out/'fit/selection_frozen.json'):hashes[str(p)]=sha256_file(p)
    atomic_json(out/'fit/result.json',dict(status='completed',operation='disable_drone_residual_without_refitting',
        candidate_hashes=hashes,training_restarted=False,model_selected=True))
    source_hashes=model_identity(source/'candidate/model.json')[1]
    source_hashes[str(original_fit/'prepared_hashes.json')]=sha256_file(original_fit/'prepared_hashes.json')
    verify_hashes(source_hashes);atomic_json(out/'source_hashes.json',source_hashes)
    atomic_json(out/'status.json',dict(status='completed'))
    print(means,flush=True)


if __name__=='__main__':main()
