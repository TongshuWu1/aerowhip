"""Isolate cable shape/velocity sensitivity under one frozen command and model."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from experimental_data.io import atomic_json,sha256_file
from experimental_data.whip_adaptation import WhipTrial,verify_hashes
from experimental_data.model_evaluation import model_identity
from experimental_data.state_initialization import project_state
from experimental_data.whip_full_fit import immutable_identity
from planning.reference_correction import CoupledRollout
from planning.local_reference_correction import strike_metrics
from simulator.workflow import read_json
from simulator.research_execution import ResearchExecutionModel


@torch.no_grad()
def run(model_path,reference_path,prepared_job,output):
    model_path,reference_path,prepared_job,output=map(lambda p:Path(p).resolve(),(model_path,reference_path,prepared_job,output))
    if output.exists():raise ValueError('Sensitivity output directory must be new')
    verify_hashes(read_json(prepared_job/'prepared_hashes.json'))
    verify_hashes(read_json(prepared_job/'source_hashes.json'))
    meta=read_json(reference_path/'reference.json')
    if sha256_file(reference_path/'reference.npz')!=meta['reference_sha256']:raise ValueError('Reference changed')
    with np.load(reference_path/'reference.npz') as z:ref={k:z[k].copy() for k in z.files}
    model=read_json(model_path);engine=ResearchExecutionModel.from_mapping(model,root=model_path.parent,device='cuda')
    if immutable_identity(model_path)!=immutable_identity(prepared_job/'source_candidate/model.json'):
        raise ValueError('Sensitivity model and prepared data have different geometry or execution contracts')
    tensor=lambda v:torch.as_tensor(v,device='cuda',dtype=torch.float64)
    nominal=ref['cable_position_m'][0];root=nominal[0]
    qs=[nominal];vs=[np.zeros_like(nominal)];labels=[dict(take=None,condition='nominal')]
    protocol=read_json(prepared_job/'protocol.json')
    for name in protocol['takes']:
        trial=WhipTrial(prepared_job,name,model,'cuda')
        state,start,projection=trial.cable_state(engine.physics,cutoff=0.)
        q=state.positions_m[0].cpu().numpy();v=state.velocities_m_s[0].cpu().numpy()
        # An explicit counterfactual isolates relative cable motion while keeping
        # the planned settled vehicle and launch fixed. No fit inputs are edited.
        q=q-q[0]+root;v=v-v[0]
        for condition,shape,velocity in [('shape',q,np.zeros_like(q)),('velocity',nominal,v),('shape_and_velocity',q,v)]:
            consistent=project_state(engine.physics,tensor(shape)[None],tensor(velocity)[None])
            projected_q=consistent.positions_m[0].cpu().numpy();projected_v=consistent.velocities_m_s[0].cpu().numpy()
            position_adjustment=float(np.abs(projected_q-shape).max());velocity_adjustment=float(np.abs(projected_v-velocity).max())
            shape,velocity=projected_q,projected_v
            qs.append(shape);vs.append(velocity)
            labels.append(dict(take=name,condition=condition,measurement_time_s=start,projection_m=projection,
                scenario_projection_max_m=position_adjustment,scenario_velocity_projection_max_m_s=velocity_adjustment,
                initial_shape_rms_m=float(np.sqrt(np.mean(np.sum((shape-nominal)**2,axis=-1)))),
                initial_tip_speed_m_s=float(np.linalg.norm(velocity[-1]))))
    q=np.stack(qs);v=np.stack(vs);count=len(q)
    executor=CoupledRollout(engine,count,meta['launch_origin_m'],q,meta['limits'],initial_velocity=v)
    prediction=executor(tensor(ref['original_command_packets'])[None].expand(count,-1,-1),ref['command_time_s'],ref['time_s'])
    tip=prediction['cable_positions_m'][:,:,-1].cpu().numpy();valid=prediction['complete_valid'].cpu().numpy()
    rows=[]
    for i,label in enumerate(labels):
        metrics=(dict(strike=strike_metrics(tip[i],ref['time_s'],ref['tip_position_m'],meta['planned_strike_time_s'],meta['physical_target_m']),
            tip_reference_rmse_m=float(np.sqrt(np.mean(np.sum((tip[i]-ref['tip_position_m'])**2,axis=-1)))),
            tip_change_from_nominal_rmse_m=float(np.sqrt(np.mean(np.sum((tip[i]-tip[0])**2,axis=-1)))) if valid[0] else None) if valid[i] else
            dict(strike=None,tip_reference_rmse_m=None,tip_change_from_nominal_rmse_m=None))
        rows.append(dict(label,valid=bool(valid[i]),**metrics))
    _,hashes=model_identity(model_path)
    for f in (reference_path/'reference.json',reference_path/'reference.npz',prepared_job/'prepared_hashes.json',Path(__file__)):
        hashes[str(f)]=sha256_file(f)
    verify_hashes(hashes)
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'scenarios.npz',initial_positions_m=q,initial_velocities_m_s=v,
        time_s=ref['time_s'],tip_positions_m=tip,valid=valid)
    result=dict(scenarios=rows,source_hashes=hashes,
        interpretation='Simulation sensitivity, not measured accuracy or evidence of physical improvement. Causal pre-command cable states are translated to the planned root, root velocity is subtracted, and scenario states are projected onto cable constraints; vehicle state, model, and commands stay fixed. Invalid rollouts are unscored.')
    atomic_json(output/'report.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','reference','prepared-job','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    run(a.model,a.reference,a.prepared_job,a.output)
    print(a.output/'report.json')


if __name__=='__main__':main()
