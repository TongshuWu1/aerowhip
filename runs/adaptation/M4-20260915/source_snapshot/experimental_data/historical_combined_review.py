"""Cross-fitted command -> attachment -> cable prediction, with ablations."""
import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
import shutil
from .historical_fit import read,read_inputs,cable_takes,cable_metrics,progress
from .force_dataset import _normalized_rotations
from .constrained_identification import combine
from .io import atomic_json,sha256_file,canonical_json_hash
from simulator.cable.residual import FrozenMotionResidual
from simulator.fullstate_execution import FullStateAttachmentModel


class MaskedResidual:
    def __init__(self,residual,mask):self.residual,self.mask=residual,mask
    def __call__(self,q,v):return self.residual(q,v)*self.mask[:,None,None]
    def drag_rates(self,q):return q.new_zeros(q.shape[-2])


@torch.no_grad()
def review(job, *, folds=None, cable_directory=None, output_name='combined', reuse=()):
    torch.set_num_threads(1)
    settings=read(job/'protocol.json');data=read_inputs(job)
    cable_path=job/(cable_directory or read(job/'cable_result.json')['directory'])
    output=job/output_name;output.mkdir(exist_ok=False)
    all_results={}
    for label in (folds or ['fold_1','fold_2','fold_3','final']):
        folder=output/label
        payload=read(cable_path/label/'candidate_model.json')
        drone_file=job/'drone_attachment'/label/'drone_residual.pt'
        sources=dict(cable_model_sha256=canonical_json_hash(payload),drone_sha256=sha256_file(drone_file),
                     input_protocol_sha256=sha256_file(job/'protocol.json'))
        cached=next((job/name/label for name in reuse if (job/name/label/'evaluation.json').exists()
            and (job/name/label/'source_models.json').exists() and read(job/name/label/'source_models.json')==sources),None)
        if cached:
            shutil.copytree(cached,folder);all_results[label]=read(folder/'evaluation.json')
            progress(output,f'{label}: reusing completed identical-model combined prediction')
            continue
        folder.mkdir()
        atomic_json(folder/'source_models.json',sources)
        drone=FullStateAttachmentModel(drone_file,sha256_file(drone_file),device='cuda')
        np.testing.assert_allclose(drone.offset_body_m,payload['recorded_data']['optitrack_to_attachment_offset_body_m'],atol=1e-10)
        names=list(data) if label=='final' else settings['folds'][int(label[-1])-1]
        selected={n:dict(a,cable_fit_valid=a['cable_fit_valid']&a['drone_fit_valid']) for n,a in data.items() if n in names}
        cable,model,takes=cable_takes(selected,payload,2.,6,'cuda')
        if set(t.take_id for t in takes)!=set(names):raise ValueError(f'{label}: a take lacks a valid combined window')
        commands=[];past=[];positions=[];velocities=[]
        lag=round(drone.delay_s/.01);count=200
        for take in takes:
            a=selected[take.take_id];rotation,_=_normalized_rotations(a['quaternion'])
            offsets=np.einsum('tij,j->ti',rotation,drone.offset_body_m)
            attachment=a['position']+offsets
            for s in take.starts:
                if s<lag+5:raise ValueError('Insufficient command history')
                cmd=a['commands'][s-lag:s-lag+count].copy();cmd[:,:3]+=offsets[s]
                h=a['commands'][s-lag-5:s-lag-5+count].copy();h[:,:3]+=offsets[s]
                t=np.arange(-10,1)*.01
                velocity=np.linalg.lstsq(np.column_stack((np.ones(11),t,t*t)),attachment[s-10:s+1],rcond=None)[0][1]
                commands.append(cmd);past.append(h);positions.append(attachment[s]);velocities.append(velocity)
        tensor=lambda x:torch.tensor(np.asarray(x),dtype=torch.float64,device='cuda')
        p,v,cmd,h=tensor(positions),tensor(velocities),tensor(commands),tensor(past)
        dt=p.new_full((len(p),count),.01)
        nominal=drone.predict(p,v,cmd,h,dt,residual=False)[0]
        corrected=drone.predict(p,v,cmd,h,dt,residual=True)[0]
        joined=combine(takes)
        variants=[];group_names=[]
        labels=['nominal_drone_physics_cable','nominal_drone_residual_cable','residual_drone_physics_cable','both_residuals']
        for index,(root,variant) in enumerate(zip([nominal,nominal,corrected,corrected],labels)):
            offset=0
            for take in takes:
                n=take.window_count
                velocity=take.initial_velocities_m_s.clone();velocity[:,0]=v[offset:offset+n]
                variants.append(replace(take,take_id=f'{variant}/{take.take_id}',
                    root_positions_m=root[offset:offset+n],initial_velocities_m_s=velocity))
                group_names.extend([index%2]*n);offset+=n
        model.motion_residual=MaskedResidual(FrozenMotionResidual(payload['motion_residual']['checkpoint'],payload['motion_residual']['sha256']),tensor(group_names))
        params=p.new_tensor([payload['cable']['EI_n_m2'],payload['cable']['Cb_n_m2_s'],0.])
        progress(output,f'{label}: predicting command-driven attachment and cable, four model combinations')
        metrics=cable_metrics(variants,model,cable,params,folder/'predictions.npz')
        atomic_json(folder/'evaluation.json',metrics)
        np.savez_compressed(folder/'attachment_predictions.npz',nominal=nominal.cpu().numpy(),residual=corrected.cpu().numpy(),
            measured=joined.root_positions_m.cpu().numpy(),take=np.array([t.take_id for t in takes for _ in t.starts]))
        all_results[label]=metrics
    summary={}
    for variant in ['nominal_drone_physics_cable','nominal_drone_residual_cable','residual_drone_physics_cable','both_residuals']:
        rows=[row for label,result in all_results.items() if label!='final' for key,row in result.items() if key.startswith(variant+'/')]
        summary[variant]={key:float(np.mean([r[key] for r in rows])) for key in ('marker_rmse_m','tip_rmse_m')} if rows else None
    atomic_json(output/'review.json',dict(cross_fitted_equal_take=summary,all_results=all_results,
        interpretation='Development whole-take command-to-cable predictions; no future measured drone or cable state after initialization',
        horizon_s=2.,independent_test=False,active_model_changed=False,ppo_started=False))
    progress(output,'Combined prediction review complete',summary=summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--fold',action='append',choices=['fold_1','fold_2','fold_3','final'])
    parser.add_argument('--cable-directory');parser.add_argument('--output-name',default='combined')
    parser.add_argument('--reuse',action='append',default=[])
    args=parser.parse_args()
    review(args.job,folds=args.fold,cable_directory=args.cable_directory,output_name=args.output_name,reuse=args.reuse)
