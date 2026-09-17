"""Unfitted nominal-pose numerical probe; never selects a model or starts fitting."""
from dataclasses import asdict
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experimental_data.adaptation_rounds import write_json
from experimental_data.io import sha256_file
from experimental_data.drone_pose_response_data import load_nominal_pose_trial
from simulator.drone_pose_response import PoseResponseParameters,predict_pose
from simulator.workflow import stamp


def main():
    if not torch.cuda.is_available():raise RuntimeError('This audit explicitly requires the local CUDA device')
    output=ROOT/'runs/audits'/(stamp()+'-nominal-drone-pose');output.mkdir(parents=True)
    source=ROOT/'data/adaptation_rounds/adaptation0/processed/20260908-041031-260837'
    # Deliberate fixed numerical examples, not estimated or tuned to these takes.
    parameters=PoseResponseParameters(4.,4.,3.,3.,1.,1.,.08,.02)
    write_json(output/'probe_parameters.json',dict(status='UNFITTED_NUMERICAL_PROBE_ONLY',parameters=asdict(parameters),
        purpose='Exercise the new engine and data adapter, not a candidate-model accuracy claim'))
    protected={}
    for directory in [source,ROOT/'config']:
        for path in directory.rglob('*'):
            if path.is_file():protected[str(path)]=sha256_file(path)
    for name in ['runs/ppo/20260906-201957-294110-measured-mass-seed653/checkpoints/best_validation.pt',
                 'simulator/drone_tracking.py','simulator/fullstate_execution.py','learning/fullstate_rollout.py',
                 'simulator/cable/residual.py','experimental_data/historical_fit.py']:
        path=ROOT/name;protected[str(path)]=sha256_file(path)
    reports=[];curves=[]
    for name in ['whip1_001','whip1_002','whip1_003']:
        folder=output/name;folder.mkdir()
        started=time.perf_counter()
        args,truth,context=load_nominal_pose_trial(source/name,parameters,device='cuda',
            alignment_mode='prehover_effective_alignment')
        with torch.no_grad():result=predict_pose(**args)
        torch.cuda.synchronize()
        arrays={key:value[0].cpu().numpy() for key,value in result.items()}
        assert all(np.isfinite(v).all() for v in arrays.values())
        orthogonal=arrays['rotation_tracking_to_world'].transpose(0,2,1)@arrays['rotation_tracking_to_world']
        np.testing.assert_allclose(orthogonal,np.broadcast_to(np.eye(3),orthogonal.shape),atol=1e-10,rtol=1e-10)
        comparisons={}
        for phase in ['csv_maneuver','post_maneuver_fullstate_hold']:
            mask=truth['execution_phase']==phase
            comparisons[phase]={}
            for point in ['origin','attachment']:
                key=f'position_{point}_m';valid=truth['position_valid' if point=='origin' else 'attachment_valid']&mask
                comparisons[phase][point+'_rmse_m']=float(np.sqrt(np.mean(np.sum((arrays[key][valid]-truth[key][valid])**2,axis=-1)))) if valid.any() else None
        state=args['initial']
        np.savez_compressed(folder/'prediction.npz',time_s=args['output_time_s'],**arrays,
            measured_position_origin_m=truth['position_origin_m'],measured_position_attachment_m=truth['position_attachment_m'],
            measured_rotation_tracking_to_world=truth['rotation_tracking_to_world'],execution_phase=truth['execution_phase'],
            initial_compensation_m_s2=state.compensation.cpu().numpy(),
            effective_rotation_command_from_tracking=state.rotation_command_from_tracking.cpu().numpy())
        write_json(folder/'context.json',context)
        reports.append(dict(trial_id=name,status='UNFITTED_ENGINE_PROBE_COMPLETED',finite=True,
            elapsed_s=time.perf_counter()-started,points=len(args['output_time_s']),
            compensation_m_s2=state.compensation[0].cpu().tolist(),diagnostic_errors_not_fitted_performance=comparisons,
            prediction_sha256=sha256_file(folder/'prediction.npz'),context_sha256=sha256_file(folder/'context.json')))
        curves.append((name,args['output_time_s']-context['csv_onset_s'],arrays,truth))
    for path,checksum in protected.items():assert sha256_file(path)==checksum,path
    sources=['simulator/drone_pose_response.py','experimental_data/drone_pose_response_data.py',
             'experimental_data/drone_pose_initialization.py','simulator/geometry.py','tools/check_nominal_drone_pose.py']
    for name in sources:
        dest=output/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    write_json(output/'verification.json',dict(platform=platform.platform(),device=torch.cuda.get_device_name(),
        status='UNFITTED_NUMERICAL_PROBE_ONLY',reports=reports,protected_sources=protected,
        source_hashes={n:sha256_file(ROOT/n) for n in sources},model_fit_started=False,training_started=False,
        active_model_changed=False,old_fullstate_execution_preserved=True))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    fig=Figure(figsize=(12,8));axes=fig.subplots(3,2)
    for i,(name,t,pred,measured) in enumerate(curves):
        for j,(key,axis,title) in enumerate([('position_origin_m',2,'Tracked-origin Z (m)'),('position_attachment_m',0,'Attachment X (m)')]):
            ax=axes[i,j];ax.plot(t,measured[key][:,axis],label='Measured');ax.plot(t,pred[key][:,axis],label='Unfitted numerical example')
            ax.axvline(0,color='gray',linestyle=':');ax.set_title(name);ax.set_ylabel(title);ax.set_xlabel('Time from observed CSV onset (s)');ax.grid(alpha=.2)
            if i==0:ax.legend(fontsize=8)
    fig.suptitle('Engine check only — parameters have NOT been fitted');fig.tight_layout();fig.savefig(output/'unfitted_probe.png',dpi=130)
    (output/'REPORT.md').write_text('# Nominal drone pose engine probe\n\nAll three phase-aware legacy whips ran through the new engine using fixed, deliberately unfitted example parameters on '+torch.cuda.get_device_name()+'. This checks numerical execution and the data connection; it is not a preliminary model fit or a flight-performance validation.\n\nFuture measured pose/attachment is kept separate from prediction inputs. Initial state and frozen effective compensation come from pre-hover data only. The effective orientation alignment is a pre-hover reference, not verified firmware mounting.\n\nThe model predicts tracked-origin P/V and tracked-frame R/omega, then computes attachment P/V/A with all rigid rotation terms. It holds original logged commands with their effective delay and freshness intervals. There is no extra gravity or cable force in effective translational acceleration. No PPO, residual training, vehicle control or active configuration change occurred.\n',encoding='utf-8')
    print(output)
    for report in reports:print(report['trial_id'],report['status'],round(report['elapsed_s'],3))


if __name__=='__main__':main()
