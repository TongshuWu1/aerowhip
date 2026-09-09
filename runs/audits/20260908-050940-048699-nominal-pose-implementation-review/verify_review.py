"""Offline verification of this review; fixed example gains, no parameter fit."""
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT))
from experimental_data.io import sha256_file
from experimental_data.drone_pose_response_data import load_nominal_pose_trial
from simulator.drone_pose_response import PoseResponseParameters,predict_pose


def main():
    if not torch.cuda.is_available():raise RuntimeError('This review requires the named local CUDA device')
    protected=json.loads((OUT/'protected-before.json').read_text())
    for path,digest in protected.items():assert sha256_file(path)==digest,path
    test_files=['tests/physics/test_drone_pose_response.py','tests/calibration/test_drone_pose_response_data.py',
        'tests/calibration/test_drone_pose_initialization.py','tests/calibration/test_geometry_conventions.py',
        'tests/calibration/test_drone_residual.py','tests/flight/test_legacy_execution.py',
        'tests/flight/test_adaptation_rounds.py','tests/training/test_fullstate_execution.py']
    result=subprocess.run([sys.executable,'-m','pytest',*test_files,'-q'],cwd=ROOT,capture_output=True,text=True)
    (OUT/'pytest.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError('Tests failed; see pytest.txt')
    print(result.stdout.strip(),flush=True)
    source=ROOT/'data/adaptation_rounds/adaptation0/processed/20260908-041031-260837'
    old=ROOT/'runs/audits/20260908-045331-240135-nominal-drone-pose'
    params=PoseResponseParameters(4.,4.,3.,3.,1.,1.,.08,.02)
    reports=[]
    for name in ['whip1_001','whip1_002','whip1_003']:
        args,truth,context=load_nominal_pose_trial(source/name,params,device='cuda',alignment_mode='prehover_effective_alignment')
        with torch.no_grad():out=predict_pose(**args)
        arrays={key:value[0].cpu().numpy() for key,value in out.items()}
        assert all(np.isfinite(a).all() for a in arrays.values())
        with np.load(old/name/'prediction.npz') as previous:
            for key,value in arrays.items():np.testing.assert_array_equal(value,previous[key])
        with np.load(source/name/'dataset.npz') as data:
            np.testing.assert_array_equal(args['schedule'].values[0].cpu(),data['controller_native_fullstate'][:-1])
            indices=np.searchsorted(data['controller_time_s'],args['output_time_s'])
            q=data['drone_quaternion_xyzw'][indices]
            independently_reconstructed=data['drone_position_m'][indices]+Rotation.from_quat(q).apply(args['offset_tracking_m'])
            np.testing.assert_allclose(independently_reconstructed,truth['position_attachment_m'],atol=1e-12,rtol=0)
            eligible=data['controller_native_csv_sample_index']
            np.testing.assert_array_equal(np.unique(eligible[eligible>=0]),np.arange(20))
        short,short_truth,short_context=load_nominal_pose_trial(source/name,params,device='cuda',alignment_mode='prehover_effective_alignment',post_hold_s=0.)
        with torch.no_grad():short_out=predict_pose(**short)
        n=len(short['output_time_s'])
        for key in out:torch.testing.assert_close(short_out[key],out[key][:,:n],atol=0,rtol=0)
        csv=truth['execution_phase']=='csv_maneuver'
        assert (short_truth['execution_phase']=='csv_maneuver').sum()==csv.sum()
        folder=OUT/name;folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/'prediction.npz',time_s=args['output_time_s'],**arrays,
            **{'measured_'+key:truth[key] for key in ['position_origin_m','rotation_tracking_to_world','position_attachment_m']},
            **{key:truth[key] for key in ['position_valid','orientation_valid','attachment_valid',
                'reference_valid','execution_phase','phase_boundary_uncertain','csv_sample_index']})
        (folder/'context.json').write_text(json.dumps(context,indent=2),encoding='utf-8')
        reports.append(dict(trial_id=name,unfitted_probe=True,all_outputs_identical_to_previous_probe=True,
            original_native_commands_preserved=True,all_20_observed_csv_rows_retained=True,
            observed_csv_tracking_samples=int(csv.sum()),maneuver_only_samples_retained=int((short_truth['execution_phase']=='csv_maneuver').sum()),
            maneuver_only_prediction_equals_full_replay_prefix=True,independent_attachment_reconstruction_passed=True,
            requested_post_hold_s=context['requested_post_hold_s'],post_hold_coverage=context['requested_post_hold_fully_observed'],
            prediction_sha256=sha256_file(folder/'prediction.npz')))
        print(name,'verified: preserved commands, masks, geometry, maneuver-only replay and unchanged example predictions',flush=True)
    files=['simulator/drone_pose_response.py','experimental_data/drone_pose_response_data.py',
        'experimental_data/drone_pose_initialization.py','simulator/geometry.py',
        'docs/NOMINAL_DRONE_POSE_RESPONSE.md','docs/NOMINAL_DRONE_IMPLEMENTATION_REVIEW_20260908.md',*test_files]
    for name in files:
        path=OUT/'after'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,path)
    for path,digest in protected.items():assert sha256_file(path)==digest,path
    summary=dict(status='REVIEWED_AND_FIXED_STANDALONE_UNFITTED_ENGINE',platform=platform.platform(),
        gpu=torch.cuda.get_device_name(),test_command=[sys.executable,'-m','pytest',*test_files,'-q'],
        pytest_summary=result.stdout.strip().splitlines()[-1],protected_files_verified=len(protected),
        reports=reports,source_hashes={name:sha256_file(ROOT/name) for name in files},
        model_fit_started=False,ppo_started=False,active_model_changed=False,
        new_engine_integrated_into_ppo_or_export=False,
        scope='Numerical and data-contract validation, not fitted accuracy or aircraft validation')
    (OUT/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Verified protected files:',len(protected),flush=True)


if __name__=='__main__':main()
