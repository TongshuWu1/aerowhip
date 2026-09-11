"""Create an unfitted zero-extension candidate; never select it or train it."""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import math
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import torch
from experimental_data.differentiable_fit import save_weights
from experimental_data.io import atomic_json, sha256_file
from simulator.cable.residual import MotionResidual, with_acceleration_correction
from planning.pva_job import freeze_model_assets


def prepare(source,output,*,acceleration_limit,frame_regularization=None):
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('Choose a new candidate directory; existing artifacts are never overwritten')
    if not 0<float(acceleration_limit)<float('inf'):
        raise ValueError('Choose a finite positive acceleration bound')
    original=json.loads(source.read_text(encoding='utf-8'))
    model=deepcopy(original)
    if frame_regularization is not None:
        if not math.isfinite(frame_regularization) or frame_regularization<0:
            raise ValueError('Frame regularization must be finite and nonnegative')
        model['cable']['curvature_frame_regularization']=float(frame_regularization)
    for key in ('motion_residual','fullstate_execution'):
        spec=model[key]
        if not spec.get('enabled'):raise ValueError('Both residual model components must be enabled')
        path=Path(spec['checkpoint'])
        if not path.is_absolute():spec['checkpoint']=str((source.parent/path).resolve())
    spec=model['motion_residual'];path=Path(spec['checkpoint'])
    if sha256_file(path)!=spec['sha256']:raise ValueError('Cable source hash mismatch')
    if model['cable']['external_drag_s_inv']!=0:raise ValueError('Do not combine this candidate with fixed drag')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    net=MotionResidual(**payload['specification']).double()
    net.load_state_dict(payload['state_dict'])
    candidate=with_acceleration_correction(net,acceleration_limit=acceleration_limit)
    source_hash=sha256_file(source)
    output.mkdir(parents=True)
    model=freeze_model_assets(model,output)
    weight=output/'assets/cable_residual.pt'
    shutil.copy2(weight,output/'assets/cable_damping_source.pt')
    save_weights(weight,candidate)
    model['motion_residual'].update(sha256=sha256_file(weight),specification=candidate.specification(),drag_mode='nn_only')
    model['adaptation_candidate']=dict(schema='cable_zero_extension_v1',status='UNFITTED',
        source_model=str(source),source_model_sha256=source_hash,
        source_residual_sha256=spec['sha256'],additional_correction_initialization='exactly zero',
        acceleration_bound_per_axis_m_s2=float(acceleration_limit),bound_is_measured=False,
        curvature_frame_regularization=model['cable'].get('curvature_frame_regularization',0.0),
        damping_numerics_changed=(model['cable'].get('curvature_frame_regularization',0.0)
            !=original['cable'].get('curvature_frame_regularization',0.0)),
        trained=False,selected_for_deployment=False)
    atomic_json(output/'model.json',model)
    manifest=dict(model['adaptation_candidate'],schema='unfitted_model_candidate_v1')
    manifest['files']={p.relative_to(output).as_posix():sha256_file(p) for p in output.rglob('*') if p.is_file()}
    atomic_json(output/'manifest.json',manifest)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--acceleration-limit',type=float,required=True,
        help='Per-axis bound in m/s^2 for the additional effective correction (a tuning choice, not a measured limit)')
    parser.add_argument('--frame-regularization',type=float,
        help='Optional new damping numerics; 0 retains legacy, positive smooths the straight/bent transition')
    args=parser.parse_args()
    result=prepare(args.source,args.output,acceleration_limit=args.acceleration_limit,
        frame_regularization=args.frame_regularization)
    print(json.dumps(result,indent=2))
