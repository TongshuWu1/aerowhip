from copy import deepcopy
from pathlib import Path
from dataclasses import asdict
import pytest
import torch
from experimental_data.current_adaptation import read
from learning.pva_env import PVAEnvironment,defaults

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
ROOT=Path(__file__).resolve().parents[2]


def test_accelerated_ticks_preserve_hits_failures_masks_and_resets():
    path=ROOT/'runs/adaptation/20260909-pva-M0-bootstrap/source_candidate/model.json'
    if not path.exists():pytest.skip('Cold bootstrap input unavailable')
    cfg=defaults('mppi');cfg['task'].update(duration_s=.4,require_pullback=False,success_criterion='legacy_strike_v1')
    model=read(path)
    def make(fast,batch):
        return PVAEnvironment(model,cfg,root=path.parent,batch_size=batch,
            fused_ticks=fast,fast_geometry=fast,fast_solve=fast)
    pilot=make(False,1)
    actions=torch.zeros(1,12,3,device='cuda');actions[:,:,0]=.4
    pilot.rollout(actions=actions,trace=True)
    # Place a tiny virtual target on a known moving tip segment to exercise a
    # real modeled hit, rather than testing only no-hit hover trajectories.
    points=torch.stack([f['cable'][0,-1] for f in pilot.frames])
    differences=points[1:]-points[:-1];k=int(differences.norm(dim=-1).argmax())
    delta=differences[k]
    cfg['task']['target_radius_m']=float(delta.norm())*.2
    cfg['task']['minimum_directed_speed_m_s']=1e-8
    cfg['task']['maximum_angle_deg']=89.
    cfg['task']['strike_direction']=(delta/delta.norm()).tolist()
    cfg['launch']['target_m']=((points[k]+points[k+1])*.5).tolist()
    reference=make(False,3);fast=make(True,3)
    for reset_number in range(2):
        origin=reference.tensor(cfg['launch']['origin_m'])[None].repeat(3,1)
        origin[1,2]=.5 # Explicit workspace failure in one independent row.
        reference.reset(origin=origin);fast.reset(origin=origin)
        for step in range(12):
            command=actions[:,step].expand(3,-1)
            expected=reference.step(command);actual=fast.step(command)
            for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-6)
            for name in ('success','failed','active','cutoff'):
                torch.testing.assert_close(getattr(fast,name),getattr(reference,name),atol=0,rtol=0)
            torch.testing.assert_close(fast.termination_time,reference.termination_time,atol=1e-9,rtol=0)
            torch.testing.assert_close(fast.state.positions_m,reference.state.positions_m,atol=1e-9,rtol=0)
        assert reference.success[0] and reference.failed[1]
