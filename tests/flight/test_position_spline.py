import json
from pathlib import Path
import numpy as np
import pytest
import torch
from planning.position_spline import PositionSpline, SCHEMA, replay
from deployment.braking_recovery import DEFAULTS, plan
from deployment.pva_rehearsal import complete_pva_packets

ROOT=Path(__file__).resolve().parents[2]


def test_spline_derivatives_start_and_knot_continuity():
    spline=PositionSpline(1.5)
    origin=torch.tensor([1.,-.2,1.255],dtype=torch.float64)
    free=origin+torch.arange(9,dtype=torch.float64)[:,None]*torch.tensor([.01,.002,.005])
    packets,jerk=spline.decode(free,origin)
    assert packets.shape==(46,11) and jerk.shape==(45,3)
    torch.testing.assert_close(packets[0,:3],origin)
    torch.testing.assert_close(packets[0,3:9],torch.zeros(6,dtype=torch.float64),atol=1e-10,rtol=0)
    points=spline.control_points(free,origin).numpy()
    t=.72;h=1e-5
    for k in (0,1,2):
        numerical=(spline.basis(t+h,nu=k)@points-spline.basis(t-h,nu=k)@points)/(2*h)
        np.testing.assert_allclose(numerical,spline.basis(t,nu=k+1)@points,atol=1e-6,rtol=1e-6)
    for t in np.unique(spline.knots)[1:-1]:
        for k in range(5):np.testing.assert_allclose(spline.basis(t-1e-10,nu=k)@points,spline.basis(t+1e-10,nu=k)@points,atol=2e-5,rtol=1e-6)
    assert packets[-1,3:6].norm()>0  # Follow-through is not forced to rest.


def test_continuous_jerk_bound_includes_between_packet_samples():
    spline=PositionSpline(1.5);origin=torch.zeros(3,dtype=torch.float64)
    free=torch.randn(3,9,3,generator=torch.Generator().manual_seed(7),dtype=torch.float64)*.1
    points=spline.control_points(free,origin)
    bound=torch.einsum('kc,bcd->bkd',spline.jerk_control_basis,points).abs().amax(1)
    dense=torch.tensor(spline.basis(np.linspace(0,1.5,2001),nu=3))
    assert bool((torch.einsum('tc,bcd->btd',dense,points).abs().amax(1)<=bound+1e-8).all())
    assert not bool(spline.jerk_valid(free,origin,torch.ones(3)).any())


def test_brake_return_preserves_pva_and_has_no_return_loop():
    cfg=json.loads((ROOT/'tests/fixtures/targeted_strike_settings.json').read_text())
    p=np.array([.7,.05,1.8]);v=np.array([1.,.1,.2]);a=np.array([.2,0.,-.1]);hover=np.array([0.,0.,1.255])
    sample,m=plan(p,v,a,hover,cfg['limits'],[60]*3,DEFAULTS)
    points,velocity,acceleration=sample(np.array([0.,m['brake_end_s'],m['return_end_s'],m['total_duration_s']]))
    np.testing.assert_allclose(points[0],p,atol=1e-12);np.testing.assert_allclose(velocity[0],v);np.testing.assert_allclose(acceleration[0],a)
    np.testing.assert_allclose(velocity[1:],0,atol=1e-12);np.testing.assert_allclose(acceleration[1:],0,atol=1e-12)
    np.testing.assert_allclose(points[-1],hover)
    from deployment.curved_recovery import evaluate
    pb,vb,ab=evaluate(np.array(m['brake_coefficients_normalized']),np.array([m['brake_end_s']]),m['brake_end_s'])
    np.testing.assert_allclose(pb[0],points[1],atol=1e-12);np.testing.assert_allclose(vb,0,atol=1e-12);np.testing.assert_allclose(ab,0,atol=1e-12)
    t=np.linspace(m['brake_end_s'],m['return_end_s'],200)
    pp,_,_=sample(t);delta=hover-points[1];fraction=((pp-points[1])@delta)/(delta@delta)
    assert np.diff(fraction).min()>-1e-12
    np.testing.assert_allclose(pp,points[1]+fraction[:,None]*delta,atol=1e-12)
    whip=np.tile(np.r_[p,v,a,0.,0.],(46,1))
    times,packets,phases,_=complete_pva_packets(whip,hover,cfg['limits'],recovery_settings=DEFAULTS,jerk_limits=[60]*3)
    np.testing.assert_array_equal(packets[:46],whip)
    np.testing.assert_allclose(np.diff(times),1/30,atol=1e-12)
    assert set(phases)=={1,2,3,4}


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_spline_coupled_replay_matches_export_pose_and_exact_packets(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    cfg=json.loads((ROOT/'tests/fixtures/targeted_strike_settings.json').read_text())
    path=ROOT/cfg['model_path']
    if not path.exists():pytest.skip('Private M0 fixture unavailable')
    from learning.pva_env import PVAEnvironment
    env=PVAEnvironment(json.loads(path.read_text()),cfg,root=path.parent,device=device)
    spline=PositionSpline(1.5,device=device);origin=env.origin0[0]
    free=origin+env.tensor(np.linspace(0,.04,9))[:,None]*env.tensor([1.,.1,0.])
    expected,jerk=spline.decode(free,origin)
    result=replay(env,free[None],trace=True)
    assert not bool(result['failed'][0]);assert not bool(result['fold_valid'][0])
    torch.testing.assert_close(result['packets'][0],expected,atol=1e-10,rtol=0)
    times,packets,_,_=complete_pva_packets(expected.cpu().numpy(),origin.cpu().numpy(),cfg['limits'],recovery_settings=cfg['recovery'],jerk_limits=[60]*3)
    grid=np.arange(round(times[-1]/env.dt)+1)*env.dt
    pose=env.engine.drone.predict(env.initial_pose,env.tensor(packets)[None],times,grid,env.engine.offset,hover_command=env.hover,maximum_tilt_deg=60)
    assert bool(pose['valid'].all())
    if device=='cuda':
        state=env.initial_state
        for k in range(1,len(grid)):
            state=env.cable_stepper(state,pose['position_attachment_m'][:,k])
            assert bool(torch.isfinite(state.positions_m).all() & torch.isfinite(state.velocities_m_s).all())
            assert float(state.positions_m[:,:,2].min())>=cfg['limits']['minimum_cable_z_m']
    torch.testing.assert_close(torch.stack([f['origin'] for f in env.frames],1),pose['position_origin_m'][:,1:len(env.frames)+1],atol=1e-9,rtol=0)


def test_spline_export_rejects_tampered_packets_before_loading_model(tmp_path):
    from deployment.pva_rehearsal import generate
    from experimental_data.io import atomic_json
    cfg=json.loads((ROOT/'tests/fixtures/targeted_strike_settings.json').read_text())
    atomic_json(tmp_path/'settings.json',cfg);atomic_json(tmp_path/'model.json',{})
    atomic_json(tmp_path/'status.json',dict(status='completed'))
    spline=PositionSpline(1.5);free=torch.tensor(cfg['launch']['origin_m'],dtype=torch.float64).expand(9,3).clone()
    packets,_=spline.decode(free,cfg['launch']['origin_m']);packets[20,0]+=.01
    np.savez(tmp_path/'plan.npz',position_control_points_m=free.numpy(),command_packets=packets.numpy(),plan_complete=True,committed_steps=45)
    with pytest.raises(ValueError,match='Saved spline packets differ'):
        generate(tmp_path,tmp_path/'output',device='cpu')
    assert not (tmp_path/'output').exists()


def test_spline_recovery_rejects_impossible_height_without_resetting_exit():
    cfg=json.loads((ROOT/'tests/fixtures/targeted_strike_settings.json').read_text())
    with pytest.raises(ValueError,match='No smooth braking'):
        plan([0.,0.,cfg['limits']['maximum_origin_z_m']],[0.,0.,1.],[0.,0.,0.],
             cfg['launch']['origin_m'],cfg['limits'],[60]*3,DEFAULTS)


def test_scratch_proposals_use_only_launch_geometry_and_rng():
    from planning.position_spline import PositionSpline,scratch_proposals
    import torch
    spline=PositionSpline(1.5);basis=spline.scratch_noise_basis()
    assert int(torch.linalg.matrix_rank(basis))==9
    torch.testing.assert_close(basis.square().sum(-1).sqrt().max(),torch.tensor(1.,dtype=torch.float64))
    origin=torch.tensor([0.,0.,1.4],dtype=torch.float64);means=origin.expand(4,9,3).clone()
    cfg=dict(samples=512,position_noise_scales_m=[.1,.4,1.],fresh_sample_fraction=.25)
    a=scratch_proposals(means,origin,basis,cfg,torch.Generator().manual_seed(657))
    b=scratch_proposals(means+10,origin,basis,cfg,torch.Generator().manual_seed(657))
    # Fresh draws do not inherit even the current search means.
    torch.testing.assert_close(a[:,:32],b[:,:32])
    torch.testing.assert_close(a[:,32:]+10,b[:,32:])
    pva,_=spline.decode(a,origin)
    torch.testing.assert_close(pva[:,:,0,:3],origin.expand(4,128,3))
    torch.testing.assert_close(pva[:,:,0,3:9],torch.zeros(4,128,6,dtype=torch.float64),atol=1e-10,rtol=0)
    assert float(pva[...,0].max())>1.4 and float(pva[...,0].min())<-.5
    assert float(pva[...,3].max())>3.  # Broad forward velocity exploration, not just a stationary seed.
