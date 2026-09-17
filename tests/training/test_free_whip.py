import torch
import numpy as np
import pytest
from types import SimpleNamespace
from planning.free_whip import advance,DEFAULT_PULLBACK
from planning.strike_objective import event_terms,OBJECTIVE
from deployment.braking_recovery import stop_height_range,brake_coefficients

def test_release_requires_loading_then_backward_motion():
    peak=torch.tensor([0.]);loaded=torch.tensor([False]);eligible=torch.tensor([True])
    def step(p,v):
        nonlocal peak,loaded
        peak,loaded,back,allowed,deficit=advance(peak,loaded,torch.tensor([p]),torch.tensor([v]),eligible,DEFAULT_PULLBACK)
        return bool(allowed),float(back)
    assert not step(-.2,-1.)[0]  # Backward motion alone is insufficient.
    assert not step(.4,2.)[0]    # Loading alone is insufficient.
    assert not step(.45,.2)[0]  # Slowing while moving forward is insufficient.
    assert not step(.43,-1.)[0] # Reversal without enough travel is insufficient.
    ok,back=step(.30,-1.)
    assert ok and abs(back-.15)<1e-6

def test_free_target_score_does_not_reward_posthoc_target_choice():
    settings=dict(OBJECTIVE,free_target=True)
    args=(torch.tensor([[0.,4.,0.]]),torch.tensor([0.,1.,0.]),settings)
    near=event_terms(torch.tensor([0.]),*args)
    far=event_terms(torch.tensor([100.]),*args)
    assert near.keys()==far.keys()
    for key in near:torch.testing.assert_close(near[key],far[key])

def test_braking_endpoint_bounds_contain_all_durations():
    rng=np.random.default_rng(4)
    for _ in range(100):
        p,v,a=rng.normal(size=3);lo,hi=stop_height_range(p,v,a,1.3,6.)
        times=np.linspace(1.3,6.,201)
        heights=np.array([brake_coefficients(np.array([0.,0.,p]),np.array([0.,0.,v]),np.array([0.,0.,a]),t).sum(0)[2] for t in times])
        assert heights.min()>=lo-1e-12 and heights.max()<=hi+1e-12
        assert lo<=hi

@pytest.mark.parametrize('earliest',[0.,.03])
def test_free_event_selection_requires_release_and_is_target_independent(earliest):
    from planning.strike_objective import initialize,observe,FOLD,StrikeCapture
    s=torch.linspace(0,1,11,dtype=torch.float64)
    q=torch.stack((s*0,s*0,-s),-1)[None]
    settings=dict(OBJECTIVE,free_target=True,pullback=dict(DEFAULT_PULLBACK),
                  speed_metric='tip_gain_over_root',minimum_tip_speed_gain_m_s=2.,minimum_release_time_s=earliest)
    results=[]
    for target in ([0.,0.,0.],[100.,-50.,20.]):
        env=SimpleNamespace(active=torch.tensor([True]),total=s.new_zeros(1),cutoff=torch.zeros(1,dtype=torch.long),
            failed=torch.tensor([False]),target=s.new_tensor([target]),dt=.01,wave_material=s[1:-1],
            settings=dict(fold_constraint=FOLD,trajectory_objective=settings,fold_requirement='diagnostic_only'),
            direction=s.new_tensor([0.,1.,0.]),encounter_distance=s.new_full((1,),float('inf')),
            encounter_tip_velocity=s.new_zeros(1,3),encounter_time=s.new_zeros(1),encounter_q=q.clone(),
            initial_state=SimpleNamespace(positions_m=q.clone()))
        initialize(env)
        previous=SimpleNamespace(positions_m=q.clone(),velocities_m_s=torch.zeros_like(q))
        for clock,root_y,root_v in [(0.,.4,2.),(.01,.25,-1.),(.02,.24,-1.)]:
            current=SimpleNamespace(positions_m=q.clone(),velocities_m_s=torch.zeros_like(q))
            current.positions_m[:,:,1]=root_y+s
            current.velocities_m_s[:,0,1]=root_v;current.velocities_m_s[:,-1,1]=4.
            observe(env,previous,current,env.active,s.new_tensor([clock]))
            assert bool(env.strike_valid[0])==(clock>0 and clock+.01>=earliest)
            previous=current
        score,_=StrikeCapture().score(env,dict(failed=env.failed),s.new_zeros(1,2,3),settings)
        results.append((score,env.strike_time,env.strike_position,env.strike_backward_travel))
    for a,b in zip(*results):torch.testing.assert_close(a,b)

def test_velocity_peak_proxy_rejects_synchronized_rotation_and_stale_pulses():
    from planning.free_whip import propagation,DEFAULT_PROPAGATION,group_speeds
    peaks=torch.tensor([[1.9,2.04,3.15]]*4)
    times=torch.tensor([[.66,.84,1.047],[1.,1.,1.],[.66,.84,1.047],[1.,.84,.66]])
    clock=torch.tensor([1.113,1.1,2.,1.2])
    allowed,_=propagation(peaks,times,clock,DEFAULT_PROPAGATION)
    assert allowed.tolist()==[True,False,False,False]
    # A material-linear rotating velocity field has increasing speed but all
    # bands peak simultaneously. Speed amplification alone cannot pass.
    material=torch.linspace(0,1,12,dtype=torch.float64)
    v=torch.zeros(1,12,3,dtype=torch.float64);v[0,:,0]=material*6
    bands=group_speeds(v,material,torch.tensor([1.,0.,0.],dtype=torch.float64))
    assert bool((bands[:,1:]>bands[:,:-1]).all())
    assert not bool(propagation(bands,torch.ones_like(bands),torch.tensor([1.1]),DEFAULT_PROPAGATION)[0])
