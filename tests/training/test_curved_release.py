import math
import torch
from planning.free_whip import curved_release,loading_turn_deg,DEFAULT_CURVED_RELEASE


def test_curved_release_rejects_straight_and_opposite_motion_and_brief_alignment():
    load=torch.tensor([[2.,0.,0.]]*5)
    tips=torch.tensor([[4.,0.,0.],[-4.,0.,0.],[0.,4.,0.],[0.,4.,0.],[0.,4.,0.]])
    durations=torch.tensor([.1,.1,.079,.08,.1],dtype=torch.float64)
    allowed,_=curved_release(load,tips,durations,DEFAULT_CURVED_RELEASE)
    assert allowed.tolist()==[False,False,False,True,True]


def test_curved_release_is_yaw_invariant_and_requires_loading():
    load=torch.tensor([[2.,1.,-.5]],dtype=torch.float64)
    tip=torch.tensor([[-1.,5.,.1]],dtype=torch.float64)
    angle=.7;c,s=math.cos(angle),math.sin(angle)
    rotation=torch.tensor([[c,-s,0.],[s,c,0.],[0.,0.,1.]],dtype=torch.float64)
    torch.testing.assert_close(loading_turn_deg(load,tip),loading_turn_deg(load@rotation.T,tip@rotation.T))
    assert not bool(curved_release(load*0,tip,torch.tensor([.1]),DEFAULT_CURVED_RELEASE)[0])


def test_alignment_requires_whole_intervals_and_resets_after_a_gap():
    from planning.free_whip import aligned_interval_duration
    duration=torch.zeros(1,dtype=torch.float64)
    previous=torch.tensor([False])
    for i in range(13):
        current=torch.tensor([True]);duration=aligned_interval_duration(duration,previous,current,1/150)
        previous=current
        assert abs(float(duration)-i/150)<1e-12
    assert abs(float(duration)-.08)<1e-12
    duration=aligned_interval_duration(duration,previous,torch.tensor([False]),1/150)
    assert float(duration)==0.


def test_separate_loading_axis_keeps_release_axis_pullback_requirement():
    from planning.free_whip import advance,DEFAULT_PULLBACK
    peak=torch.tensor([0.]);loaded=torch.tensor([False]);eligible=torch.tensor([True])
    # Forward loading can complete before sideways displacement begins.
    peak,loaded,back,allowed,_=advance(peak,loaded,torch.tensor([0.]),torch.tensor([0.]),eligible,DEFAULT_PULLBACK,
        loading_position=torch.tensor([.3]),loading_speed=torch.tensor([1.5]),loading_allowed=torch.tensor([True]))
    assert bool(loaded) and not bool(allowed)
    peak,loaded,_,allowed,_=advance(peak,loaded,torch.tensor([.4]),torch.tensor([2.]),eligible,DEFAULT_PULLBACK,
        loading_position=torch.tensor([.4]),loading_speed=torch.tensor([0.]),loading_allowed=torch.tensor([False]))
    assert not bool(allowed)
    _,_,back,allowed,_=advance(peak,loaded,torch.tensor([.25]),torch.tensor([-.8]),eligible,DEFAULT_PULLBACK,
        loading_position=torch.tensor([.4]),loading_speed=torch.tensor([0.]),loading_allowed=torch.tensor([False]))
    assert bool(allowed) and float(back)>.1
    _,loaded,_,_,_=advance(torch.tensor([0.]),torch.tensor([False]),torch.tensor([.4]),torch.tensor([2.]),eligible,DEFAULT_PULLBACK,
        loading_position=torch.tensor([.4]),loading_speed=torch.tensor([1.5]),loading_allowed=torch.tensor([False]))
    assert not bool(loaded)  # Sideways motion cannot masquerade as forward loading.
