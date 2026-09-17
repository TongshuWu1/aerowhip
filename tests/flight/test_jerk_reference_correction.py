import numpy as np
import torch
from planning.jerk_reference_correction import JerkCorrection
from planning.local_reference_correction import active_spline_basis
from simulator.pva_commands import jerk_packets


def adapter():
    rng=np.random.default_rng(31)
    initial=np.zeros(11);initial[:3]=[0.,0.,1.7917713383723264]
    return JerkCorrection(initial,rng.normal(size=(46,3))*10)


def test_zero_correction_preserves_original_packets_exactly():
    a=adapter();zero=torch.zeros((9,3),dtype=torch.float64)
    packets,jerk=a.decode(zero,[0,0,1.7917713383723264])
    assert torch.equal(packets,a.original_packets)
    assert torch.equal(jerk,a.original_jerk)


def test_correction_matches_native_integration_and_preserves_initial_pva():
    a=adapter();free=torch.randn((2,9,3),dtype=torch.float64)
    packets,jerk=a.decode(free,None)
    expected=jerk_packets(a.original_packets[0].expand(2,-1),jerk)
    torch.testing.assert_close(packets,expected,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(packets[:,0],a.original_packets[0].expand(2,-1),rtol=0,atol=0)


def test_packet_and_jerk_maps_are_exact_and_bounds_include_original_jerk():
    a=adapter();basis,active=active_spline_basis(a,47)
    assert basis.shape==(27,27) and len(active)==9
    free=torch.as_tensor((basis@np.ones(27)).reshape(9,3))
    packets,jerk=a.decode(free,None)
    for i,b in enumerate(a.matrices):
        torch.testing.assert_close(packets[:,3*i:3*i+3]-a.original_packets[:,3*i:3*i+3],b[:,3:]@free)
    torch.testing.assert_close(jerk-a.original_jerk,a.jerk_control_basis[:,3:]@free)
    assert bool(a.jerk_valid(torch.zeros_like(free),None,torch.tensor([60.,60.,60.])))
    assert not bool(a.jerk_valid(torch.ones_like(free)*200,None,torch.tensor([60.,60.,60.])))
