from pathlib import Path
import json
import zipfile
import numpy as np
import torch
from deployment.research_rehearsal import complete_packets,FIELDS


def test_complete_export_preserves_native_whip_and_recovers_continuously():
    t=np.arange(31)/30
    whip=np.zeros((31,11));whip[:,:3]=[0,0,1.5];whip[:,0]=.4*t*t;whip[:,3]=.8*t;whip[:,6]=.8
    times,commands,phases,details=complete_packets(whip,[0,0,1.5])
    np.testing.assert_array_equal(commands[:31],whip)
    np.testing.assert_allclose(np.diff(times),1/30,atol=1e-12)
    np.testing.assert_allclose(commands[-1,:9],[0,0,1.5,0,0,0,0,0,0],atol=1e-10)
    assert phases[:31].tolist()==[1]*31 and phases[-1]==4
    assert details['schema']=='curved_moving_recovery_v1' and details['return_s']>0
    # Independent finite difference of the analytic recovery P/V agrees with supplied V/A.
    from deployment.curved_recovery import plan_curved_recovery
    sample,_=plan_curved_recovery(whip[-1,:3],whip[-1,3:6],whip[-1,6:9],[0,0,1.5])
    x=np.array([.1,.45,1.2]);h=1e-5;p,v,a=sample(x);pm,vm,_=sample(x-h);pp,vp,_=sample(x+h)
    np.testing.assert_allclose((pp-pm)/(2*h),v,atol=1e-8)
    np.testing.assert_allclose((vp-vm)/(2*h),a,atol=1e-8)
    assert FIELDS[-2:]==['yaw_rad','yaw_rate_rad_s']


def test_reference_correction_ignores_unexecuted_tail():
    from simulator.research_reference import reference_packets
    t=torch.arange(16,dtype=torch.float64)/150
    p=torch.zeros(2,16,3,dtype=torch.float64);v=p.clone();v[:,:,0]=1;p[:,:,0]=t
    p[0,6:]=p[0,5]  # The shorter virtual rollout has frozen after its cutoff.
    packets,source=reference_packets(p,v,1/150,torch.tensor([5,15]),[0,0,0])
    torch.testing.assert_close(source['integration_position_correction_m'],torch.zeros(2,dtype=p.dtype),atol=1e-15,rtol=0)
    torch.testing.assert_close(packets[0,1:,:3],p[0,5].expand(3,3))
