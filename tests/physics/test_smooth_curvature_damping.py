from dataclasses import replace
import math
import torch
import pytest

from simulator.cable.dder import _curvature_rate_jacobian_impl
from simulator.drone_pose_response import rotation_exp


def cable(angle):
    angles=torch.arange(5,dtype=angle.dtype)*angle
    edges=.1*torch.stack((angles.sin(),torch.zeros_like(angles),-angles.cos()),-1)
    return torch.cat((torch.zeros_like(edges[:1]),edges.cumsum(0)),0)[None]


@pytest.mark.parametrize('angle',[0.,math.acos(1-1e-7),.001])
def test_smooth_damping_gradient_at_straight_and_legacy_switch(angle):
    a=torch.tensor(angle,dtype=torch.float64,requires_grad=True)
    v=torch.linspace(-.2,.3,18,dtype=a.dtype).reshape(1,6,3)
    def loss(a):
        q=cable(a);j,affine,weight=_curvature_rate_jacobian_impl(q,q.new_full((5,),.1),None,None,
            frame_regularization=2e-7)
        rate=(j@v.flatten(1)[...,None])[...,0]+affine
        return (weight*rate.square()).sum()
    gradient=torch.autograd.grad(loss(a),a)[0]
    epsilon=1e-8
    finite_difference=(loss(a.detach()+epsilon)-loss(a.detach()-epsilon))/(2*epsilon)
    assert torch.isfinite(gradient)
    torch.testing.assert_close(gradient,finite_difference,rtol=2e-5,atol=1e-6)


def test_regularized_damping_is_dissipative_translation_free_and_rotation_equivariant():
    q=cable(torch.tensor(.0005,dtype=torch.float64))
    v=torch.linspace(-.2,.3,18,dtype=q.dtype).reshape_as(q)
    def force(q,v):
        j,affine,w=_curvature_rate_jacobian_impl(q,q.new_full((5,),.1),None,None,frame_regularization=2e-7)
        return -(j.transpose(-1,-2)@(w*((j@v.flatten(1)[...,None])[...,0]+affine))[...,None]).reshape_as(q)
    f=force(q,v)
    assert (f*v).sum()<=0
    torch.testing.assert_close(force(q,torch.ones_like(q)),torch.zeros_like(q),atol=1e-10,rtol=0)
    rotation=rotation_exp(q.new_tensor([.2,-.3,.1]))
    torch.testing.assert_close(force(q@rotation.T,v@rotation.T),f@rotation.T,atol=1e-8,rtol=1e-8)


def test_configuration_defaults_to_legacy_and_rejects_bad_regularization():
    from simulator.cable import CableConfiguration
    from pathlib import Path
    import json
    payload=json.loads((Path(__file__).parents[2]/'config/research_30hz/model.json').read_text())['cable']
    config=CableConfiguration.from_mapping(payload)
    assert config.curvature_frame_regularization==0.
    new=replace(config,curvature_frame_regularization=2e-7)
    assert new.dder_parameters(EI=1e-7,Cb=1e-4).curvature_frame_regularization==2e-7
    for value in [-1.,float('nan'),float('inf')]:
        with pytest.raises(ValueError,match='regularization'):replace(config,curvature_frame_regularization=value)
