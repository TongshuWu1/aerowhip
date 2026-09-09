from copy import deepcopy

import pytest
import torch

from experimental_data.differentiable_fit import save_weights
from experimental_data.io import sha256_file
from simulator.cable.residual import MotionResidual, FrozenMotionResidual, with_acceleration_correction


def fitted_damping():
    torch.manual_seed(34)
    net=MotionResidual(12,hidden=32,mode='dissipative').double()
    with torch.no_grad():
        net.net[-1].weight.normal_(std=.05)
        net.net[-1].bias.fill_(.3)
    return net


def test_extension_preserves_fitted_outputs_source_and_roundtrips(tmp_path):
    original=fitted_damping()
    weights=deepcopy(original.state_dict())
    candidate=with_acceleration_correction(original,acceleration_limit=.5)
    q=torch.randn(3,12,3,dtype=torch.float64);v=torch.randn_like(q)
    torch.testing.assert_close(candidate(q,v),original(q,v),atol=0,rtol=0)
    assert original.mode=='dissipative' and not hasattr(original,'correction_head')
    for key,value in original.state_dict().items():
        torch.testing.assert_close(value,weights[key],atol=0,rtol=0)
        assert value.data_ptr()!=candidate.state_dict()[key].data_ptr()
    path=tmp_path/'candidate.pt';save_weights(path,candidate)
    loaded=FrozenMotionResidual(path,sha256_file(path))
    assert loaded.payload['specification']['mode']=='dissipative_plus_acceleration'
    torch.testing.assert_close(loaded(q,v),candidate(q,v),atol=0,rtol=0)


def test_correction_can_act_at_rest_is_bounded_invariant_and_cannot_move_root():
    net=with_acceleration_correction(fitted_damping(),acceleration_limit=.5)
    q=torch.randn(2,12,3,dtype=torch.float64);v=torch.zeros_like(q)
    with torch.no_grad():
        net.correction_head.weight.normal_(std=5)
        net.correction_head.bias.copy_(torch.linspace(-3,3,33))
    damping,extra=net.components(q,v)
    assert torch.count_nonzero(damping)==0
    assert extra[:,1:].abs().max()<=.5
    assert (extra>0).any() and (extra<0).any()
    assert torch.count_nonzero(extra[:,0])==0
    torch.testing.assert_close(net(q,v),extra)
    torch.testing.assert_close(net(q+q.new_tensor([7.,-2.,3.]),v),extra,atol=1e-12,rtol=0)
    # The new branch is deliberately not claimed to be dissipative.
    v=extra.detach().clone()
    with torch.no_grad():net.net[-1].bias.fill_(-100);net.net[-1].weight.zero_()
    assert (net(q,v)*v).sum()>0


def test_zero_head_has_useful_weight_gradient_and_frozen_input_derivatives(tmp_path):
    net=with_acceleration_correction(fitted_damping(),acceleration_limit=.5)
    q=torch.randn(2,12,3,dtype=torch.float64);v=torch.randn_like(q)
    extra=net.components(q,v)[1]
    g=torch.autograd.grad(extra[:,-1,0].sum(),net.correction_head.bias)[0]
    assert g[-3].item()==1.
    path=tmp_path/'candidate.pt';save_weights(path,net)
    frozen=FrozenMotionResidual(path,sha256_file(path))
    v.requires_grad_()
    grad=torch.autograd.grad(frozen(q,v).square().sum(),v)[0]
    assert torch.isfinite(grad).all() and grad.norm()>0
    assert not any(p.requires_grad for p in frozen._network(q).parameters())


@pytest.mark.parametrize('limit',[0.,-1.,float('nan'),float('inf')])
def test_reject_invalid_bound(limit):
    with pytest.raises(ValueError,match='positive bound'):
        with_acceleration_correction(fitted_damping(),acceleration_limit=limit)


def test_reject_silent_reinterpretation_of_acceleration_checkpoint():
    with pytest.raises(ValueError,match='Only a dissipative'):
        with_acceleration_correction(MotionResidual(12),acceleration_limit=.5)
