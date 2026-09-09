import torch
from simulator.cable.residual import MotionResidual,FrozenMotionResidual
from experimental_data.differentiable_fit import save_weights
from experimental_data.io import sha256_file


def test_zero_initialization_without_scalar_drag_and_nonpositive_power():
    torch.manual_seed(3)
    net=MotionResidual(12,hidden=32,mode='dissipative').double()
    q=torch.randn(4,12,3,dtype=torch.float64);v=torch.randn_like(q)
    assert torch.count_nonzero(net(q,v))==0
    assert not hasattr(net,'raw_drag')
    with torch.no_grad():net.net[-1].weight.normal_();net.net[-1].bias.normal_()
    a=net(q,v)
    assert torch.count_nonzero(a[:,0])==0
    assert ((a*v).sum((1,2))<=0).all()
    assert (a.abs()<=2*v.abs()+1e-12).all()


def test_initial_symmetric_gradient_is_nonzero_and_matches_difference():
    net=MotionResidual(12,hidden=32,mode='dissipative').double()
    q=torch.zeros(2,12,3,dtype=torch.float64);v=torch.ones_like(q)
    value=net(q,v).sum();g=torch.autograd.grad(value,net.net[-1].bias)[0]
    assert g.abs().sum()>0
    epsilon=1e-5
    with torch.no_grad():
        net.net[-1].bias.fill_(epsilon);positive=net(q,v).sum()
        net.net[-1].bias.fill_(-epsilon);negative=net(q,v).sum()
    torch.testing.assert_close((positive-negative)/(2*epsilon),g.sum(),rtol=1e-4,atol=1e-6)


def test_dissipative_checkpoint_and_translation_invariance(tmp_path):
    net=MotionResidual(12,hidden=32,mode='dissipative').double()
    with torch.no_grad():net.net[-1].bias.fill_(.1)
    path=tmp_path/'residual.pt';save_weights(path,net)
    frozen=FrozenMotionResidual(path,sha256_file(path))
    q=torch.randn(2,12,3,dtype=torch.float64);v=torch.randn_like(q)
    torch.testing.assert_close(frozen(q,v),net(q,v))
    torch.testing.assert_close(frozen(q+torch.tensor([4.,-3.,2.]),v),net(q,v))
