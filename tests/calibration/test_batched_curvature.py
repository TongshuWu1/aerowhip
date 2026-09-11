import pytest
import torch
from simulator.cable.dder import _curvature_rate_jacobian_impl


@pytest.mark.parametrize('regularization',[0.,2e-7])
@pytest.mark.parametrize('curved',[False,True])
def test_batched_vertex_geometry_and_gradients(regularization,curved):
    torch.manual_seed(18)
    q=torch.zeros(3,12,3,dtype=torch.float64);q[:,:,2]=-torch.arange(12,dtype=torch.float64)[None]*.09
    if curved:q+=torch.randn_like(q)*.008
    q.requires_grad_();lengths=torch.full((11,),.09,dtype=torch.float64)
    a=_curvature_rate_jacobian_impl(q,lengths,None,None,frame_regularization=regularization)
    b=_curvature_rate_jacobian_impl(q,lengths,None,None,frame_regularization=regularization,vectorized=True)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=1e-12,rtol=1e-12)
    weight=torch.randn_like(a[0])
    ag=torch.autograd.grad((a[0]*weight).sum(),q)[0];bg=torch.autograd.grad((b[0]*weight).sum(),q)[0]
    torch.testing.assert_close(ag,bg,atol=1e-9,rtol=1e-11)
