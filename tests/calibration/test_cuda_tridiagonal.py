import pytest
import torch
from simulator.cable.cuda_tridiagonal import solve

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('n',[1,11,31,64])
def test_direct_solver_and_adjoint_match_dense_spd_system(n):
    torch.manual_seed(3)
    e=(torch.randn(5,n-1,device='cuda',dtype=torch.float64)*.2).requires_grad_()
    d=(torch.nn.functional.pad(e.detach().abs(),(0,1))+torch.nn.functional.pad(e.detach().abs(),(1,0))+1).requires_grad_()
    b=torch.randn_like(d,requires_grad=True)
    matrix=torch.diag_embed(d)+torch.diag_embed(e,offset=1)+torch.diag_embed(e,offset=-1)
    expected=torch.linalg.solve(matrix,b)
    actual=solve(d,e,b)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    eg=torch.autograd.grad(expected.square().sum(),(d,e,b));ag=torch.autograd.grad(actual.square().sum(),(d,e,b))
    for a,v in zip(ag,eg):torch.testing.assert_close(a,v,atol=1e-12,rtol=1e-12)
