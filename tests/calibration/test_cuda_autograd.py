import pytest
import torch
from simulator.cuda_autograd import CudaAutogradBlock

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_reused_recurrent_graph_preserves_all_inputs_and_parameter_gradients():
    torch.manual_seed(14)
    weight=torch.nn.Parameter(torch.randn(3,3,device='cuda',dtype=torch.float64)*.1)
    def function(x, u):return (torch.tanh(x@weight+u),)
    x=torch.randn(2,3,device='cuda',dtype=torch.float64,requires_grad=True)
    u=torch.randn(2,3,device='cuda',dtype=torch.float64,requires_grad=True)
    block=CudaAutogradBlock(function,(x,u),(weight,))
    def run(fn):
        h=x
        for i in range(5):h=fn(h,u*(i+1))[0]
        return h,torch.autograd.grad(h.square().sum(),(x,u,weight))
    for _ in range(2):
        expected,eg=run(function);actual,ag=run(block)
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
        for a,e in zip(ag,eg):torch.testing.assert_close(a,e,atol=1e-12,rtol=1e-12)
        with torch.no_grad():weight.add_(.01)
