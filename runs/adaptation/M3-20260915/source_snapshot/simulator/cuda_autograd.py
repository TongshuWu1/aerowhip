"""Reusable CUDA forward/VJP graphs with explicit saved inputs.

Each backward replay recomputes its own forward from saved inputs. Unlike
reusing a captured autograd tape, this is safe for recurrent applications of
the same block before any backward. First-order derivatives only. Parameters
are explicit Function inputs, so ordinary autograd accumulates their gradients.
"""
import torch
from torch.autograd.function import once_differentiable


class CudaAutogradBlock:
    def __init__(self, function, examples, parameters=()):
        self.function = function
        self.parameters = tuple(parameters)
        if not examples or any(not x.is_cuda or not x.is_floating_point() for x in examples):
            raise ValueError('CUDA floating-point inputs required')
        # The Windows MAGMA Cholesky path allocates outside the graph pool.
        # This is also the backend used by the existing inference graph path.
        torch.backends.cuda.preferred_linalg_library('cusolver')
        self.inputs = tuple(x.detach().clone().requires_grad_(True) for x in examples)
        self.signature = tuple((x.shape, x.dtype, x.device) for x in examples)
        # Warm up allocations and linear algebra on a side stream.
        stream = torch.cuda.Stream(device=examples[0].device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.enable_grad():
            for _ in range(2):
                outputs = tuple(function(*self.inputs))
                cotangents = tuple(torch.ones_like(x) for x in outputs)
                torch.autograd.grad(outputs, self.inputs + self.parameters, cotangents)
        torch.cuda.current_stream().wait_stream(stream)
        self.cotangents = tuple(torch.zeros_like(x) for x in outputs)
        del outputs, cotangents
        self.forward_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.forward_graph, stream=stream), torch.no_grad():
            self.outputs = tuple(function(*self.inputs))
        self.backward_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.backward_graph, stream=stream), torch.enable_grad():
            outputs = tuple(function(*self.inputs))
            self.gradients = torch.autograd.grad(outputs, self.inputs + self.parameters, self.cotangents)

    def __call__(self, *inputs):
        if tuple((x.shape, x.dtype, x.device) for x in inputs) != self.signature:
            raise ValueError('Captured input signature changed')
        return _RecomputedCudaBlock.apply(self, len(inputs), *inputs, *self.parameters)


class _RecomputedCudaBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, block, count, *values):
        ctx.block = block
        ctx.count = count
        ctx.save_for_backward(*values)
        for target, value in zip(block.inputs, values[:count]):
            target.copy_(value)
        block.forward_graph.replay()
        return tuple(x.clone() for x in block.outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, *cotangents):
        block = ctx.block
        values = ctx.saved_tensors
        for target, value in zip(block.inputs, values[:ctx.count]):
            target.copy_(value)
        for target, value in zip(block.cotangents, cotangents):
            if value is None:
                target.zero_()
            else:
                target.copy_(value)
        block.backward_graph.replay()
        return (None, None, *(g.clone() if needed else None for g, needed in zip(block.gradients, ctx.needs_input_grad[2:])))
