"""Fused float64 linear solves for a single free 12-node cable rehearsal.

Same 32-iteration Jacobi-PCG and Thomas recurrences as the graph reference;
no material parameters, timestep, iteration budget or constraint count change.
"""
import ctypes
import ctypes.util
import os
import re

import torch

from .cuda_fixed_pcg import _FixedPcgKernel, _nvrtc_library_path, _specialized_mechanics_source


def mechanics_source():
    source = _specialized_mechanics_source(12, pinned_start_nodes=0, iterations=32)
    source = re.sub(r'\bfloat\b', 'double', source)
    source = re.sub(r'\b(sqrt|rsqrt|fabs|fmax|fmin|acos|sin|cos|atan2)f\b', r'\1', source)
    source = re.sub(r'(?<=[0-9])f\b', '', source)
    source = source.replace('7.62939453125e-6', '1.4210854715202004e-14')
    source = source.replace('1.52587890625e-5', '2.842170943040401e-14')
    source = source.replace('9.5367431640625e-7', '1.7763568394002505e-15')
    return source

class RehearsalSolvers(_FixedPcgKernel):
    def __init__(self):
        path = _nvrtc_library_path()
        if hasattr(os, 'add_dll_directory'):
            self._dll_directory = os.add_dll_directory(str(path.parent))
        loader = ctypes.WinDLL if os.name == 'nt' else ctypes.CDLL
        self._nvrtc = loader(str(path))
        self._cuda = loader('nvcuda.dll' if os.name == 'nt' else (ctypes.util.find_library('cuda') or 'libcuda.so.1'))
        self._configure_signatures()
        self._check_cuda(self._cuda.cuInit(0), 'cuInit')
        # CUDA's current context is thread-local. A new Qt rehearsal worker
        # may reuse cached tensors without making a context current. Explicitly
        # select PyTorch's device before loading our driver-API module.
        torch.cuda.set_device(torch.cuda.current_device())
        major, minor = torch.cuda.get_device_capability()
        ptx = self._compile_ptx(f'--gpu-architecture=compute_{major}{minor}',
                               source=mechanics_source(), source_name='rehearsal_solvers.cu')
        self._module = ctypes.c_void_p()
        self._check_cuda(self._cuda.cuModuleLoadDataEx(ctypes.byref(self._module),
            ctypes.c_char_p(ptx), 0, None, None), 'cuModuleLoadDataEx')
        self.functions = {}
        for name in ('fixed_damping_12node_0pinned_32pcg',
                     'fixed_projection_12node_0pinned_4plus1'):
            function = ctypes.c_void_p()
            self._check_cuda(self._cuda.cuModuleGetFunction(ctypes.byref(function),
                self._module, name.encode()), 'cuModuleGetFunction')
            self.functions[name] = function

    def mechanics(self, name, inputs, outputs, threads):
        if any(t.dtype != torch.float64 or not t.is_cuda for t in inputs):
            raise ValueError('Fused rehearsal mechanics requires float64 CUDA tensors.')
        inputs = [t.contiguous() for t in inputs]
        values = [ctypes.c_void_p(t.data_ptr()) for t in (*inputs, *outputs)]
        values.append(ctypes.c_int(outputs[0].shape[0]))
        arguments = (ctypes.c_void_p * len(values))(*[
            ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in values])
        stream = ctypes.c_void_p(torch.cuda.current_stream(outputs[0].device).cuda_stream)
        self._check_cuda(self._cuda.cuLaunchKernel(self.functions[name], outputs[0].shape[0],1,1,
            threads,1,1,0,stream,arguments,None), name)
        return outputs

    def damping(self, q, v, boundary, lengths, masses, dt, damping):
        if q.shape[1:] != (12,3) or boundary.shape[1:] != (0,3):
            raise ValueError('Fused rehearsal requires 12 free nodes.')
        return self.mechanics('fixed_damping_12node_0pinned_32pcg',
            [q,v,boundary,lengths,masses,dt,damping], [torch.empty_like(q)], 64)[0]

    def project(self, predicted, previous, lengths, masses, boundary, boundary_v, dt):
        return self.mechanics('fixed_projection_12node_0pinned_4plus1',
            [predicted,previous,lengths,masses,boundary,boundary_v,dt],
            [torch.empty_like(predicted),torch.empty_like(predicted)], 32)
