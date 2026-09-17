"""Float64 batched SPD tridiagonal solve and its exact first-order adjoint.

Specialized direct elimination for the positive, regularized constraint
systems. No iteration-count approximation or change to the constraints.
"""
import ctypes
import ctypes.util
import os
import torch
from torch.autograd.function import once_differentiable
from .cuda_fixed_pcg import _FixedPcgKernel,_nvrtc_library_path

SOURCE=r'''
extern "C" __global__ void tridiagonal(const double* d,const double* e,
    const double* b,double* x,int batch,int n) {
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=batch) return;
    double c[64],y[64];
    int base=row*n,off=row*(n-1);
    double pivot=d[base];
    c[0]=n>1?e[off]/pivot:0.;y[0]=b[base]/pivot;
    for(int i=1;i<n;++i){
        pivot=d[base+i]-e[off+i-1]*c[i-1];
        c[i]=i<n-1?e[off+i]/pivot:0.;
        y[i]=(b[base+i]-e[off+i-1]*y[i-1])/pivot;
    }
    x[base+n-1]=y[n-1];
    for(int i=n-2;i>=0;--i)x[base+i]=y[i]-c[i]*x[base+i+1];
}
'''


class Kernel(_FixedPcgKernel):
    def __init__(self):
        path=_nvrtc_library_path()
        if hasattr(os,'add_dll_directory'):self._dll_directory=os.add_dll_directory(str(path.parent))
        loader=ctypes.WinDLL if os.name=='nt' else ctypes.CDLL
        self._nvrtc=loader(str(path))
        self._cuda=loader('nvcuda.dll' if os.name=='nt' else (ctypes.util.find_library('cuda') or 'libcuda.so.1'))
        self._configure_signatures();self._check_cuda(self._cuda.cuInit(0),'cuInit')
        torch.cuda.set_device(torch.cuda.current_device());major,minor=torch.cuda.get_device_capability()
        ptx=self._compile_ptx(f'--gpu-architecture=compute_{major}{minor}',source=SOURCE,source_name='tridiagonal.cu')
        self._module=ctypes.c_void_p()
        self._check_cuda(self._cuda.cuModuleLoadDataEx(ctypes.byref(self._module),ctypes.c_char_p(ptx),0,None,None),'load tridiagonal')
        self.function=ctypes.c_void_p()
        self._check_cuda(self._cuda.cuModuleGetFunction(ctypes.byref(self.function),self._module,b'tridiagonal'),'find tridiagonal')

    def __call__(self,d,e,b):
        d,e,b=d.contiguous(),e.contiguous(),b.contiguous();x=torch.empty_like(b)
        values=[ctypes.c_void_p(t.data_ptr()) for t in (d,e,b,x)]+[ctypes.c_int(len(d)),ctypes.c_int(d.shape[1])]
        args=(ctypes.c_void_p*len(values))(*(ctypes.cast(ctypes.byref(v),ctypes.c_void_p) for v in values))
        stream=ctypes.c_void_p(torch.cuda.current_stream(d.device).cuda_stream)
        self._check_cuda(self._cuda.cuLaunchKernel(self.function,(len(d)+63)//64,1,1,64,1,1,0,stream,args,None),'tridiagonal')
        return x


_kernels={}


def kernel():
    device=torch.cuda.current_device()
    if device not in _kernels:_kernels[device]=Kernel()
    return _kernels[device]


class _Solve(torch.autograd.Function):
    @staticmethod
    def forward(ctx,d,e,b):
        x=kernel()(d,e,b);ctx.save_for_backward(d,e,x);return x

    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        d,e,x=ctx.saved_tensors;adjoint=kernel()(d,e,g)
        return -adjoint*x,-(adjoint[:,:-1]*x[:,1:]+adjoint[:,1:]*x[:,:-1]),adjoint


def solve(d,e,b):
    if d.ndim!=2 or d.shape!=b.shape or e.shape!=(len(d),d.shape[1]-1) or not 1<=d.shape[1]<=64:
        raise ValueError('Expected batched tridiagonal systems with 1..64 unknowns')
    if any(t.dtype!=torch.float64 or not t.is_cuda or t.device!=d.device for t in (d,e,b)):
        raise ValueError('Float64 CUDA tridiagonal systems required')
    return _Solve.apply(d,e,b)
