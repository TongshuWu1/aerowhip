"""Bounded read-only benchmarks. Never trains or edits the source fit job."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import time
from dataclasses import replace
import numpy as np
import torch
from experimental_data.current_adaptation import Trial, read
from experimental_data.current_adaptation_fit import load_engine, cable_windows, join_windows, cable_forward
from simulator.cable import DderState
from simulator.research_physics import research_physics_step
from simulator.cuda_autograd import CudaAutogradBlock
from experimental_data.cuda_cable_fit import CudaCableFit


def setup(job):
    torch.backends.cuda.preferred_linalg_library('cusolver')
    engine = load_engine(job, trainable=True)
    model = read(job/'source_candidate/model.json')
    trials = [Trial(job, p.name, model, end=1.) for p in sorted((job/'inputs').iterdir())]
    records, rejected = cable_windows(trials, engine.physics, [0.], 1.02)
    assert len(records) == 5 and not rejected
    data = join_windows(records)
    checkpoint = torch.load(job/'cable/residual_full_whip/progress.pt', map_location='cuda', weights_only=False)
    engine.physics.motion_residual.load_state_dict(checkpoint['current'])
    physical = read(job/'cable/physics/parameters.json')
    params = data['q'].new_tensor([physical['EI_n_m2'], physical['Cb_n_m2_s']])
    return engine, data, params


def block_benchmark(job, steps, repeats):
    engine, data, params = setup(job)
    physics = engine.physics
    constants = replace(physics.runtime_constants(data['q']),
        bending_stiffness_n_m2=params[0].expand(5), bending_damping_n_m2_s=params[1].expand(5))
    dt = data['q'].new_full((5,), engine.dt_s)
    def advance(q, v, roots):
        qs, vs = [], []
        for root in roots.unbind(1):
            state = research_physics_step(physics, DderState(q, v), root, dt, constants, create_graph=True)
            q, v = state.positions_m, state.velocities_m_s
            qs.append(q); vs.append(v)
        return torch.stack(qs, 1), torch.stack(vs, 1)
    samples = (data['q'], data['v'], data['roots'][:,1:steps+1])
    parameters = tuple(physics.motion_residual.parameters())
    def measure(function):
        values = tuple(x.detach().clone().requires_grad_(True) for x in samples)
        torch.cuda.synchronize(); start = time.perf_counter()
        outputs = function(*values)
        loss = outputs[0].square().sum() + .001*outputs[1].square().sum()
        grads = torch.autograd.grad(loss, values+parameters)
        torch.cuda.synchronize()
        return time.perf_counter()-start, tuple(x.detach().clone() for x in outputs), grads
    base = measure(advance)
    print(json.dumps({'stage':'eager', 'seconds':base[0]}), flush=True)
    torch.cuda.synchronize(); start=time.perf_counter()
    captured = CudaAutogradBlock(advance, samples, parameters)
    torch.cuda.synchronize(); capture_seconds=time.perf_counter()-start
    print(json.dumps({'stage':'captured', 'seconds':capture_seconds}), flush=True)
    accelerated=[measure(captured) for _ in range(repeats)]
    forward_error=max(float((a-b).abs().max()) for a,b in zip(base[1], accelerated[-1][1]))
    gradient_error=max(float((a-b).abs().max()) for a,b in zip(base[2], accelerated[-1][2]))
    for a,b in zip(base[1], accelerated[-1][1]):torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-9)
    for a,b in zip(base[2], accelerated[-1][2]):torch.testing.assert_close(a,b,atol=1e-8,rtol=1e-7)
    return dict(steps=steps, eager_seconds=base[0],capture_seconds=capture_seconds,
        replay_seconds=[r[0] for r in accelerated],maximum_forward_error=forward_error,
        maximum_gradient_error=gradient_error,peak_allocated_bytes=torch.cuda.max_memory_allocated())


def full_benchmark(job,steps,repeats):
    engine,data,params=setup(job)
    parameters=tuple(engine.physics.motion_residual.parameters())
    indices=list(engine.cable.marker_node_indices[1:])
    # Match the production full-whip robust marker/tip objective, including NN penalty.
    model=read(job/'source_candidate/model.json')
    trials=[Trial(job,p.name,model,end=1.) for p in sorted((job/'inputs').iterdir())]
    records,_=cable_windows(trials,engine.physics,[0.],1.02)
    mask=data['q'].new_tensor(np.stack([(r['time'][1:]>=0)&(r['time'][1:]<=1.) for r in records]))
    weights=mask/mask.sum(1,keepdim=True)
    def measure(fn):
        torch.cuda.synchronize();start=time.perf_counter()
        q,v=fn()
        error=q[:,1:,indices]-data['truth'][:,1:]
        robust=2*(torch.sqrt(1+error.square().sum(-1)/.05**2)-1)
        _,extra=engine.physics.motion_residual.components(q[:,1:].flatten(0,1),v[:,1:].flatten(0,1))
        penalty=extra.square().reshape(len(q),len(mask[0]),-1).mean(-1)/.5**2
        loss=((.7*robust.mean(-1)+.3*robust[:,:,-1]+.01*penalty)*weights).sum(1).mean()
        grads=torch.autograd.grad(loss,parameters)
        torch.cuda.synchronize()
        return time.perf_counter()-start,float(loss.detach()),q.detach(),v.detach(),tuple(g.detach().clone() for g in grads)
    base=measure(lambda:cable_forward(engine,data,params,gradients=True))
    print(json.dumps({'stage':'full eager','seconds':base[0],'loss':base[1]}),flush=True)
    start=time.perf_counter();fast=CudaCableFit(engine,data,params,steps)
    torch.cuda.synchronize();capture=time.perf_counter()-start
    results=[]
    for _ in range(repeats):
        result=measure(lambda:fast(data,gradients=True));results.append(result)
        print(json.dumps({'stage':'full graph','seconds':result[0],'loss':result[1]}),flush=True)
    maximum=max(float((a-b).abs().max()) for a,b in zip(base[4],results[-1][4]))
    for a,b in zip(base[4],results[-1][4]):torch.testing.assert_close(a,b,atol=1e-9,rtol=1e-7)
    torch.testing.assert_close(base[2],results[-1][2],atol=1e-10,rtol=1e-9)
    torch.testing.assert_close(base[3],results[-1][3],atol=1e-9,rtol=1e-9)
    return dict(block_steps=steps,eager_seconds=base[0],capture_seconds=capture,
        replay_seconds=[r[0] for r in results],loss=base[1],maximum_gradient_error=maximum,
        maximum_position_error=float((base[2]-results[-1][2]).abs().max()),
        maximum_velocity_error=float((base[3]-results[-1][3]).abs().max()),
        peak_allocated_bytes=torch.cuda.max_memory_allocated())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,default=Path('runs/adaptation/20260909-pva-M0-bootstrap'))
    parser.add_argument('--steps',type=int,default=3);parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(4)
    result=(full_benchmark if args.full else block_benchmark)(args.job,args.steps,args.repeats)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
