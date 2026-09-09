"""Compare fixed-shape eager and captured physics without changing a policy."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gc
import argparse
import time
import torch
from simulator.workflow import read_json
from simulator.point_mass import ForceControlledPointCable
from simulator.cuda_graph_physics import CudaGraphPhysics
from experimental_data.io import atomic_json


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser()
    parser.add_argument('--batches',type=int,nargs='+',default=[512,1024,2048,4096])
    parser.add_argument('--output',type=Path,default=root/'results/benchmarks/gpu_graph_physics.json')
    parser.add_argument('--graph-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    rows=[]
    for batch in args.batches:
        if not 1<=batch<=32768:raise ValueError('Benchmark batch must be between 1 and 32768.')
        torch.cuda.reset_peak_memory_stats()
        model_config=read_json(root/'config/model.json')
        model=ForceControlledPointCable.from_mapping(model_config)
        dt=model_config['simulation']['dt_s']
        state=model.hanging_state(torch.tensor([[0.,0.,1.5]],device='cuda',dtype=torch.float64).expand(batch,3))
        force=model.hover_force_world_n(dtype=torch.float64,device='cuda')[None].expand(batch,3).clone()
        force[:,0]=.2;force[:,1]=.1
        start=time.perf_counter();graph=CudaGraphPhysics(model,state,dt);torch.cuda.synchronize()
        row={'batch':batch,'capture_s':time.perf_counter()-start}
        reference=model.step_runtime(state,force,dt).state
        actual=graph(state,force)
        row['maximum_position_error_m']=float((reference.positions_m-actual.positions_m).abs().max())
        assert row['maximum_position_error_m']<1e-10
        for label,fn in [('eager',lambda:model.step_runtime(state,force,dt)),('graph',lambda:graph(state,force))]:
            if args.graph_only and label=='eager':continue
            for _ in range(2):fn()
            torch.cuda.synchronize();start=time.perf_counter()
            for _ in range(10):fn()
            torch.cuda.synchronize();row[label+'_seconds_per_physics_step']=(time.perf_counter()-start)/10
        if not args.graph_only:row['speedup']=row['eager_seconds_per_physics_step']/row['graph_seconds_per_physics_step']
        row['peak_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3
        row['graph_environment_steps_per_s']=batch/row['graph_seconds_per_physics_step']
        rows.append(row);print(row,flush=True)
        atomic_json(args.output,{
            'scope':'Physics step only; excludes scoring, actor update, validation and graph creation.',
            'precision':'float64','substeps':model.cable_configuration.substeps,'results':rows})
        del graph,model,state,force,reference,actual,fn
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
