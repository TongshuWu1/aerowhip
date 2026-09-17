"""Model lineage registration and explicit same-flight CUDA evaluation."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experimental_data import model_evaluation as evaluation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['seed','register','flight','evaluate'])
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--job',type=Path);parser.add_argument('--comparison',type=Path)
    parser.add_argument('--models',nargs='+');parser.add_argument('--output',type=Path)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    required={'seed':[],'register':['job'],'flight':['comparison'],'evaluate':['job','models','output']}[args.stage]
    for name in required:
        if getattr(args,name) is None:parser.error('--'+name+' is required')
    if args.stage=='seed':evaluation.seed_catalog(args.root)
    elif args.stage=='register':
        from simulator.workflow import read_json
        if read_json(args.job/'protocol.json').get('full_update'):
            from experimental_data.whip_full_fit import register
            print(register(args.job,Path(read_json(args.job/'fit/result.json')['comparison'])))
        else:print(evaluation.add_candidate(args.root,args.job))
    elif args.stage=='flight':evaluation.add_flight_report(args.root,args.comparison)
    else:
        import torch
        torch.set_num_threads(4)
        from simulator.workflow import read_json
        if read_json(args.job/'protocol.json').get('full_update'):
            from experimental_data.whip_full_fit import evaluate_pair
            from experimental_data.whip_adaptation import verify_hashes
            catalog=evaluation.load_catalog(args.root);models={}
            for name in args.models:
                item=next(m for m in catalog['models'] if m['id']==name);verify_hashes(item['hashes']);models[name]=Path(item['model'])
            evaluate_pair(args.job,args.output,models,args.device)
        else:evaluation.evaluate_models(args.root,args.job,args.models,args.output,args.device)


if __name__=='__main__':main()
