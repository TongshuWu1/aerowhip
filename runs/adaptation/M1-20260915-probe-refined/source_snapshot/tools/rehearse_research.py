"""Run isolated GPU rehearsal so Qt cannot invalidate the CUDA context."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experimental_data.io import atomic_json
from deployment.research_rehearsal import generate

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--origin',nargs=3,type=float,required=True);parser.add_argument('--target',nargs=3,type=float,required=True)
    parser.add_argument('--device',default='cuda');args=parser.parse_args()
    def progress(label,step,total):atomic_json(Path(args.output)/'progress.json',dict(label=label,step=step,total=total))
    generate(args.checkpoint,args.output,args.origin,args.target,args.device,progress)
