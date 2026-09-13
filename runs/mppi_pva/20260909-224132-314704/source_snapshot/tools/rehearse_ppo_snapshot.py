from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from simulator.workflow import read_json
from experimental_data.io import atomic_json


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--complete',action='store_true')
    args=parser.parse_args()
    if args.complete:
        from deployment.pva_rehearsal import generate
        result=generate(args.job,args.output,checkpoint=args.job/'checkpoints/policy.pt',progress=lambda label,*_:print(label,flush=True))
        result['policy_snapshot']=read_json(args.job/'policy_snapshot.json')
        atomic_json(args.output/'rehearsal.json',result)
    else:
        from deployment.pva_policy_preview import generate_preview
        print('Replaying frozen policy',flush=True);result=generate_preview(args.job,args.output)
    print('Saved: '+str(args.output),flush=True)
