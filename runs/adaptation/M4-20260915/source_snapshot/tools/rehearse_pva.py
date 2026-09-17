from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from deployment.pva_rehearsal import generate
from experimental_data.io import atomic_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--origin',nargs=3,type=float);p.add_argument('--target',nargs=3,type=float)
    p.add_argument('--device',default='cuda');a=p.parse_args()
    # Status is separate: the completed artifact directory is created atomically by generation only after checks.
    def progress(label,step,total):print(label,flush=True)
    generate(a.job,a.output,checkpoint=a.checkpoint,origin=a.origin,target=a.target,device=a.device,progress=progress)
