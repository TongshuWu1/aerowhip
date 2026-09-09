"""Create a separate translated native rehearsal; no policy inference or flight."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from deployment.translate_rehearsal import translate_rehearsal

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--offset',type=float,nargs=3,default=[0,0,-.3])
    args=parser.parse_args()
    translate_rehearsal(args.source,args.output,args.offset)
