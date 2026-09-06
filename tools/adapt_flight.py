"""Offline flight-data adaptation commands; never connects to a drone."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from simulator.workflow import read_json,atomic_json
from experimental_data.flight_trials import template,import_trial
from experimental_data.flight_adaptation import fit_candidate,replay


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['template','import','replay','fit','refine'])
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source',type=Path)
    parser.add_argument('--trials',type=Path,nargs='+')
    parser.add_argument('--model',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--updates',type=int,default=12)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.mode=='template':template(args.output);return
    if args.mode=='import':print(import_trial(args.root,args.source));return
    payload=read_json(args.model or args.root/'config/model.json')
    def progress(label):
        print(label,flush=True)
        # BackgroundJob owns the parent job folder; result subfolders stay immutable.
        atomic_json(args.output.parent/'progress.json',dict(label=label))
    if args.mode=='replay':result=replay(args.source,payload,args.output)
    elif args.mode=='fit':result=fit_candidate(args.trials,payload,args.output,max_evaluations=args.updates,progress=progress)
    else:
        from learning.strike_adaptation import refine
        result=refine(args.source,payload,read_json(args.root/'config/ppo.json'),args.output,updates=args.updates,progress=progress)
    print(result,flush=True)


if __name__=='__main__':main()
