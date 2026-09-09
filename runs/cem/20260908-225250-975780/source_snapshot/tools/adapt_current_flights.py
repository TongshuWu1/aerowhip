"""Fit current adp0 into a separate candidate; never launch PPO or flight."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from experimental_data.current_adaptation import JOB,prepare
from experimental_data.current_adaptation_fit import drone_run,cable_run

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','drone','cable']);p.add_argument('--job',type=Path,default=JOB)
    args=p.parse_args();torch.set_num_threads(1)
    {'prepare':prepare,'drone':drone_run,'cable':cable_run}[args.stage](args.job)
