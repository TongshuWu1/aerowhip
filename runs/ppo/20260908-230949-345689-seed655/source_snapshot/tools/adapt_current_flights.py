"""Fit current adp0 into a separate candidate; never launch PPO or flight."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from experimental_data.current_adaptation import JOB,prepare
from experimental_data.current_adaptation_fit import drone_run,cable_run
from experimental_data.adaptation_ensemble import run as cable_batched
from experimental_data.current_adaptation_validation import baseline_run,validation_run
from experimental_data.adaptation_attitude import run as attitude_refine

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','drone','cable','cable_batched','baseline','attitude_refine','validate']);p.add_argument('--job',type=Path,default=JOB)
    args=p.parse_args();torch.set_num_threads(1)
    {'prepare':prepare,'drone':drone_run,'cable':cable_run,'cable_batched':cable_batched,'baseline':baseline_run,'attitude_refine':attitude_refine,'validate':validation_run}[args.stage](args.job)
