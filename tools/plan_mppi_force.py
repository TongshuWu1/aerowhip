"""Isolated offline direct-force MPPI worker; no aircraft interface."""
from pathlib import Path
import sys
import argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planning.mppi_force_run import run_job
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True)
    run_job(parser.parse_args().job)
