"""Isolated CEM worker; never creates a Qt or OpenGL context."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planning.cem_run import run_job

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',required=True)
    run_job(parser.parse_args().job)
