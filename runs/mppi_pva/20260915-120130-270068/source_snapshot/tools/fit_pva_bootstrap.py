"""Independent process for cold normalized-adp0 fitting."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from experimental_data.pva_bootstrap import prepare_job, run

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true')
    parser.add_argument('--continue-after-drone',action='store_true')
    parser.add_argument('--continue-cable-full-whip',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare_job(args.job)
    if args.run:run(args.job,continue_after_drone=args.continue_after_drone,continue_cable_full_whip=args.continue_cable_full_whip)
