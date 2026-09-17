"""Correct any completed generation toward the frozen experiment reference."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model-job','reference','job','output','export'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--previous-rehearsal',type=Path,help='Warm-start command controls; reference stays fixed')
    p.add_argument('--iterations',type=int,default=30)
    p.add_argument('--strike-guard',choices=['none','reference','target'],default='none')
    p.add_argument('--strike-tolerance-m',type=float,default=0.)
    a=p.parse_args()
    from planning.correction_job import run
    run(a.model_job,a.reference,a.job,a.output,a.export,previous_rehearsal=a.previous_rehearsal,
        iterations=a.iterations,strike_guard=a.strike_guard,strike_tolerance_m=a.strike_tolerance_m)


if __name__=='__main__':main()
