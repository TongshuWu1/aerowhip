"""Explicit, separately frozen full-model continuation with no neural update ceiling."""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experimental_data.whip_full_continuation import prepare,run
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','fit'])
    p.add_argument('--job',required=True,type=Path);p.add_argument('--source',type=Path);p.add_argument('--numerical-review',type=Path)
    a=p.parse_args()
    if a.stage=='prepare':
        if a.source is None or a.numerical_review is None:p.error('Preparation requires source and numerical review')
        print(prepare(a.source,a.job,a.numerical_review))
    else:print(run(a.job))
if __name__=='__main__':main()
