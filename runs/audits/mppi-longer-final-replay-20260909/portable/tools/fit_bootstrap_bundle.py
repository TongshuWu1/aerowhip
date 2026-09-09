"""Prepare and run explicit corrected bootstrap stages; never activates them."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experimental_data.bootstrap_bundle import prepare
from experimental_data.io import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--job',type=Path)
    p.add_argument('--stage',choices=['cable','drone','combined','report']);a=p.parse_args()
    if not a.prepare and a.job is None:p.error('Provide --prepare or --job')
    job=prepare(ROOT) if a.prepare else a.job.resolve()
    if a.stage:
        try:
            if a.stage=='cable':
                from experimental_data.bootstrap_cable import run
            elif a.stage=='drone':
                from experimental_data.bootstrap_drone import run
            elif a.stage=='combined':
                from experimental_data.bootstrap_combined import run
            else:
                from experimental_data.bootstrap_report import run
            run(job)
        except Exception as exc:
            atomic_json(job/(a.stage+'_failure.json'),dict(type=type(exc).__name__,reason=str(exc)));raise

if __name__=='__main__':main()
