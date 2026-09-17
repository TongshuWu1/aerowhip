"""Prepare full-state flight recordings without fitting or training."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experimental_data.adaptation_rounds import import_round,process_round,write_json
from experimental_data.legacy_execution import create_profile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--number',type=int)
    parser.add_argument('--parent')
    parser.add_argument('--notes',default='')
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--round',type=Path)
    parser.add_argument('--overrides',type=Path)
    parser.add_argument('--legacy-controller',type=Path,help='Archive reviewed CSV-only legacy controller; never execute it')
    parser.add_argument('--logger-source',type=Path)
    parser.add_argument('--geometry-model',type=Path)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    directory=args.round
    if args.source:
        if args.number is None:parser.error('--number required with --source')
        directory=import_round(args.root,args.source,args.number,parent=args.parent,notes=args.notes,reference=args.reference)
    if directory is None:parser.error('Supply --round or --source')
    context=[args.legacy_controller,args.logger_source,args.geometry_model]
    if any(context):
        if not all(context):parser.error('Legacy profile requires --legacy-controller, --logger-source and --geometry-model')
        create_profile(directory,*context)
    overrides=json.loads(args.overrides.read_text()) if args.overrides else None
    output=process_round(directory,overrides)
    write_json(args.receipt,dict(round=str(directory),processed=str(output)))
    reports=json.loads((output/'processing.json').read_text())['reports']
    print(json.dumps(reports,indent=2))
    return int(any(r['status']=='failed' for r in reports))


if __name__=='__main__':raise SystemExit(main())
