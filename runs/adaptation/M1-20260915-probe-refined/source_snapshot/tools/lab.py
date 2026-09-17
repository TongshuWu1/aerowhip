"""Explicit lab actions. No action arms or communicates with an aircraft."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from deployment.lab_workflow import LabWorkspace


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--study',help='Study name; required for actions other than overview')
    commands=parser.add_subparsers(dest='action',required=True)
    commands.add_parser('overview');commands.add_parser('create');commands.add_parser('stop')
    p=commands.add_parser('import')
    p.add_argument('--stage',choices=['M0','M1','final'],required=True);p.add_argument('--take',required=True)
    p.add_argument('--tracking',type=Path,required=True);p.add_argument('--controller',type=Path,required=True)
    p.add_argument('--offset',type=float,required=True);p.add_argument('--clock-source',required=True)
    p.add_argument('--clock-verified',action='store_true');p.add_argument('--drone-label')
    p=commands.add_parser('align')
    p.add_argument('--stage',choices=['M0','M1','final'],required=True);p.add_argument('--take',required=True)
    p.add_argument('--offset',type=float,required=True);p.add_argument('--clock-source',required=True);p.add_argument('--clock-verified',action='store_true')
    p=commands.add_parser('estimate-timing')
    p.add_argument('--tracking',type=Path,required=True);p.add_argument('--controller',type=Path,required=True);p.add_argument('--drone-label')
    p=commands.add_parser('review')
    p.add_argument('--stage',choices=['M0','M1','final'],required=True);p.add_argument('--take',required=True)
    p.add_argument('--reviewer',required=True);p.add_argument('--free-motion-end',type=float)
    p.add_argument('--contact',choices=['none','at_or_after_end'],default='none')
    p.add_argument('--clock-reviewed',action='store_true');p.add_argument('--same-hardware',action='store_true')
    p.add_argument('--no-intervention',action='store_true');p.add_argument('--notes',default='');p.add_argument('--exclude',action='store_true')
    p=commands.add_parser('prepare');p.add_argument('--generation',choices=['M1','M2'],required=True);p.add_argument('--retry',action='store_true')
    for name in ('fit','plan','export'):
        p=commands.add_parser(name);p.add_argument('--generation',choices=['M1','M2'] if name=='fit' else ['M0','M1','M2'],required=True)
        p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    p=commands.add_parser('evaluate');p.add_argument('--device',default='cuda',choices=['cpu','cuda']);p.add_argument('--physical-only',action='store_true')
    args=parser.parse_args(argv)
    if args.action!='overview' and not args.study: parser.error('--study is required for this action')
    lab=LabWorkspace(args.root,args.study)
    try:
        if args.action=='overview': result=lab.overview()
        elif args.action=='create': result=lab.create()
        elif args.action=='import': result=lab.import_take(args.stage,args.take,args.tracking.resolve(),args.controller.resolve(),offset_s=args.offset,clock_source=args.clock_source,clock_verified=args.clock_verified,drone_label=args.drone_label)
        elif args.action=='review': result=lab.review_take(args.stage,args.take,reviewer=args.reviewer,free_motion_end_s=args.free_motion_end,physical_contact=args.contact,clock_reviewed=args.clock_reviewed,same_hardware=args.same_hardware,no_intervention=args.no_intervention,notes=args.notes,exclude=args.exclude)
        elif args.action=='align': result=lab.align_take(args.stage,args.take,offset_s=args.offset,clock_source=args.clock_source,clock_verified=args.clock_verified)
        elif args.action=='estimate-timing': result=lab.estimate_timing(args.tracking,args.controller,args.drone_label)
        elif args.action=='prepare': result=lab.prepare_update(args.generation,retry=args.retry)
        elif args.action=='fit': result=lab.fit_update(args.generation,args.device)
        elif args.action=='plan': result=lab.plan(args.generation,args.device)
        elif args.action=='export': result=lab.export(args.generation,args.device)
        elif args.action=='stop': result=lab.stop()
        else: result=lab.evaluate(args.device,predictions=not args.physical_only)
        print(json.dumps(result,indent=2,allow_nan=False,default=str),flush=True)
        return 0
    except (ValueError,FileNotFoundError,FileExistsError) as error:
        print('Action could not complete: '+str(error),file=sys.stderr,flush=True)
        return 2


if __name__=='__main__': raise SystemExit(main())
