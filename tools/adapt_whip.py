"""Reviewed flight data, diagnostics, and explicit staged model adaptation."""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experimental_data import whip_adaptation as data
from experimental_data import whip_adaptation_fit as fitting

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['setup','compare','prepare','diagnose','diagnose-drone','fit','fit-response','prepare-full','fit-full'])
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--batch',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--comparison',type=Path)
    p.add_argument('--review',type=Path);p.add_argument('--job',type=Path);p.add_argument('--device',default='cuda')
    p.add_argument('--selection',type=Path,help='Explicit frozen flight selection for a later generation')
    p.add_argument('--whip-source',type=Path);p.add_argument('--preliminary-source',type=Path)
    p.add_argument('--full-model',action='store_true',help='Prepare reviewed raw takes for full adaptation, including learned parent residuals')
    p.add_argument('--contract',type=Path,help='Frozen full update contract, including candidate/parent IDs and prior replay sources')
    p.add_argument('--rehearsal',type=Path);p.add_argument('--csv',type=Path)
    p.add_argument('--take-count',type=int);p.add_argument('--take-prefix',default='whip')
    p.add_argument('--all-training',action='store_true',help='Predeclare every take for training after frozen-parent evaluation')
    p.add_argument('--parent-id');p.add_argument('--candidate-id')
    p.add_argument('--prior-whip-source',type=Path,action='append',default=[])
    a=p.parse_args()
    required={'setup':['batch'],'compare':['batch','output'],'prepare':['batch','comparison','review','job'],
              'diagnose':['job'],'diagnose-drone':['job','output'],'fit':['job','review'],'fit-response':['job'],
              'prepare-full':['job','whip_source','preliminary_source'],'fit-full':['job']}[a.stage]
    for name in required:
        if getattr(a,name) is None:p.error('--'+name+' is required for '+a.stage)
    import torch
    torch.set_num_threads(4)
    if a.stage=='prepare-full':
        from experimental_data.whip_full_data import prepare,default_contract
        from simulator.workflow import read_json
        contract=read_json(a.contract) if a.contract else default_contract()
        if a.parent_id:contract['parent_id']=a.parent_id
        if a.candidate_id:contract['candidate_id']=a.candidate_id
        if a.prior_whip_source:contract['prior_whip_sources']=[str(v.resolve()) for v in a.prior_whip_source]
        if a.all_training:
            if any(v['role']!='adaptation' for v in read_json(a.whip_source/'protocol.json')['takes'].values()):
                p.error('--all-training requires reviewed adaptation roles for every included take')
            contract['evaluate_before_training']=True
        result=prepare(a.job,a.whip_source,a.preliminary_source,contract)
    elif a.stage=='fit-full':
        from experimental_data.whip_full_fit import fit
        result=fit(a.job,a.device)
    elif a.stage=='setup':result=data.setup(a.root,a.batch,a.selection,rehearsal=a.rehearsal,command_csv=a.csv,
        take_count=a.take_count,take_prefix=a.take_prefix,all_training=a.all_training)
    elif a.stage=='compare':result=data.compare(a.root,a.batch,a.output)
    elif a.stage=='prepare':result=data.prepare(a.root,a.batch,a.comparison,a.review,a.job,full_model=a.full_model)
    elif a.stage=='diagnose':result=fitting.diagnose(a.job,a.device)
    elif a.stage=='diagnose-drone':
        from experimental_data.drone_response_diagnostic import diagnose_drone
        result=diagnose_drone(a.job,a.output,a.device)
    elif a.stage=='fit-response':
        from experimental_data.whip_response_fit import fit
        result=fit(a.job,a.device)
    else:result=fitting.fit(a.job,a.review,a.device)
    print(result,flush=True)

if __name__=='__main__':main()
