"""Accept an independently replayed full MPPI proposal as a NEW saved plan.

The generating rolling run remains a separate partial/stopped run. This is
feasible-candidate acceptance, not completion of its one-action replanning loop.
No optimizer or flight is started. A failed recovery leaves the parent untouched.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import torch
from planning.pva_job import prepare
from learning.pva_env import PVAEnvironment
from simulator.workflow import read_json
from experimental_data.io import atomic_json,sha256_file
from deployment.pva_rehearsal import generate,export_package

ROOT=Path(__file__).resolve().parents[1]


def finalize(parent,candidate,output):
    parent=parent.resolve();candidate=candidate.resolve();output=output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    cfg=read_json(parent/'settings.json')
    if cfg['method']!='mppi':raise ValueError('Only MPPI proposals can be finalized')
    cfg['model_path']=str(parent/'model.json')
    with np.load(candidate) as z:sequence=z['normalized_jerk'].copy()
    if sequence.ndim!=2 or sequence.shape[1]!=3 or not np.isfinite(sequence).all() or np.abs(sequence).max()>1:
        raise ValueError('Finite bounded XYZ jerk candidate required')
    job,_=prepare(ROOT,cfg,'MPPI wave - accepted full candidate (simulation)')
    identity=read_json(job/'identity.json');identity.update(acceptance='independent_full_candidate_replay',
        parent_run=str(parent),candidate_source=str(candidate),candidate_sha256=sha256_file(candidate),
        note='New plan accepted from a rolling MPPI proposal. Parent rolling loop did not complete; no optimizer runs in this derived job.')
    atomic_json(job/'identity.json',identity)
    atomic_json(output/'acceptance.json',dict(job=str(job),**identity))
    try:
        env=PVAEnvironment(read_json(job/'model.json'),cfg,root=job)
        result=env.rollout(actions=env.tensor(sequence)[None],max_steps=len(sequence),trace=True)
        if not bool(env.success[0]) or bool(env.failed[0]):raise ValueError('Candidate fails independent strike/envelope checks')
        actions=torch.stack(env.actions,1)[0].cpu().numpy()
        np.savez_compressed(job/'plan.npz',normalized_jerk=actions,committed_steps=env.index,
            plan_complete=True,planner_mode='mppi_full_candidate_acceptance_v1')
        summary=dict(best_success=True,best_failed=False,best_reward=float(env.total[0]),
            best_minimum_tip_distance_m=float(env.minimum_distance[0]),command_steps=env.index,
            maneuver_duration_s=env.index/30,scored_duration_s=float(env.termination_time[0]),
            iterations=0,stop_reason='accepted_independently_replayed_full_candidate',
            parent_run=str(parent),candidate_sha256=sha256_file(candidate),
            lookahead_s=cfg['mppi']['horizon_s'],planner_mode='mppi_full_candidate_acceptance_v1',
            evidence='Historical normalized M1 simulation only; no new optimizer in derived job')
        rehearsal=ROOT/'runs/rehearsals_pva'/(job.name+'-mppi-wave')
        metadata=generate(job,rehearsal)
        package=output/'MPPI-wave-simulation.zip';export_package(rehearsal,package)
        atomic_json(job/'result.json',summary)
        atomic_json(job/'status.json',dict(status='completed',**summary))
        atomic_json(output/'result.json',dict(job=str(job),parent_run=str(parent),rehearsal=str(rehearsal),
            package=str(package),metadata=metadata,acceptance=summary))
        print(str(output/'result.json'),flush=True)
        return job
    except Exception as error:
        atomic_json(job/'status.json',dict(status='failed',stage='candidate acceptance/recovery',error=str(error)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();finalize(a.parent,a.candidate,a.output)
