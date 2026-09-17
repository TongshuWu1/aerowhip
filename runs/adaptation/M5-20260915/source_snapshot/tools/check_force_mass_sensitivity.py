"""Offline paired sensitivity study; no training, controller changes or flight sender."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deployment.rehearsal import assumed_hanging_state
from deployment.planner import write_plan
from simulator.live_flight import LiveFlight
from simulator.gpu_rehearsal import prepare_gpu_rehearsal, GpuRehearsalPhysics
from simulator.cable.cuda_rehearsal_solvers import RehearsalSolvers
from simulator.point_mass import ForceControlledPointCable


def main():
    torch.set_num_threads(1)
    run = ROOT/'runs/ppo/20260906-174733-192436-seed652'
    checkpoint = run/'checkpoints/terminal.pt'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == '214fb97211c1b57a04a7352d9629a0369c185a97efc0ff665e0336b5188d303d'
    configs = [json.loads((run/f'{name}.json').read_text()) for name in ('model','task','ppo')]
    output = ROOT/'runs/sensitivity'/time.strftime('%Y%m%d-%H%M%S-force-mass')
    output.mkdir(parents=True, exist_ok=False)
    for name, config in zip(('model','task','ppo'), configs):
        (output/f'{name}.json').write_text(json.dumps(config, indent=2))
    (output/'study_source.py').write_bytes(Path(__file__).read_bytes())
    planner = LiveFlight.from_checkpoint(*configs, checkpoint)
    _, planner.physics, planner.policy = prepare_gpu_rehearsal(planner)
    solvers = RehearsalSolvers()
    masses = [-5, -3, 0, 3, 5]
    physics = {}
    for grams in masses:
        model = copy.deepcopy(configs[0])
        model['point_mass']['mass_kg'] += grams/1000
        physics[grams] = GpuRehearsalPhysics(ForceControlledPointCable.from_mapping(model),
                                          planner.state, planner.dt_s, solvers)
    scenarios = [(g, 1.) for g in masses] + [(0, gain) for gain in (.98, .95, .90, .85)]
    scenarios += [(g, gain) for g in (3, 5) for gain in (.95, .90)]
    rng = np.random.default_rng(90606)
    def offset():
        x = rng.normal(size=3)
        return x/np.linalg.norm(x)*.05*rng.random()**(1/3)
    conditions = [(np.zeros(3), np.zeros(3))] + [(offset(), offset()) for _ in range(8)]
    rows = []
    for index, (start_delta, target_delta) in enumerate(conditions):
        task = copy.deepcopy(configs[1])
        task['initial_root_position_m'] = (np.array(task['initial_root_position_m'])+start_delta).tolist()
        task['target_position_m'] = (np.array(task['target_position_m'])+target_delta).tolist()
        initial = assumed_hanging_state(configs[0], task['initial_root_position_m'], [0,0,0])
        from simulator.strike_plan import compile_strike_plan
        plan = compile_strike_plan(configs[0], task, configs[2], initial,
                                   planner.policy, physics=planner.physics)
        directory = output/f'initial_{index:02d}'
        write_plan(plan, directory, task=task, metadata=dict(policy_sha256=digest))
        baseline_end = None
        # Always run nominal first for paired endpoint errors.
        ordered = [(0, 1.)] + [s for s in scenarios if s != (0, 1.)]
        for grams, gain in ordered:
            plant_config = copy.deepcopy(configs[0])
            plant_config['point_mass']['mass_kg'] += grams/1000
            flight = LiveFlight(plant_config, task, configs[2], policy=lambda _: None)
            flight.state = initial
            flight.settled_s = 10.
            flight.physics = lambda q, v, f, g=grams, k=gain: physics[g](q, v, f*k)
            flight.start_strike(plan)
            positions = [initial.positions_m[0].numpy().copy()]
            velocities = [initial.velocities_m_s[0].numpy().copy()]
            hits = []
            failure = None
            try:
                for _ in plan.forces_world_n:
                    frame = flight.step()
                    positions.append(frame['positions'])
                    velocities.append(frame['velocities'])
                    hits.append(frame['hit'])
            except RuntimeError as error:
                failure = str(error)
            q = np.asarray(positions)
            target = np.asarray(task['target_position_m'])
            a, d = q[:-1,-1], np.diff(q[:,-1], axis=0)
            t = np.clip(np.sum((target-a)*d, axis=1)/np.maximum(np.sum(d*d, axis=1), 1e-30), 0, 1)
            closest = np.linalg.norm(a+t[:,None]*d-target, axis=1).min()
            if (grams, gain) == (0, 1.):
                baseline_end = q[-1,0].copy()
            row = dict(initial=index, mass_error_g=grams, delivered_force_fraction=gain,
                valid_hit=bool(flight.success), tip_contact=bool(flight.tip_contact),
                non_tip_first=bool(flight.non_tip_first), closest_tip_distance_m=float(closest),
                drone_endpoint_error_m=float(np.linalg.norm(q[-1,0]-baseline_end)),
                drone_endpoint_z_error_m=float(q[-1,0,2]-baseline_end[2]),
                cutoff_s=plan.duration_s, completed=failure is None, failure=failure)
            rows.append(row)
            label = f'mass_{grams:+d}g_gain_{gain:.2f}'
            np.savez_compressed(directory/f'{label}.npz', positions_m=q, velocities_m_s=velocities,
                commanded_force_world_n=plan.forces_world_n.numpy(),
                delivered_force_world_n=plan.forces_world_n.numpy()*gain, valid_hits=hits)
            (directory/f'{label}.json').write_text(json.dumps(row, indent=2))
        print(f'Finished initial condition {index+1}/{len(conditions)}', flush=True)
    with (output/'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = dict(checkpoint=str(checkpoint), sha256=digest, device=torch.cuda.get_device_name(),
        os=sys.platform, seed=90606, initial_conditions=len(conditions), trials=len(rows),
        scope='Ideal settled vertical cable, zero initial velocity; strike only, no recovery or hover drift. Fixed plan per initial condition. Uniform total-thrust gain after controller gravity addition. Gain is hypothetical, not calibrated battery voltage. Diagnostic sensitivity, not independent validation.',
        scenarios=[])
    for grams, gain in scenarios:
        selected = [r for r in rows if r['mass_error_g']==grams and r['delivered_force_fraction']==gain]
        summary['scenarios'].append(dict(mass_error_g=grams, delivered_force_fraction=gain,
            valid_hits=sum(r['valid_hit'] for r in selected), trials=len(selected),
            nominal_initial_hit=selected[0]['valid_hit'],
            max_drone_endpoint_error_cm=100*max(r['drone_endpoint_error_m'] for r in selected),
            mean_drone_z_error_cm=100*np.mean([r['drone_endpoint_z_error_m'] for r in selected]),
            failures=sum(not r['completed'] for r in selected)))
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    print(str(output), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
