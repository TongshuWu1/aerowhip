"""One-shot post-training rehearsal audit. Never starts or resumes training."""
from pathlib import Path
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import torch

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[2]
sys.path.insert(0, str(ROOT))
from experimental_data.io import atomic_json, sha256_file
from simulator.workflow import read_json


def main():
    spec = read_json(STUDY / 'study.json')
    if read_json(STUDY / 'status.json')['status'] != 'COMPLETED':
        raise RuntimeError('Both bounded training arms must finish before this audit.')
    with (STUDY / 'export.lock').open('x') as stream:
        stream.write(str(os.getpid()))
    jobs = {label: Path(job['directory']) for label, job in spec['jobs'].items()}
    configs = [read_json(path / 'ppo.json') for path in jobs.values()]
    tasks = [read_json(path / 'task.json') for path in jobs.values()]
    assert configs[0] == configs[1] and tasks[0] == tasks[1]
    assert configs[0]['reward'] == read_json(STUDY / 'parent_policy/ppo.json')['reward']
    assert not configs[0]['early_stopping']['enabled']
    outcomes = {}
    # The M1 artifact is first for the upcoming data collection. M0 is the control.
    for label in ('M1', 'M0'):
        run = jobs[label]
        status = read_json(run / 'status.json')
        record = read_json(run / 'run.json')
        assert status['status'] == 'COMPLETED'
        assert status['episodes'] == spec['target_attempts']
        assert not record['optimizer_state_restored']
        checkpoint = run / 'checkpoints/best_validation.pt'
        digest = sha256_file(checkpoint)
        selected = torch.load(checkpoint, map_location='cpu', weights_only=False)
        validation = read_json(run / 'best_validation.json')
        assert selected['episodes'] == validation['training_episodes']
        output = ROOT / 'runs/rehearsals' / (STUDY.name + '-' + label + '-updated-PPO')
        if output.exists():
            raise RuntimeError('Do not overwrite a saved prediction: ' + str(output))
        command = [sys.executable, '-u', str(run / 'source_snapshot/tools/rehearse_research.py'),
                   '--checkpoint', str(checkpoint), '--output', str(output),
                   '--origin', '-2', '0', '1.255', '--target', '-1', '0', '1.1', '--device', 'cuda']
        atomic_json(STUDY / 'export_status.json', dict(status='GENERATING', condition=label, output=str(output)))
        with (STUDY / (label + '-rehearsal.log')).open('wb') as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        meta = read_json(output / 'rehearsal.json')
        assert meta['checkpoint_sha256'] == digest
        assert sha256_file(output / 'checkpoints/policy.pt') == digest
        assert meta['reference_feasible']
        assert meta['training_export_prefix_max_difference_m'] <= 1e-8
        assert meta['initial_tracking_origin_m'] == spec['launch']['initial_tracking_origin_m']
        assert meta['target_position_m'] == spec['launch']['target_position_m']
        with np.load(output / 'rehearsal.npz') as arrays:
            csv = np.loadtxt(output / 'fullstate_30hz.csv', delimiter=',', skiprows=1)
            expected = np.c_[arrays['command_time_s'], arrays['commands']]
            np.testing.assert_allclose(csv, expected, atol=1e-12, rtol=0)
            np.testing.assert_allclose(np.diff(csv[:, 0]), 1 / 30, atol=1e-12, rtol=0)
            force = np.loadtxt(output / 'virtual_force_30hz.csv', delimiter=',', skiprows=1)
            np.testing.assert_allclose(force, np.c_[arrays['force_time_s'][::5], arrays['virtual_force_n'][::5]], atol=1e-12, rtol=0)
            maximums = dict(command=float(arrays['commands'][:, 2].max()),
                            predicted_origin=float(arrays['origin_positions_m'][:, 2].max()),
                            predicted_cable=float(arrays['cable_positions_m'][:, :, 2].max()))
            assert all(np.isfinite(arrays[key]).all() for key in ('commands', 'origin_positions_m', 'origin_rotations', 'cable_positions_m'))
        package = ROOT / 'policies' / (label + '-PPO-adaptation-study-20260909-003414.zip')
        if package.exists():
            raise RuntimeError('Do not overwrite existing export: ' + str(package))
        export_code = ('import sys; sys.path.insert(0,sys.argv[1]); '
                       'from deployment.research_rehearsal import export_package; '
                       'export_package(sys.argv[2],sys.argv[3])')
        subprocess.run([sys.executable, '-c', export_code, str(run / 'source_snapshot'), str(output), str(package)], cwd=ROOT, check=True)
        with zipfile.ZipFile(package) as archive:
            assert archive.read('fullstate_30hz.csv') == (output / 'fullstate_30hz.csv').read_bytes()
            assert archive.read('checkpoints/policy.pt') == checkpoint.read_bytes()
            assert archive.read('deployment/research_rehearsal.py') == (run / 'source_snapshot/deployment/research_rehearsal.py').read_bytes()
        outcomes[label] = dict(run=str(run), checkpoint=str(checkpoint), checkpoint_sha256=digest,
                               selected_at_attempt=selected['episodes'], additional_attempts=20480,
                               simulation_validation=validation, rehearsal=str(output), package=str(package),
                               package_sha256=sha256_file(package), csv_sha256=sha256_file(output / 'fullstate_30hz.csv'),
                               nominal_prediction=meta, maximum_sampled_height_m=maximums,
                               sampled_heights_below_2_8m=all(v < 2.8 for v in maximums.values()),
                               exact_csv_force_and_packaged_source=True, real_flight_performance='UNKNOWN')
        atomic_json(STUDY / 'export_results.json', outcomes)
    original = read_json(STUDY / 'protected_inputs.json')
    changed = [name for name, digest in original.items() if not Path(name).is_file() or sha256_file(name) != digest]
    atomic_json(STUDY / 'final_preservation_check.json', dict(files=len(original), changed=changed, passed=not changed))
    assert not changed
    atomic_json(STUDY / 'export_status.json', dict(status='COMPLETED', simulation_only=True, real_flights_collected=False))
    print(json.dumps({label: dict(checkpoint=value['checkpoint'], simulated_hit=value['nominal_prediction']['predicted_valid_hit'],
                                 heights=value['maximum_sampled_height_m'], package=value['package']) for label, value in outcomes.items()}, indent=2))


if __name__ == '__main__':
    main()
