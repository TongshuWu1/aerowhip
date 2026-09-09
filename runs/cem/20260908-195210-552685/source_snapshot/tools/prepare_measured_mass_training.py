"""Prepare a bounded, source-frozen PPO refinement on the measured mass model."""
from pathlib import Path
import json
import hashlib
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.workflow import stamp


def main():
    candidate = Path(sys.argv[1]).resolve()
    parent = ROOT/'runs/ppo/20260906-174733-192436-seed652/checkpoints/terminal.pt'
    directory = ROOT/'runs/ppo'/f'{stamp()}-measured-mass-seed653'
    launch = directory/'launch_config'
    launch.mkdir(parents=True)
    for name in ('model','task','ppo'):
        config = json.loads((candidate/f'{name}.json').read_text())
        if name == 'ppo':
            config['seed'] = 653
            config['deployment']['reset_optimizer_on_resume'] = True
            config['deployment']['force_gain_fraction'] = .05
            config['deployment']['evaluate_final_holdout'] = False
            config['training'].update(requested_episodes=48128, collection_batch=4096, device='cuda')
            config['validation'].update(episodes=256, every_episodes=4096)
            config['status'] = 'measured_mass_bounded_refinement_8192_additional_attempts'
        (launch/f'{name}.json').write_text(json.dumps(config, indent=2))
    snapshot = directory/'source_snapshot'
    snapshot.mkdir()
    for folder in ('learning','simulator','experimental_data','tools','deployment'):
        for source in (ROOT/folder).rglob('*'):
            if source.is_file() and source.suffix in ('.py','.cu','.cuh','.h') and '__pycache__' not in source.parts:
                target = snapshot/source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source,target)
    shutil.copy2(ROOT/'run_ppo.py',snapshot/'run_ppo.py')
    shutil.copytree(launch,snapshot/'config')
    files = {p.relative_to(snapshot).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
             for p in snapshot.rglob('*') if p.is_file()}
    (directory/'source_snapshot_manifest.json').write_text(json.dumps(dict(files=files),indent=2))
    command = [sys.executable,'-u',str(snapshot/'run_ppo.py'),'--train','--config-directory',str(launch),
               '--artifact-directory',str(directory),'--resume-checkpoint',str(parent),
               '--device','cuda','--episodes','48128','--batch-size','4096']
    (directory/'launch.json').write_text(json.dumps(dict(command=command,working_directory=str(snapshot)),indent=2))
    (directory/'run.json').write_text(json.dumps(dict(schema='point_force_training_run_v1',
        algorithm='PPO',display_name='PPO · measured 157 g drone / 18 g cable · 5% force tolerance',
        resumed_from=str(parent),parent_checkpoint_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
        change='Measured mass distribution (proportional cable scaling assumption), reset Adam, provisional ±5% force gain instead of ±2%, 8192 additional attempts. Reward and timing unchanged.'),indent=2))
    print(directory)


if __name__ == '__main__':
    main()
