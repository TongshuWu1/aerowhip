"""Paired development evaluation of measured mass; never starts training."""
import json
from pathlib import Path
import sys
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_ppo import build_agent, _load_checkpoint, evaluate
from learning.experiment_records import save_evaluation
from simulator.workflow import stamp


def main():
    torch.set_num_threads(1)
    parent = ROOT/'runs/ppo/20260906-174733-192436-seed652'
    corrected = Path(sys.argv[1])
    output = ROOT/'runs/sensitivity'/f'{stamp()}-measured-mass'
    output.mkdir(parents=True)
    for name, directory in [('previous', parent), ('measured', corrected)]:
        configs = [json.loads((directory/f'{key}.json').read_text()) for key in ('model','task','ppo')]
        agent = build_agent(configs[2], torch.device('cuda'))
        _load_checkpoint(agent, parent/'checkpoints/terminal.pt', load_optimizer=False)
        result = evaluate(*configs, agent, episodes=256, batch_size=256, device=torch.device('cuda'))
        np.savez_compressed(output/f'{name}_scenarios.npz', **result.scenarios)
        arrays, metadata = result.recording
        np.savez_compressed(output/f'{name}_first_trial.npz', **arrays)
        (output/f'{name}_first_trial.json').write_text(json.dumps(metadata, indent=2))
        (output/f'{name}_trials.json').write_text(json.dumps(result.trials, indent=2))
        (output/f'{name}.json').write_text(json.dumps(result, indent=2))
        print(name, json.dumps(result), flush=True)
    print(output, flush=True)


if __name__ == '__main__':
    main()
