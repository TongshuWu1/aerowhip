"""Check the installed runtime and a finite CPU/CUDA physics step, without training."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from simulator.point_mass import ForceControlledPointCable


def check(device='cpu', batch=1):
    if batch < 1:
        raise ValueError('Batch must be positive')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; install a compatible driver and CUDA torch wheel')
    torch.set_num_threads(1)
    config = json.loads((ROOT/'config/model.json').read_text(encoding='utf-8'))
    model = ForceControlledPointCable.from_mapping(config)
    q = torch.tensor([0., 0., 1.5], dtype=torch.float64, device=device).expand(batch, -1)
    state = model.hanging_state(q)
    force = (-model.system_mass_kg * model.gravity_world_m_s2).to(device).expand(batch, -1)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = model.step_runtime(state, force, config['simulation']['dt_s'])
    if device == 'cuda':
        torch.cuda.synchronize()
    if not bool(torch.isfinite(result.state.positions_m).all() & torch.isfinite(result.state.velocities_m_s).all()):
        raise RuntimeError('Nonfinite physics result')
    report = {'python': platform.python_version(), 'platform': platform.platform(),
              'interpreter': sys.executable, 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
              'device': device, 'batch': batch, 'physics_step_seconds': time.perf_counter()-started,
              'finite': True, 'scope': 'one runtime step; not a full training memory or flight acceptance test'}
    if device == 'cuda':
        report.update(gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                      total_vram_bytes=torch.cuda.get_device_properties(0).total_memory,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated())
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = check(args.device, args.batch)
    value = json.dumps(report, indent=2)+'\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(value)
    print(value)
