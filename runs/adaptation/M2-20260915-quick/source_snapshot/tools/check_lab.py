"""Read-only environment and baseline checks; never starts an experiment job."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PACKAGES = ('numpy', 'scipy', 'matplotlib', 'torch', 'PySide6', 'vtk', 'pyvista', 'pyvistaqt')


def check(root=ROOT, *, compute=False, require_cuda=False, require_baseline=False):
    root = Path(root).resolve()
    result = dict(python=sys.version.split()[0], interpreter=sys.executable,
                  platform=platform.platform(), root=str(root), packages={}, errors=[], notes=[])
    if sys.version_info[:2] < (3, 12):
        result['errors'].append('Use Python 3.12 or 3.13 for the pinned desktop environment.')
    for package in PACKAGES:
        try:
            result['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result['packages'][package] = None
            result['errors'].append(f'Missing package: {package}')
    if result['packages']['torch']:
        try:
            import torch
            available = torch.cuda.is_available()
            result['cuda'] = dict(available=available, build=torch.version.cuda,
                                  devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
            if require_cuda and not available:
                result['errors'].append('CUDA is unavailable; full model fitting requires a working NVIDIA CUDA setup.')
            if compute:
                device = 'cuda' if available else 'cpu'
                values = torch.arange(16, dtype=torch.float64, device=device).reshape(4, 4)
                actual = values @ values.T
                if not torch.isfinite(actual).all().item() or actual[0, 0].item() != 14.0:
                    raise RuntimeError('Float64 tensor check returned an incorrect result')
                if available:
                    torch.cuda.synchronize()
                result['compute_check'] = dict(device=device, dtype='float64', passed=True,
                                               scope='Small tensor operation; no model rollout, fit, planner or flight.')
        except Exception as exc:
            result['errors'].append(f'PyTorch/CUDA check failed: {exc}')
    if sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        result['notes'].append('No desktop display detected. Launch the GUI in a graphical session; CLI inspection remains available.')
    try:
        from deployment.lab_seed import load_baseline
        baseline = load_baseline(root)
        result['baseline'] = dict(available=True, description=str(baseline.get('model', baseline.get('model_path', 'Verified imported M0'))))
    except (OSError, ValueError, KeyError, ImportError) as exc:
        result['baseline'] = dict(available=False, reason=str(exc))
        message = 'Import the retained M0 baseline bundle in the app before starting the experiment.'
        (result['errors'] if require_baseline else result['notes']).append(message)
    result['ok'] = not result['errors']
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--compute', action='store_true', help='Run a tiny float64 tensor check, not a simulation.')
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--require-baseline', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    result = check(args.root, compute=args.compute, require_cuda=args.require_cuda,
                   require_baseline=args.require_baseline)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Aerial Whip Lab — {'ready' if result['ok'] else 'needs attention'}")
        print(f"Python {result['python']} | {result['platform']}")
        print(f"Project: {result['root']}")
        for name, version in result['packages'].items():
            print(f"  {name}: {version or 'MISSING'}")
        cuda = result.get('cuda', {})
        print(f"CUDA: {cuda.get('build')} | {', '.join(cuda.get('devices', [])) or 'not available'}")
        print(f"M0 baseline: {'available' if result['baseline']['available'] else 'not imported'}")
        for item in result['errors']:
            print(f'ERROR: {item}')
        for item in result['notes']:
            print(f'NOTE: {item}')
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
