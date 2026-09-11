"""Create a repo-local environment and install the lab desktop dependencies."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent


def environment_python(root: Path) -> Path:
    return root / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def installation_commands(python: Path, root: Path, device: str, tests: bool = False):
    """Argument lists work in paths containing spaces; no shell needed."""
    index = 'https://download.pytorch.org/whl/' + ('cu128' if device == 'cuda' else 'cpu')
    requirement = 'test.txt' if tests else 'desktop.txt'
    return [
        [str(python), '-m', 'pip', 'install', '--upgrade', 'pip'],
        [str(python), '-m', 'pip', 'install', 'torch==2.11.0', '--index-url', index],
        [str(python), '-m', 'pip', 'install', '-r', str(root / 'requirements' / requirement),
         '-c', str(root / 'requirements/deployment-constraints.txt')],
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda',
                        help='CUDA for the NVIDIA lab workstation; CPU for inspection/UI only.')
    parser.add_argument('--tests', action='store_true', help='Also install pytest.')
    args = parser.parse_args(argv)
    if not (3, 12) <= sys.version_info[:2] < (3, 14):
        parser.error('Use Python 3.12 (recommended) or 3.13 to create this environment.')
    if args.device == 'cuda' and sys.platform not in ('win32', 'linux'):
        parser.error('The CUDA installation is for Windows/Linux. Use --device cpu here.')
    python = environment_python(ROOT)
    if not python.exists():
        print(f'Creating environment: {ROOT / ".venv"}', flush=True)
        venv.EnvBuilder(with_pip=True).create(ROOT / '.venv')
    try:
        for command in installation_commands(python, ROOT, args.device, args.tests):
            subprocess.run(command, cwd=ROOT, check=True)
        check = [str(python), str(ROOT / 'tools/check_lab.py')]
        if args.device == 'cuda':
            check.append('--require-cuda')
        subprocess.run(check, cwd=ROOT, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f'Setup did not finish: {exc}\nSee docs/INSTALL.md for recovery.', file=sys.stderr)
        return 1
    print(f'Environment ready. Start the app with:\n"{python}" "{ROOT / "run_lab.py"}"')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
