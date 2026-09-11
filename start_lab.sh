#!/usr/bin/env sh
set -eu
LAB_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ ! -x "$LAB_ROOT/.venv/bin/python" ]; then
    printf '%s\n' 'Run python3 setup_lab.py --device cuda first (or --device cpu for inspection).'
    exit 1
fi
exec "$LAB_ROOT/.venv/bin/python" "$LAB_ROOT/run_lab.py" "$@"
