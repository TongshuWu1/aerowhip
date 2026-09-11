"""Portable setup command construction and failure visibility."""
from pathlib import Path
import subprocess

import setup_lab


def test_cuda_install_uses_official_wheel_and_literal_space_paths(tmp_path):
    root = tmp_path / 'lab with spaces'
    python = root / '.venv/bin/python'
    commands = setup_lab.installation_commands(python, root, 'cuda')
    assert all(command[0] == str(python) for command in commands)
    assert commands[1][-1] == 'https://download.pytorch.org/whl/cu128'
    assert commands[2][-1] == str(root / 'requirements/deployment-constraints.txt')
    assert str(root / 'requirements/desktop.txt') in commands[2]


def test_cpu_install_and_test_dependencies_are_explicit(tmp_path):
    commands = setup_lab.installation_commands(tmp_path / 'python', tmp_path, 'cpu', True)
    assert commands[1][-1] == 'https://download.pytorch.org/whl/cpu'
    assert str(tmp_path / 'requirements/test.txt') in commands[2]


def test_existing_environment_is_reused_and_health_check_runs(monkeypatch, tmp_path):
    python = tmp_path / '.venv' / ('Scripts/python.exe' if setup_lab.os.name == 'nt' else 'bin/python')
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(setup_lab, 'ROOT', tmp_path)
    monkeypatch.setattr(setup_lab.sys, 'version_info', (3, 12, 10))
    calls = []
    monkeypatch.setattr(setup_lab.subprocess, 'run', lambda command, **kw: calls.append((command, kw)))
    assert setup_lab.main(['--device', 'cpu']) == 0
    assert calls[-1][0] == [str(python), str(tmp_path / 'tools/check_lab.py')]
    assert all(call[1]['cwd'] == tmp_path and call[1]['check'] for call in calls)


def test_failed_install_does_not_launch_app_or_delete_data(monkeypatch, tmp_path):
    python = tmp_path / '.venv' / ('Scripts/python.exe' if setup_lab.os.name == 'nt' else 'bin/python')
    python.parent.mkdir(parents=True)
    python.touch()
    evidence = tmp_path / 'measurement.csv'
    evidence.write_bytes(b'original')
    monkeypatch.setattr(setup_lab, 'ROOT', tmp_path)
    monkeypatch.setattr(setup_lab.sys, 'version_info', (3, 12, 10))
    def fail(command, **kw):
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(setup_lab.subprocess, 'run', fail)
    assert setup_lab.main(['--device', 'cpu']) == 1
    assert evidence.read_bytes() == b'original'
