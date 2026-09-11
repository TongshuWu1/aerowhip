# Local setup and transfer

The current development evidence is Windows with an NVIDIA RTX 4080, Python 3.12
and PyTorch 2.11.0+cu128. Ubuntu and other GPU configurations need their own checks;
old workstation references are not proof of support on another machine.

Follow [INSTALL.md](INSTALL.md) for environment creation. Use the project root as
the working directory and the virtual environment's Python interpreter. Opening
the desktop application does not require starting fitting or training.

For an existing clean checkout:

```text
git fetch origin
git checkout twin-rewrite
git pull --ff-only origin twin-rewrite
git lfs pull
```

Preserve uncommitted work before switching an existing checkout. LFS pointers
must be hydrated to read checkpoints, trajectory arrays, packages and research CSVs.
Model/run snapshots may contain historical absolute paths; use portable rehearsal
bundles or identical-hash relative assets rather than rewriting original evidence.

```powershell
.venv/Scripts/python.exe run_simulation.py
```

This opens the six-page PVA desktop. Inspect the named completed MPPI in
[HANDOFF.md](../../HANDOFF.md). Fitting/PPO remain stopped; no automatic diagnostic
restart or physical flight is part of setup.

The ROS/Crazyswarm2 flight computer remains a separate interface/environment.
Do not infer its firmware, topics or motor configuration from the offline planner.
See [flight data and replay](FLIGHT_ADAPTATION_QUICKSTART.md) and the historical
[controller interface review](CONTROLLER_INTERFACE_REVIEW.md).

For paper work, begin with [PAPER_WRITING_HANDOFF.md](../paper/PAPER_WRITING_HANDOFF.md).
Older transfer packages and workstation-specific validation reports are archived;
they are not the current setup instructions.
