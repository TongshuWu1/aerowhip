# Two-branch workflow

| Branch | Role | Entry point |
|---|---|---|
| `main` | Canonical research code, current paper guides and retained evidence | `python run_simulation.py` |
| `deployment` | Portable lab defaults and guided operator workflow | `python run_lab.py` |

Both branches contain the same shared numerical and application code. Main also
includes `run_lab.py`. The deployment branch keeps empty source catalogs and
imports a verified M0 baseline separately. Its private lab ZIP includes that
baseline. Do not merge research selections or data into the lab defaults.

Use separate checkouts for research and deployment. Switching a populated lab
study to main in place may collide with tracked research files. Commit code
changes, merge the shared implementation, and explicitly preserve each branch's
configuration and documentation entry points. Verify both test subsets before
pushing; never force-push to simplify a merge.

The former `twin-rewrite` and `Simulator` branches were consolidated into main's
ancestry. The paused Isaac prototype was committed before retirement. Its local
checkout is retained as a detached historical snapshot with its ignored synthetic
logs. It is not the active paper model and is not restarted by consolidation.

The `aerowhip` local path remains a compatibility junction to the original
physical research checkout. Older experiment paths are preserved; a Git branch
cleanup is not permission to rewrite measurements, models, or frozen snapshots.
