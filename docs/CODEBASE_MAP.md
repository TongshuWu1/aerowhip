# Codebase map

This file distinguishes supported execution paths from reproducibility tools
and retained legacy code. Existence in the repository does not make a module a
public workflow.

## Supported workflow

```text
run_offline_fitting.py
  -> optitrack_offline.gui
  -> OptiTrack CSV review, EI/Cb fitting, held-out validation
  -> optitrack_offline/models/cable_model.json

run_online.py
  -> drone_mpc.receding_mppi_gui
  -> accelerated DDER-MPPI receding control
  -> optional between-strike distributed EI/Cb adaptation
```

There is no supported learned-policy branch.

## Active online modules

| Module | Responsibility |
|---|---|
| `drone_mpc/problem.py` | Lightweight strike/task contract, independent of CasADi |
| `drone_mpc/model.py` | Strict cable-artifact loading and provenance |
| `drone_mpc/reduced.py` | Mass-conserving model reduction and distributed-state transfer |
| `drone_mpc/simulator.py` | Point-mass root plus captured DDER candidate propagation |
| `drone_mpc/mppi.py` | Authoritative MPPI settings, objective, sampling, and plan update |
| `drone_mpc/cuda_mppi_cost.py` | Fused CUDA continuous-event/cost evaluation |
| `drone_mpc/receding_mppi.py` | Shift/solve/execute/replan loop and execution artifacts |
| `drone_mpc/distributed_adaptation.py` | Distributed EI/Cb monitoring, fitting, validation, runtime store |
| `drone_mpc/online_adaptation.py` | Persistent between-strike adapter integration |
| `drone_mpc/trajectory_canvas.py` | Reusable active trajectory viewer without legacy solver imports |
| `drone_mpc/receding_mppi_gui.py` | Supported research UI |

## Active offline modules

`optitrack_offline/` is the authoritative one-attachment/free-tip
identification package. Its parser, configuration, fitting, validation, viewer,
and GUI are all active. The CSV and generated validation/model directories are
experiment data, not source.

## Reproducibility and engineering tools

`research_tools/` contains benchmarks, frozen-reference generation, profiler
scripts, sensitivity/ablation studies, report renderers, and earlier experiment
drivers. They may import slower reference implementations intentionally. They
are not launched by the public UIs.

`reports/` contains generated evidence. A report can describe a historical
experiment and should not be interpreted as the current public architecture.

`isaac_whip/` is a separate 6-DoF simulator-validation project. It does not
replace the point-mass MPPI plant until its physical equivalence and controller
integration are validated.

## Retained legacy modules

| Module/path | Status |
|---|---|
| `drone_mpc/mpc.py` | Legacy IPOPT controller; imports CasADi; not imported by `run_online.py` |
| `drone_mpc/realtime.py`, `realtime_gui.py` | Earlier realtime/CEM experiments |
| `drone_mpc/oracle.py`, `perfect_model.py`, `perfect_model_gui.py` | Solve-once and diagnostic experiments |
| `drone_mpc/trajectory_gui.py` | Earlier full application; active UI uses extracted `trajectory_canvas.py` |
| `drone_mpc/adaptation.py` | Earlier tip-only parameter benchmark, not production adaptation |
| `cable_twin/offline`, `cable_twin/online` | Retained RGB/ZED particle-filter prototypes |
| `research_tools/legacy_*` | Explicit compatibility launchers for those prototypes |

Legacy files are retained where tests, reports, or scientific reproduction
still depend on them. They should be archived or removed only through a
separate migration with artifact provenance preserved.

## Known boundaries

1. The public online UI is still an exact-state simulation testbed.
2. The drone plant tracks commanded acceleration and ignores cable reaction.
3. The physical observation contract is 11 ordered material points; changing
   DDER node count changes only the simulation grid.
4. The tracked model may be a labelled provisional transfer until a final
   one-attachment/free-tip artifact is published.
5. Settings/execution data under `data/drone_mpc/` are user experiment data and
   are intentionally not deleted by code cleanup.
6. Large captured/training datasets need a deliberate external-storage or
   deduplication migration; they are not safe automatic cleanup targets.

## Dependency boundary

The supported online launcher imports without CasADi. CUDA PyTorch is required
to run it. CasADi remains optional and is needed only for the legacy IPOPT
module. Tkinter is supplied by the Windows Python installation.
