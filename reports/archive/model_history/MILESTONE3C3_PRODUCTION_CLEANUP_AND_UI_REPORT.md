# Milestone 3C.3 — Production Cleanup and Identification UI Report

Date: 2026-08-28  
Repository: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project`  
Git HEAD: `cbdb59b4f05472506599b6f7e08be08e05541ff3`

## Executive result

The repository now has one normal simulator/identification path:

```text
FullState command
    → attitude-coupled UAV Physics
    + causal 100-ms UAV Residual
    → predicted UAV pose
    → rigid two-node centerline clamp
    → 12-node isotropic Cable DDER
    → c1...c10
```

The normal GUI no longer selects historical UAV models, a prescribed-root
model, a delay ablation, a physics-only production predictor, or a 21-node
cable. Its Identification page is a read-only scientific status view backed
by explicit model/data/results APIs.

The checkpoint is deliberately **not** labelled ready for MPPI. No Milestone
3C.2/PRE_MPPI freeze exists, and the latest 12-node EI/Cb artifact was fitted
with the superseded 0.943-m geometry. The active geometry is now 0.9525 m.
Consequently, `config/active_model.json` states:

```text
MODEL_NOT_YET_FROZEN_FOR_MPPI
```

The existing 3C.1 EI/Cb values are pinned only as a development fallback so
the simulator can be inspected. They are never presented as a validated fit
for the new geometry.

No UAV fitting, residual training, EI/Cb fitting, long validation rollout,
protected-test prediction, or MPPI run was performed in this milestone.

## 1. Before the cleanup

The previous Identification page mixed several generations of work:

- Physics Baseline, Fixed Delay, and Physics + Residual model selection;
- Stage-A/B/C controls and historical readiness gates;
- old one-second fitting-window concepts;
- multiple-shooting/AL and forward-solver diagnostics;
- a direct protected-test evaluator;
- large artifact/implementation tables intended for developer debugging.

The Simulator sidebar independently exposed “Prescribed Root / Pivot” and
“FullState UAV / Clamped” as peer choices and allowed live editing of gains
labelled “NOT IDENTIFIED”. Thus the GUI could construct a different model from
the one represented by the current fit artifacts.

Configuration also retained rejected delay and long-horizon solver settings,
and `FullStateUAVModel` retained a selectable historical direct-translation
branch.

## 2. Final Identification page

The final page answers one question: **what model is active, what data
identifies it, what has been fitted, and how well does it predict?**

It has five sections:

1. **Active Model** — architecture, coupling, observations, current geometry,
   backend, and expandable numerical details.
2. **Dataset / Identification Data** — whole-take roles, PhysicalEpisode
   counts/durations, residual-eligible suffixes, and cable eligibility.
3. **Parameter Identification** — separate read-only cards for UAV Physics,
   UAV Residual, and Cable DDER EI/Cb, including provenance and parameter
   status.
4. **Validation** — Conditional Cable Validation and End-to-End Validation,
   with a shared lead-time table and 0.70 s visually emphasized.
5. **Production Freeze / Production Status** — explicit manifest, hashes,
   geometry, component sources, protected-test status, and readiness reason.

The red status banner makes the current non-ready state impossible to mistake
for a final production freeze.

![Final Identification page](reports/milestone3c3_identification_page.png)

## 3. Active scientific workflow

There is one fitting entry point:

```powershell
.\.venv\Scripts\python.exe run_milestone3c.py
```

It routes to `fitting/decomposed.py` and retains the decomposed scientific
structure:

1. fit UAV Physics on continuous Training PhysicalEpisodes;
2. fit cable EI/Cb using the measured UAV boundary and measured c1...c10;
3. train the causal UAV Residual on causally eligible episode suffixes;
4. validate Physics + Residual + DDER end to end.

A PhysicalEpisode has one causal initialization followed by continuous
open-loop propagation. Historical overlapping one-second windows are no
longer the primary or selectable fitting unit.

The GUI names the four production operations—Fit UAV Physics, Fit Cable
EI/Cb, Train UAV Residual, and Run Provisional Validation—but this checkpoint
provides no execution buttons. This prevents an expensive fit or protected
evaluation from being launched while the current geometry remains unfrozen.

## 4. Active model and geometry

### UAV Physics

The pinned component values are:

| Parameter | Value |
|---|---:|
| `K_p` | 4.020097778647703 |
| `K_v` | 12.05728865003419 |
| `k_a` | 0.7327301468333923 |
| `K_R` | 69.18419375930429 |
| `K_omega` | 11.456583174321011 |

The source is
`data/model_freezes/MODEL_FREEZE_DECOMPOSED_PRETEST/physical_parameters.json`.

### UAV Residual

- type: causal translational acceleration residual;
- history: 100 ms;
- architecture: 90 → 32 → 32 → 3;
- parameter count: 4,067;
- weight SHA-256:
  `f39b9e63c90f550bd07b6c9c2f91aec23469762aee9df6abc391334a5c75a18c`;
- same-suffix saved validation position improvement: 42.6%.

### Cable DDER

- topology: 12 nodes / 11 edges;
- `node0`: attachment root;
- `node1`: prescribed clamp-support node;
- `node2...node11`: c1...c10;
- observations: c1...c10 only;
- isotropic uncertainty: EI and Cb only;
- coupling: one-way UAV → cable;
- CUDA runtime: float32, PCG32, fused projection, three DDER substeps, four
  position projections.

### Remeasured geometry

- UAV reference → connector: 0.055 m downward;
- connector → c10 cable length: 0.9525 m;
- marker intervals:
  `[0.063, 0.087, 0.100, 0.100, 0.100, 0.100, 0.100, 0.1025, 0.100, 0.100]` m;
- 12-node rest lengths:
  `[0.0315, 0.0315, 0.087, 0.100, 0.100, 0.100, 0.100, 0.100, 0.1025, 0.100, 0.100]` m.

`config/default.json` is the geometry source, while
`config/active_model.json` explicitly pins every component artifact. The
production builder verifies geometry, all five UAV parameters, EI/Cb, the
PCG32 selection, and the residual weight hash before construction. It never
chooses an artifact by modification time.

## 5. Current fit and validation evidence

The UAV Physics and UAV Residual are loaded from the decomposed component
freeze. The displayed cable result is the pinned 3C.1 development artifact:

```text
data/fit_results_12node/2026-08-28T221032.685038+0000_0eea8215
```

Its values are:

| Parameter | Value | Identification status |
|---|---:|---|
| EI | 0.0008243154916666281 N m² | contained |
| Cb | 0.00015000000000000004 N m² s | boundary-seeking |

Cb is not presented as an intrinsic identified constant. More importantly,
both values were fitted using the old `measured_0p943m_12node_v1` geometry.
The UI labels them **DEVELOPMENT VALUES — GEOMETRY REFIT REQUIRED**.

The saved old-geometry artifact passes its own frozen 0.70-s thresholds:

| 0.70-s saved metric | Value |
|---|---:|
| Conditional distributed c1...c10 RMSE | 42.1 mm |
| Conditional c10 RMSE | 66.3 mm |
| End-to-end UAV position RMSE | 11.5 mm |
| End-to-end UAV orientation RMSE | 4.11° |
| End-to-end distributed c1...c10 RMSE | 53.3 mm |
| End-to-end c10 RMSE | 103.7 mm |

Those numbers are shown only as saved development evidence. They are not a
validation of the active 0.9525-m geometry, so `active_model_pass` remains
false even though `saved_artifact_pass` is true.

## 6. Dataset shown by the UI

| Take | Role | PhysicalEpisodes | Physical duration | Residual-eligible duration |
|---|---|---:|---:|---:|
| fig8_001 | Training | 1 | 40.49 s | 40.39 s |
| fig8_002 | Training | 3 | 27.84 s | 27.54 s |
| fig8_003 | Provisional Validation | 2 | 56.41 s | 56.21 s |
| fig8vertical_001 | Training | 10 | 67.44 s | 66.71 s |
| **fig8vertical_002** | **Protected Test** | 2 | 50.82 s | 50.62 s |
| osc_001 | Training | 2 | 17.86 s | 17.66 s |
| osc_002 | Training | 5 | 34.41 s | 34.06 s |
| osc_003 | Provisional Validation | 1 | 15.50 s | 15.40 s |

The protected row is red and states **PROTECTED — NOT EVALUATED**. Episode
counts are metadata/eligibility summaries; no predictor is executed on the
protected take.

## 7. GUI actions and controls

### Production workflow descriptions retained

- Fit UAV Physics;
- Fit Cable EI/Cb;
- Train UAV Residual;
- Run Provisional Validation.

They are explanatory/read-only at this checkpoint.

### Obsolete controls removed

- Physics Baseline / Fixed Delay / Physics + Residual selector;
- direct-effective UAV model selection;
- prescribed-root/pivot selection in the normal simulator;
- editable “NOT IDENTIFIED” gains and EI/Cb in the simulator sidebar;
- short-window Stage-A controls;
- protected-test execution;
- multiple-shooting/AL and GGN/SQP controls;
- forward-DE and synthetic-recovery controls;
- 21-node topology and old marker mapping;
- old 0.943-m geometry as a production option;
- physics-only P as a normal production predictor.

The remaining simulator motion presets all produce a FullState command. The
attitude debug preset was corrected to yaw motion so the GUI does not encode
non-stock roll/pitch in the yaw-only command quaternion.

## 8. Source cleanup

Confirmed dead source was deleted rather than renamed inside the active tree.
Major groups removed were:

- historical Stage-A one-second fitting and reports;
- rejected fixed-delay fitter and report generator;
- historical short-window residual and cross-motion transfer runners;
- multiple-shooting/augmented-Lagrangian implementation;
- long-horizon synthetic/GGN/forward-AD benchmark scripts;
- forward-DE benchmark implementation;
- one-off 0.943-m Milestone 3C.1 fitting runner;
- protected-test evaluator/launcher;
- obsolete fit gate/evaluator/project-state routing;
- the historical direct-effective translation branch;
- tests whose only purpose was to maintain those removed implementations.

Small causal initialization, quaternion metric, and measured residual-history
helpers needed by the current decomposed fitter were moved to
`fitting/production_support.py` before their historical owners were removed.

The machine-readable path-by-path disposition is:

```text
reports/milestone3c3_cleanup_manifest.json
```

### Configuration keys removed

- `prediction_horizon_s` and `window_stride_s` for historical one-second
  windows;
- `command_delay_ablation`;
- abandoned `long_horizon` multiple-shooting/AL configuration;
- `translation_mode` and the direct-effective UAV alternative.

The normal production configuration now resolves to one 12-node geometry,
one rigid clamp, one attitude-coupled UAV model, one active residual, and
explicitly pinned development EI/Cb.

## 9. Preserved scientific provenance

The cleanup did not remove:

- raw experimental CSV data;
- processed immutable takes;
- dataset role/segment manifests;
- `fig8vertical_002` protected data;
- milestone reports;
- fit-result directories;
- model-freeze directories;
- parameter JSON files;
- loss landscapes and EI/Cb profile plots;
- model-freeze source/hash manifests;
- `legacy/` snapshots.

`offline_dder/` remains as an explicitly historical, self-contained reference
because it was previously requested as a named research archive. No active
simulator, fitter, GUI, configuration, or test imports it. Its README now
states that its free-tangent boundary, historical geometries, and one-second
reinitialization problem are not the current aerial identification problem.

`PrescribedRootModel` remains only as a low-level DDER diagnostic/public API
dependency used by focused simulator tests. It is not selectable in the
normal GUI and is not constructed by `build_production_simulator()`.

## 10. Backend/API separation

Physics and artifact selection are outside Qt:

- `simulator/production.py` validates and builds the active predictor;
- `fitting/production_status.py` provides read-only model, dataset, fit,
  validation, and freeze summaries;
- `fitting/decomposed.py` remains the fitting backend;
- `simulator/gui/fit_widget.py` only renders those summaries and opens saved
  plots.

The GUI contains no CSV parsing, loss calculation, geometry construction,
model selection by timestamp, or residual weight loading.

## 11. Protected-test integrity

`fig8vertical_002` remains sealed.

- `config/active_model.json` records
  `protected_test_predictively_evaluated: false`;
- the status API reports `PROTECTED / NOT EVALUATED`;
- the GUI has no protected-test action;
- the removed historical protected evaluator is no longer in the active
  source tree;
- no protected prediction was run during tests or screenshot generation.

## 12. Verification

### Automated tests

```text
51 passed in 26.27 s
```

The suite includes geometry, DDER, clamp, UAV dynamics, residual behavior,
data contracts, PhysicalEpisodes, decomposed fitting primitives, GUI contract,
explicit active-manifest selection, artifact/parameter/hash pinning, protected
status, and a three-step production CUDA smoke test.

### GUI startup

The actual Windows/PyVista application was started, all three pages opened,
and the application closed normally:

```text
['Simulator', 'Dataset', 'Identification']
CoupledSimulator residual_enabled=True node_count=12
closed 0
```

### Production smoke test

The explicit active builder loaded the frozen residual and ran three finite
steps on CUDA float32. The DDER reported `production_cuda_fused`; the state
shape was `[1, 12, 3]` and all cable positions were finite.

### Static checks

- Python `compileall`: passed;
- `git diff --check`: passed with no whitespace errors (Git emitted only the
  repository's existing LF/CRLF conversion warnings);
- removed-model symbol/reference scan: passed for active source;
- `offline_dder` active-import scan: zero references.

No expensive scientific computation was used as a software test.

## 13. Hashes and repository state

| Item | SHA-256 |
|---|---|
| Active source aggregate, 58 files | `439bfb4c048fe9664d140c786490650408bbed8dd926dd2f4843df0db04c8c4a` |
| `config/active_model.json` | `4b2b6e057e4f23ec2731041f640984ea5bf0ef84ee39c64010fbd7b4fb38b3e9` |
| `config/default.json` | `14a7892bc197d11fb4c905653600d693b2aec7f5b2dd2e9edeb020bc1b035f93` |
| Residual weights | `f39b9e63c90f550bd07b6c9c2f91aec23469762aee9df6abc391334a5c75a18c` |
| UAV/residual component manifest | `1b8c60b0ae57035a41f461b9c8caef6c741bdd769bb091d64efb612fb1109a42` |
| Development cable parameter artifact | `ad564589afdbd0ab706062e846943d8807df9de71234e319651f271aee942753` |
| Dataset manifest | `3ede299ed528175e33d3949ce1437e5a6d6dca0ef41b19a433eb9cbaee2cdb08` |

The aggregate algorithm is recorded in the machine-readable cleanup manifest.

The working tree was already an extensive uncommitted reconstruction relative
to Git HEAD and remains dirty (1,863 porcelain entries at final audit). No
commit was created. This milestone did not reset, discard, or rewrite
unrelated user changes; the explicit active manifest and source hash provide
the reproducible checkpoint within that working tree.

## 14. Final status

The software/UI cleanup is complete:

- one obvious normal simulator model;
- one explicit active-model manifest;
- one active decomposed fitting pipeline;
- one clean Identification page;
- current 0.9525-m geometry and c1...c10 mapping visible;
- historical implementations removed from active source when safe;
- scientific artifacts preserved;
- protected test sealed;
- no expensive fit rerun.

The scientific model is **not yet frozen for MPPI**. The next valid scientific
action, when instructed separately, is the missing EI/Cb refit/validation for
the 0.9525-m geometry and creation of a genuine PRE_MPPI freeze. This report
does not perform or authorize that next stage.
