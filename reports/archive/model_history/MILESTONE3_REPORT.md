# Milestone 3 — Experimental Take, Dataset, and Joint-Fit Readiness Report

Date: 2026-08-27 (America/New_York)  
Repository: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project`  
Code base commit at audit: `cbdb59b4f05472506599b6f7e08be08e05541ff3`  
Working tree: dirty because this milestone and the previously requested legacy
reorganization are not committed.

## Executive result

The experimental-data side of Milestone 3 is implemented and verified for the
first physical take. The two source CSVs are preserved byte-for-byte, detected
by content, explicitly synchronized, converted into a deterministic immutable
`aerial_cable_take_v1` product, quality-audited, exposed in the production GUI,
and consumable by causal window and 10-marker-to-21-node initialization code.

All four takes are intentionally **NOT_FIT_READY**, and no parameter fit was
run. The supplied logger now establishes the exact CSV mapping, but the
FullState publisher and Crazyflie command/controller path are still absent, so
the intended meanings of `q_cmd` and angular-rate frame cannot be established
with the confidence required to identify `K_R` or `K_omega`. The data itself
strongly suggests that the logged quaternion is not the complete commanded roll/pitch:
its x/y components are exactly zero while measured pitch spans roughly
`[-30.55, +18.04] deg` during command-valid motion. That evidence is enough to
reject an unverified assumption, but not enough to identify the real API
semantics.

Four physically separate exports are now present (`osc_001` and
`fig8_001...003`), so a take-level training/validation split is possible after
scientific review. All default to disabled/Ignore; no roles were silently
assigned.

Consequently, this is a scientifically guarded partial completion:

- raw processing, synchronization, schema, masks, annotation, windows,
  initialization, losses, result-snapshot infrastructure, and GUI: **complete**;
- logger-field audit: **complete**;
- controller-command audit: **blocked by missing publisher/firmware configuration**;
- Stage A/B/C fitting, fitted parameters, and held-out validation results:
  **not run and not claimed**.

No MPPI, drag, delay, noise, downwash, learned residual, new coupling, or new
fitted parameter was added.

## 1. Architecture

The new production structure is:

```text
experimental_data/
    default_processing.json
    io.py                 content detection, hashes, atomic JSON/NPZ
    logger.py             logger parser; no cable observation export
    motive.py             labeled Motive parser and quaternion normalization
    sync.py               robust clocks, exact frame match, command ZOH
    quality.py            explicit automatic quality flags
    processing.py         immutable take construction and provenance

fitting/
    default_fit.json      visible provisional bounds and settings
    config.py             validated fit configuration
    dataset.py            processed-only take/role loading
    segments.py           layered manual Use/Exclude masks
    windows.py            causal overlapping windows and audit records
    initialization.py     causal UAV/cable state initialization
    losses.py             physical masked and robust losses
    evaluate.py           physical validation summaries
    artifacts.py          immutable dataset/config snapshots
    optimize.py           hard scientific gate; fitting deliberately blocked

simulator/gui/
    dataset_widget.py     take table, roles, segments, measured playback
    fit_widget.py         bounds and readiness diagnostics
    main_window.py        Simulator / Takes & Dataset / Fit & Validate
    viewer_3d.py          measured UAV and sparse marker rendering

data/
    raw_takes/{osc_001,fig8_001,fig8_002,fig8_003}/
    processed_takes/{osc_001,fig8_001,fig8_002,fig8_003}/
        take.npz, metadata.json, sync_report.json
    dataset_manifest.json
    fit_results/

process_all_takes.py
tests/test_experimental_take_processing.py
```

All later physical prediction continues to use the existing
`CoupledSimulator.rollout`; no duplicate fitting physics was created.

## 2. Raw processing

### Content detection

The processor does not infer type from filename.

- A logger CSV is identified by its ordinary header containing the required
  logger fields, including `ros_time`, `motive_time`, `natnet_frame`,
  `cmd_ros_time`, `cmd_valid`, UAV pose, and command fields.
- A Motive CSV is identified from the Motive metadata/multi-row header,
  including `Format Version`, `Take Name`, and the `Frame,Time...` component
  header structure.
- A take directory must resolve to exactly one file of each type. Missing,
  ambiguous, or invalid pairs fail explicitly and may produce a processing
  failure artifact; partial data is not silently accepted.

### Scientific source fields

Motive is the authoritative measurement source for:

- UAV rigid body `cf_7` position and quaternion;
- labeled cable markers `cable:c1` through `cable:c10`.

The logger supplies clocks, frames, availability, UAV pose only for alignment
audit, and recorded command channels. Its live cable fields are not represented
in `LoggerTake` and cannot enter fitting.

### Motive mapping

The parser resolves fields from persistent labels plus measurement/component
names, not lexical ordering. This matters because lexical ordering would place
`c10` before `c2`.

| Quantity | Motive zero-based columns |
|---|---:|
| `cf_7` quaternion X,Y,Z,W | 2,3,4,5 |
| `cf_7` position X,Y,Z | 6,7,8 |
| `cable:c1` position | 721,722,723 |
| `cable:c10` position | 724,725,726 |
| `cable:c2` position | 727,728,729 |
| `cable:c3` ... `cable:c9` | 730...750 |

Quaternions are normalized and made sign-continuous while preserving Motive's
header order `xyzw`. Missing marker coordinates become `NaN` plus an explicit
boolean validity mask; they are never interpolated in the immutable take.

### Idempotence and provenance

`take.npz` is written with fixed member ordering and fixed ZIP timestamps, so a
forced regeneration from identical evidence is byte-identical. The processing
fingerprint covers source hashes, processing configuration, and all processor
source hashes. Unchanged reruns are skipped. Raw input hashes are checked by
tests before and after processing.

## 3. Synchronization

Two robust affine clocks are fitted with centered least squares and iterative
MAD rejection:

```text
t_logger_motive = a_ros * t_ros + b_ros
t_take           = a_take * t_logger_motive + b_take
```

Exact NatNet/Motive integer-frame identity is used for the second fit and for
frame-level validation. Results for `osc_001` are:

| Mapping | slope | offset [s] | RMS [s] | max [s] | used | rejected |
|---|---:|---:|---:|---:|---:|---:|
| ROS -> logger Motive | 0.9999792223610201 | -1787827663.1379054 | 0.0008906283 | 0.0044965553 | 3792 | 3 |
| logger Motive -> Motive take | 0.9999999999999998 | -12554.4500000 | 6.824e-13 | 1.311e-12 | 1930 | 0 |

`osc_001` frame statistics:

- logger frames: 3,795 (`155...3951`), covering a wider interval;
- Motive take frames: 1,931 (`1274...3204`);
- exact matches: 1,930;
- Motive-only: frame 2805;
- logger-only: 1,865, outside the take plus wider-run gaps;
- Motive match fraction: `0.9994821336` (99.948%).

Command events are deduplicated by `cmd_ros_time` and mapped through both
clocks. The command at each Motive frame is the latest mapped event at or before
that frame (causal zero-order hold). `cmd_valid` at the matching logger frame
remains an independent availability gate.

- unique events: 531;
- duplicate-value conflicts: 0;
- event rate: 30.0004768 Hz;
- valid command coverage: 0.9249093734 (92.491%);
- first/last event in Motive source clock: 13.4381762 / 31.1045621 s,
  corresponding to about 0.698 / 18.365 s relative to the exported take.

No future command sample is used to fill an earlier measurement frame.

The three added Figure-8 pairs also synchronize successfully:

| Take | Frames | Exact matched | ROS sync RMS | Command coverage | Logger frame reset |
|---|---:|---:|---:|---:|---|
| fig8_001 | 4,285 | 4,284 | 0.152 ms | 94.516% | 26381 -> 0 at logger row 387 |
| fig8_002 | 3,035 | 3,033 | 0.956 ms | 91.730% | none |
| fig8_003 | 5,730 | 5,729 | 0.620 ms | 98.464% | 158388 -> 0 at logger row 42 |
| osc_001 | 1,931 | 1,930 | 0.891 ms | 92.491% | none |

The supplied logger continues running across a NatNet/Motive frame-counter
reset. The parser now accepts a unique nonnegative frame sequence with an
explicit reset, records the reset in `sync_report.json`, and still rejects
duplicate frame IDs because they would make epoch selection ambiguous. Row
chronology and Motive timestamps are not reordered. Exact matching selects the
post-reset frame epoch represented by each Motive export.

## 4. Command semantics audit

The exact supplied logger is preserved at
`experimental_data/source_audit/experiment_logger.py` with SHA-256
`f27cdcd9b34c15f5ef5b63a4dfb6ecb4586af4ff9e77be92cb776cc5e82c1b95`.
It establishes that the logger:

- subscribes to `crazyflie_interfaces.msg.FullState`;
- defaults to `/<tracking.drone_name>/cmd_full_state`;
- copies `msg.pose.position`, `msg.twist.linear`, and `msg.acc` directly;
- copies `msg.pose.orientation` directly as x/y/z/w;
- copies `msg.twist.angular` without conversion;
- computes CSV `cmd_yaw` itself from the quaternion using `math.atan2`, so
  `cmd_yaw` is radians and is **not** an independent command channel;
- uses the FullState header timestamp to select the latest causal command for
  each consumed NatNet frame, with a 0.2-s default timeout.

The required end-to-end controller audit still cannot be completed because the
publisher/flight script and firmware configuration are not supplied. The
logger observes a message; it does not reveal how that message was constructed
or how the Crazyflie interpreted it.

The following remain unresolved and are recorded explicitly in processed
metadata:

- exact API function and Crazyswarm command;
- Crazyflie firmware controller and configuration;
- estimator and configuration;
- whether `p_cmd`, `v_cmd`, and `a_cmd` are feedforward/setpoint terms and how
  the firmware combines them;
- whether `q_cmd` is full attitude or a yaw-only/placeholder quaternion;
- `cmd_omega` frame and units.

`cmd_omega` values are copied from ROS `Twist.angular`; radians/second is the
standard expected unit, but the publisher source is still required to verify
that it obeys that convention. The frame is not saved by this logger.

Observed command-valid data:

- horizontal commanded acceleration reaches 4.284808 m/s^2;
- logged command quaternion x/y norm is exactly zero;
- logged command yaw is constant at raw value `-0.0020402962`;
- logged command omega is zero;
- measured UAV roll spans about `[-1.241, 2.610] deg`;
- measured UAV pitch spans about `[-30.550, 18.045] deg`.

The evidence is consistent with a position/velocity/acceleration controller
deriving roll/pitch internally, but this is an inference, not a verified API
contract. `FullStateUAVModel` was therefore not changed and the data is marked
`fit_ready=false`. This prevents `K_R`/`K_omega` from absorbing a semantic
modeling error.

To open the gate, supply the FullState **publisher/flight script** and record
the firmware, controller, estimator, and configuration used for these
experiments. The logger source itself is now audited.

## 5. Coordinate system

Motive metadata declares Global coordinates, meters, and quaternion rotation.
The logger/Motive pose comparison at 1,930 matched frames verifies that the
current files already share the same metric world:

- position RMS difference: 1.4904 mm;
- maximum position difference: 3.0211 mm;
- quaternion absolute-dot median: 0.9999999209;
- quaternion absolute-dot minimum: 0.9999925999.

The configured source-to-simulator transform is identity. The simulator frame
is right-handed, Z-up, SI. The old unrelated coordinate permutation is not
reused. Processing deliberately refuses a future nonidentity transform until
its position, quaternion, and command transformations are explicitly tested.

## 6. Processed physical takes

| Take | Frames | Duration | Command | UAV valid | Cable valid | Max interval error | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| fig8_001 | 4,285 | 42.84 s | 94.516% | 100% | 100% | 9.077 mm | NOT_FIT_READY |
| fig8_002 | 3,035 | 30.34 s | 91.730% | 100% | 100% | 21.233 mm | NOT_FIT_READY |
| fig8_003 | 5,730 | 57.29 s | 98.464% | 100% | 99.9616% | 82.953 mm | NOT_FIT_READY |
| osc_001 | 1,931 | 19.30 s | 92.491% | 100% | 99.9793% | 8.228 mm | NOT_FIT_READY |

All geometry/marker/jump failures remain explicit automatic masks. In
particular, `fig8_003` contains 82 geometry-invalid frames, 19 marker-dropout
frames, and six marker-jump transition frames. Those regions must be reviewed
in the dataset timeline and are excluded from windows by the quality rules;
they were not deleted or repaired. `fig8_001` and `fig8_002` have complete
cable marker coverage and no automatically geometry-invalid frames.

### Detailed original take: `osc_001`

| Field | Result |
|---|---:|
| Motive take name | `figure8_001` |
| Capture/export rate | 100 / 100 Hz |
| Frames | 1,931 |
| Duration | 19.300 s |
| UAV valid | 100.000% |
| Cable observations valid overall | 99.9793% |
| Command coverage | 92.4909% |
| Max measured interval error | 8.228 mm |
| Geometry-invalid frames | 0 |
| Quality status | `NOT_FIT_READY` |

Marker validity:

- c1 through c8: 100%;
- c9: 99.9482% (one missing sample);
- c10: 99.8446% (three missing samples).

Automatic diagnostic regions:

- command unavailable: 0.00–0.69 s, isolated at 15.31 s, and 18.57–19.30 s;
- marker dropout: isolated frames at 5.63, 11.61, 12.10, and 14.78 s;
- stale held command diagnostic: 18.47–18.56 s;
- UAV jump: none;
- marker jump: none;
- geometry invalidity: none;
- long marker dropout: none.

The four isolated marker dropouts remain masked observations. They do not
invalidate an otherwise eligible prediction window unless they occur in the
causal initialization history, where the current derivative initializer needs
the complete marker set.

## 7. Segment and role system

`data/dataset_manifest.json` is the only mutable scientific decision layer.
The processed take is never edited. Each take has:

```json
{
  "role": "training | validation | ignore",
  "enabled": true,
  "note": "...",
  "segments": [
    {"start_s": 1.0, "end_s": 5.0, "use": true, "note": "..."},
    {"start_s": 2.0, "end_s": 2.3, "use": false, "note": "..."}
  ]
}
```

Whole takes—not windows from the same take—define Training versus Validation.
This prevents overlapping windows or adjacent parts of one physical run from
being presented as independent validation.

Manual behavior is explicit:

- if one or more Use intervals exist, their union defines the selected domain;
- if only Exclude intervals exist, all unannotated data stays usable;
- Exclude intervals always override Use intervals;
- automatic critical quality masks are layered afterward;
- changing annotations does not change the raw/processed hashes.

The GUI writes the manifest atomically. All four takes default to
disabled/Ignore.

## 8. Causal windows

Frozen current configuration:

- horizon: 1.00 s;
- stride: 0.25 s;
- initialization history: five 100-Hz frames, ending at the prediction start;
- initialization marker RMSE limit: 10 mm;
- cable robust scale: 2 mm;
- validation lead times: 0.1, 0.25, 0.5, and 1.0 s.

A candidate is accepted only if:

- the complete prediction interval lies in the effective usable domain;
- the complete command input interval has causal command coverage;
- UAV initialization history is valid;
- all ten markers are available throughout initialization history.

Marker dropout after the prediction start is retained and handled by loss
masks. A prediction interval is not rejected merely for one missing later
marker. Every proposed window gets an auditable accepted/rejected record.

Before whole-take role gating:

| Take | Proposed | Structurally accepted | Rejected |
|---|---:|---:|---:|
| fig8_001 | 168 | 158 | 10 command coverage |
| fig8_002 | 118 | 99 | 19 command coverage |
| fig8_003 | 226 | 188 | 8 command coverage; 29 critical quality; 1 initialization-history dropout |
| osc_001 | 74 | 64 | 10 command coverage |

The take remains excluded from optimization because it is Ignore and not fit
ready.

## 9. Causal initialization

At each window start:

- UAV linear velocity is estimated by a causal local quadratic least-squares
  derivative over the five past/current pose samples;
- UAV world angular velocity is estimated from the last two normalized
  body-to-world quaternions using their relative rotation;
- measured marker velocities use the same causal polynomial derivative;
- UAV position/orientation are the current Motive measurements;
- cable node 0 and node 1 are prescribed by the production rigid attachment
  and clamped tangent geometry;
- c1...c10 populate DDER nodes `[2,4,6,8,10,12,14,16,18,20]`;
- odd latent nodes `[3,5,...,19]` start at adjacent measured-node midpoints;
- the seed is projected using the production DDER length projection and then
  the production velocity projection with the two-node clamp fixed.

No future measurement enters initialization. No measured trajectory is used as
the later root boundary: scientific rollout remains full command -> simulated
UAV -> simulated clamp -> DDER cable.

The first three accepted windows in each take initialize within the 10-mm
limit. Their ranges are:

- fig8_001: 6.343–6.848 mm;
- fig8_002: 6.897–7.042 mm;
- fig8_003: 2.732–2.841 mm;
- osc_001: 6.741–7.101 mm.

All are below the configured 10-mm per-window threshold. A failed
initialization rejects the window, not the whole take.

## 10. Losses and intended fitting stages

The implemented loss primitives are:

- UAV position: mean squared Euclidean distance, reported as m/mm RMSE;
- UAV orientation: sign-invariant quaternion geodesic angle, reported in
  degrees (never quaternion component subtraction);
- cable: masked pseudo-Huber distance over only measured nodes
  `[2,4,...,20]`, with ordinary all-marker and c10 tip RMSE reported separately;
- missing observations: excluded through explicit validity masks;
- hierarchical structure: windows average within each take, then take losses
  average across training takes so a long take does not dominate by count.

The intended positive log-parameter vector and visible **provisional** bounds
are:

| Parameter | Lower | Upper |
|---|---:|---:|
| K_p | 1.0 | 60.0 |
| K_v | 0.5 | 30.0 |
| k_a | 0.1 | 2.0 |
| K_R | 1.0 | 80.0 |
| K_omega | 0.5 | 30.0 |
| EI [N m^2] | 1e-8 | 4e-4 |
| Cb [N m^2 s] | 1e-10 | 1.5e-5 |

Configured optimizer plan, not executed:

- Stage A: UAV response from UAV position/orientation; EI/Cb fixed;
- Stage B: EI/Cb from markers using the **simulated** fitted UAV/clamp motion;
- Stage C: joint refinement;
- scrambled Sobol candidate initialization: 32 per stage;
- Adam: 100 iterations per stage, learning rate 0.02, gradient clip 10;
- all rollouts through the existing differentiable `CoupledSimulator.rollout`.

### Why no fit result exists

It would be scientifically invalid to implement or run the final command
adapter while its input semantics are unknown. The staged fit entry point
therefore raises `FitNotReadyError` and the GUI Fit button remains disabled.
There are no fitted parameters, training metrics, validation metrics, bound
hits, lead-time claims, or identifiability claims to report.

After the command contract is supplied, the four takes can be reviewed and
assigned to complete take-level training and validation roles. Only then should
the production command adapter and staged optimizer be completed and executed.
No per-window split of one take may be used to bypass independent validation.

## 11. GUI

The existing command remains the single entry point:

```powershell
.\.venv\Scripts\python.exe run_simulator.py
```

It now contains:

1. **Simulator** — unchanged production simulator controls and visualization;
2. **Takes & Dataset** — processed-take table, command/UAV/cable validity,
   sync and fit status, whole-take role selector, timeline, manual Use/Exclude
   intervals, playback slider, measured `cf_7` body, connector, c1...c10, and
   command ghost when causally available;
3. **Fit & Validate** — current initial parameters, visible provisional
   bounds, take/window counts, and explicit blocking reasons.

The dataset tab imports processed NPZ artifacts only; it contains no raw CSV
parser. Playback is visualization only and does not reconstruct hidden DDER
nodes or alter simulation. The same persistent viewer implementation is reused.

The Fit button remains disabled and the reason is shown. This is not a UI
failure; it is the required scientific safety behavior.

## 12. Reproducibility

### Data hashes

| Artifact | SHA-256 | Bytes |
|---|---|---:|
| raw Motive `osc_001.csv` | `be813abc8742ae3ac0ab251e4388123469878ffa858c0810ca14865cff827d43` | 2,534,471 |
| raw logger `experiment_osc001.csv` | `ea63f4333c011e6628eae243ec7981349dd8696361ccf50ccd7ff13f86a8721b` | 3,626,974 |
| processed osc_001 `take.npz` | `08481352225a3e01f96ae7c84998f998b3ffeb98078e1fa6008c859123eaec2e` | 498,907 |
| fig8_001 logger | `3bec24506c242b7cbd88d641f827cb57caca862e771fc5f7fd81ae2a2d4bfdba` | 6,285,172 |
| fig8_001 Motive | `faa278bdb7a42f14b8500e6e38b87de9353129f085b057e71c37907c62b634a4` | 9,962,399 |
| fig8_002 logger | `5ccda8ad1db9f83e2f48a8165a4396778cbc6847cf7824e3eb204f0a87a00921` | 5,316,541 |
| fig8_002 Motive | `18e48b38c8c8734d1c3e3386500f8452de7f8e840a1a009621d4cfde7e41ff60` | 4,785,341 |
| fig8_003 logger | `339f1cf0afbc6fc5223576cb992fac23723ec7d4815ba9a532fb465ec18d7ae3` | 8,480,046 |
| fig8_003 Motive | `39ade6cab3155f5ea113d57be6d1f6365616c364b5eb205a6ce5145351203566` | 13,646,672 |
| processed fig8_001 `take.npz` | `9013eb961fb6f16b66ce848804b6e4367a9c7d002dbb6843f0e6300ac8e36619` | 1,090,806 |
| processed fig8_002 `take.npz` | `d742a704475b6c2e428755d80bde9b9ecdfb9a6f0c48b7c91c9614548e777355` | 778,463 |
| processed fig8_003 `take.npz` | `4703bf2c2744114a1b7556b274af153a7dc185188660734906dae63463d138f4` | 1,496,834 |

### Configuration hashes

| File | SHA-256 |
|---|---|
| `config/default.json` | `e976eb498f506c5670b87a26439f86ad5cd566874df6cfe8e28916f247623ae3` |
| `experimental_data/default_processing.json` | `95a54c740adc3f676d991592e7d1d89ee4b0b042536747b03e2e0c66d16392a7` |
| `fitting/default_fit.json` | `895f0bf7880367c8f71e2dd8dff7c3d148958d2ab92ce71b549ae7a9b0058f54` |
| `data/dataset_manifest.json` | `35ce6cecf13a3542c9431d29c761a93c44aae52fb16f98ff055c11d19a6c8665` |

Processed metadata also stores every processor-module hash, processing
configuration hash, raw hash, output hash, software version, field mapping,
coordinate contract, command-semantics contract, and warnings.

Environment used for verification:

- Python 3.12.10;
- NumPy 2.5.1;
- PyTorch 2.11.0+cu128 / CUDA 12.8;
- PySide6 6.11.2;
- NVIDIA GeForce RTX 4080.

## 13. Testing and integrity

Milestone-focused tests verify:

- content-based raw type detection and frozen raw hashes;
- explicit label mapping, including c10/c2 ordering;
- logger cable observations cannot enter the processed contract;
- deterministic forced processing and unchanged rerun skipping;
- source/config/source-code fingerprinting;
- schema, frame, clock, coordinate, command, validity, and NaN-mask facts;
- causal window counts and 21-node clamped initialization;
- scientific fitting gates;
- exclude-only annotations preserve unannotated evidence;
- the production GUI exposes all three tabs and keeps fitting disabled.

Verification result:

- production repository suite: `40 passed`;
- preserved `offline_dder` standalone suite (run from that project root):
  `20 passed`;
- targeted Python compilation: passed;
- `git diff --check`: passed.

The root `pytest.ini` deliberately excludes the independently preserved
`offline_dder` and `legacy` trees. The offline suite retains its own package
root and is therefore invoked from `offline_dder/`.

## 14. Definition-of-done status

Items 1–34 and GUI structure item 40 are implemented subject to the explicit
command-semantics gate. Items 35–39, 41, and the fit-specific part of 42 cannot
be honestly completed from the supplied evidence:

- Stage A/B/C were not run;
- validation is not available;
- no fit artifact or parameter claims exist;
- no used fitting windows exist, although proposed/rejected window auditing is
  implemented.

This negative gate is the correct result under the instruction: *if controller
semantics cannot be determined with confidence, processing may proceed but
fit_ready=false*.

## 15. Required next input

To continue without guessing, provide:

1. the exact FullState publisher/flight-controller source that generated the
   subscribed messages;
2. Crazyswarm/Crazyflie API and version;
3. firmware controller, estimator, and relevant configurations;
4. confirmation of quaternion intent and the angular-rate frame/unit contract;
5. a reviewed whole-take Training/Validation assignment for the four processed
   experiments.

Until then, the trustworthy result is a reproducible, inspectable experimental
take and fitting-ready infrastructure—not a fitted physical model.
