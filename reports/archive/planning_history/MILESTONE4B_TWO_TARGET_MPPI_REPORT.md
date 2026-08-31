# Milestone 4B — Streamlined Production UI and MPPI Replay Video Addendum

## Outcome

The production GUI addendum is implemented and verified. The normal interface now has exactly four pages:

1. **Simulator**
2. **Data**
3. **Model**
4. **Planning**

The existing `canonical_whip_v1` deterministic replay was rendered to a phone-compatible H.264 MP4 without rerunning MPPI or physics. The replay remains a scientifically failed near-miss and is visibly labeled **FAIL**.

The two-task Milestone 4B scientific run is **not complete** in the current repository. No `figure8_endpoint_whip_v1` task configuration or completed result exists, so no Figure-8 result, video, or success claim was invented. The Planning page shows this task as unavailable.

Email delivery was not attempted because no already-authorized email mechanism is available in the repository or local execution environment. No credentials were requested, stored, scraped, or added.

## Scope and scientific invariants

This addendum did not change:

- the frozen model;
- UAV or cable parameters;
- residual weights or normalization;
- DDER equations or backend;
- task definitions;
- MPPI sampling, cost, update, or success criteria;
- the protected-test state;
- any physical experiment or real-flight interface.

The active model remains:

`MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`

with status `MODEL_FROZEN_FOR_MPPI`, model integrity `Verified`, and `ready_for_mppi = true`.

`fig8vertical_002` remains **PROTECTED — NOT EVALUATED**.

No real Crazyflie command was transmitted and no hardware execution occurred.

## Existing planning-result audit

Only one completed planning result was present:

`data/planning_results/canonical_whip_v1/2026-08-29T041300.459080Z`

Its authoritative batch-one deterministic replay reports:

| Quantity | Value |
|---|---:|
| Task result | FAIL |
| Minimum/reported tip error | 56.416 mm |
| Tip total speed | 9.634 m/s |
| Directed tip speed | 7.808 m/s |
| Direction error | 35.860 deg |
| Maximum UAV displacement | 1.094 m |
| Maximum UAV speed | 3.955 m/s |
| Valid hit time | none |
| First target-entry marker | none |

The saved 4A numerical-consistency artifact records that the short three-step batch test passed to `1.397e-7`, but the sampled batch winner and batch-one replay diverged over the aggressive 0.70-s horizon. Both paths classified the task as FAIL, so the scientific outcome did not change. The MP4 uses only the authoritative saved batch-one `final_replay.npz`; it does not present the sampled population result as authoritative. This unresolved long-horizon batch-shape difference means the full Milestone 4B consistency gate is not claimed as passed.

No Figure-8 endpoint result directory exists. Consequently:

- Figure-8 MPPI was not rerun;
- Figure-8 deterministic replay is unavailable;
- Figure-8 video is unavailable;
- no second-task status is claimed.

## UI cleanup

### Final top-level pages

#### Simulator

The Simulator page is now a replay-first visualization page. It shows:

- the saved UAV trajectory;
- the complete cable;
- observed nodes `c1...c10`;
- an identifiable `c10` tip;
- target and desired direction;
- UAV and tip trails.

Normal controls are limited to:

- Play/Pause;
- Reset;
- Load latest plan;
- timeline slider;
- playback speed.

The compact header reports:

- active model: `PR + 12-node DDER`;
- model integrity: `Verified`;
- MPPI readiness: `Yes`.

Replay loads `final_replay.npz` through the read-only planning-results API. It does not run the simulator.

#### Data

The Data page contains one read-only table:

- Take;
- Role;
- Duration;
- Status.

`fig8vertical_002` is visibly labeled `PROTECTED — NOT EVALUATED`. The normal page exposes no protected prediction/evaluation action. Episode internals remain under one collapsed Advanced area and in artifacts.

#### Model

The Model page begins with `MODEL READY` and displays four compact cards:

- UAV;
- Cable;
- End-to-End;
- Geometry.

The normal view shows the frozen 0.7-s validation values:

| Card | Quantity | Value |
|---|---|---:|
| UAV | Position | 26.97 mm |
| UAV | Orientation | 4.01 deg |
| Cable conditional | Distributed | 44.10 mm |
| Cable conditional | Tip | 71.54 mm |
| End-to-End | Distributed cable | 55.37 mm |
| End-to-End | Tip | 82.55 mm |
| Geometry | Cable length | 0.9525 m |
| Geometry | UAV-to-connector | 55 mm downward |

Exact gains, EI/Cb, rest lengths, backend, artifact paths, and hashes are available only under the collapsed `Advanced / Reproducibility` section.

#### Planning

The Planning page is the single normal MPPI workflow. It provides:

- task selection;
- target and desired-direction summary;
- verified/frozen model status;
- one prominent `RUN MPPI` action;
- iteration, tip-error, directed-speed, displacement, feasible-count, ESS, and runtime progress;
- a large PASS/FAIL result;
- REPLAY, SAVE VIDEO, and OPEN RESULT FOLDER actions;
- one collapsed Advanced Planner Settings section.

The current canonical result is shown honestly as FAIL. Figure-8 Endpoint Whip can be selected for status inspection, but its run action is disabled because no authorized task configuration/result exists.

The GUI uses the read-only production status API and the planning/replay APIs. It contains no DDER physics, MPPI cost, candidate sampling, data parsing, or artifact-selection science inside Qt widgets.

### Controls removed from normal navigation

The normal interface no longer instantiates or exposes:

- the old Identification/fitting page;
- raw take-processing controls;
- historical model selectors;
- rejected delay/physics-only variants;
- old topology and geometry controls;
- editable gains, EI, Cb, or node count;
- solver/backend experiments;
- raw source hashes;
- giant fitting/debug tables;
- simulator developer motion/test controls.

The old detailed widgets remain in source history but are not reachable from the normal four-page application.

### GUI screenshots

- `reports/milestone4b_ui_simulator.png`
- `reports/milestone4b_ui_data.png`
- `reports/milestone4b_ui_model.png`
- `reports/milestone4b_ui_planning.png`

## Planner progress integration

An optional, side-effect-free MPPI iteration callback was added. The production runner emits one JSON progress line after each already-computed iteration. The GUI consumes those lines to update progress values. With no callback, the optimizer follows the original path. This does not change candidates, costs, weights, nominal updates, stopping logic, or numerical outputs.

The GUI did not run MPPI during this addendum or during screenshot testing.

## Video artifacts

### Authoritative source

The renderer accepts only a completed planning-result directory and reads:

- `task_config_snapshot.json`;
- `final_metrics.json`;
- `mppi_iteration_history.json`;
- `final_replay.npz`.

It does not construct a simulator, evaluate physics, or rerun MPPI. The source replay SHA-256 is recorded in `video_metadata.json` so the video can be tied to the trajectory it depicts.

### Canonical Single Target Whip

Video:

`data/planning_results/canonical_whip_v1/2026-08-29T041300.459080Z/canonical_whip_v1_final_replay.mp4`

Metadata:

`data/planning_results/canonical_whip_v1/2026-08-29T041300.459080Z/video_metadata.json`

| Property | Value |
|---|---|
| Task label | Single Target Whip |
| Scientific status | FAIL |
| Source | authoritative saved batch-one `final_replay.npz` |
| Container | MP4 |
| Codec | H.264 / libx264 |
| H.264 profile | Constrained Baseline |
| Level | 3.1 |
| Pixel format | yuv420p |
| Resolution | 1280 × 720 |
| Frame rate | 30 fps |
| Frames | 129 |
| Playback duration | 4.30 s |
| Physical duration | 0.70 s |
| Slowdown | 4× |
| Start hold | 0.50 s |
| Final hold | 1.00 s |
| Fast-start metadata | enabled |
| Audio | none |
| File size | 136,983 bytes (0.131 MiB) |
| Source trajectory SHA-256 | `4d41244cdb4a84a5573d240b825d860905b675c7e3880cb01598ddf298e625ca` |
| Video SHA-256 | `e8a64bd02ef80781d088a4ef0093170ac076f612dcc8904d6ce1714f5b9bee76` |

The H.264 Constrained Baseline profile, Level 3.1, `yuv420p`, standard MP4 container, and fast-start metadata were selected specifically for broad iPhone/Android compatibility.

The video includes a fixed camera, UAV, full cable, cable nodes, highlighted tip, target, desired-direction arrow, UAV trail, tip trail, actual simulation time, tip distance, total/directed speed, and final FAIL label. The overlay states `SIMULATION ONLY` and `No real flight performed`.

### Figure-8 Endpoint Whip

Video status: `NOT AVAILABLE — NO COMPLETED TASK RESULT`.

No video placeholder was fabricated and MPPI was not run to satisfy a UI/video requirement.

## Email delivery

`EMAIL_DELIVERY = NOT_AVAILABLE_IN_LOCAL_ENVIRONMENT`

The repository and local environment were checked for an already-authorized email path. None was available. In accordance with the security requirement:

- no SMTP implementation was added;
- no email password or app password was requested;
- no credentials were scraped;
- no plaintext credential file was created;
- no email was sent.

The delivery status is recorded at:

`data/planning_results/canonical_whip_v1/2026-08-29T041300.459080Z/email_delivery.json`

This does not change the scientific result. Once the user provides an already-authorized email mechanism in a future task, only finalized, gate-accepted replay videos should be attached.

## Verification

### Automated tests

Repository suite:

`59 passed in 26.98 s`

Focused GUI/planning/video tests verified:

- application construction;
- exactly four normal pages;
- active model shown as verified/frozen;
- Data page and protected label;
- Model page readiness;
- saved result display;
- deterministic replay loading without simulator execution;
- video cache/source linkage;
- MP4 decoding;
- 1280 × 720 resolution;
- 30-fps frame rate;
- 129-frame count;
- H.264/yuv420p phone-compatible encoding metadata.

### Manual artifact checks

FFmpeg inspection reported:

`h264 (Constrained Baseline), yuv420p, 1280x720, 30 fps`

OpenCV successfully opened the MP4 and reported:

- width: 1280;
- height: 720;
- fps: 30;
- frame count: 129;
- codec FourCC: `h264`.

The video was visually inspected at an intermediate frame. All four normal GUI pages were captured and inspected.

Python compilation completed successfully for planning, simulator, fitting, tools, and tests. `git diff --check` reported no whitespace errors in the files changed by this addendum.

## Files added or changed by this addendum

Production result/replay boundary:

- `planning/results.py`
- `planning/video.py`
- `planning/__init__.py`

GUI:

- `simulator/gui/main_window.py`
- `simulator/gui/replay_page.py`
- `simulator/gui/data_page.py`
- `simulator/gui/model_page.py`
- `simulator/gui/planning_page.py`

Non-scientific progress reporting:

- `planning/mppi.py`
- `run_milestone4a.py`

Verification/support:

- `tests/test_experimental_take_processing.py`
- `tests/test_milestone4b_ui_video.py`
- `tools/capture_milestone4b_gui.py`

Artifacts:

- canonical H.264 MP4;
- `video_metadata.json`;
- `email_delivery.json`;
- four GUI screenshots;
- one video inspection frame.

## Acceptance summary

| Item | Status |
|---|---|
| Four-page streamlined UI | PASS |
| One obvious Planning workflow | PASS |
| Historical/debug clutter absent from normal pages | PASS |
| Advanced details collapsed by default | PASS |
| Replay from deterministic saved trajectory | PASS |
| Phone-compatible canonical MP4 | PASS |
| Canonical task scientific result | FAIL (existing 4A near-miss) |
| Full-horizon batch/replay consistency gate | NOT PASSED in existing 4A artifact |
| Figure-8 endpoint task result | NOT AVAILABLE / NOT RUN |
| Figure-8 endpoint MP4 | NOT AVAILABLE |
| Email delivery | NOT AVAILABLE IN LOCAL ENVIRONMENT |
| Protected test | NOT EVALUATED |
| Real hardware execution | NOT PERFORMED |

The UI and deterministic-video infrastructure are ready. The repository does not yet contain the two scientifically finalized, consistency-gate-accepted task results required to claim completion of the full two-target Milestone 4B.

