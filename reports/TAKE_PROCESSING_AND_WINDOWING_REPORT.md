# Experimental Take Processing and Prediction-Window Report

**Repository:** `C:\Users\wts28\Documents\PHD\particle_filter_cable_project`  
**Workspace inspected:** 2026-08-28  
**Git HEAD:** `cbdb59b4f05472506599b6f7e08be08e05541ff3`  
**Scope:** documentation of the current production take-processing, dataset-selection, window-generation, and causal-initialization paths. No processing, fitting, simulator, or windowing method was changed for this report.

## 1. Executive summary

The take workflow has four distinct layers:

```text
paired raw CSV files
    -> deterministic parsing, synchronization, ZOH command reconstruction
    -> immutable processed take on the Motive timeline
    -> take role + manual Use/Exclude annotations in dataset_manifest.json
    -> overlapping causal 1.0-s prediction windows
    -> causal UAV/DDER initialization and open-loop rollout
```

The most important windowing facts are:

- The **Motive export defines the scientific take boundary and timeline**. Logger samples outside that interval do not extend the take.
- The current canonical data rate is 100 Hz (`dt = 0.01 s`).
- A base prediction window is 1.00 s long, advances every 0.25 s, and uses five causal frames ending at its initial time to initialize UAV and cable velocities.
- Each accepted base window contains one initial state, 100 command intervals, and 100 predicted states. Commands are indexed `[start:stop]`; measured targets are indexed `[start+1:stop+1]`.
- The causal residual model has a stricter gate. Its 100-ms FIFO uses the ten samples immediately **before** the prediction boundary, while its velocity features require four still-earlier samples. It therefore needs valid causal data back to `t0 - 0.14 s`.
- After `t0`, no measured state is fed back into an open-loop fitting rollout. The nominal and residual models use commands and simulated state only.
- Current Stage A residual fitting uses 320 Training windows and 184 Provisional Validation windows.

## 2. Production files and ownership

| Responsibility | Authoritative implementation |
|---|---|
| Raw-pair classification and deterministic file writing | `experimental_data/io.py` |
| Logger parsing | `experimental_data/logger.py` |
| Motive parsing | `experimental_data/motive.py` |
| Clock fitting and causal command reconstruction | `experimental_data/sync.py` |
| Automatic quality flags | `experimental_data/quality.py` |
| Processed-take creation | `experimental_data/processing.py` |
| Take roles and processed-take loading | `fitting/dataset.py` |
| Manual Use/Exclude masks | `fitting/segments.py` |
| Base prediction-window generation | `fitting/windows.py` |
| Causal UAV and DDER initialization | `fitting/initialization.py` |
| Additional residual-history eligibility | `fitting/stage_a_residual.py` |
| Dataset GUI | `simulator/gui/dataset_widget.py` |
| Processing settings | `experimental_data/default_processing.json` |
| Fitting/window settings | `fitting/default_fit.json` |
| Geometry and simulator settings | `config/default.json` |

The GUI does not contain an alternative CSV parser or alternative window generator. It launches `process_all_takes.py` for processing and calls the same production dataset/window functions used by fitting.

## 3. Raw take collection and pairing

Each raw take is a directory under:

```text
data/raw_takes/<take_id>/
```

The directory must contain exactly:

1. one logger CSV; and
2. one Motive CSV.

Files are identified by their contents, not their filenames. A logger CSV must contain the required logger header fields, including ROS time, logger Motive time, NatNet frame, UAV state, FullState command time/validity, and translational command fields. A Motive CSV is recognized from its metadata header, including Take Name, frame rates, rotation type, length units, and coordinate space.

If there is not exactly one file of each type, processing fails explicitly. Unknown or ambiguous CSV files are reported rather than guessed.

The parser records SHA-256 hashes of both raw inputs. The processed NPZ is written deterministically with sorted members and fixed ZIP timestamps, so unchanged inputs and configuration can be recognized reproducibly.

## 4. What is extracted from each source

### 4.1 Logger CSV

The logger parser preserves:

- ROS timestamp;
- logger-side Motive timestamp;
- NatNet frame number;
- logged UAV pose;
- FullState command source timestamp;
- command validity and command age;
- commanded position, velocity, and acceleration;
- logged command quaternion and yaw;
- raw commanded angular velocity.

NatNet frame order is treated chronologically. A counter reset is permitted, but ambiguous duplicate frame IDs are rejected.

The resolved command contract stored in the processing configuration is:

- position: world-frame metres;
- velocity: world-frame m/s;
- acceleration: world-frame m/s2;
- yaw: radians;
- command quaternion: yaw-derived provenance, not a full commanded roll/pitch/yaw attitude;
- angular velocity: body-frame rad/s, identically zero in the current takes.

### 4.2 Motive CSV

The Motive parser resolves exact labels:

- UAV rigid body: `cf_7`;
- cable markers: `cable:c1` through `cable:c10`.

It retains:

- Motive source time and frame number;
- UAV position and quaternion;
- all ten cable-marker positions;
- a validity mask for the UAV;
- a separate validity mask for every cable marker.

Motive quaternions are read in `x,y,z,w` order, normalized, and made sign-continuous. Missing marker samples remain missing (`NaN`) and are accompanied by masks; they are not silently interpolated during take processing.

Only Motive exports explicitly marked as metres and Global coordinate space are accepted. The present source-to-simulator transform is identity in a right-handed, Z-up world. A non-identity transform is deliberately rejected until quaternion and command-frame transforms are separately implemented and tested.

## 5. Synchronization and command reconstruction

The Motive timestamps define the final array:

```text
time_s[k] = motive_source_time_s[k] - motive_source_time_s[0]
```

Every processed scientific sample is therefore one Motive frame. Logger data is mapped onto this timeline in two robust affine stages:

```text
ROS command time
    -> logger Motive time
    -> Motive take source time
```

Matched NatNet frame numbers supply the clock correspondences. Robust affine fits are used rather than assuming identical offsets or rates.

Valid finite command events are deduplicated by command ROS timestamp. If a timestamp appears more than once, the latest chronological logger row is retained and conflicting duplicates are counted.

At each Motive frame, the command is reconstructed causally with:

```python
selected = searchsorted(command_event_time, observation_time, side="right") - 1
```

This is zero-order hold: the processed frame receives the most recent command event whose mapped timestamp is less than or equal to the Motive frame time. Future commands are never used. A command is marked available only when the corresponding logger/NatNet availability flag is also valid.

Logger pre-history is recorded as metadata and can support causal command lookup, but it does not add scientific samples before Motive time zero. Logger post-history is excluded from the take.

## 6. Processed take artifact

Each processed take is written under:

```text
data/processed_takes/<take_id>/
    take.npz
    metadata.json
    sync_report.json
```

The NPZ contains synchronized Motive-frame arrays for:

- time and Motive frame number;
- UAV position, orientation, and validity;
- ten cable-marker positions and per-marker validity;
- FullState position, velocity, acceleration, yaw/quaternion, and omega;
- command source time, age, and validity;
- automatic quality flags and `auto_frame_valid`.

`metadata.json` records source hashes, field mappings, frame conventions, processing configuration hash, processor source hashes, processed artifact hash, duration, command semantics, and quality status. `sync_report.json` records clock fits, frame matches, command coverage, missing samples, problem intervals, and the scientific-timeline boundary.

Processing is idempotent. If the source hashes, processing configuration, and processor fingerprint match an existing artifact, “Process All Changed” skips that take. The processed arrays are not edited when a user changes a fitting role or segment.

## 7. Automatic quality treatment

The current automatic thresholds are:

| Check | Threshold |
|---|---:|
| UAV position jump between adjacent valid frames | 0.100 m |
| Any cable-marker jump between adjacent valid frames | 0.100 m |
| Attachment/marker interval-length error | 0.025 m |
| Command age flagged stale | 0.100 s |
| Long marker dropout reporting threshold | 0.100 s |

The geometry check constructs the physical attachment point from UAV pose and the measured body-frame attachment offset, then evaluates the ten consecutive intervals from attachment to `c1`, `c1` to `c2`, and so on.

The automatic **critical** mask is:

```text
critical = UAV invalid
        OR UAV jump
        OR marker jump
        OR geometry invalid

auto_frame_valid = NOT critical
```

Two nuances matter:

1. Aggressive motion is not rejected merely because it is fast. Only the explicit jump/geometry/validity tests determine automatic exclusion.
2. An isolated missing cable marker is recorded as dropout but is not, by itself, inserted into `critical`. The base-window initializer still requires all markers over its causal five-frame initialization history. Future cable losses retain per-marker masks. Command validity is handled by the window command-coverage gate rather than by `auto_frame_valid`.

Command staleness is reported diagnostically; the current critical mask does not independently reject a frame only because its age exceeds 0.100 s.

## 8. Whole-take roles and manual segments

Roles and annotations live separately in:

```text
data/dataset_manifest.json
```

The four roles are:

- `training`;
- `validation` (shown as **Provisional Validation** in the GUI);
- `untouched_test`;
- `ignore`.

Only enabled Training and Validation takes enter current fitting/evaluation assembly. Untouched Test is protected from fitting, normalization, and checkpoint selection. Ignore takes remain visible and processable but do not enter fitting.

Manual interval behavior is exact:

- No intervals: the entire Motive take is manually usable.
- At least one **Use** interval: start with everything excluded, then union all Use intervals.
- Only **Exclude** intervals: start with everything usable, then remove the Exclude intervals.
- Exclude is applied after Use, so Exclude wins any overlap.
- Interval endpoints are inclusive.

The final mask used by the base window generator is:

```text
effective_use_mask = manual_use_mask AND auto_frame_valid
```

The current manifest contains no manual segments. It assigns `osc_001`, `fig8_001`, and `fig8_002` to Training; `fig8_003` to Provisional Validation; and the remaining processed takes to Ignore.

## 9. Exact base-window construction

The authoritative settings are:

| Setting | Value |
|---|---:|
| Prediction horizon | 1.00 s |
| Window stride | 0.25 s |
| Current median sample interval | 0.01 s |
| Horizon steps | 100 |
| Stride steps | 25 |
| Causal initialization frames | 5 |

For each take, the code computes the median sample interval and rounds the requested horizon and stride into sample counts. Candidate starts are generated as:

```python
for start in range(initialization_history_frames - 1,
                   len(time) - horizon_steps,
                   stride_steps):
    stop = start + horizon_steps
    history_start = start - initialization_history_frames + 1
```

At the present 100-Hz rate, a window is indexed as follows:

```text
                    causal initialization              open-loop prediction
index:       s-4  s-3  s-2  s-1   s | s+1  s+2 ...                 s+100
time:       t0-.04 .............  t0 | t0+.01 ........              t0+1.00
state data:  [--------- 5 --------] | [------ 100 targets ------------]
commands:                              [command s ........ command s+99]

start = s
stop  = s + 100
```

This means:

- the simulator state is initialized at sample `s`, time `t0`;
- the five-frame causal history is `[s-4:s+1]`, including the state at `t0`;
- 100 commands are taken from `[s:s+100]` with the Python upper bound excluded;
- the command at `s+k` propagates state `s+k` to state `s+k+1`;
- predicted states are compared with measurements `[s+1:s+101]`;
- the reported window interval is `[time[s], time[s+100]]`, exactly 1.00 s at 100 Hz;
- starts advance by 25 samples, so adjacent windows overlap by 0.75 s.

Although the first mathematical candidate can start at sample 4, initial command unavailability commonly rejects early candidates. The first accepted time therefore differs by take.

## 10. Base-window acceptance and rejection order

A candidate is accepted only when all of these gates pass:

1. **Usable trajectory:** `effective_use_mask[start:stop+1]` is true for all 101 state samples, including both endpoints.
2. **Complete causal command sequence:** `command_valid[start:stop]` is true for all 100 propagation intervals.
3. **UAV initialization:** `uav_valid[history_start:start+1]` is true for all five causal initialization samples.
4. **Cable initialization:** every one of the ten marker-valid entries is true over the same five causal samples.

The checks run in exactly that order. Each rejected candidate receives only the first applicable reason:

1. `outside_usable_segment_or_critical_quality_failure`;
2. `command_coverage_incomplete`;
3. `uav_initialization_history_incomplete`;
4. `cable_initialization_history_incomplete`.

The audit also records command coverage and the fraction of valid marker observations over the complete 101-state interval. A partial marker dropout later in a window does not automatically reject the base window if the four gates above pass; marker-level loss masks preserve which later observations are actually available.

## 11. Causal state initialization at the window boundary

### 11.1 UAV state

At `t0`, the UAV state is initialized as follows:

- position: measured UAV position at sample `s`;
- velocity: derivative of a least-squares quadratic fitted to the five causal UAV-position samples `s-4 ... s`;
- orientation: measured quaternion at `s`;
- angular velocity: quaternion increment from the last two causal orientation samples divided by their time difference, using the simulator's world-frame angular-velocity state convention.

No sample after `t0` enters initialization.

### 11.2 Cable/DDER state

The physical measurements are one UAV rigid body plus ten cable markers. The production DDER is more finely discretized: the ten measured intervals are subdivided into two segments each, giving **21 DDER nodes and 20 edges**.

Initialization proceeds as follows:

1. Rigid attachment kinematics prescribe DDER nodes 0 and 1 from UAV pose, velocity, angular velocity, the measured attachment offset, and the clamped tangent.
2. The ten measured marker positions are assigned to their corresponding every-second DDER nodes.
3. Unmeasured intermediate nodes are midpoint-interpolated.
4. Marker velocities are obtained with the same five-frame causal quadratic derivative; unmeasured-node velocities are midpoint-interpolated.
5. Eight production DDER length-projection iterations smooth the measured-site-conditioned seed while keeping the two-node clamp prescribed.
6. One velocity projection enforces the cable velocity constraints and analytic boundary velocity.
7. Marker reconstruction RMSE is checked against 0.010 m. Exceeding the limit raises an explicit initialization error.

This is causal measured-site conditioning, not use of future observations.

Stage A UAV-only fitting calls the shared UAV initializer and rolls out only the UAV portion of the same simulator. EI/Cb remain fixed and cable loss is zero. The full DDER initializer is retained for later cable/joint fitting.

## 12. Additional 100-ms residual-history window gate

The causal UAV residual does not change the base 1.0-s window. It adds an earlier-history eligibility check and an explicit FIFO state.

The residual input at one sample is exactly:

```text
[e_p (3), e_v (3), a_cmd (3)] = 9 features
```

with `e_p = p_cmd - p` and `e_v = v_cmd - v`. Ten samples produce a `10 x 9 = 90` dimensional causal history.

The FIFO used to initialize a prediction at `t0` is:

```text
feature samples: t0-0.10, t0-0.09, ... , t0-0.01
```

Each pre-window measured velocity feature is itself estimated with the accepted five-frame causal derivative. Therefore the earliest raw state sample needed is:

```text
t0 - 0.10 s - 0.04 s = t0 - 0.14 s
```

The exact index layout is:

```text
s-14 ... s-10  s-9 ... s-1 | s | s+1 ... s+100
  ^ derivative support         |t0| open-loop outputs
              [10 FIFO features]
```

For a base-accepted window, the residual gate rejects when:

1. the required earlier index is before the take start;
2. any sample from the earliest derivative support through `t0` lies outside the effective usable mask;
3. UAV measurements are invalid anywhere in that same required prehistory;
4. a command needed for the ten feature samples through `t0` is invalid.

The initial FIFO may use measured causal history because it defines the state at the prediction boundary. During rollout, the first current feature at `t0` is appended and the oldest sample is removed. From that point onward, `e_p` and `e_v` use **simulated** position and velocity. No measured state after `t0` enters the residual.

The Dataset GUI's “Accepted Windows” count and orange window track currently use this stricter residual-eligible set, not merely the base-window count.

## 13. How accepted windows enter fitting

For each physical take, accepted windows are stacked into one batched rollout:

```text
commands:          [100 time steps, number of windows, command dimension]
initial UAV state: [number of windows, state dimension]
measured targets:  [100 time steps, number of windows, output dimension]
```

The output at `t0` is not counted as a prediction target. The rollout states after each of the 100 commands are compared with Motive samples `s+1 ... s+100`.

The Stage A loss hierarchy is:

```text
average prediction samples within each window
    -> average windows within a physical take
    -> average physical training takes equally
```

Thus a longer take does not receive greater top-level weight merely because it produces more overlapping windows. The current residual trainer may draw up to 32 windows per training take for a gradient update, using a fixed seed, but evaluates the complete Training set periodically and retains the best full-training objective. Feature-normalization statistics are computed from nominal-model rollouts of Training windows only; Provisional Validation and Untouched Test do not contribute.

## 14. Current per-take window audit

These counts were regenerated directly from the current processed artifacts and production generators while writing this report.

| Take | Role | Frames | Duration [s] | Base accepted / candidates | Base rejection summary | Residual accepted / base | First base window [s] | Last base window [s] |
|---|---|---:|---:|---:|---|---:|---|---|
| `fig8_001` | Training | 4,285 | 42.84 | 158 / 168 | 10 command coverage | 158 / 158 | 2.54–3.54 | 41.79–42.79 |
| `fig8_002` | Training | 3,035 | 30.34 | 99 / 118 | 19 command coverage | 99 / 99 | 2.29–3.29 | 28.79–29.79 |
| `fig8_003` | Provisional Validation | 5,730 | 57.29 | 188 / 226 | 8 command; 29 use/critical; 1 cable initialization | 184 / 188 | 1.04–2.04 | 56.29–57.29 |
| `fig8vertical_001` | Ignore | 6,976 | 69.75 | 241 / 275 | 34 command coverage | 239 / 241 | 2.29–3.29 | 68.54–69.54 |
| `fig8vertical_002` | Ignore | 5,307 | 53.06 | 198 / 209 | 11 command coverage | 197 / 198 | 2.29–3.29 | 51.54–52.54 |
| `osc_001` | Training | 1,931 | 19.30 | 64 / 74 | 10 command coverage | 63 / 64 | 0.79–1.79 | 17.54–18.54 |
| `osc_002` | Ignore | 3,629 | 36.28 | 126 / 141 | 15 command coverage | 124 / 126 | 1.54–2.54 | 34.79–35.79 |
| `osc_003` | Ignore | 1,800 | 17.99 | 58 / 68 | 10 command coverage | 57 / 58 | 2.29–3.29 | 16.54–17.54 |

Residual-specific rejections are:

| Take | Additional residual rejection reason |
|---|---|
| `fig8_001` | none |
| `fig8_002` | none |
| `fig8_003` | 3 residual-history usable-mask failures; 1 residual-history command failure |
| `fig8vertical_001` | 2 residual-history command failures |
| `fig8vertical_002` | 1 residual-history command failure |
| `osc_001` | 1 residual-history command failure |
| `osc_002` | 2 residual-history command failures |
| `osc_003` | 1 residual-history command failure |

Current active totals are therefore:

| Set | Base windows | Residual-eligible windows |
|---|---:|---:|
| Training: `osc_001`, `fig8_001`, `fig8_002` | 321 | 320 |
| Provisional Validation: `fig8_003` | 188 | 184 |

The final accepted window need not end exactly at the Motive take boundary because starts lie on a fixed 0.25-s stride lattice and every full horizon must pass all gates.

## 15. What the Dataset GUI shows and changes

The **Takes & Dataset** tab:

- scans only processed artifacts for display;
- can launch deterministic processing for one take or all changed takes;
- displays Motive duration, command/UAV/cable validity, residual-eligible window count, and quality status;
- draws separate timeline tracks for command availability, UAV validity, all-marker availability, manual Use mask, and accepted residual windows;
- plays measured UAV/cable data from the processed Motive timeline;
- writes only take roles and Use/Exclude annotations to `dataset_manifest.json`;
- never edits the raw CSV files or processed NPZ when a role/segment changes.

The orange **Windows** timeline rectangles are the accepted residual-history windows returned by `residual_eligible_windows()`. They show overlapping eligible prediction intervals, not raw command segments and not separate physical takes.

## 16. Causality and leakage guarantees

The current architecture enforces these boundaries:

- command reconstruction uses only the latest command at or before each Motive timestamp;
- base initialization uses only five samples ending at `t0`;
- residual initialization uses only samples before `t0` plus the initialized state at `t0` when rollout begins;
- predicted targets begin after `t0` and are used only in the loss;
- no measured UAV state is fed into the simulator after initialization;
- Training-only rollouts determine residual normalization;
- validation data does not train network weights or normalization;
- Untouched Test is excluded from fitting and checkpoint selection;
- take identity and role are not residual input features;
- the GUI reads the same processed artifacts and production window functions as the command-line fit path.

## 17. Important interpretation

A prediction window is not an independently recorded experiment and it is not statistically independent of neighboring windows. It is a causal, overlapping evaluation interval cut from one complete physical take. The physical take remains the top-level experimental unit, which is why loss aggregation gives equal weight to takes after averaging their windows.

The initial five-frame state estimate and the residual's longer prehistory are allowed measured information because both occur at or before the prediction boundary. The 1.0-s trajectory after that boundary is genuinely open loop. This separation is the key property that makes the windows suitable for identifying whether the model can predict UAV/cable evolution rather than merely track it with repeated measurement correction.

