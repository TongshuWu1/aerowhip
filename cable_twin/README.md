# Cable Twin Perception Runtime

This is the first clean slice of the rewritten cable twin: synchronized ZED
RGB-D acquisition, the preserved three-channel PIDNet, cable observation-graph
extraction, lossless source recording, deterministic replay, and a metric 3D
scene viewer. The live path contains no particle filter, object tracker, or
contact model and does not yet execute cable physics.

The separate [offline DDER workflow](../offline_der/README.md) now extracts
fixed-length reference trajectories from deterministic replay and identifies a
versioned free-cable bending/damping model. The live runtime still does not run
that model or a particle filter.

Launch its dedicated desktop application separately:

```powershell
.\.venv\Scripts\python.exe run_offline_der.py
```

The offline controller has capture, extraction, fitting, artifact review, and
process-log pages. During capture it launches this exact live runtime and its
OpenGL viewer as a separate process, so camera ownership and rendering remain
unchanged. After the recording is finalized and the viewer is closed, the
controller runs extraction and fitting without camera or OpenGL contention.

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe run_cable_twin.py
```

The live window is a metric RGB point cloud built from the same registered
depth consumed by the pipeline. It overlays measured 3D graph edges, endpoint
observations, complete route hypotheses, and crossing-pairing alternatives;
invalid depth and hidden geometry are deliberately not drawn. A small
top-right inset shows RGB with the three PIDNet masks. `D` temporarily switches
that inset to registered depth; depth does not occupy a permanent pane. `Q`
quits and `R` starts or stops a recording.

Use left-drag to orbit, right-drag to pan, the mouse wheel to zoom, and `Home`
to refit the view to the latest cloud. The viewport is fitted once and then
stays fixed until that explicit refit.

Record without the window:

```powershell
.\.venv\Scripts\python.exe run_cable_twin.py --headless --record-to recordings\experiment.svo2
```

Replay every recorded frame through the same PIDNet and observation path:

```powershell
.\.venv\Scripts\python.exe run_cable_twin.py --svo recordings\experiment.svo2
```

SVO replay is ordered and never drops frames. `Space` pauses or resumes and
`N` advances one frame while paused. Add `--realtime` to pace replay using the
recorded camera timestamps. `--headless` uses the identical source and PIDNet
path without importing OpenGL or constructing viewer snapshots.

## Recording identity

Each recording has exactly two files:

- `experiment.svo2`: lossless ZED stereo imagery and camera metadata.
- `experiment.svo2.json`: exact per-frame live camera timestamps, calibration,
  ZED/depth settings, PIDNet config and checkpoint hashes, thresholds,
  observation configuration, Git identity, and completion state.

Canonical replay requires and validates the JSON sidecar. The exact timestamp
timeline is used because SVO quantizes its embedded timestamp by less than one
microsecond.

Automatic stereo self-calibration is disabled for both live capture and replay
so the same stored factory rectification is used on both paths. This setting is
part of the validated camera configuration.

ZED SVO does not store the computed NEURAL depth image. Replay recomputes
registered depth from the lossless stereo pair with the recorded SDK and depth
configuration. This is reproducible at the configured algorithm level but is
not a promise of byte-identical depth; byte-exact depth would require a separate
lossless depth stream later.

## Runtime behavior

- Live capture and SVO writing stay on one camera-owning thread.
- The consumer receives only the latest live RGB-D frame, so visualization or
  inference cannot slow camera capture or source recording.
- Deterministic SVO replay is demand-driven and returns every frame once.
- PIDNet remains unchanged in `NN_collection_training`; `perception.py` is the
  single compatibility boundary.
- `cable_observation.py` skeletonizes each body component, constructs a
  compressed graph, associates visible endpoint components, enumerates
  complete edge-simple routes, and retains all three degree-four crossing
  pairings. Graph samples are lifted from registered depth with explicit
  validity and metric covariance; the mixed central crossing pixel is never
  asserted as a 3D cable point.
- One dedicated thread owns the FreeGLUT/OpenGL context. It receives a
  capacity-one latest snapshot, so a slow display cannot slow live capture or
  deterministic SVO consumption. Viewer drops and source replacements are
  counted separately.
- HD1080 RGB-D is sampled once at the configured stride for visualization.
  Deprojection preserves the original calibrated pixel coordinates and ZED's
  right-handed Y-up camera frame.
- Rendering is capped at 15 FPS to avoid taking GPU scheduling time from
  PIDNet; 3D deprojection, inset composition, buffer upload, and drawing all
  run on the viewer thread.
- Scene contracts accept future posterior cable surfaces/centerlines and a
  posed cube mesh. Those inferred geometry fields remain empty in this slice;
  the viewer draws only current measured cable evidence and never fabricates a
  cable, cube, table, or floor.
