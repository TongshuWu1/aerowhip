# Cable Twin Perception Runtime

This is the first clean slice of the rewritten cable twin: synchronized ZED
RGB-D acquisition, the preserved three-channel PIDNet, lossless source
recording, deterministic replay, and a small diagnostic viewer. It contains no
particle filter, object tracker, contact model, or physics code.

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe run_cable_twin.py
```

The live window shows the PIDNet overlay beside registered metric depth. `Q`
quits and `R` starts or stops a recording.

Record without the window:

```powershell
.\.venv\Scripts\python.exe run_cable_twin.py --headless --record-to recordings\experiment.svo2
```

Replay every recorded frame through the same PIDNet path:

```powershell
.\.venv\Scripts\python.exe run_cable_twin.py --svo recordings\experiment.svo2
```

SVO replay is ordered and never drops frames. `Space` pauses or resumes and
`N` advances one frame while paused. Add `--realtime` to pace replay using the
recorded camera timestamps. `--headless` uses the identical source and PIDNet
path without OpenCV.

## Recording identity

Each recording has exactly two files:

- `experiment.svo2`: lossless ZED stereo imagery and camera metadata.
- `experiment.svo2.json`: exact per-frame live camera timestamps, calibration,
  ZED/depth settings, PIDNet config and checkpoint hashes, thresholds, Git
  identity, and completion state.

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
- The viewer is CPU-only and owns no camera, CUDA, recording, or tracking state.
