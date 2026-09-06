# Data

`raw_takes/` and `processed_takes/` are immutable real measurements. The processed files contain synchronized 100 Hz drone pose, ten cable markers, and historical FullState commands. They contain no direct thrust or motor-force measurement.

The active numerical model is `../config/model.json`; its `parameter_source`
identifies the applied fit provenance. `baselines/` stores applied snapshots,
`calibration_audits/` stores candidate comparisons. The superseded `cable_model/`
pivot report was retired outside this repository. A directory name alone does not identify the current
best fit. Consult the saved split and evaluation records before making a claim
about held-out performance.

`force_takes/` is a reproducible, generated view for the simplified model. Each
take contains the dynamic attachment state, all 12 DDER node positions and
velocities, validity masks, and `estimated_point_force_world_n`. The extra node
between the attachment and marker `c1` is linearly interpolated to match the
current DDER topology. Positions are differentiated with an 11-sample centered
cubic fit.

The point-force input is estimated from whole-system linear momentum balance
using the 0.159 kg physical vehicle as the point mass and the cable vertex
masses. Internal cable forces cancel in this balance. Aerodynamic drag and
OptiTrack differentiation error remain unmodeled,
so this field is useful for validation, initial states, and force-scale checks;
it is not measured thrust or a supervised ground-truth action.

`dataset_manifest.json` preserves the training, validation, and untouched-test roles. `fig8vertical_002` remains an untouched test and must not be used during model development.

Processing entry points (run only when deliberately regenerating derived data):

```powershell
.\.venv\Scripts\python.exe tools\process_takes.py
.\.venv\Scripts\python.exe tools\build_force_dataset.py
```

The force builder converts the five training and two validation takes by
default. It protects and excludes `fig8vertical_002` unless the explicit
`--include-untouched-test` flag is supplied.

`adaptation_preflight/` contains synthetic pipeline checks, not physical-flight
validation. Flight templates specify the tracking, force and clock contract;
new flight imports and fit candidates are managed through Real-world Updates.
Task/timing versions and workflow jobs retain provenance for local operations.

Measurements, derived datasets, fitted reports and local audit outputs are not
included in the source-only release. Keep originals immutable and select any
public data artifact separately. Ignore rules do not remove previously tracked
data or its Git history; see [publication preparation](../docs/PUBLICATION.md).
