> Historical document archived on 9 September 2026. For current work, read [HANDOFF.md](../../../HANDOFF.md). Old running-job and launch instructions below are historical.

# Corrected bootstrap bundle: scope and reuse

The September 8 bootstrap run is
`data/bootstrap_model_runs/20260908-064040-669809`.
Its final numerical review is `BOOTSTRAP_COMPLETE_MODEL_20260908.md` in this
directory. Check the run's `status.json` before treating it as completed.

## Lasting model code

- `simulator/drone_pose_response.py`: timestamped effective loaded-drone
  translation and attitude response, including the exact rigid attachment P/V/A.
- `simulator/drone_pose_residual.py`: bounded acceleration correction with
  an explicit feature/checkpoint schema.
- `simulator/cable/dder.py` and `simulator/cable/residual.py`: constrained cable
  mechanics and dissipative NN correction.
- Causal pose/cable initialization: use past samples only. A hanging-cable
  initialization is a separately evaluated assumption, not measured cable truth.

The execution chain consumes desired FullState commands, predicts the tracked
origin and attitude, transforms to the attachment, and prescribes that boundary
to the cable. The fitted drone already represents the loaded vehicle; adding
another explicit cable reaction to that response would double count loading.
It is an empirical execution model, not an identified motor/thrust plant.

## Temporary historical-data bootstrap

`experimental_data/bootstrap_bundle.py` deliberately pins the reviewed legacy
processed-data version and corrected nominal fit. `bootstrap_cable.py`,
`bootstrap_drone.py` and `bootstrap_combined.py` implement this recorded bootstrap
protocol. They are **not** generic importers for tomorrow's recordings.

All eight preliminary takes may train cable physics/NN. Only the three
whip-through-simulated-reference takes train the drone response/NN. The legacy
whip CSV is maneuver-only: pre/post holds retain their actual logged commands
and separate phases. No unflown CSV tail is invented. Exclusions are saved masks;
the original recordings, historical splits and old model artifacts remain intact.

After prospective assessment of a frozen starting candidate, subsequent
adaptation should use compatible recordings from the corrected route. Old
weights can initialize the next model without replaying old samples in its loss.
New-data selection and per-take complete-export/event association still need
their explicit integration; the current bootstrap does not enforce retirement
across every older loader or UI page.

## Portable component files

After completion, `bundle/` contains:

- `drone_model.json` and `drone_residual.pt`;
- `cable_model.json` and `cable_residual.pt`;
- `manifest.json`, with hashes and the coupling/activation contract.

`experimental_data.bootstrap_combined.load_bundle(folder)` validates hashes
and resolves relative checkpoint paths after copying the folder. The cable
component has its own schema; it is not a replacement old point-force model
JSON. Loading these files does not change active calibration or select a PPO.

The selected GUI/PPO/export path is preserved. A new 30 Hz force policy and
matching FullState integration, common feasibility checks and consistent
rehearsal/export remain separate work. This bundle is not a flight command
package and has not been validated for recovery or the unrecorded hit-time tail.

## Reproduction

Use the repository virtual environment. Each stage is explicit:

```powershell
.venv\Scripts\python.exe tools/fit_bootstrap_bundle.py --prepare --stage cable
# Use the NEW job path printed above for the remaining stages:
.venv\Scripts\python.exe tools/fit_bootstrap_bundle.py --job <new-job> --stage drone
.venv\Scripts\python.exe tools/fit_bootstrap_bundle.py --job <new-job> --stage combined
.venv\Scripts\python.exe tools/fit_bootstrap_bundle.py --job <new-job> --stage report
```

Do not rerun a fit into a completed stage directory. Preserve its source/input
snapshots and failures. The report generator currently targets the named
September 8 review document; preserve that document before using the generator
for a different study. No stage starts PPO/SAC, changes the selected model, or
sends flight commands.
