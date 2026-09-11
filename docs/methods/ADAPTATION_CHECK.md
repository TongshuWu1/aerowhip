# Adaptation Check

**Current evaluation convention:** all active flight scores and replay use batch hover-normalized Z. Valid calibration is required; the raw toggle is hidden. Drone/tip RMS and target distances use corrected measured geometry against the exact original saved forecast/reference target. Older numeric summaries below are historical and superseded by `docs/methods/HOVER_HEIGHT_CALIBRATION.md` and `runs/audits/normalized-flight-evaluation/results.json`. No raw results are substituted for missing calibration, and these normalized distances are not labeled as uncorrected physical-world hitting errors.

Open the application with `run_simulation.py`, then select **07 Adaptation Check**. Restart an already-running application to load the new page.

1. Select the adaptation batch and flight, then click **Load flight**. The current policy-scoped `adp0` and all five corrected flight pairs are discovered automatically.
2. Press **Play**, scrub the time slider, or select a playback speed. **Whip only** restricts playback and plots to the scored maneuver; **Predicted strike** seeks the nearest recorded frame to the saved prediction's strike time.
3. Compare the **solid orange measured drone, cable and tip** with the **translucent blue predicted execution**. Toggle either body, trails, opacity or camera view. Mouse controls rotate, pan and zoom the scene.
4. **Errors over time** shows measured-versus-predicted drone/tip distances and measured/predicted tip-to-target distances. The purple dashed line marks the predicted strike, gray dotted line the whip end, and dark cursor the replay time. Values are geometric errors, not confirmed physical impacts.

## Inputs and meaning

Drone identity is detected from a complete pose named `cf_3`, `cf3`, `cf_7` or another `cf` number. Exactly one candidate is required for automatic selection. Position XYZ and rotation XYZW are resolved by header names, not fixed column indices. If multiple drones are present, use **Recordings → Current flight batches → Drone rigid body**, enter the exact name and save. This writes `<take>.tracking.json` beside the raw CSV, binding the selection to its hash; it does not edit the recording. Replay and adaptation preparation record the resolved drone identity. Missing pose/cable samples remain missing. Cable labels remain `cable1:c1` through `cable1:c10`.

Changing the drone name does not change the tracked-origin coordinates, attachment offset, mass, fitted physics or saved forecast. These remain the frozen experiment configuration. Historical cf_7 data is not relabeled as cf_3. Previously audited cf_7 timing is not reused for an explicitly selected different body.

Each batch has `simulation_csv/fullstate_30hz.csv` and `flight_take/`. OptiTrack `NAME.csv` pairs with controller `experiment_NAME.csv`. **Open batch…** selects another folder with that layout. The batch's original prediction is found by exact CSV SHA256 among saved rehearsals/CEM results. If it is elsewhere, **Choose saved prediction…** selects the original rehearsal folder; its CSV must still match. Missing/ambiguous pairs are not selectable.

The blue ghost uses `origin_positions_m`, `origin_rotations` and `cable_positions_m` from the original saved `rehearsal.npz`, not the commanded PVA path and not a new rollout under the current model. The prediction's packet array is also checked against its CSV. Loading does not generate trajectories, fit parameters, change configuration, or resume PPO.

Measured drone position/orientation and cable markers come exclusively from raw OptiTrack. The active replay and current adaptation preparation read controller logs in commands-only mode: measured XYZ, velocity and other state fields are not required or exposed. Derived measured velocities/initialization use OptiTrack timestamps and samples. Original saved model outputs remain the prediction, not measurements.

CSV onset is reconstructed from matching moving command packets and controller `cmd_age`. Clock alignment for a new take must be supplied in the batch's `time_alignment.json`, with an entry keyed by take name containing `offset_s`, `source` (the timestamp/shared-event evidence), `clock_verified`, `optitrack_sha256`, and `controller_sha256`. The convention is **controller_time = OptiTrack_time + offset_s**; replay time then subtracts CSV onset. Hashes bind the offset to the exact files/trim. The alignment file is included in replay and fitting provenance. Missing or changed alignment is reported as an error rather than guessed. Do not infer this offset by matching actual motion to commanded motion: that would mix physical response delay with clock offset. This release reads the sidecar; no graphical offset editor is implemented.

Previously audited historical pairs can reuse their exact saved offset from `runs/audits/20260909-adp0-hover-height/hover_height.json` only when both raw hashes still match. Those offsets were originally estimated from measured-stream agreement, retain logging-latency uncertainty, and are labeled **Frozen legacy audit offset**. No new measured-controller alignment is computed. Historical preprocessing tools and immutable old datasets retain their diagnostic controller fields for reproducibility; active state fitting and evaluation use OptiTrack.

Replay uses native measured timestamps within the shared prediction coverage. Predicted arrays are interpolated onto those times (rotation uses SLERP), without extrapolation. File beginnings are never assumed to be the same event. No spatial shift or target-error minimization is applied.

Positions use the original world frame in metres. Measured cable attachment is the cf_7 tracked origin plus its measured orientation applied to the saved tracking-to-attachment offset. The drone glyph represents the tracking frame, not an independently identified physical body/COM frame. Only cable1:c1 through c10 are used; prediction marker dots correspond to those physical sites in the saved cable discretization. Missing measured markers hide their dots and adjoining cable segments. The geometry between observed markers is a visualization, not a measurement of continuous cable shape. Missing samples/time gaps are not joined by trails.

Coverage and stream-alignment RMS are shown separately from flight prediction errors. The current five flights were confirmed by the user to have no contact/intervention: the target is virtual. Initial-state differences remain visible. Loading a flight does not approve it for fitting, and this page is independent of the legacy `data/adaptation_rounds/adaptation0` pipeline.

## Verification

### Adaptation progress: real flights versus model fitting

The **Adaptation progress** view opens **Real flight progress** by default. The four charts show drone prediction RMS, tip prediction RMS, measured tip-to-target distance at the saved planned strike, and closest measured tip-to-target distance during the whip. RMS is the square root of the mean squared Euclidean position error at valid native measured samples in the whip. Round summaries give each complete flight equal weight. Individual flight values remain visible; partial coverage is excluded from round means, missing strike samples remain missing, and pairing errors are shown.

Every real-flight metric uses that batch's exact CSV-matched original preflight prediction. No replay or current model is regenerated to supply its ghost. **M1/adp1 has no score until its actual paired flight logs arrive.** Missing scores display an em dash and an awaiting-flights message, never a zero bar or a held-out model score.

**Model-fit diagnostics** and **Model-fit trajectories** separately retain the held-out and all-data results for investigating identification. These are not new M1 flight performance. A reduction in prediction RMS establishes better prediction under the recorded conditions; measured hitting error separately evaluates the executed maneuver. The training status shown below these tabs is explicitly simulation validation, not flight evidence.

The five adp0 flights currently give 15.05 cm drone RMS and 20.01 cm tip RMS against the original saved forecast. Model-fit RMS differs because it uses recorded initialization and separately fitted predictions. Audit: `runs/audits/20260909-real-flight-progress`; 11 focused progress/alignment/library tests passed, and the native seven-page Qt UI was rendered and checked. These additions supersede the older progress display that put held-out metrics in the overview.

### Original replay verification

- Eight targeted unit/UI tests passed: timing from packet reception, ambiguous playback rejection, missing-sample/time-gap handling, exact reference match, flight pairing, native page integration.
- Windows native Qt/VTK verification replayed all five current flights, forward and backward; checked both drone pose matrices, fixed camera, missing-marker line connectivity, strike seek, playback pause on page exit, background loading, and shutdown.
- Raw CSV and retained checkpoint hashes stayed unchanged. No flight, fit, or production training was launched.
- Audit scripts, screenshots and results: `runs/audits/20260908-adaptation-check-ui/`.
- A broader historical workflow test still assumes target X=+1 and fails against the existing X=-1 configuration (`test_launch_snapshots_isolate_algorithms_and_future_edits`). Its obsolete target assertion was not changed as part of this page. The broad run was stopped before completing its unrelated fitting/training tests; no test worker remains running.
