# Legacy bootstrap data and the next recording session

Checked 8 September 2026. This is the data inventory and collection contract;
it does not apply new trims, fit models, retire files or change flight software.
Inventory evidence: `runs/audits/20260908-052743-639452-data-lifecycle-review/inventory.json`.

## What we have and what currently uses it

| Material | Intended use now | Actual status |
|---|---|---|
| `fig8_001/002/003`, `fig8vertical_001/002`, `osc_001/002/003` | Temporary bootstrap cable physics and cable residual data | Eight preliminary takes remain enabled as training in `data/dataset_manifest.json`; every manual `segments` list is empty. Excluded from the new drone-fit scope. |
| `whip1_001/002/003` in adaptation0 | Temporary bootstrap drone nominal/residual and cable physics/residual data | Latest phase-aware version is `20260908-041031-260837`. The subsequent [nominal-only fit](history/20260909-doc-cleanup/NOMINAL_DRONE_FIT_RESULTS_20260908.md) consumes explicit masks but reveals attitude-domain/post-hold limitations. Residual/cable stages remain pending. |
| Old fitted bundles and old processed versions | Historical comparisons and reproduction | These are not a fitted replacement for the newly designed pose model. The old historical loader still pins whip version `20260907-194802-922485`. |
| Measured mass, cable lengths, rigid geometry and marker identities | Continue as setup calibration when the setup is unchanged | Preserve their provenance and distinguish measured quantities from fitted/provisional values. A new recording date does not invalidate the physical dimensions. |
| Next-session recordings through the corrected route | Assess frozen M0, then become the active adaptation data | Not collected or imported yet; prospective pipeline integration is still pending. |

Preliminary raw CSV pairs are in `data/raw_takes/`, with prepared takes in
`data/processed_takes/`. Legacy whip pairs are in
`data/adaptation_rounds/adaptation0/raw/`. Several supplied Motive exports,
including the whips, were already trimmed before import. We have preserved the
files as supplied; we cannot recover omitted motion from those CSVs.

There is an older `ARCHIVED.json` inside adaptation0 and later authorizations
that reopened preparation and expanded fit scope. Neither that old archive
file nor the word `training_ready` alone implements the intended new lifecycle.
Currently **automatic retirement is not enforced across the old entry points**.
Do not rerun a historical fitting button and assume it uses the redesigned
model, latest phase-aware source, or tomorrow's new-data-only scope.

## How we treat the old data

The three legacy whips contain a 10-second commanded pre-hold, approximately
0.66–0.67 seconds of observed CSV maneuver, and a separate hold at a measured
end position. Take 003 intentionally used a longer post-hold. Only 20 of the
25 supplied reference rows were observed. The missing final five are not flown
data and are never appended. The legacy hold is not the new exported recovery.

The latest whip preparation retains every supplied frame, commands in their
original coordinates, validity masks, native logger timestamps, phase labels
and alignment uncertainty. No named cable markers are missing during the
observed CSV maneuver in these three prepared takes; the observed cable
dropouts occur during post-hold. This does not prove marker identity, contact
status or every measurement is correct.

Historical fits already saved automatic exclusion masks for missing markers,
jumps, excess cable chord and possible contact. For example, the old audit
excluded 25 cable frames in fig8_003 and four in osc_001. It quarantined the
first 1.05 seconds of whip1_002 as *possible* near-ground contact; that was a
heuristic, not manually confirmed contact. These are historical fit masks,
not automatically the approved masks for the redesigned fit.

The three latest whip review files still have unreviewed outcomes/alignment
and no pre-contact start/end interval. The UI can save one such interval and
notes, but **the new pose adapter does not consume these review files**, and
neither does the old historical loader. Saving a UI trim is not equivalent to
applying a fit mask. The subsequent standalone nominal fitter explicitly reads
the matching reviews and saves effective masks. Its library also supports
multiple reasoned exclusions by component; this run required none. That
separate path does not make the old UI fitting buttons consume these masks.

Before fitting, inspect each take's plots/3D trajectory and any available video.
Record each accepted window or excluded interval with take, source version,
time axis, start, end, reason and the affected component (drone, cable or both).
Produce the fit's effective masks and inspect what remains. Keep the raw file
whole and preserve a separate processed version; do not splice time across
removed sections or replace the logged command with a simulated one.

A failed hit is useful adaptation data. Exclude corrupt measurements or motion
outside the free-flight model, not high prediction error or unsuccessful trials.
Cable contact/intervention must terminate or exclude affected free-flight fit
windows. Missing cable markers can invalidate a cable loss while the drone
recording remains useful. Good measured attachment/cable motion may support a
cable-only replay even when commands are missing. Initial-state estimation
still needs sufficient valid, contiguous preceding observations.

## Temporary versus lasting components

| Temporary legacy treatment | Lasting workflow |
|---|---|
| Hand-salvaging previously trimmed exports | Record whole takes; select versioned intervals afterward |
| Parsing the old hard-coded 20-row controller and explaining its missing tail | Associate every take with the exact full reference actually used |
| Inferring legacy CSV/post-hold transitions and special take-003 duration | Explicit phase/row evidence from playback |
| Approximate measured-XYZ alignment for the simplified old logger | Prefer shared frame/time evidence; retain estimated alignment with uncertainty if unavailable |
| Old version-specific loaders, masks and historical model candidates | Explicit source versions, reproducible masks and component-specific fitting |
| Manual rescue of old data before M0 is saved | Automated quality checks plus human review of contact, identity, aborts and unusual events |

Geometry, units, validity masks, causal initialization, phase-aware losses,
keeping failures, and comparisons against a frozen model remain in use.
Improved logging reduces manual rescue; it does not eliminate measurement
quality review or contact annotation.

After saving a satisfactory preliminary M0 bundle, exclude these 11 legacy
takes from the active adaptation manifest and default UI fitting selection.
Retain one provenance archive linking M0 to inputs, masks and fitted weights.
Preserve historical study snapshots; permanent deletion is unnecessary for
new-data-only fitting. This retirement has **not** happened yet.

Keep the learned M0 bundle as the starting point. If M1 starts from M0's
weights, it retains that initialization even though old recording samples no
longer enter its fitting loss. This is the intended way to retain the useful
old information while adapting with the new data. Retain cumulative compatible
recordings from the corrected route across later rounds.

## Recording tomorrow / the next session

The recording date alone does not establish a new model. The corrected nominal
fit, residuals and new 30 Hz force-policy/export integration are not complete
yet. Save the actual model/policy/export used; label any legacy-route flight
honestly rather than automatically calling it the corrected M0 experiment.

For each take:

1. Save the exact export folder used on the flight computer, including the CSV,
   model/policy identifiers, planned initial state, target position and predicted
   trajectory when available. The exported P/V/A is the desired controller
   input, not the model's predicted actual response. Keep each take's reference
   even if another take uses a newly generated CSV with the same filename.
2. Start the controller logger **first**. Start the Motive recording while the
   vehicle is still on the ground. Record one continuous take through takeoff,
   the full 10-second pre-hold, the entire CSV maneuver/recovery/final hold, and
   landing. Stop Motive after landing, then stop the logger last. This gives
   command-log coverage around the whole tracking recording.
3. Keep actual OptiTrack capture/export at **100 Hz**, Global metric coordinates
   and quaternion orientation. Preserve original frame numbers/time columns.
   Include cf_7 position **and orientation**, and the named cable markers
   c1 through c10. For the current whip importer the expected names are
   `cable1:c1`…`cable1:c10`; verify C1 nearest the attachment and C10 at the tip.
   If names change, record the mapping instead of guessing or silently renaming
   raw measurements. Preserve the native Motive take/project as well as CSV.
4. Use the verified complete **30 Hz FullState reference** through the maneuver,
   gentle recovery and final hold, with takeoff/pre-hold and landing outside
   that reference as agreed. Confirm the playback program reads all intended
   rows and timestamps. The supplied local `full_state_pva.py` is still the
   old hard-coded 20-row version; it is not evidence that your colleague's
   current program plays a complete export. Preserve their actual program and
   console output showing sample count, completion/abort and lateness.
5. Give the OptiTrack file, controller log and notes the same unique take ID.
   Record fixed target XYZ in the same world frame independently of the desired
   target when physically measured; if a target is tracked, save its identity
   and pose. Record target size/definition and contact evidence where available.
   Save a side video if practical, especially for target/cable contact or
   intervention. Do not require live cable reconstruction as a policy input;
   cable markers are for offline model fitting and diagnostics.
6. Keep failures and aborted takes, with the reason and the last known executed
   row. Never overwrite a prior take. First inspect one complete take for
   coverage/columns/marker labels, then collect several repeats of the same
   reference before varying it. Three to five repeats are a practical starting
   batch, not a statistical guarantee or an instruction to change flight limits.

Record the setup once per session and note changes per take: drone/cable identity,
measured masses, marker/attachment geometry, coordinate origin, controller and
firmware settings/revision if known, and actual playback/logger source. Keep
the measured 157 g drone and 18 g cable-assembly values only while they still
describe that setup. Do not infer a body/COM offset from the recording date.

## Existing logger and the smallest useful improvement

The supplied `experiment_logger_pva.py` logs at a requested 100 Hz and records
latest cached P/V/A/yaw/rate, command age/validity, and `cf.get_position()`.
It does not record every command publication, CSV row/phase, measured drone
quaternion, cable markers, or a shared Motive frame ID. In the old whips its
position values updated about 10 Hz, so its numeric velocity is not our
100 Hz OptiTrack velocity estimate. Use the Motive pose for measured motion.

That existing CSV can remain the controller log. Use a unique output filename;
the supplied logger opens its output with `w` and can overwrite an existing
file. Start/stop order and full recordings improve usability without changing
the log format. Timing would still be approximate with this minimum setup.

For more reliable new data, add a **separate timing/event log** without changing
the existing CSV columns or flight command values. Record take/reference ID,
phase, CSV sample index, planned time, actual command-publication time and
completion/abort. Associate motion-capture frame number/time and its receive
time on the same stated host clock. When publisher and observer use different
clocks, record their mapping; independent wall-clock timestamps alone are not
verified synchronization. Do not substitute subscriber receipt time for
onboard actuation time.

The old preliminary logger already archived in
`experimental_data/source_audit/experiment_logger.py` provides an example of
NatNet frame/Motive timestamps and command timestamps. Its setup dependencies
must be checked before reuse on the colleague's computer. The new event log
and its importer are proposed work, not implemented here. Without this
addition we can still review the minimum files, but some manual timing/phase
alignment remains necessary.

## File handoff and subsequent use

Suggested session layout (collection organization; not a new importer schema):

```
session/
  paired_logs/
    trial_001.csv                 # full Motive export
    experiment_trial_001.csv      # existing controller log
    trial_002.csv
    experiment_trial_002.csv
  plans/trial_001/                # exact export used
  plans/trial_002/
  notes/trial_001.txt
  notes/trial_002.txt
  motive_native/                 # original captures
  timing/                        # event logs if added
  session_setup/                 # actual software/settings and geometry
```

The current UI imports the flat pairs folder and supports one reference CSV
per round. It does not yet associate multiple per-take references or consume
the proposed event logs. Preserve those files now so the forthcoming processor
can use them. Do not assign one round-wide reference to takes that used
different CSVs, or apply the legacy 20-row profile to complete new exports.

Freeze M0 before collecting the first corrected-route batch. Evaluate it on
that batch before fitting the batch: use only the pre-maneuver observations
for initialization and keep full measured future trajectories as targets.
Then fit M1 using the new compatible recordings, with legacy sample replay
disabled. Those same recordings are now M1 fitting data; assess M1 on another
fresh execution. This is the continuing record → assess → adapt cycle.
