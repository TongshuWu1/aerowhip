# One-day operator runbook

Keep the existing M0 and preliminary recordings. Collect fresh whip recordings
for a new M0 to M1 to M2 chain. Use one target and the same vehicle/controller setup.
This app exports CSVs; execute them in the laboratory's separate flight program.

## Before collection

1. Run the environment check and open the lab app. Confirm the retained M0 bundle
   verifies correctly. Do not refit M0.
2. Create a study with a short unique name. Use the app's assigned take slots.
3. Confirm the target coordinate frame and the physical setup still match M0.
   The current reference target is [1.25, 0, 1.00] m; nominal tracked-origin start
   is [0, 0, 1.255] m.
4. Check launch preparation, tip visibility and command/onset logging using the
   existing lab procedure. Finish setup debugging before collecting study takes.
5. Export M0 and preserve the exact CSV, manifest and original forecast together.

## The 20 executions

| Phase | Executions | Recorded-data roles | Next operation |
|---|---:|---|---|
| M0 update batch | 5 | 001/002/004 adaptation; 003/005 operational validation | Fit new M1; plan and export its whip |
| M1 update batch | 5 | 001/002/004 adaptation; 003/005 operational validation | Fit new M2; plan and export its whip |
| Final comparison | 10 | Every take is final test | Evaluate frozen M0/M1/M2; no fitting |

Final block order: **M2â€“M0, M0â€“M2, M0â€“M2, M2â€“M0, M0â€“M2**.
Keep the paired flights close in time, with comparable battery and preparation
conditions. Do not collect all M0 repeats under one condition and M2 under another.

## After each execution

Import the original controller log and native OptiTrack CSV into the assigned
slot. The app copies files into the study; preserve the originals too.
Review the command match, timing/alignment, usable interval, marker tracking,
and any contact or operator intervention. Record battery and launch notes.

Use **Estimate timing from recorded motion** to obtain a candidate clock offset
from the two measured position streams. Inspect the reported alignment error and
offset spread. This estimate does not establish a shared hardware clock; review
it before importing or using **Save corrected timing**. Save the take review
separately after inspecting the recorded-flight preview.

A large target distance is still a valid experimental observation. Keep poor
approaches, aborted attempts and incomplete tracking with their reasons. Do not
replace a poor-distance trial in pursuit of a better result.

Only the assigned adaptation takes enter fitting. Operational-validation takes
support review of the selected candidate. Final takes are excluded from every
fit, restart, planning choice and threshold adjustment.

## Model updates and planning

Run the existing full update after a reviewed batch. Preserve preliminary
training/holdout roles and replay only designated training inputs. New M1 starts
from retained M0; new M2 starts from this new M1. Historical M1/M2 models and old
whip recordings do not substitute for the new chain.

After a completed update, plan using the frozen experiment settings. Use
**Rehearse + export CSV**, then preview the saved complete motion before executing
the CSV. Select by simulation only. Keep the
same command/recovery package for all repeats in its batch. The app records
failures; it does not automatically retry until an improvement appears.

If a fit is stopped or fails, **Prepare a new attempt** preserves its folder and
freezes a fresh job from the same reviewed inputs and settings. Starting that fit
is a separate action. Do not retry a completed candidate based on its final-test
performance.

Reserve time for the final comparison. Two sequential fits and planning/review
turnaround are the main scheduling uncertainty. If an update fails or cannot
finish, preserve the partial study and continue later. Do not relabel an
unfinished chain as a completed M0 to M2 experiment.

## What the results mean

**Primary task metric:** minimum observed 3D tip-to-target distance during the
fixed 0â€“1.5 s interval after command onset, in centimeters. Smaller is better.
There is no binary 5 cm success requirement. Show coverage and the closest-point
time; the fixed interval can include early recovery.

**Supporting metric:** command-to-tip prediction RMS. Evaluate all three frozen
models on the same final recordings with common commands, causal initialization,
time grid and masks. This needs no extra flights.

Show final paired M0/M2 distances and all per-recording prediction errors.
The final M0 recordings are new repeats of a command seen during adaptation.
Call the M2 command unseen only if it actually differs from both training-stage
command sequences. One target and one update chain do not establish broad
hardware or task generalization.

Preserved M0 has no active cable residual; full M1/M2 introduce it. Describe the
overall refinement procedure, not a controlled extra-data-only improvement.
Keep prediction improvement distinct from physical target improvement.
