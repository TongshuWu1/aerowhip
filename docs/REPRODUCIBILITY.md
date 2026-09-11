# Method and reproducibility

The main deployment method is a cascaded loaded-UAV response and distributed
cable model, staged regularized identification, and offline MPPI-inspired
whole-maneuver planning. Onboard UAV tracking feedback remains active; the cable
task follows frozen 30 Hz desired PVA commands.

The lab app organizes the existing method. It does not implement a new fitting
algorithm, online cable-feedback MPC, or an aircraft sender.

## Preserved evidence

The baseline bundle retains original M0 model/weight identities, the exact CSV
and original preflight prediction, and preliminary replay inputs. Imported
provenance records source identities; portable derived metadata resolves assets
inside the destination checkout.

A study binds its take roles, raw bytes, review decisions, parent models, fitting
jobs, planned commands and forecasts. Preserve failures and intermediate records.
Do not regenerate an old forecast with a newer model and call it the original.

The baseline cable residual is disabled, while the full M1/M2 updates enable it.
A preserved-M0 study evaluates the complete refinement procedure including this
capacity difference. M1/M2 matched prediction is a same-class comparison.

## Reporting

Use continuous observed minimum 3D target distance over 0â€“1.5 s as the main task
endpoint. Do not equate it with full trajectory RMS. Keep missing coverage
visible, and do not interpolate across large tracking gaps.

Original nominal-start forecasts measure operational prediction. Common-history
postflight replays compare frozen models on identical observations and commands.
Both are useful; they are separate measurements.

The existing 13 historical whip takes are development evidence. The deployment
study collects new whip recordings while retaining the existing preliminary
baseline. Final comparison recordings stay outside fitting and tuning.

## Source and assets

The deployment branch contains runnable source and portable defaults. Local data
and generated jobs are ignored by Git. The private colleague ZIP also contains
the verified baseline. A public source release requires a separately distributed
baseline/data artifact to reproduce the retained model's recorded trajectories.

Use tools/build_lab_release.py to package the current source, optionally including
the verified baseline for private lab transfer. It does not export Git history,
old studies, caches or virtual environments.

The numerical implementation remains in simulator/, planning/, learning/ and
experimental_data/. Legacy force/PPO/SAC entry points remain for research
compatibility and are not the main lab workflow. No selected PPO/SAC job restarts
when the app opens.

See docs/VALIDATION.md for the actual checks and their limits. A passing GUI or
packaging test does not establish physical accuracy, paper success or validation
on an untested GPU.
