# Fine-tuning the selected flight PPO at the lower setup

The user explicitly rejected post-export translation and requested continuation
of `20260908-152029-040639-seed655/checkpoints/best_validation.pt` in a lower
training environment. They then requested drone initial X = -2 m and target
X = -1 m. This supersedes the short-lived translation prototype; CEM is paused.

## Coordinates and run

The requested drone X refers to the unshifted OptiTrack tracking origin used by
FullState. At the nominal settled start:

- Tracking origin: `[-2, 0.01287427254333901, 1.255]` m.
- Cable attachment: `[-1.9933450027145172, 0, 1.2]` m.
- Target: `[-1, 0, 1.1]` m.

The measured body offset remains `[0.0066549972854827175, -0.01287427254333901,
-0.055]` m. Y is unchanged. Start and target retain their existing 5 cm sampling
radii. Explicit use of tracked-origin X makes the relative target displacement
about 6.65 mm different from a rigid -2 m translation of the old attachment;
this is intentional to satisfy the user's exact drone coordinate.

Continuation: `runs/ppo/20260908-192755-046598-seed655`, display name
**M0 flight PPO - lower setup - start X -2 target X -1**. It starts at the exact
selected **293,888-attempt** checkpoint, preserving policy/value tensors, both
optimizers, 9,184 gradient updates and all checkpoint counters. Parent SHA256:
`466f5d8e392357f33df3a7f7b312ee09ad3487cf5cf4f2a8c5e2b75c2c9a6f75`.

Initial continuation budget is **20,480 additional attempts**, ending at
**314,368**, with 2,048 CUDA environments. Stopping history resets for the changed
setup; the inherited 20,000-attempt plateau patience/minimum remains. The
one-second frozen-reference whip objective, reward, 30/30 Hz clocks, model and
both residuals belong to this selected flight PPO. The five-second dynamic
strike v3 objective is not inherited. Lowering the environment does not add a
new ceiling penalty or guarantee visibility: the preferred ~2.6 m and review
limit 2.8 m must be checked on each complete exported rehearsal, including cable
and recovery.

`prepare_training(..., task_overrides=...)` now supports explicit initial-root
and target XYZ amendments, recorded in `run.json`. Changed setups cannot reuse
convergence history. The parent's configs/checkpoint stay unchanged. The native
workspace now contains this setup; its former configuration is preserved in
`config/experiments/20260908-lower-flight-setup/previous_workspace`.

## Verification before continuation improvement

All initial checkpoint contents match the parent exactly after loading; archive
serialization bytes differ but tensors/optimizers/counters do not. Verified all
165 parent, 174 stopped-v3 and 176 new-run frozen source files. **28 targeted
tests passed**, including task amendment provenance/history reset, actual
optimizer continuation, recovery, export and UI regressions.

A real GPU rehearsal re-ran the original selected policy at the requested
coordinates; no saved trajectory was translated. Result:
`runs/rehearsals/20260908-192859-616554-lower-setup-baseline`.

- Valid predicted hit at 0.94 s; zero training/export prefix discrepancy.
- Full reference feasible and full recovery prediction completed.
- Commanded tracked-origin maximum: 2.6623 m.
- Predicted tracked-origin maximum: 2.4228 m.
- Predicted cable range: 0.2469–2.6651 m.

The initial 256-case development evaluation achieved 242 hits (94.53125%), mean
reward 223.0891 and no numerical failures. This is performance **before new
updates**, not evidence of fine-tuning improvement or independent hardware
validation. The policy observes relative positions, so lowering the start and
target together may require little adaptation. Subsequent validation uses the
same lower-setup development cases.

Audit: `runs/audits/20260908-lower-flight-finetune/`. Tested Windows / RTX 4080.
Review the eventual selected continuation checkpoint's full rehearsal before
treating these baseline height numbers as its performance.

## Preserved and withdrawn work

Separate saved PPO bundles, including weights, configs and fitted assets:

- `policies/PPO-flight-baseline-20260908-152029-preserved`
- `policies/PPO-stopped-v3-80896-preserved`

The unsuccessful v3 run remains STOPPED at 80,896 attempts. Its latest checkpoint
SHA256 is `6dfd88a5f7bf247b0bef4d6c02ea8e5ff92937b8ce4e0fc9882c92b09712b17c`.
Do not resume stopped ancestors.

The translation UI, active utility and export recipe were removed. The rejected
prototype code/tests and transformed rehearsal are preserved only under
`runs/audits/20260908-lowered-rehearsal/withdrawn_translation`. They are not flight
or adaptation data. No raw recording or original rehearsal was changed. New
rehearsals use actual lower coordinates during policy inference and simulation.
