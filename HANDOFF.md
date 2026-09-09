# Current project handoff

Updated 9 September 2026. Branch: `twin-rewrite`.

Start with [the documentation index](docs/README.md). The detailed
[paper-writing handoff](docs/PAPER_WRITING_HANDOFF.md) covers equations, implementation,
candidate contributions, data lineage, results, limitations and manuscript planning.

## Current work and stop state

Latest MPPI wave work is complete. New accepted plan `20260909-160208-467697`
comes from a cold search in `20260909-155433-585040`, which was stopped after an
independent full-proposal replay, recovery and export passed. The parent has only
13 committed actions; its receding loop did not finish. The derived job performs
no new optimization and must not be reported as a zero-cost cold solve.
No worker remains. See [the MPPI task record](docs/MPPI_PULLBACK_20260909.md).

The user requested this result in Rehearsals and authorized commit/push.
`config/pva/replay.json` selects its exact saved rehearsal at 1.03 s, Side XZ,
quarter speed. Startup loads the existing arrays and command; it performs no
new optimization or ghost regeneration. The six-page native replay UI is verified.

The user is writing the paper in another Codex and investigating physical drone
tracking. Fitting, PPO, campaigns and both heartbeats remain stopped/paused.
No fitting/PPO optimizer is scheduled to restart. PhysX work is paused. No battery
calibration or new flight has been performed.

Fresh PVA M0 `runs/adaptation/20260909-pva-M0-bootstrap` stopped during full-whip
cable residual fitting at update 105, before plateau. Its weights, optimizer and
stopping state are retained. It is unfinished and unpublished. Campaign
`runs/pva_campaign/20260909-M0-PVA` stopped before PPO training.

## Active implementation

Direct desired tracked-origin P/V/A at 30 Hz is generated from bounded XYZ jerk.
The fitted loaded-drone model and drone residual predict pose; rotated attachment
motion drives DDER and the cable residual. Planning is offline, with a frozen
FullState CSV executed through onboard feedback. No new verified flight sender exists.

MPPI uses a 2 s / 60-action rolling horizon, 1,024 sampled trajectories plus
the deterministic mean, and commits one action per replan. It has plateau stopping
without a fixed iteration/wall-clock cap, and a separate 5 s maneuver limit.
Start `[-2,0,1.255]` m; target `[-1,0,1.1]` m. The ordered forward-pull/backward-release
task is checked using modeled aircraft motion at interpolated tip contact.

PPO and MPPI configurations remain independent. New direct-PVA PPO has no trained
matched performance baseline. Historical force-PPO/CEM runs retain their saved
actions, tasks, models, rewards and exported forecasts.

## Latest completed MPPI evidence

Accepted wave plan `20260909-160208-467697`: 41 commands / 1.36667 s whip;
modeled tip contact at 1.33338125 s, minimum distance 2.68814 cm. Bend stages
complete at 0.81333, 1.15333 and 1.30000 s. Peak local turning angle 0.79806 rad;
peak tip speed 5.78353 m/s. At contact drone velocity is -0.62381 m/s and tip
velocity +5.21677 m/s along the strike axis; backward travel 0.132719 m.
Complete CSV is 10.5 s, modeled drone peak height 2.32653 m. The stronger bend
uses more height than the earlier result; provisional envelope checks pass but
physical performance remains unverified. Portable CSV and all 12 arrays reproduce
exactly. 28 targeted tests pass (Windows/RTX 4080). PPO settings are unchanged.

The parent first found a predicted wave-qualified hit at iteration 17 / 94.69 s.
It completed 97 iterations / 508.99 s before being stopped. The accepted candidate
was captured earlier with four committed actions plus the current optimized
proposal; later parent iterations are not its generation cost or extra training.

Read [final audit](runs/audits/mppi-wave-final-20260909/result.json),
[verification](runs/audits/mppi-wave-final-20260909/verification.json), and
[animation](runs/audits/mppi-wave-final-20260909/motion/wave.gif).
The bend criterion is a kinematic proxy, not proof of energy transfer.

## Previous hit-and-reversal result

Run **`20260909-135429-797997`** is completed. Historical normalized M1 simulation only:

- Peak forward aircraft speed +1.30192 m/s, then -0.62040 m/s at tip contact.
- Tip forward speed +5.02739 m/s; backward travel at contact 0.100000267 m.
- Contact at 1.26666754 s; minimum tip distance 1.34697 cm.
- 39 commands / 1.3 s whip; complete recovery/hold CSV 10.4333 s.
- First 30 commands explicitly reused from provisional parent `20260909-133459-410882`;
  last nine refined in 36 new iterations / 56.934 s. This is not cold-solve timing.
- Full configured reference/model checks and exact portable CSV/all-array replay pass.
- 73 relevant tests and the six-page native UI passed on Windows/RTX 4080,
  Python 3.12 / PyTorch 2.11.0+cu128.

Read [the task/result record](docs/MPPI_PULLBACK_20260909.md),
[final audit](runs/audits/mppi-pullback-contact-20260909/result.json),
[verification](runs/audits/mppi-pullback-contact-20260909/verification.json), and
[motion plot](runs/audits/mppi-pullback-final-motion-20260909/pullback.png).
The modeled reversal barely exceeds the configured distance threshold; physical
robustness and real-flight success remain unverified.

Source model: `data/model_candidates/20260908-normalized-M1/model.json`, SHA256
`51d11f38bf8727f52c3641cbdce29f01cb1715c9313752a51299bfa43465e760`.

## Physical issue and battery discussion

The user states that the old sudden drop was caused by acceleration demands
exceeding drone capability. Keep that separate from the unconfirmed battery/height
hypothesis. The Bolt uses a 2S battery. Proposed calibration repeats PVA trajectories
at different voltages while logging actual motor outputs, loaded voltage, pose and
acceleration, accounting for cable forces. Eleven inspected experiment CSV headers
lacked battery/motor fields. Actual flashed firmware, ESC configuration and landing
cutoff remain unspecified; no compensation has been implemented.

## Evidence and history

Real-flight metrics require exact CSV/model/policy lineage. Hover Z normalization
uses retrospective pre/post information; normalized fit diagnostics are not
prospective physical strike validation. Preserve cf7/cf3 distinctions and all raw data.

Current code/research artifacts and the paper handoff were pushed in commits
`74d7c1b` and `93f4fe9`. Documentation cleanup now keeps current guides here and
archives superseded progress/test/UI notes. The full prior decision records are in
[the documentation archive](docs/history/20260909-doc-cleanup/README.md).
