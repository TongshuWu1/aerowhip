# Cable-only pilot: initialization explains more than this physical refit

Completed 9 September 2026 under the user's authorization to run the small
cable-only diagnostic. Original preliminary M0, planner selection and raw data
are unchanged. MPPI, PPO, drone fitting and residual training remain stopped.

## Result

Weighted causal initialization substantially improves this pilot's forecasts.
The subsequent two-parameter physical search offers a modest training benefit
and little improvement on the separate take. It does not establish identified
stiffness or a precise strike model. No candidate was published or selected.

Tip position RMS, in centimetres:

| Configuration | Training 1 s | Separate take 1 s | Training 2 s | Separate take 2 s |
| --- | ---: | ---: | ---: | ---: |
| M0 physical coefficients, residual off, uniform 1 s initialization | 14.29 | 29.43 | 17.43 | 39.11 |
| M0 physical coefficients, residual off, weighted 1 s initialization | 4.75 | 5.50 | 11.46 | 9.92 |
| Pilot physical coefficients, residual off, weighted 1 s initialization | 4.25 | 5.48 | 11.44 | 9.44 |
| Published M0 including its frozen residual, weighted 1 s initialization | 4.75 | 5.29 | 11.47 | 9.71 |

The uniform row is the recently requested uniform one-second initializer,
not the original completed M0 fit's mixed 0.4/0.1-second initializer. These are
conditional cable rollouts driven by measured attachment motion. Do not compare
these numbers directly with the earlier 64.6 cm command-driven coupled result.
The existing residual has a small effect here; this does not evaluate whether a
new correctly trained residual could help.

Pilot comparison (`runs/audits/preliminary1-cable-only-pilot/comparison.png`, not included in this source-only release)

## Exact procedure and data treatment

- Reused the completed fit's frozen, masked, time-aligned preliminary inputs;
  no raw CSV editing, new normalization, clock fitting or gap filling.
- Used native OptiTrack pose to drive the rotated attachment. Controller XYZ
  remains the same cached measurement source, not an independent sensor.
- Selected three evenly spaced eligible windows per take before prediction:
  twelve windows across the four training takes and three from `figure8_002`.
  Eligibility required a full valid past second and a valid two-second future
  trajectory; the offline reference also required its valid local neighborhood.
- The separate take was excluded from the physical search. It has been reviewed
  in earlier diagnostics, so it is not a newly blind or prospective test.
- Kept masses, marker spacing, attachment offset, gravity and solver geometry
  fixed. Cable learned correction was disabled in private diagnostic engines.
- Compared identical initial positions with three velocity estimates: uniform
  quadratic regression over the past second; exponentially weighted regression
  over that same second; and an offline centered quadratic reference.
- Weighted regression uses a 0.02 s time constant previously selected using only
  training takes. It effectively emphasizes recent measurements. This pilot did
  not tune that time constant using the separate take.
- The offline reference uses eleven native samples, from 50 ms before to 50 ms
  after initialization (100 ms first-to-last span). Its saved method key is
  `offline_centered_110ms`; that key counts eleven 100 Hz samples and must not be
  interpreted as the elapsed span. It uses future data and is only a diagnostic.
- Length projection and compatible velocity projection are applied consistently.
  Maximum absolute coordinate adjustment during initial position projection was
  1.17 cm. This adjustment is recorded, not treated as measured calibration.
- Forecasts run uninterrupted from each initial state. Measured interior cable
  states never reset a forecast. Measured future attachment is deliberately
  supplied for this conditional cable test.
- Loss is the existing robust marker-position objective: 70% all markers and
  30% tip, with equal total weight per training take, over one second. Reported
  RMS values are unscaled Euclidean position errors, not the robust objective.
- Evaluation uses approximately 0.25, 0.5, 1 and 2 s. The 150 Hz grid's rounded
  quarter-second endpoint is 0.25333 s; JSON records the actual endpoint.

## Physical search and interpretation

One 7-by-7 logarithmic grid covered EI from 1e-9 to 1e-4 N m² and Cb from
1e-9 to 1e-3 N m² s. It additionally evaluated the M0 physical pair and cold
engineering pair. A seeded, bounded differential-evolution search used population
16 and stopped at generation 6 by the declared practical plateau rule: minimum
6 generations, five stale generations, 0.5% meaningful-improvement threshold.
Its safety ceiling of 24 generations was not reached. This is not proof of
global convergence. Best/current population, random-generator state, every
evaluation and stopping history are saved.

The best pilot pair was EI = 7.3123e-6 N m² and Cb = 1.0604e-5 N m² s.
The training robust objective changed from 0.327747 at the M0 physical pair to
0.302254. However, around Cb = 1e-5, the grid barely distinguishes stiffnesses
across several orders of magnitude. The printed EI is an optimizer output,
not a well-determined material constant. The small separate-take improvement
does not justify replacing the selected M0.

Initial-state uncertainty is not the entire problem. In the predetermined
training figure-eight midpoint, substantial long-rollout errors remain even
with the weighted and offline reference velocities. This leaves model structure,
boundary behavior, physical losses, mass distribution and residual learning as
unresolved contributors; this pilot does not isolate one of them.

Figure-eight midpoint (`runs/audits/preliminary1-cable-only-pilot/figure8_midpoint.png`, not included in this source-only release)

## Numerical checks and GPU execution

- Identical-input forward rollouts reproduced exactly on the GPU.
- At the M0 physical pair without residual, the largest training-window tip
  trajectory difference was 1.77 cm between 8 and 16 substeps, and 0.775 cm
  between 16 and 32. Numerical sensitivity is smaller than the uniform-history
  error here, but is not zero.
- At the pilot pair, separate-take one-second tip RMS is 5.48 / 5.56 / 5.59 cm
  with 8 / 16 / 32 substeps. Two-second training RMS grows from 11.44 to 12.32 cm.
  Thus the search should not be described as integration-independent.
- Executed float64 on Windows 11 / RTX 4080. CUDA graph replay batches
  sixteen parameter candidates across twelve windows: 192 simultaneous
  one-second cable rollouts per batch. Recorded search batches took roughly
  0.96–1.00 seconds, excluding graph setup and reporting.
- 163 unpadded parameter evaluations; final partial batches are padded only for
  graph shape reuse and padded entries are excluded from scores.
- Nine targeted initialization/preparation and artifact-preservation tests passed. All 325 protected
  file hashes checked unchanged after the diagnostics.

This derivative-free pilot does not repair or validate the earlier long-window
autograd disagreement. No neural optimizer was run.

## What follows from this pilot

Retain the consistent weighted initializer as the current diagnostic baseline,
with the offline reference as a check rather than a deployment input. Do not
adopt the new EI value as an identified stiffness. Before another residual fit,
resolve long-window gradient behavior and examine the remaining figure-eight
mismatch with measured attachment. A fixed-attachment release recording would
provide additional evidence separating cable dynamics from moving-state
initialization. Existing data remain useful and need not be discarded.

The small pilot is complete; a full refit, new model publication or MPPI restart
has not been performed as part of it.

## Reproducibility

Saved run (`runs/audits/preliminary1-cable-only-pilot`, not included in this source-only release) contains `protocol.json`,
`windows.json`, `states.npz`, `protected_before.json`, `integrity.json`,
`code_snapshot/`, `diagnostic_source.py`, `forward_checks.json`, `grid.json`,
`evaluations.json`, `optimizer_state.json`, `best_physics.json`, `stopping.json`,
`evaluation.json`, `existing_residual_comparison.json`, `summary.json`, and
prediction arrays. `finish_report.py` generates the summary and figures.

The reusable command is `tools/diagnose_preliminary_cable.py` with a new output
directory and `--stage checks`, followed by `--stage search` only after reviewing
the numerical checks. It refuses to overwrite a prior search. Source snapshots
preserve this execution; no archived research artifacts were restored.
