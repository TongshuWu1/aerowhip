# First-contact PPO acquisition pilot

Run: `runs/ppo/20260905-181856-007834-seed651`.
Target: 20,480 attempted episodes, batch 512, seed 651, CUDA.
This is an exploratory pilot, not the final paper experiment.

## Applied changes

- Baseline `20260905-181108-847814` applies the reviewed geometry/drag candidate:
  twelve substeps, external cable drag .3/s, calibrated lateral attachment
  offset, EI 2.83994e-8 and Cb 3.75231e-5. Neural residual is off.
- The first contact must qualify. Invalid tip contact or non-tip-first contact
  terminates nominal planning/scoring; the compiler refuses that plan. Live
  scoring also cannot award success after an earlier invalid contact. Executed
  forces still follow their frozen cutoff, without hit feedback.
- PPO can learn X, Y and Z force outputs. The refinement rollback guard is off;
  clipping, KL stopping, gradient clipping and deterministic validation remain.
- Initial conditions include small, independently perturbed segment directions,
  with exact segment lengths and rigid angular velocities consistent with those
  lengths. This adds mild bent-cable coverage near hover, not arbitrary shapes
  or independent deformation-velocity modes.
- Nominal fraction remains .25. Other tolerances remain provisional. New bend
  range is 1 degree per segment, with .2-degree shape estimation perturbations.
- Live preparation drift thresholds are .002 m and .01 m/s; these replace the
  substantially looser .04 m and .15 m/s acceptance thresholds for new settings.
- Physics is float64 on CUDA, while observations and neural networks remain
  float32. Under a one-second common 3D force sequence, float32 GPU positions
  differed from CPU by up to 1.785 mm; float64 differed by 9.07e-14 m. This short
  agreement test does not replace validation of a learned seven-second strike.
- Reward weights remain unchanged, including the low 1 point/second time cost.
- Training and validation now distinguish nonfinite states, position-limit
  violations and speed-limit violations. These flags may overlap.
- Final holdout evaluation is disabled for this pilot. A future locked paper
  protocol must explicitly enable it with reserved scenarios.

Old snapshots without `first_contact_only` or `physics_dtype` retain their old
semantics. The shared task/initial-condition changes also affect new SAC runs;
the PPO rollback setting only affects PPO. Existing runs retain their snapshots.

## UI

The reward page explains the first-contact rule, including that a zero invalid
contact penalty does not allow a retry. PPO identifies acquisition mode, active
axes and saved-model provenance, defaults to this pilot budget, and displays
failure categories after each training batch. Externally started runs are
automatically selected and their logs/Stop button remain available.

## Review before expanding the budget

Check first-contact success and hit+recovery rate, planning refusals, failure
causes, return, exploration and wall time. Review actual first-trial recordings.
If progress stalls, diagnose it before increasing episode count or changing
multiple reward weights. Keep pilot selection separate from the final holdout.
The million-attempt run is not launched by this change.
