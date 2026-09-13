# Travelling-fold strike validation — 12 September 2026

Development on Windows with NVIDIA GeForce RTX 4080. Ubuntu/RTX 5080 has not
been tested. The model is the retained M0 calibration; no physical model refit,
PPO/SAC training, new measurement evaluation, or protected-recording access was
performed. These simulations are development evidence only.

## Checks completed

- Synthetic geometry: moving fold, static fold, rigid swing, reverse propagation,
  shallow bending, disconnected peak jumps, rigid transformations and edge
  subdivision. The fold test is a geometric operational definition.
- Objective: distance/speed co-timing, no credit for backward velocity, increasing
  speed credit above the normalization scale, rejection of distant fast misses,
  future-fold exclusion and preservation of a fast event during a later slow pass.
- Workflow: frozen new-study profile/templates, unchanged M0 model, prohibition
  on exporting a plan without fold confirmation, historical study compatibility,
  and rejection of an unrecoverable exit without modifying flight limits.
- Physical reporting: gap-aware speed estimates use the reviewed free-motion
  interval; derivative windows containing post-contact samples remain unavailable.
- CPU and CUDA eager full-trajectory replay agreed within 1e-8 m in final cable
  position and 1e-6 in objective score. This checks implementation agreement,
  not the accuracy of the physical model.
- CUDA batched search and independent single-candidate replay reproduce the
  selected objective and fold condition. Full recovery is separately required
  before CSV generation.

## Preserved development failures

The first closest-approach-only objective selected a slow pass (0.15 m/s forward
speed at about 6.2 cm). The scored encounter was separated from the minimum-error
reporting event to avoid replacing an energetic strike with a later slow pass.
An intermediate forward-strike candidate reached 5.22 m/s at 1.82 cm but could
not recover inside the existing reference limits. Its rehearsal correctly
produced no CSV. Recovery-reference feasibility now gates incumbent selection.
Neither failure was hidden or fixed by relaxing flight/fold constraints.

## Final integration run

Job `20260912-212707-357664`, retained M0, one target, seed 657,
512 random candidates x 120 iterations (61,440 random candidates; also evaluates
proposal means and incumbent). Search completed in 332.2 s on the development
machine; this is an observed duration, not a controlled hardware benchmark.

- Geometric fold completion: 1.220 s.
- Scored encounter: 1.252084 s.
- Minimum distance / strike-event distance: 1.179 cm.
- Directed tip speed at that event: 7.083 m/s.
- Independent replay score difference: 8.67e-13.
- Complete 30 Hz CSV: 10.9 s, including recovery and hold.
- Full coupled recovery prediction passed the existing envelopes. No limits were
  relaxed. Minimum predicted cable height: 0.2475 m.
- Planner/export prefix agreement: 1.48e-14 m
  for quadrotor position and 4.4e-12 m for cable position.
- CSV SHA-256: `cfce4e432bee184ff13a849f126432c485f720e3205606d0dcb4e7f6f5a53dea`.

The installed deployment checkout passed **88 targeted tests; 1 skipped**
(the unavailable historical cold-seed fixture). The saved new rehearsal also
loaded through the offscreen Qt preview and displayed fold acceptance without
calling it a hit. Shared changed files match between the two working copies,
except their intentionally separate handoffs. No commit or push was performed.

## Limits of this evidence

The geometric detector has not been validated as a physical fold classifier on
new lab recordings. Dominant-bend tracking can be ambiguous for multiple nearby
bends; inspect the complete saved centerline sequence. Simulated fold acceptance
is not confirmation of a fold in a real flight, and tip speed is not measured
impact energy. One model, target and search seed do not establish robustness or
M0-to-M2 improvement. Freeze all settings before starting the new physical chain.
