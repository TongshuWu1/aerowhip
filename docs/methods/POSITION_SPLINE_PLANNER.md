# M0 B-spline planner and firm recovery

Current main setup, 13 September 2026. The user restored the selected
`20260910-022818-648386-M0-development-whip` model and objective, requested
B-spline initialization from that motion, soft strike alignment, and a firm
smooth brake followed by a return to launch.

## Active configuration

- Launch: [0, 0, 1.255] m; target: [1.25, 0, 1.0] m.
- Retained M0 coupled model and `preferred_fold_v1` objective.
- The saved M0 full command initializes the planner. Later speed-gain searches
  and their solutions are not used as seeds.
- Quintic clamped position B-spline: 12 points, 3 fixed at launch,
  9 adjustable XYZ points. P/V/A are analytic derivatives, sampled at 30 Hz.
- Bounded least-squares conversion respects the continuous 60 m/s^3 jerk
  envelope. Smooth correlated position perturbations refine the converted seed.
- Existing adaptive-temperature MPPI scoring, candidate ordering and plateau
  stopping remain. Default: 512 samples, 4 proposals, at least 20 updates,
  plateau patience 12, no fixed update ceiling.

## Objective and angle

The original saved cable-shape preference, contact/reversal guidance and
motion costs remain. This is a preference based on a saved simulation,
not proof of a travelling fold. Cast reward is

`200 * proximity * forward_speed_factor * reach * max(cos(angle), 0)^2`,

with the original tip-first factor at cable contact. All encounter quantities
refer to the same saved event. Straighter forward strikes receive more reward.
There is no angle cutoff and no minimum tip-over-root speed-gain requirement.
The original `tip_contact_v1` outcome uses a 5 cm target sphere; reported
minimum distance remains continuous. The unused historical angle value is
retained for reading old settings but is not shown as a current limit.

## Recovery

Brake from the exit P/V/A to rest, then return on a smooth straight path to
launch and hold. P/V/A are continuous at both handovers. Braking does not
reverse along the exit-velocity direction. Current braking settings allow
8 m/s^2 horizontal acceleration and 60 degrees tilt, subject to the existing
vehicle and jerk envelopes. Return speed is limited to 1 m/s; final hold is 3 s.

A short integration refinement (128 samples, 3 updates) produced
`20260913-000522-283096`, a recovery revision of the same spline plan.
Its independently replayed score agrees within 1e-5. Complete coupled replay
and CSV checks pass: 6.7 s total, 0.567 s brake, 2 s return, 3 s hold.
Predicted minimum tip distance is 0.02693 m. This is development simulation,
not a completed systematic experiment or a physical performance claim.

## Run and inspect

Run `python run_simulation.py`, then MPPI > M0 B-spline setup.
The original M0 and the verified new B-spline candidate are the active entries.
Rehearsals opens the new complete recovery. The CSV is also available at
`exports/M0_Bspline_firm_brake/fullstate_30hz.csv`.
`python tools/plan_fold_strike.py --check` validates the current inputs.
No optimizer or flight sender is launched automatically.

Superseded generated runs and M1/M2 models are marked ARCHIVED, and the PPO
page and old selected M2 profiles are retired. Original measurements and
immutable run/source snapshots remain available on disk.
