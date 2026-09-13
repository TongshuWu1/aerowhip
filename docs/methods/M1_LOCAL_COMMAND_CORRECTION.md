# M1 correction of the original M0 command

Decision: 13 September 2026. The user confirmed that fitting is unchanged.
Physical recordings identify M1; they do not enter the command objective as a
measured-trajectory offset. The original M0 MPPI motion is not replanned.

`tools/correct_m1_local_reference.py` loads the finalized M1 and the immutable
reference in `runs/reference_tracking/M0-paper-fixed-reference`. It initializes
from the original executed M0 B-spline controls. The predicted physical tip and
tracked-origin trajectories are compared at the original timestamps through
34/30 s. The command remains a quintic B-spline on the original 1.5 s domain.

The objective is unchanged from the earlier sampling correction:

    mean ||M1 tip - original M0 tip reference||^2
  + 0.1 mean ||M1 quadrotor - original M0 quadrotor reference||^2
  + 0.01 mean ||corrected command position - original command position||^2.

Each term uses squared 3D Euclidean distances in m^2. Quadrotor tracking is a
soft preference; it is not a fixed maximum path-deviation guarantee. The
physical target remains (1.25, 0, 1.25) m and launch remains (0, 0, 1.4) m.
Original planned strike time remains 1.1172482457473654 s.

## Numerical update

Only eight of the nine free XYZ controls affect the executed prefix. The last
control remains fixed; all original continuous jerk constraints, including
those on the unused spline suffix, remain enforced. Twenty-four coordinates
are conditioned using the spline's position, velocity, acceleration and jerk
maps. Conditioning scales do not change physical limits or objective weights.

Central differences estimate the complete command-to-tracking-residual
Jacobian. Every update compares estimates at h and h/2. Relative Frobenius
disagreement must be <= 5%; h can be reduced by four up to three times. Invalid
model probes are explicitly detected; one-sided differences, if necessary,
are recorded. All accepted updates undergo actual nonlinear rollout checks.

A regularized Gauss-Newton subproblem uses bounded coordinate increments,
unchanged continuous jerk bounds, and exact command-envelope inequalities.
Nonlinear model-domain and complete recovery checks remain outside this local
approximation. Backtracking accepts only an actual objective reduction with
valid command and model rollouts. Settings are frozen before each job: at most
12 updates, damping 1e-4, initial coordinate radius 0.5, maximum radius 1.0,
and a 0.1% small-improvement stopping rule. These are numerical settings, not
physical task-success definitions.

The final command is checked using an independent single-candidate rollout
and the standard production predictor. Full recovery retains the original
brake/return/hold settings, including a minimum one-second brake. Its duration
can increase to satisfy the unchanged envelope. Checks do not certify
cable-to-propeller clearance or future physical tracking accuracy.

## Evidence and next executions

Each job saves settings, input hashes, baseline prediction, finite-difference
checks, accepted-update history, optimized controls, and final replay results.
The optimization's rollout counter counts prefix predictions; full recovery
checks and final production verification are additional computations. The
reported total wall time includes the final production verification.
The earlier sampling-based CSV remains available as historical evidence.
No physical M1 result is inferred from simulation improvement.

For subsequent physical comparison use the same frozen M0 reference and timing:
report tip and quadrotor RMSE over [0, 34/30] s, target error at the original
planned strike time, and closest target error over the existing [0, 1.5] s
interval. Preserve every raw take and its exact executed CSV. Timing comes
from recorded streams; do not time-warp or spatially normalize the comparison.
