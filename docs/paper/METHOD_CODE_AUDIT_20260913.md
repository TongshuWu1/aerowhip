# Problem Statement and methodology audit

Reviewed the live Dropbox manuscript, lines 156–912, against current implementation and the selected M0/M1/M2 study. No manuscript, model, objective or measurement changes were made. The user intends to recollect M2 on Monday; existing recordings remain preserved. The suspected collection problem is not established by this audit.

## Confirmed correspondence

| Manuscript component | Implementation | Assessment |
|---|---|---|
| Material-coordinate centerline, simulation nodes and shape samples | `planning/whip_objective.py::material_resample`, `simulator/cable/dder.py` | Consistent. Nonuniform simulation spacing and uniform material shape samples have distinct roles. |
| Curvature binormal and isotropic bending energy | `simulator/cable/dder.py` | Correct for the stated reduced model away from antiparallel singularities; numerical denominator floors are disclosed. |
| Regularized curvature rate and Rayleigh damping | `simulator/cable/dder.py::_curvature_rate_jacobian_impl` | Formula matches the implemented rate; direct float64 checks include straight and nearly straight configurations. |
| Delayed translational response and nominal-acceleration attitude drive | `simulator/research_pose.py::tensor_derivatives`, `simulator/drone_pose_response.py` | Consistent. Vehicle residual is added to translation after constructing the attitude drive. |
| Tracking-frame alignment and rotated attachment offset | `simulator/research_pose.py`, `simulator/drone_pose_response.py` | Consistent. Root motion is prescribed from vehicle motion; no explicit cable reaction feedback enters the vehicle model. |
| Vehicle/cable network features and bounded outputs | `simulator/drone_pose_residual.py`, `simulator/cable/residual.py` | Matches the selected acceleration-only residual models. Broader optional modes in the code are not required in the paper. |
| Quintic B-spline, first three equal controls, sampled PVA | `planning/position_spline.py` | Consistent. Jerk constrains the derived spline; the optimization variables are position control points. |
| Encounter, target, shape and extension scores | `planning/whip_objective.py`, `planning/mppi_trajectory.py` | Matches the active preferred-shape objective. It does not impose a separate travelling-fold constraint. |
| Correlated proposals and exponential score weighting | `planning/position_spline.py::RetainedM0SplineProposals`, `planning/mppi_trajectory.py` | Consistent with an MPPI-inspired offline search. Fixed seed offsets do not change the weighted-mean expression. |
| Fixed-reference correction objective | `planning/reference_correction.py::tracking_cost` | Exactly mean squared 3D tip, vehicle and command-position deviations with separate sample grids. |
| Local command solver | `planning/local_reference_correction.py` | Bounded, damped Gauss–Newton with finite-difference sensitivities, backtracking and nonlinear feasibility checks. The original M0 reference stays fixed; the most recently executed command initializes correction. |
| Staged fitting, robust data loss and neural penalties | `experimental_data/whip_full_optim.py`, `experimental_data/whip_full_cable.py`, `experimental_data/whip_bounded_fit.py` | Consistent with the active equal magnitude/change regularization weights. Cable fitting is driven by measured attachment motion; complete command-to-tip error is evaluated separately. |

## Recommended manuscript clarifications

1. **Yaw, lines 173–175 versus 584–587:** say position, velocity and acceleration commands with fixed yaw. The current planner does not optimize a yaw trajectory; the general predictor can accept yaw inputs.
2. **Source of the two references, lines 658–661 and 752–757:** make explicit that the earlier shape history is a design prior, while the initial M0 forecast is the fixed physical-motion reference for later correction. Do not collapse these into one reference or imply an externally observed desired trajectory.
3. **Meaning of whipping, lines 208–214:** keep the travelling bend as a description of the schematic and physical motivation. The implemented shape-history reward encourages the selected motion; it is not a propagation detector or guarantee. The current text mostly respects this distinction.
4. **Fitted bias, lines 405–406 and 558–560:** identify the constant bias as effective hover compensation estimated from pre-maneuver measurements, rather than an unexplained physical force. Preserve the separate zero-bias nominal planning initialization.
5. **Candidate fitting versus deployment selection, lines 799–804 and 900–905:** say the four-stage procedure produces candidate updates, followed by selection of the model used for command correction. The selected M2 retained the M1 vehicle residual. Specific candidate scores and selection details belong in Experiments.
6. **Marker mass, lines 162–164 and 474:** uniform cable material does not imply uniform node masses. A short statement that node masses include cable and marker contributions would make the experimental model clearer.
7. **Compact presentation:** keep the physical dynamics, attachment mapping, primary score, shape score, correction cost and fitting losses. Implementation constants, derivative-probe rules, optimizer budgets and solver backend details need not be added. The expanded regularized damping expression is accurate but lengthy; shorten its exposition only if the implemented rate remains unambiguously specified.

The continuous constraint-force equation is an appropriate theoretical representation of the projected numerical dynamics; it need not be replaced by a step-by-step solver description. The current initialization distinction between nominal planning and measurement-initialized postflight prediction should also remain.

## Validation

65 targeted tests passed, with one warning, on Windows. The manuscript bending-rate formula and implemented operator differed by at most 5.33e-15 on four deterministic CPU float64 states; the damping-power identity differed by at most 8.89e-16. No new model fitting, MPPI campaign or physical evaluation was run. These checks do not establish physical accuracy or convergence guarantees.

Exact source hashes, manuscript hash and the test command are recorded in `METHOD_CODE_AUDIT_20260913.json`.
