# Travelling-fold strike objective

**Current implementation:** see [Position spline planner](POSITION_SPLINE_PLANNER.md).
New runs initialize from rest with random smooth position proposals and no
saved command templates. The latest profile uses intensity weight 8 and a maximum 30-degree tip-velocity
deviation at the scored event. The formulas and mandatory-fold/jerk details
below describe earlier development versions, not the current experiment profile.

**Latest decision:** new runs set `fold_requirement: diagnostic_only`. The detector
below is retained for diagnosis, not as a score/acceptance gate. Historical runs
without this field preserve the original requirement. The physical task is a
close, fast forward tip encounter under the saved feasibility constraints.


**Superseded implementation details:** new main runs use [position splines and brake-return-hold recovery](POSITION_SPLINE_PLANNER.md). The old `20260912-212707-357664` candidate failed the later predicted-attitude audit; its numbers below are historical development diagnostics, not feasible-flight evidence.

The user requested replacement of the active MPPI reward for the upcoming
systematic experiment on 12 September 2026. Every accepted new plan must contain
a travelling fold. The calibrated M0 model and historical recordings, commands,
objectives and forecasts remain unchanged. New M0/M1/M2 commands use the new
objective. This supersedes reuse of the historical M0 command for new studies.

## Task and score

The planner searches a fixed 1.5 s, 30 Hz jerk sequence in the existing coupled
quadrotor/DDER model. It runs the complete interval; simulated sphere entry does
not truncate the new maneuver. The new settings contain no legacy reward block.

For a candidate encounter, let d be tip-to-target distance and v+ the positive
component of tip velocity along the prescribed world-frame strike direction.
Set e=(d/0.10 m)^2 and z=(v+/4 m/s)^2. The event score is

    -e + 4 exp(-e) z/(1+z).

The trajectory score is its best eligible event score minus 0.01 times the mean
squared normalized jerk. Both distance and speed belong to the same event.
Events are sampled at each physics interval's endpoints and linearly interpolated
closest point. This is a discretized maximization, not an exact continuous-time
optimal event solver. Only forward-moving events after fold completion qualify.

The 0.10 m and 4 m/s values are normalization scales, not success thresholds.
The speed term continues increasing above 4 m/s. Its bounded value prevents
unbounded speed credit; its spatial factor suppresses credit for distant motion.
Weight 4 expresses the accuracy/intensity tradeoff. These are engineering design
choices, not parameters identified from physical impact data.

For new studies, physical evaluation also reports directed tip speed at the
observed closest point when it can be estimated from five contiguous samples
inside the reviewed free-motion interval. It uses a centered quadratic derivative;
missing or contact-contaminated windows yield no speed estimate. The speed at
closest approach can differ from the planner's selected-event speed. Physical
fold confirmation remains a review of recorded geometry/video.

The reporting metric remains minimum 3D distance across the entire 0–1.5 s
window. It can occur at a different time from the scored strike. Report the
strike distance and directed speed together; do not pair speed from one instant
with a smaller distance from another. Squared directed speed is a kinetic
intensity proxy. Contact force, impulse and energy transferred to a target are
not modeled or measured by this objective.

## Required travelling fold

The cable is resampled on 21 uniform **rest-material** coordinates. A fold track
starts in the proximal 40% when segment tangents span at least 90 degrees and
the dominant local bend turns by at least 15 degrees. Local turning compares
secants spanning 10% of cable length on either side. The dominant bend must then
remain visible and move to at least 75% of the cable length. The saved bounds
allow at most 5% backward or 20% forward movement per physics observation, with
at least four observations. A disconnected jump resets tracking. Fold completion
must precede the scored encounter; a later fold cannot qualify an earlier event.

This geometric test rejects rigid swings and static folds. It is an operational
definition of a travelling fold, not a proof of wave-energy transport. Review
`fold_diagnostics.npz` and the predicted centerline sequence before freezing a
physical study. Its thresholds and sampling resolution must remain fixed across
model generations. No claim of a physical fold follows from simulated acceptance;
retain video/marker evidence of what the real cable does, including failures.

## Planning and experiment workflow

The dedicated optimizer uses the existing retimed command-template sampler and
MPPI weighting, with four proposal groups, 512 random candidates per iteration
and a fixed 120-iteration budget. Templates initialize commands; their predicted
cable shapes never enter the score. All generations use the same templates,
seed, objective, constraints and budget. Physics/fold failures and known recovery failures receive no weight. Before
promoting a higher-scoring candidate, the existing recovery planner must find a
reference inside the saved command and altitude limits. These checks are lazy:
lower-scoring proposals are not all replayed through recovery. The complete
coupled recovery prediction is checked during CSV rehearsal.
If none qualify, planning fails and publishes no plan. A selected plan must
reproduce its score and fold condition in an independent single-candidate replay.

Create a **new study** through `run_lab.py`, or run:

    python tools/lab.py --study fold-study create
    python tools/lab.py --study fold-study plan --generation M0
    python tools/lab.py --study fold-study export --generation M0

Creation requires the existing private lab baseline import. It freezes
`config/pva/systematic_strike.json` and the command templates inside the study.
M0 retains its physical model but requires a new plan. All five repetitions use
the same frozen exported command. Existing studies retain their original task.
Neither creating nor opening a study starts an optimizer or physical flight.

For development in a research checkout without an imported lab baseline:

    python tools/plan_fold_strike.py --model PATH/model.json --templates PATH/proposal_baselines.npz
    python tools/rehearse_pva.py --job runs/mppi_pva/JOB --output runs/rehearsals_pva/NEW_NAME

Rehearsal repeats the fold check and validates the complete recovery before
writing a CSV. It reports fold validity and continuous strike metrics, never
mislabels fold acceptance as a target hit. CSV export remains offline; the
colleague's flight program is unchanged. Outputs stay inside the repository.

## Validation status

This is a new experimental method. Simulation development checks and their
limitations are recorded in `TARGETED_FOLD_STRIKE_VALIDATION.md`. Do not treat
them as physical evidence or evidence that M2 improves over M0. Keep final
recordings outside all objective, model and optimizer tuning.

## Main checkout entry point

Main defaults to its retained M0 model and command templates. Run
`python tools/plan_fold_strike.py --check` to verify paths without optimizing,
then `python tools/plan_fold_strike.py` to create a new offline plan.
`--model` and `--templates` remain available as explicit overrides.
The saved example and CSV are in
`runs/rehearsals_pva/20260912-212707-357664/`; all five development search jobs,
including unsuccessful/intermediate jobs, are in `runs/mppi_pva/`.
