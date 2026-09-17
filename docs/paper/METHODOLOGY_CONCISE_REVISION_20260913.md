# Methodology notation and presentation review

The live manuscript was shortened without changing its section structure or
numbered equations. The methodology decreased from 2798 to 2304 whitespace-delimited
LaTeX tokens (17.7%). This measure includes math and is not a prose word count.

## Applied changes

- Clarified fixed yaw, hover compensation, tracking-frame alignment, cable and
  marker mass contributions, and network-specific acceleration bounds.
- Distinguished the initial simulated shape prior from the fixed M0 trajectory
  and command reference used by subsequent correction.
- Explained that staged fitting produces candidates and that deployment may
  retain an inherited component. Kept measured-attachment cable fitting distinct
  from complete command-to-tip evaluation.
- Removed the effective-sample-count expression and proposal bookkeeping;
  shortened recovery, solver termination, and repeated data-processing prose.
- Moved the absence of an active M0 cable residual and the resulting first-update
  capacity change to experimental setup. The physical results remain unchanged;
  old M2 flight data remain excluded pending recollection.

## Equations and notation retained deliberately

The attachment mapping requires both attitude and tracking-frame alignment.
The continuous command spline and held commands have distinct roles. Likewise,
material shape samples and simulation nodes need separate indices. These were
not merged merely because they both describe positions.

The bending-rate definition remains explicit: replacing it with an unspecified
curvature derivative would hide the actual damping model. The directed-motion
and shape scores remain because they define what initial whip design rewards.
The command-correction objective keeps separate tip, vehicle, and command terms;
its reference and time grid remain fixed. Neither shape matching nor command
correction is described as a guarantee of travelling-fold propagation.

## Verification and limits

All numbered equations are textually unchanged, all section headings preserved,
and labels are unique. Every numerical source hash in METHOD_CODE_AUDIT_20260913.json
still matches the code. The earlier 65-test and damping-operator checks therefore
refer to the same implementation; they were not rerun for these prose edits.
Focused current reads confirmed hover initialization, residual inputs and bounds,
and the correction objective. No models, objectives, calibration, recordings,
or forecasts were modified.

The seven-page PDF compiled using Tectonic with the local OT1 QA wrapper. Pages
2-7 were visually checked; no overfull boxes or unresolved references were found.
Existing class, missing-author, font-substitution and two underfull-paragraph
warnings remain. The live preamble and bibliography are unchanged. Local Dropbox
publication does not verify Overleaf cloud synchronization or establish that the
paper is submission-ready.
