> Historical document archived on 9 September 2026. For current work, read [HANDOFF.md](../../../HANDOFF.md). Old running-job and launch instructions below are historical.

# Measured initial state and open-loop maneuver

User-confirmed experiment contract, 2026-09-05: measure the initial drone and
cable state, prepare a maneuver using the calibrated physical simulator, execute
the maneuver open-loop, then return to normal PID control. Open-loop is the
first experimental condition. Feedback during the maneuver is a possible later
extension, not an automatic or implicit change to this condition.

## Measured state and node mapping

`experimental_data/default_processing.json` explicitly names cable:c1 through
cable:c10, ordered from the drone toward the tip. The pipeline stores all ten
marker positions and their validity flags, together with drone rigid-body pose.
The retained recordings are synchronized at 100 Hz.

With the current `config/model.json`, `reconstruct_dder_nodes` in
`experimental_data/force_dataset.py` creates:

| DDER node | Source |
| --- | --- |
| 0 | Drone attachment position, calculated from rigid-body pose and rotated body-frame offset [0, 0, -0.055] m |
| 1 | Interpolated midpoint between the attachment and c1 |
| 2–11 | The ten measured markers c1–c10; c10 is the tip |

The model state contains 12 three-dimensional positions and 12 corresponding
velocities. Ten markers provide an estimated discretized cable state; the
additional midpoint and attachment are reconstructed, not extra measured
cable markers. Drone attitude is used to locate the attachment; it does not
clamp the first cable tangent in the free-pivot model.

The fitter's `_prepare_take` in `experimental_data/cable_fit.py` loads these
positions and velocities, projects them onto the cable length/velocity
constraints, and initializes each fitting window from that state. During cable
fitting, the measured attachment trajectory drives the boundary while EI/Cb
are estimated. During force-controlled deployment, the attachment is dynamic.

## Policy and execution

`PointForceWhipEnvironment.reset(state)` accepts the full `DderState`.
`observe()` constructs 79 normalized inputs:

- 36 node-position components relative to the initial attachment;
- 36 node-velocity components;
- 3 target-position components relative to the initial attachment;
- 3 desired strike-direction components;
- 1 remaining-planning-time component.

`compile_strike_plan` initializes a private model from this state. The current
actor outputs one force action per query; queries on predicted model states
prepare the complete sequence. It is not currently a network that emits the
entire sequence in one forward pass. Nevertheless, the only measured state
input is the initial state: physical execution uses the frozen force sequence
without further policy observations or queries.

The sequence ends at its predicted first valid hit. PID recovery starts at the
predetermined end regardless of the measured outcome. If planning predicts no
valid hit, the maneuver is refused. Logging may continue throughout physical
execution for scoring and later model identification; logging does not make
the maneuver a feedback policy. Low-level stabilization that realizes the
commanded force is distinct from maneuver-level feedback.

## Live integration and training work still needed

1. Build the live synchronized marker/attachment state adapter using the same
   marker identities, coordinates, attachment offset, node topology and units.
   Timestamp every state estimate and check marker validity before planning.
2. Estimate velocities causally from a short pre-maneuver observation buffer.
   The offline derivative is an 11-sample centered cubic fit: at 100 Hz it uses
   50 ms of future samples relative to its center. It cannot directly estimate
   the latest live frame without latency. Use a past-only estimate, or explicitly
   compensate a delayed estimate to the planned release time.
3. Align positions and velocities to one timestamp and account for planning
   delay while PID is holding the system. Apply consistent state projection
   and normalization across fitting, training and live initialization.
4. Cover the intended release-state distribution in training. The current
   `sample_batch` uses hanging/straight tilted cable states with small motion
   perturbations; it does not yet sample general measured bent cable shapes.
   Any initial-state bank must preserve the preliminary fitting/validation/test
   split and match the states expected at maneuver release.
5. Keep open-loop and any later feedback-enabled runs explicitly labeled, with
   sensor access and controller changes reported separately in the paper.

The retained `fig8_001` arrays were read to check the mapping: its measured
marker array has shape (4285, 10, 3), and its reconstructed node position and
velocity arrays have shape (4285, 12, 3). Nodes 2–11 match the measured markers;
node 1 matches the attachment-to-c1 midpoint. This read-only check created no
new checkpoint or experiment-result artifacts.
