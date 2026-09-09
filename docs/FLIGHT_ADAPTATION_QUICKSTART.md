# Flight data, saved replay and adaptation

Read [HANDOFF.md](../HANDOFF.md) for the active model/run and stopped-job state.
This guide describes the current data/replay workflow; it does not launch a vehicle.

1. Associate each recording with its actual vehicle, policy/plan, exact sent
   `fullstate_30hz.csv`, saved rehearsal forecast, model and source hashes.
2. Retain original tracking and command logs, timestamps, validity, gaps and
   intervention/phase information. Do not invent missing commands or markers.
3. Use the desktop's Recordings and Flight comparison pages to inspect measured
   geometry against that exact saved forecast. Do not substitute command positions
   for predicted aircraft motion or regenerate the ghost with another model.
4. Apply the existing recorded normalization once. Hover-normalized Z is a
   retrospective evaluation convention, not proof of physical target contact.
5. Evaluate new flights before any later fit. Fitting and PPO currently remain
   stopped while the physical drone is investigated.

Read [data lifecycle](DATA_LIFECYCLE_AND_RECORDING_GUIDE.md),
[adaptation comparison](ADAPTATION_CHECK.md), [height normalization](HOVER_HEIGHT_CALIBRATION.md),
and [future fitting requirements](FUTURE_ADAPTATION_FITTING.md) for details.

Current planning/export uses [direct PVA](DIRECT_PVA_WORKFLOW.md). Acceleration
is kinematic; do not add/subtract gravity or apply battery scaling to the CSV.
The physical firmware/interface still needs its own verified mapping.

Battery work is proposed: repeated PVA trajectories on a 2S Bolt while logging
voltage, actual motor output, pose and acceleration. No compensation or collection
script has been implemented. Existing recordings lack the required battery/motor
fields in the inspected experiment CSVs. Account for cable forces when interpreting
motion as thrust, and do not confuse the old acceleration-limit drop with battery evidence.

Older force-correction/import command examples are retained in the documentation
archive for reproducing their original experiments, not as current launch instructions.
