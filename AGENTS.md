# Simulator branch - complete application, no historical data

Read HANDOFF.md first. The user explicitly wants the WHOLE application/UI and
all training, policy, model, fitting, rehearsal/export, recording and diagnostic
code. Do not reduce this to a standalone physics viewer or remove app features.

Latest cable material is user-confirmed braided nylon paracord. Standalone
default is `config/isaac_physx/rig_153g_paracord.json`, 32 links at 1,200 Hz.
Read `docs/PARACORD_MODEL.md`: separate implicit bend/twist drives, exact
marker mass COM/inertia (under estimated spherical-marker assumptions), and
distributed air loads at physical COM. Do not re-add explicit spring torques
in this mode. The old config retains legacy behavior. Native Newton and
48-link prototypes failed checks and are not active options. Do not promote
them, claim hardware calibration, or call fast-whip spatial convergence proven.

This is a fresh source-only branch. Do not import collected real-flight data,
old fitted M0/M1, NN weights, trained policy checkpoints, or historical outputs.
Keep model/policy/data libraries empty until new simulator experiments produce
artifacts. The nominal example profile is unfitted and must not be called M0.

The separate independent Isaac Lab/PhysX runner is implemented in
`isaac_simulation/`, launched by `launch_isaac_simulation.py`. Read
`docs/ISAAC_PHYSX_RUNNER.md`. User selected the 153 g experimental drone with an
18 g cable. Its engineering-estimate plant config is separate from the model
to be fitted; never silently import old fitted parameters. Preserve this
separate process rather than merging it into the research Qt UI.

Use D6 ball joints with locked translation and passive implicit angular damping;
USD SphericalJoint ignores drive damping, and the earlier explicit-damping
prototype was unstable on longer flights. Keep the top pivot free. Do not
advance `app.update()` alone while a take is paused or capturing screenshots:
it steps physics without the controller. Use `sim.render()`.

Next: synthetic command playback / measurement adapter, then new preliminary
data collection and NEW M0 identification. Existing Isaac Lab training hosting
uses our external model; it is not the independent PhysX plant. Keep hidden
plant parameters separate from the model being fitted. Preserve the 30 Hz
FullState command contract and plan for 100 Hz synthetic measurements.

Do not launch fits or training without user authorization. Future fits should
stop on practical convergence/plateau and use essential validation. The inherited
five-take adaptation runner needs preliminary/synthetic-data integration before
M0 fitting. Keep existing code available, but do not claim this is already done.
