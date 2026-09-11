# Braided nylon paracord upgrade — latest

PAUSED by user on 9 September 2026. Active work returns to the real-flight
checkout `../particle_filter_cable_project`, branch `twin-rewrite`. Preserve
this prototype; do not resume simulator development or run more experiments
unless requested. Next priority is one-drone real-flight M0/M1 collection
after investigating battery compensation.

User confirmed braided nylon paracord; no diameter measurement supplied.
Standalone default is now `config/isaac_physx/rig_153g_paracord.json`: 153 g
drone, 18 g cable including markers, 0.9525 m length, 32 segments, 1,200 Hz
PhysX. Existing 300 Hz controller / 30 Hz commands / 100 Hz logging remain.
See `docs/PARACORD_MODEL.md` for assumptions, sources, rejected prototypes
and essential numerical checks. No fitting, training or M0 collection started.

Changes: distinct implicit bending/twist stiffness and damping, marker COM
and inertia at actual distances, marker sphere collisions, physical contact
material, distributed translation/rotation-dependent cord drag plus marker
drag. Force application explicitly uses COM because tensor API's default
is link origin. Startup verifies authored COM/inertia as well as total masses.
Twelve CPU math checks pass. Ten-second hover force balance passes. Enhanced
600 Hz circle and figure-eight passed 45-second, three-loop checks. Matched
1,200/2,400 Hz figure-eight tests showed 0.283 mm drone and 8.610 mm cable-tip
RMS difference during motion/final hold; tip peak difference 23.170 mm.
This is numerical sensitivity, not hardware error or full whip convergence.
Final rendered default run `runs/isaac_physx/paracord_final11` completed 45 s /
three figure-eight loops: motion tracking RMS 0.766 cm, peak 1.664 cm, no motor
saturation. Screenshot inspected; normal exit confirmed. Wall time 147.8 s.

The native Newton VBD prototype had unacceptable coupled hover drift and
was rejected, as was the unstable 48-link PhysX refinement. Their debug
sources/results remain under ignored runs; they are not active launcher
choices. Newton 1.5.1 was added only to env_isaaclab, with existing Warp1.16
unchanged; Isaac's bundled Warp1.8 remains isolated. Working PhysX needs no
Newton import. Do not restore unvalidated experimental viewer code.

Original `rig_153g.json` preserves the prior baseline. Material stiffness,
diameter, marker mass split and size are explicit estimates, not calibrated
nylon properties. Axial stretch, braid hysteresis, downwash and a measured
attachment-clamp law remain absent. Original twin-rewrite checkout is clean.

# Independent PhysX drone/cable runner — earlier baseline

Latest task: build a separate Isaac Lab process, realistic-looking drone and
passive PhysX cable, able to fly a circle or figure-eight. User explicitly chose
the 153 g experimental drone, with the existing 18 g cable. Implemented in
`isaac_simulation/`, with `run_isaac_simulation.py` (Isaac interpreter) and
`launch_isaac_simulation.py` (ordinary Python subprocess launcher), plus two
PyCharm run configurations. See `docs/ISAAC_PHYSX_RUNNER.md`.

Official Crazyflie visuals; own mass/inertia; bounded four-motor actuation and
attitude/PVA controller; 32-link D6 articulation with passive angular damping,
elastic bending, drag, self-collision, floor and two-way reaction. Root API is
on the drone body. Tracking frame and rotated lever arm are explicit. Physics
600 Hz, controller 300 Hz, commands 30 Hz, measurements 100 Hz. CPU PhysX by
default (measured substantially faster for this single articulation), RTX
rendering; CUDA PhysX remains selectable. No real-time wall-speed guarantee.

All plant parameters beyond the known mass/geometry conventions are engineering
estimates, not a calibrated aircraft or firmware match. No old data/weights
imported. No M0, NN fit, policy training or real flight started. Logs are native
synthetic CSVs, not yet input-compatible Motive/controller recordings; playback
and the recording adapter remain the next research integration work.

Essential checks found and fixed automatic cable-root selection, uncontrolled
physics during screenshot/pause, and the installed Lab STOP render-loop hang.
Short flights originally passed but default-duration circle exposed explicit
cable damping instability. USD spherical joints also silently ignored drive
damping. Final joints use the supported D6 ball-joint representation and
implicit passive damping. A 45-second, three-loop circle completed after this
fix (motion RMS 1.018 cm, peak 1.803 cm). Final figure-eight/rendering results
and launch instructions are recorded in the runner guide. Eight math checks
pass; native Kit STOP was exercised and saved/closed correctly.

Generated tests and source snapshots are ignored under `runs/isaac_physx`.
Earlier probes are debug artifacts, including failed prototypes; do not
present all of them as successful validations.

# Complete application port for simulator experiments (earlier)

Latest user correction: retain the entire UI and all training/policy/model code.
The earlier reduced single-scene UI was discarded and the full application was
restored. The seven original pages and their subpanels remain available.

Source: twin-rewrite commit 19dc9a2694f4edafc0ebf94482139ca16e334b2c.
That original checkout and research artifacts are untouched. Simulator has fresh
root history and a separate checkout.

No original recordings, processed data, learned policies/checkpoints, fitted
models/residual weights, or historical run outputs were copied. Startup uses
an explicitly unfitted nominal profile; both NNs are disabled. Small changes
allow nominal-only asset loading, prediction, and snapshots without fake weights.
The model page identifies this profile as unfitted, not an already collected M0.

Next requested direction: independent Isaac Lab/PhysX plant, preliminary data
collection, then new M0 identification. No PhysX plant was built, no preliminary
data collected, and no fitting or training started during the branch port.
The existing external-model Isaac host and five-take adaptation runner remain
code foundations, not completed independent-plant/preliminary-fit workflows.

Verification: 37 focused physics/geometry/FullState/nominal-only checks passed.
All seven UI pages and their sub-tabs opened with Qt offscreen; empty libraries
and disabled continuation/fit-without-data controls verified. Original checkout
remains clean. This is not an Isaac/PhysX benchmark or a full historical suite.
