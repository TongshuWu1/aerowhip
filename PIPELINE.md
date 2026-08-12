# Cable Twin pipeline

The repository has three layers:

- `cable_twin/shared`: PIDNet, cable-route extraction, constrained rod dynamics,
  and schemas.
- `cable_twin/offline`: image-plane recording, 2D extraction, and material fitting.
- `cable_twin/online`: the existing RGB-D particle filter using the shared fitted dynamics.

Offline and online use the same cable equations. Only their observations differ.

## Phase 1: planar identification of one cable

```text
one or more lossless HD1080@30 SVO2 recordings
    fixed level camera; approximately image-parallel cable motion
    slow shapes + faster endpoint changes + stationary settling periods
                       |
                       v
left RGB -> canonical PIDNet -> body skeleton + two endpoint components
                       |
                       v
one ordered endpoint-to-endpoint route
    one recording-level scale from known cable length
    pixels -> metric image-plane (x,z) curve
                       |
                       v
24 ordered PIDNet observation nodes
    no position filter; three-frame quadratic supplies velocity
                       |
                       v
differentiable multi-frame rollouts
    exact length belongs only to the physical rod state
    compare every prediction directly with PIDNet observations
    fit one homogeneous EI and one homogeneous Cb
                       |
                       +--> cable model JSON
                       +--> fit rollout residual
```

No calibration target or wall is required. The camera remains fixed and level,
and the cable is held approximately in a plane parallel to the image sensor at
roughly constant depth. Out-of-plane motion creates foreshortening and is outside
this Phase 1 observation model. No stereo matching, ZED depth, or point-cloud
reconstruction is used for offline identification.

Capture records the untouched lossless SVO2. The live window displays RGB only;
PIDNet and 2D trajectory extraction run afterward so display or inference cannot reduce
the recorded frame rate. Manual exposure and gain can be tuned from the offline
UI before capture; the SDK readback is recorded with the SVO metadata.

The known cable length and median complete PIDNet route length define one
pixel-to-metre scale for each recording. The same scale maps every frame to
horizontal-right/vertical-up coordinates; individual frames are never
renormalized. A 2D observation is accepted only when PIDNet supplies two endpoint
components and one connected skeleton route between them. The route is resampled
by arc length into 24 observation nodes. Their positions are not temporally
filtered or projected onto the physical model. A three-frame local quadratic is
used only to estimate initial velocity. At the start of each rollout, the first
observation initializes one feasible exact-length rod state; all later rod
predictions are compared directly with the PIDNet nodes.

## Shared cable model

The cable has `N=24` ordered points and `N-1` capsule segments. Its state is the
point positions and velocities. Segment lengths are hard constraints:

```math
\|x_{i+1}-x_i\| = L/(N-1).
```

The straight, homogeneous, isotropic, torsion-free rod has bending energy and
Rayleigh dissipation

```math
U_b = \frac{EI}{2}\sum_i \frac{\|\kappa_i\|^2}{\ell_i},
\qquad
R_b = \frac{C_b}{2}\sum_i \frac{\|\dot\kappa_i\|^2}{\ell_i}.
```

`EI` has units `N m^2`; `Cb` has units `N m^2 s`. The damping force is
`-dR_b/dv`, so it damps deformation without damping rigid translation. The
previous global velocity-decay parameter is not part of this model.

Mechanically, this is an inextensible spring-joint chain. Its joint energy uses
the nonlinear discrete-elastic-rod curvature binormal rather than a small-angle
angular-spring approximation. It is differentiable and torsion-free, but it is
not the full material-frame, twist-aware DDER formulation. The precise model
and claim boundaries are documented in `cable_twin/offline/CABLE_MODEL.md`.

Gravity, bending, and curvature-rate damping are integrated explicitly with
substeps. A coupled mass-weighted RATTLE-style solve enforces all segment
lengths and prescribed endpoint positions and velocities. Constraint reactions
are axial; the present implementation does not report force or tension from its
multipliers. All operations are differentiable and batched on CUDA.

The offline fit uses short multi-frame rollouts rather than noisy frame-to-frame
accelerations. Every valid window from every selected recording contributes to
one joint `EI`, `Cb` fit. Slowly changing shapes provide stiffness information;
faster endpoint changes followed by stationary endpoints provide damping
information. Bounded log parameters are optimized using full-batch L-BFGS with
a strong-Wolfe line search, and the finite pair with the lowest rollout loss is
retained.
PIDNet is the Phase 1 measurement reference, not a claim of independent
physical ground truth.

## Internal model check

`run_model_check.py` generates an exactly inextensible trajectory using known
`EI` and `Cb`, then calls the production rollout fitter and saves only the true
and recovered parameters and rollout RMSE. It isolates the dynamics and
optimizer; it does not replace validation using real cable motion.

## Online particle filter

The current online application retains its registered-depth 3D observation,
but loads the new planar-identified model artifact. Isotropy makes the planar
bending parameters valid in 3D. Each particle contains 3D ordered point
positions and velocities and uses the shared constrained rod as its transition.

The online 3D observation path remains separate from offline identification and
must be validated after a new model is fitted. Contact is not yet implemented.

## Next phase: cable-object contact

Each cable segment will be treated as a capsule with the measured cable radius.
The cube will be a rigid body with known mass and inertia and an analytic
oriented-box signed-distance query. Nonpenetration will be a unilateral
constraint solved together with cable length and cube dynamics. Normal contact
comes first; Coulomb friction follows after contact behavior is validated.
