# Offline model decisions

The canonical baseline is a homogeneous, three-dimensional DER with one
prescribed attachment position and one free distal end. The two identified
parameters are `EI` and `Cb`.

## Retained

- nonlinear isotropic DER bending and objective Kelvin--Voigt bending damping;
- a prescribed-position hinged/swivel attachment: position fixed, tangent and
  material roll free;
- a dynamic distal end with no prescribed terminal force or moment, whose
  measured position is used only as truth;
- measured cable and marker masses, gravity, and hard inextensibility;
- equal weight per clean, non-overlapping one-second Training window;
- explicit measurement-quality rejection and held-out validation; and
- bounded deterministic initialization followed by differentiable rollout
  fitting.

## Torsion reduction

For a naturally straight, circular, bending-isotropic rod, bending energy is
independent of material-frame roll. Quasistatic twist is constant along the rod,
and the free-end zero-torque condition makes that constant zero. The material
twist variables and `GJ` are therefore analytically eliminated. The physical
cable is not assigned `GJ = 0`; `GJ` is absent because every relaxed twist
state has zero torsional energy in this boundary-value problem. Spatial
centerline torsion remains allowed through a Bishop/parallel-transport frame.

This reduction would no longer be sufficient with a torsionally driven or
rigidly welded attachment, a distal rigid body, frictional/contact moments,
significant rotational aerodynamics, intrinsic curvature, cross-section
anisotropy, or constitutive bend--twist coupling. Those effects require new
measurements, including material-director observations if `GJ` is to be
identified; they must not be absorbed into the present `EI`/`Cb` fit.

## Not retained

- ambient drag, because the previous held-out gain was below 1 mm and drag can
  absorb unmodelled forces needed for later contact inference;
- a neural residual or hidden smoothing;
- endpoint compliance without an independently observable attachment model;
- per-window latent initial-state optimization; and
- torsional stiffness or damping in the free-tip centerline model.

## Data and artifact compatibility

New CSVs contain one rigid-body pivot followed by `c1...c10`, with `c10` on
the free end. Training uses only complete one-pivot/free-tip windows; Validation
takes are content-disjoint and never select parameters. The primary held-out
result is attachment-only open-loop prediction, supplemented by periodic
full-cable position observations at fixed, predeclared intervals.

The earlier two-holder `c1...c9` dataset and the
`optitrack_twist_aware_rod_v5` `EI`/`GJ`/`Cb` artifact encode two
prescribed endpoint poses and a different energy. Their fitted values and
reported errors are not results for the free-tip model and cannot initialize or
validate it. A new canonical fit using schema
`optitrack_one_attached_free_rod_v1` must be produced from the new capture
contract.
