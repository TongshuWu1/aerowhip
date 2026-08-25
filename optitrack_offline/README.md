# OptiTrack offline cable identification

Run:

```powershell
.\.venv\Scripts\python.exe run_offline_fitting.py
```

This experiment identifies a cable driven at one attachment point and predicts
the motion of its mechanically free distal end. Fitting and validation use the
same one-attachment boundary condition.

## Required Motive CSV

Each new take must contain one attachment rigid body and ten ordered cable
markers, `cable1:c1` through `cable1:c10`. Put the rigid-body pivot at the
physical cable attachment, place `c1` nearest that pivot, and place `c10` at
the free distal end. Do not hold the distal end or attach a second tracking body
to it. Export the untouched 100 Hz Motive CSV in metres and world coordinates,
including marker and rigid-body quality data and the complete header.

The model's eleven observed material sites are **attachment pivot**, then `c1`
through `c10`. The rigid body's pivot position is used; its orientation is not
a boundary input. Older takes with two held endpoint rigid bodies and
`c1...c9` represent a different mechanical experiment. They may be retained
for provenance, but they cannot train or validate this model.

## Workflow

1. Assign dynamic identification takes to **Training**.
2. Assign independent motions, preferably recorded in a separate session, to
   **Validation**. Do not select parameters after inspecting their errors.
3. Enter the measured segment lengths, cable mass, marker masses, diameter, and
   DER node count; then save the cable description.
4. Run **Check & select windows**. Bad transitions split a take, and a take with
   no complete usable window is excluded.
5. Fit the physical model. A safe cancellation never overwrites the last
   completed artifact.
6. Select one held-out take and run the attachment-only and periodic-observation
   validation protocols.

No endpoint-frame calibration or zero-twist calibration take is required.
Every clean, non-overlapping Training window has equal weight. Validation data
never enters initialization, optimization, model selection, or parameter
bounds.

## Reduced physical model

The default model is a 21-node, three-dimensional, inextensible discrete
elastic rod observed at eleven sites. The measured attachment position
`x0(t)` is prescribed. Its tangent and material roll remain free, so this is a
prescribed-position hinged/swivel attachment, not a clamp. All other nodes,
including the distal node, are dynamic. The measured `c10` trajectory is a
prediction target, never a prescribed boundary.

The identified parameters are:

- homogeneous bending stiffness `EI` (`N m^2`); and
- objective Kelvin--Voigt bending damping `Cb` (`N m^2 s`).

For a naturally straight cable with a circular, bending-isotropic cross-section
and a free material frame, quasistatic twist minimizes to zero everywhere.
Torsional degrees of freedom and `GJ` are therefore eliminated analytically.
This is not the claim that the physical cable has `GJ = 0`; `GJ` simply does
not appear in this reduced centerline problem. A Bishop/parallel-transport frame
still permits fully three-dimensional centerline motion. This reduction follows
the stress-free isotropic case in [Discrete Elastic Rods](https://doi.org/10.1145/1360612.1360662).

Gravity and measured cable/marker masses are fixed inputs. Hard length
constraints are enforced while fixing only the attachment node. Ambient drag,
contact, a learned residual, attachment stiffness, intrinsic curvature, and
torsional damping are not fitted. Torsion must be reconsidered if the hardware
transmits root torque, a distal body or frictional contact transmits moment, or
the cable has appreciable cross-section anisotropy, coil set, or bend--twist
coupling.

The optimizer uses complete one-second differentiable rollouts. A fixed-seed
scrambled Sobol search in bounded log `EI`/`Cb` space initializes projected
CUDA-batched Adam. Periodic full-Training evaluations alone choose the saved
parameters. Candidate points, updates, source hashes, rejected windows, and
full-set scores are saved with the artifact.

The preflight rejects unreliable rigid-body pivots, discontinuous pivot or
marker motion, overlong measured chords, and excessive initialization
correction. A rejected transition invalidates both adjacent frames so it cannot
contaminate a causal velocity estimate or later rollout.

## Held-out validation

All protocols initialize once from causal measured history and then prescribe
only the attachment-pivot position:

- **Attachment-only**: no cable observation after initialization. This is the
  primary open-loop free-tip test.
- **Every 1 s**, **Every 5 s**, or **Every 10 s**: predict continuously and
  apply a complete cable-position observation at the stated interval.

A normal periodic observation corrects cable position while retaining and
projecting the predicted velocity; it does not restart the simulation. If a
scheduled complete observation is unavailable, the correction is skipped and
prediction continues. If the attachment pivot is temporarily unreliable, its
last trustworthy position is held and reported. The distal end is never held.
Results record protocol, correction frames, data/model hashes, held attachment
frames, and time-resolved error.

Legacy two-held-end CSVs and twist-aware `EI`/`GJ`/`Cb` artifacts are
mechanically and structurally incompatible with this experiment. Record new
one-pivot/free-tip takes and refit `EI`/`Cb`; do not migrate the old point
estimates.
