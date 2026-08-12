# Differentiable torsion-free constrained elastic rod

## Model used in this project

The cable is represented by `N=24` ordered point masses joined by `N-1`
inextensible links. Each interior joint stores bending energy and dissipates
curvature-rate motion. The same homogeneous material parameters are shared by
all joints:

- bending stiffness `EI` in `N m^2`;
- effective bending damping `Cb` in `N m^2 s`.

The implementation is differentiable in PyTorch, so `EI` and `Cb` can be
identified by differentiating through multi-frame dynamics rollouts.

This is best described as a **differentiable torsion-free constrained elastic
rod using discrete-elastic-rod bending geometry**. Mechanically, it is an
inextensible spring-joint chain. It is not the full twist-aware DDER model.

## Assumptions

Phase 1 models one homogeneous cable with measured length, mass, and diameter.
Its undeformed centreline is straight and its circular cross-section is treated
as isotropic. Axial stretch, shear, material twist, contact, and friction are
omitted. These assumptions leave centreline bending as the relevant deformation.

The offline experiment is planar. The shared rod state and equations remain 3D,
which allows the identified bending parameters to be used by the online 3D
particle filter under the isotropic-cross-section assumption.

## State and hard constraints

For positions `x_0 ... x_(N-1)` and velocities `v_0 ... v_(N-1)`, the uniform
link length is

```math
l=\frac{L}{N-1}.
```

Every link satisfies

```math
h_i(x)=\|x_{i+1}-x_i\|-l=0.
```

The two measured endpoints are `x_0` and `x_(N-1)` and enter the dynamics as
prescribed boundary positions. A coupled mass-weighted multiplier solve enforces
the position constraints, followed by the corresponding velocity projection.
The reactions are axial along the links. The current implementation does not
retain consistently time-scaled multipliers and therefore does not claim cable
tension or contact-force estimates.

## Nonlinear bending and damping

For adjacent unit tangents `t_(i-1)` and `t_i`, the discrete-elastic-rod
curvature binormal is

```math
\kappa_i = \frac{2\,t_{i-1}\times t_i}
                 {1+t_{i-1}\cdot t_i}.
```

For a straight rest centreline, the bending energy is

```math
U_b=\frac{EI}{2}\sum_i \frac{\|\kappa_i\|^2}{\bar l_i},
```

where `l_bar_i` is the dual length around joint `i`. The internal bending force
is `-dU_b/dx`.

The material derivative of the same nonlinear curvature defines the
Kelvin-Voigt bending dissipation

```math
R_b=\frac{C_b}{2}\sum_i \frac{\|\dot\kappa_i\|^2}{\bar l_i},
\qquad
f_b^d=-\frac{\partial R_b}{\partial v}.
```

This damps changes of shape without applying an artificial decay to rigid-body
translation.

For a planar joint angle `theta`, `|kappa|=2 tan(theta/2)`. In the small-angle
limit the model becomes a uniform angular spring-damper chain:

```math
k_\theta \approx \frac{EI}{\bar l},
\qquad
c_\theta \approx \frac{C_b}{\bar l}.
```

The implemented finite-angle energy uses the nonlinear curvature above, not the
small-angle `theta^2` approximation.

## Time integration

Each frame interval is divided into fixed substeps. Gravity, bending force, and
curvature-rate damping are advanced with a semi-implicit Euler step. Endpoint
positions are interpolated over the substeps. The coupled length projection and
velocity projection then enforce the endpoint and inextensibility constraints.
The explicit bending and damping updates are checked against conservative
mass/grid/time-step stability limits.

## Planar RGB measurement

PIDNet supplies the cable body and its two endpoints in the left RGB image. A
connected skeleton geodesic between the endpoints provides one ordered route.
The camera is fixed and level, and the cable is assumed to move approximately
in a plane parallel to the image sensor at nearly constant depth. Identification
recordings exclude occlusion, crossings, and contact.

For complete route pixel lengths `P_t` and measured cable length `L`, one scale
is estimated for the entire recording:

```math
s = \frac{L}{\operatorname{median}_t P_t}.
```

With image centre `(c_x,c_y)`, the metric image-plane coordinates are

```math
x=s(u-c_x), \qquad z=s(c_y-v).
```

The route is resampled by arc length into 24 ordered observation nodes. Their
positions remain PIDNet measurements; they are not forced onto the rod. A
three-frame local quadratic is used only to estimate the velocity required to
initialize each rollout. The first observation of a rollout is projected once
to produce a feasible rod state.

This mapping does not reconstruct depth. Out-of-plane motion appears as
foreshortening and violates the Phase 1 observation model.

## Identification of `EI` and `Cb`

`EI` and `Cb` are positive global parameters represented in bounded log space.
Because there are only two parameters, the full-batch objective is minimized by
L-BFGS with a strong-Wolfe line search.
For every valid multi-frame window:

1. initialize a feasible position and velocity state;
2. prescribe the two observed endpoint trajectories;
3. roll the rod dynamics forward;
4. compare predicted interior nodes directly with the PIDNet nodes using a
   robust metric loss; and
5. backpropagate through the rollout.

All selected recordings contribute to one joint fit. Slowly changing shapes
provide most of the stiffness information. Transverse motion followed by
stationary endpoints provides damping information. Because aerodynamic and
other unresolved losses are not modeled separately, `Cb` is reported as
**effective bending damping**.

The optimizer retains the finite parameter pair with the lowest rollout loss
encountered during its evaluations.

PIDNet is the Phase 1 measurement reference. It is not independent physical
ground truth unless it is separately validated against manual annotations or
motion capture.

## Synthetic recovery check

`run_model_check.py` generates a known, exactly inextensible planar trajectory
with prescribed `EI` and `Cb`, then calls the same production rollout fitter.
The generated motion contains a transverse velocity mode, slow endpoint shape
change, a faster endpoint change, and settling. Non-overlapping windows avoid
repeatedly weighting nearly identical synthetic frames.

The check saves the true and recovered parameters and rollout RMSE. Exact
simulator velocities initialize its windows so the result isolates the dynamics
and parameter optimizer. It is a necessary implementation sanity check; it is
not an evaluation of RGB velocity estimation and not independent experimental
validation.

## Scope of the claim

The model includes:

- ordered 3D centreline point positions and velocities;
- exact link-length and endpoint constraints;
- nonlinear DER centreline bending;
- curvature-rate bending damping;
- gravity; and
- differentiable, batched rollouts.

It does not include material directors, twist angles, torsional stiffness,
axial stretch, shear, contact, friction, learned residual dynamics, or force
estimation. Internal names such as `DderModel` and the existing artifact schema
are retained for code compatibility; they do not imply a full twist-aware DDER
implementation.
