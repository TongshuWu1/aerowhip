# Cable--Object Contact and Force Estimation Methodology

## Status and purpose

This document records the research methodology for cable--object interaction
estimation. The raw/refined cube observation, pose covariance, separate
temporal cube-state filter, analytical cube surface query, and passive binary
contact observer are implemented. Bidirectional estimator feedback and the
mechanics sections remain a design specification. The implemented tracking
pipeline is documented separately in `PIPELINE.md`.

The immediate scope is contact between either tracked cable and the known
rigid cube. Cable--cable and cable self-contact are deliberately deferred.
Their contact geometry can later use the same interaction interface without
changing the cable--cube method.

The intended final system estimates, in order:

1. the cable and cube states from synchronized RGB-D;
2. the probability and location of cable--cube contact;
3. a bidirectional contact-consistent update of both state estimates;
4. the cable--cube action--reaction force from inverse mechanics;
5. later, sticking, sliding, and friction parameters.

The neural network remains responsible only for cable segmentation. Contact,
force, and friction are inferred from geometry, temporal motion, uncertainty,
and mechanics rather than by training another image-to-contact network.

## Main design decisions

- Contact is initially binary: `free` or `contact`.
- Each cable has its own cable--cube contact state.
- A single temporal cube-state distribution is shared by both cables.
- A single frame can propose contact but cannot establish high confidence.
- Cable and cube motion provide bidirectional evidence for contact.
- Initial contact feedback constrains only nonpenetration, surface gap, and
  relative normal motion.
- Tangential cable motion remains unconstrained until sticking and sliding are
  explicitly modelled.
- Contact correlation is evidence of interaction, not a measurement of force.
- Force is estimated later from an explicit cable mechanics model.
- Friction is estimated only after force and tangential contact mode are
  observable.
- Contact evidence is applied once per frame to avoid counting the same
  evidence repeatedly.

## System architecture

```text
Synchronized RGB-D
        |
        +--> PIDNet and skeleton graph
        |           |
        |           v
        |    Two independent cable PF visual updates
        |    shape, velocity, weights, ESS, covariance
        |
        +--> Yellow-cube geometric observation
                    |
                    v
             Rigid cube state filter
             pose, velocity, uncertainty
                    |
                    v
       Pairwise cable--cube interaction factors
       physical gap, normal motion, co-motion
                    |
                    v
       Binary temporal contact inference       [implemented, passive]
                    |
                    v
       Bidirectional cable/cube reweighting     [future]
                    |
                    v
       Contact-conditioned prediction/proposals
                    |
                    v
       Inverse cable mechanics for contact force
                    |
                    v
       Later: sticking, sliding, and friction
```

The visual cable and cube updates are performed before the interaction update.
This keeps the two visual likelihoods independent. The interaction factor then
links their post-visual state hypotheses through one physical relationship.

## State representation

### Cable state

Cable \(c\), particle \(i\), has the existing ordered centerline and node
velocities:

\[
X_{c,t}^{(i)}
=
\left(
\mathbf x_0,\ldots,\mathbf x_{N-1},
\mathbf v_0,\ldots,\mathbf v_{N-1}
\right).
\]

The current implementation uses 16 nodes, 800 particles per cable, fixed total
length, and a physical cable radius

\[
r_c = 0.0045\ {\rm m}.
\]

The cable PF continues to supply:

- particle states and weights;
- the posterior centerline;
- ordered local velocities;
- ESS;
- per-node posterior covariance;
- visible graph-edge attribution and arc intervals.

### Cube state

The raw geometric cube observation remains an unsmoothed visual measurement.
For interaction inference, it feeds a separate rigid-body temporal state
distribution. The implemented representation is a Gaussian error state about
the filtered rigid pose and velocity:

\[
\delta \mathbf o_t
=
\left(
\delta\mathbf p_t,
\delta\boldsymbol\theta_t,
\delta\mathbf v_{o,t},
\delta\boldsymbol\omega_{o,t}
\right).
\qquad
\delta\mathbf o_t\sim\mathcal N(\mathbf 0,P_{o,t}).
\]

The nominal state contains cube pose \(T\), centre velocity \(\mathbf v_o\),
and angular velocity \(\boldsymbol\omega_o\). A robust joint fit supplies a
six-dimensional tangent-space pose measurement and covariance to a separate
Gaussian error-state filter. Centre and linear velocity follow a
constant-velocity model. Orientation is symmetry-aligned to the previous
accepted visual orientation and updated only from accepted measurements;
prediction-only frames hold it while its covariance grows. Angular velocity
is estimated by smoothing finite differences between consecutive accepted
visual orientations, but it does not drive nominal orientation prediction.
This prevents an untextured symmetric cube from acquiring false rotation from
measurement noise while retaining angular motion as evidence for later
interaction inference. This is deliberately not a persistent cube particle
population: after resolving the cube's finite rotation symmetry, the local
rigid-pose posterior is expected to be unimodal, and the Gaussian
representation is substantially cheaper to predict and update.

When the later nonlinear contact factor needs numerical marginalization,
index \(j\) denotes a small deterministic set of temporary cubature or sigma
hypotheses drawn from this Gaussian. These hypotheses exist only during the
interaction calculation; they are not another temporal tracker.

The cube's 24 geometrically equivalent proper rotations describe the same
physical surface. A temporally continuous representative is required for
angular-velocity estimation, but the signed-distance geometry itself is
invariant to this choice.

### Contact state

Each cable has one binary temporal contact variable:

\[
z_{c,t}\in\{0,1\}
=
\{\text{free},\text{contact}\}.
\]

The internal result is always a probability distribution. A displayed
contact/no-contact decision, if needed, is only a view of that distribution.

### Force and material parameters

The later mechanics layer introduces:

- contact force \(\boldsymbol\lambda_{c,t}\);
- cable tension multipliers;
- endpoint reaction forces when endpoints are constrained;
- cable mechanical parameters \(\theta_c\), including bending stiffness and
  rest curvature;
- later, static and kinetic friction parameters.

## Physical cable--cube geometry

The cube is represented as an oriented box with known side length. The contact
query uses the cube signed-distance function and the physical cable radius.
For cable particle \(i\), cube hypothesis \(j\), and cable arc coordinate
\(s\),

\[
g_{ij}(s)
=
\operatorname{SDF}_{O^{(j)}}\!\left(
\mathbf x^{(i)}(s)
\right)
- r_c.
\]

This is equivalent to testing the cable centerline against the cube inflated
by the cable radius.

- \(g>0\): physical surface separation;
- \(g\approx0\): geometrically possible contact;
- \(g<0\): apparent physical penetration.

The query returns:

\[
\mathcal C_{ij}
=
\left(
g_{ij},
s_{ij},
\mathbf p_{ij},
\mathbf n_{ij},
[s_a,s_b]_{ij}
\right),
\]

where:

- \(g_{ij}\) is the minimum gap;
- \(s_{ij}\) is its cable arc coordinate;
- \(\mathbf p_{ij}\) is the closest cube surface point;
- \(\mathbf n_{ij}\) is the outward cube surface normal;
- \([s_a,s_b]_{ij}\) is the candidate contact interval.

The contact interval is derived from contiguous arc samples whose gaps are
consistent with the geometric uncertainty. It represents a possible contact
patch rather than only one minimum-distance point.

## Relative motion and object-response evidence

The rigid-body velocity of cube surface point \(\mathbf p\) is

\[
\mathbf v_{\rm surface}(\mathbf p)
=
\mathbf v_o
+
\boldsymbol\omega_o
\times
(\mathbf p-\mathbf c_o).
\]

The cable velocity at the candidate arc position is obtained from its ordered
node velocities or a timestamp-aware local interpolation:

\[
\mathbf v_{\rm cable}(s).
\]

The relative velocity is

\[
\mathbf v_{\rm rel}
=
\mathbf v_{\rm cable}(s)
-
\mathbf v_{\rm surface}(\mathbf p).
\]

Its normal component is

\[
v_n
=
\mathbf n^\top\mathbf v_{\rm rel}.
\]

Binary contact requires compatible normal motion but does not require
tangential co-motion. A cable can be in contact while sliding.

Persistent cable/cube co-motion supplies additional positive contact evidence.
For example, if a nearby cable starts moving and the cube surface subsequently
moves with it, the odds of contact should increase substantially. Absence of
co-motion is not decisive negative evidence because the cable may be sliding
or may not be transmitting enough force to move the cube.

The motion evidence must be timestamp-aware and evaluated over a short
temporal history. It should compare the local cable material-point motion with
the corresponding rigid cube-surface motion, rather than correlate only their
global image displacement.

## Binary contact likelihood

For one cable-particle/cube-hypothesis pair, the contact likelihood is based on
near-zero physical gap and compatible normal motion:

\[
\Psi_{\rm contact}^{ij}
=
\exp\left[
-
\frac{g_{ij}^2}{2\sigma_g^2}
-
\frac{v_{n,ij}^2}{2\sigma_n^2}
\right].
\]

Cube pose covariance and surface measurement resolution determine
\(\sigma_g\). Cable uncertainty is not counted a second time in this scalar:
the compatibility is evaluated for every cable particle and then marginalized
with its normalized PF weight.

The implemented free-space compatibility is the Gaussian probability that the
uncertain physical gap lies on the separated side of the contact boundary:

\[
\Psi_{\rm free}^{ij}
= \Phi\!\left(\frac{g_{ij}}{\sigma_g}\right).
\]

Thus clear positive separation supports `free`, physical penetration opposes
it, and a zero measured gap remains ambiguous rather than being declared
contact by geometry alone. Both state likelihoods are mixed with a fixed
outlier component before the Bayesian update, bounding the influence of any
single frame.

Co-motion supplies a temporal Bayes factor that can increase the contact odds
when spatial proximity is already plausible. It must not turn distant
co-moving bodies into contact.

The current image must support the cable arc at which each particle is closest
to the cube. Let (q_i\in\{0,1\}) denote that local support lookup. The
implemented marginalized likelihood is

\[
L_z = \sum_i w_i\left[q_i\,\Psi_z^i + (1-q_i)\right].
\]

The neutral term is identical for `contact` and `free`, so geometry inferred
only from a hidden or missing arc cannot change the contact odds. Particles
whose closest arc is observed still contribute normally, making
\(\sum_i w_iq_i\) a continuous local support mass rather than introducing a
global visibility threshold. Co-motion is gated by the same (q_i), because a
predicted velocity at a hidden contact arc is not new motion evidence.

The representative contact geometry is selected from the particle maximizing
\(w_i q_i \Psi_{\rm contact}^i\), not from the unconstrained PF MAP particle.
This reporting rule makes the displayed gap, surface point, normal, and arc
interval consistent with both the observed local cable support and the contact
hypothesis. It does not alter the marginalized likelihood above.

## Temporal contact inference

The predicted binary contact probability is

\[
\pi_{c,t}^{-}(z)
=
\sum_{z_{t-1}}
P(z\mid z_{t-1})
\pi_{c,t-1}(z_{t-1}).
\]

The transition model gives contact persistence without a hard requirement for
a fixed number of frames. Missing visual measurements propagate the temporal
probability while growing the state uncertainty; they do not create new
contact evidence.

When cable and cube are both motionless and their separation is below the
sensor resolution, true contact may be unobservable. In that case the contact
posterior should remain uncertain rather than force a decision.

## Bidirectional cable--cube interaction update

After the independent visual updates, the joint interaction posterior for
cable particle \(i\), cube error state \(\delta\mathbf o\), and contact state
\(z\) is

\[
P_t(i,\delta\mathbf o,z)
\propto
w_{c,t}^{(i)}
\mathcal N(\delta\mathbf o;\mathbf0,P_{o,t})
\pi_{c,t}^{-}(z)
\Psi_z^{i}(\delta\mathbf o).
\]

Marginalizing this joint distribution gives the updated cable weights,

\[
\widetilde w_{c,t}^{(i)}
=
\sum_z\int P_t(i,\delta\mathbf o,z)\,d\delta\mathbf o,
\]

the contact-conditioned cube distribution,

\[
p_t(\delta\mathbf o\mid\text{interaction})
=
\sum_{i,z}P_t(i,\delta\mathbf o,z),
\]

and the updated contact probability,

\[
\pi_{c,t}(z)
=
\sum_i\int P_t(i,\delta\mathbf o,z)\,d\delta\mathbf o.
\]

The integrals can be evaluated with a small deterministic sigma set and the
resulting cube message moment-matched back to the Gaussian error state. This
keeps the interaction cost proportional to cable particles times a small
fixed sigma count, without running another object-particle population.

This is the central two-way inference mechanism:

- cube pose and motion reweight cable hypotheses;
- cable shape and motion update the cube state distribution;
- both state distributions contribute to contact confidence.

The interaction update is evaluated once per frame. Iterating the same factor
within the frame would count the same evidence multiple times and could create
self-confirming contact.

For two cables, each cable has its own joint interaction factor with the same
cube distribution. If both are likely to contact the cube, their messages
jointly update the cube Gaussian. The cable PF populations remain separate but
become conditionally coupled through the shared cube state only when physical
interaction is supported.

## Contact-conditioned PF feedback

The initial feedback uses only properties shared by sticking and sliding:

1. nonpenetration;
2. near-zero physical surface gap when contact is probable;
3. compatible normal velocity.

Tangential velocity is not constrained in the binary model.

The primary operation is probabilistic reweighting. After resampling,
contact-conditioned proposals may correct the candidate cable interval along
the cube normal. The existing fixed-length and smoothness operations then
restore the cable's internal constraints.

The proposal population is a mixture determined by the contact posterior:

- free hypotheses retain ordinary cable prediction;
- contact hypotheses receive normal contact conditioning;
- uncertain contact preserves both possibilities.

This is direct sampling from the current binary interaction posterior.

The cube receives the corresponding probabilistic message, but strong current
visual cube evidence remains authoritative. Contact should help carry the cube
state through partial observation; it must not drag a well-observed cube away
from its image evidence.

## Force estimation from inverse cable mechanics

Geometric proximity and motion correlation establish contact evidence but do
not determine force magnitude. Force is introduced only through a mechanical
model.

### Cable mechanics

The first force model should be an inextensible discrete elastic rod defined
on the ordered cable state. Required terms are:

- fixed segment lengths;
- bending stiffness;
- rest curvature;
- linear density and gravity;
- endpoint boundary conditions;
- optionally damping and inertia for faster motion.

For slow manipulation, the initial model is quasi-static:

\[
\mathbf 0
=
\mathbf f_{\rm bend}(X;\theta)
+
\mathbf f_g
+
J_L^\top\boldsymbol\tau
+
J_c^\top\boldsymbol\lambda
+
J_e^\top\mathbf f_e.
\]

Here:

- \(\theta\) contains the cable constitutive parameters;
- \(\boldsymbol\tau\) are the inextensibility tension multipliers;
- \(\boldsymbol\lambda\) is the cable--cube contact force;
- \(\mathbf f_e\) contains endpoint reactions;
- \(J_L\), \(J_c\), and \(J_e\) map these forces into cable-node space.

The contact force and nuisance reactions are estimated with an
uncertainty-weighted constrained optimization:

\[
\min_{\boldsymbol\lambda,\boldsymbol\tau,\mathbf f_e}
\left\|
\mathbf f_{\rm bend}
+
\mathbf f_g
+
J_L^\top\boldsymbol\tau
+
J_c^\top\boldsymbol\lambda
+
J_e^\top\mathbf f_e
\right\|_W^2
+
\alpha
\left\|
\boldsymbol\lambda_t-\boldsymbol\lambda_{t-1}
\right\|^2.
\]

The weight matrix \(W\) derives from cable posterior uncertainty. Hidden or
poorly constrained geometry therefore contributes less force information.

The normal force must satisfy

\[
\lambda_n\ge0.
\]

When the binary contact posterior strongly favours free space,

\[
\boldsymbol\lambda=\mathbf0.
\]

Force should be estimated per connected contact interval as an aggregate
contact wrench, not as an arbitrary independent force at every centerline
sample. A cable passing over two cube faces can create two patches because the
surface normals differ.

### Dynamic extension

For motion where inertial terms are not negligible, the same model becomes

\[
M(X)\ddot X
+
D\dot X
=
\mathbf f_{\rm bend}
+
\mathbf f_g
+
J_L^\top\boldsymbol\tau
+
J_c^\top\boldsymbol\lambda
+
J_e^\top\mathbf f_e.
\]

The quasi-static formulation is the starting point because acceleration
estimated from RGB-D is substantially noisier than position and velocity.

## Action--reaction and cube dynamics

The same physical contact force acts oppositely on the cube:

\[
\mathbf F_{\rm cube}
=
-\boldsymbol\lambda.
\]

At cube surface point \(\mathbf p\), the corresponding wrench about the cube
centre is

\[
W_{\rm contact}
=
\begin{bmatrix}
-\boldsymbol\lambda\\
(\mathbf p-\mathbf c_o)
\times
(-\boldsymbol\lambda)
\end{bmatrix}.
\]

The final force estimator should prefer a force that simultaneously explains
cable deformation and cube motion:

\[
\boldsymbol\lambda^*
=
\arg\min_{\boldsymbol\lambda}
\left[
\|r_{\rm cable}(\boldsymbol\lambda)\|_{W_c}^2
+
\|r_{\rm cube}(-\boldsymbol\lambda)\|_{W_o}^2
\right].
\]

Mass and inertia belong to this mechanics layer, not to the visual cube-state
filter. The later implementation will require the measured cube mass \(m_o\)
and centre-frame inertia \(I_o\). For a homogeneous cube of side \(a\),

\[
I_o=\frac{m_o a^2}{6}I_3,
\]

but the printed cube must be weighed and its actual mass distribution checked
before using this expression. Hollow walls, ballast, or attached markers make
the homogeneous approximation inappropriate.

The current cube rests on a supporting surface. Its dynamics therefore also
contain table normal force and table friction:

\[
M_o\dot V_o
=
W_{\rm contact}
+
W_{\rm table}
+
W_g.
\]

Cube acceleration alone cannot separate cable force from unknown table
reaction. Initially:

- cable mechanics provide the primary force estimate;
- cube motion provides contact evidence and a force-consistency term;
- absolute force from cube dynamics requires a known or separately identified
  table-contact model.

The support model will therefore be a separate mechanics component with a
known table plane, unilateral normal reaction, gravity, and later a calibrated
table-friction law. None of these parameters is inserted into pose refinement
or the temporal visual filter.

## Later sticking, sliding, and friction

Friction is deliberately deferred until binary contact and force are
established.

The later contact state becomes

\[
z_t
\in
\{\text{free},\text{sticking},\text{sliding}\}.
\]

During sticking,

\[
\|\boldsymbol\lambda_t\|
\le
\mu_s\lambda_n,
\qquad
\mathbf v_{\rm rel}\approx\mathbf0.
\]

Ordinary sticking supplies only a lower bound:

\[
\mu_s
\ge
\frac{\|\boldsymbol\lambda_t\|}{\lambda_n}.
\]

During sliding,

\[
\boldsymbol\lambda_t
=
-\mu_k\lambda_n
\frac{\mathbf v_t}{\|\mathbf v_t\|}.
\]

The kinetic coefficient is estimated as a persistent material parameter over
multiple sliding frames:

\[
\widehat\mu_k
=
\arg\min_{\mu\ge0}
\sum_t w_t
\left\|
\boldsymbol\lambda_{t,t}
+
\mu\lambda_{n,t}
\widehat{\mathbf v}_{t,t}
\right\|^2.
\]

A stick-to-slip transition provides the strongest estimate of limiting static
friction. Without such a transition, an exact static coefficient should not
be reported.

## Deferred cable--cable contact

Cable--cable contact will reuse:

- the same binary temporal contact model;
- the same pairwise population update;
- the same relative-normal-motion likelihood;
- the same action--reaction formulation.

Only the geometric contact adapter changes. For two cable centerlines,

\[
g_{12}(s_1,s_2)
=
\left\|
\mathbf x_1(s_1)-\mathbf x_2(s_2)
\right\|
-
(r_1+r_2).
\]

This deferred extension is not part of the immediate cable--cube
implementation.

## Planned outputs

### Binary contact stage

For each cable:

- cable posterior centerline and velocity;
- cable posterior covariance and ESS;
- cube pose and motion distribution;
- minimum physical cable--cube gap;
- contact arc coordinate and interval;
- closest cube surface point and normal;
- temporal motion-coupling evidence;
- probability of contact;
- contact-state uncertainty.

### Force stage

Additionally:

- estimated normal contact force;
- aggregate contact force and cube wrench when tangential mechanics are
  identifiable;
- force covariance or confidence;
- cable and cube mechanics residuals.

### Later friction stage

Additionally:

- sticking and sliding probabilities;
- tangential contact force;
- static-friction lower bound;
- kinetic-friction estimate;
- confidence and observability status.

## Planned implementation sequence

Completed foundations:

1. Retain the raw plane observation and add robust refined pose/covariance.
2. Add the separate temporal Gaussian rigid cube-state filter.
3. Add the analytical oriented-cube signed-distance, closest-point, and normal
   query.
4. Apply the physical cable radius and evaluate weighted cable-particle gap,
   normal motion, and bounded positive co-motion evidence on CUDA.
5. Add a passive binary temporal contact posterior for each cable, with
   measurement gating and prediction-only propagation.
6. Gate geometry and co-motion by posterior-weighted support at the candidate
   contact arc, leaving hidden arcs neutral.

Remaining sequence:

1. Analyse passive contact recordings before enabling any estimator feedback.
2. Add one-pass bidirectional cable-PF reweighting and a moment-matched
   Gaussian cube message.
3. Add contact-conditioned normal proposals behind an isolated feature switch.
4. Add the quasi-static inverse cable mechanics force estimator.
5. Add cube action--reaction consistency after mass, inertia, and the support
   model are available.
6. Later extend contact to sticking/sliding and estimate friction.
7. Later add cable--cable contact through the common interaction interface.

## Required methodological invariants

- A two-dimensional image overlap is never physical contact evidence by
  itself.
- Contact uses the physical cable radius, not centerline intersection.
- Contact likelihood never treats tangential co-motion as mandatory.
- Binary contact feedback never imposes tangential sticking.
- Free-space feedback rejects penetration but never attracts a separated
  cable to the cube.
- A single frame cannot create high-confidence persistent contact.
- Missing observations cannot create new contact evidence.
- The same interaction evidence is applied only once per frame.
- Contact uncertainty must increase when the supporting state estimates become
  uncertain.
- Contact correlation is not reported as force.
- Force is not reported without explicit mechanical assumptions.
- Static friction is not reported as an exact coefficient without an
  informative slip transition.
- Cable--cube interaction remains isolated from the visual measurement code
  and can be disabled for ablation.
