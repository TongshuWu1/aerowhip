# Contact-Aware Cable Digital Twin: Research Direction and Codex Handoff

**Status:** Forward research design, not yet implemented
**Date:** 2026-07-31
**Project:** `PF_cable` / `particle_filter_cable_project`

## 1. How to use this document

This document records the intended next-generation research direction so a new
Codex task can continue without relying on the previous conversation history.

- `PIPELINE.md` remains the authority for what the repository currently
  implements.
- `CONTACT_FORCE_METHODOLOGY.md` records the earlier PF-centred contact and
  force plan. It is still useful background, but this document supersedes it
  wherever the two future designs conflict.
- The current tracker should be preserved as a working baseline, perception
  frontend, recorder, and comparison method. It is not a constraint on the
  final estimator architecture.
- The working tree contains substantial intentional uncommitted work. Do not
  reset, discard, overwrite, or broadly reformat it.

No major redesign described below should be integrated directly into the live
pipeline before its mechanical model and interfaces have been established in a
standalone offline prototype.

## 2. Research objective

The long-term objective is to use a cable suspended or actuated by two aerial
endpoints as a lightweight distributed manipulator. The system should observe
the full cable and a rigid object with one RGB-D camera, infer when and where
the cable contacts the object, predict their coupled motion through occlusion,
and control the cable endpoints to acquire, maintain, exploit, lose, and
recover contact while manipulating the object.

The intended first-paper scope is deliberately narrower:

- one physical cable;
- one known rigid object, beginning with the 150 mm cube;
- slow planar dragging or rolling on a known table;
- a single RGB-D camera;
- measured or commanded endpoint trajectories;
- binary free/contact belief with a cable arc interval;
- contact-aware endpoint control;
- deliberate visual occlusion and contact-loss recovery.

Cable-cable contact, hitches, multilayer wraps, arbitrary deformable objects,
general high-speed 6-DoF manipulation, and detailed friction-state estimation
are later extensions.

## 3. Precise research gap

The motivation must not claim that previous catenary-robot work cannot
manipulate objects. The D'Antonio research line already demonstrates object
dragging and rolling, feedback on object pose, adaptive control in simulation,
and hitch-based transportation.

The defensible missing capability is observation-driven manipulation during
physical cable-object interaction:

- no full observed 3D cable state while the cable contacts and occludes an
  object;
- contact locations or cable paths are commonly prescribed or simplified;
- no probabilistic inference of contact acquisition, persistence, loss, or
  hidden contact;
- no physically interpretable online contact-arc estimate driven jointly by
  cable and object response;
- no predictive endpoint controller using this contact belief and recovering
  when its assumed contact is lost.

A concise working thesis is:

> An offline system-identified, RGB-D-corrected interaction twin that infers
> hidden cable-object contact from coupled cable and object motion and uses
> that belief for closed-loop two-endpoint aerial manipulation.

The cable should be described as a **distributed end effector**, not as a
universal replacement for a rigid robot arm. A cable is tensile-only,
underactuated, and depends on environmental contact for controllability.

## 4. Central architectural decision

The final system should use an expensive, high-fidelity model and physical
parameter identification offline, then use a reduced model with visual
correction online.

```text
Controlled RGB-D and endpoint-motion recordings
                     |
                     v
      High-resolution rod--rigid-body simulator
                     |
       staged physical system identification
                     |
       model reduction / coarse re-identification
                     |
                     v
RGB-D --> reduced online interaction twin --> contact belief
  ^                    |                           |
  |                    v                           v
  +---------- endpoint-controlled scene <-- MPC / MPPI
```

This separation allows expensive optimization, calibration, and possibly
training to occur once offline. The online process performs only reduced
simulation, short-window state correction, a small number of contact-mode
hypotheses, and batched control rollouts.

The current 16-node, 800-particle-per-cable PF should not be assumed to be the
final architecture. It remains the experimental baseline until a replacement
demonstrably provides a better observation/dynamics/contact formulation.

## 5. What to learn from PhysTwin

PhysTwin offers two important principles:

1. reconstruct or retain complete physical geometry so collision is not
   performed on a visually convenient zero-thickness abstraction;
2. perform expensive physical identification offline and deploy only a fast
   CUDA simulator online.

However, its implementation should not be copied literally. PhysTwin first
reconstructs a triangle mesh for general unknown deformable objects, then
samples surface, volume, and observation points and simulates a spring-mass
graph. Its Warp `HashGrid` bins mass points and finds nearby collision
candidates. The hash is a broad-phase acceleration structure; it is not the
contact law, a surface representation, or a physics model.

Useful ideas to borrow:

- offline system identification from synchronized observations;
- a high-fidelity offline model and compact online model;
- CUDA/Warp parallel dynamics and collision queries;
- physics prediction through visual occlusion;
- derivative-free optimization for non-smooth parameters followed by
  gradient-based refinement where appropriate;
- separation of appearance/observation modelling from mechanics.

Ideas not to copy by default:

- a generic dense spring network when cable topology and dimensions are known;
- Gaussian-splat appearance reconstruction for a known coloured cable and
  cube;
- thousands of surface vertices as the estimator state;
- point-point collision when analytic tube-to-SDF contact is available;
- interpreting simulator constraint forces as calibrated Newtons without
  parameter and sensor calibration.

Primary references:

- Local PhysTwin paper:
  `C:\Users\wts28\Downloads\Jiang_PhysTwin_Physics-Informed_Reconstruction_and_Simulation_of_Deformable_Objects_from_Videos_ICCV_2025_paper.pdf`
- Official PhysTwin repository: <https://github.com/Jianghanxiao/PhysTwin>
- PhysTwin project page: <https://jianghanxiao.github.io/phystwin-web/>

## 6. Three representations, each with one job

The cable should not be represented by one data structure for every purpose.

| Layer | Representation | Purpose |
|---|---|---|
| Latent mechanics | Discrete elastic rod or Cosserat rod | Mass, bending, damping, endpoint boundary conditions, and dynamics |
| Physical collision | Swept finite-radius tube or segment capsules | Cable-object, table, and later self/cable-cable collision |
| Observation/rendering | Watertight swept triangle mesh | Predicted silhouette, depth, occlusion, and visualization |

For a rod centreline `x(s)`, cross-section orientation `R(s)`, and cable radius
`r`, the physical surface is

\[
\mathbf p(s,\theta)
=
\mathbf x(s)+
R(s)
\begin{bmatrix}
r\cos\theta\\
r\sin\theta\\
0
\end{bmatrix}.
\]

This is a real surface even though the mechanical state remains compact. A
dense triangle tube may be generated for rendering, while collision is more
robustly evaluated with capsule distances or a signed-distance field.

For object SDF `phi_O`, the cable-surface gap is

\[
g(s)=\phi_O(\mathbf x(s))-r.
\]

- `g > 0`: separated surfaces;
- `g = 0`: geometrically compatible contact;
- `g < 0`: penetration that the mechanics must resolve.

The SDF gradient supplies the object surface normal. For the current cube, an
analytic oriented-box or rounded-box SDF is preferable. For a future arbitrary
known object, precompute an SDF from its mesh.

For the current circular, visually textureless cable, axial twist is weakly
observable. The initial mechanical model may omit twist and retain centreline
bending only. Cross-section directors and torsion should be introduced only if
experiments show that twist materially affects contact or manipulation.

## 7. Offline high-fidelity interaction twin

### 7.1 State and mechanics

The offline simulator should initially use approximately 64--128 rod nodes and
contain:

- fixed cable length and measured radius;
- measured linear density and gravity;
- inextensibility or very stiff axial response;
- bending stiffness and rest curvature;
- internal damping and, if required, aerodynamic drag;
- measured endpoint position and velocity boundary conditions;
- a rigid cube with measured mass and inertia;
- a known table plane with support and friction;
- unilateral finite-radius cable-cube contact;
- equal-and-opposite contact impulses on cable and cube;
- frictional contact, initially using an offline calibrated coefficient.

The 0.518 m cable with 64 nodes has roughly 8 mm spacing, which is suitable for
resolving bending near a 9 mm diameter cable and cube edges. The exact
resolution must be selected by spatial convergence, not by assumption.

An implicit or constraint-based integrator is preferable for stiff
inextensibility and contact. The model should avoid arbitrary soft penalties
whose apparent force changes substantially with timestep or mesh resolution.

### 7.2 Collision acceleration

- One cable against the current cube: use direct batched tube/capsule-to-SDF
  queries.
- A triangle-mesh object: use its SDF or a BVH for narrow-phase queries.
- Cable self-contact, several cables, or many bodies: use a CUDA spatial hash
  to produce candidate segment pairs, followed by exact segment/tube contact.
- A spatial hash must never replace the narrow-phase gap, normal, and contact
  constraint calculations.

At higher speeds, collision detection may require swept or continuous segment
queries so a thin cable cannot tunnel through an object between timesteps.

### 7.3 Parameters and staged identification

Measure directly whenever possible:

- cable length and radius;
- cable linear density;
- cube dimensions, mass, and inertia or mass distribution;
- camera intrinsics/extrinsics;
- table plane;
- endpoint trajectories and timestamps.

Identify the remaining parameters in separate experiments:

1. **Free static shapes and free swings:** bending stiffness, rest curvature,
   internal damping, and possibly aerodynamic drag.
2. **Cube-only pushes/slides:** cube-table friction and support behaviour.
3. **Controlled cable contact/sliding:** cable-cube friction and contact
   regularization/compliance.
4. **Full cable-cube interactions:** final joint predictive refinement without
   allowing one parameter group to compensate arbitrarily for another.

A candidate parameter vector is

\[
\vartheta=
\{EI,c_{\rm internal},c_{\rm drag},
\mu_{\rm cable-object},\mu_{\rm table},
\text{contact regularization}\}.
\]

Do not jointly infer cable mass, bending stiffness, damping, both friction
coefficients, tension, and contact force from a single RGB-D sequence. Many of
those combinations can produce nearly identical visible motion.

Use synchronized RGB-D, endpoint trajectories, estimated centrelines, cube
poses, and preferably endpoint tension or UAV thrust as offline residuals.
Derivative-free search can initialize discontinuous contact/friction
parameters; differentiable simulation can refine smooth material parameters.

### 7.4 Reduction for online use

After calibrating the high-resolution model:

1. generate representative free and contact rollouts;
2. construct a 24--32-node rod using the same physical structure;
3. re-identify its effective coarse parameters against high-resolution
   trajectories, contact intervals, cube response, and endpoint reactions;
4. verify that reducing resolution does not materially change contact onset or
   the predicted object response.

A small learned residual model is optional later, only after systematic
one-step model error is demonstrated. It must preserve length,
nonpenetration, and action-reaction. Do not begin with an unconstrained neural
contact dynamics model.

## 8. Recommended online estimator

The recommended final estimator is a **hybrid fixed-lag interaction smoother**
rather than a large population of complete cable-shape particles.

Maintain a short window of approximately 4--6 frames and a small bank of
contact hypotheses, for example:

- free cable;
- contact on candidate arc interval A;
- contact on candidate arc interval B;
- optionally a persistent previous-contact interval.

For each hypothesis, the continuous state contains:

\[
X_t=
\{q_t,\dot q_t,T^O_t,V^O_t\},
\]

where `q` is the reduced rod state, `T^O` is cube pose, and `V^O` is cube
twist. The discrete state contains free/contact and its connected cable arc
interval. Contact impulses and endpoint reactions should normally be nuisance
variables solved within the dynamics transition, not persistent unconstrained
state variables.

Each frame should:

1. run the existing semantic perception and registered-depth acquisition;
2. obtain cable/endpoints and cube observations from the exact synchronized
   frame;
3. generate only geometrically and temporally plausible contact intervals;
4. predict every retained hypothesis with the reduced CUDA simulator;
5. render its cable tube and cube silhouette/depth;
6. correct the short trajectory using PIDNet logits, depth, endpoints, cube
   pose, dynamics, and contact residuals;
7. score marginal evidence, update mode probabilities, and prune weak modes;
8. extract continuous covariance from the local Hessian together with the
   remaining discrete mode uncertainty.

One to three warm-started Gauss--Newton iterations may be sufficient if the
previous window is shifted forward each frame. Exact optimizer and contact
relaxation choices must be tested in the standalone replay before live
integration.

This recommendation does not require deleting the current PF immediately. The
current PF is the baseline. A rod transition could first be evaluated inside
offline replay, but production should ultimately contain one chosen estimator
rather than accumulating permanent PF, smoother, and fallback paths.

## 9. Contact inference from coupled motion

Single-frame surface proximity proposes contact but cannot establish it. The
key evidence is temporal and bidirectional.

The free hypothesis predicts cable and cube independently. A contact
hypothesis couples them with a unilateral, equal-and-opposite impulse. The
contact probability rises when that coupled simulation explains both:

- observed local cable deformation and motion; and
- the cube's measured translation or rotation.

This makes the user's desired correlation precise:

> If cable motion followed by a physically compatible object response is much
> better explained by a coupled model than by independent motion, confidence
> in contact increases.

Important invariants:

- 2D overlap is never sufficient contact evidence.
- Near-zero SDF gap is a proposal, not a decision.
- Sliding contact must not require tangential co-motion.
- Stationary near-touching bodies may remain fundamentally ambiguous.
- Missing visual evidence must not create new contact confidence.
- During occlusion, physics may propagate existing contact belief, but that
  prediction is not counted again as an independent measurement.
- Contact feedback must retain competing free/contact hypotheses until the
  temporal evidence separates them.
- Cable and cube image likelihoods must remain independent of the inferred
  contact factor to avoid self-confirming contact.

The initial result should report contact probability and one connected cable
arc interval. Multiple patches, cable-cable contact, and explicit
stick/slide modes can follow after binary contact works.

## 10. Online contact-aware control

Use the same reduced simulator inside a receding-horizon endpoint controller.

```text
RGB-D estimator (approximately 30 Hz)
              |
        contact belief
              |
reduced simulator + MPC/MPPI (approximately 10--20 Hz)
              |
 desired endpoint trajectories / tension-safe commands
              |
existing high-rate UAV position and attitude controllers
```

Candidate objectives include:

- cube target-pose error;
- acquire or retain the intended contact arc;
- avoid penetration and unintended collisions;
- maintain cable length and safe endpoint separation;
- avoid excessive tension or aggressive endpoint acceleration;
- smooth control;
- expected cost across free/contact hypotheses when contact is uncertain.

Gradient-based SQP/iLQR is attractive if the reduced contact model is smooth
enough. CUDA-batched MPPI is attractive when contact transitions remain
nonsmooth. The first controller can be quasi-static or slow dynamic, provided
that the paper states the operating regime honestly.

When contact confidence collapses, the controller must stop relying on a
contact-dependent prediction and execute an explicit reacquisition behaviour.
This contact-loss recovery is part of the research contribution, not merely a
software fallback.

## 11. Contact, force, and friction are separate claims

### Stage A: contact

Initially estimate:

- free/contact probability;
- contact onset and loss;
- cable arc interval;
- surface point and normal;
- uncertainty and whether inference is observation-supported or propagated.

### Stage B: model-implied contact wrench

The simulator will produce constraint impulses and an aggregate object wrench.
These are useful latent variables and control signals, but they should not be
reported as accurate Newtons merely because the solver returns numbers.

Metric force requires identifiable mechanics and calibration. Useful added
measurements include:

- endpoint tension sensors;
- calibrated UAV thrust or motor/IMU force estimates;
- a force plate or instrumented cube for ground truth;
- known cube-table support and friction.

RGB-D motion alone generally cannot distinguish cable force, table friction,
damping, and modelling error uniquely.

### Stage C: friction and stick/slide

Friction should be addressed after reliable binary contact. Static scenes do
not identify a unique friction coefficient. Informative experiments require
sliding or a stick-to-slip transition under known or estimable normal load.

The first paper should headline contact-aware manipulation. A calibrated
contact wrench may be secondary; friction estimation should be included only
if the available sensing makes it defensible.

## 12. Relationship to current implementation

The following current components remain valuable:

- native ZED HD1080 at 30 FPS and registered depth;
- the three-channel PIDNet contract: common cable body, endpoint set 1, and
  endpoint set 2;
- exact runtime/training mask configuration;
- skeleton graph, complete routes, visible graph edges, and ordered 3D lifting;
- current two-cable PF as a working tracking baseline;
- physical cable radius and swept watertight tube visualization;
- raw and robust refined cube pose, covariance, and temporal rigid-state
  filter;
- analytical oriented-cube distance, closest point, and normal queries;
- passive contact observer;
- synchronized recording/offline replay and timing infrastructure where
  available;
- asynchronous visualization.

These components should become measurements, initialization, baselines, or
interfaces for the twin. They should not dictate the new mechanical state or
online inference algorithm.

In particular:

- PIDNet observes visible pixels; it should not be expected to infer hidden
  mechanics.
- The skeleton graph is an observation, not the physical cable model.
- The rendered tube is currently visual geometry; it can inform the new
  rendering interface but should not be simulated vertex-by-vertex.
- The passive contact observer is useful evidence and a baseline, but the
  future contact state should arise from competing coupled/free dynamics over
  time.

## 13. Proposed implementation sequence

### Phase 0: preserve the current baseline

- Keep the live system operational.
- Record exact synchronized RGB-D, endpoints, cube state, cable estimates,
  configuration, timestamps, and endpoint/UAV commands.
- Do not introduce the new mechanics as live PF feedback yet.

### Phase 1: standalone high-resolution simulator

- Create an isolated interaction-twin module or tester.
- Implement a 64--128-node rod with measured cable geometry.
- Implement cube and table rigid-body mechanics.
- Implement finite-radius tube-to-SDF contact and action-reaction.
- Drive endpoints from recorded trajectories.
- Render predicted cable/cube depth and silhouettes for comparison.

### Phase 2: offline system identification

- Run the staged free-cable, cube-only, and interaction calibrations.
- Save parameter values, uncertainty, dataset identity, solver configuration,
  and random seed in a versioned model manifest.
- Confirm that calibrated parameters transfer to held-out motions without
  silently retuning each sequence.

### Phase 3: online reduced simulator

- Construct and re-identify a 24--32-node model.
- Batch dynamics and contact hypotheses on CUDA.
- Target real-time prediction and short-horizon rollouts before integration
  with live perception.

### Phase 4: hybrid fixed-lag estimator

- Use recorded replay first.
- Add a small free/contact hypothesis bank.
- Correct with logits, depth, endpoints, and cube pose.
- Publish state covariance and contact-mode probabilities.
- Compare against the existing PF on identical sequences, then choose one
  production architecture.

### Phase 5: contact-aware endpoint control

- Begin with slow planar cube manipulation.
- Demonstrate approach, contact acquisition, manipulation, deliberate
  occlusion, contact loss, recovery, and release.
- Integrate with aerial endpoints only after the estimator/controller is safe
  with recorded or benchtop endpoint trajectories.

### Phase 6: calibrated wrench and friction extensions

- Add tension/thrust sensing and force ground truth.
- Validate aggregate contact wrench.
- Introduce stick/slide modes and friction adaptation only under informative
  motion.

## 14. Minimal evaluation priorities

Avoid a large collection of weak metrics. The most important evidence is:

1. cable and cube prediction error during visible and controlled-occlusion
   intervals;
2. contact onset/loss and contact-arc accuracy against a small amount of
   explicit ground truth;
3. recovery time after deliberate contact loss or occlusion;
4. object-manipulation success and final cube pose error;
5. runtime distribution for estimator and control, not only average runtime;
6. paired ablation of independent visual tracking versus coupled physical
   inference on exactly the same recordings.

If metric force is claimed, force/wrench error against an external sensor is
mandatory.

## 15. ICRA novelty boundary

GPU simulation, a rendered deformable twin, or MPC alone is not sufficient
novelty. Related recent systems already combine real-time rod/deformable
simulation, visual correction, or differentiable contact-aware planning.

Relevant current references include:

- GaussTwin: <https://arxiv.org/abs/2603.05108>
- CORD-SLS: <https://arxiv.org/abs/2606.14188>
- DLO-Lab: <https://arxiv.org/abs/2606.04206>
- The Catenary Robot local paper:
  `C:\Users\wts28\Downloads\The_Catenary_Robot_Design_and_Control_of_a_Cable_Propelled_by_Two_Quadrotors.pdf`

The intended contribution is their combination in the specific unresolved
aerial-cable setting:

1. system-identified cable--rigid-object interaction dynamics from controlled
   RGB-D and endpoint-motion data;
2. probabilistic hidden-contact and contact-arc inference from joint cable and
   object response;
3. belief-aware two-endpoint control that acquires, exploits, and recovers
   physical contact.

## 16. Non-goals and methodological constraints

- Do not rebuild arbitrary-object geometry with TRELLIS or Gaussian splats when
  cable and cube geometry are already known.
- Do not make a dense cable surface mesh the stochastic estimator state.
- Do not call a CUDA spatial hash a contact model.
- Do not retain several permanent legacy estimators or silent fallbacks.
- Do not learn full black-box contact dynamics before calibrating the physical
  rod model.
- Do not infer strong contact from one frame, image overlap, or proximity
  alone.
- Do not force tangential sticking in the binary contact stage.
- Do not report correlation as force.
- Do not report force in Newtons without calibration and observability.
- Do not claim exact static friction without informative slip behaviour.
- Keep visualization asynchronous from tracking and control.
- Keep features isolatable for scientific ablation, but avoid duplicating
  production pathways solely for defensive fallback behaviour.

## 17. Open design choices to resolve experimentally

- 64 versus 128 offline rod nodes and 24 versus 32 online nodes;
- centreline-only bending versus full directors/torsion;
- implicit discrete elastic rod, Cosserat rod, or an equivalent constrained
  formulation;
- differentiable regularized contact versus a nonsmooth constraint solver;
- fixed-lag Gauss--Newton/SQP versus another continuous optimizer;
- SQP/iLQR versus CUDA MPPI for endpoint control;
- availability and type of endpoint tension or UAV force measurements;
- whether self-contact is necessary for the first manipulation task.

These choices should be made through small standalone experiments and spatial,
temporal, and runtime convergence—not by preserving the current PF design or
copying PhysTwin wholesale.

## 18. Copyable prompt for a future Codex task

```text
Continue the PF_cable research in:
C:\Users\wts28\Documents\PHD\particle_filter_cable_project

First read these files completely:
1. NEXT_GENERATION_CABLE_TWIN_HANDOFF.md — forward research architecture.
2. PIPELINE.md — authoritative description of current implementation.
3. CONTACT_FORCE_METHODOLOGY.md — earlier PF-centred methodology and useful
   background; it is superseded by the handoff where they conflict.

Important constraints:
- The working tree contains substantial intentional uncommitted work. Preserve
  it; do not reset, discard, or broadly reformat it.
- Do not assume the current 16-node/800-particle PF is the final architecture.
- Preserve the current system as a baseline and data-collection frontend.
- The forward design is an offline-calibrated high-resolution rod--rigid-body
  twin, reduced for online hybrid contact estimation and endpoint control.
- The cable has a physical 9 mm diameter. Use a finite-radius swept surface for
  collision; do not simulate a zero-thickness line or make every mesh vertex a
  latent state.
- Geometry proposes contact; temporal coupled cable/object response establishes
  confidence.
- Contact is the immediate target. Metric force and friction require later
  sensing and calibration.
- Prefer simple, direct, efficient, mathematically defensible CUDA
  implementations. Avoid silent fallbacks and parallel legacy pathways.
- Discuss major pipeline redesign decisions before integrating them into the
  live tracker.

Begin by inspecting the dirty tree and the relevant current interfaces. Do not
make unrelated changes.

The next requested task is: [INSERT THE SPECIFIC TASK HERE]
```
