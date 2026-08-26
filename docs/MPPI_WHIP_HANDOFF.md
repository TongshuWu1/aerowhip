# Drone–cable MPPI whipping: technical handoff

Paste this document into another GPT and ask it to act as a research collaborator.
The current goal is to analyze and improve the method, not to randomly replace working
components.

## 1. Research objective

We want a drone carrying a cable from one rigidly attached endpoint to strike a
specified target with the free cable tip. The command is

\[
g=(\mathbf p^*,\mathbf d^*,v^*),
\]

where \(\mathbf p^*\) is the target center, \(\mathbf d^*\) is the desired unit
impact direction, and \(v^*\) is the minimum directed tip speed. The desired behavior
is a dynamic cast/whip: the drone injects momentum, recoils or brakes, and the cable tip
overtakes it. The controller should not be told a desired cable shape or a prescribed
wind-up/release schedule.

The current experiment is a **matched-model numerical feasibility test**. The optimizer
and the independently constructed replay plant use the same identified cable model.
It therefore tests the trajectory parameterization, cost, constraints, optimizer, and
simulator consistency. It does not yet demonstrate robustness to model mismatch or
real flight.

## 2. Current physical and control model

- Cable: full discrete differential elastic rod (DDER-style) model using the current
  offline-identified homogeneous cable parameters. It includes bending stiffness and
  damping; the saved artifact is still marked provisional.
- Boundary: one cable endpoint is attached to the drone; the other endpoint is free.
- Drone in this test: a point mass commanded by a three-dimensional acceleration
  vector. Attitude, angular velocity, rotor dynamics, aerodynamic drag, and thrust
  allocation are not modeled yet.
- Physics rate: 100 Hz.
- Control rate: 50 Hz.
- Planning horizon: 3.0 s.
- Acceleration-vector limit: 20 m/s\(^2\).
- Drone-speed limit: 3 m/s.
- Full simulated cable state is available to the optimizer.

The optimized action is not one acceleration per physics frame. MPPI optimizes 16
low-frequency 3-D acceleration knots. They are linearly interpolated to 150 control
steps and vector-norm clipped to the acceleration limit. This keeps the search space
small enough for batched full-cable simulation without prescribing a casting phase.

## 3. Impact-event selection

For each sampled rollout, let \(\mathbf p_{\mathrm{tip},k}\) and
\(\mathbf v_{\mathrm{tip},k}\) be the free-tip position and velocity.

1. If the swept tip path enters the geometric target sphere, use its **first**
   continuous entry time between physics frames.
2. Otherwise, use the continuous closest point over every piecewise-linear
   tip interval.
3. Interpolate tip position/velocity, drone position, and event time at the
   selected event. The stored integer frame is only its upper bracket.

The current simulator has no rigid target-contact engine, so “contact” means entry into
the target sphere. `physical_tip_contact` therefore remains false even for a valid
geometric hit.

## 4. Rollout cost

Let

\[
d=\|\mathbf p_{\mathrm{tip},k^*}-\mathbf p^*\|,
\qquad
v_{\parallel}=\mathbf v_{\mathrm{tip},k^*}^{T}\mathbf d^*,
\]

and

\[
c=\frac{\mathbf v_{\mathrm{tip},k^*}^{T}\mathbf d^*}
{\|\mathbf v_{\mathrm{tip},k^*}\|+\epsilon}.
\]

The task cost is

\[
J_{\mathrm{pos}}=w_p\frac{d^2}{d^2+\sigma_p^2},
\]

\[
g_p(d)=\exp\!\left(-\frac{d^2}{2\sigma_v^2}\right),
\]

\[
J_{\mathrm{speed}}=w_v g_p(d)[v^*-v_{\parallel}]_+^2,
\]

and

\[
J_{\mathrm{dir}}=w_\theta g_p(d)
[\cos\theta_{\max}-c]_+^2.
\]

A valid strike requires all of the following:

- tip inside the target sphere;
- \(v_{\parallel}\ge v^*\);
- \(c\ge\cos\theta_{\max}\);
- zero encoded safety violation.

A valid safe strike receives

\[
J_{\mathrm{success}}=-C_s.
\]

The drone-displacement term is

\[
J_{\mathrm{disp}}=w_d
\|\mathbf p_{d,k^*}-\mathbf p_{d,0}\|^2.
\]

Weak action regularization is applied only through the selected event:

\[
J_u=w_u\sum_{k\le k^*}\|\mathbf u_k\|^2
+w_{\Delta u}\sum_{k\le k^*}\|\mathbf u_k-\mathbf u_{k-1}\|^2.
\]

The total cost is

\[
J=J_{\mathrm{pos}}+J_{\mathrm{speed}}+J_{\mathrm{dir}}
+J_{\mathrm{success}}+J_{\mathrm{disp}}+J_{\mathrm{safety}}+J_u.
\]

There is deliberately **no** positive cable-energy reward, bending-energy reward,
straightness reward, cable-shape target, explicit “whip” reward, predefined injection
phase, predefined release time, or phase boundary. The cast emerges from the simulated
cable dynamics and the terminal directed-impact task.

## 5. Safety cost and an important contact correction

The safety term is a weighted sum of normalized squared violations for:

- optional drone workspace excursion (disabled in the public objective);
- drone–target keepout;
- drone speed;
- ground and altitude limits;
- cable–drone clearance;
- actuator acceleration;
- a non-tip cable section entering the target before the free tip.

The non-tip check is performed on complete cable segments with conservative
between-frame collision detection and must be **strictly before** the selected
continuous tip event. The final cable segment may enter simultaneously with its
tip. Treating that simultaneous event as non-tip-first incorrectly rejects valid
strikes whenever target radius approaches the DDER node spacing.

## 6. MPPI configuration for the final refinement

- Samples per iteration: 256.
- CUDA rollout batch: 128.
- Iterations: 15.
- Acceleration knots: 16.
- Temperature: 1.0.
- Initial knot-noise standard deviation: 2.5 m/s\(^2\).
- Noise decay: 0.92 per iteration.
- \(\sigma_p=0.18\) m.
- \(\sigma_v=0.12\) m.
- \(w_p=40\).
- \(w_v=25\).
- \(w_\theta=20\).
- \(C_s=600\).
- \(w_d=2\).
- safety weight: 180.
- weak action effort and smoothness weights: \(10^{-5}\) each.

Antithetic Gaussian perturbations are sampled around the nominal knot sequence. An
exact unperturbed nominal candidate is included. Candidate weights are proportional to

\[
\exp(-(J_i-J_{\min})/\lambda).
\]

The best sampled trajectory is retained separately from the weighted nominal update.

## 7. Why target continuation was necessary

The final target-and-speed feasible set is narrow. Starting MPPI from zero control at
the final target produced occasional close trajectories that violated speed, workspace,
or non-tip-first requirements. The importance-weighted nominal sequence then drifted
away instead of preserving a useful strike.

We used numerical homotopy/continuation only to initialize the final optimization. This
is not a reward curriculum and does not insert phases into the motion.

First, a verified 3.37 m/s solution at target \((0.48,0,1.30)\) m was continued through

\[
(0.53,0,1.32),
(0.58,0,1.34),
(0.63,0,1.36),
(0.68,0,1.38),
(0.74,0,1.39),
(0.80,0,1.40),
\]

with a 3.0 m/s minimum directed-speed requirement. This produced a verified 4.00 m/s
strike at \((0.80,0,1.40)\) m.

That trajectory was then continued to the final task using:

| Stage | Target position (m) | Minimum directed speed (m/s) |
|---:|---|---:|
| 1 | (0.84, 0, 1.40) | 4.0 |
| 2 | (0.88, 0, 1.40) | 4.2 |
| 3 | (0.92, 0, 1.40) | 4.5 |
| 4 | (0.96, 0, 1.40) | 4.8 |
| 5 | (1.00, 0, 1.40) | 5.0 |

Each continuation stage used 6 iterations, 128 samples, 16 knots, 3.5 m/s\(^2\)
initial noise, and 0.90 noise decay. The final target was then independently refined
with the settings in Section 6. Only the final target defines the reported result.

## 8. Final verified result

Initial drone position:

\[
(0.00,0.00,1.50)\ \mathrm{m}.
\]

Final target:

\[
(1.00,0.00,1.40)\ \mathrm{m},
\]

so the target is 1.00 m forward and only 0.10 m lower. Desired impact direction is
world \(+x\), target radius is 50 mm, maximum impact angle is 35 degrees, and required
directed speed is 5.0 m/s.

Independent matched-model replay:

- valid geometric tip-first strike: yes;
- impact time: 0.72 s;
- target error: 15.46 mm;
- directed tip speed: 5.682 m/s;
- total tip speed: 6.229 m/s;
- direction error: 24.20 degrees;
- maximum forward drone sweep: 0.690 m;
- drone x-position at impact: 0.444 m;
- backward recoil from peak forward position: 0.246 m;
- drone displacement at impact: 0.741 m;
- maximum drone excursion: 0.862 m;
- maximum drone speed: 2.546 m/s;
- total tip-speed / maximum drone-speed ratio: 2.45;
- minimum drone–target distance: 0.538 m;
- maximum acceleration: 19.69 m/s\(^2\);
- encoded safety violations: zero;
- independent replay position and velocity difference: exactly zero numerically.

The motion is a forward sweep followed by backward recoil while the cable tip continues
forward. It can look like a “backward whip,” but the signed motion confirms that the
recoil transfers momentum to the free tip.

## 9. Crucial limitations and cautions

1. **Geometric contact only.** A target sphere is used; there is no rigid collision,
   impulse, or post-impact dynamics.
2. **Matched model.** Controller and replay use the same cable artifact. Zero replay
   mismatch validates implementation consistency, not real-world prediction accuracy.
3. **Point-mass drone.** Attitude, body torque, thrust direction, motor lag, and rotor
   saturation are absent. The current acceleration sequence is not yet a directly
   flyable command.
4. **Open-loop solve.** This is a solve-once trajectory feasibility result, not yet
   real-time receding-horizon MPPI.
5. **Full-state assumption.** The optimizer currently receives the simulated full cable
   state. Real observation/estimation and model adaptation are separate future stages.
6. **Small non-tip margin.** Minimum pre-impact non-tip target clearance is 50.61 mm
   for a 50 mm target radius—only about 0.61 mm margin. The strike is valid in this
   discretization but fragile to model, contact, and state-estimation error.
7. **Near actuator limit.** Peak acceleration is 19.69 m/s\(^2\) under a 20 m/s\(^2\)
   limit. Real drone feasibility must be checked with full thrust/attitude dynamics.
8. **Continuation dependence.** The final solution is valid, but cold-start reliability
   is not established. Report continuation transparently and evaluate multiple seeds.

## 10. Recommended research discussion

Treat this as a working nominal optimal-control baseline. The next defensible steps are:

1. Replace geometric target entry with physical tip/target contact and report impulse or
   delivered momentum.
2. Replace point-mass acceleration control with rigid-body quadrotor dynamics and
   thrust/body-rate or motor commands.
3. Test final-target cold starts, multiple MPPI seeds, and continuation ablations.
4. Add model mismatch only after the nominal controller remains reliable: perturb cable
   stiffness, damping, mass, attachment transform, and aerodynamic drag.
5. Introduce state estimation and online adaptation separately, with clear nominal,
   oracle, and adapted baselines.
6. Preserve the simple terminal task objective unless a specific failure mode provides
   evidence for an additional term. Do not add vague “whipiness” or energy rewards just
   to make the animation look more intuitive.

## 11. Distributed-motion diagnostic and controlled ablations

The verified replay now has a post-hoc node-by-time diagnostic. It computes node speed
and kinetic energy relative to the drone, physical DER curvature magnitude, and the
time of each node's peak relative speed through impact. None of these quantities is
added to the MPPI cost.

Run:

```powershell
C:\Users\wts28\env_isaaclab\Scripts\python.exe `
  -m research_tools.mppi_propagation
```

For the current verified strike, the drone reaches its peak forward position and
begins recoil at 0.47 s, the tip strikes at 0.72 s, and 79.7% of cable kinetic energy
relative to the drone is in the distal quarter at impact. The node-speed timing and
curvature heat maps support a stroke--reversal--distal-lash interpretation. This is
descriptive evidence of distributed dynamics, not an identified material wave speed.

The ablation runner checkpoints every solve, preserves the baseline model hash, stores
the complete replay for every run, and never mixes different compute budgets in one
summary row:

```powershell
# Fast software-path check; never use one seed as statistical evidence.
.\.venv\Scripts\python.exe -m research_tools.mppi_ablation `
  --study all --profile smoke

# Multi-seed pilot.
.\.venv\Scripts\python.exe -m research_tools.mppi_ablation `
  --study initialization --profile pilot `
  --output-dir data/drone_mpc/ablations/mppi_discovery_pilot

# Twenty paired seeds at the final MPPI budget. This is deliberately expensive.
.\.venv\Scripts\python.exe -m research_tools.mppi_ablation `
  --study initialization --profile discovery `
  --output-dir data/drone_mpc/ablations/mppi_discovery
```

The studies are:

- horizons 1.0, 1.5, 2.0, and 3.0 s at approximately fixed 0.20 s knot spacing,
  initialized by the time-preserved verified trajectory;
- zero, random smooth, forward--recoil, backward--forward, lateral, and verified
  continuation initializations on the same fixed 2.0 s/11-knot task. MPPI sampling
  seeds are paired across strategies;
- the current 0.12 m proximity gate, a 0.24 m gate, and the current gate plus a weak
  far-range directed-speed-deficit cue, all from zero initialization.

The weak predictive cue is an ablation only. Its default weight is exactly zero, and
the main controller cost remains the formulation in Section 4.

The three-seed discovery pilot used six MPPI iterations and 128 samples per iteration.
Zero, random smooth, backward--forward, and lateral initializations each succeeded in
0/3 trials. Forward--recoil and continuation each succeeded in 3/3. Forward--recoil
first sampled a valid plan at iterations 2, 2, and 4; continuation did so at iteration
1 in every trial. Forward--recoil's median directed speed was 6.23 m/s and median
non-tip clearance margin was 4.4 mm, compared with 5.64 m/s and 1.4 mm for
continuation. With only three paired seeds, Wilson confidence intervals remain wide;
these are pilot results supporting the local-basin hypothesis, not final success-rate
estimates.

Generate the six-panel discovery figure after a pilot or final run with:

```powershell
C:\Users\wts28\env_isaaclab\Scripts\python.exe `
  -m research_tools.mppi_discovery_plot `
  --runs data/drone_mpc/ablations/mppi_discovery_pilot/runs.csv `
  --profile pilot `
  --output data/drone_mpc/ablations/mppi_discovery_pilot/discovery_pilot.png
```

The one-seed/two-iteration smoke run verified all paths. It retained feasible strikes
at 1.0, 2.0, and 3.0 s. The 1.5 s time-resampled seed entered the target at 42.5 mm and
5.06 m/s but was invalid because a non-tip node entered the target region first. Zero
and random cold starts remained far from success. The weak predictive cue improved the
selected cold-start direction and directed speed in that tiny budget but did not
improve target distance. These observations are hypotheses for the pilot, not final
comparative claims.

When discussing this project, separate three claims:

- **trajectory feasibility** under the nominal simulator (shown here);
- **closed-loop robustness** under model/state errors (not yet shown);
- **real-drone executability and target impact** (not yet shown).
