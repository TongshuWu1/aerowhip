# UAV cable-tip targeting: visual paper summary

## Project overview

**Goal:** hit the target using cable motion while reducing UAV travel.

**Pipeline:** PPO desired 3D force -> dynamic 159 g root -> DDER cable -> tip strike.

**Cable:** 12-node elastic rod with identified bending/damping; the new dynamic root includes cable reaction.

**State:** full simulated 3D root and cable state will be available to PPO; the final observation design is not frozen.

**Success gate:** tip first within 5 cm, actual world-frame tip velocity projected on the desired strike direction >=4 m/s, and tip velocity within 45 degrees of that direction; explicit angle reward off and cable tangent diagnostic only.

## Paper 01. Learning a whip strike

**Deep Reinforcement Learning of Robotic Manipulation for Whip Targeting**
Bai et al. | SIMPAR 2025 | Simulation

[Paper](https://doi.org/10.1109/SIMPAR62925.2025.10979109) | [Full text](https://findresearcher.sdu.dk/ws/files/290228125/Deep_reinforcement_learning_of_robotic_manipulation_for_whip_targeting.pdf)

- **Goal:** Teach a robot arm to hit a target with a whip.
- **Method:** PPO learns from trial rewards; PD servos turn requested joint angles into torques.
- **How it works:** Tip-to-target position error -> Policy chooses joint angles -> PD servos move the arm.
- **Cable:** 25-link spring-damper chain; 50 rotational degrees of freedom. MuJoCo.
- **Full state?** NO - policy sees tip error only. Arm joints feed PD; cable shape and velocity are hidden.
- **Takeaway:** The simulator knows the cable; the policy does not.

## Paper 02. One smooth movement, nine parameters

**Learning to manipulate a whip with simple primitive actions - A simulation study**
Nah et al. | iScience 2023 | Simulation

[Paper](https://doi.org/10.1016/j.isci.2023.107395) | [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10405071/)

- **Goal:** Hit whip targets by optimizing a smooth arm swing.
- **Method:** A motion primitive defines the swing. Impedance acts like a spring and damper around it.
- **How it works:** Choose endpoints and duration -> Execute the smooth swing -> DIRECT-L refines it without gradients.
- **Cable:** 25-segment flexible chain; 50 rotational degrees of freedom. MuJoCo.
- **Full state?** NO online cable feedback. Arm joints are sensed; simulated cable nodes score each trial.
- **Takeaway:** A small search space simplifies control, while the simulator still models the whole cable.

## Paper 03. Improve the next attempt

**Iterative Residual Policy: for Goal-Conditioned Dynamic Manipulation of Deformable Objects**
Chi et al. | Robotics: Science and Systems 2022 | Real robot

[Paper](https://arxiv.org/abs/2203.00663) | [Full text](https://irp.cs.columbia.edu/irp_2022.pdf)

- **Goal:** Adapt a rope strike to different ropes using repeated trials.
- **Method:** IRP learns how a small action correction changes the observed tip trajectory.
- **How it works:** Camera records the tip path -> Predict candidate action corrections -> Execute the best next attempt.
- **Cable:** 25 linked capsules; randomized length and density. MuJoCo.
- **Full state?** NO - 2D tip trajectory from a 60 fps RGB camera. No full cable shape or velocity.
- **Takeaway:** Corrections happen between attempts, not during a strike.

## Paper 04. Learn the right state at the right event

**Learning Dynamic Rope Manipulation Using Task-Level Iterative Learning Control**
Suresh and Atkeson | 2026 preprint v2 | Real robot

[Paper](https://arxiv.org/abs/2602.21302v2) | [Full text](https://arxiv.org/html/2602.21302v2)

- **Goal:** Learn a flying knot from one human demonstration.
- **Method:** ILC converts measured event error into a command correction for the next trial.
- **How it works:** Measure rope state near collision -> Use a model gradient and constrained QP -> Update the smooth robot command.
- **Cable:** 11-link point-mass chain: fixed lengths, bending stiffness and damping. No collision model.
- **Full state?** PARTIAL, distributed sensing: 11 motion-capture markers and derived velocity. Tracking fails after contact.
- **Takeaway:** Measure the rope across its length; learn from errors at the critical event.

## Paper 05. Throwing with a flying base

**Learning to Throw: Agile and Accurate Cable-Suspended Payload Delivery with a Quadrotor**
Zhai et al. | June 2026 preprint v1 | Real flights

[Paper](https://arxiv.org/abs/2606.27603) | [Full text](https://arxiv.org/html/2606.27603v1)

- **Goal:** Swing and release a suspended payload toward a landing target.
- **Method:** Asymmetric PPO gives the training critic more state information than the deployed actor.
- **How it works:** UAV state + payload position history -> Choose thrust, rates and release -> Coupled physics advances the system.
- **Cable:** 15 passive-jointed segments in PhysX. Cable reaction affects the UAV; physics runs at 500 Hz.
- **Full state?** NO - actor has 35 inputs, not all segment states. Critic adds payload velocity. State sensors unspecified.
- **Takeaway:** Detailed coupled physics can train a controller with partial observations.

## Paper 06. Agile flight from endpoint observations

**FLARE: Agile Flights for Quadrotor Cable-Suspended Payload System via Reinforcement Learning**
Cao et al. | IEEE RA-L 2026; preprint v2 | Real flights

[Paper](https://arxiv.org/abs/2508.09797v2) | [Full text](https://arxiv.org/html/2508.09797v2)

- **Goal:** Move a suspended payload through targets and gates.
- **Method:** PPO learns a direct observation-to-command mapping from simulated rewards.
- **How it works:** UAV state + payload angles + goal -> Neural policy chooses thrust and body rates -> Onboard feedback at 100 Hz.
- **Cable:** Serial chain of rigid links in Genesis; flexibility comes from the joints.
- **Full state?** NO - endpoint geometry, not distributed cable shape or velocity. Real flights use OptiTrack.
- **Takeaway:** Endpoint swing information may suffice for payload tasks; whip waves are a different challenge.

## Paper 07. Better rod physics, transferable motion

**DeformX: A Versatile Co-Simulation Framework for Deformable Linear Objects**
Yang et al. | 2026 preprint; author-listed IROS 2026 | Real replay

[Paper](https://arxiv.org/abs/2606.22116) | [Full text](https://arxiv.org/html/2606.22116v1)

- **Goal:** Transfer learned rope-swinging commands to a real UR5e.
- **Method:** Cosserat rods track centerline and cross-section orientation; deformation creates forces and moments.
- **How it works:** Isaac Sim supplies robot motion -> Rod solver advances with finer time steps -> Return reaction forces to rigid bodies.
- **Cable:** Stretch, shear, bend, twist and contact. Specialized Cosserat-rod solver coupled to Isaac Sim.
- **Full state?** NO - PPO sees robot and tip data: 18 inputs. Hardware replays commands without online tip feedback.
- **Takeaway:** Full rod simulation; partial policy observations; open-loop real execution.

## Paper 08. Identify the cable mechanics

**Accurate Simulation and Parameter Identification of Deformable Linear Objects using Discrete Elastic Rods in Generalized Coordinates**
Chen, Bretl and Pham | IROS 2025; arXiv v3 | Physical rod tests

[Paper](https://arxiv.org/abs/2310.00911v3) | [Full text](https://arxiv.org/html/2310.00911v3)

- **Goal:** Fit and simulate bending and twisting stiffness.
- **Method:** DER computes forces from elastic-energy gradients: deformation stores energy, which drives restoring forces.
- **How it works:** Read the cable node geometry -> Calculate bending and twisting forces -> Convert forces into MuJoCo joint torques.
- **Cable:** Inextensible elastic rod; quasistatic twist. Stretching and shearing are neglected.
- **Full state?** SIMULATOR: yes. No feedback policy. Depth-camera geometry supports fitting; no online dynamic estimator.
- **Takeaway:** A mechanics and identification method; full-state control is not demonstrated.

## Paper 09. Learn physical uncertainty from vision

**A Distributional Treatment of Real2Sim2Real for Object-Centric Agent Adaptation in Vision-Driven Deformable Linear Object Manipulation**
Kamaras and Ramamoorthy | IEEE RA-L 2025; arXiv v4 | Real robot

[Paper](https://arxiv.org/abs/2502.18615v4) | [Full text](https://arxiv.org/html/2502.18615v4)

- **Goal:** Adapt deformable-object reaching to different stiffness and length.
- **Method:** BayesSim infers plausible parameter distributions; PPO trains across samples from those distributions.
- **How it works:** Observe RGB keypoint trajectories -> Infer parameter uncertainty -> Randomize parameters and train PPO.
- **Cable:** Corotational finite elements in Isaac Gym/FleX; Young's modulus controls elastic response.
- **Full state?** NO - 12 inputs: end-effector position plus five 2D image points. No complete 3D state or velocity.
- **Takeaway:** Estimating material parameters does not reveal the cable's instantaneous state.

## Paper 10. Predict with fewer cable variables

**Nonlinear Predictive Control of the Continuum and Hybrid Dynamics of a Suspended Deformable Cable for Aerial Pick and Place**
Rapuano et al. | ICRA 2026; preprint v1 | Simulation

[Paper](https://arxiv.org/abs/2602.17199) | [Full text](https://arxiv.org/html/2602.17199v1)

- **Goal:** Track a cable tip through payload pickup and release.
- **Method:** POD compresses deformation into modes. MPC predicts motion, optimizes commands, then replans.
- **How it works:** Project cable shape into modal coordinates -> Optimize future UAV acceleration -> Apply first command; repeat at 40 Hz.
- **Cable:** Extensible tension string; no bending stiffness, internal damping or self-contact.
- **Full state?** REDUCED state assumed available: boundaries, modes and rates. No real observer validated.
- **Takeaway:** A reduced controller state still needs a practical sensing solution.
