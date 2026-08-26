# Accelerated online DDER–MPPI and between-strike adaptation

`run_online.py` launches the research UI in
`drone_mpc.receding_mppi_gui`. The normal online path now requires CUDA
full-horizon capture and the fused CUDA MPPI objective. It does not expose the
former step-by-step reference path.

## Acceleration tiers

Every fixed node count supported by the identified cable artifact uses:

- one captured CUDA graph for the complete prediction horizon;
- one maximally parallel candidate batch per MPPI iteration;
- the arbitrary-node fused CUDA contact/cost evaluator.

The 11-node controller additionally uses the topology-specialized fused
Kelvin–Voigt damping and four-position-plus-one-velocity projection kernels.
This is the maximum-throughput tier and remains the UI default. Other node
counts use the same DDER equations and the captured full-horizon runtime, with
generic DDER mechanics inside the graph.

The graph-captured arbitrary-node rollout was compared against the former
per-step runtime at 6, 11, 15, and 21 nodes. Maximum position and velocity
differences were exactly zero for the deterministic test trajectories. The
arbitrary-node fused cost was also checked against the PyTorch reference.

For the saved `11node_tru_phys.json` workload (11 nodes, 0.7 s horizon, 50 Hz,
2,048 samples, two MPPI iterations), the production optimizer measured a
median warmed update of 80.5 ms on the current RTX 4080 system.

## Online adaptation semantics

The UI adaptation tab controls a persistent
`BetweenStrikeAdaptationSession`. Its scheduling contract is:

1. Snapshot one immutable accelerated controller runtime.
2. Run the complete strike without changing EI or Cb.
3. Convert the realized distributed cable trajectory into adaptation
   observations.
4. Apply the existing mismatch, excitation, and information gates.
5. Fit EI and Cb only when the data justify fitting.
6. Accept a candidate only after held-out prediction improves.
7. Build and prewarm a fresh accelerated runtime.
8. Atomically publish the complete runtime for the next strike.

Fitting never runs inside an MPPI update. Changing cable model, node count,
simulation timing, or simulated truth starts a new adaptation session. The UI
shows the published generation and EI/Cb ratios. Its Adaptation tab also plots
the absolute relative EI and Cb error against the hidden simulated truth after
every completed strike, beginning with the nominal model at strike zero. A
separate numeric readout reports the symmetric joint log-parameter error and,
when fitting is attempted, held-out all-node position RMSE before and after the
candidate update.

The truth-error plot is deliberately labeled as a simulation diagnostic. A
physical OptiTrack experiment does not reveal true EI or Cb, so that deployment
must use held-out prediction residual rather than parameter error as its online
model-quality signal.

The current UI observation source uses exact distributed state from the
simulated plant. This validates online orchestration, not physical sensing.
The Motive/OptiTrack path should later provide the same observation contract
after a causal cable-state estimator has been validated.

## Regression evidence

- Complete repository suite: 202 tests passed.
- Matched simulated plant: no trigger, no fit, no parameter drift.
- Simulated truth ratios EI=0.8 and Cb=0.7: one first-strike fit was accepted,
  prewarmed, and published; the first update produced EI=0.868 and Cb=0.849.
- The fitted runtime remained outside the active strike and became visible only
  at the next solve boundary.
