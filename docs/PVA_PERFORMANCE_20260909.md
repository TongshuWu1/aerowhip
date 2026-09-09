# PVA performance work — 9 September 2026

User requested stopping the overnight work and optimizing fitting, training,
and planning. The fit stopped cooperatively at update **105**, retaining
`runs/adaptation/20260909-pva-M0-bootstrap/cable/residual_full_whip/progress.pt`
(current weights, best weights, optimizer and plateau state). Its selection
loss is 0.13627655661558855; it has **not plateaued**. The campaign
`runs/pva_campaign/20260909-M0-PVA` is stopped; its PPO never started.
The old thread heartbeat `finish-pva-research-workflow` is PAUSED.
No production fit, training or optimization was restarted in this work.

## Measured results

Windows, RTX 4080, PyTorch 2.11.0+cu128 / CUDA 12.8. Timings are synchronized
wall-clock measurements, not GPU utilization estimates. Reports and the
read-only benchmark programs are in `runs/audits/pva-performance/` and `tools/`.

| Operation | Reference | Accelerated, warm | Gain |
|---|---:|---:|---:|
| Five-flight full-whip cable loss + all NN gradients | 236.32 s | 3.13 s | 75.4× |
| Five-flight drone residual loss + all NN gradients | 0.249 s CPU | 0.119–0.123 s CUDA | about 2× |
| 1,024 PVA trajectories, one-second horizon, shared PPO/MPPI rollout | 4.57–4.59 s | 2.60–2.62 s | about 1.75× |

The cable comparison uses the saved update-105 weights and the **same** exact
0–1 s robust marker/tip loss and residual penalty as the production fitter.
Graph setup costs 2.43 s for cable and 4.89 s for drone. First replays are slower
than warmed replays. These measurements exclude optimizer/checkpoint I/O and
do not imply that every stage or complete PPO training speeds up by 75×.
The cable loss differs by about 1.0e-13, maximum position by 1.27e-12 m,
velocity by 4.03e-11 m/s, and NN gradient by 7.24e-13. Drone gradient difference
is 1.39e-12. PPO/MPPI packets/actions and termination outputs match; maximum
cable-position difference is 1.05e-12 m and reward difference 2.42e-11.

## Accepted implementation

- `CudaAutogradBlock` captures forward and recomputed backward/VJP blocks.
  Every recurrent call saves its own inputs, and backward reloads them before
  replay. NN parameters remain ordinary optimizer-owned tensors. It supports
  first-order derivatives; it does not claim higher-order differentiation.
- `CudaCableFit` advances the **whole** whip through three-step captured blocks.
  No state detachment, shortened objective, reduced substeps, float32 dynamics,
  changed material parameters, or extra/duplicated training flights are used.
- Cable damping geometry evaluates independent interior vertices together.
  Both legacy hard-frame and current smooth-frame equations are retained.
- A float64 NVRTC tridiagonal direct solver replaces general dense constraint
  factorization on the accelerated path. Its adjoint is checked against the
  dense reference. The existing cuSOLVER damping factorization remains in use.
- `CudaDroneFit` replays the existing weighted, batched translation recurrence
  and regularized loss on CUDA. Small SciPy nominal/attitude solves remain CPU
  work; forcing them onto a GPU was not justified by these measurements.
- The PVA environment captures complete physics/reward ticks, copies exact
  resolved delayed command events and durations, and retains per-row invalidity,
  first contact, hit/revocation, termination and export-boundary logic.
- PPO collects its six update metrics with one host transfer, preserving every
  finite check, optimizer update and KL stop decision.

New PVA settings/run snapshots record the performance flags. Explicit
`fused_ticks=False, fast_geometry=False, fast_solve=False` selects the reference
PVA execution for comparison. Historical force paths retain their existing
defaults and immutable source snapshots. The specialized constraint solver is
CUDA/float64 only; CPU PVA execution uses the tensor solver.

Larger tick graphs alone only improved rollout by about 2–4%. An experimental
custom damping Cholesky kernel was slower than cuSOLVER and was removed; its
benchmark result remains as an audit. The useful changes were batched vertex
geometry, graph replay for gradients and specialized constraint solves.

## Verification and limits

- **31 focused tests passed** across curvature values/gradients, recurrent graph
  reuse after parameter updates, direct solve/adjoint, fit stopping/hover
  conventions, PVA execution, actual modeled hits, failures, masks, resets,
  PPO optimizer/KL parity, export matching and UI integration.
- Full five-flight cable loss/gradient parity and CPU/GPU drone parity passed.
- Saved MPPI smoke rehearsal regenerated in the audit directory: complete CSV
  is byte-identical, drone prediction identical, maximum cable difference
  7.31e-12 m. It remains an integration smoke case, not a trained flight plan.
- Original stop/checkpoint, flight data, normalization, old policies and saved
  flight ghosts were not rewritten. Old artifacts are not relabeled.

This is a measured optimization pass, not proof of maximum possible GPU
throughput, numerical equivalence on every state, or Linux/other-GPU validation.
The five-flight dataset limits independent batch parallelism; increasing batch
size by duplicating flights would waste work. PPO batch size, rewards, learning
rate, task, numerical precision and convergence rules were not changed merely
to raise utilization. The saved model still needs its intended convergence and
essential coupled diagnostics before new production PPO. Resume requires a
separately recorded continuation using retained optimizer/best/stopping state;
do not overwrite the stopped fit or run the old campaign automatically.
