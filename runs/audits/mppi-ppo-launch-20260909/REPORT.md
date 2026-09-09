# MPPI at the PPO starting and target positions

Requested coordinates saved in `config/pva/mppi.json`:
start[-2,0,1.255]m, target[-1,0,1.1]m. Exact match to current PPO central
positions. Only these two vectors changed; PPO config hash is unchanged.
MPPI keeps a2s horizon,1024 random GPU candidates plus the proposal mean,
historical normalized M1, and the same rewards, limits and sampling settings.

Run `runs/mppi_pva/20260909-120300-901531` completed31 iterations in162.43s on
Windows/RTX4080. Best saved reward298.4565267, modeled valid hit, closest
tip-target distance0.04400159m. Reward plateau stop; no live worker remains.

Full rehearsal was attempted and rejected before any CSV was written. The
current curved recovery planner found no return inside the saved reference
limits for this exit. This is a successful simulated strike, not a verified
complete command sequence. No rehearsal, CSV or portable ZIP was produced.

Exit command at1.56667s: position[0.631573,0.181261,1.558478]m,
velocity[4.883248,1.039406,0.038778]m/s,
acceleration[5.295830,1.570226,-1.046266]m/s². The speed is near the5m/s
ceiling while acceleration is still forward. The failed search does not prove
that all possible recovery constructions are infeasible. No constraint was
relaxed and the saved whip was not edited.

Evidence: `result.json`, `setup_check.json`, `recovery-boundary.json`,
`optimization.log` and `rehearsal.log`. Earlier runs retain their original
coordinates and artifacts. Fitting/PPO remain stopped; no physical flight.
