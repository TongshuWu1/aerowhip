# M0 MPPI accuracy comparison

Exploratory single-seed simulation comparison. No physical improvement or statistical significance is established.

All searches use the same frozen M0, launch, target position, spline representation, seed (657), and limits. The existing saved initialization is identical across searches; the runs do not resume the previous optimizer state. The minimum braking duration is 1.3 s. No active command or manuscript was changed.

The stronger target reward changes contact weight 450 -> 900, miss weight 300 -> 1200, and proximity scale 0.35 -> 0.10 m. Other reward weights stay unchanged. The final case additionally uses the user-approved 2 cm planning sphere instead of 5 cm; physical success remains actual contact with the real target.

| Case | Planning tolerance (cm) | Closest predicted distance (cm) | Full recovery |
|---|---:|---:|---|
| Current M0 command | 5 | 4.289 | passed |
| Original reward, 512 x 40 | 5 | 4.539 | passed |
| Target reward, 512 x 40 | 5 | 4.089 | passed |
| Target reward, 512 x 80 | 5 | 4.462 | passed |
| Target reward, 1024 x 40 | 5 | 4.615 | passed |
| Target reward + 2 cm tolerance, 512 x 40 | 2 | 1.773 | passed |

Distances use the same [0, 1.5] s interval of each complete saved command and swept linear interpolation between 150 Hz tip predictions. The simulations omit target contact forces, so these are free-motion predictions. This avoids ranking candidates using their differently truncated optimizer distance alone.

| Search | Random candidate rollouts including initialization | Updates | Runtime (s) |
|---|---:|---:|---:|
| Original reward, 512 x 40 | 20992 | 40 | 119.6 |
| Target reward, 512 x 40 | 20992 | 40 | 118.8 |
| Target reward, 512 x 80 | 41472 | 80 | 231.9 |
| Target reward, 1024 x 40 | 41984 | 40 | 163.7 |
| Target reward + 2 cm tolerance, 512 x 40 | 20992 | 40 | 118.8 |

The 512 x 80 and 1024 x 40 cases each evaluate 40,960 random update candidates. Initialization and deterministic candidates add slightly different overhead. All searches stopped at their declared iteration budgets, not a demonstrated convergence threshold. Each selected plan passed independent batch-one outcome and score checks.

The encounter-distance part of the reward is constant at the sphere radius for successful tip entries. Higher reward can therefore reflect shape, speed, or command cost rather than tighter aiming. Additional random seeds would be needed to establish a reliable sample-count versus iteration-count ranking.

Candidate CSV: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\exports\M0_MPPI_2cm_brake_1p3s\fullstate_30hz.csv`

Candidate full duration: 7.933 s. Commanded braking stop: [-1.1727769287657455, -0.1487431576696609, 1.987239615050687] m.
