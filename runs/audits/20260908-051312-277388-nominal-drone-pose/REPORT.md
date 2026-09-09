# Nominal drone pose engine probe

All three phase-aware legacy whips ran through the new engine using fixed, deliberately unfitted example parameters on NVIDIA GeForce RTX 4080. This checks numerical execution and the data connection; it is not a preliminary model fit or a flight-performance validation.

Future measured pose/attachment is kept separate from prediction inputs. Initial state and frozen effective compensation come from pre-hover data only. The effective orientation alignment is a pre-hover reference, not verified firmware mounting.

The model predicts tracked-origin P/V and tracked-frame R/omega, then computes attachment P/V/A with all rigid rotation terms. It holds original logged commands with their effective delay and freshness intervals. There is no extra gravity or cable force in effective translational acceleration. No PPO, residual training, vehicle control or active configuration change occurred.
