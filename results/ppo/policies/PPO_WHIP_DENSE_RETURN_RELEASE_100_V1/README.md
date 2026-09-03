# PPO whip — dense return/release candidate

Status: **retained trade-off candidate; not selected for deployment**.

The final checkpoint was trained for 501,760 episodes from the preceding
return/release pilot. Relative to the selected D50 controller, it adds bounded
dense potential credit for forward loading, target-axis return, backward UAV
motion, forward attachment-relative cable motion, and simultaneous strike
quality. Release credit is evaluated at impact.

Final deterministic held-out validation:

- 474/512 = **92.58%** success.
- Median directed speed: 5.20 m/s.
- Median direction error: 21.72 degrees.
- Mean maximum UAV displacement: 0.915 m.
- Mean terminal UAV displacement: 0.806 m.
- Mean successful return before impact: 0.119 m.

The return-oriented checkpoint at `checkpoints/best_return.pt` reached 91.02%
success with 0.121 m mean return. `checkpoints/terminal.pt` has the better
overall success/return balance. Neither replaces the selected D50 controller.
