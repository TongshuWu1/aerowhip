# PPO whip — selected D50 forward/reverse controller

Status: **selected simulation controller**.

This folder is a self-contained tracked copy of the completed run
`whip_ppo_forward_reverse_release_v1/2026-09-02T174750.625177Z`.

Key deterministic held-out validation results:

- 480/512 = **93.75%** task success.
- Median tip error: 30.44 mm.
- Median directed speed: 5.10 m/s.
- Median direction error: 21.29 degrees.
- Median strike time: 0.88 s.
- Mean maximum UAV displacement: 0.916 m.
- Mean terminal UAV displacement: 0.829 m.
- Mean successful forward excursion: 0.625 m.
- Mean return before impact: 0.0966 m.
- Mean UAV target-axis velocity at impact: -1.445 m/s.
- Mean attachment-relative cable-tip target-axis speed at impact: 6.580 m/s.
- Numerical failures: 0.

`checkpoints/terminal.pt` is the policy selected by the desktop Run & Replay
page. The 512 validation states are distinct physically propagated held-out
states; the target and physics parameters are canonical/nominal.

This is simulation-only and is not a hardware-ready policy.
