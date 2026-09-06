# Project instructions for the next coding agent

Read `HANDOFF.md` first; it is the current research and deployment decision record.

- Preserve the user's selected PPO, saved experiment configurations, active physical calibration and original measurements. Keep the five-page workflow and 20 Hz open-loop strike semantics unless the user requests a methodological change.
- Do not inspect, evaluate, plot, fit or tune against the protected `fig8vertical_002` recording. Copying it opaquely for the authorized private lab transfer is permitted.
- PPO and SAC were intentionally stopped. Do not restart training/supervisors automatically. The selected checkpoint hash is in the handoff.
- Offline planning/export is not flight authorization. No ROS flight sender is implemented or validated. Ask for actual vehicle/interface details rather than inventing frames, firmware parameters or topic names.
- Preserve raw logs, failed trials and immutable study snapshots. Do not change rewards, physics or report selected validation as independent paper evidence during deployment work.
- Do not reset the working tree or rewrite Git history as cleanup. Superseded artifacts were moved outside the repo; permanent deletion was blocked by approval review, so do not retry via another mechanism.
- Prefer targeted existing tests and record what hardware/OS was actually tested. Do not claim Ubuntu/4080/5080 validation from a Windows/5090 run.
