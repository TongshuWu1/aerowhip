> Historical document archived on 9 September 2026. For current work, read [HANDOFF.md](../../../HANDOFF.md). Old running-job and launch instructions below are historical.

# Current project context

The authoritative current handoff is [HANDOFF.md](../../../HANDOFF.md). It consolidates research intent, physical/state initialization, selected PPO, stopped SAC evidence, deployment semantics and unfinished hardware/adaptation work.

As of 6 September 2026:

- Windows development / RTX 4080, small offline deployment package on Ubuntu / RTX 5080. See [lab setup](../../LAB_SETUP.md).
- PPO is deliberately stopped at 229,376 attempts; the selected checkpoint and hash are in the handoff. SAC is deliberately stopped at 32,768. Do not restart the comparison queue automatically.
- Initial full drone/cable state → private simulated policy rollout → one frozen 20 Hz force sequence → position-controller recovery. No real-state/hit feedback changes the strike.
- Current calibration fixes EI/Cb and measured geometry while fitting lateral attachment position and cable drag. No neural residual is active. Protected `fig8vertical_002` remains untouched.
- The supplied ROS 2/Crazyswarm2 controller is a reviewed reference, not a flight executor. Offline package planning works separately from the unimplemented ROS tracking/controller bridge.
- New-run batch defaults are reduced to 1,024 for lab GPUs. Physics/reward/timing and immutable saved-run settings are unchanged; new-run update cadence differs from the old large-batch study.

The previous running decision log is retained in [history](../PROJECT_CONTEXT_BEFORE_LAB_TRANSFER_20260906.md). Its older requests, artifact locations and active-run descriptions are historical, not current instructions.
