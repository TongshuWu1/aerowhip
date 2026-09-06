# One-shot strike execution (updated)

The live strike uses the initial state only. A background model rollout queries
PPO on predicted states and saves a finite force sequence, ending at its first
predicted valid hit. Actual execution consumes those commands once and resumes
PID when the sequence ends. Live motion and simulated hit diagnostics never feed
back into force selection or stopping. A prediction without a valid hit is not
executed. There is no automatic retry or looping.

Recordings include the initial state, planned force sequence, actual simulated
motion and contact outcomes. Previous exploratory recordings and their
checkpoint-specific measurements were cleared on 2026-09-05.

The continuous flight implementation retains position PID, the existing
point-mass/cable dynamics, fixed physics steps, and Windows timer pacing. The
standalone recorded policy rollout also truncates at its predicted hit rather
than adding the remaining seven-second episode tail.

Policy queries occur during planning. Execution consumes the frozen sequence;
contact diagnostics are recorded without changing its force commands or cutoff.
