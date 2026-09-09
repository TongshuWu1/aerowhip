> Historical document archived on 9 September 2026. For current work, read [HANDOFF.md](../../../HANDOFF.md). Old running-job and launch instructions below are historical.

# Longer-horizon MPPI experiment

Superseded task interpretation: the user subsequently clarified that the drone
must pull forward then move backward before contact. The run below was still
moving forward at contact and does not demonstrate that requested whip. Its
old tip-hit result/provenance is preserved. Current work:
docs/MPPI_PULLBACK_20260909.md.

The user authorized broad MPPI improvements and longer horizons after the
corrected rolling 0.4s controller missed by 0.51465m. All work is historical
normalized M1 simulation. Fitting/PPO remain stopped; no physical flight.

## Candidate comparisons

`runs/audits/mppi-longer-horizon-20260909/result.json` compares cold windows
from identical launch [-2,0,1.255] to target [-1,0,1.1], with 1024 random
candidates plus mean, noise 0.05, correlation 0.7, temperature 1, no extra
zero-jerk prior and zero geometric terminal guidance. Original task rewards,
model, jerk bounds and hit/feasibility criteria remain unchanged. Window
stopping is plateau (minimum20, patience15), without iteration/wall-clock cap.

| Lookahead | Predicted valid hit | Closest tip | Hit time | Task return | Compute |
|---|---|---|---|---|---|
| 1.2s | Yes | 0.029155m | 1.054698s | 305.3885 | 64.89s / 21 iterations |
| 2.0s | Yes | 0.043642m | 1.683959s | 297.8079 | 119.44s / 23 iterations |

These are optimized candidate windows, not committed receding maneuvers. This
is a single seed comparison, not statistical evidence of universal superiority.
The 1.2s candidate's curved recovery reached 3.08374m command height, exceeding
the unchanged 2.8m limit; no flight CSV was exported for that candidate.

## Rolling diagnostic: successful simulation

`runs/audits/mppi-longer-rolling-20260909/result.json` records the completed job
`20260909-131121-677417`.
It uses 1.2s/36-action lookahead with the above sampling, one committed 30Hz
action per replan, and unchanged 5s maneuver limit. The first window uses
minimum20/patience15; subsequent shifted windows use minimum3/patience3.
This avoids both premature cold-start stopping and repeating cold-start effort
at every command. Neither is a hard iteration or wall-clock budget.

All prior geometric terminal guidance is disabled. New explicit planner costs
apply only to a predicted successful strike's last command: 0.5*speed^2,
5*positive_vertical_speed^2 and 1*positive_acceleration_along_velocity^2.
They favor a manageable recovery handover. Actual task rewards and hit labels
are unchanged. They are heuristic costs, not a recovery feasibility proof;
complete command/recovery/model replay checks must still pass before export.

It achieved a modeled valid hit at 1.1266667s, closest tip 0.0235112m, task
return304.37965, no modeled constraint failure. It committed34 actions and
stopped at1.133333s after168 optimizer iterations in381.845s (6.36 minutes).
The previous0.4s miss took971.71s; this is a comparison of these specific runs,
not a general algorithm speed benchmark. The longer run also ends sooner
because it hits instead of reaching the5s maneuver limit.

Active config and new MPPI defaults now use these verified1.2s settings.
Historical saved settings and PPO config are unchanged. No optimizer remains
running and no further automatic reruns are requested.

## Recovery selection bug found during the run

An early retained proposal had a valid predicted strike with low exit speed,
but the existing recovery selector minimized peak height before applying a
floor. It selected a curve dipping to -0.38457m although another duration
satisfied both height bounds. The PVA exporter previously rejected that curve
after selection rather than searching among height-feasible alternatives.

`plan_curved_recovery` now accepts optional height bounds and filters exact
polynomial minimum/maximum height before choosing a turn. PVA supplies its
saved bounds; historical force recovery defaults are unchanged. Regression
and existing recovery/export tests pass (24 tests). The whip is unchanged.
The optimizer's frozen source predates this correction. Its final strike had
an already height-feasible recovery and passed the original export. To verify
the current code as well, derived job `20260909-131858-327997` retained the
EXACT original plan (SHA25696eea21b82aab049724a4e38186f44dea7e20527d177b6c4b79bf7ec74235f5f)
and froze the corrected recovery code. No optimizer was rerun. It selected the
same recovery commands; both original and corrected exports have CSV SHA256
f9c8ef14ec05318cbdc790bbc69d9d9f2413fcc0a1a4cadf1b1d63c21010daea.

## Final verification and artifact

Final rehearsal: `runs/rehearsals_pva/20260909-131858-327997-mppi-diagnostic`.
Full command duration10.266667s, including unchanged whip, recovery and hold.
Command height1.255–1.670926m; predicted drone height1.255–1.724960m; minimum
predicted cable height0.2475m. Original model/command limits pass throughout.
Original whip replay differences: pose6.88e-15m, cable1.29e-12m.

Portable package: `runs/audits/mppi-longer-final-replay-20260909/MPPI-PVA-simulation-diagnostic.zip`.
CSV regenerates identically and every saved array has maximum difference zero.
69 tests pass on Windows/RTX4080/PyTorch2.11.0+cu128, including legacy curved
recovery and force export. Native six-page UI and exact new rehearsal render
pass with no errors: `runs/audits/mppi-longer-final-ui-20260909`. Desktop was
reopened with1.2s/36-action controls. Fitting/PPO and heartbeats remain stopped.

This is one successful historical-model simulation with complete replay.
It is not physical flight validation, independent robustness testing, or a
demonstration of equivalence to a trained same-contract PPO policy.
