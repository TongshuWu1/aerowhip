> Historical document archived on 9 September 2026. For current work, read [HANDOFF.md](../../../HANDOFF.md). Old running-job and launch instructions below are historical.

# Receding MPPI correction and verification

Historical record of the0.4s correction. Later user-authorized tuning selected
1.2s and achieved a modeled valid hit with complete export; current settings
and results are in docs/history/20260909-doc-cleanup/MPPI_LONGER_HORIZON_20260909.md.

User explicitly corrected the horizon bug:0.4 seconds is lookahead, not the
whole maneuver. Work is authorized to complete planning/recovery/UI checks.
Fitting/PPO/heartbeat stay stopped; no physical flight. Historical normalized
M1 is the frozen simulator for all new diagnostics.

Implemented `planning/mppi_receding.py`, selected only by `mppi.mode=receding`.
Old snapshot workflows and mode-less/open_loop jobs retain their semantics.
Lookahead12 actions; commit ONE30Hz action to independent continuing simulator;
shift best proposal, append zero jerk, repeat. Candidates branch all pose/cable
state, contact/hit flags, cumulative reward state, command queue and absolute
index. GPU branch continuity/independence tests pass, including a failed row.

Separate `task.duration_s=5` is a physical maneuver safety limit, not optimizer
runtime. `mppi.iterations=0` still means no iteration ceiling. Minimum3 iterations
and patience3 per warm-started window replace whole-sequence20/15; this is a
plateau condition, not a fixed per-window budget.1024 random candidates plus
mean, noise0.3, temperature1, explicit control_prior0 remove the additional
zero-centered preference. Actual task rewards/hit rules remain unchanged.

Terminal guidance is explicit and separate from actual task reward. Initial
tip-position/velocity guidance tended to lift the cable without a strike.
Diagnostic20260909-123448-035589 was cooperatively stopped; partial plan saved,
not exportable. Its files and source remain unchanged. Later settings also
guide the predicted drone toward a staging position0.6 cable lengths along
the strike direction and0.7 lengths above target, representing continuation
beyond the short window without extending its physics rollout. Active weights:
tip position10, tip velocity2, drone staging30, command speed0.1, command
acceleration0.01. These are planner heuristics requiring verification.

Completed diagnostic: runs/mppi_pva/20260909-124046-219074, evidence at
runs/audits/mppi-receding-guidance-20260909/result.json. It committed 150
actions over 5 seconds, using 938 window optimization iterations in 971.71s.
It terminated on the maneuver limit with no modeled constraint failure, but
NO valid hit. Closest tip distance was 0.51465269m; actual return -73.01517.
No live optimizer remains; do not restart automatically.

New plans include committed_steps/plan_complete and keep actual actions only.
Partial files are rejected for rehearsal/export; completed files replay from
original hover through all committed actions, not only one lookahead window.
Exporter checks original limits and complete predicted recovery. UI separates
lookahead from maneuver safety limit, and plots committed simulated state
metrics using windows.json, rather than showing future candidate hits as actual.

49 focused tests pass on Windows / RTX 4080 / PyTorch 2.11.0+cu128. The suite
covers CUDA derivatives/solvers, PVA physics/performance parity, PPO contracts,
rolling state continuation, optimistic candidate versus committed hit,
manual STOP retaining a partial plan, export guards and independent UI fields.
Actual committed-plan branches at steps 0,20,40,55 match independent replay:
maximum cable difference 5.07e-13 m and reward difference 1.58e-12.
Evidence: runs/audits/mppi-receding-guidance-20260909/state-parity.json.

Native six-page UI render passed with no errors; lookahead and maneuver limit
are separately visible and progress shows committed simulation over time.
Evidence: runs/audits/mppi-receding-ui-20260909. The UI audit's rehearsal was
the explicitly identified historical 0.4s artifact; it is not this pending run.
Desktop GUI was normally closed and reopened to load the corrected controls.
Final native UI audit also loaded this exact new rehearsal and rendered its
closest approach: runs/audits/mppi-receding-final-ui-20260909, errors empty.

## Complete replay/export result

Rehearsal: runs/rehearsals_pva/20260909-124046-219074-mppi-diagnostic.
Complete whip/recovery/hold is 13.7 seconds and passes saved command and
predicted drone/cable height limits. Original 5-second whip is preserved.
Streaming versus export differences: pose 1.66e-14m, cable 2.24e-10m.
Portable ZIP under the diagnostic audit directory regenerates the exact CSV
hash and every saved array with maximum difference zero. This is an exportable
simulation MISS, not a successful strike or physical flight validation.

The horizon/state/replay/UI bugs are corrected. Short-window terminal guidance
and exploration still do not yield a valid strike in this diagnostic; PPO-level
performance is unproven. No trained same-contract PVA PPO comparator exists.
Do not hide the miss, relax hit criteria, or relabel the earlier force PPO.
PPO config hash remains 195562600bff1b6b05d5fd1cbf5ab7391d9ca5260dd106339984d33b4db71c75.
The fresh fit/campaign remain stopped and both monitoring automations paused.
