# Milestone 4B.1 — MPPI Numerical Consistency and Two-Target Planning Report

## Outcome

The aggressive full-horizon batch/replay numerical inconsistency is fixed. The saved Milestone 4A command now produces exactly matching results for logical B=1, B=8, and B=2048 evaluation, including final UAV position, final c10 position, minimum target distance, success classification, and first-entry classification.

Adaptive MPPI temperature and feasibility-first candidate ordering are implemented and verified. One authoritative seed-42 rerun of `canonical_whip_v1` was completed. Its deterministic replay is numerically trustworthy and physically feasible, but the task remains **FAIL** because the cable tip does not reach the target with the required speed/direction.

`figure8_endpoint_whip_v1` was **not run**. The repository contains no single current authoritative Figure-8 command generator or frozen trajectory definition. The three current horizontal Figure-8 command records have different centers and amplitudes, while the only analytic Figure-8 source is explicitly under `legacy/`. The instruction requires Task B to stop rather than guess when the authoritative definition is ambiguous, so no target, direction, result, or video was invented.

Overall:

`TWO_TASK_MPPI = FAIL / INCOMPLETE`

This result does not authorize changing the horizon or tuning MPPI. The best feasible Task A event occurs at the 0.70-s horizon boundary; that fact is recorded for a later scientific decision.

## 1. Frozen model confirmation

The planning simulator remained exactly:

`MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`

with:

- frozen UAV Physics;
- frozen causal UAV residual;
- rigid two-node clamp;
- frozen 12-node DDER;
- CUDA float32;
- PCG32;
- three DDER substeps;
- four position projections.

No UAV parameter, residual weight/normalization/history rule, EI, Cb, cable mass, rest length, attachment offset, node count, DDER equation, projection rule, or solver iteration count was changed. No fitting or residual retraining occurred.

The active freeze passed the existing integrity/hash checks before planning. `fig8vertical_002` remained sealed and was not opened or predictively evaluated.

## 2. Milestone 4A batch/replay problem

The Milestone 4A sampled winner reported a minimum target distance of 25.936 mm, while its logical batch-one replay reported 56.416 mm. This 30.481-mm discrepancy was caused solely by the evaluation batch shape and was unacceptable for contact-sensitive planning.

The exact saved 4A acceleration knots and post-hover state were replayed at B=1, B=8, and B=2048. Every row inside B=8 and B=2048 was an identical clone and received the identical FullState command.

### First observed divergence before the fix

For B=1 versus B=2048:

| Component | First differing time | First absolute difference |
|---|---:|---:|
| Residual acceleration output | 0.01 s | `2.9802e-08` m/s² |
| UAV velocity | 0.01 s | `6.9849e-10` m/s |
| UAV orientation | 0.01 s | `1.1642e-10` |
| UAV position | 0.02 s | `7.2760e-12` m |
| Cable position | 0.02 s | `4.0163e-09` m |
| Cable velocity | 0.02 s | `1.0976e-06` m/s |

At 0.70 s, the B=1/B=2048 maximum component differences had amplified to:

| Component | Difference |
|---|---:|
| UAV position | `7.4506e-08` m |
| UAV velocity | `1.1921e-06` m/s |
| Residual output | `5.7220e-06` m/s² |
| Cable position | `4.3048e-02` m |
| Cable velocity | `6.7662e-01` m/s |

The same B=1 root-boundary sequence was then supplied to DDER at B=1, B=8, and B=2048. Cable position and velocity differences were exactly zero through the complete 0.70-s rollout. This isolates the origin upstream of DDER. PCG32, fused damping, and fused projection were not the source.

## 3. Root cause and exact numerical fix

The earliest observable discrepancy is in the float32 UAV/residual response path. CUDA evaluates the small residual/UAV tensor operations through batch-shape-dependent arithmetic paths. The tiny upstream boundary difference is dynamically amplified by the aggressive cable trajectory.

The fix adds an explicit planning-only fixed UAV evaluation size:

`fixed_uav_evaluation_batch_size = 2048`

When the logical UAV batch is smaller than 2048, `FullStateUAVModel` pads the UAV state, residual FIFO, and FullState command with identical copies, evaluates the unchanged equations at the canonical 2048-row shape, and immediately slices back to the logical rows. The DDER cable solver still advances its real logical batch size. Default non-planning simulator behavior remains unchanged because the fixed evaluation size is disabled unless explicitly selected by planning.

This changes neither model equations nor parameter values. It standardizes the numerical evaluation shape used by sampled candidates and deterministic replay.

## 4. Post-fix B=1/B=8/B=2048 gate

The exact saved aggressive 4A command was rerun after the fix.

| Comparison | Final UAV difference | Final c10 difference | Minimum-distance difference | Classification | Result |
|---|---:|---:|---:|---|---|
| B=1 vs B=8 | 0.000 mm | 0.000 mm | 0.000 mm | identical | PASS |
| B=1 vs B=2048 | 0.000 mm | 0.000 mm | 0.000 mm | identical | PASS |

Required tolerances were 1 mm final UAV, 2 mm final c10, and 2 mm minimum target distance. All measured differences were exactly zero.

Artifact:

`reports/mppi_batch_consistency_4b1_gate.json`

## 5. Adaptive temperature

Fixed `lambda = 1.0` is no longer used for the authoritative update. For each already-computed cost vector, a deterministic scalar bisection chooses the temperature whose normalized exponential weights target:

`ESS_target = 0.02 × N`

For N=2048, the target is 40.96. No rollout is rerun during the temperature search.

### ESS before and after

| Iteration | 4A fixed-lambda ESS | 4B.1 selected lambda | 4B.1 ESS |
|---:|---:|---:|---:|
| 1 | 1.00 | 614.1664 | 40.9600 |
| 2 | 1.00 | 149.3485 | 40.9600 |
| 3 | 4.38 | 127.6568 | 40.9600 |
| 4 | 1.60 | 122.7606 | 40.9600 |

The adaptive result is inside the required 1–5% population band in every iteration.

## 6. Feasibility-first handling

Each population now separates:

- task objective;
- hard-envelope feasibility;
- continuous normalized feasibility-violation magnitude.

Within a population, the infeasible offset is derived from that population's finite task-cost span. Consequently, every finite feasible candidate ranks before every infeasible candidate, without assigning all infeasible rows the same infinity or using an arbitrary `1e9` feasibility penalty. Infeasible candidates retain continuous ordering from task cost and violation magnitude.

Across iterations, the retained command is:

1. the lowest task-cost feasible candidate, if any feasible candidate exists;
2. otherwise, the candidate with the lowest feasibility violation, using task cost as the tie-breaker.

The closest overall Task A trajectory came within 58.380 mm but was correctly rejected from final selection because it used 1.291 m UAV displacement and 8.614 m/s UAV speed. The selected trajectory was feasible: 0.476 m displacement and 2.627 m/s maximum speed.

## 7. Figure-8 endpoint authority audit

No current non-legacy Figure-8 command producer or frozen analytic trajectory definition exists in the repository. `experimental_data/source_audit/experiment_logger.py` records `cmdFullState`; it does not generate it.

The current recorded horizontal command streams are not one common definition:

| Command record | Approx. center [m] | Approx. amplitudes X/Y [m] | Rightmost recorded command [m] |
|---|---|---|---|
| `fig8_001` | `[0.0644,-0.0968,1.4424]` | `0.500/0.250` | `[0.5644,-0.0957,1.4424]` |
| `fig8_002` | `[0.0553,-0.0479,1.4397]` | `0.500/0.250` | `[0.5553,-0.0460,1.4397]` |
| `fig8_003` | `[0.0847,0.0138,1.4458]` | `1.000/0.500` | `[1.0847,0.0152,1.4458]` |

The only analytic path source found is:

`legacy/current_baseline_2026-08-27/drone_mpc/figure8_tracking.py`

It defines a legacy geometric cable-tip reference with different semantics, not the current physical-experiment FullState command generator. Using it would violate the instruction not to use superseded Figure-8 code.

There is also an unresolved semantic ambiguity for a closed Figure-8: “endpoint” could mean its closed start/end crossing or a lobe extremum. No current source freezes that choice.

Therefore:

- no Figure-8 target was guessed;
- no impact direction was selected from an optimized result;
- no `figure8_endpoint_whip_v1` task configuration was fabricated;
- Task B MPPI was not run;
- no Task B replay or video was fabricated.

Audit artifact:

`reports/figure8_endpoint_authority_audit.json`

Task B can proceed only after the exact current Figure-8 generator/config and endpoint convention are identified or archived.

## 8. Task A iteration history

Task: `canonical_whip_v1`

Configuration remained 0.70 s, 11 knots, 2048 samples, four maximum iterations, sigma 2.5 m/s², seed 42, and the same structured forward-recoil nominal.

| Iteration | Best feasible cost | Best feasible tip error | Best overall tip error in population | Feasible | Successful | Lambda | ESS | Runtime |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 763.664 | 690.386 mm | 53.949 mm | 45 | 0 | 614.166 | 40.96 | 0.665 s |
| 2 | 606.160 | 613.522 mm | 99.627 mm | 737 | 0 | 149.349 | 40.96 | 0.738 s |
| 3 | 335.353 | 443.860 mm | 124.532 mm | 872 | 0 | 127.657 | 40.96 | 0.672 s |
| 4 | 175.878 | 320.554 mm | 123.315 mm | 744 | 0 | 122.761 | 40.96 | 0.681 s |

The best feasible event is at exactly the final 0.70-s frame.

## 9. Task A deterministic replay

Result directory:

`data/planning_results/canonical_whip_v1/2026-08-29T060719.197533Z`

The selected sampled row and authoritative logical batch-one replay agree exactly:

| Quantity | Difference |
|---|---:|
| Final UAV position | 0.000 mm |
| Final c10 position | 0.000 mm |
| Minimum target distance | 0.000 mm |
| Success classification | identical |
| First-entry classification | identical |

### Final Task A metrics

| Quantity | Value |
|---|---:|
| Valid hit time | none |
| Best event time | 0.700 s |
| Minimum / event tip error | 320.554 mm |
| Tip total speed | 7.014 m/s |
| Directed tip speed | 2.913 m/s |
| Direction error | 65.457 deg |
| First target-entry marker | none |
| Maximum UAV displacement | 0.4764 m |
| Maximum UAV speed | 2.6271 m/s |
| UAV-to-target distance at event | 1.0837 m |
| Tip/UAV speed ratio at event | 3.703 |
| Maximum command acceleration | 14.6736 m/s² |
| Rollout finite | yes |

### Hard gates

| Gate | Requirement | Observed | Result |
|---|---:|---:|---|
| Tip position | ≤ 50 mm | 320.554 mm | FAIL |
| Directed tip speed | ≥ 4.0 m/s | 2.913 m/s | FAIL |
| Direction error | ≤ 30 deg | 65.457 deg | FAIL |
| Tip first | c10 first | no entry | FAIL |
| UAV displacement | ≤ 0.50 m | 0.4764 m | PASS |
| UAV speed | ≤ 3.0 m/s | 2.6271 m/s | PASS |
| Command acceleration | ≤ 20 m/s² | 14.6736 m/s² | PASS |
| Finite rollout | required | yes | PASS |

Task A classification:

`canonical_whip_v1 = FAIL`

The task is not numerically untrustworthy; it is a genuine planner failure under the frozen 0.70-s configuration.

## 10. Task B status

Task B classification:

`figure8_endpoint_whip_v1 = NOT RUN — AUTHORITATIVE SOURCE AMBIGUOUS`

This is the explicit stop condition from the instruction, not an MPPI or numerical failure.

## 11. Video artifacts

Task A deterministic replay video:

`data/planning_results/canonical_whip_v1/2026-08-29T060719.197533Z/canonical_whip_v1_final_replay.mp4`

Properties:

- source: saved authoritative `final_replay.npz`;
- 1280×720;
- 30 fps;
- H.264 Constrained Baseline;
- yuv420p;
- four-times slow motion;
- 145,217 bytes;
- MPPI/physics was not rerun for rendering;
- labeled simulation-only and FAIL.

Task B video:

`NOT CREATED — TASK B NOT AUTHORIZED BY AN UNAMBIGUOUS SOURCE`

Email:

`EMAIL_DELIVERY = NOT_AVAILABLE_IN_LOCAL_ENVIRONMENT`

No credential or email implementation was added.

## 12. Runtime

Task A on NVIDIA GeForce RTX 4080 / PyTorch 2.11.0+cu128:

| Quantity | Value |
|---|---:|
| MPPI iterations | 4 |
| Candidate rollouts | 8192 |
| MPPI solve | 2.763 s |
| Effective rollouts/s | 2964.7 |
| Deterministic replay | 0.730 s |
| Peak CUDA allocated memory | 26.62 MiB |

Task B runtime is not applicable because it was not run.

## 13. Verification

Verification completed:

- aggressive saved-command B=1/B=8/B=2048 full-horizon consistency: PASS;
- adaptive-temperature synthetic ESS target test: PASS;
- feasible-near-miss versus illegal-close-hit ranking test: PASS;
- Task A sampled winner versus deterministic replay gate: PASS;
- repository tests: `61 passed in 27.54 s`;
- Python compilation: PASS;
- `git diff --check`: PASS (only pre-existing line-ending warnings were printed).

The requested Figure-8 endpoint extraction test could not be truthfully created because no authoritative current definition exists. The authority audit records the exact blocker instead.

## 14. Safety and scope

- Protected test: **NOT EVALUATED**.
- Real Crazyflie hardware: **NOT EXECUTED**.
- Generated commands: **SIMULATION ONLY / NOT AUTHORIZED FOR REAL FLIGHT**.
- No horizon extension, sample increase, extra seed, target change, model refit, or automatic tuning was performed.

## Final summary

    Batch consistency:
        PASS

    Adaptive MPPI:
        ESS target = 40.96 / 2048 (2%)
        achieved ESS = 40.96 in every Task A iteration

    Task A:
        canonical_whip_v1
        target = [1.0, 0.0, 1.4]
        hit time = none; best feasible event = 0.700 s
        tip error = 320.554 mm
        directed speed = 2.913 m/s
        direction error = 65.457 deg
        UAV displacement = 0.476 m
        UAV speed = 2.627 m/s
        result = FAIL

    Task B:
        figure8_endpoint_whip_v1
        target = NOT FROZEN — authoritative source ambiguous
        direction = NOT FROZEN
        hit time = NOT RUN
        tip error = NOT RUN
        directed speed = NOT RUN
        direction error = NOT RUN
        UAV displacement = NOT RUN
        UAV speed = NOT RUN
        result = NOT RUN — AUTHORITATIVE SOURCE AMBIGUOUS

    TWO_TASK_MPPI:
        FAIL / INCOMPLETE

    VIDEOS:
        Task A = data/planning_results/canonical_whip_v1/2026-08-29T060719.197533Z/canonical_whip_v1_final_replay.mp4
        Task B = NOT CREATED

    PROTECTED TEST:
        NOT EVALUATED

    REAL HARDWARE:
        NOT EXECUTED
