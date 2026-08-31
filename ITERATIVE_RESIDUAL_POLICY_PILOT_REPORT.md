# Iterative Residual Policy Pilot Report

## 1. Decision

This pilot tests the actual iterative-residual mechanism used by successful dynamic-rope work, rather than another direct action-regression policy. It uses **zero new CEM solves**, no SAC, no diffusion, no scorer, no final test, no protected data, and no hardware.

## 2. Detailed current pipeline

The complete experimental path is:

1. A physically propagated UAV/cable state and target are converted to the frozen root-centered, yaw-aligned 83-D `PolicyContext`.
2. The already-trained 7B deterministic residual MLP supplies the initial complete normalized 49-D maneuver. It is only an initializer; its development success was 16.22%.
3. That maneuver is decoded by the production codec into 16 three-axis acceleration knots and `T_maneuver` in [0.45, 1.80] s.
4. The command executes as `ACTIVE -> 0.30-s analytic SETTLE -> HOLD`, observed to 2.40 s in the fixed-2048 frozen simulator.
5. The observed rollout is compressed into six physical continuous outcomes (tip distance, directed speed, direction cosine, UAV displacement, UAV speed, command acceleration) plus tip-first, feasibility, and scientific-success flags.
6. A delta-outcome MLP receives normalized context, current action, observed outcome, and one proposed action correction. It predicts the physical result of that correction.
7. Exactly 1024 corrections are generated in a 13-D local basis fitted from existing CEM-family residuals. The authoritative action remains 49-D; the compact basis changes only local search.
8. Candidates are ranked using predicted unchanged hard-gate margins. Neither CEM nor the production simulator participates in ranking.
9. One correction is selected and executed. Steps 5-9 repeat up to 5 times, stopping per context on scientific success.

This is an iterative physical-trial method, not the earlier one-query deployment objective. It is deliberately tested now because direct one-query imitation failed on knife-edge CEM teachers.

## 3. Existing data

- Teacher contexts: 252
- State-disjoint training contexts: 215
- State-disjoint development contexts: 37
- Saved perturbations/context: 449
- Physical outcome rows used for gradients: 96535
- New CEM solves: 0

The data are the existing 7C exact/IID/smooth/duration perturbations evaluated under the final production action, settle, and physics contract.

## 4. Compact correction primitive

The correction basis has 13 dimensions. It is an orthonormal SVD basis of the training-only difference between the 7B initial action and the exact CEM teacher action. It does not approximate or reconstruct the complete teacher action, which avoids the spline reconstruction failure from 7B. Proposed corrections remain bounded and are canonicalized by the production 49-D codec.

## 5. Learned delta-outcome model

Architecture: `(83 context + 49 current action + 9 observed outcomes + 49 delta) -> 384 SiLU -> 384 SiLU -> 384 SiLU`, followed by six continuous and three binary heads. Parameters: 372,489. Best development loss: 0.996608 at update 2600.

Held-out pair prediction used 65536 development pairs. Binary accuracy was {'tip_first': 0.627899169921875, 'feasible': 0.6851043701171875, 'scientific_success': 0.5532379150390625}.

## 6. Authoritative iterative result

| Executions | Train success | Train feasible | Development success | Development feasible |
|---:|---:|---:|---:|---:|
| 1 | 28.84% | 66.05% | 16.22% | 70.27% |
| 2 | 38.60% | 73.02% | 37.84% | 78.38% |
| 3 | 43.72% | 80.47% | 40.54% | 81.08% |
| 4 | 47.91% | 81.40% | 43.24% | 78.38% |
| 5 | 50.23% | 74.88% | 43.24% | 70.27% |
| 6 | 51.63% | 68.84% | 43.24% | 62.16% |


Every entry is the one model-selected maneuver actually replayed through the complete production simulator. Candidate actions that were not selected were not simulated.

## 7. Interpretation

Train success changed by +22.79 percentage points and state-disjoint development success changed by +27.03 points. Classification: **IRP_STYLE_REFINEMENT_PARTIAL**.

This result answers whether learned local physical response plus iterative correction is more useful than direct coordinate imitation. It does not establish a one-query policy and it does not authorize hardware. If refinement is weak, the next issue is the saved perturbation experiment: it records outcome summaries around successful teachers, not complete tip trajectories around arbitrary failed starting maneuvers. More CEM winners would not repair that mismatch.

Development success first reached its maximum after 4 executions (3 corrections). Later corrections did not add development successes and reduced failure-row feasibility, so the practical stopping point for this frozen pilot is 3 corrections. The next scientifically useful dataset would record complete tip trajectories and local responses around the actual failed initializer—not generate more nominal CEM winners.

## 8. Scientific restrictions

- Model: `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`
- Production model modified: **NO**
- New CEM solves: **0**
- SAC/diffusion/scorer: **NOT USED**
- Final TEST: **NOT EVALUATED**
- Protected `fig8vertical_002`: **NOT EVALUATED**
- Hardware: **NOT EXECUTED**

## Final summary

    Method:
        ITERATIVE RESIDUAL OUTCOME MODEL

    Initial maneuver:
        7B DETERMINISTIC RESIDUAL MLP

    Correction dimension:
        13

    Candidate corrections/iteration:
        1024

    Maximum correction iterations:
        5

    Recommended corrections for frozen pilot:
        3

    New CEM solves:
        0

    Training contexts:
        215

    Development contexts:
        37

    Initial train success:
        28.84%

    Final train success:
        51.63%

    Initial development success:
        16.22%

    Final development success:
        43.24%

    Classification:
        IRP_STYLE_REFINEMENT_PARTIAL

    Final TEST:
        NOT EVALUATED

    Protected test:
        NOT EVALUATED

    Hardware:
        NOT EXECUTED
