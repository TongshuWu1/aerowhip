# Targeted Iterative Residual Training Report

## Outcome

This continuation trained the same 372k-parameter residual-outcome architecture on simulator responses centered on actions that actually failed. It ran zero CEM solves and preserved the frozen production model/action contract.

## Targeted data

- Failed training contexts: 64
- Failed state-disjoint development contexts: 24
- Candidate responses/context: 256
- New physical response rows: 22528
- Successful response rows: 3710
- Feasible response rows: 12422
- Saved trajectory feature: 25 time samples of tip-target position, tip velocity, and UAV displacement
- New CEM solves: 0

Candidate corrections were generated in the existing 13-D basis around the failed 7B initializer, then every response was propagated by the unchanged full production simulator.

### Local support by context group

| Split/group | Contexts | Successful candidates | Candidate success rate | Contexts with any success |
|---|---:|---:|---:|---:|
| Train target-only | 16 | 1,891/4,096 | 46.17% | 100.0% |
| Train state-only | 16 | 843/4,096 | 20.58% | 100.0% |
| Train joint | 16 | 244/4,096 | 5.96% | 56.25% |
| Train edge | 16 | 64/4,096 | 1.56% | 50.0% |
| Development state-only | 8 | 518/2,048 | 25.29% | 75.0% |
| Development joint | 8 | 135/2,048 | 6.59% | 87.5% |
| Development edge | 8 | 15/2,048 | 0.73% | 25.0% |

The local correction family contains abundant support for target-only and ordinary state variation, but very little for joint/edge contexts. Training cannot select a successful action in a context where the sampled correction family contains none.

## Training

The prior IRP checkpoint was continued with 50% original teacher-neighborhood pairs and 50% targeted failed-initializer pairs. Outcome-balanced target sampling prevents the rare useful targeted responses from disappearing. Best targeted development loss: 0.778045 at update 800 (initial 0.982455).

## Authoritative correction result

| Executions | Train success | Train feasible | Development success | Development feasible |
|---:|---:|---:|---:|---:|
| 1 | 28.84% | 66.05% | 16.22% | 70.27% |
| 2 | 44.19% | 77.67% | 29.73% | 78.38% |
| 3 | 53.02% | 74.88% | 32.43% | 67.57% |
| 4 | 56.28% | 77.67% | 35.14% | 56.76% |
| 5 | 56.28% | 72.56% | 37.84% | 45.95% |
| 6 | 57.21% | 68.37% | 37.84% | 56.76% |


Best stopping point: 5 corrections. At that point train success improved by +28.37 percentage points and development success by +21.62 points from the one-action initializer.

The previous teacher-neighborhood IRP reached 43.24% development success. This targeted continuation reached 37.84%. Classification: **TARGETED_RESIDUAL_DATA_NO_IMPROVEMENT**.

## Interpretation

This isolates whether relevant residual-response coverage—not merely longer neural optimization—was limiting the previous model. The trajectory features are durably saved but were not yet added to the network input, so any improvement comes strictly from moving the training distribution around the real failure actions. Candidate ranking remained neural-only; unselected candidates were not searched with physics during evaluation.

The targeted checkpoint is **not promoted**. It fits the selected training failures better but generalizes worse than the earlier 43.24% checkpoint. Continuing gradient updates on this same representation is not justified. The durable trajectory data remain useful for a later timing-sensitive diagnostic, but the primary limitation in joint/edge contexts is already candidate support, not only outcome ranking.

## Restrictions

- Production model modified: **NO**
- CEM: **NOT RUN**
- SAC/diffusion/scorer: **NOT USED**
- Final TEST: **NOT EVALUATED**
- Protected test: **NOT EVALUATED**
- Hardware: **NOT EXECUTED**

## Final summary

    New CEM solves:
        0

    New targeted simulator rows:
        22528

    Training updates:
        2000

    Initial development success:
        16.22%

    Best development success:
        37.84%

    Recommended corrections:
        5

    Result:
        TARGETED_RESIDUAL_DATA_NO_IMPROVEMENT

    Final TEST:
        NOT EVALUATED

    Protected test:
        NOT EVALUATED

    Hardware:
        NOT EXECUTED
