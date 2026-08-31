# Milestone 7A Partial Progress Report

## Scope and stop decision

The long-running Milestone 7A teacher campaign was stopped cleanly at the
user-requested checkpoint. Both CEM worker processes received an interrupt and
exited. No generated artifact was deleted or overwritten, no additional CEM
context was started after the stop request, and no diffusion training, scorer
training, validation campaign, or aggregation round was started.

This is a viability checkpoint, not a completed Milestone 7A result. In
particular, neural task-success numbers are not reported because the explicit
minimum-data condition for running the partial viability experiment was not
met.

## 1. Current run status

| Item | Status |
|---|---:|
| Artifact timestamp | `2026-08-30T14:10:19.803847Z` |
| Stop/inventory timestamp | `2026-08-30T15:51:32.693185Z` |
| Elapsed wall time | 1 h 41 min 13 s |
| Stage at interruption | Teacher generation |
| Completed CEM contexts | 42 / 768 |
| Remaining CEM contexts | 726 |
| Authoritatively solved contexts | 36 |
| Unsolved after the allowed CEM attempts | 6 |
| Diffusion training | Not started |
| Scorer training | Not started |
| Validation | Not started |
| Aggregation | Not started |

Generation is resumable without recomputing any of the 42 durably committed
contexts. The main progress manifest and two disjoint worker progress manifests
identify every completed context and its shards. Context indices 39 and 46 have
partial CEM iteration checkpoints but no committed result; on resume, only
those interrupted contexts' uncommitted work must restart. The 42 completed
contexts are skipped by the manifest-driven resume path.

## 2. Teacher data generated so far

The split was frozen before any teacher optimization and is state-disjoint.
All completed rows happened to be in the first, TRAIN portion of the manifest.

| Split | Designed contexts | Attempted/completed | Authoritative solved | Unsolved | Successful actions | Mean / median actions per solved context | Scorer rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 512 | 42 | 36 | 6 | 605 | 16.806 / 17 | 21,504 |
| VALIDATION | 128 | 0 | 0 | 0 | 0 | N/A | 0 |
| TEST | 128 | 0 | 0 | 0 | 0 | N/A | 0 |
| **Total** | **768** | **42** | **36** | **6** | **605** | **16.806 / 17** | **21,504** |

The teacher solve rate over attempted contexts is 85.71%. Successful-action
counts range from 1 to 32 per solved context. The 21,504 scorer rows are exactly
512 outcome-stratified rows for each completed context.

### Actual CEM runtime

Runtime per context is the sum of all CEM seed attempts made for that context.
Fifty-five seed attempts were executed across the 42 completed contexts.

| Statistic | Runtime |
|---|---:|
| Mean / completed context | 178.453 s |
| Median / completed context | 112.874 s |
| P90 / completed context | 469.337 s |
| Total accumulated CEM compute | 7,495.012 s (2 h 4 min 55 s) |

The accumulated CEM time exceeds wall time because two workers ran
concurrently for part of the campaign.

## 3. Data-quality verification

- **Population to authoritative replay mismatch count admitted to the teacher
  set: 0.** A diffusion label is written only when the exact final normalized
  action passes the authoritative fixed-2048 replay. All 605 saved labels are
  authoritative successes. The partial shards do not retain a separate
  all-population flip table, so this statement is specifically the count of
  mismatches admitted as labels; the permanent Milestone-6A replay diagnostic
  remains zero flips.
- **Split leakage: none.** TRAIN, VALIDATION, TEST, edge-reserved, and
  aggregation-pool state ownership intersections are all zero. Completed
  context IDs are unique.
- **Shard integrity: pass.** All 36 teacher-shard SHA-256 values and all 42
  scorer-shard SHA-256 values match their manifests. No shard hash failure was
  found.
- **Action representation:** final normalized production `[49]`: 16 x 3
  acceleration knots plus one linearly normalized active-duration coordinate.
  All saved teacher arrays have 49 columns. Maximum observed absolute
  coordinate is 0.939766.
- **Command semantics:** final Milestone-6A `ACTIVE -> SETTLE -> HOLD` contract.
  Active duration is `[0.45, 1.80]` s, deterministic analytic settle is 0.30 s,
  and evaluation horizon is 2.40 s. Fixed numerical physics batch is 2048.
- **No old 1.20-s semantics:** confirmed. The decoder range is 1.80 s and the
  smooth terminal continuation is active. Saved successful active durations
  span 1.015764--1.289774 s (median 1.142320 s).
- **No SAC data:** no SAC action, replay, reward, critic, or policy weight enters
  either dataset. The source state banks live under a historically named
  one-shot-SAC artifact directory, but they are physically propagated initial
  states, not SAC experiences.
- **Theta:** nominal only. Theta is nevertheless stored explicitly in every
  83-D context under the permanent schema.
- **Frozen model:** `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`; the production
  CEM source hash and teacher configuration remain the launch versions.

The launch-time source hash manifest is preserved, not rewritten. Four
non-planner source files now differ from that launch manifest because
deployment/scorer fixes and aggregation-resume tests were completed while the
teacher workers were running. The production CEM implementation and the
teacher configuration hashes still match. These post-launch changes do not
alter any committed CEM rollout or label; a future resumed campaign should
record an additional current-source manifest rather than overwrite the launch
manifest.

## 4. Action-data diagnostics already available

The analysis below uses only the 605 completed authoritative-success actions.
PCA is diagnostic and is not used as a policy bottleneck or decoder.

### PCA explained variance

| Threshold | Components required |
|---|---:|
| 95% | 40 |
| 99% | 45 |
| 99.9% | 48 |

The first eight principal components explain only 48.87% of variance. The
partial teacher set therefore does not resemble a very low-dimensional linear
action manifold.

### Multiple solutions and action distances

| Diagnostic | Value |
|---|---:|
| Solved contexts with more than one success | 34 / 36 (94.44%) |
| Within-context action pairs | 7,257 |
| Mean within-context Euclidean distance | 0.442730 |
| Median within-context Euclidean distance | 0.439831 |
| P90 within-context Euclidean distance | 0.539435 |
| Maximum within-context Euclidean distance | 0.799056 |
| Median of per-context median distances | 0.441654 |
| Median nearest action from another context | 0.397597 |

Distances are computed directly in normalized 49-D production action space.
The within-context spread is substantial and is comparable to, rather than
negligible beside, between-context nearest-neighbor separation. Together with
the high PCA dimension and the 94.44% multiple-solution rate, this is evidence
against treating each context as having one tight deterministic MSE label.

It is **consistent with meaningful multimodality**, but it does not yet prove
strong, cleanly separated discrete modes: most alternatives are final-top-32
solutions from one successful CEM basin/seed. More completed states and
multi-seed successful contexts would be needed for a strong discrete-mode
claim. The diagnostic nevertheless supports retaining a generative policy
rather than reverting to averaged deterministic regression.

## 5. Diffusion implementation status

`ConditionalActionDiffusion` is implemented and runnable in direct normalized
49-D production action space. It contains **1,249,329 trainable parameters**.

Architecture:

1. Context encoder: `83 -> 256 SiLU -> 256`.
2. Sinusoidal timestep embedding: 64 dimensions, then
   `64 -> 256 SiLU -> 256`.
3. Noisy-action projection: `49 -> 256`.
4. Sum action, context, and timestep embeddings.
5. Four residual MLP blocks, each `LayerNorm -> 256 -> 512 -> SiLU -> 256`,
   followed by residual addition.
6. Output: `LayerNorm -> 49` epsilon prediction.

Implemented training/inference contract:

- 100-step variance-preserving DDPM with cosine cumulative-alpha schedule.
- Standard epsilon-prediction MSE.
- Deterministic 25-step DDIM sampler with `eta = 0`.
- Fixed `[32,49]` noise-bank input and deterministic repeatability.
- Final-only clamp to `[-1,1]`; intermediate diffusion states are not clamped.
- EMA implementation with decay 0.999, including checkpoint state round-trip.

No diffusion training checkpoint or frozen noise-bank artifact exists because
training was never started. The implementation and permanent data path are
complete; the final combined Milestone-6A/7A regression run reports **16
passed, 0 failed** (12 Milestone-7A tests and 4 Milestone-6A production-CEM
tests).

## 6. Scorer implementation status

`ManeuverOutcomeScorer` is implemented and runnable. It contains **168,202
trainable parameters**.

Architecture:

- Input: normalized 83-D context plus normalized 49-D action = 132 dimensions.
- Trunk: `132 -> 256 SiLU -> 256 SiLU -> 256 SiLU`.
- Seven continuous outputs: legacy optimizer reward,
  `log(tip_distance + 1e-4)`, directed speed, cosine of direction error, maximum
  UAV displacement, maximum UAV speed, and maximum command acceleration.
- Three binary logits: tip-first, finite, and scientific success.

Training utilities are implemented for TRAIN-only continuous-target
standardization with a safe standard-deviation floor, SmoothL1/Huber loss on
the seven standardized continuous outputs, BCE-with-logits on binary outputs,
and TRAIN-only class balancing capped at 20. The deterministic candidate
selector implements the predicted hard gates, highest predicted legacy reward
among predicted passes, and the required lexicographic minimum-margin fallback.

Target normalization was not permanently fitted or saved because scorer
training did not start. The temporary inference-latency measurement fitted a
normalizer in memory from the available TRAIN scorer rows only; it was not
saved and is not a policy-performance result. Scorer schema, finite loss,
normalization, and deterministic selection tests pass. No scorer checkpoint
exists.

## 7. Small architecture viability test

**Not run.** Only 36 solved TRAIN contexts are available, below the requested
roughly-50-context gate. In addition, teacher generation never reached a
VALIDATION context, so there are zero generated validation teacher/scorer
shards. Starting final-architecture training here would make validation-based
early stopping and scorer validation ill-defined and would invite an
overconfident conclusion from a very small state sample.

No toy architecture, reduced network, shortened substitute training loop, or
TRAIN-as-validation result was created.

## 8. Critical viability metrics

Because Section 7 did not pass its data gate, the required offline production
simulator evaluation was not run.

| Metric | Partial result |
|---|---:|
| Oracle best-of-32 scientific success | NOT RUN |
| Scorer-selected scientific success | NOT RUN |
| First-candidate scientific success | NOT RUN |
| Selected-action feasibility | NOT RUN |
| Median tip error | NOT RUN |
| Median directed speed | NOT RUN |
| Median direction error | NOT RUN |
| Median UAV displacement | NOT RUN |

There are 128 validation contexts in the immutable design manifest, but zero
of them currently have completed teacher/scorer data. The final TEST split and
the separate sealed joint test were not accessed.

## 9. Interpretation

**Partial classification: `INSUFFICIENT_DATA_TO_JUDGE`.**

The architecture itself is complete and tested, so
`IMPLEMENTATION_INCOMPLETE` would be inaccurate. The action diagnostics support
the motivation for a direct generative model: successful labels are
high-dimensional and show material within-context diversity. They do not,
however, demonstrate that the diffusion generator learns a useful conditional
distribution. No trained generator exists, so neither
`ARCHITECTURE_PROMISING`, `GENERATOR_NOT_LEARNING`, nor `SCORER_NOT_LEARNING`
is scientifically supported yet.

The partial data therefore answers an important but narrower question: the
teacher dataset has the diversity the generative architecture was designed to
represent, and the permanent implementation is runnable, but the run stopped
before enough TRAIN and VALIDATION coverage existed to test amortization.

## 10. Learning-curve test

**Not run.** Nested 25%, 50%, 75%, and 100% training would contain only about
9, 18, 27, and 36 solved TRAIN contexts, respectively, while holding zero
generated validation contexts fixed. Such a curve would measure small-sample
instability rather than scaling with teacher data. No new CEM label was
generated and no data split was repurposed.

## 11. Latency

The complete untrained final architecture was benchmarked only to verify its
engineering inference cost. The test used the implemented direct 49-D model,
32 fixed candidate noises, 25 DDIM steps, and scorer/selector on the current
CUDA workstation. It used 30 warm-up queries and 200 measured queries. These
times are **implementation latency only** and do not imply a useful policy.

| Component | Median GPU / wall | P95 GPU / wall |
|---|---:|---:|
| 32-candidate, 25-step diffusion | 24.653 / 24.672 ms | 26.996 / 27.030 ms |
| 32-candidate scorer | 0.259 / 0.276 ms | 0.547 / 0.645 ms |
| Full context-tensor policy query, including selection | 26.280 / 26.310 ms | 28.501 / 28.680 ms |

Maximum measured full-query wall latency was 39.950 ms. Thus the implemented
architecture is compatible with the nominal `<50 ms` median and `<100 ms` P95
engineering goals before training. Raw sensor/state acquisition is not part of
this partial timing; no CEM or simulator is included.

## 12. Artifact integrity and resume record

Artifact root:

`data/policy_training/amortized_cem_diffusion_nominal_v1/2026-08-30T141019.803847Z`

Completed persistent artifacts include:

- `context_split_manifest.json` and `target_manifest.json`;
- `context_table.npz` and `context_normalizer.json`;
- `context_schema.json`, `action_schema.json`, and
  `production_command_contract.json`;
- `teacher_generation_config.json` and main
  `teacher_generation_progress.json`;
- main `diffusion_teacher_manifest.json` and `scorer_dataset_manifest.json`;
- two disjoint `teacher_workers/worker_*_of_02/progress.json` manifests;
- 36 diffusion-teacher NPZ shards;
- 42 compact scorer NPZ shards;
- 806 per-iteration CEM checkpoint NPZ files, including preserved partial
  checkpoints for interrupted context indices 39 and 46;
- `teacher_cem_failures.json` (main-worker portion; combined failures are
  reconstructed from all three progress manifests);
- `sealed_evaluation_context_manifest.json`;
- the preserved launch-time `source_hash_manifest.json`.

Total current artifact size is approximately 21.49 MB. There are no diffusion,
EMA, scorer, aggregation, or policy-freeze checkpoints because those stages did
not start.

Resume status is **YES**. To resume, use the same artifact root and the same
two-worker assignment. The worker manifests skip all 42 completed contexts.
The two interrupted, uncommitted contexts restart; their existing partial
checkpoints remain available for forensic inspection. Teacher manifests should
be merged only after all 768 contexts complete. No aggregation should start
before that merge and the intended model-training/validation stages.

Protected `fig8vertical_002` was not accessed. Real hardware was not connected
or executed.

## 13. Final partial summary

    CEM contexts completed:
        42

    CEM contexts remaining:
        726

    Verified successful teacher contexts:
        36

    Successful teacher actions:
        605

    Scorer rows:
        21,504

    Diffusion implemented:
        YES

    Scorer implemented:
        YES

    Validation contexts available:
        0 generated / 128 designed

    Oracle best-of-32 success:
        NOT RUN

    Scorer-selected success:
        NOT RUN

    First-candidate success:
        NOT RUN

    Current inference latency:
        median 26.310 ms
        p95 28.680 ms
        UNTRAINED ARCHITECTURE TIMING ONLY

    Learning curve:
        NOT RUN — INSUFFICIENT SOLVED TRAIN / NO GENERATED VALIDATION DATA

    Partial classification:
        INSUFFICIENT_DATA_TO_JUDGE

    More CEM data justified:
        CANNOT JUDGE

    Safe to resume:
        YES

