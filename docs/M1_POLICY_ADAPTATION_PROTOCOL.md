# M1 policy adaptation: research protocol

Historical force-PPO study: the authorized paired study and its rehearsal/export
audit are COMPLETED. Both arms received exactly 20,480 additional attempts. Do not
restart or extend automatically. This protocol preserves the study's original
method; see [HANDOFF.md](../HANDOFF.md) for current MPPI and PVA status.
**Real M1 policy performance remains unknown until new flights.**

Use the retained flown PPO as initialization for a new M1 PPO run. This is continued policy optimization in an updated simulator. Real logs fit the physical model and residuals; PPO receives fresh simulated rollouts. The real aircraft still executes the exported 30 Hz FullState CSV, not the neural policy online.

Keep tracked origin [-2,0,1.255] m, target [-1,0,1.1] m, the retained checkpoint's one-second task, reward, force bounds, 30 Hz clocks and launch/target variation. Freeze M1 physics and both residuals during PPO. Preserve the original policy and artifacts. Start a new optimizer and convergence history; this reset is an explicit design choice, not a theoretical requirement. Apply identical optimizer handling to the continued-M0 control. Record additional simulator attempts, simulated transitions, wall time, random seeds and the original policy's training cost.

## Why warm start

This task, observation/action space and apparatus are unchanged; the update corrects the learned dynamics. Reusing an established maneuver is a reasonable hypothesis for efficient adaptation and limits the need to rediscover a strike. It does not guarantee better final performance or prevent a poor local optimum. Training from scratch remains a useful initialization comparison.

SimOpt explicitly reports initialization from the previous iteration's policy, reducing the PPO iterations required from 200 to 10 in its simulated cabinet adaptation experiment. This is precedent, not an expected speedup for aerial whipping: [Chebotar et al., Section IV-C](https://arxiv.org/html/1810.05687v3). Its model-distribution adaptation differs from our fitted drone/cable residual cascade.

Improved trajectory prediction does not establish improved task execution. AdaptSim likewise distinguishes simulation matching from task performance, although its meta-learned simulator adaptation is different from our identification procedure: [Ren et al., AdaptSim](https://irom-lab.princeton.edu/AdaptSim/).

## Comparisons and attribution

| Condition | Generation model | Policy | Question |
|---|---|---|---|
| Original baseline | M0 | Original frozen PPO | What was flown before adaptation? |
| Model update only | M1 | Same frozen PPO weights | Does regenerating commands under M1 help without a policy update? |
| Extra-training control | M0 | Original PPO continued for the same additional budget | Is improvement explained by more optimization alone? |
| Proposed adaptation | M1 | Original PPO continued for that budget | Does training with the adapted model improve the exported maneuver? |
| Initialization comparison | M1 | Randomly initialized PPO | Does warm starting improve efficiency or constrain the resulting solution? |

Frozen policy weights do not imply frozen forces or CSV: simulated observations change under M1, so both the action sequence and its conversion to FullState can change. Save both. Do not compare an old M0 CSV with a regenerated M1 forecast as if they were the same experiment.

Prioritize the M1 continuation and equal-budget M0 continuation for attributing benefit to model adaptation; evaluate frozen-policy regeneration too. Run the scratch comparison at an explicitly matched additional budget and report the warm start's pretraining cost. If assessing eventual performance rather than adaptation speed, also allow scratch training to its stated stopping rule. Prefer multiple training seeds; repeated hardware execution of one checkpoint is not a substitute for independent training seeds.

Choose checkpoints with a fixed simulation validation protocol, then freeze them before collecting the next real flights. Real-flight comparisons need comparable initial conditions and execution settings. If possible interleave conditions to reduce session drift. Five repeats per condition give initial repeatability evidence, not a strong broad-generalization claim. If control conditions are tested only in simulation, state that they do not establish causal real-flight improvement.

## Separate outcomes

- Model accuracy: drone and tip whip RMS, and prediction-versus-measurement error at the planned strike time. Evaluate all saved models against the same recorded command sequence and initial-state protocol.
- Task performance: measured tip-to-target distance at the preregistered strike time, closest approach within the whip, and speed/direction-gated geometric success. Keep timing error visible. No physical contact occurred in adp0.
- Compute: additional simulation attempts/transitions and time, with total pretraining cost disclosed.

M1 was fitted using all five adp0 flights; the existing leave-one-flight-out results evaluate the adaptation procedure with separate fold models. Score new adp1 flights before using them to fit M2. Existing adp0 improvement alone is not evidence that an M1-generated new maneuver transfers better.

## Authorized execution

Study: `runs/policy_adaptation/20260909-003414-347814-M1-policy-adaptation`. Its runner is bounded to two sequential jobs; no retry or automatic extension. Original retained PPO stays stopped and unchanged.

| Arm | New PPO run | Additional attempts |
|---|---|---:|
| M0 extra-training control | `20260909-003414-738407-seed655` | 20,480 |
| M1 adaptation | `20260909-003414-507606-seed655` | 20,480 |

Each starts from the original best checkpoint at 314,368 attempts and ends at 334,848, with fresh actor/critic optimizer state and no inherited convergence history. Plateau stopping is disabled for this fixed equal budget. Both use seed 655, 2,048 environments, the same 256 fixed development-validation scenarios, identical task/reward/configuration and the visible Isaac Lab host with the existing CUDA physics and both residuals. There is no PhysX replacement. The model and residuals stay frozen during PPO.

Both arms use actual flown tracked-origin [-2,0,1.255] m and target [-1,0,1.1] m. The old parent training task's origin Y was +0.012874 m; both new arms explicitly correct it to zero. This shared amendment is recorded in the study and run provenance. The original one-second objective and 0.1-point/s time weight remain, not another stopped policy's five-second objective. One training seed is a preliminary comparison, not evidence across independent seeds.

The unchanged-policy M1 ablation is saved at `runs/rehearsals/20260909-003414-347814-M1-policy-adaptation-unchanged-PPO-M1`. It predicts a miss with 5.483 cm minimum tip distance in the nominal launch. Its complete 30 Hz command is model-feasible and the whip scoring/export prefix matches exactly. This is a simulation result, not a real-flight result. `baseline_verification.json` records CSV/force checks. `protected_inputs.json` and the final `preservation_check.json` track original policy, rehearsal, models, settings and raw data.

## UI and evidence boundaries

`Adaptation Check → Adaptation progress → Real flight progress` is the default. Its drone/tip RMS uses actual measured samples against the exact saved preflight prediction matched to the executed CSV. It never fills an adapted-flight cell from a held-out or all-data model fit. M1/adp1 remains **Awaiting adp1 flights**, with missing values rather than zero bars. Individual flights and equal-flight means are shown. The separate measured tip-to-target metrics describe hitting accuracy; prediction RMS and task accuracy answer different questions.

The five current adp0 flights give mean drone RMS **15.05 cm**, tip RMS **20.01 cm**, measured tip-to-target distance at the saved planned strike **39.88 cm**, and closest whip distance **18.53 cm**. These compare real recordings with the original preflight forecast, whereas the model-fitting diagnostic initializes from recorded state and uses a different prediction protocol. Do not interchange them. Raw data and original native 3D ghost remain unchanged.

Held-out/all-data comparisons live only under **Model-fit diagnostics** and **Model-fit trajectories**, with explicit labels. The selected M1 policy's training status is simulation-only. The dashboard model-selection button stages a future run without launching it; the explicit model selector snapshots provenance.

After this correction, 11 focused progress/alignment/library tests passed, including a check that saved fit results cannot create real M1 metrics and that missing strike samples remain missing. Native Windows Qt verification preserves seven pages and the original CSV-matched ghost; screenshots, PDF and real metric audit are in `runs/audits/20260909-real-flight-progress`. Earlier UI audits predate the user's real-flight-only progress correction and are historical.


## Completed simulation artifacts and open height issue

The fixed study completed in 1,074.53 seconds including the unchanged-policy baseline and both sequential training arms. Best simulation development validation: M0 control **249/256 (97.265625%)**, M1 **250/256 (97.65625%)**; authoritative values are in the saved `simulation_results.json`. The two arms use the same scenario identity but different dynamics; these are model-conditional simulation scores, not a cross-model real-flight comparison. Selection remained the predefined validation ranking. No convergence claim is made from the fixed budget.

M1 checkpoint `runs/ppo/20260909-003414-507606-seed655/checkpoints/best_validation.pt` was selected at 334,848 attempts, SHA256 `ab7f1630505a28ff9f509de4e888e5459e315d1be63926dcae3cc14022872d0a`. Its saved rehearsal is `runs/rehearsals/20260909-003414-347814-M1-policy-adaptation-M1-updated-PPO`; package `policies/M1-PPO-adaptation-study-20260909-003414.zip`. CSV SHA256 `21f04c5f3d2b65b97eee4d636d0b146c350602b92eb5c07a8ae5c54dc49560de`. Nominal simulated hit at 0.87333 s, minimum tip distance 3.916 cm, unchanged one-second whip, full 11.5 s prediction. Real performance is UNKNOWN.

The M1 recovery command peaks at **2.92688 m** continuously, at about 1.7 s into the complete CSV, outside the one-second whip. The whip command peaks at 2.14019 m; full predicted drone and cable peaks are 2.73821 and 2.74013 m. Thus the command exceeds the user's approximately 2.8 m OptiTrack preference and the predicted response has little margin. The saved export is a review artifact, not an approved height-compliant flight trajectory. No recovery or whip was silently changed to hide this issue. Resolve recovery height before selecting the next flight CSV; regenerate and preserve its exact forecast and identity if changed.

M0 control checkpoint was selected at 330,752 attempts; its rehearsal is the corresponding `...-M0-updated-PPO` folder, package `policies/M0-PPO-adaptation-study-20260909-003414.zip`. Nominal simulated hit at 0.94 s, minimum distance 3.268 cm, full 11.2333 s prediction, maximum command height 2.67105 m continuously. It is a control, not the M1 flight-policy selection.

Both exports use each job's frozen source and selected checkpoint. Complete CSV, native 30 Hz forces, package bytes and source were checked; whip scoring/export prefix difference is zero. All 289 protected original files remain unchanged. Evidence: study `export_results.json`, `final_preservation_check.json`, and `finalize.py`. No aircraft commands or new real flights occurred.
