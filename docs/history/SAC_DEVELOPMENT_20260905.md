# SAC development run

## 30 Hz update

The user requested 30 Hz after the first pilot started. The 10 Hz pilot stopped
cooperatively at 3,072 attempts and its checkpoints remain available. The new
active run is `runs/sac/20260905-230045-842321-seed651`, displayed as
**SAC · 30 Hz · searched strike initialization**. It starts fresh, with a budget
of 8,192 attempts. The SAC critic's action-prefix dimension changes with the
horizon, so the old critic and replay are not resumed.

New-run defaults for both algorithms are 30 Hz commands and 150 Hz physics,
with eight internal DDER substeps. The internal solver step remains 1/1200 s.
Each command lasts five physics steps; the 20 ms follow-through lasts three.
The one-second horizon now has 30 actions. Each original prior phase lasts
three actions, preserving its physical duration. Fitted physical parameters
are unchanged; the numerical configuration has its own baseline version and
the previous defaults are archived in `data/timing_versions`.

Task & Rewards now displays policy frequency in Hz, avoiding rounding 1/30 s
when saving. Existing run snapshots and checkpoints keep their original timing.
The new 30 Hz result must be evaluated before claiming comparable performance
to the selected 10 Hz PPO result.

## Original 10 Hz pilot

Run: `runs/sac/20260905-224544-763195-seed651`.
Display name: **SAC · searched strike initialization**.

This is an 8,192-attempt CUDA pilot with 1,024 parallel environments. Its
immutable launch configuration is in the run folder. The project-wide SAC
defaults have not been promoted to this experimental configuration.

The task, physical baseline, reward, uncertainty assumptions and validation
scenarios match the selected PPO protocol. The actor starts from the same
8,192-attempt CEM force-sequence search used to initialize PPO; it does not
inherit trained PPO weights. Search cost must be reported separately from SAC
training attempts. This is one training seed, not a publication-level comparison.

The actor privately plans against the nominal model. The independent plant
executes the frozen force sequence once, with a 20 ms predetermined follow-through
and then PID recovery. The strike horizon is 1 s and the scored recovery allowance
is 15 s. The real plant provides no state or hit feedback during the strike.

Exploration samples around the searched action prior. Initial latent standard
deviations are 0.05, 0.02 and 0.05 on XYZ. The actor learning rate is 1e-5;
the critic rate is 3e-4. The first 512 gradient updates train only the critics
(two 1,024-attempt collections at 256 updates each). Thereafter SAC updates
the actor, twin critics and entropy temperature normally. Target entropy is
-7 and initial temperature is 0.02. These are pilot choices, not established
optimal settings.

Initial deterministic validation: 55/128 valid impacts (42.97%), 100% recovery,
5.69 m/s mean speed among successful impacts, and 0.815 m mean peak drone
travel including recovery. This is the initialization baseline, not evidence
of SAC learning. Validation uses fixed development seed 90651. No reserved
evaluation has been performed for this SAC pilot.

The SAC page shows its own learning plots, actual validation replay and
Run latest policy / Execute controls. Restarting the UI loads the updated
initialization label without stopping the detached training worker. Avoid
starting another run while this pilot is active. The worker stops automatically
at its configured budget, and the existing Stop button requests a cooperative
stop preserving the last durable checkpoint.

After completion, select checkpoints using development validation, then evaluate
the selected checkpoint on the reserved scenarios. Report impact success,
impact plus settled recovery, successful impact speed and time, and mean/p95/max
drone travel. Do not select a checkpoint using the reserved result. The generic
`tools/evaluate_selected_strike.py` now accepts saved SAC checkpoints as well as PPO.
`tools/plot_strike_learning.py --run RUN --output OUTPUT_STEM` exports raw learning
curves to PNG and vector PDF. Curves should be accompanied by multiple independent
training seeds before making algorithm comparisons in the paper.
