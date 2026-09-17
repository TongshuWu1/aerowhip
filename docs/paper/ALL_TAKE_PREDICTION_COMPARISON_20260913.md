# Frozen-model tip prediction on all M0/M1 takes

All five M0 and five M1 recordings were predicted by the frozen M0, M1 and
selected M2 models, using each take's recorded commands and the same measured
initialization information. No fitting, model selection, command optimization,
or physical flight was performed. Old M2 physical data remain excluded.

The mean of the ten per-take 3D tip-prediction RMSEs, over [0, 34/30] s, is:

| Model | Mean tip-prediction RMSE (cm) |
|---|---:|
| M0 | 19.65 |
| M1 | 8.66 |
| M2 | 8.37 |

This is a retrospective development comparison: takes were previously used in
fitting, replay, model selection or development checks. It does not establish
independent generalization or improvement on every individual take.

The original masks and clock alignment were retained. All scored tip samples
were valid for these ten windows. Each take shares commands, grid, truth, masks,
projected cable state and measured vehicle physical state across models.
Effective hover compensation and frame alignment use the same causal procedure
with each model's coefficients. Rollouts receive no later measurement resets.

All 30 RMSEs were independently recomputed from saved prediction arrays; maximum
difference was 2.78e-17 m. Shared arrays were checked for equality. Four previously
reported M1/M2 predictions on M1_003/005 were reproduced within 1e-12 m.
Original model, input and code hashes were verified after inference.

Reproduction command:

```powershell
.venv/Scripts/python.exe tools/evaluate_paper_tip_predictions.py --output runs/evaluation/NEW_UNUSED_DIRECTORY --device cuda
```

The evaluation preserves every per-take metric and prediction. The paper uses
only the three overall means requested by the user. Exact inputs, output paths,
hashes and validation are recorded in the adjacent JSON record.
