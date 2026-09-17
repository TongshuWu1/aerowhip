# Each model on its own flights

| Model / own flights | Whips | All-10-marker RMSE (cm) |
|---|---:|---:|
| M0 | 11 | 11.808 |
| M1 | 12 | 4.125 |
| M2 | 11 | 5.182 |
| M3 | 12 | 4.319 |
| M4 | 12 | 4.165 |

RMSE uses the 3D prediction error over all valid samples of all ten cable markers, with no extra tip weight. Each whip contributes equally to the reported mean. Scoring is from onset to the fixed planned strike at 1.113800079 s; no post-strike motion is included. Predictions use the recorded commands and causal measured pre-whip initial state.

| Model | c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 9.662 | 10.011 | 10.225 | 10.450 | 10.706 | 11.151 | 11.716 | 12.834 | 14.152 | 15.612 |
| M1 | 3.428 | 3.465 | 3.421 | 3.352 | 3.441 | 3.603 | 3.959 | 4.546 | 5.207 | 5.915 |
| M2 | 4.686 | 4.717 | 4.821 | 4.928 | 4.991 | 5.116 | 5.226 | 5.359 | 5.583 | 5.915 |
| M3 | 3.522 | 3.610 | 3.632 | 3.607 | 3.770 | 3.970 | 4.268 | 4.823 | 5.276 | 5.760 |
| M4 | 3.628 | 3.672 | 3.623 | 3.672 | 3.620 | 3.878 | 4.081 | 4.586 | 5.078 | 5.293 |

Per-marker values are centimetres; c10 is the tip. Each model is tested only on its own subsequently collected flights, which were not used to fit that model. Commands and flight collections differ across rows, so this describes the performance of each iteration rather than isolating the model change on identical flights. Missing observations remain masked. No M5 fitting was performed.
