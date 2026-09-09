# Complete fitted candidate — not active

`model.json` enables the fitted cable physics, dissipative cable NN, effective full-state attachment response and drone NN together. Both checkpoints are stored beside it; the model uses paths relative to the project root. This is a model bundle for future policy training, not an exported PPO policy or a flight command file.

The current calibration and selected PPO were not replaced. The candidate passed a 1,024-case Windows/RTX 4080 numerical probe, without training the actor. Accuracy is mixed: pooled cable-tip RMSE is 6.15 cm versus the previous calibration's 6.14 cm; on historical whips it is 7.69 cm versus 7.27 cm. Passing the runtime check does not establish better hitting or flight readiness.

The parent job contains the input audit, masks, fits, rotated development checks and the complete-model review. See `docs/HISTORICAL_MODEL_FITTING.md` and `docs/HISTORICAL_MODEL_FIT_RESULTS.md` in the repository. All historical recordings are development data. Future flights must assess a frozen candidate and policy prospectively.

The cable candidate uses zero external scalar drag and a zero-initialized, learned dissipative NN. It does not initialize from 0.3 s⁻¹. Measured masses remain drone 157 g plus cable/markers 18 g. The full-state predictor represents this fixed loaded setup and receives no additional cable-reaction force.

Do not copy this model over `config/model.json` without reviewing its regression. The existing PPO was trained under different saved physics; use a deliberately named new run if this candidate is selected later.
