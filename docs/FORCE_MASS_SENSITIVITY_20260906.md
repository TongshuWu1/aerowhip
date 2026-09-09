# Fixed-policy mass and thrust sensitivity

117 offline strikes were evaluated on Windows with an NVIDIA RTX 4080. No training, saved calibration, policy weights, existing logs or controller settings were changed.

Output: `runs/sensitivity/20260906-160422-force-mass/`. `summary.json` and `results.csv` summarize every trial; each initial-condition directory includes its frozen plan and all successful/failed trial trajectories and commanded/delivered force arrays. The directory timestamp uses the local machine clock. `study_source.py` and model/task/PPO configuration copies preserve the evaluation recipe. Reproduce with `tools/check_force_mass_sensitivity.py`.

Checkpoint: stopped run `20260906-174733-192436-seed652`, `terminal.pt`, 39,936 saved attempts. SHA-256: `214fb97211c1b57a04a7352d9629a0369c185a97efc0ff665e0336b5188d303d`, checked unchanged after evaluation.

One nominal start/target plus eight independently sampled pairs within their 5 cm spheres (seed 90606) were used. Each starts with a perfectly vertical cable and zero velocity. One nominal-model plan is generated per pair; every mass/thrust scenario executes that exact sequence and cutoff with no policy calls during execution. The actual plant mass alone is perturbed around 159 g. Cable/marker masses remain fixed. A uniform delivered-thrust multiplier is applied to the total XYZ force, after the conceptual controller gravity addition, throughout the strike. This is a residual force-calibration error, not an experimentally identified voltage model. The gravity-subtracted interface conversion cannot itself correct this actuator mismatch.

| Actual drone mass error | Delivered thrust loss | Valid hits / 9 | Mean drone Z difference at cutoff |
|---|---|---|---|
| 0 g | 0% | 9 | 0 cm |
| -3 g | 0% | 5 | +6.8 cm |
| +3 g | 0% | 5 | -6.6 cm |
| -5 g | 0% | 0 | +11.4 cm |
| +5 g | 0% | 0 | -10.8 cm |
| 0 g | 2% | 3 | -7.8 cm |
| 0 g | 5% | 0 | -19.6 cm |
| 0 g | 10% | 0 | -39.2 cm |
| 0 g | 15% | 0 | -58.8 cm |
| +3 g | 5% | 0 | -25.8 cm |
| +3 g | 10% | 0 | -45.1 cm |
| +5 g | 5% | 0 | -29.9 cm |
| +5 g | 10% | 0 | -48.9 cm |

Z differences are relative to the corresponding nominal trajectory at its frozen cutoff, not displacement from takeoff. Valid hits use the existing target contact, speed, direction and first-contact rules. All 117 numerical executions completed; a completed simulation can still miss the target. The nominal pair's cutoff is 0.81 s. At that pair, 5% thrust loss increased closest tip approach from 0.31 cm to 17.36 cm and shifted the drone down 19.44 cm relative to nominal.

This small paired diagnostic does not estimate real-flight success rates or establish a battery-voltage threshold. No takeoff, prelaunch hover drift, motor lag, attitude dynamics, noise, collision response or recovery performance was evaluated. Thrust reductions are assumed fractions, not equivalent percentage voltage reductions. Training uncertainty coverage is not inferred from these trials, and this is not independent paper evidence.

Recommendation: first measure the actual vehicle mass and force response across operating battery voltages and strike force levels. Keep controller/export mass conventions consistent and correct the delivered-thrust mapping below the fixed policy. Then rerun a larger sensitivity study with measured residual errors; policy adaptation or retraining may be needed if those errors remain substantial. No compensation or retraining was automatically applied by this study.
