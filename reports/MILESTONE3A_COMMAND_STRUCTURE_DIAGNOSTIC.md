# Milestone 3A — Corrected FullState Structural Diagnostic

## Outcome

The acceleration-plus-gravity body-z construction has the correct tilt sign and smooth behavior on all four physical takes. This check used the pre-fit nominal gains only to test structure; it is not a parameter-accuracy result.

| Take | roll corr. | pitch corr. | acceleration vs measured tilt corr. | roll RMSE [deg] | pitch RMSE [deg] | best diagnostic lag [s] | max R_des step [deg] |
|---|---:|---:|---:|---:|---:|---:|---:|
| fig8_001 | 0.956 | 0.914 | 0.754 | 1.27 | 4.56 | -0.030 | 3.56 |
| fig8_002 | 0.953 | 0.882 | 0.870 | 2.74 | 5.02 | 0.010 | 5.44 |
| fig8_003 | 0.945 | 0.863 | 0.886 | 6.07 | 7.23 | 0.010 | 8.78 |
| osc_001 | 0.331 | 0.920 | 0.681 | 2.85 | 7.09 | 0.170 | 6.28 |

All command-valid samples have exactly zero `q_cmd` x/y and exactly zero `omega_cmd`. The standard cmdFullState contract defines omega as body-frame rad/s; because every recorded value is zero, this clarification does not alter the diagnostic or fit. No common repeatable delay is established by the structural check.

Diagnostic plot: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\reports\milestone3a_stage_a_data\desired_vs_measured_attitude.png`
