# Existing-data initialization and parameter audit

The controlled comparison is complete. Seven development recordings supplied 70
common two-second windows, with physical parameters frozen and the neural residual
disabled. The protected take was not opened. The recordings contain meaningful
motion variety; no new recording was required.

| Initializer | Validation marker RMSE | Validation tip RMSE |
| --- | ---: | ---: |
| Centered offline reference, includes future samples | 62.06 mm | 103.95 mm |
| Past-only polynomial | 62.74 mm | 105.04 mm |
| Past-only DER history fit | 70.93 mm | 121.34 mm |

These are full-window RMSE values averaged equally across the two validation takes.
All methods share starts. Sampling balances initial motion intensity; these numbers
are not directly comparable to previous jobs with different windows or weighting.

**Implemented decision:** new fitting jobs initialize velocity using a quadratic
fit to the preceding 11 position samples, including the current sample (100 ms at
100 Hz). Initial positions remain measured and positions/velocities are projected
to cable constraints. Grid search, differentiable refinement and validation use
the same estimator. The centered force-analysis derivatives are preserved. Old
job configurations retain their old initialization, and warm-start refinement
rejects changes in initialization settings. The DER-assisted estimator remains an
offline experimental method because it worsened prediction in this comparison.

The physical parameter audit then fixed the causal initializer and evaluated 49
log-grid EI/Cb combinations plus the previous fitted pair. 36 pairs were within
1% of the best training loss, spanning the full EI search range. Individual takes
favored substantially different coefficients. The selected grid candidate improved
validation marker RMSE only from 62.74 to 62.61 mm and reached the lower EI boundary.
It was not applied. The 1% tolerance is descriptive, not a confidence interval.

This supports a practical causal initializer, but it does not resolve the larger
model mismatch or establish true material properties. The next physical-model work
should investigate boundary/geometry and model discrepancy using the existing
recordings, rather than simply increasing optimizer updates.

Detailed artifacts:

- [Initialization protocol and results](../../data/calibration_audits/20260905_initialization_2s/README.md)
- [Prediction-error figure](../../data/calibration_audits/20260905_initialization_2s/prediction_error.pdf)
- [Parameter sensitivity report](../../results/20260905-235530-029769-ppo-sac-500k/unused_artifact_archive/data/calibration_audits/20260905_causal_parameter_surface/README.md)
- [Parameter loss surface](../../results/20260905-235530-029769-ppo-sac-500k/unused_artifact_archive/data/calibration_audits/20260905_causal_parameter_surface/loss_surface.pdf)

Verification: 117 tests passed, including causal data-access behavior, endpoint
velocity recovery, initial-state gradients checked against finite differences,
batched candidate equivalence, and existing fitting/runtime integration. Four
existing Torch JIT deprecation warnings remain. No policy was retrained, no physical
baseline was replaced, and no new live MoCap interface was introduced.
