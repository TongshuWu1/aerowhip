# Constrained physical fitting candidate — 5 September 2026

Completed the recommended geometry-first calibration on the existing development recordings. This is a provisional research candidate. The active baseline has not been replaced and the protected test has not been opened.

## Recommended candidate

- Body-frame attachment offset: [6.655, -12.874, -55.000] mm.
- Attachment height fixed at the reported 55 mm; first cable span fixed at 63 mm. Other measured lengths and masses retained.
- EI: 2.8399397e-08 N m²; Cb: 3.7523085e-05 N m² s; effective external decay rate: 0.3 /s.
- For comparison, joint optimization gave EI 0.00071428571, Cb 7.1428571e-05 and decay rate 0.325, but selection also considers development validation.
- Pivot attachment; 12 physics substeps per 10 ms. Geometry and material values are conditional model estimates, not independently measured constants.
- External decay is applied to free cable vertices in the coupled point-mass runtime. It does not silently add the same drag rate to the drone.

## Prediction results

| Horizon | Model | Fit marker RMSE [mm] | Development validation marker RMSE [mm] | Validation tip RMSE [mm] |
| --- | --- | ---: | ---: | ---: |
| 2 s | Joint physical fit | 27.16 | 41.09 | 70.27 |
| 2 s | Joint fit + neural residual | 26.85 | 40.91 | 70.05 |
| 2 s | Constrained geometry + drag seed | 27.69 | 37.93 | 66.34 |
| 2 s | Prior fitted physics | 47.24 | 62.74 | 105.04 |
| 5 s | Joint physical fit | 44.20 | 61.50 | 106.83 |
| 5 s | Joint fit + neural residual | 43.91 | 61.43 | 106.77 |
| 5 s | Constrained geometry + drag seed | 46.16 | 58.68 | 100.21 |
| 5 s | Prior fitted physics | 76.66 | 103.03 | 175.88 |

Errors average Euclidean position RMSE equally across takes, excluding time zero. Models within a horizon share complete-valid prediction windows. Two- and five-second cohorts differ. `metrics.csv` supplies per-take values and sample counts. Root motion is measured input; no future cable observations correct the prediction.

## Geometry calibration

Float64 L-BFGS with strong-Wolfe line search fitted only the lateral offset, using equal training-take weight, a robust first-span feasibility term and a weaker quiet-tangent extrapolation term. The lateral offset was bounded to ±20 mm with a weak zero-centered prior. Quiet tangent extrapolation remains an approximation. Sensitivity profiles used heights 52/55/58 mm and spans 60/63/66 mm; those are assumed ranges, not measured uncertainty. They did not select the height or span. Leaving out individual training takes yielded similar lateral offsets.

On fig8_003, the fraction of frames with attachment-to-c1 chord exceeding 63 mm plus 2 mm tolerance fell from 29.93% to 7.97%. On osc_003 it fell from 9.22% to 1.56%. This improves consistency but does not eliminate measurement/geometry uncertainty.

## Physical optimization and the gradient limitation

The initial 61-candidate search was followed by three bounded 3D pattern-search passes, each testing 27 neighboring EI/Cb/drag combinations against full two-second rollouts. Selection used only training loss, equally weighting takes and nonempty initial-motion bins. The search extended EI to 0.004 N m² as an upper bound with the finer integrator, and evaluated values above the initial 0.001 bound. The final candidate is not at that extended bound. A finite search is not proof of a unique optimum.

The initial grid has 4 combinations within 1% of its minimum training loss. Discrete leave-one-training-take-out minima are recorded in `parameter_sensitivity.json`; they should be inspected before interpreting the fitted quantities as reusable material constants.

An attempted Adam refinement through long rollouts was rejected. At the real fig8_002 starting frame 35 over 0.5 s, the log-EI adjoint was approximately -6.45, whereas central finite differences were +0.0064 (step 1e-4) and +0.204 (step 1e-5). The finite-difference estimates themselves vary with step size. This indicates unsuitable local sensitivity for optimization at that horizon; it does not yet locate one confirmed kernel defect. The interrupted four-update history is preserved. Its evaluated candidate was worse than the grid seed and was not selected.

Short checks at 0.05 and 0.1 s passed on two real initial states for log EI, log Cb, log drag and a learned-residual gain. Each used two finite-difference steps. Those are local checks, not a guarantee for all states. The simulator is differentiable in code, but this experiment does not support a claim of reliable long-horizon differentiable identification or force-plan optimization yet.

## Neural residual

Physical parameters and geometry were frozen. A bounded 32-unit MLP was trained with Adam on random 100 ms segments inside the stratified training windows. Each segment starts from measured markers and an 11-frame causal velocity estimate. There is no state reset inside a segment. Such supervised training does not introduce feedback into policy execution: evaluation still runs continuously for two or five seconds from one initial state. Corrections are bounded at 0.5 m/s² per coordinate and regularized on sampled initial states. Updates with excessive gradient norm are skipped and logged. Checkpoint selection compares full two-second training rollout error, including the zero-residual model.

The post-hoc development review recommends **Constrained geometry + drag seed**. A residual must improve validation mean marker and tip error at both horizons and avoid worsening any validation take’s marker RMSE by more than 5%. This is development model selection, not an independent test.

The updated material model must also match or improve the simpler geometry-plus-drag seed’s validation marker and tip errors at both horizons. Otherwise retain the earlier material values and the constrained geometry/drag correction. A small training improvement alone does not establish a better physical baseline. `recommended_model.json` contains the resulting choice.

## Remaining limits and next use

Across the seven takes, doubling substeps from 12 to 24 changes all-marker predictions by 0.19–1.10 mm RMSE. This is small relative to the measured-data error, but is not a bound on continuum discretization error.

Use this candidate for simulation comparisons, not as certified hardware calibration. Preserve the protected recording for a frozen final protocol. Investigate the long-horizon sensitivity problem before relying on gradients for rapid policy/force-sequence adaptation. Validate command-to-drone motion separately; the cable fit alone cannot establish open-loop strike success.

Candidate files are `candidate_model.json` and `candidate_with_residual.json`; the latter references a hash-checked immutable checkpoint. `candidate_review.json` records the recommended variant. The `figures` directory contains PDF, SVG and PNG exports. Protocols, per-stage source snapshots, input hashes, loss surfaces, gradient checks and per-take results are retained alongside this report. The final full software suite passed 123 tests with four existing Torch JIT deprecation warnings.
