# Physical fitting audit — 5 September 2026

The existing recordings are sufficient to diagnose important model mismatches. They do not currently identify unique EI and damping values. More optimization iterations on the original two-parameter model are unlikely to resolve the observed error.

## Scope and protocol

Completed 51 model/solver sensitivity variants on the same 70 two-second windows (53 fit, 17 development validation), plus five models on 56 common five-second windows (43 fit, 13 development validation). All predictions were finite; no variant failed. Additional completed checks include three initialization methods and a 50-pair EI/Cb surface. The full software suite passed 119 tests, with four existing Torch JIT deprecation warnings.

Five fit takes: fig8_001, fig8_002, fig8vertical_001, osc_001, osc_002. Development validation: fig8_003, osc_003. The protected fig8vertical_002 recording was not opened. Validation has been examined repeatedly, so these are development results, not final paper test results.

Every model receives the measured attachment trajectory during prediction. This isolates conditional cable dynamics; it does not validate an open-loop force-controlled drone. Initial velocity uses only the previous 11 measured frames (including the current frame); no future cable feedback is used, except the separately labelled oracle-c1 diagnostic. Boundary, geometry and drag variants keep the prior fitted EI=2.83994e-8 N m² and Cb=3.75231e-5 N m² s fixed unless explicitly named otherwise. A gamma of 0.3 /s was selected from 0.1, 0.3 and 0.5 using training loss within each combined model family. This is a small sensitivity sweep, not a completed joint calibration.

RMSE is the Euclidean 3D position error over prediction frames, averaged as RMSE within each take and then equally across takes. Time zero is excluded from trajectory RMSE. Overlapping windows are not independent experiments. Five-second and two-second cohorts differ; compare models within each cohort.

## Measured results

| Model | 2 s fit markers [mm] | 2 s validation markers [mm] | 2 s validation tip [mm] |
| --- | ---: | ---: | ---: |
| Active EI/Cb, current geometry | 49.69 | 70.61 | 117.14 |
| Prior fitted EI/Cb, current geometry | 47.24 | 62.74 | 105.04 |
| Prior fit + frozen neural residual | 44.12 | 60.18 | 100.83 |
| Tentative attachment offset | 38.50 | 56.08 | 97.02 |
| External drag only (0.5 /s) | 39.62 | 51.33 | 84.81 |
| Offset + pivot + drag (0.3 /s)* | 26.19 | 36.51 | 64.46 |
| Offset + 31.5 mm clamp + drag* | 21.53 | 29.47 | 53.75 |

*Combined offset/drag models use 12 substeps. The same original physics at 12 substeps has validation marker RMSE 69.35 mm. Consequently, the combined improvement is not explained by using a coarser solver.

| Model | 5 s fit markers [mm] | 5 s validation markers [mm] | 5 s validation tip [mm] |
| --- | ---: | ---: | ---: |
| Prior fitted physics (3 substeps) | 76.66 | 103.03 | 175.88 |
| Same physics (12 substeps) | 85.27 | 118.07 | 198.46 |
| Prior fit + frozen neural residual | 73.92 | 100.22 | 171.76 |
| Offset + pivot + drag (12 substeps) | 44.81 | 57.25 | 98.58 |
| Offset + 31.5 mm clamp + drag (12 substeps) | 40.11 | 42.59 | 75.78 |

The offset-plus-drag pivot hypothesis improves five-second marker RMSE on every development take relative to the original fitted physics. Errors still differ substantially: its marker RMSE is 29.77 mm on fig8_003 and 84.73 mm on osc_003. The physical hypotheses are promising but do not constitute an accurate, validated baseline yet. Per-take values are exported in `per_take_5s_results.csv` and plotted in `per_take_5s`.

## What explains the mismatch?

- **Initialization is not the main fix.** Centered offline differentiation and causal differentiation gave 62.06 and 62.74 mm validation marker RMSE. Optimizing the initial state against 0.2 seconds of preceding DER motion worsened it to 70.93 mm. Keep the simpler causal estimator; the current imperfect dynamics should not pull the observed starting cable shape away from measurements.
- **EI/Cb are weakly identifiable under the current model.** In the 50-pair causal profile, 36 pairs were within 1% of the minimum training loss; the selected pair improved training loss by only 0.032%. Extending EI to 0.001 and 0.004 N m² at finer integration also failed to improve prediction. This is evidence against simply adding fitting epochs, not proof of zero bending stiffness.
- **Geometry has unresolved inconsistencies.** The original attachment-to-c1 chord exceeds the configured 63 mm arc plus a 2 mm tolerance in 29.9% of valid fig8_003 frames. Curvature alone cannot explain a chord longer than the arc. Possible causes include the attachment transform, first-span length/compliance, or measurement errors. Quiet training segments suggest a conditional body offset of approximately [7.63, -13.83, -50.41] mm, but that estimate assumes a straight first span aligned with c1–c2. It is not metrology. It actually increases the overall violation fraction in several takes. Preserve the user’s approximately 55 mm top-plane measurement as a prior, rather than silently replacing it.
- **A long hard clamp can compensate for errors.** Fixing the first 31.5 mm along the body axis lowers RMSE but leaves insufficient free span to reach c1 in many frames. At the tentative offset, this geometric inconsistency occurs in about 62% of fig8_002 and 51% of fig8_003 valid frames. Shortening the fixed segment changes the result substantially. A hard-clamp model would also require an attachment-attitude prediction in the force-driven deployment model.
- **External dissipation is a useful missing term.** The diagnostic applies exponential decay exp(-gamma dt) to world-frame nodal velocity, in addition to internal bending damping. The 0.3 /s candidate is an effective dissipation rate, not an identified air-drag coefficient or proof of downwash. Its benefit survives using 12 substeps and should be tested in a constrained joint fit with geometry and Cb.
- **Numerical resolution must be fixed before interpreting material parameters.** Increasing substeps from 3 to 6, 12 and 24 changes original-model validation marker RMSE from 62.74 to 66.97, 69.35 and 70.63 mm. Better data agreement at coarse resolution can reflect numerical damping. Constraint iterations, projection passes, reasonable differentiation choices, marker mass perturbations and small interval-length changes do not resolve the bulk discrepancy.

### Combined-model integration check

| Model | 12-substep validation markers [mm] | 24-substep validation markers [mm] |
| --- | ---: | ---: |
| offset_pivot_drag0p3 | 36.51 | 37.08 |
| offset_clamp_drag0p3 | 29.47 | 30.16 |

These data-error comparisons measure sensitivity; they are not a formal solver-error bound.

## Recommended fitting procedure using the existing recordings

1. **Calibrate measurement geometry first.** Use quiet training segments to jointly assess lateral attachment offset and first-span consistency, with ruler measurements as priors. Profile uncertainty in vertical offset and first-span length instead of freely fitting both to a single optimum. Do not replace arc lengths with median chords. Check residuals versus body attitude and speed; a single transform should explain all training takes without requiring systematic stretch.
2. **Keep causal state initialization.** Reconstruct measured markers, use an 11-frame past-only velocity estimate, and apply minimal length/velocity constraint projection. Save the initial projection error as a separate metric. Avoid optimizing the starting state to conceal model error.
3. **Use a physically defensible attachment model and adequate integration.** Start with the pivot model and compare a short compliant attachment only if geometry supports it. Use 12 substeps for development and check selected candidates at 24; choose the final resolution using a predeclared prediction-change tolerance. Match that model in subsequent policy compilation.
4. **Fit EI, Cb and a nonnegative effective drag rate together after geometry is constrained.** Use bounded log-parameter multistart optimization and differentiable multi-step rollouts, robust marker-position loss, equal take weighting, and balanced motion intensities. Begin with shorter horizons and finish at two seconds; use five-second predictions as a drift check. Profile parameter sensitivity and leave-one-training-take-out stability. Keep weakly identified quantities fixed to defensible priors or report ranges; do not claim unique material constants from a flat loss surface.
5. **Train the neural motion residual last.** Freeze the selected physical model first, then fit a small regularized correction. Require improvement across takes, speeds and long rollouts over the improved physics alone. The previous residual helps much less than the new physical hypotheses, so training it harder now risks learning geometry and numerical errors.
6. **Freeze the complete protocol before opening the protected take.** Publish per-take errors, tip and all-marker error versus prediction horizon, initialization displacement, model ablations, and parameter sensitivity. Use take-level uncertainty when defensible; do not count overlapping windows as independent trials. With one protected recording, explicitly limit generalization claims.

## Later real-flight updates

Log complete successful and failed trajectories with timestamps, commanded force, initial state and recovery transition. Offline logging does not violate open-loop execution. Fit the physical model on the force-sequence portion with clear treatment of controller transitions, retaining preliminary data to prevent drift. Failure labels alone cannot identify dynamics. Update the next open-loop force plan through differentiable rollout optimization initialized by the policy; retrain or distill the policy periodically after enough validated updates. This audit only tests cable predictions under measured root motion, so command-to-drone/attachment dynamics need separate validation before claiming end-to-end sim-to-real performance.

## Reproduction and files

Run `.venv/Scripts/python.exe -m experimental_data.fitting_audit_report` from the repository. The audit runners are `experimental_data/comprehensive_fit_audit.py`, `initialization_benchmark.py`, and `initialization_parameter_audit.py`. Each experiment directory records its protocol, input hashes, source snapshot and per-window/per-take results. All 51 two-second variants are included in `all_2s_results.csv`; geometry consistency is in `attachment_consistency.json`. Figures are supplied as PDF, SVG and 200 dpi PNG. No active physical baseline or policy was changed by these sensitivity experiments.

Figures: `model_comparison_2s`, `model_comparison_5s`, `solver_sensitivity`, `attachment_geometry`, `per_take_5s`. All are development-study figures; they must not be labelled as independent final-test results.
