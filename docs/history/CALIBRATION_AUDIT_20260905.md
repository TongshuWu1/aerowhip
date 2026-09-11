# Calibration audit — 5 September 2026

Later evidence: [Comprehensive fitting audit](FITTING_AUDIT_20260905.md) reports 51 model/solver variants, causal initialization checks and five-second comparisons. This document preserves the earlier investigation; use the later report for the current fitting recommendation.

The large measured-versus-predicted discrepancy remains unresolved. A numerical damping defect was corrected, and a fresh candidate was fitted, but the improvement is small. The active physical parameter baseline was not replaced.

## Measurement reference

The user clarified that all drone markers are mounted on the top prop-guard plane and that the OptiTrack tracked center lies at that top plane. The ruler measurement is approximately 55 mm from that plane down to the cable attachment, then approximately 63 mm from the attachment to c1. These are separate intervals. The configured attachment-to-c1 interval is 63 mm, subdivided into two simulation edges.

The parser reads the position and orientation of rigid body `cf_7`. Processing computes attachment position as `p_cf7 + R_cf7 @ [0, 0, -0.055]`. Retain this transform based on the user's clarification. The origin need not coincide with any individual marker for a height measured from their plane to apply. The configured lateral alignment and body-axis direction remain model assumptions, not quantities newly calibrated by this audit.

The first 500 exported frames of fig8_003 contain five rigid-body markers. Their median body-frame offsets from the rigid-body origin, in mm, are Marker1 (-68.26, 15.73, 4.12), Marker2 (18.19, 40.19, -1.51), Marker3 (-30.38, -3.16, 1.12), Marker4 (71.35, -9.80, -4.59), and Marker5 (9.13, -42.81, 0.97). None coincides with that origin, but these primarily lateral offsets do not establish an error in the 55 mm vertical offset. The earlier interpretation that a specific marker must coincide with the origin was incorrect.

Measured adjacent-marker chords also differ from configured arc lengths. Curved cable spans can have shorter chords; these observations alone do not justify replacing physical cable lengths.

## Corrections and checks

- Replaced the bent-joint corotational damping calculation with an equivalent stable tangent-space calculation in both Torch and fused CUDA. The old eigenvalue floor depended on floating-point precision and substantially distorted small-angle damping in float32.
- Changed offline physical fitting to float64 with the direct damping solver.
- Preserved saved parameter precision when unchanged GUI fields are applied. Display rounding no longer silently changes those values.
- Added fit validation changes and solver information to the UI, plus the initialization tip displacement.
- Fixed relative-root handling for command-line data jobs.
- Full test suite: 105 passed; two existing Torch JIT deprecation warnings.

Corrected-solver checks on three initial windows of fig8_003 gave tip RMSE of 40.020 mm (CPU float64), 40.025 mm (CPU float32), and 39.797 mm (GPU float32). Thus numerical agreement improved without eliminating the physical prediction error.

Increasing constraint iterations from 4 to 12 or substeps from 3 to 24 did not remove the discrepancy on three sampled takes. Constraint length errors were near double-precision roundoff. Changing initial velocity differentiation from 7 to 31 frames changed sampled tip RMSE by less than 1 mm within each take. These are local sensitivity checks, not proof of global convergence.

## Fresh fit

Job: `data/workflow_jobs/20260905-052802-619476-fit`.
Training: 278 windows across five takes. Validation: 94 windows across two separate takes. Horizon: 0.7 seconds. The protected test take was not used. Parameters were selected using training data only.

Both columns below use the corrected solver and the same data. Metrics are equal-take averages.

| Metric | Active parameters | Candidate parameters |
| --- | ---: | ---: |
| Training marker RMSE | 35.357 mm | 35.177 mm |
| Training tip RMSE | 51.954 mm | 51.568 mm |
| Validation marker RMSE | 60.026 mm | 58.653 mm |
| Validation tip RMSE | 93.219 mm | 91.768 mm |

Candidate EI: 1.7481368661988278e-6 N m². Candidate Cb: 1.0709455109087161e-4 N m² s. These are conditional estimates under the current geometry, attachment reference, and pivot model; they should not be presented as established physical constants. The candidate is saved for review and has not been applied.

## Evidence and remaining work

`data/calibration_audits/20260905_initial_audit/` contains geometry, velocity sensitivity, precision, and convergence results. `corrected_solver/` contains the post-correction convergence audit. Pre-correction solver sources are preserved there for reproducibility. The fit job contains its source hashes, configuration snapshots, candidate grids, metrics, and measured/active/candidate validation trace.

The exploratory attachment offset estimate assumes a straight 63 mm first span and reaches its -20 mm lateral bound. It is confounded by cable curvature and origin calibration and must not be applied.

Retain the clarified attachment geometry. Further investigation should examine initialization projection and boundary/physical effects; this audit does not identify one confirmed explanation for the full mismatch. The clarification itself changes no numerical configuration and requires no repeat fit. Policy training has not been restarted on this uncertain calibration.
