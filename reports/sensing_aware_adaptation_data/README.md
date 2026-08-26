# Sensing-aware adaptation artifacts

These files support `../SENSING_AWARE_DDER_ADAPTATION_REPORT.md`.

| Artifact | Content |
|---|---|
| `baseline_matched.json` | exact-state matched regression |
| `baseline_joint_mismatch.json` | exact-state `(0.8,0.7)` regression |
| `baseline_mppi_timing.json` | warmed MPPI and concurrent-fit timing |
| `zero_noise_causal_velocity.json` | raw versus constrained causal velocity ablation and phase observability |
| `representative_noise_sweep_projected.json` | requested 0--2 mm synthetic position-noise sweep |
| `fine_noise_detection_boundary.json` | 0.05--0.20 mm diagnostic boundary sweep |
| `mode_b_full_sensed_control.json` | initial reconstructed-state controller diagnostic |
| `sensing_diagnostics.json` | latency, dropout, cache, and trigger ablations |
| `representative_paired_control_10seeds.json` | paired Mode A/Mode B control comparison |
| `statistical_summary.json` | compact Wilson intervals, paired outcomes, prediction, and noise summaries |
| `plots/` | figures referenced by the report |

Run `python -m research_tools.sensing_aware_adaptation_study --help` from the
repository root for study stages. Plot generation uses
`research_tools/render_sensing_adaptation_report.py` and requires Matplotlib.

The full nine-condition final matrix is intentionally absent. The method was
not frozen after the diagnostic sweep showed that the present health/state
estimation path misses the representative mismatch at 0.10 mm synthetic
position noise and above.
