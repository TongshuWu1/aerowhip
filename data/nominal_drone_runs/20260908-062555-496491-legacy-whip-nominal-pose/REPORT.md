# Independent attitude mapping correction

All twelve pose assessments finish without the earlier attitude-domain
failure. Final maneuver orientation RMS is 2.16/2.17/2.20 degrees; left-out
RMS is 3.44/2.45/1.72 degrees. Translation parameters remain frozen, and the
separate post-hold position mismatch remains. This is a development candidate,
not a complete M0 or selected flight/export model.

Read the [full report](../../../docs/ATTITUDE_MAPPING_CORRECTION_20260908.md)
and the archived `RESULTS_REPORT_SNAPSHOT.md` in this folder.

Evidence: `protocol.json`, `mask_report.json`, `results.json`, `comparison.json`,
`verification.json`, `test_verification.json`, `pytest.txt` and per-fold files.
`review_comparison.py` reproduces the native script/command transition audit
and compares the old and new predictions. `nominal_fit_review.png` shows the
recorded and predicted trajectories with the maneuver shaded.

No active configuration or selected PPO changed. No NN/cable fit, training,
data retirement or flight occurred.
