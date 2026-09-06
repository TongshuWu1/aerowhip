# Workflow tools

These scripts run from the repository root using `.venv/Scripts/python.exe`.

| Task | Script |
|---|---|
| Export private development and selected-PPO deployment bundles | `export_lab_transfer.py --output <new-folder-outside-project>` |
| CPU/CUDA environment check without training | `lab_preflight.py --device cpu` (or `--device cuda --batch 1024`) |
| Build a portable source-only review bundle | `build_source_release.py` |
| Flight template, import, replay, physical fit, force correction | `adapt_flight.py` |
| Process preliminary OptiTrack/Motive recordings | `process_takes.py` |
| Build cable-state/force datasets | `build_force_dataset.py` |
| Existing pivot material-parameter fitting | `fit_pivot_cable.py` |
| Start a fresh paired PPO/SAC comparison | `run_training_comparison.py` |
| Export comparison figures | `plot_training_comparison.py` |
| Monitor the active comparison for reward plateaus | `early_stop_comparison.py` |
| Evaluate an explicit checkpoint on reserved scenarios | `evaluate_selected_strike.py` |
| Physics throughput and memory checks | `benchmark_gpu_physics.py`, `benchmark_comparison_memory.py` |

**Do not relaunch an existing comparison to check its status.** Its `status.json`, `early_stopping_status.json` and per-run validation history are the live records. Comparison preparation creates new runs.

Calibration audits and active comparison tools remain. One-off reward searches, old strike reports and completed migration scripts have been retired outside the repository. `source_snapshot.py` supplies experiment source capture without importing an old reward-tuning launcher. Historical frozen experiment source remains with the corresponding result for provenance.

The UI’s recommended calibration workflow uses the geometry/drag audit result. Flight adaptation defaults to a bounded drag-only update; it does not automatically refit weakly identified EI/internal damping or apply a candidate.
