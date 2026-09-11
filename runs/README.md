# Experiment runs and evidence

| Folder | Contents |
|---|---|
| `adaptation/` | Prepared fitting inputs, fitted M0/M1/M2 models and fit diagnostics |
| `mppi_pva/` | MPPI planning jobs, frozen settings, model copies and predictions |
| `ppo_pva/` | PPO training runs, checkpoint selection and logs |
| `rehearsals_pva/` | Saved PVA commands, original predicted motion and recovery |
| `data_review/` | Recording reviews and comparisons against original forecasts |
| `evaluation/` | Comparisons across frozen models and common recordings |
| `flight_packages/` | Reviewed flight-package staging and metadata |
| `pva_jobs/` | Background worker job records |
| `audits/` | Verification reports and source/evidence hashes |

Keep run IDs and saved files unchanged: they bind paper evidence to its inputs.
Use `exports/` for files handed to the flight program, `data/flight_batches/`
for paired recordings and flown commands, and `paper/figures/` for manuscript
figures. Opening this folder does not authorize restarting a job.
