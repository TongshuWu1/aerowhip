# Code map

The lab interface is an explicit workflow around the existing numerical code.

| Entry point | Responsibility |
|---|---|
| run_lab.py | Desktop launcher |
| deployment/lab_gui.py | Guided operator pages and asynchronous job display |
| deployment/lab_workflow.py | Studies, fixed take roles, reviews, model updates, exports and results |
| deployment/lab_seed.py | Verified retained-M0 import and portable assets |
| tools/lab.py | Explicit command-line actions used by the desktop |
| tools/check_lab.py | Read-only environment/baseline diagnostics |
| planning/pva_job.py | Frozen model/settings and offline planning jobs |
| deployment/pva_rehearsal.py | Complete PVA/recovery prediction and packaging |
| experimental_data/whip_adaptation.py | Raw pairing, review, causal preparation |
| experimental_data/whip_full_data.py | Full update inputs, replay and hash checks |
| experimental_data/whip_full_fit.py | Staged full identification and evaluation |
| simulator/research_execution.py | Command-to-drone-to-cable predictions |
| simulator/cable/ | Cable mechanics and residual models |

The guided app records local relative paths and uses the running Python
interpreter. Expensive actions run in explicit child processes. Startup/import
does not fit or optimize. New output metadata may be made portable before
publication; original raw inputs and provenance remain preserved.

experiments/ contains the operator's study state, runs/ contains immutable
computation jobs, exports/ contains files for the external flight program, and
workspace/baseline/ contains the retained baseline.

The older run_simulation.py entry opens research tools. Existing force-based,
PPO/SAC and numerical test paths remain available for compatibility; they are
not the guided deployment path.
