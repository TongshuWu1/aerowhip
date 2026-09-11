# Documentation

Start with the current guides below. Old test reports, run diaries, obsolete UI
instructions and superseded designs are separated into the [history archive](history/README.md).

## Start here

- [Authoritative whole-system paper audit: theory, actual implementation, evidence and related work](PAPER_PIPELINE_AUDIT.md)
- [Clean paper experiment: selected pipeline and release conditions](PAPER_EXPERIMENT_PROTOCOL.md)

The two guides above supersede conflicting paper proposals/status notes below.
The clean study is designed but not yet released for collection.


- [M0/M1/M2: complete system comparison, latest flight evidence, related work and paper design](M0_M1_M2_SYSTEM_COMPARISON.md)

- [M2 aggressive MPPI: stronger contact-speed reward and saved comparison](M2_AGGRESSIVE_MPPI.md)

- [Two-target continuous-whip trials, ordered contacts and replay](TWO_TARGET_WHIP.md)

- [Frozen staged identification method and separate M2 refit](FROZEN_SYSTEM_IDENTIFICATION.md)

- [Historical combined-fitting proposal, superseded as the active pipeline](SYSTEMATIC_ADAPTATION_PROTOCOL.md)

- [Why M2's component gains do not consistently improve the full prediction](M2_REGRESSION_ANALYSIS.md)

- [Current M1-full to M2 adaptation and soft strike-speed objective](M1_TO_M2_ADAPTATION.md)

- [Training-window duration, residual contribution and stopping: paper/code review](TRAINING_HORIZON_RESIDUAL_REVIEW.md)

- [Complete adaptation implementation, GPU checks and measured run](FULL_MODEL_ADAPTATION.md)

- [Project goal and complete adaptation: drone, cable and residuals](ADAPTATION_MODEL_CONTRACT.md)

- [Learning drone response: literature, math and adaptation diagnostics](DRONE_RESPONSE_ADAPTATION.md)

- [Drone fitting, optimized commands and predicted flight: experiment audit](DRONE_COMMAND_CHAIN_AUDIT.md)

- [PPO with the selected MPPI objective: training, parity and replay](PPO_MPPI_OBJECTIVE.md)

- [Current M2 PPO: exact selected single-target MPPI reward and live training](M2_PPO_MPPI_MATCH.md)

- [Theory, implementation and paper-readiness audit](PAPER_READINESS_REVIEW.md)

- [M0 → M1 → M2: model comparison UI and sim-real evaluation protocol](SIM_REAL_EVALUATION.md)

- [M0→M1 adaptation: reviewed raw data, small physical update and verification](M0_TO_M1_ADAPTATION.md)

- [Selected M2 MPPI package: exact command and frozen forecast](../runs/flight_packages/20260910-211435-608306/README.md)

- [Fresh unseen-system MPPI check: start with preliminary recordings](NEW_SYSTEM_CHECK.md)

- [Current project status and stop decisions](../HANDOFF.md)
- [Detailed paper-writing handoff](PAPER_WRITING_HANDOFF.md)
- [Direct PVA workflow](DIRECT_PVA_WORKFLOW.md)
- [Current architecture](ARCHITECTURE.md)
- [Installation](INSTALL.md) and [local setup/transfer](LAB_SETUP.md)

## Models, data and planning

- [Whip success-condition audit: paper definitions versus our custom gates](WHIP_SUCCESS_CONDITION_REVIEW.md)

- [MPPI timing and strength search: both baselines and unchanged reward/model](MPPI_TIMING_SEARCH.md)

- [Preferred-fold MPPI objective: tested shape preference and one development-M0 trial](MPPI_PREFERRED_FOLD.md)

- [Preferred whip wave: paper evidence, archived-motion comparison and reward redesign](WHIP_WAVE_REWARD_REVIEW.md)

- [Complete-whip MPPI: development M0, control-point search and explicit score components](MPPI_WHOLE_WHIP.md)

- [Current MPPI pullback task and verified simulation](MPPI_PULLBACK_20260909.md)
- [Loaded-drone pose model](NOMINAL_DRONE_POSE_RESPONSE.md)
- [Geometry and reference points](GEOMETRY_COORDINATE_CONVENTIONS.md)
- [Data lifecycle and recording](DATA_LIFECYCLE_AND_RECORDING_GUIDE.md)
- [Flight data and replay](FLIGHT_ADAPTATION_QUICKSTART.md)
- [Exact saved-forecast comparison](ADAPTATION_CHECK.md)
- [Retrospective height normalization](HOVER_HEIGHT_CALIBRATION.md)
- [Current preliminary1 preparation and GPU fit](PRELIMINARY1_FIT.md)
- [Preliminary1 fitting audit: initialization and optimization concerns](PRELIMINARY1_FIT_AUDIT.md)
- [Preliminary1 method comparison with published identification and whip work](PRELIMINARY1_METHOD_COMPARISON.md)
- [Completed cable-only pilot: initialization, physical search and separate-take results](PRELIMINARY1_CABLE_PILOT.md)
- [M0 development: gradient resolution, simple damping and the future whip-data loop](PRELIMINARY1_DAMPING_RESOLUTION.md)
- [Future fitting requirements](FUTURE_ADAPTATION_FITTING.md)
- [Recovery design](CURVED_RECOVERY_20260908.md)
- [Measured CUDA performance](PVA_PERFORMANCE_20260909.md)

## Paper references

- [Candidate contributions, equations and evidence ledger](PAPER_WRITING_HANDOFF.md)
- [Between-trial research proposal](RESEARCH_PROPOSAL_ADAPTIVE_AERIAL_WHIP.md)
- [Identification research design](IDENTIFICATION_RESEARCH_DESIGN.md)
- [Adaptation literature and design](ADAPTATION_LITERATURE_AND_DESIGN.md)
- [Adaptation related work](ADAPTATION_RELATED_WORK_20260905.md)
- [Detailed literature notes](RELATED_PAPERS_DETAILED_REVIEW.md) and [bibliography](related_papers.bib)
- [Historical equal-budget M0/M1 study protocol](M1_POLICY_ADAPTATION_PROTOCOL.md)
- [Reproducibility](REPRODUCIBILITY.md) and [publication scope](PUBLICATION.md)

Dated technical references retain their experiment-specific caveats. Their old
run-status sentences do not override the current handoff. Raw data, saved runs,
tests, code and artifact provenance are preserved by this documentation cleanup.
