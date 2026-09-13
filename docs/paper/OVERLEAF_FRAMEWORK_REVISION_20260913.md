# Overleaf framework revision — 13 September 2026

The user authorized editing the Dropbox manuscript directly, preserving its
section structure and focusing the methodology on the theoretical basis.
Routine numerical choices belong in the future experimental setup. No new
Experiments or Conclusion section was inserted.

Active manuscript:
`C:/Users/wts28/Lehigh University Dropbox/TonyLehigh Wu/Apps/Overleaf/AeroWhip-ICRA/main.tex`.
The older `paper/manuscript.tex` in the code repository is not the editing base.

## Editorial basis

### Subsequent local-correction pipeline update

The user's latest clarification is applied to the Abstract, Introduction,
subproblem wording, framework overview, optimization procedure and closing.
Initial MPPI planning fixes the desired predicted motion. Physical fitting is
unchanged; the updated model supplies local command sensitivities for bounded,
regularized Gauss-Newton correction. Backtracking verifies objective improvement
and feasibility in nonlinear simulation. The manuscript no longer applies MPPI
sampling to correction. It explains that the refined predictor can support later
MPPI planning while keeping the current study's motion reference fixed.

Verified against `planning/local_reference_correction.py`,
`tools/correct_m1_local_reference.py`, and the user's clarification in
"Icra paper technical audit." Solver settings remain for Experiments. No new
equation was needed; the 17 equation bodies are byte-identical by label. Sampling
now precedes correction, so equations (14) and (15) exchange order. Section
structure, author notes, figure sources and bibliography are preserved.
Seven pages compiled and visually inspected on Windows; no overfull boxes or
unresolved references, two mild paragraph underfull-box warnings. No numerical
tests, fits, planning runs or physical evaluations were performed for this edit.
Publication and backup manifest:
`tmp/local_correction_manuscript_20260913/publication.json`.

The revision applies [IEEE's conference-paper guidance](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/structure-your-paper/)
on defining relevant terminology, presenting necessary equations, and making
the method understandable and reproducible. It also follows the reader-focused
advice in [Nature Methods' writing editorial](https://www.nature.com/articles/nmeth.4532)
to use direct language, make each paragraph serve a purpose, and avoid unnecessary
information and repetition. These are editorial sources, not manuscript citations.
The user's structure and self-contained-paper requirements take precedence over
publication-specific templates or suggestions to rely on supplementary text.

## Changes

- Preserved the preamble, Abstract, Introduction, author notes, figure inputs,
  and exact section/subsection hierarchy.
- Aligned the Problem Statement with the restored reference-shape objective;
  removed the superseded tip-over-root speed-gain objective.
- Preserved a prescribed horizontal strike direction independent of the target
  bearing, without fixing the general statement to the experimental +x direction.
- Replaced two-template, 39-variable retimed jerk commands with a generic clamped
  quintic position spline. Its free control coordinates are the search variables;
  desired PVA is sampled from its analytic derivatives and held between updates.
- Described the selected baseline command as the spline initialization and the
  earlier simulated shape history as a distinct scoring prior. Both remain fixed
  across the model-update chain; each candidate is predicted with the current model.
- Replaced C/D shape matrices, auxiliary E_C/E_D scores, and Frobenius norms with
  equivalent arithmetic means of squared paired position and direction errors.
  Defined normalization and phase pairing where the score is introduced.
- Kept the total objective, directed-motion factors, regularized bending-rate law,
  damping dissipation, robust observation loss, and neural correction penalties.
  Generalized the fixed approach-credit floor to the symbolic h_min.
- Replaced diagonal proposal noise with correlated Gaussian perturbations;
  retained exponential weighting and adaptive effective-sample-count temperature.
- Distinguished search-time checks from full-sequence export verification.
  The active preferred_fold_v1 optimizer does not apply the targeted-strike-only
  predicted-tilt gate during search; full export prediction does apply the limit.
- Clarified initial R(0), effective frame alignment, correction-component bounds,
  stage-initial parameter priors, quadratic exit costs, and early recovery handover.
- Preserved staged fitting, measured-attachment cable identification, training-only
  selection, freezing before validation, and M0's first-update capacity change.
- Fixed a space in one unused BibTeX key (`..._et al._2026` to `..._et_al._2026`).
  Publication metadata and cited keys were not changed.

## Numerical choices retained for the future experimental setup

These are notes for writing the experimental setup, not new experimental results
or permission to change/freeze an experiment. Recheck the selected study's saved
configuration before reporting them.

| Item | Current verified setting / presentation note |
|---|---|
| Selected baseline | Original retained M0, `20260910-022818-648386-M0-development-whip`; command seed matches the original full best_actions |
| Command reference | Quintic clamped spline, 12 total control points, first 3 fixed, 9 adjustable XYZ points = 27 variables; six uniform interior knots |
| Initial spline | Constrained position least squares from selected M0 command positions, subject to jerk bounds; no exact PVA preservation claim |
| Command schedule | 30 Hz; current T=1.5 s gives 45 intervals and 46 PVA packets; yaw zero in the current study |
| Initial conditions | Nominal planning hover with R(0)=Q=I, zero bias/velocities and a straight hanging cable; postflight initialization uses causal measurements |
| Reference shape comparison | M=21 material samples, K=21 phases; Gamma={3/4,1,4/3}; sigma_c=0.12 and sigma_d=0.55; beta=0.1 |
| Approach-credit floor | h_min=0.2; it provides partial reward and is not a hard reversal gate |
| Objective scales | Tip-speed saturation 4 m/s, reversal-speed saturation 0.5 m/s, retraction scale 0.1 m; proximity scale 0.35 m |
| Objective weights | Shape 600, encounter 450, extension 200, miss 300; forward/lateral/jerk 15/30/0.02; exit speed/upward/acceleration 0.5/5/1 |
| Physical paper outcome | Continuous minimum observed 3D tip-to-target distance; distinct from the planner's 0.05 m contact sphere |
| Direction | Experimental +x; chosen independently of the target's bearing. Shape normalization retains world orientation |
| Proposal distribution | Four groups and 512 random samples; correlated full-rank spline geometry covariance; position scales 0.005, 0.015, 0.04 m |
| Temperature/stopping | Target ESS fraction 0.2; minimum 20 updates, plateau patience 12, improvement max(0.1,0.005 times absolute anchor score); no default iteration ceiling |
| Jerk | Continuous spline jerk bounded by third-derivative control coefficients; motion-cost quadrature samples jerk at interval midpoints |
| Bending regularization | epsilon=2e-5; the vector-rate law is retained in the theory, including its approximate rigid-rotation removal |
| Data family weights | Latest/prior/preliminary 1/0.5/0.5, normalized over present groups; equal recording weights within groups |
| Observation loss | Position scale 0.02 m, orientation scale 0.05 rad; cable weight half all observed non-attachment markers (including tip), half tip alone |
| Nominal fitting | Bounded nonlinear least squares; log-parameter prior coefficient 0.03; delay prior coefficient 0.03 with change normalized by 0.04 s; finite candidate delay set |
| Neural fitting | Component correction bound 0.5 m/s^2, lambda_a=0.01; Adam learning rate 1e-3, weight decay 1e-4; full-rollout gradients |
| Neural architecture and input scaling | Preserve the selected checkpoint specifications; document them compactly with the experimental model settings rather than expanding the theoretical subsection |
| Refinement study | Retained M0 followed by two planned updates, with M0 cable correction disabled and a zero-output cable network introduced at the first update; final recordings outside fitting/tuning |
| Recovery | Smooth brake, return, hold; full-sequence verification before export. Detailed durations/bounds belong with experiment/export settings |

The recent 128-sample, 3-update spline development run is simulation integration
evidence. It is not convergence evidence or a completed physical study. Archived
M1/M2 comparisons do not establish outcomes for the newly selected planner.

## Verification and provenance

- Original main/bibliography backup: `tmp/paper_revision_20260913/before_20260913T003253/`.
- Staged source and scripts: `tmp/paper_revision_20260913/`.
- Static checks preserve the heading tree, preamble/Abstract/Introduction, author
  notes and figure inputs. Eleven existing geometry, model, and fitting-loss
  equation blocks are unchanged. Braces/environments, bibliography keys and all
  cross-references pass. Shape-score changes were independently checked for
  mathematical equivalence to the prior matrix means.
- Portable official Tectonic 0.17.0 compiled the draft with an OT1 QA wrapper
  matching the IEEE class's intended Times-compatible metrics; the synced
  preamble/class was not modified for this wrapper.
- Seven-page preview: `output/pdf/AeroWhip_framework_revision_20260913.pdf`.
  All pages were visually inspected; no errors, unresolved references/citations,
  overfull boxes, font substitutions, clipping or overlaps. Existing class/caption
  and missing-author warnings, plus mild underfull paragraphs, remain.
- Experiments, quantitative results and Conclusion remain unwritten. The source
  still has the existing quantitative-results author note. This edit does not
  establish submission readiness or physical performance.
- No numerical project code, selected models, recordings, fit settings or jobs
  were changed or launched for this writing revision. The preview is local;
  cloud synchronization and Overleaf/pdfLaTeX recompilation were not checked.
