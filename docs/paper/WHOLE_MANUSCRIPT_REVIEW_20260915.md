# AeroWhip manuscript review - 15 September 2026

## Scope and outcome

Reviewed the entire live [manuscript](<C:/Users/wts28/Lehigh University Dropbox/TonyLehigh Wu/Apps/Overleaf/AeroWhip-ICRA/main.tex>), including equations, captions, tables, references, notation, and paragraph transitions. Checked the predictor, fitting, spline, and correction descriptions against their implementations. Revised the prose and filled supported method-definition gaps.

The user's from-hover planning formulation, experimental results, figure assets, and numerical values are preserved. The numerical table has one label clarification: **Prediction grid rate** distinguishes its 150 Hz output grid from internal integration steps. The requested marker-node mapping and tip/whole-cable weighting explanations are omitted from the paper. No stopping or fallback discussion was added.

## Changes made

- **Notation:** define the shared acceleration bound at first use; identify the batch recordings `D_k`; state the nonnegative damping rate; expand RMSE before using it; distinguish vehicle-position error in model selection.
- **Averages:** distinguish arithmetic averages on the correction sample grids, weighted observation averages in fitting, and acceleration-regularization averages over time and free nodes. Clarify that the correction compares trajectories at matching times.
- **Commands:** make the zero initial desired velocity and acceleration explicit; define spline derivatives, jerk, and specific force; distinguish sampled command limits, predicted-state checks, and the continuous spline jerk bound.
- **Prediction:** clarify the zero-yaw nominal initialization, delayed-command network inputs, and reconstruction between measured cable markers. Complete the integration description with midpoint steps split at delayed command updates, exponential velocity damping, backward-Euler bending damping, and constraint projections.
- **Optimization:** explain correlated smooth spline proposals and effective sample size. Clarify the adaptive step bound and damping in command correction, and distinguish local-model cost predictions from full-rollout evaluations. State that recovery segments are also checked for feasibility.
- **Refinement:** identify the vehicle network held fixed during parameter fitting, and explain that automatic differentiation passes through integration and cable constraint projections. Distinguish command-driven model selection from the measured-attachment diagnostic.
- **Writing:** improve transitions between the prediction challenge, rod models, and iterative refinement; remove repeated explanations of the fixed reference and model freezing; shorten the conclusion. Move the quadrotor citation to the vehicle-dynamics claim it supports. Avoid presenting downwash as the proven dominant source of error.

No additional nonstandard symbol lacking a definition was found in the revised equations. Standard vector operations and rotation-matrix notation remain compact. A symbol glossary or another algorithm box was not needed for this revision.

## Major methods checked

| Component | Methods described in the paper | Implementation evidence |
|---|---|---|
| Vehicle and cable prediction | Delayed response model; reduced rod dynamics; learned acceleration corrections; midpoint and split cable integration | [Vehicle integration](../../simulator/research_pose.py), [cable dynamics](../../simulator/cable/dder.py) |
| Parameter fitting | Bounded nonlinear least squares in log coordinates; TRF; central-difference rollout Jacobians; separate delay search | [Vehicle fitting](../../experimental_data/whip_full_optim.py), [cable fitting](../../experimental_data/whip_full_cable.py) |
| Network fitting | Adam, weight decay, rollout differentiation, magnitude/change regularization | [Training](../../experimental_data/whip_full_continuation.py), [network objectives](../../experimental_data/whip_full_optim.py) |
| Command correction | Fixed-reference least squares; damped bounded Gauss-Newton; SLSQP subproblems; backtracking and adaptive trust region | [Correction solver](../../planning/local_reference_correction.py), [tracking cost](../../planning/reference_correction.py) |
| Initial planning | Quintic spline; Gaussian proposals; exponential score weighting; adaptive effective sample size | [Spline implementation](../../planning/position_spline.py), [from-hover search](../../planning/strike_mppi.py), [weight calculation](../../planning/mppi_trajectory.py) |

The initial planner's **objective and event rule** remain an exception to full implementation correspondence, as explained below.

## Remaining issues

### 1. Initial-planner formulation and implementation still differ

The manuscript's preserved formulation combines a from-hover search, first free-tip entry, and proximity/retraction/extension scores. The retained [preferred-shape planner](../../planning/mppi_trajectory.py) instead uses an earlier command seed and a shape-history term; its [encounter rule](../../planning/whip_objective.py) latches the first contact by any tracked cable site and includes a tip-first factor. The available [from-hover planner](../../planning/strike_mppi.py) uses a different [strike objective](../../planning/strike_objective.py).

The user's choice to preserve the manuscript formulation is respected. It does not establish that either implementation exactly reproduces that formulation. This remains a substantive correspondence issue, not an omitted optimizer name. Planner-specific weights, score normalizations, sampling settings, and numerical limits should be tied to the matching implementation before claiming complete reproducibility; values from the two variants should not be mixed.

### 2. A visible author note remains in Experiments

The orange `\Tony{...}` note requesting an additional non-whipping task is still rendered. It was preserved with the experiment section. It is an author note, not evidence of a completed experiment.

## Verification and limits

- Final LaTeX compilation produces **8 pages**, with the numerical table on page 6. No overfull boxes, unresolved references/citations, duplicate labels, or LaTeX errors. All 23 citation keys resolve. Four underfull-box warnings remain; the rendered text was visually reviewed.
- All 20 displayed equation blocks are preserved apart from one punctuation correction. The planning objective and encounter definition are unchanged. The experiment section is unchanged except for the numerical-table label above; figures and bibliography are unchanged.
- Four focused CPU float64 checks passed against the implementation: initial spline conditions, the inverse jerk mapping, the correction residual's squared-norm identity, and the cable fitting loss identity. These check the descriptions and algebra; they are not a new simulation-accuracy or physical-performance validation.
- Reviewed all eight final rendered pages for layout, equations, table alignment, and figure readability. The title's author area remains as supplied, and the existing experiment note remains visible.
- Targeted primary-source checks covered [rod reduction](https://www.cs.columbia.edu/cg/pdfs/143-rods.pdf), [SciPy least squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html), [SLSQP terminology](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html), [quadrotor dynamics learning](https://arxiv.org/abs/2102.05773), [MPPI with repeated model training](https://nolanwagener.github.io/media/mppi/paper.pdf), and [DEFORM](https://proceedings.mlr.press/v270/chen25d.html). This was not an exhaustive literature or bibliographic-metadata audit.
- No fitting campaign, flight experiment, or experimental provenance audit was performed. Quantitative claims remain the supplied manuscript's claims.

The source backup, final diff, dependency hashes, and check results are retained in `tmp/whole_paper_review_20260915/`.
