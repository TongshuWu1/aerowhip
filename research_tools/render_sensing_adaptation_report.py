"""Render plots and aggregate statistics for the sensing-aware study."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np


DATA = Path("reports/sensing_aware_adaptation_data")
PLOTS = DATA / "plots"


def _load(name: str):
    return json.loads((DATA / name).read_text(encoding="utf-8"))["results"][0]


def _wilson(success: int, total: int) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = success / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total**2)) / denominator
    return center - radius, center + radius


def _paired_binomial_p(repaired: int, degraded: int) -> float:
    total = repaired + degraded
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, index) for index in range(min(repaired, degraded) + 1)) / 2**total
    return min(1.0, 2.0 * tail)


def _iqr(values):
    return [float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75))]


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    zero = _load("zero_noise_causal_velocity.json")
    noise = _load("representative_noise_sweep_projected.json")
    diagnostics = _load("sensing_diagnostics.json")
    benchmark = _load("representative_paired_control_10seeds.json")

    # 1--2: retained parameter estimates over strikes.
    mismatch = next(case for case in zero["cases"] if case["truth_ratio"] == [0.8, 0.7])
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for key, label, color in (
        ("exact_position_exact_velocity", "exact state", "#444444"),
        ("causal_first_order_three_sample", "causal + constraint projection", "#1f77b4"),
    ):
        values = mismatch[key]["estimates_by_strike"]
        axes[0].plot([x["strike"] for x in values], [x["ei_ratio"] for x in values], "o-", label=label, color=color)
        axes[1].plot([x["strike"] for x in values], [x["cb_ratio"] for x in values], "o-", label=label, color=color)
    axes[0].axhline(0.8, color="#d62728", linestyle="--", label="truth")
    axes[1].axhline(0.7, color="#d62728", linestyle="--", label="truth")
    axes[0].set(xlabel="strike", ylabel="EI / EI0", title="EI estimate")
    axes[1].set(xlabel="strike", ylabel="Cb / Cb0", title="Cb estimate")
    axes[0].set_xticks((1, 2, 3)); axes[1].set_xticks((1, 2, 3))
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    figure.tight_layout(); figure.savefig(PLOTS / "01_02_parameter_vs_strike.png", dpi=180); plt.close(figure)

    # 3--5: error and velocity versus noise.
    noise_levels = sorted({row["noise_std_m"] for row in noise["rows"]})
    mismatched_rows = [row for row in noise["rows"] if row["truth_ratio"] != [1.0, 1.0]]
    ei_error = []; cb_error = []; velocity = []
    for level in noise_levels:
        selected = [row for row in mismatched_rows if row["noise_std_m"] == level]
        ei_error.append(np.median([abs(math.log(row["adaptation"]["final_ei_ratio"] / row["truth_ratio"][0])) for row in selected]))
        cb_error.append(np.median([abs(math.log(row["adaptation"]["final_cb_ratio"] / row["truth_ratio"][1])) for row in selected]))
        velocity.append(np.median([row["state_metrics"]["all_marker_velocity_rmse_m_s"] for row in selected]))
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    axes[0].plot(np.asarray(noise_levels) * 1000, ei_error, "o-")
    axes[1].plot(np.asarray(noise_levels) * 1000, cb_error, "o-", color="#d62728")
    axes[2].plot(np.asarray(noise_levels) * 1000, velocity, "o-", color="#2ca02c")
    axes[0].set(xlabel="position-noise std (mm)", ylabel="median |log EI error|", title="EI sensing sensitivity")
    axes[1].set(xlabel="position-noise std (mm)", ylabel="median |log Cb error|", title="Cb sensing sensitivity")
    axes[2].set(xlabel="position-noise std (mm)", ylabel="velocity RMSE (m/s)", title="Causal velocity error")
    figure.tight_layout(); figure.savefig(PLOTS / "03_05_noise_sensitivity.png", dpi=180); plt.close(figure)

    # 6--8: information diagnostics by phase/noise.
    phases = ["stroke", "reversal", "distal_lash", "weak_late_motion"]
    joint_rows = [row for row in noise["rows"] if row["truth_ratio"] == [0.8, 0.7]]
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
    for phase in phases:
        phase_rows = [next(item for item in row["phase_information"] if item["phase"] == phase) for row in joint_rows]
        x = np.asarray([row["noise_std_m"] for row in joint_rows]) * 1000
        axes[0].plot(x, [item["minimum_eigenvalue"] for item in phase_rows], "o-", label=phase)
        axes[1].plot(x, [item["condition"] for item in phase_rows], "o-", label=phase)
        axes[2].plot(x, [item["sensitivity_correlation"] for item in phase_rows], "o-", label=phase)
    axes[0].set_yscale("log")
    axes[0].set(xlabel="noise std (mm)", ylabel="lambda_min", title="Information magnitude")
    axes[1].set(xlabel="noise std (mm)", ylabel="condition number", title="Information conditioning")
    axes[2].set(xlabel="noise std (mm)", ylabel="rho_EC", title="EI/Cb correlation")
    for axis in axes: axis.legend(fontsize=7)
    figure.tight_layout(); figure.savefig(PLOTS / "06_08_information_by_phase.png", dpi=180); plt.close(figure)

    # 9: health distributions at the first requested noisy level.
    noisy_rows = [row for row in noise["rows"] if row["noise_std_m"] == 0.00025]
    matched_health = next(row for row in noisy_rows if row["truth_ratio"] == [1.0, 1.0])["adaptation"]["health_trace"]
    mismatch_health = next(row for row in noisy_rows if row["truth_ratio"] == [0.8, 0.7])["adaptation"]["health_trace"]
    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    axis.hist([row["ema_error_m2"] for row in matched_health], bins=18, alpha=0.65, label="matched")
    axis.hist([row["ema_error_m2"] for row in mismatch_health], bins=18, alpha=0.65, label="mismatch (0.8, 0.7)")
    axis.set(xlabel="health EMA (m²)", ylabel="count", title="0.25 mm sensing: health distributions overlap")
    axis.legend(); figure.tight_layout(); figure.savefig(PLOTS / "09_health_distribution.png", dpi=180); plt.close(figure)

    # 10: prediction RMSE at zero sensing noise over mismatched cases.
    zero_noise_rows = [row for row in noise["rows"] if row["noise_std_m"] == 0.0 and row["truth_ratio"] != [1.0, 1.0]]
    prediction = []
    for horizon in (0.1, 0.2, 0.3):
        for model in ("nominal", "adapted", "oracle"):
            values = [entry["all_node_position_rmse_m"] for row in zero_noise_rows for entry in row["truth_evaluated_prediction_rows"] if entry["horizon_s"] == horizon and entry["model"] == model]
            prediction.append({"horizon_s": horizon, "model": model, "position_rmse_m": float(np.mean(values))})
    figure, axis = plt.subplots(figsize=(6.4, 3.9))
    width = 0.022
    for index, model in enumerate(("nominal", "adapted", "oracle")):
        values = [row["position_rmse_m"] * 1000 for row in prediction if row["model"] == model]
        axis.bar(np.asarray((0.1, 0.2, 0.3)) + (index - 1) * width, values, width=width, label=model)
    axis.set(xlabel="prediction horizon (s)", ylabel="all-node position RMSE (mm)", title="Truth-scored prediction after causal adaptation")
    axis.set_xticks((0.1, 0.2, 0.3)); axis.legend(); figure.tight_layout(); figure.savefig(PLOTS / "10_prediction_rmse.png", dpi=180); plt.close(figure)

    # 11--12: control success and paired outcomes.
    controls = benchmark["control_rows"]
    control_summary = []
    paired_summary = []
    for noise_value in (0.0, 0.00005):
        for mode in ("mode_a_exact_controller_state", "mode_b_sensed_controller_state"):
            subset = [row for row in controls if row["noise_std_m"] == noise_value and row["mode"] == mode]
            for model in ("fixed_nominal", "adapted_after_strike_3", "oracle"):
                group = [row for row in subset if row["baseline"] == model]
                success = sum(bool(row["valid_strike"]) for row in group)
                lower, upper = _wilson(success, len(group))
                errors = [row["target_error_m"] for row in group]
                control_summary.append({"noise_std_m": noise_value, "mode": mode, "model": model, "success": success, "total": len(group), "wilson_95": [lower, upper], "target_error_median_m": float(np.median(errors)), "target_error_iqr_m": _iqr(errors)})
            fixed = {row["seed"]: bool(row["valid_strike"]) for row in subset if row["baseline"] == "fixed_nominal"}
            adapted = {row["seed"]: bool(row["valid_strike"]) for row in subset if row["baseline"] == "adapted_after_strike_3"}
            both = repaired = degraded = neither = 0
            for seed in fixed:
                if fixed[seed] and adapted[seed]: both += 1
                elif not fixed[seed] and adapted[seed]: repaired += 1
                elif fixed[seed] and not adapted[seed]: degraded += 1
                else: neither += 1
            paired_summary.append({"noise_std_m": noise_value, "mode": mode, "both_succeed": both, "adapted_only_repaired": repaired, "fixed_only_degraded": degraded, "both_fail": neither, "exact_mcnemar_binomial_p": _paired_binomial_p(repaired, degraded)})
    mode_b = [row for row in control_summary if row["mode"] == "mode_b_sensed_controller_state"]
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    labels = ["0 mm", "0.05 mm"]
    x = np.arange(2); width = 0.24
    for index, model in enumerate(("fixed_nominal", "adapted_after_strike_3", "oracle")):
        values = [next(row for row in mode_b if row["noise_std_m"] == level and row["model"] == model)["success"] / 10 for level in (0.0, 0.00005)]
        axis.bar(x + (index - 1) * width, values, width, label=model)
    axis.set_xticks(x, labels); axis.set_ylim(0, 1.05); axis.set(ylabel="strike success", title="Mode B: reconstructed-state control")
    axis.legend(fontsize=8); figure.tight_layout(); figure.savefig(PLOTS / "11_control_success.png", dpi=180); plt.close(figure)
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    labels = ["A 0", "B 0", "A .05", "B .05"]
    components = ("both_succeed", "adapted_only_repaired", "fixed_only_degraded", "both_fail")
    bottom = np.zeros(4)
    colors = ("#2ca02c", "#1f77b4", "#d62728", "#777777")
    for component, color in zip(components, colors):
        values = [row[component] for row in paired_summary]
        axis.bar(labels, values, bottom=bottom, label=component, color=color)
        bottom += values
    axis.set(ylabel="paired seeds", title="Fixed versus adapted paired outcomes")
    axis.legend(fontsize=7); figure.tight_layout(); figure.savefig(PLOTS / "12_paired_control_outcomes.png", dpi=180); plt.close(figure)

    # 13--14: cache and trigger ablations.
    figure, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))
    cache = diagnostics["cache_rows"]
    axes[0].bar(("FIFO", "FIFO + cache"), [row["fit_segments"] + row["validation_segments"] for row in cache], color=("#999999", "#1f77b4"))
    axes[0].set(ylabel="selected informative segments", title="Delayed fitting opportunity")
    trigger = [row for row in diagnostics["trigger_rows"] if row["truth_ratio"] == [0.8, 0.7]]
    names = [row["policy"] for row in trigger]
    axes[1].bar(names, [row["fit_attempts"] for row in trigger], label="attempts")
    axes[1].bar(names, [row["accepted_fits"] for row in trigger], label="accepted")
    axes[1].tick_params(axis="x", rotation=20); axes[1].set(title="Trigger-component ablation", ylabel="fits")
    axes[1].legend(); figure.tight_layout(); figure.savefig(PLOTS / "13_14_cache_trigger_ablation.png", dpi=180); plt.close(figure)

    summary = {
        "schema": "sensing_aware_adaptation_statistics_v1",
        "prediction_summary": prediction,
        "control_summary": control_summary,
        "paired_control_summary": paired_summary,
        "noise_sensitivity": [
            {"noise_std_m": level, "median_abs_log_ei_error": float(e), "median_abs_log_cb_error": float(c), "median_velocity_rmse_m_s": float(v)}
            for level, e, c, v in zip(noise_levels, ei_error, cb_error, velocity)
        ],
    }
    (DATA / "statistical_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

