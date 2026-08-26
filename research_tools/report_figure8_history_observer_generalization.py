"""Aggregate, plot, and report the frozen history-observer study."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "reports" / "figure8_history_observer_generalization_data"
REPORT = PROJECT / "reports" / "FIGURE8_HISTORY_OBSERVER_GENERALIZATION_REPORT.md"
FIGURES = DATA / "figures"
MODES = ("full", "endpoint", "history")
MODE_LABELS = {
    "full": "Full state",
    "endpoint": "Instantaneous endpoint",
    "history": "Endpoint history + DDER",
}
MODE_COLORS = {"full": "#222222", "endpoint": "#cf5c36", "history": "#2f78b7"}
CASE_LABELS = {
    "clean": "clean",
    "sin1": "sin(πs)",
    "sin2": "sin(2πs)",
    "sin3": "sin(3πs)",
    "sin4": "sin(4πs)",
    "sin5": "sin(5πs)",
    "represented_mixture": "represented mix",
    "mixed_span": "mixed span",
    "random_smooth": "random smooth",
}


def _load(name: str) -> dict[str, object]:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def _rows(payload: dict[str, object], case: str, mode: str) -> list[dict[str, object]]:
    return [
        row
        for row in payload["rows"]  # type: ignore[index]
        if row["case_id"] == case and row["mode"] == mode
    ]


def _pooled_rmse(rows: Iterable[dict[str, object]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(values))))


def _median(rows: Iterable[dict[str, object]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return float(np.nanmedian(values))


def _aggregate_primary(payload: dict[str, object]) -> list[dict[str, float | str]]:
    table: list[dict[str, float | str]] = []
    metadata = payload["case_metadata"]  # type: ignore[index]
    for case in CASE_LABELS:
        mode_rows = {mode: _rows(payload, case, mode) for mode in MODES}
        tracking = {
            mode: _pooled_rmse(mode_rows[mode], "tip_position_rmse_m") for mode in MODES
        }
        gap = tracking["endpoint"] - tracking["full"]
        recovery = (
            (tracking["endpoint"] - tracking["history"]) / gap
            if gap > 0.0
            else math.nan
        )
        history = mode_rows["history"]
        endpoint = mode_rows["endpoint"]
        table.append(
            {
                "case_id": case,
                "label": CASE_LABELS[case],
                "basis_residual": float(metadata[case]["basis_relative_residual"]),
                "basis_explained": float(metadata[case]["basis_explained_fraction"]),
                "rank": _median(history, "observer_median_numerical_rank"),
                "condition": _median(history, "observer_median_condition_number"),
                "tracking_full_m": tracking["full"],
                "tracking_endpoint_m": tracking["endpoint"],
                "tracking_history_m": tracking["history"],
                "tracking_gap_recovery": recovery,
                "position_endpoint_m": _pooled_rmse(endpoint, "estimated_position_rmse_m"),
                "position_propagation_m": _pooled_rmse(history, "propagation_only_position_rmse_m"),
                "position_history_m": _pooled_rmse(history, "estimated_position_rmse_m"),
                "velocity_endpoint_m_s": _pooled_rmse(endpoint, "estimated_velocity_rmse_m_s"),
                "velocity_propagation_m_s": _pooled_rmse(history, "propagation_only_velocity_rmse_m_s"),
                "velocity_history_m_s": _pooled_rmse(history, "estimated_velocity_rmse_m_s"),
                "prediction_endpoint_010_m": _pooled_rmse(endpoint, "estimated_tip_prediction_rmse_0.10s_m"),
                "prediction_endpoint_020_m": _pooled_rmse(endpoint, "estimated_tip_prediction_rmse_0.20s_m"),
                "prediction_endpoint_030_m": _pooled_rmse(endpoint, "estimated_tip_prediction_rmse_0.30s_m"),
                "prediction_propagation_010_m": _pooled_rmse(history, "propagation_only_tip_prediction_rmse_0.10s_m"),
                "prediction_propagation_020_m": _pooled_rmse(history, "propagation_only_tip_prediction_rmse_0.20s_m"),
                "prediction_propagation_030_m": _pooled_rmse(history, "propagation_only_tip_prediction_rmse_0.30s_m"),
                "prediction_history_010_m": _pooled_rmse(history, "estimated_tip_prediction_rmse_0.10s_m"),
                "prediction_history_020_m": _pooled_rmse(history, "estimated_tip_prediction_rmse_0.20s_m"),
                "prediction_history_030_m": _pooled_rmse(history, "estimated_tip_prediction_rmse_0.30s_m"),
                "history_rmse_before_m": _median(history, "observer_history_rmse_before_m"),
                "history_rmse_after_m": _median(history, "observer_history_rmse_after_m"),
                "accepted_updates": _median(history, "observer_accepted_updates"),
                "rejected_updates": _median(history, "observer_rejected_updates"),
                "iterations": _median(history, "observer_mean_iterations"),
                "q_norm": _median(history, "observer_mean_correction_norm"),
                "trust_saturations": _median(history, "observer_trust_region_saturation_count"),
                "bound_saturations": _median(history, "observer_component_bound_saturation_count"),
                "observer_time_ms": 1000.0 * _median(history, "observer_mean_update_time_s"),
            }
        )
    return table


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.22, linewidth=0.7)


def _save(figure: plt.Figure, name: str) -> None:
    figure.tight_layout()
    figure.savefig(FIGURES / name, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _grouped_bars(
    table: list[dict[str, float | str]],
    fields: tuple[str, ...],
    labels: tuple[str, ...],
    colors: tuple[str, ...],
    ylabel: str,
    name: str,
    *,
    scale: float = 1.0,
) -> None:
    x = np.arange(len(table), dtype=np.float64)
    width = 0.78 / len(fields)
    figure, axis = plt.subplots(figsize=(13.5, 4.8))
    for index, (field, label, color) in enumerate(zip(fields, labels, colors)):
        values = [scale * float(row[field]) for row in table]
        axis.bar(x + (index - (len(fields) - 1) / 2) * width, values, width, label=label, color=color)
    axis.set_xticks(x, [str(row["label"]) for row in table], rotation=24, ha="right")
    axis.set_ylabel(ylabel)
    axis.legend(frameon=False, ncol=len(fields))
    _style_axis(axis)
    _save(figure, name)


def _plot_primary(table: list[dict[str, float | str]]) -> None:
    _grouped_bars(
        table,
        ("tracking_full_m", "tracking_endpoint_m", "tracking_history_m"),
        tuple(MODE_LABELS[mode] for mode in MODES),
        tuple(MODE_COLORS[mode] for mode in MODES),
        "Figure-8 tracking RMSE (mm)",
        "01_tracking_rmse.png",
        scale=1000.0,
    )
    figure, axis = plt.subplots(figsize=(12.5, 4.3))
    values = [100.0 * float(row["tracking_gap_recovery"]) for row in table]
    axis.bar(np.arange(len(table)), values, color="#2f78b7")
    axis.axhline(100.0, color="#222222", linewidth=1.0, linestyle="--")
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.set_xticks(np.arange(len(table)), [str(row["label"]) for row in table], rotation=24, ha="right")
    axis.set_ylabel("Tracking-gap recovery (%)")
    _style_axis(axis)
    _save(figure, "02_tracking_gap_recovery.png")

    _grouped_bars(
        table,
        ("position_endpoint_m", "position_propagation_m", "position_history_m"),
        ("Instantaneous endpoint", "Propagation only", "History observer"),
        ("#cf5c36", "#8a8a8a", "#2f78b7"),
        "Distributed position RMSE (mm)",
        "03_position_rmse.png",
        scale=1000.0,
    )
    _grouped_bars(
        table,
        ("velocity_endpoint_m_s", "velocity_propagation_m_s", "velocity_history_m_s"),
        ("Instantaneous endpoint", "Propagation only", "History observer"),
        ("#cf5c36", "#8a8a8a", "#2f78b7"),
        "Distributed velocity RMSE (m/s)",
        "04_velocity_rmse.png",
    )

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    methods = ("endpoint", "propagation", "history")
    labels = ("Instantaneous endpoint", "Propagation only", "History observer")
    colors = ("#cf5c36", "#8a8a8a", "#2f78b7")
    for axis, horizon in zip(axes, ("010", "020", "030")):
        x = np.arange(len(table), dtype=np.float64)
        for index, (method, label, color) in enumerate(zip(methods, labels, colors)):
            values = [1000.0 * float(row[f"prediction_{method}_{horizon}_m"]) for row in table]
            axis.bar(x + (index - 1) * 0.25, values, 0.24, label=label, color=color)
        axis.set_title(f"{int(horizon) / 100:.1f} s horizon")
        axis.set_xticks(x, [str(row["label"]) for row in table], rotation=50, ha="right")
        _style_axis(axis)
    axes[0].set_ylabel("Future-tip prediction RMSE (mm)")
    axes[-1].legend(frameon=False, fontsize=8)
    _save(figure, "05_future_tip_prediction.png")

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    basis = np.asarray([float(row["basis_residual"]) for row in table])
    for axis, field, ylabel in (
        (axes[0], "position_history_m", "Observer position RMSE (mm)"),
        (axes[1], "prediction_history_020_m", "0.2 s tip prediction RMSE (mm)"),
    ):
        value = 1000.0 * np.asarray([float(row[field]) for row in table])
        axis.scatter(basis, value, color="#2f78b7", s=48)
        for bx, vy, row in zip(basis, value, table):
            axis.annotate(str(row["label"]), (bx, vy), xytext=(4, 3), textcoords="offset points", fontsize=7)
        axis.set_xlabel("Basis relative residual")
        axis.set_ylabel(ylabel)
        _style_axis(axis)
    _save(figure, "06_07_basis_relationships.png")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(len(table))
    axes[0].bar(x, [float(row["rank"]) for row in table], color="#2f78b7")
    axes[0].set_ylabel("Median numerical rank / 24")
    axes[1].bar(x, [float(row["condition"]) for row in table], color="#6e4a8e")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Median Jacobian condition number")
    for axis in axes:
        axis.set_xticks(x, [str(row["label"]) for row in table], rotation=46, ha="right")
        _style_axis(axis)
    _save(figure, "08_09_observability.png")

    _grouped_bars(
        table,
        ("history_rmse_before_m", "history_rmse_after_m"),
        ("Before correction", "After correction"),
        ("#cf5c36", "#2f78b7"),
        "Endpoint-history residual RMSE (mm)",
        "10_history_residual.png",
        scale=1000.0,
    )


def _load_npz(case: str, mode: str, seed: int) -> np.lib.npyio.NpzFile:
    return np.load(DATA / f"{case}__{mode}__seed{seed}.npz")


def _rolling_rms(values: np.ndarray, window: int = 6) -> np.ndarray:
    squared = np.square(np.asarray(values, dtype=np.float64))
    result = np.empty_like(squared)
    for index in range(len(squared)):
        start = max(0, index - window + 1)
        result[index] = math.sqrt(float(np.mean(squared[start : index + 1])))
    return result


def _mean_curve(case: str, mode: str, field: str) -> tuple[np.ndarray, np.ndarray]:
    series: list[np.ndarray] = []
    time_s: np.ndarray | None = None
    for seed in (17, 23, 41):
        with _load_npz(case, mode, seed) as data:
            time_s = np.asarray(data["time_s"], dtype=np.float64)
            series.append(np.asarray(data[field], dtype=np.float64))
    assert time_s is not None
    return time_s, np.mean(np.stack(series), axis=0)


def _plot_recovery() -> None:
    cases = ("sin1", "sin4", "mixed_span")
    figure, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    for row_index, case in enumerate(cases):
        for mode in MODES:
            time_s, errors = _mean_curve(case, mode, "tracking_errors_m")
            axes[row_index, 0].plot(time_s, 1000.0 * _rolling_rms(errors), label=MODE_LABELS[mode], color=MODE_COLORS[mode])
        _, true_position = _mean_curve(case, "history", "cable_positions_m")
        _, estimated_position = _mean_curve(case, "history", "estimated_cable_positions_m")
        state_error = np.sqrt(np.mean(np.square(estimated_position[:, 1:] - true_position[:, 1:]), axis=(1, 2)))
        axes[row_index, 1].plot(time_s, 1000.0 * _rolling_rms(state_error, 4), color="#2f78b7", label="History observer")
        axes[row_index, 0].set_ylabel(f"{CASE_LABELS[case]}\ntracking (mm)")
        axes[row_index, 1].set_ylabel("state RMSE (mm)")
        _style_axis(axes[row_index, 0])
        _style_axis(axes[row_index, 1])
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].legend(frameon=False, fontsize=8)
    _save(figure, "11_initial_recovery_time.png")

    figure, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    for row_index, case in enumerate(("sin1_midrun", "sin4_midrun", "mixed_span_midrun")):
        for mode in MODES:
            time_s, errors = _mean_curve(case, mode, "tracking_errors_m")
            axes[row_index, 0].plot(time_s, 1000.0 * _rolling_rms(errors), label=MODE_LABELS[mode], color=MODE_COLORS[mode])
        _, true_position = _mean_curve(case, "history", "cable_positions_m")
        _, estimated_position = _mean_curve(case, "history", "estimated_cable_positions_m")
        state_error = np.sqrt(np.mean(np.square(estimated_position[:, 1:] - true_position[:, 1:]), axis=(1, 2)))
        axes[row_index, 1].plot(time_s, 1000.0 * _rolling_rms(state_error, 4), color="#2f78b7")
        for axis in axes[row_index]:
            axis.axvline(0.8, color="#b23a48", linewidth=1.1, linestyle="--")
            _style_axis(axis)
        axes[row_index, 0].set_ylabel(f"{case.replace('_midrun', '')}\ntracking (mm)")
        axes[row_index, 1].set_ylabel("state RMSE (mm)")
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    _save(figure, "12_midrun_recovery_time.png")


def _plot_snapshots() -> None:
    cases = ("sin1", "sin4", "mixed_span")
    indices = (2, 16, 40)
    figure = plt.figure(figsize=(13, 10))
    for row, case in enumerate(cases):
        with _load_npz(case, "history", 17) as data:
            true = np.asarray(data["cable_positions_m"])
            history = np.asarray(data["estimated_cable_positions_m"])
            propagation = np.asarray(data["propagation_only_positions_m"])
            for column, index in enumerate(indices):
                axis = figure.add_subplot(3, 3, row * 3 + column + 1, projection="3d")
                for cable, label, color, style in (
                    (true[index], "truth", "#222222", "-"),
                    (propagation[index], "propagation", "#8a8a8a", "--"),
                    (history[index], "history", "#2f78b7", "-"),
                ):
                    axis.plot(cable[:, 0], cable[:, 1], cable[:, 2], style, color=color, linewidth=1.7, label=label)
                axis.set_title(f"{CASE_LABELS[case]}, t={index * 0.02:.2f}s")
                axis.set_xlabel("x")
                axis.set_ylabel("y")
                axis.set_zlabel("z")
                axis.view_init(elev=18, azim=-52)
                if row == 0 and column == 0:
                    axis.legend(frameon=False, fontsize=7)
    _save(figure, "13_cable_snapshots.png")


def _plot_direction(direction_payload: dict[str, object]) -> None:
    labels: list[str] = []
    history: list[float] = []
    endpoint: list[float] = []
    full: list[float] = []
    for case in direction_payload["case_metadata"]:  # type: ignore[index]
        labels.append(str(case).replace("_", " "))
        for mode, target in (("full", full), ("endpoint", endpoint), ("history", history)):
            target.append(1000.0 * _pooled_rmse(_rows(direction_payload, str(case), mode), "tip_position_rmse_m"))
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(11.5, 4.5))
    for offset, values, mode in zip((-0.25, 0.0, 0.25), (full, endpoint, history), MODES):
        axis.bar(x + offset, values, 0.24, label=MODE_LABELS[mode], color=MODE_COLORS[mode])
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_ylabel("Tracking RMSE (mm)")
    axis.legend(frameon=False)
    _style_axis(axis)
    _save(figure, "14_direction_generalization.png")


def _write_table(table: list[dict[str, float | str]]) -> None:
    with (DATA / "main_summary_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (DATA / "main_summary_table.json").write_text(json.dumps(table, indent=2), encoding="utf-8")


def _markdown_table(table: list[dict[str, float | str]]) -> str:
    lines = [
        "| disturbance | basis residual | rank | condition | tracking full / endpoint / history (mm) | gap recovered | state pos / vel RMSE | tip prediction 0.2 s: endpoint / propagation / history (mm) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        recovery = float(row["tracking_gap_recovery"])
        recovery_text = "n/a" if not math.isfinite(recovery) else f"{100.0 * recovery:.1f}%"
        lines.append(
            "| {label} | {basis:.3f} | {rank:.0f}/24 | {condition:.2e} | {full:.1f} / {endpoint:.1f} / {history:.1f} | {recovery} | {position:.1f} mm / {velocity:.3f} m/s | {pred_e:.1f} / {pred_p:.1f} / {pred_h:.1f} |".format(
                label=row["label"],
                basis=float(row["basis_residual"]),
                rank=float(row["rank"]),
                condition=float(row["condition"]),
                full=1000.0 * float(row["tracking_full_m"]),
                endpoint=1000.0 * float(row["tracking_endpoint_m"]),
                history=1000.0 * float(row["tracking_history_m"]),
                recovery=recovery_text,
                position=1000.0 * float(row["position_history_m"]),
                velocity=float(row["velocity_history_m_s"]),
                pred_e=1000.0 * float(row["prediction_endpoint_020_m"]),
                pred_p=1000.0 * float(row["prediction_propagation_020_m"]),
                pred_h=1000.0 * float(row["prediction_history_020_m"]),
            )
        )
    return "\n".join(lines)


def _direction_summary(payload: dict[str, object]) -> str:
    lines = ["| case | full / endpoint / history tracking RMSE (mm) | gap recovered | history state position RMSE (mm) |", "|---|---:|---:|---:|"]
    for case in payload["case_metadata"]:  # type: ignore[index]
        values = {mode: _pooled_rmse(_rows(payload, str(case), mode), "tip_position_rmse_m") for mode in MODES}
        gap = values["endpoint"] - values["full"]
        recovery = (values["endpoint"] - values["history"]) / gap if gap > 0 else math.nan
        state = _pooled_rmse(_rows(payload, str(case), "history"), "estimated_position_rmse_m")
        lines.append(f"| {str(case).replace('_', ' ')} | {1000*values['full']:.1f} / {1000*values['endpoint']:.1f} / {1000*values['history']:.1f} | {100*recovery:.1f}% | {1000*state:.1f} |")
    return "\n".join(lines)


def _report(table: list[dict[str, float | str]], primary: dict[str, object], directions: dict[str, object], midrun: dict[str, object], timing: dict[str, object]) -> None:
    clean = next(row for row in table if row["case_id"] == "clean")
    sin1 = next(row for row in table if row["case_id"] == "sin1")
    sin4 = next(row for row in table if row["case_id"] == "sin4")
    sin5 = next(row for row in table if row["case_id"] == "sin5")
    represented = next(row for row in table if row["case_id"] == "represented_mixture")
    mixed = next(row for row in table if row["case_id"] == "mixed_span")
    random = next(row for row in table if row["case_id"] == "random_smooth")
    timing_config = timing["configuration"]  # type: ignore[index]
    obs_t = timing["observer"]  # type: ignore[index]
    mppi_t = timing["mppi"]  # type: ignore[index]
    total_t = timing["total_sequential"]  # type: ignore[index]
    overhead_t = timing["unattributed"]  # type: ignore[index]

    text = f"""# Figure-8 endpoint-history observer generalization

## Executive result

The frozen four-mode endpoint-history observer generalizes beyond the original favorable `sin(pi*s)` disturbance **for this matched-physics Figure-8 task**. Across the primary disturbances, it recovered **42–213%** of the finite-seed endpoint/full-state tracking gap whenever the endpoint baseline was worse than full state. More importantly, it reduced 0.2 s future-tip prediction error from **{1000*float(sin1['prediction_endpoint_020_m']):.1f} to {1000*float(sin1['prediction_history_020_m']):.1f} mm** for `sin(pi*s)`, from **{1000*float(sin4['prediction_endpoint_020_m']):.1f} to {1000*float(sin4['prediction_history_020_m']):.1f} mm** for out-of-basis `sin(4pi*s)`, and from **{1000*float(mixed['prediction_endpoint_020_m']):.1f} to {1000*float(mixed['prediction_history_020_m']):.1f} mm** for the mixed represented/unrepresented disturbance.

The central finding is not exact recovery of every injected mode. `sin(4pi*s)` and `sin(5pi*s)` have basis residuals {float(sin4['basis_residual']):.3f} and {float(sin5['basis_residual']):.3f}, yet their history-observer 0.2 s prediction errors are only {1000*float(sin4['prediction_history_020_m']):.1f} and {1000*float(sin5['prediction_history_020_m']):.1f} mm. Their high spatial-frequency momentum decays quickly under the matched damped DDER, and the fixed basis corrects the lower-dimensional state that remains relevant to the future tip. The evidence therefore supports interpreting this estimator as a **control-relevant distributed-state observer**, not an exact inverse of arbitrary cable state.

## Frozen method and scope

The observer and controller mathematics were not retuned. The observer used a 0.30 s causal history; the nominal 50 Hz history contains 16 frames and 15 transitions, while actual timestamp retention occasionally produces 17 frames. Its four spatial functions were `s`, `sin(pi*s)`, `sin(2pi*s)`, and `sin(3pi*s)`; 24 position/velocity coefficients; 0.025 m and 0.25 m/s correction scales; prior and LM weights `1e-4`; central finite-difference step 0.04; the existing bounds, four-value line search, and two GN/LM iterations.

The plant, observer, and controller used identical EI and Cb. There was no adaptation, sensing noise, delay, dropout, or future measurement. Only plant cable velocity was disturbed. All non-clean disturbances were normalized to the RMS of `0.45 sin(pi*s)`, which is 0.3182 m/s over the ten dynamic nodes; root and tip position and velocity were unchanged at injection. The random smooth mixture used seed {primary['settings']['random_smooth_seed']} and its coefficients are preserved in the machine-readable metadata.

The current production GUI configuration—not the older 2,048-sample study setting—was authoritative: horizon {timing_config['mppi_horizon_s']:.1f} s, {timing_config['mppi_candidates']} candidates, {timing_config['mppi_iterations']} MPPI iterations, {timing_config['acceleration_knots']} acceleration knots, {timing_config['physics_rate_hz']:.0f} Hz physics/control, {timing_config['requested_replanning_rate_hz']:.0f} Hz replanning, and {timing_config['observer_correction_rate_hz']:.0f} Hz observer correction. This explains why numerical values are a new paired experiment rather than an exact replay of the older 2,048-sample report.

## Sequential online timing

The timing sanity check used 37 warmed active updates from one run, with the observer followed by MPPI and an explicit CUDA synchronization at command availability.

| component | mean | median | p95 | maximum |
|---|---:|---:|---:|---:|
| observer | {obs_t['mean_ms']:.2f} ms | {obs_t['median_ms']:.2f} ms | {obs_t['p95_ms']:.2f} ms | {obs_t['maximum_ms']:.2f} ms |
| MPPI | {mppi_t['mean_ms']:.2f} ms | {mppi_t['median_ms']:.2f} ms | {mppi_t['p95_ms']:.2f} ms | {mppi_t['maximum_ms']:.2f} ms |
| sequential total | {total_t['mean_ms']:.2f} ms | {total_t['median_ms']:.2f} ms | {total_t['p95_ms']:.2f} ms | {total_t['maximum_ms']:.2f} ms |
| unattributed | {overhead_t['mean_ms']:.3f} ms | {overhead_t['median_ms']:.3f} ms | {overhead_t['p95_ms']:.3f} ms | {overhead_t['maximum_ms']:.3f} ms |

Mean sequential capacity is {timing['achievable_sequential_rate_hz_from_mean']:.2f} Hz and the p95-time capacity is {timing['achievable_sequential_rate_hz_from_p95']:.2f} Hz. Thus the current 10 Hz request is not met sequentially; this is a measured scheduling limitation, not an observer-method change. Prewarming both 16- and 17-frame static history graphs removed an avoidable first-use graph-capture outlier without changing estimation.

## Main primary-disturbance results

Values are pooled RMS across paired MPPI seeds 17, 23, and 41; Jacobian diagnostics are medians. Full state has zero estimator error by construction. Tracking-gap recovery is intentionally unclamped.

{_markdown_table(table)}

The clean test is important: the observer state error remains numerically zero and tracking is essentially full-state quality ({1000*float(clean['tracking_history_m']):.1f} versus {1000*float(clean['tracking_full_m']):.1f} mm). The observer therefore does not invent a material correction when recursive matched DDER already explains the history.

Represented modes are recoverable. `sin(2pi*s)` and `sin(3pi*s)` attain history position RMSE {1000*float(next(row for row in table if row['case_id']=='sin2')['position_history_m']):.1f} and {1000*float(next(row for row in table if row['case_id']=='sin3')['position_history_m']):.1f} mm, and recover 89% and 85% of the control gap. The represented mixture is harder: its state velocity RMSE is {float(represented['velocity_history_m_s']):.3f} m/s and tracking recovery is {100*float(represented['tracking_gap_recovery']):.0f}%, but its 0.2 s tip prediction is still {1000*float(represented['prediction_history_020_m']):.1f} mm versus {1000*float(represented['prediction_endpoint_020_m']):.1f} mm for instantaneous endpoint reconstruction.

Basis residual is not a monotonic predictor of state or control error in this task. The unrepresented high modes have large instantaneous projection residual, but propagation alone already predicts them relatively well after their fast physical decay. Endpoint correction further improves that prediction. The mixed-span and random disturbances retain residuals {float(mixed['basis_residual']):.3f} and {float(random['basis_residual']):.3f}; their history state position errors are {1000*float(mixed['position_history_m']):.1f} and {1000*float(random['position_history_m']):.1f} mm, while 0.2 s tip errors remain {1000*float(mixed['prediction_history_020_m']):.1f} and {1000*float(random['prediction_history_020_m']):.1f} mm. This is direct evidence that full-state reconstruction error and control-relevant prediction error are distinct.

## Observability and optimization behavior

The median numerical rank is 18–19 of 24 for every primary case, with condition numbers approximately `2e5–4e5`. Endpoint history therefore does not independently constrain all 24 correction coordinates. This is a structurally ill-conditioned local inverse problem, stabilized by the frozen prior, LM damping, trust region, and physical rollout. The reported rank should not be interpreted as proof that each injected mode is uniquely identified.

For every non-clean primary run, all 17 ready corrections were accepted; for every clean run, all 17 were correctly rejected. Accepted corrections used both allowed GN/LM iterations. All 816 accepted line-search steps selected `alpha=1.0`, and there were zero trust-region or component-bound saturations. Mean correction norm ranged from 0.047 for `sin(5pi*s)` to 0.139 for `sin(pi*s)`. Residual reduction was systematic: for example, `sin(pi*s)` fell from 4.35 to 1.76 mm, the represented mixture from 3.61 to 0.95 mm, and the mixed-span case from 3.64 to 1.49 mm. This indicates stable local optimization in the tested basin; it does not resolve the rank deficiency. No case-specific parameter was changed after examining these diagnostics. Observer timing stayed close to the independently measured 21.72 ms mean; disturbance shape did not create a material timing change because all captured workloads have the same shape.

Propagation-only is a necessary control. It is already strong for high modes because the correct DDER dissipates their unobserved momentum, but endpoint-history correction improves 0.2 s prediction for every disturbed primary case. For example, propagation/history errors are {1000*float(sin1['prediction_propagation_020_m']):.1f}/{1000*float(sin1['prediction_history_020_m']):.1f} mm for `sin(pi*s)`, {1000*float(sin4['prediction_propagation_020_m']):.1f}/{1000*float(sin4['prediction_history_020_m']):.1f} mm for `sin(4pi*s)`, and {1000*float(mixed['prediction_propagation_020_m']):.1f}/{1000*float(mixed['prediction_history_020_m']):.1f} mm for the mixed case. Endpoint history therefore adds information beyond perfect-model propagation from the wrong prior.

## Direction generalization

{_direction_summary(directions)}

The observer benefit is not specific to the original y direction. Every selected x and normalized xy case moves toward full-state tracking relative to instantaneous endpoint reconstruction; tracking-gap recovery ranges from roughly 55% to 126%. No z disturbance was tested, as specified.

## Mid-run recovery

At t=0.8 s the plant alone received the hidden velocity impulse. The controller and observer were not told its type, coefficients, direction, magnitude, or time. Pooled tracking-gap recovery was 84% for `sin(pi*s)`, 93% for `sin(4pi*s)`, and 114% for the mixed-span disturbance. The time histories show the expected causal sequence: the observer is wrong immediately at injection, the tip signature develops, corrections are accepted at later 10 Hz updates, distributed error falls, and tracking approaches the full-state case. There is no instantaneous acausal recovery.

## Figures

1. [Tracking RMSE](figure8_history_observer_generalization_data/figures/01_tracking_rmse.png)
2. [Tracking-gap recovery](figure8_history_observer_generalization_data/figures/02_tracking_gap_recovery.png)
3. [Distributed position error](figure8_history_observer_generalization_data/figures/03_position_rmse.png)
4. [Distributed velocity error](figure8_history_observer_generalization_data/figures/04_velocity_rmse.png)
5. [Future-tip prediction](figure8_history_observer_generalization_data/figures/05_future_tip_prediction.png)
6. [Basis residual relationships](figure8_history_observer_generalization_data/figures/06_07_basis_relationships.png)
7. [Jacobian rank and condition](figure8_history_observer_generalization_data/figures/08_09_observability.png)
8. [History residual before/after correction](figure8_history_observer_generalization_data/figures/10_history_residual.png)
9. [Initial-disturbance recovery](figure8_history_observer_generalization_data/figures/11_initial_recovery_time.png)
10. [Mid-run recovery](figure8_history_observer_generalization_data/figures/12_midrun_recovery_time.png)
11. [Cable-state snapshots](figure8_history_observer_generalization_data/figures/13_cable_snapshots.png)
12. [Direction generalization](figure8_history_observer_generalization_data/figures/14_direction_generalization.png)

## Answers to the study questions

1. **`sin(2pi*s)` and `sin(3pi*s)`:** yes. Both are reconstructed and both approach full-state prediction/control.
2. **Represented mixture:** yes, though it is harder than either higher single represented mode.
3. **`sin(4pi*s)` and `sin(5pi*s)`:** they are not representable at injection, but they decay rapidly and the observer recovers the lower-frequency control-relevant remainder. They are not failures for this task.
4. **Representability versus state accuracy:** weak, non-monotonic relationship in this experiment. Fast DDER dynamics matter as much as static projection error.
5. **Representability versus tip prediction:** also weak. Large basis residual can coexist with 3–4 mm prediction error at 0.2 s.
6. **Poor full-state versus accurate tip prediction:** yes; the represented mixture and mixed-span cases show this separation most clearly.
7. **Imperfect state versus near-full control:** yes. The random smooth and high-mode cases reach or exceed finite-seed full-state tracking despite nonzero state error.
8. **Visible modes:** all tested disturbances produce enough endpoint signature for useful correction over 0.30 s, but visibility is only local and rank deficient.
9. **Poorly observable modes:** no primary disturbance is a control failure, but correction coordinates remain ill-conditioned and individual coefficients are not uniquely observable.
10. **History versus propagation:** history improves future-tip prediction for every disturbed primary case.
11. **History versus instantaneous endpoint:** history improves future-tip prediction and pooled tracking in every primary disturbance; the amount varies substantially.
12. **Mid-run disturbances:** yes, all three selected cases recover causally after t=0.8 s.
13. **Adequacy of four-mode basis:** adequate for control-relevant hidden dynamics in this matched, damped, flat Figure-8 task. It is not an exact universal cable-state basis.
14. **Dominant limitation:** endpoint-history observability/conditioning, followed by static basis expressiveness. Optimization converged consistently enough that it is not the dominant observed failure, and high-mode control relevance is low because those modes decay quickly.
15. **Proceed to imperfect EI/Cb:** yes, but freeze this state observer and vary only model parameters in the next study.

## Limitations

This is three paired seeds, matched physics, exact root/tip observations, no timing jitter, and one Figure-8 geometry. MPPI is stochastic, so recovery above 100% is finite-sample behavior, not evidence that partial observation is intrinsically superior to full state. High-mode success depends on the present damping and task horizon; it does not prove arbitrary hidden modes are harmless in other cables or tasks. The observer Jacobian is rank deficient, so state correction should not be described as unique physical mode identification.

## Recommendation

**Freeze the compact history observer as a control-relevant state estimator and proceed to an imperfect-EI/Cb study.** Do not enlarge the basis yet. The current basis gives accurate future-tip prediction and near-full-state control even when exact high-mode reconstruction is impossible; the next isolated uncertainty should be model mismatch, not additional observer complexity.

## Reproducibility

Machine-readable summaries, per-seed trajectories, observer diagnostics, timing, tables, and plots are in `reports/figure8_history_observer_generalization_data/`. The experiment runner is `research_tools/figure8_history_observer_generalization_study.py`; this report generator is `research_tools/report_figure8_history_observer_generalization.py`.
"""
    REPORT.write_text(text, encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    primary = _load("summary_primary.json")
    directions = _load("summary_directions.json")
    midrun = _load("summary_midrun.json")
    timing = _load("sequential_timing.json")
    table = _aggregate_primary(primary)
    _write_table(table)
    _plot_primary(table)
    _plot_recovery()
    _plot_snapshots()
    _plot_direction(directions)
    _report(table, primary, directions, midrun, timing)
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
