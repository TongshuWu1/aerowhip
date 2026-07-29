"""Small live evaluation plot and CSV logger for the cable particle filter."""

from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import threading
import time

import cv2
import numpy as np


WINDOW_NAME = "Cable PF - live evaluation"
CABLE_COLORS = ((255, 120, 40), (70, 220, 90))


@dataclass(frozen=True)
class EvaluationSample:
    time_s: float
    frame_index: int
    trace_error_mm: tuple[float, float]
    uncertainty_mm: tuple[float, float]
    visible_fraction: tuple[float, float]
    measurement_valid: tuple[bool, bool]


def _finite_or_nan(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else float("nan")


class LiveEvaluation:
    """Sample three primary PF signals, plot them, and save one CSV."""

    def __init__(
        self,
        output_directory: Path,
        *,
        sample_hz: float = 10.0,
        history_seconds: float = 30.0,
    ):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_directory = Path(output_directory) / stamp
        self.run_directory.mkdir(parents=True, exist_ok=False)
        self.csv_path = self.run_directory / "metrics.csv"
        self.plot_path = self.run_directory / "metrics.png"
        self.sample_period_s = 1.0 / max(0.5, float(sample_hz))
        self.history_seconds = max(5.0, float(history_seconds))
        self.samples: deque[EvaluationSample] = deque(
            maxlen=max(50, int(self.history_seconds / self.sample_period_s) + 2)
        )
        self.lock = threading.Lock()
        self.first_timestamp: float | None = None
        self.next_sample_timestamp = float("-inf")
        self.next_render_time = 0.0
        self.window_open = False
        self.closed = False
        self.last_canvas = self._render(())

        self.csv_stream = self.csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
        )
        self.csv_writer = csv.writer(self.csv_stream)
        self.csv_writer.writerow(
            (
                "time_s",
                "frame_index",
                "cable_1_trace_error_mm",
                "cable_2_trace_error_mm",
                "cable_1_uncertainty_mm",
                "cable_2_uncertainty_mm",
                "cable_1_visible_fraction",
                "cable_2_visible_fraction",
                "cable_1_measurement_valid",
                "cable_2_measurement_valid",
            )
        )
        self.csv_stream.flush()

    def init_window(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 960, 720)
        cv2.imshow(WINDOW_NAME, self.last_canvas)
        cv2.waitKey(1)
        self.window_open = True

    def should_sample(self, timestamp: float) -> bool:
        """Reserve the next low-rate diagnostic sample on the tracking thread."""

        timestamp = float(timestamp)
        with self.lock:
            if timestamp < self.next_sample_timestamp:
                return False
            self.next_sample_timestamp = timestamp + self.sample_period_s
            return True

    def append(self, frame_index: int, timestamp: float, particle_filter) -> None:
        if self.closed or not particle_filter.diagnostics_refreshed:
            return
        timestamp = float(timestamp)
        if self.first_timestamp is None:
            self.first_timestamp = timestamp
        relative_time = timestamp - self.first_timestamp
        diagnostics = tuple(
            cable.diagnostics for cable in particle_filter.cables
        )
        sample = EvaluationSample(
            time_s=float(relative_time),
            frame_index=int(frame_index),
            trace_error_mm=tuple(
                _finite_or_nan(item.trace_mean_mm) for item in diagnostics
            ),
            uncertainty_mm=tuple(
                _finite_or_nan(item.maximum_node_uncertainty_mm)
                for item in diagnostics
            ),
            visible_fraction=tuple(
                float(np.clip(item.visible_fraction, 0.0, 1.0))
                for item in diagnostics
            ),
            measurement_valid=tuple(
                bool(item.measurement_valid) for item in diagnostics
            ),
        )
        with self.lock:
            self.samples.append(sample)
            self.csv_writer.writerow(
                (
                    f"{sample.time_s:.6f}",
                    sample.frame_index,
                    *(
                        "" if not np.isfinite(value) else f"{value:.6f}"
                        for value in (
                            *sample.trace_error_mm,
                            *sample.uncertainty_mm,
                            *sample.visible_fraction,
                        )
                    ),
                    int(sample.measurement_valid[0]),
                    int(sample.measurement_valid[1]),
                )
            )
            self.csv_stream.flush()

    def poll(self) -> None:
        if self.closed:
            return
        now = time.perf_counter()
        if now < self.next_render_time:
            return
        self.next_render_time = now + 1.0 / 15.0
        with self.lock:
            samples = tuple(self.samples)
        self.last_canvas = self._render(samples)
        if not self.window_open:
            return
        cv2.imshow(WINDOW_NAME, self.last_canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            cv2.destroyWindow(WINDOW_NAME)
            self.window_open = False
            return
        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                self.window_open = False
        except cv2.error:
            self.window_open = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        with self.lock:
            samples = tuple(self.samples)
            self.csv_stream.flush()
            self.csv_stream.close()
        self.last_canvas = self._render(samples)
        cv2.imwrite(str(self.plot_path), self.last_canvas)
        if self.window_open:
            try:
                cv2.destroyWindow(WINDOW_NAME)
                cv2.waitKey(1)
            except cv2.error:
                pass
        self.window_open = False

    def _render(self, samples: tuple[EvaluationSample, ...]) -> np.ndarray:
        width, height = 960, 720
        canvas = np.full((height, width, 3), (20, 24, 29), dtype=np.uint8)
        cv2.putText(
            canvas,
            "CABLE PARTICLE FILTER - LIVE EVALUATION",
            (28, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (235, 240, 245),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Cable 1",
            (740, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            CABLE_COLORS[0],
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Cable 2",
            (840, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            CABLE_COLORS[1],
            2,
            cv2.LINE_AA,
        )
        panels = (
            (
                "Visible trace error",
                "mm",
                lambda item: item.trace_error_mm,
                None,
                10.0,
            ),
            (
                "Maximum node uncertainty",
                "mm",
                lambda item: item.uncertainty_mm,
                None,
                10.0,
            ),
            (
                "Visible cable fraction",
                "",
                lambda item: item.visible_fraction,
                1.0,
                1.0,
            ),
        )
        for panel_index, panel in enumerate(panels):
            top = 62 + panel_index * 216
            self._draw_panel(
                canvas,
                (55, top, 885, 170),
                samples,
                *panel,
            )
        cv2.putText(
            canvas,
            f"CSV: {self.csv_path.name}    Q/Esc closes this plot only",
            (28, 706),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (145, 155, 165),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def _draw_panel(
        self,
        canvas: np.ndarray,
        bounds: tuple[int, int, int, int],
        samples: tuple[EvaluationSample, ...],
        title: str,
        unit: str,
        selector,
        fixed_maximum: float | None,
        minimum_maximum: float,
    ) -> None:
        x, y, width, height = bounds
        plot_left = x + 58
        plot_right = x + width - 16
        plot_top = y + 28
        plot_bottom = y + height - 28
        cv2.rectangle(
            canvas,
            (x, y),
            (x + width, y + height),
            (36, 42, 49),
            -1,
        )
        cv2.putText(
            canvas,
            title,
            (x + 12, y + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.49,
            (220, 226, 232),
            1,
            cv2.LINE_AA,
        )

        values = [
            value
            for sample in samples
            for value in selector(sample)
            if np.isfinite(value)
        ]
        if fixed_maximum is not None:
            y_maximum = float(fixed_maximum)
        elif values:
            y_maximum = max(
                float(minimum_maximum),
                float(np.percentile(values, 95.0)) * 1.25,
            )
        else:
            y_maximum = float(minimum_maximum)
        for fraction in (0.0, 0.5, 1.0):
            line_y = int(round(plot_bottom - fraction * (plot_bottom - plot_top)))
            cv2.line(
                canvas,
                (plot_left, line_y),
                (plot_right, line_y),
                (60, 68, 76),
                1,
            )
            label = f"{fraction * y_maximum:.1f}"
            cv2.putText(
                canvas,
                label,
                (x + 8, line_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (145, 155, 165),
                1,
                cv2.LINE_AA,
            )
        if unit:
            cv2.putText(
                canvas,
                unit,
                (x + 12, plot_top + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (145, 155, 165),
                1,
                cv2.LINE_AA,
            )

        end_time = samples[-1].time_s if samples else 0.0
        start_time = max(0.0, end_time - self.history_seconds)
        time_span = max(self.history_seconds, 1e-6)
        for cable_index, color in enumerate(CABLE_COLORS):
            previous = None
            for sample in samples:
                if sample.time_s < start_time:
                    continue
                value = float(selector(sample)[cable_index])
                if not np.isfinite(value):
                    previous = None
                    continue
                point_x = int(
                    round(
                        plot_left
                        + (sample.time_s - start_time)
                        / time_span
                        * (plot_right - plot_left)
                    )
                )
                point_y = int(
                    round(
                        plot_bottom
                        - np.clip(value / y_maximum, 0.0, 1.0)
                        * (plot_bottom - plot_top)
                    )
                )
                point = (point_x, point_y)
                if previous is not None:
                    cv2.line(canvas, previous, point, color, 2, cv2.LINE_AA)
                previous = point
            current = (
                float(selector(samples[-1])[cable_index])
                if samples
                else float("nan")
            )
            current_label = (
                f"{current:.2f}{unit}" if np.isfinite(current) else "--"
            )
            cv2.putText(
                canvas,
                current_label,
                (x + width - 155 + cable_index * 75, y + 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                color,
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            canvas,
            f"last {self.history_seconds:.0f} s",
            (plot_right - 65, plot_bottom + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (125, 135, 145),
            1,
            cv2.LINE_AA,
        )
