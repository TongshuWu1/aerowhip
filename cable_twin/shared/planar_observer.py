"""PIDNet endpoint-to-endpoint observations in the RGB image plane."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .config import AppSettings
from .contracts import PartialCurveObservation, StereoFrame, ViewObservation
from .pidnet_runtime import PidnetResult, PidnetRuntime
from .routes import extract_complete_endpoint_route, skeletonize_body


@dataclass(frozen=True, slots=True)
class PlanarFrameObservation:
    view: ViewObservation
    skeleton: np.ndarray
    pidnet: PidnetResult
    complete: bool
    timings_ms: dict[str, float]


class PlanarCableObserver:
    """Complete non-occluded cable route from one RGB image."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        pidnet: PidnetRuntime | None = None,
    ) -> None:
        self.settings = settings
        self.pidnet = pidnet or PidnetRuntime(settings.pidnet_runtime_config)
        self._previous_endpoints_xy: np.ndarray | None = None

    def _orient(self, view: ViewObservation) -> ViewObservation:
        curve = view.curve
        if np.count_nonzero(curve.endpoint_valid) != 2:
            return view
        centers = np.asarray(curve.endpoint_centers_xy)
        reverse = False
        if self._previous_endpoints_xy is not None:
            direct = float(
                np.linalg.norm(centers - self._previous_endpoints_xy, axis=1).sum()
            )
            flipped = float(
                np.linalg.norm(centers[::-1] - self._previous_endpoints_xy, axis=1).sum()
            )
            reverse = flipped < direct
        if reverse:
            oriented_curve = PartialCurveObservation(
                curve.points_xy[::-1].copy(),
                curve.point_valid[::-1].copy(),
                curve.segment_ids[::-1].copy(),
                centers[::-1].copy(),
                curve.endpoint_valid[::-1].copy(),
                curve.endpoint_component_count,
                curve.body_component_count,
            )
            view = ViewObservation(
                view.body_mask,
                view.endpoint_mask,
                oriented_curve,
                view.failure,
            )
            centers = oriented_curve.endpoint_centers_xy
        self._previous_endpoints_xy = np.array(centers, copy=True)
        return view

    def process(self, frame: StereoFrame) -> PlanarFrameObservation:
        started = time.perf_counter()
        result = self.pidnet.infer(frame.left_bgr)
        route_started = time.perf_counter()
        skeleton = skeletonize_body(result.masks[0])
        view = extract_complete_endpoint_route(
            result.masks[0],
            result.masks[self.settings.cable_identity],
            result.body_component_count,
            self.settings.route,
            skeleton=skeleton,
        )
        view = self._orient(view)
        route_ms = (time.perf_counter() - route_started) * 1000.0

        curve = view.curve
        complete = bool(
            np.all(curve.point_valid)
            and np.count_nonzero(curve.endpoint_valid) == 2
            and view.failure is None
        )
        total_ms = (time.perf_counter() - started) * 1000.0
        return PlanarFrameObservation(
            view=view,
            skeleton=skeleton,
            pidnet=result,
            complete=complete,
            timings_ms={
                "pidnet_ms": result.inference_ms,
                "mask_postprocess_ms": result.postprocess_ms,
                "route_ms": route_ms,
                "total_ms": total_ms,
            },
        )
