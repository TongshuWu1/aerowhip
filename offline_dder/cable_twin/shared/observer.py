"""One canonical RGB-D cable observation step for offline and online use."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .config import AppSettings
from .contracts import StereoCalibration, StereoFrame, ViewObservation
from .metric_curve import (
    MetricCurveObservation,
    lift_partial_curve_from_registered_depth,
    orient_view_to_previous,
)
from .pidnet_runtime import PidnetRuntime, PidnetResult
from .routes import extract_partial_curve, extract_skeleton_evidence, skeletonize_body


@dataclass(frozen=True, slots=True)
class CableFrameObservation:
    view: ViewObservation
    metric: MetricCurveObservation | None
    skeleton: np.ndarray
    pidnet: PidnetResult
    timings_ms: dict[str, float]


class CableObserver:
    """PIDNet, ordered 2-D route, and registered-depth lifting for one cable."""

    def __init__(
        self,
        settings: AppSettings,
        calibration: StereoCalibration,
        *,
        pidnet: PidnetRuntime | None = None,
    ) -> None:
        self.settings = settings
        self.calibration = calibration
        self.pidnet = pidnet or PidnetRuntime(settings.pidnet_runtime_config)
        self._previous_endpoints_xy: np.ndarray | None = None

    def process(
        self,
        frame: StereoFrame,
        *,
        ordered_route: bool = True,
    ) -> CableFrameObservation:
        started = time.perf_counter()
        result = self.pidnet.infer(frame.left_bgr)

        skeleton_started = time.perf_counter()
        skeleton = skeletonize_body(result.masks[0])
        skeleton_ms = (time.perf_counter() - skeleton_started) * 1000.0
        route_started = time.perf_counter()
        extractor = extract_partial_curve if ordered_route else extract_skeleton_evidence
        view = extractor(
            result.masks[0],
            result.masks[self.settings.cable_identity],
            result.body_component_count,
            self.settings.route,
            skeleton=skeleton,
        )
        view, self._previous_endpoints_xy = orient_view_to_previous(
            view, self._previous_endpoints_xy
        )
        route_ms = (time.perf_counter() - route_started) * 1000.0

        lift_started = time.perf_counter()
        metric = None
        if np.any(view.curve.point_valid) or np.any(view.curve.endpoint_valid):
            if frame.depth_m is None:
                raise RuntimeError("Cable observation requires registered ZED depth.")
            depth = self.settings.depth
            metric = lift_partial_curve_from_registered_depth(
                view,
                frame.depth_m,
                self.calibration.left_intrinsics,
                radius_px=depth.neighborhood_radius_px,
                minimum_support=depth.minimum_support,
                maximum_cluster_span_m=depth.maximum_cluster_span_m,
                depth_min_m=depth.minimum_m,
                depth_max_m=depth.maximum_m,
            )
        depth_lift_ms = (time.perf_counter() - lift_started) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0
        return CableFrameObservation(
            view=view,
            metric=metric,
            skeleton=skeleton,
            pidnet=result,
            timings_ms={
                "pidnet_ms": result.inference_ms,
                "mask_postprocess_ms": result.postprocess_ms,
                "skeleton_ms": skeleton_ms,
                "route_ms": route_ms,
                "depth_lift_ms": depth_lift_ms,
                "total_ms": total_ms,
            },
        )
