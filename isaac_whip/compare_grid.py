"""Compare rigorously compatible Isaac-whip grid-resolution runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


_COMPARABILITY_FIELDS: tuple[tuple[str, ...], ...] = (
    ("root_reference_schema",),
    ("mode",),
    ("num_envs",),
    ("device",),
    ("physics_dt_s",),
    ("achieved_control_hz",),
    ("cable", "source_sha256"),
    ("cable", "length_m"),
    ("cable", "diameter_m"),
    ("cable", "total_mass_kg"),
    ("cable", "bending_stiffness_n_m2"),
    ("cable", "bending_damping_n_m2_s"),
    ("drone", "config_sha256"),
)


def _nested(report: dict[str, Any], keys: tuple[str, ...]) -> Any:
    value: Any = report
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"report is missing comparability field {'.'.join(keys)}")
        value = value[key]
    return value


def _comparable_value_equal(left: Any, right: Any) -> bool:
    """Compare metadata exactly, except for round-off in derived numeric scalars."""

    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return math.isclose(float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-12)
    return left == right


def _load(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "isaac_whip_physics_run_v1":
        raise ValueError(f"{path} is not an Isaac-whip physics report")
    if report.get("status") != "PASS" or report.get("finite") is not True:
        raise ValueError(f"{path} is not a finite PASS run")
    if not report.get("trace_env_0_10_hz"):
        raise ValueError(f"{path} has no 10 Hz trajectory trace")
    for keys in _COMPARABILITY_FIELDS:
        _nested(report, keys)
    report["_source_path"] = str(path.resolve())
    return report


def _distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b, strict=True)))


def _samples(report: dict[str, Any], until_s: float | None) -> dict[float, dict[str, Any]]:
    result: dict[float, dict[str, Any]] = {}
    for sample in report["trace_env_0_10_hz"]:
        time_s = round(float(sample["time_s"]), 6)
        if until_s is not None and time_s > until_s + 1.0e-9:
            continue
        if time_s in result:
            raise ValueError(f"report contains duplicate trace time {time_s:g} s")
        result[time_s] = sample
    return result


def _validate_compatible(
    left: dict[str, Any], right: dict[str, Any], until_s: float | None
) -> tuple[dict[float, dict[str, Any]], dict[float, dict[str, Any]]]:
    for keys in _COMPARABILITY_FIELDS:
        left_value = _nested(left, keys)
        right_value = _nested(right, keys)
        if not _comparable_value_equal(left_value, right_value):
            field = ".".join(keys)
            raise ValueError(
                f"reports are not comparable: {field} differs ({left_value!r} != {right_value!r})"
            )

    left_duration = float(left["simulated_time_s"])
    right_duration = float(right["simulated_time_s"])
    dt = float(left["physics_dt_s"])
    if until_s is None:
        if not math.isclose(left_duration, right_duration, rel_tol=0.0, abs_tol=0.5 * dt):
            raise ValueError(
                "reports are not comparable: simulated durations differ and --until-s was not supplied"
            )
    elif left_duration + 0.5 * dt < until_s or right_duration + 0.5 * dt < until_s:
        raise ValueError(f"a report does not cover the requested {until_s:g} s horizon")

    left_samples = _samples(left, until_s)
    right_samples = _samples(right, until_s)
    if not left_samples or not right_samples:
        raise ValueError("reports do not contain trace samples in the requested horizon")
    if set(left_samples) != set(right_samples):
        raise ValueError("reports do not contain identical 10 Hz trace times")
    if until_s is not None:
        expected_last = math.floor((until_s + 1.0e-9) * 10.0) / 10.0
        if max(left_samples) + 1.0e-9 < expected_last:
            raise ValueError(f"trace does not cover the requested {until_s:g} s horizon")
    return left_samples, right_samples


def compare(left: dict[str, Any], right: dict[str, Any], until_s: float | None) -> dict[str, Any]:
    """Return adjacent-grid trajectory errors after strict compatibility checks."""

    left_samples, right_samples = _validate_compatible(left, right, until_s)
    common_times = sorted(left_samples)
    tip_errors = [
        _distance(
            left_samples[time_s]["tip_position_m"],
            right_samples[time_s]["tip_position_m"],
        )
        for time_s in common_times
    ]
    drone_errors = [
        _distance(
            left_samples[time_s]["drone_position_m"],
            right_samples[time_s]["drone_position_m"],
        )
        for time_s in common_times
    ]

    def metrics(values: list[float]) -> dict[str, float]:
        return {
            "rmse_m": math.sqrt(sum(value * value for value in values) / len(values)),
            "maximum_m": max(values),
        }

    return {
        "left_links": int(left["cable"]["link_count"]),
        "right_links": int(right["cable"]["link_count"]),
        "sample_count": len(common_times),
        "first_time_s": common_times[0],
        "last_time_s": common_times[-1],
        "tip": metrics(tip_errors),
        "drone": metrics(drone_errors),
    }


def _input_manifest(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": report.get("_source_path"),
        "links": int(report["cable"]["link_count"]),
        "model_sha256": report["cable"]["source_sha256"],
        "drone_config_sha256": report["drone"]["config_sha256"],
        "root_reference_schema": report["root_reference_schema"],
        "mode": report["mode"],
        "num_envs": int(report["num_envs"]),
        "device": report["device"],
        "physics_dt_s": float(report["physics_dt_s"]),
        "control_hz": float(report["achieved_control_hz"]),
        "simulated_time_s": float(report["simulated_time_s"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare adjacent Isaac-whip grid-resolution PASS reports at common 10 Hz times."
    )
    parser.add_argument("reports", nargs="+", type=Path, help="Two or more physics report JSON files.")
    parser.add_argument("--until-s", type=float, default=None, help="Only compare samples up to this time.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional machine-readable output.")
    args = parser.parse_args()
    if len(args.reports) < 2:
        parser.error("at least two reports are required")
    if args.until_s is not None and args.until_s <= 0.0:
        parser.error("--until-s must be positive")

    reports = [_load(path.resolve()) for path in args.reports]
    reports.sort(key=lambda report: int(report["cable"]["link_count"]))
    link_counts = [int(report["cable"]["link_count"]) for report in reports]
    if len(set(link_counts)) != len(link_counts):
        raise ValueError("each report must use a distinct link count")
    comparisons = [
        compare(left, right, args.until_s)
        for left, right in zip(reports[:-1], reports[1:], strict=True)
    ]
    result = {
        "schema": "isaac_whip_grid_comparison_v1",
        "until_s": args.until_s,
        "inputs": [_input_manifest(report) for report in reports],
        "comparisons": comparisons,
    }
    for item in comparisons:
        print(
            f"{item['left_links']}->{item['right_links']} links | "
            f"tip RMS={item['tip']['rmse_m'] * 1000.0:.3f} mm "
            f"max={item['tip']['maximum_m'] * 1000.0:.3f} mm | "
            f"drone RMS={item['drone']['rmse_m'] * 1000.0:.3f} mm "
            f"max={item['drone']['maximum_m'] * 1000.0:.3f} mm"
        )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
