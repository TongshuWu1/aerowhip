"""Measure same-GPU asynchronous fitting interference with accelerated MPPI."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time

import torch

from drone_mpc.distributed_adaptation import (
    DistributedAdaptationSettings,
    DistributedParameterFitter,
    ParameterEstimate,
    RollingDistributedBuffer,
    select_informative_segments,
)
from drone_mpc.mppi import optimize_mppi
from drone_mpc.simulator import WhipSimulator
from research_tools.distributed_adaptation_study import (
    generate_truth_rollout,
    observations_from_rollout,
)
from research_tools.profile_mppi_forward import DEFAULT_PROFILE, load_workload


SCHEMA = "distributed_adaptation_mppi_interference_v1"
DEFAULT_OUTPUT = Path(
    "data/drone_mpc/adaptation/adaptation_mppi_interference.json"
)


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "count": len(values),
        "mean_s": statistics.fmean(values),
        "median_s": statistics.median(values),
        "p95_s": p95,
        "maximum_s": max(values),
        "minimum_s": min(values),
        "standard_deviation_s": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()
    workload = load_workload(args.profile)
    settings = DistributedAdaptationSettings()
    truth_rollout, _truth, _simulation = generate_truth_rollout(workload, 0.8, 0.7)
    observations = observations_from_rollout(
        truth_rollout, workload, ParameterEstimate()
    )
    buffer = RollingDistributedBuffer(settings)
    for observation in observations:
        buffer.append(observation)
    fitting_segments, validation_segments = select_informative_segments(
        buffer.candidate_segments(), settings
    )
    if not fitting_segments or not validation_segments:
        raise RuntimeError("Interference benchmark could not select fitting data.")
    fitter = DistributedParameterFitter(workload.controller, settings, device="cuda")
    planner = WhipSimulator(workload.controller, workload.simulation, device="cuda")
    initial = planner.initial_state(workload.initial_xyz)

    # Warm both static CUDA paths before measuring contention.
    optimize_mppi(
        planner,
        initial,
        workload.problem,
        workload.mppi,
        warm_start_knots_m_s2=workload.warm_knots,
    )
    fitter.fit(fitting_segments, validation_segments, ParameterEstimate())
    torch.cuda.synchronize()

    baseline = []
    for _ in range(args.repetitions):
        started = time.perf_counter()
        optimize_mppi(
            planner,
            initial,
            workload.problem,
            workload.mppi,
            warm_start_knots_m_s2=workload.warm_knots,
        )
        torch.cuda.synchronize()
        baseline.append(time.perf_counter() - started)

    concurrent = []
    fitting_times = []
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="fit-interference") as executor:
        for _ in range(args.repetitions):
            future = executor.submit(
                fitter.fit,
                fitting_segments,
                validation_segments,
                ParameterEstimate(),
            )
            started = time.perf_counter()
            optimize_mppi(
                planner,
                initial,
                workload.problem,
                workload.mppi,
                warm_start_knots_m_s2=workload.warm_knots,
            )
            torch.cuda.synchronize()
            concurrent.append(time.perf_counter() - started)
            result = future.result()
            fitting_times.append(result.timing.total_s)

    baseline_stats = _stats(baseline)
    concurrent_stats = _stats(concurrent)
    payload = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile": str(args.profile.resolve()),
        "settings": asdict(settings),
        "fit_segments": len(fitting_segments),
        "validation_segments": len(validation_segments),
        "mppi_without_adaptation": baseline_stats,
        "mppi_with_concurrent_same_gpu_fit": concurrent_stats,
        "concurrent_fitting": _stats(fitting_times),
        "median_latency_ratio": concurrent_stats["median_s"]
        / baseline_stats["median_s"],
        "p95_latency_ratio": concurrent_stats["p95_s"]
        / baseline_stats["p95_s"],
        # The planning-only target is 70--80 ms so that sensing,
        # communication, and scheduling still fit inside a 100 ms cycle.
        "recommend_between_strikes": concurrent_stats["p95_s"] > 0.08,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
