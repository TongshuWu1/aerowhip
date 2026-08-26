"""Profile one warmed exact-accelerated DDER--MPPI update by CUDA kernel."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from drone_mpc.mppi import optimize_mppi
from drone_mpc.simulator import WhipSimulator
from research_tools.profile_mppi_forward import DEFAULT_PROFILE, load_workload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/drone_mpc/optimization/final_kernel_profile.json"),
    )
    args = parser.parse_args()
    workload = load_workload(args.profile)
    simulator = WhipSimulator(workload.controller, workload.simulation, device="cuda")
    state = simulator.initial_state(workload.initial_xyz)

    # Compile/capture every static path before the measured update.
    for _ in range(2):
        optimize_mppi(
            simulator,
            state,
            workload.problem,
            workload.mppi,
            warm_start_knots_m_s2=workload.warm_knots,
        )
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        )
    ) as profile:
        optimize_mppi(
            simulator,
            state,
            workload.problem,
            workload.mppi,
            warm_start_knots_m_s2=workload.warm_knots,
        )
        torch.cuda.synchronize()

    operations = []
    for event in profile.key_averages():
        device_us = float(event.self_device_time_total)
        if device_us <= 0.0:
            continue
        operations.append(
            {
                "name": event.key,
                "calls": int(event.count),
                "self_device_time_us": device_us,
            }
        )
    operations.sort(key=lambda item: float(item["self_device_time_us"]), reverse=True)
    total_us = sum(float(item["self_device_time_us"]) for item in operations)

    def category(predicate) -> dict[str, float | int]:
        selected = [item for item in operations if predicate(str(item["name"]))]
        time_us = sum(float(item["self_device_time_us"]) for item in selected)
        return {
            "calls": sum(int(item["calls"]) for item in selected),
            "self_device_time_us": time_us,
            "share_of_summed_device_time": time_us / total_us if total_us else 0.0,
        }

    categories = {
        "fixed_damping": category(lambda name: "fixed_damping_11node_60pcg" in name),
        "fixed_projection": category(lambda name: "fixed_projection_11node_4plus1" in name),
        "fixed_cost": category(lambda name: "fixed_mppi_cost_11node" in name),
    }
    accounted = sum(float(value["self_device_time_us"]) for value in categories.values())
    categories["other"] = {
        "calls": sum(int(item["calls"]) for item in operations)
        - sum(int(value["calls"]) for value in categories.values()),
        "self_device_time_us": total_us - accounted,
        "share_of_summed_device_time": (total_us - accounted) / total_us if total_us else 0.0,
    }
    payload = {
        "schema": "exact_accelerated_mppi_kernel_profile_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(),
        "summed_self_device_time_us": total_us,
        "categories": categories,
        "top_operations": operations[:30],
        "interpretation": (
            "Self-device times are mutually exclusive kernel times. Their sum can differ "
            "slightly from the CUDA-event wall span; category shares use this sum."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
