"""Offline unrestricted-control reachability oracle for the drone-whip task."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from cable_twin.shared.dder import DderState

from .model import CableModelSnapshot
from .mpc import (
    CostWeights,
    MpcProblem,
    total_rollout_cost,
    variable_impact_rollout_cost_terms,
)
from .simulator import (
    DroneCableState,
    SimulationResult,
    SimulationSettings,
    WhipSimulator,
    tensor_rollout_to_result,
)
from .reduced import stable_controller_model


ORACLE_SCHEMA = "drone_whip_multifidelity_trajectory_v3"
REACHABILITY_RANK_TERMS = (
    "safety_violation",
    "hit_violation",
    "negative_directional_tip_energy_j",
    "impact_time",
    "regularization",
)
ProgressCallback = Callable[[str], None]
CancellationCallback = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class ReachabilitySettings:
    horizon_s: float = 4.0
    physics_dt_s: float = 0.02
    control_interval_s: float = 0.10
    basis_count: int = 6
    search_node_count: int = 7
    candidates: int = 512
    elite_count: int = 64
    iterations: int = 40
    global_exploration_fraction: float = 0.25
    initial_std_fraction: float = 1.0
    minimum_std_fraction: float = 0.05
    refinement_candidates: int = 64
    refinement_elite_count: int = 8
    refinement_iterations: int = 8
    refinement_std_fraction: float = 0.15
    refinement_minimum_std_fraction: float = 0.005
    attachment_drop_m: float = 0.10
    maximum_acceleration_m_s2: float = 20.0
    maximum_speed_m_s: float = 3.0
    seed: int = 42

    def __post_init__(self) -> None:
        simulation = SimulationSettings(
            horizon_s=self.horizon_s,
            simulation_dt_s=self.physics_dt_s,
            control_interval_s=self.control_interval_s,
            attachment_drop_m=self.attachment_drop_m,
            maximum_acceleration_m_s2=self.maximum_acceleration_m_s2,
            maximum_speed_m_s=self.maximum_speed_m_s,
        )
        del simulation
        if not 2 <= self.basis_count <= 12:
            raise ValueError("Oracle basis count must be between 2 and 12.")
        if self.search_node_count < 6:
            raise ValueError("Search DER node count must be at least six.")
        if not 8 <= self.candidates <= 4096:
            raise ValueError("Oracle candidates must be between 8 and 4096.")
        if not 2 <= self.elite_count < self.candidates:
            raise ValueError("Oracle elite count must be at least two and below candidates.")
        if not 1 <= self.iterations <= 500:
            raise ValueError("Oracle iterations must be between 1 and 500.")
        if not 8 <= self.refinement_candidates <= 512:
            raise ValueError("Full-model refinement candidates must be between 8 and 512.")
        if not 2 <= self.refinement_elite_count < self.refinement_candidates:
            raise ValueError(
                "Full-model refinement elite count must be at least two and below candidates."
            )
        if not 1 <= self.refinement_iterations <= 100:
            raise ValueError("Full-model refinement iterations must be between 1 and 100.")
        fractions = (
            self.global_exploration_fraction,
            self.initial_std_fraction,
            self.minimum_std_fraction,
            self.refinement_std_fraction,
            self.refinement_minimum_std_fraction,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in fractions):
            raise ValueError("Oracle exploration and standard-deviation fractions must be positive.")
        if self.global_exploration_fraction >= 1.0:
            raise ValueError("Global exploration fraction must be below one.")
        if self.minimum_std_fraction > self.initial_std_fraction:
            raise ValueError("Minimum oracle spread cannot exceed the initial spread.")
        if self.refinement_minimum_std_fraction > self.refinement_std_fraction:
            raise ValueError(
                "Minimum full-model refinement spread cannot exceed its initial spread."
            )
        if self.seed < 0:
            raise ValueError("Oracle random seed must be non-negative.")

    @property
    def control_count(self) -> int:
        return int(round(self.horizon_s / self.control_interval_s))


@dataclass(frozen=True, slots=True)
class ReachabilityResult:
    basis_coefficients_m: np.ndarray
    controls_m_s2: np.ndarray
    prediction: SimulationResult
    terms: dict[str, float]
    impact_frame: int
    search_history: tuple[float, ...]
    refinement_history: tuple[float, ...]
    source_model_sha256: str
    search_model_sha256: str
    source_node_count: int
    search_node_count: int
    source_model_provisional: bool

    @property
    def feasible(self) -> bool:
        return bool(self.terms["feasible"])

    @property
    def speed_amplification(self) -> float:
        denominator = max(self.terms["maximum_drone_speed_m_s"], 1.0e-9)
        return self.terms["directional_speed_m_s"] / denominator


def _bounded_acceleration(controls: torch.Tensor, maximum_m_s2: float) -> torch.Tensor:
    norms = torch.linalg.vector_norm(controls, dim=2, keepdim=True)
    return controls * torch.clamp(
        maximum_m_s2 / torch.clamp(norms, min=1.0e-12), max=1.0
    )


def _basis_controls(
    coefficients_m: torch.Tensor,
    settings: ReachabilitySettings,
) -> torch.Tensor:
    """Decode a smooth zero-initial-velocity position basis into acceleration.

    For mode j, ``p_j(t)=c_j(1-cos(2*pi*j*t/T))``. Each mode starts from
    rest at the mission center and returns there. Mixtures can express several
    forward/recoil pumping cycles without independent-knot random-walk drift.
    """

    if coefficients_m.ndim != 3 or coefficients_m.shape[1:] != (
        settings.basis_count,
        3,
    ):
        raise ValueError("Oracle coefficients must have shape Bxbasis_countx3.")
    times = (
        torch.arange(
            settings.control_count,
            dtype=coefficients_m.dtype,
            device=coefficients_m.device,
        )
        + 0.5
    ) * settings.control_interval_s
    modes = torch.arange(
        1,
        settings.basis_count + 1,
        dtype=coefficients_m.dtype,
        device=coefficients_m.device,
    )
    omega = 2.0 * math.pi * modes / settings.horizon_s
    basis = omega[:, None] ** 2 * torch.cos(omega[:, None] * times[None])
    controls = torch.einsum("bmc,mk->bkc", coefficients_m, basis)
    return _bounded_acceleration(controls, settings.maximum_acceleration_m_s2)


def _vertical_target_plane(
    coefficients_m: torch.Tensor,
    target_position_m: tuple[float, float, float],
    workspace_center_m: tuple[float, float, float],
) -> torch.Tensor:
    """Project coefficients into the gravity/target plane.

    For a straight isotropic cable, stationary target, and symmetric vehicle
    limits, reflection symmetry makes out-of-plane excitation unnecessary for
    reaching a target in this plane. The physical rollout remains fully 3-D.
    """

    target = coefficients_m.new_tensor(target_position_m)
    center = coefficients_m.new_tensor(workspace_center_m)
    forward = target - center
    forward[2] = 0.0
    norm = torch.linalg.vector_norm(forward)
    if float(norm.detach().cpu()) <= 1.0e-9:
        forward = coefficients_m.new_tensor((1.0, 0.0, 0.0))
    else:
        forward = forward / norm
    horizontal = torch.sum(coefficients_m * forward, dim=2, keepdim=True) * forward
    vertical = torch.zeros_like(coefficients_m)
    vertical[:, :, 2] = coefficients_m[:, :, 2]
    return horizontal + vertical


def _repeat_state(state: DroneCableState, batch_size: int) -> DroneCableState:
    if state.batch_size != 1:
        raise ValueError("The offline oracle requires one initial state.")
    return DroneCableState(
        state.drone_position_m.repeat(batch_size, 1),
        state.drone_velocity_m_s.repeat(batch_size, 1),
        DderState(
            state.cable.positions_m.repeat(batch_size, 1, 1),
            state.cable.velocities_m_s.repeat(batch_size, 1, 1),
        ),
    )


def _lexicographic_order(*keys: torch.Tensor) -> torch.Tensor:
    if not keys:
        raise ValueError("At least one oracle ranking key is required.")
    order = torch.arange(keys[0].shape[0], device=keys[0].device)
    for key in reversed(keys):
        order = order[torch.argsort(key[order], stable=True)]
    return order


def _candidate_order(terms: dict[str, torch.Tensor]) -> torch.Tensor:
    return _lexicographic_order(*(terms[name] for name in REACHABILITY_RANK_TERMS))


def _rank(terms: dict[str, torch.Tensor], index: int) -> tuple[float, ...]:
    return tuple(
        float(terms[name][index].detach().cpu())
        for name in REACHABILITY_RANK_TERMS
    )


def reachability_result_rank(result: ReachabilityResult) -> tuple[float, ...]:
    """Return the canonical lexicographic oracle rank for one verified result."""

    return tuple(float(result.terms[name]) for name in REACHABILITY_RANK_TERMS)


def _terms_at(
    terms: dict[str, torch.Tensor], index: int
) -> dict[str, float]:
    return {
        name: float(value[index].detach().cpu()) for name, value in terms.items()
    }


def _seed_coefficients(
    settings: ReachabilitySettings,
    problem: MpcProblem,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Deterministic target-aligned pump modes used to seed basis CEM."""

    target = torch.tensor(problem.target_position_m, dtype=dtype, device=device)
    center = torch.tensor(
        problem.drone_workspace_center_m,
        dtype=dtype,
        device=device,
    )
    forward = target - center
    forward[2] = 0.0
    if float(torch.linalg.vector_norm(forward).detach().cpu()) <= 1.0e-9:
        forward = torch.tensor(problem.impact_direction, dtype=dtype, device=device)
        forward[2] = 0.0
    if float(torch.linalg.vector_norm(forward).detach().cpu()) <= 1.0e-9:
        forward = torch.tensor((1.0, 0.0, 0.0), dtype=dtype, device=device)
    forward = forward / torch.linalg.vector_norm(forward)
    seeds = [torch.zeros((settings.basis_count, 3), dtype=dtype, device=device)]
    for mode in range(settings.basis_count):
        for peak_fraction in (0.5, 0.75, 1.0):
            coefficients = torch.zeros_like(seeds[0])
            # One basis mode has peak displacement 2*coefficient.
            coefficients[mode] = (
                0.5 * peak_fraction * problem.maximum_drone_excursion_m * forward
            )
            seeds.append(coefficients)
    return torch.stack(seeds)


@dataclass(frozen=True, slots=True)
class _CemSearchResult:
    coefficients: torch.Tensor
    controls: torch.Tensor
    terms: dict[str, float]
    impact_frame: int
    history: tuple[float, ...]


def _cem_search(
    simulator: WhipSimulator,
    initial: DroneCableState,
    problem: MpcProblem,
    settings: ReachabilitySettings,
    *,
    candidates: int,
    elite_count: int,
    iterations: int,
    initial_std_fraction: float,
    minimum_std_fraction: float,
    seed: int,
    phase: str,
    initial_mean: torch.Tensor | None = None,
    global_exploration_fraction: float = 0.0,
    add_deterministic_seeds: bool = False,
    progress: ProgressCallback | None = None,
    cancelled: CancellationCallback | None = None,
) -> _CemSearchResult:
    """Run one deterministic CEM phase on one fixed DER resolution."""

    report = progress if progress is not None else (lambda _message: None)
    repeated = _repeat_state(initial, candidates)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if initial_mean is None:
        mean = torch.zeros(
            (settings.basis_count, 3),
            dtype=simulator.dtype,
            device=simulator.device,
        )
    else:
        mean = torch.as_tensor(
            initial_mean,
            dtype=simulator.dtype,
            device=simulator.device,
        ).clone()
    coefficient_scale = problem.maximum_drone_excursion_m / (
        2.0 * math.sqrt(settings.basis_count)
    )
    std = torch.full_like(mean, initial_std_fraction * coefficient_scale)
    minimum_std = minimum_std_fraction * coefficient_scale
    global_count = int(round(candidates * global_exploration_fraction))
    seeds = (
        _seed_coefficients(
            settings,
            problem,
            dtype=simulator.dtype,
            device=simulator.device,
        )
        if add_deterministic_seeds
        else None
    )
    best_rank: tuple[float, ...] | None = None
    best_coefficients: torch.Tensor | None = None
    best_controls: torch.Tensor | None = None
    best_terms: dict[str, float] = {}
    best_impact_frame = 0
    history: list[float] = []
    weights = CostWeights()

    for iteration in range(1, iterations + 1):
        if cancelled is not None and cancelled():
            raise RuntimeError("Whip trajectory optimization stopped by user.")
        noise = torch.randn(
            (candidates, settings.basis_count, 3),
            generator=generator,
            dtype=simulator.dtype,
        ).to(simulator.device)
        coefficients = mean[None] + std[None] * noise
        if global_count:
            coefficients[-global_count:] = (
                settings.initial_std_fraction
                * coefficient_scale
                * torch.randn(
                    (global_count, settings.basis_count, 3),
                    generator=generator,
                    dtype=simulator.dtype,
                ).to(simulator.device)
            )
        coefficients[0] = mean
        if iteration == 1 and seeds is not None:
            seed_count = min(len(seeds), candidates)
            coefficients[:seed_count] = seeds[:seed_count]
        assert problem.drone_workspace_center_m is not None
        coefficients = _vertical_target_plane(
            coefficients,
            problem.target_position_m,
            problem.drone_workspace_center_m,
        )
        controls = _basis_controls(coefficients, settings)
        with torch.no_grad():
            rollout = simulator.rollout(
                repeated,
                controls,
                create_graph=False,
                cancelled=cancelled,
            )
            terms, impact_frames = variable_impact_rollout_cost_terms(
                rollout,
                repeated,
                problem,
                simulator,
                weights,
            )
            costs = total_rollout_cost(terms, weights)
            if not bool(torch.all(torch.isfinite(costs)).detach().cpu()):
                raise RuntimeError("Whip trajectory search produced a non-finite cost.")
            order = _candidate_order(terms)
            winner = int(order[0].detach().cpu())
            candidate_rank = _rank(terms, winner)
            if best_rank is None or candidate_rank < best_rank:
                best_rank = candidate_rank
                best_coefficients = coefficients[winner].detach().clone()
                best_controls = controls[winner].detach().clone()
                best_terms = _terms_at(terms, winner)
                best_impact_frame = int(impact_frames[winner].detach().cpu())
            elites = coefficients[order[:elite_count]]
            mean = 0.25 * mean + 0.75 * torch.mean(elites, dim=0)
            std = torch.clamp(
                0.25 * std + 0.75 * torch.std(elites, dim=0, unbiased=False),
                min=minimum_std,
            )
            assert best_rank is not None
            history.append(best_rank[0] + best_rank[1])
        if (
            iteration == 1
            or iteration == iterations
            or (phase == "search" and iteration % 10 == 0)
        ):
            report(
                f"{phase} {iteration}/{iterations}: "
                f"feasible={'yes' if bool(best_terms.get('feasible')) else 'no'} "
                f"violation={best_terms.get('constraint_violation', math.inf):.4g} "
                f"error={1000.0 * best_terms.get('position_error_m', math.nan):.1f}mm "
                f"speed={best_terms.get('directional_speed_m_s', math.nan):.2f}m/s"
            )

    if best_coefficients is None or best_controls is None:
        raise RuntimeError("Whip trajectory search produced no finite candidate.")
    return _CemSearchResult(
        coefficients=best_coefficients,
        controls=best_controls,
        terms=best_terms,
        impact_frame=best_impact_frame,
        history=tuple(history),
    )


def optimize_reachability(
    source_snapshot: CableModelSnapshot,
    problem: MpcProblem,
    initial_drone_position_m: tuple[float, float, float],
    settings: ReachabilitySettings = ReachabilitySettings(),
    *,
    device: str = "cuda",
    progress: ProgressCallback | None = None,
    cancelled: CancellationCallback | None = None,
) -> ReachabilityResult:
    """Optimize a smooth return-to-center motion and verify the exact full rod.

    The broad search uses a mass-conserving seven-node DER only to screen many
    trajectories efficiently.  Its winner initializes a local CEM refinement
    on the immutable full fitted DER.  Feasibility and all reported metrics are
    taken exclusively from the final full-model rollout.
    """

    report = progress if progress is not None else (lambda _message: None)
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError("The canonical reachability oracle requires CUDA.")
    active_problem = replace(
        problem,
        drone_workspace_center_m=tuple(float(value) for value in initial_drone_position_m),
    )
    simulation_settings = SimulationSettings(
        horizon_s=settings.horizon_s,
        simulation_dt_s=settings.physics_dt_s,
        control_interval_s=settings.control_interval_s,
        attachment_drop_m=settings.attachment_drop_m,
        maximum_acceleration_m_s2=settings.maximum_acceleration_m_s2,
        maximum_speed_m_s=settings.maximum_speed_m_s,
    )
    if settings.search_node_count > source_snapshot.node_count:
        raise ValueError("Search DER cannot contain more nodes than the full fitted model.")
    search_snapshot = stable_controller_model(
        source_snapshot,
        simulation_dt_s=settings.physics_dt_s,
        node_count=settings.search_node_count,
        constraint_iterations=4,
    )
    search_simulator = WhipSimulator(
        search_snapshot, simulation_settings, device=torch_device
    )
    search_initial = search_simulator.initial_state(initial_drone_position_m)
    report(
        f"search model: {search_snapshot.node_count} nodes; final verification: "
        f"{source_snapshot.node_count} nodes"
    )
    search = _cem_search(
        search_simulator,
        search_initial,
        active_problem,
        settings,
        candidates=settings.candidates,
        elite_count=settings.elite_count,
        iterations=settings.iterations,
        initial_std_fraction=settings.initial_std_fraction,
        minimum_std_fraction=settings.minimum_std_fraction,
        seed=settings.seed,
        phase="search",
        global_exploration_fraction=settings.global_exploration_fraction,
        add_deterministic_seeds=True,
        progress=report,
        cancelled=cancelled,
    )
    simulator = WhipSimulator(source_snapshot, simulation_settings, device=torch_device)
    initial = simulator.initial_state(initial_drone_position_m)
    refinement = _cem_search(
        simulator,
        initial,
        active_problem,
        settings,
        candidates=settings.refinement_candidates,
        elite_count=settings.refinement_elite_count,
        iterations=settings.refinement_iterations,
        initial_std_fraction=settings.refinement_std_fraction,
        minimum_std_fraction=settings.refinement_minimum_std_fraction,
        seed=settings.seed + 1,
        phase="full model",
        initial_mean=search.coefficients,
        progress=report,
        cancelled=cancelled,
    )
    weights = CostWeights()
    target = torch.tensor(
        active_problem.target_position_m, dtype=simulator.dtype, device=torch_device
    )
    direction = torch.tensor(
        active_problem.impact_direction, dtype=simulator.dtype, device=torch_device
    )
    final_rollout = simulator.rollout(
        initial, refinement.controls[None], create_graph=False
    )
    final_terms_tensor, final_frames = variable_impact_rollout_cost_terms(
        final_rollout,
        initial,
        active_problem,
        simulator,
        weights,
    )
    final_terms = _terms_at(final_terms_tensor, 0)
    final_impact_frame = int(final_frames[0].detach().cpu())
    prediction = tensor_rollout_to_result(
        final_rollout,
        batch_index=0,
        target_position_m=target,
        impact_direction=direction,
        model_sha256=source_snapshot.sha256,
    )
    controls = refinement.controls.detach().cpu().numpy().copy()
    coefficients = refinement.coefficients.detach().cpu().numpy().copy()
    controls.setflags(write=False)
    coefficients.setflags(write=False)
    return ReachabilityResult(
        basis_coefficients_m=coefficients,
        controls_m_s2=controls,
        prediction=prediction,
        terms=final_terms,
        impact_frame=final_impact_frame,
        search_history=search.history,
        refinement_history=refinement.history,
        source_model_sha256=source_snapshot.sha256,
        search_model_sha256=search_snapshot.sha256,
        source_node_count=source_snapshot.node_count,
        search_node_count=search_snapshot.node_count,
        source_model_provisional=source_snapshot.provisional,
    )


def save_reachability_result(
    path: str | Path,
    result: ReachabilityResult,
    settings: ReachabilitySettings,
    problem: MpcProblem,
    *,
    provenance: dict[str, object] | None = None,
    overwrite: bool = True,
) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction = result.prediction
    payload: dict[str, np.ndarray] = {
        "schema": np.asarray(ORACLE_SCHEMA),
        "created_utc": np.asarray(datetime.now(timezone.utc).isoformat()),
        "feasible": np.asarray(result.feasible),
        "basis_coefficients_m": result.basis_coefficients_m,
        "controls_m_s2": result.controls_m_s2,
        "search_history": np.asarray(result.search_history),
        "refinement_history": np.asarray(result.refinement_history),
        "time_s": prediction.time_s,
        "drone_positions_m": prediction.drone_positions_m,
        "drone_velocities_m_s": prediction.drone_velocities_m_s,
        "cable_positions_m": prediction.cable_positions_m,
        "cable_velocities_m_s": prediction.cable_velocities_m_s,
        "impact_frame": np.asarray(result.impact_frame),
        "term_names": np.asarray(tuple(result.terms)),
        "term_values": np.asarray(tuple(result.terms.values())),
        "source_model_sha256": np.asarray(result.source_model_sha256),
        "search_model_sha256": np.asarray(result.search_model_sha256),
        "source_node_count": np.asarray(result.source_node_count),
        "search_node_count": np.asarray(result.search_node_count),
        "source_model_provisional": np.asarray(result.source_model_provisional),
        "speed_amplification": np.asarray(result.speed_amplification),
        "target_position_m": np.asarray(problem.target_position_m),
        "impact_direction": np.asarray(problem.impact_direction),
        "settings_json": np.asarray(json.dumps(asdict(settings), sort_keys=True)),
        "problem_json": np.asarray(json.dumps(asdict(problem), sort_keys=True)),
    }
    if provenance is not None:
        payload["provenance_json"] = np.asarray(
            json.dumps(provenance, sort_keys=True)
        )
    if overwrite:
        np.savez_compressed(output, **payload)
        return output
    try:
        stream = output.open("xb")
    except FileExistsError:
        raise FileExistsError(
            f"Refusing to overwrite an existing reachability result: {output}"
        ) from None
    try:
        with stream:
            np.savez_compressed(stream, **payload)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return output
