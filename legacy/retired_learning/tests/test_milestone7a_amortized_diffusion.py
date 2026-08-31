from __future__ import annotations

import ast
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from learning.action_diffusion import (
    ConditionalActionDiffusion,
    ExponentialMovingAverage,
    cosine_alpha_bar_schedule,
    ddim_timestep_schedule,
    diffusion_epsilon_loss,
    sample_ddim,
)
from learning.amortized_cem_data import (
    AmortizedCemContextRecord,
    ContextBalancedTeacherSampler,
    build_initial_context_records,
    load_scorer_rows,
    pending_context_records,
    write_context_shards,
)
from learning.outcome_scorer import (
    ManeuverOutcomeScorer,
    OutcomeTargetNormalizer,
    PredictedPhysicalOutcomes,
    binary_positive_weights,
    scorer_loss,
    select_candidate_from_predictions,
)
from learning.state_bank import InitialStateBank
from learning.policy_action import decode_policy_action
from learning.normalization import FixedContextNormalizer
from learning.amortized_cem_training import train_diffusion, train_scorer
from planning.cem_task import load_variable_duration_task
from run_milestone7a import _append_aggregation_records


ROOT = Path(__file__).resolve().parents[1]
STATE_ROOT = (
    ROOT
    / "data"
    / "policy_training"
    / "oneshot_sac_nominal_v1"
    / "2026-08-29T192059.907360Z"
)


def _state_banks() -> tuple[InitialStateBank, InitialStateBank]:
    return (
        InitialStateBank.load(
            STATE_ROOT / "training_state_bank.npz",
            STATE_ROOT / "training_state_bank_manifest.json",
        ),
        InitialStateBank.load(
            STATE_ROOT / "validation_state_bank.npz",
            STATE_ROOT / "validation_state_bank_manifest.json",
        ),
    )


def test_state_level_split_is_deterministic_and_disjoint() -> None:
    training, heldout = _state_banks()
    arguments = dict(theta_nominal=(1.0, 2.0, 3.0, 4.0, 5.0, 0.001, 0.01), seed=42)
    first, manifest, _ = build_initial_context_records(training, heldout, **arguments)
    second, second_manifest, _ = build_initial_context_records(training, heldout, **arguments)
    assert first == second
    assert manifest == second_manifest
    train = set(manifest["training_state_ids"])
    validation = set(manifest["validation_state_ids"])
    test = set(manifest["test_state_ids"])
    edge = set(manifest["edge_reserved_test_state_ids"])
    aggregation = set(manifest["aggregation_train_pool_state_ids"])
    assert len(first) == 768
    assert not (train & validation or train & test or validation & test)
    assert not (edge & validation or edge & test)
    assert not (aggregation & train)
    assert all(state_id.startswith("training:") for state_id in aggregation)
    assert all(state_id.startswith("heldout:") for state_id in validation | test | edge)
    for state_id in train | validation | test:
        assert len({row.split for row in first if row.state_id == state_id}) == 1


def test_generation_resume_filters_only_durably_committed_contexts() -> None:
    records = [
        AmortizedCemContextRecord(
            index,
            f"context_{index}",
            "TRAIN",
            "training",
            f"training:{index:04d}",
            index,
            0,
            (1.0, 0.0, -0.04),
            (1.0, 0.0, 0.0),
            (1.0, 2.0, 3.0, 4.0, 5.0, 0.001, 0.01),
        )
        for index in range(4)
    ]
    pending = pending_context_records(
        records,
        [{"source_pool_context_id": "context_1"}, {"source_pool_context_id": "context_3"}],
        progress_id_key="source_pool_context_id",
    )
    assert [record.context_id for record in pending] == ["context_0", "context_2"]


def test_aggregation_append_is_train_only_and_preserves_validation_test(tmp_path: Path) -> None:
    theta = (1.0, 2.0, 3.0, 4.0, 5.0, 0.001, 0.01)
    existing = [
        AmortizedCemContextRecord(
            index,
            f"existing_{split.lower()}",
            split,
            "heldout",
            f"heldout:{index:04d}",
            index,
            0,
            (1.0, 0.0, -0.04),
            (1.0, 0.0, 0.0),
            theta,
        )
        for index, split in enumerate(("VALIDATION", "TEST"))
    ]
    np.savez_compressed(
        tmp_path / "context_table.npz",
        schema=np.asarray("amortized_cem_context_table_v1"),
        contexts=np.zeros((2, 83), dtype=np.float32),
        context_indices=np.arange(2, dtype=np.int64),
        context_ids=np.asarray([record.context_id for record in existing]),
        state_ids=np.asarray([record.state_id for record in existing]),
        splits=np.asarray([record.split for record in existing]),
        theta=np.asarray([theta, theta], dtype=np.float32),
    )
    (tmp_path / "context_split_manifest.json").write_text(
        json.dumps({"records": [asdict(record) for record in existing]}), encoding="utf-8"
    )
    added = AmortizedCemContextRecord(
        2,
        "aggregation_train",
        "TRAIN",
        "training",
        "training:0007",
        7,
        0,
        (0.9, 0.1, -0.08),
        (0.99, 0.1, 0.0),
        theta,
    )
    _append_aggregation_records(tmp_path, [added], np.ones((1, 83), dtype=np.float32))
    manifest = json.loads((tmp_path / "context_split_manifest.json").read_text(encoding="utf-8"))
    expected_existing = json.loads(json.dumps([asdict(record) for record in existing]))
    assert manifest["records"][:2] == expected_existing
    assert manifest["records"][2]["split"] == "TRAIN"
    assert manifest["aggregation_train_context_ids"] == ["aggregation_train"]
    with np.load(tmp_path / "context_table.npz", allow_pickle=False) as table:
        assert table["splits"].tolist() == ["VALIDATION", "TEST", "TRAIN"]
        assert np.array_equal(table["contexts"][:2], np.zeros((2, 83), dtype=np.float32))


def test_context_balanced_sampler_does_not_overweight_many_solution_contexts() -> None:
    contexts = np.zeros((3, 83), dtype=np.float32)
    actions = {
        0: np.zeros((1, 49), dtype=np.float32),
        1: np.ones((100, 49), dtype=np.float32),
        2: -np.ones((5, 49), dtype=np.float32),
    }
    sampler = ContextBalancedTeacherSampler(contexts, actions, seed=7)
    _, _, indices = sampler.sample(30_000)
    fractions = torch.bincount(indices, minlength=3).float() / indices.numel()
    assert torch.all(torch.abs(fractions - 1.0 / 3.0) < 0.02)


def test_diffusion_equation_loss_and_deterministic_ddim_are_finite() -> None:
    torch.manual_seed(5)
    model = ConditionalActionDiffusion()
    context = torch.randn((8, 83))
    action = torch.empty((8, 49)).uniform_(-1.0, 1.0)
    alpha = cosine_alpha_bar_schedule(100)
    generator = torch.Generator().manual_seed(9)
    loss = diffusion_epsilon_loss(model, context, action, alpha_bar=alpha, generator=generator)
    selected = alpha[loss.timesteps][:, None]
    expected = torch.sqrt(selected) * action + torch.sqrt(1.0 - selected) * loss.noise
    assert torch.allclose(loss.noisy_action, expected)
    assert torch.isfinite(loss.loss)
    loss.loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())

    noise = torch.randn((32, 49), generator=torch.Generator().manual_seed(11))
    schedule = ddim_timestep_schedule(100, 25)
    assert schedule[0] == 95
    assert schedule[-1] == 0
    assert torch.unique(schedule).numel() == 25
    first = sample_ddim(model, context[:1], noise, alpha_bar=alpha, timestep_schedule=schedule)
    second = sample_ddim(model, context[:1], noise, alpha_bar=alpha, timestep_schedule=schedule)
    assert torch.equal(first.bounded_action, second.bounded_action)
    assert first.bounded_action.shape == (32, 49)
    assert bool((first.bounded_action.abs() <= 1.0).all())


def test_diffusion_ema_round_trip() -> None:
    model = ConditionalActionDiffusion()
    ema = ExponentialMovingAverage(model, decay=0.999)
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    ema.update(model)
    restored = ExponentialMovingAverage(ConditionalActionDiffusion(), decay=0.5)
    restored.load_state_dict(ema.state_dict())
    copy = ConditionalActionDiffusion()
    restored.copy_to(copy)
    for key, value in ema.shadow.items():
        assert torch.equal(copy.state_dict()[key], value)


def test_scorer_loss_schema_and_candidate_selection() -> None:
    scorer = ManeuverOutcomeScorer()
    context, action = torch.randn((16, 83)), torch.randn((16, 49))
    continuous = torch.randn((16, 7))
    binary = torch.randint(0, 2, (16, 3)).float()
    normalizer = OutcomeTargetNormalizer.fit(continuous)
    predicted_continuous, predicted_binary = scorer(context, action)
    losses = scorer_loss(
        predicted_continuous,
        predicted_binary,
        continuous,
        binary,
        target_normalizer=normalizer,
        positive_weights=binary_positive_weights(binary),
    )
    assert torch.isfinite(losses.total)
    losses.total.backward()

    physical = torch.tensor(
        [
            [2.0, np.log(0.0201), 4.5, 1.0, 0.2, 2.0, 15.0],
            [5.0, np.log(0.1001), 3.0, 0.0, 0.2, 2.0, 15.0],
        ],
        dtype=torch.float32,
    )
    probabilities = torch.tensor([[0.9, 0.9, 0.7], [0.9, 0.9, 0.1]])
    selection = select_candidate_from_predictions(PredictedPhysicalOutcomes(physical, probabilities))
    assert selection.selected_index == 0
    assert bool(selection.predicted_gate_pass[0])


def test_deployment_module_has_no_cem_or_simulator_dependency() -> None:
    path = ROOT / "learning" / "amortized_cem_policy.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(name.startswith("planning") or name.startswith("simulator") for name in imports)
    text = path.read_text(encoding="utf-8")
    assert "optimize_production_cem" not in text
    assert "build_production_simulator" not in text


def test_milestone7a_config_preserves_final_contract() -> None:
    payload = json.loads(
        (ROOT / "config" / "learning" / "amortized_cem_diffusion_nominal_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["model_freeze"] == "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI"
    assert payload["teacher"]["train_states"] == 256
    assert payload["teacher"]["validation_states"] == 64
    assert payload["teacher"]["test_states"] == 64
    assert payload["diffusion"]["training_steps"] == 100
    assert payload["diffusion"]["sampling_steps"] == 25
    assert payload["diffusion"]["candidates"] == 32
    assert all(payload["prohibitions"].values())


def test_scorer_shard_is_compact_versioned_and_loadable(tmp_path: Path) -> None:
    record = AmortizedCemContextRecord(
        0,
        "train_synthetic",
        "TRAIN",
        "training",
        "training:0000",
        0,
        0,
        (1.0, 0.0, -0.04),
        (1.0, 0.0, 0.0),
        (1.0, 2.0, 3.0, 4.0, 5.0, 0.001, 0.01),
    )
    metric = {
        "task_cost": -5.0,
        "best_event_tip_distance_m": 0.02,
        "best_event_directed_speed_m_s": 4.5,
        "best_event_direction_angle_deg": 10.0,
        "maximum_uav_displacement_m": 0.2,
        "maximum_uav_speed_m_s": 2.0,
        "maximum_command_acceleration_m_s2": 10.0,
        "first_entry_marker": 10,
        "finite": True,
        "success": True,
    }
    teacher_entry, scorer_entry = write_context_shards(
        tmp_path / "teacher",
        tmp_path / "scorer",
        record=record,
        successful_actions=np.zeros((2, 49), dtype=np.float32),
        successful_metrics=[metric, metric],
        successful_provenance=[{"seed": 1}, {"seed": 2}],
        scorer_actions=np.zeros((3, 49), dtype=np.float32),
        scorer_metrics=[metric, metric, metric],
        scorer_provenance=[{"source": "test"}] * 3,
    )
    assert teacher_entry is not None
    context_table = tmp_path / "contexts.npz"
    np.savez_compressed(context_table, contexts=np.zeros((1, 83), dtype=np.float32))
    loaded = load_scorer_rows(
        context_table,
        tmp_path / "scorer",
        {"shards": [scorer_entry]},
        split="TRAIN",
    )
    assert [value.shape for value in loaded] == [(3, 83), (3, 49), (3, 7), (3, 3)]
    with np.load(tmp_path / "scorer" / scorer_entry["path"], allow_pickle=False) as shard:
        assert "metrics_json" not in shard.files
        assert str(shard["schema"]) == "amortized_cem_outcome_scorer_shard_v1"


def test_generated_action_contract_decodes_through_production_codec() -> None:
    task = load_variable_duration_task(
        ROOT / "config" / "tasks" / "canonical_whip_variable_duration_tuned_reward_v1.json"
    )
    action = torch.empty((32, 49)).uniform_(-1.0, 1.0)
    decoded = decode_policy_action(action, task, duration_max_s=1.80)
    assert decoded.acceleration_knots_local_m_s2.shape == (32, 16, 3)
    assert bool(
        (torch.linalg.vector_norm(decoded.acceleration_knots_local_m_s2, dim=-1) <= 20.0 + 1e-5).all()
    )
    assert bool(((decoded.duration_s >= 0.45) & (decoded.duration_s <= 1.80)).all())


def test_training_loops_checkpoint_and_reload_on_sharded_data(tmp_path: Path) -> None:
    contexts = np.stack((np.zeros(83, dtype=np.float32), np.ones(83, dtype=np.float32)))
    np.savez_compressed(tmp_path / "context_table.npz", contexts=contexts)
    FixedContextNormalizer.fit(torch.from_numpy(contexts[:1])).save(
        tmp_path / "context_normalizer.json"
    )
    teacher_entries, scorer_entries = [], []
    for index, split in enumerate(("TRAIN", "VALIDATION")):
        record = AmortizedCemContextRecord(
            index,
            f"{split.lower()}_synthetic",
            split,
            "training" if split == "TRAIN" else "heldout",
            f"synthetic:{index}",
            index,
            0,
            (1.0, 0.0, -0.04),
            (1.0, 0.0, 0.0),
            (1.0, 2.0, 3.0, 4.0, 5.0, 0.001, 0.01),
        )
        metrics = []
        for row in range(6):
            success = row % 3 == 0
            metrics.append(
                {
                    "task_cost": float(row - 3),
                    "best_event_tip_distance_m": 0.02 if success else 0.2,
                    "best_event_directed_speed_m_s": 4.5 if success else 2.0,
                    "best_event_direction_angle_deg": 10.0 if success else 70.0,
                    "maximum_uav_displacement_m": 0.2 if row < 4 else 0.7,
                    "maximum_uav_speed_m_s": 2.0 if row < 4 else 4.0,
                    "maximum_command_acceleration_m_s2": 10.0,
                    "first_entry_marker": 10 if success else 5,
                    "finite": True,
                    "success": success,
                }
            )
        teacher_entry, scorer_entry = write_context_shards(
            tmp_path / "diffusion_teacher_shards",
            tmp_path / "scorer_dataset_shards",
            record=record,
            successful_actions=np.zeros((2, 49), dtype=np.float32) + 0.1 * index,
            successful_metrics=metrics[:2],
            successful_provenance=[{"seed": 1}, {"seed": 2}],
            scorer_actions=np.zeros((6, 49), dtype=np.float32),
            scorer_metrics=metrics,
            scorer_provenance=[{"source": "synthetic"}] * 6,
        )
        teacher_entries.append(teacher_entry)
        scorer_entries.append(scorer_entry)
    (tmp_path / "diffusion_teacher_manifest.json").write_text(
        json.dumps({"shards": teacher_entries}), encoding="utf-8"
    )
    (tmp_path / "scorer_dataset_manifest.json").write_text(
        json.dumps({"shards": scorer_entries}), encoding="utf-8"
    )
    config = {
        "seed": 42,
        "diffusion": {
            "training_steps": 100,
            "learning_rate": 2e-4,
            "weight_decay": 1e-6,
            "batch_size": 4,
            "gradient_clip": 1.0,
            "ema_decay": 0.999,
            "maximum_updates": 1,
            "validation_interval": 1,
            "minimum_updates": 1,
            "plateau_updates": 1,
        },
        "scorer": {
            "learning_rate": 3e-4,
            "weight_decay": 1e-6,
            "batch_size": 6,
            "maximum_updates": 1,
            "validation_interval": 1,
            "minimum_updates": 1,
            "plateau_updates": 1,
        },
    }
    diffusion_summary = train_diffusion(tmp_path, config, device="cpu")
    scorer_summary = train_scorer(tmp_path, config, device="cpu")
    assert diffusion_summary["updates"] == 1
    assert scorer_summary["updates"] == 1
    assert (tmp_path / "diffusion_ema_best.pt").is_file()
    assert (tmp_path / "scorer_best.pt").is_file()
