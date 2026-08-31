"""Active learning interfaces and reproducible policy diagnostics."""

from .action_robustness import build_action_perturbation_bank
from .cem_teacher_support import TeacherRecord, load_production_cem_teachers
from .one_shot_env import OneShotEpisodeResult, evaluate_open_loop_batch
from .policy_action import (
    POLICY_ACTION_DIM,
    DecodedPolicyAction,
    canonicalize_normalized_action,
    decode_policy_action,
    encode_physical_action,
)
from .policy_context import (
    POLICY_CONTEXT_DIM,
    PhysicsContext,
    PolicyContext,
    PolicyFrame,
    build_policy_context,
    policy_context_tensor_metadata,
)
from .normalization import FixedContextNormalizer
from .iterative_residual import (
    IterativeResidualOutcomeModel,
    OutcomeNormalizer,
    deterministic_compact_candidates,
    fit_compact_residual_basis,
)
from .sequential_sac_env import SEQUENTIAL_SAC_OBSERVATION_DIM, SequentialWhipEnvironment
from .simple_sac import SIMPLE_SAC_ACTION_DIM, SimpleSACAgent

__all__ = [
    "POLICY_ACTION_DIM",
    "POLICY_CONTEXT_DIM",
    "SEQUENTIAL_SAC_OBSERVATION_DIM",
    "SIMPLE_SAC_ACTION_DIM",
    "DecodedPolicyAction",
    "FixedContextNormalizer",
    "IterativeResidualOutcomeModel",
    "OneShotEpisodeResult",
    "OutcomeNormalizer",
    "PhysicsContext",
    "PolicyContext",
    "PolicyFrame",
    "SequentialWhipEnvironment",
    "SimpleSACAgent",
    "TeacherRecord",
    "build_action_perturbation_bank",
    "build_policy_context",
    "canonicalize_normalized_action",
    "decode_policy_action",
    "deterministic_compact_candidates",
    "encode_physical_action",
    "evaluate_open_loop_batch",
    "fit_compact_residual_basis",
    "load_production_cem_teachers",
    "policy_context_tensor_metadata",
]
