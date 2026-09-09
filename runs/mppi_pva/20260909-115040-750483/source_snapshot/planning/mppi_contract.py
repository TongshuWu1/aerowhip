"""MPPI-owned task and rollout settings; no policy/config/checkpoint inputs.

The batched force and fitted-execution simulator is shared with learning.
Its historical ``ppo_config`` argument is only a numerical rollout mapping:
this module supplies it without constructing an actor or a training job.
"""
from copy import deepcopy
import numpy as np
from learning.point_force_env import PointForceRewardWeights
from .cem_objective import task_with_settings

REWARD_DEFAULTS = dict(
    progress_weight=60., strike_quality_improvement_weight=60., success_bonus=200.,
    point_displacement_integral_weight=.5, maximum_displacement_weight=15.,
    success_forward_return_bonus_weight=0., success_release_bonus_weight=0.,
    non_tip_first_penalty=25., invalid_tip_entry_penalty=0., timeout_penalty=0.,
    terminal_displacement_weight=0., time_to_success_weight_per_s=10.,
    numerical_failure_penalty=100., proximity_scale_m=.3,
    directed_speed_reward_cap_m_s=4., forward_excursion_scale_m=.35,
    point_backward_speed_scale_m_s=1., relative_tip_forward_speed_scale_m_s=4.,
    displacement_cost_scale_m=.35)
ACTION_DEFAULTS = dict(delta_force_scale_n=[2.,2.,1.6], maximum_force_norm_n=3.2,
                       minimum_vertical_force_n=0.)
SUCCESS_DEFAULTS = dict(tip_target_distance_m=.05, minimum_directed_tip_speed_m_s=4.,
    maximum_tip_velocity_to_desired_direction_error_deg=45.,
    first_contact_only=True, tip_must_enter_before_other_markers=True)
LAUNCH_DEFAULTS = dict(initial_tracking_origin_m=[0.,0.,1.225], target_position_m=[1.5,0.,1.1])


def resolve_reward(overrides=None):
    overrides=overrides or {}
    if set(overrides)-set(REWARD_DEFAULTS):raise ValueError('Unknown MPPI reward setting.')
    reward=dict(REWARD_DEFAULTS,**overrides)
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) for v in reward.values()):
        raise ValueError('MPPI rewards must be finite numbers.')
    PointForceRewardWeights.from_mapping(reward)
    return reward


def resolve_action(overrides=None):
    overrides=overrides or {}
    if set(overrides)-set(ACTION_DEFAULTS):raise ValueError('Unknown MPPI force limit.')
    action=deepcopy(ACTION_DEFAULTS);action.update(overrides)
    scale=np.asarray(action['delta_force_scale_n'],float)
    maximum=action['maximum_force_norm_n'];minimum=action['minimum_vertical_force_n']
    if scale.shape!=(3,) or not np.isfinite(scale).all() or (scale<=0).any() or not np.isfinite([maximum,minimum]).all() or not 0<=minimum<maximum:
        raise ValueError('Use positive XYZ force scales and 0 <= minimum vertical force < force norm limit.')
    return action


def rollout_contract(settings):
    """Resolved snapshot used by MPPI only; settings never read from PPO."""
    config=dict(schema='mppi_force_rollout_v1',physics_dtype='float64',cuda_graph_physics=True,
        action=resolve_action(settings.get('action')),reward=resolve_reward(settings.get('reward')),
        numerical_limits=dict(absolute_position_m=20.,node_speed_m_s=100.),
        observation=dict(clip=10.,position_scale_m=1.,velocity_scale_m_s=5.),
        deployment=dict(enabled=True,termination='execution_success_or_timeout',
            planning_cable_state='hanging',require_predicted_success=False,
            force_gain_fraction=0.,force_lag_max_s=0.,stiffness_fraction=0.,damping_fraction=0.,
            recovery_failure_penalty=0.,strike_followthrough_s=0.))
    config['reward'].update(directed_speed_shaping_reference='world',
        displacement_allowance_mode='none',displacement_allowance_margin_m=0.,angle_shaping_weight=0.)
    task=task_with_settings(dict(schema='force_whip_task_v1',control_dt_s=1/30,
        episode_duration_s=settings['horizon_s'],success=deepcopy(SUCCESS_DEFAULTS),
        desired_strike_direction_world=[1.,0.,0.]),settings)
    return task,config
