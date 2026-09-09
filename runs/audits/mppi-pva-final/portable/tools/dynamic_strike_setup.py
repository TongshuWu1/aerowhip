"""Reproducible reward candidate. Does not start training or overwrite configs."""
from copy import deepcopy


def configure_dynamic_strike(model, task, config):
    model, task, config = deepcopy((model, task, config))
    if model.get('fullstate_execution', {}).get('schema') != 'tracked_pose_execution_v1':
        raise ValueError('This experiment requires the native 30 Hz M0 model.')
    task.update(episode_duration_s=5., target_position_m=[1.5, 0., 1.4])
    config['reward'].update(
        source='dynamic_strike_v1_20260908',
        directed_speed_shaping_reference='world_and_attachment_relative',
        directed_speed_reward_cap_m_s=4., relative_directed_speed_reward_cap_m_s=6.,
        displacement_allowance_mode='required_reach_plus_margin', displacement_allowance_margin_m=.25,
        displacement_cost_scale_m=.5, progress_weight=80., strike_quality_improvement_weight=120.,
        success_bonus=250., point_displacement_integral_weight=1., maximum_displacement_weight=10.,
        success_forward_return_bonus_weight=0., success_release_bonus_weight=0., terminal_displacement_weight=0.,
        time_to_success_weight_per_s=.2, non_tip_first_penalty=50., invalid_tip_entry_penalty=25.,
        timeout_penalty=0., numerical_failure_penalty=500.,
        equation='80 Δbest_progress + 120 Δbest(proximity × world_speed_quality × relative_speed_quality) '
                 '+ 250 first_valid_hit − 1 integral(excess_excursion_cost) − 10 max(excess_excursion_cost) '
                 '− 0.2 planned_seconds − 50 non_tip_first − 25 invalid_tip_contact − 500 invalid_execution')
    config.update(seed=656, status='research_30hz_dynamic_strike_v1', bootstrap={'enabled': False})
    config['curriculum']['enabled'] = False
    config['curriculum']['meaning'] = 'Fixed 5 cm / 4 m/s / 45 degree world hit gate; no changing reward during this run.'
    config['deployment'].update(reset_optimizer_on_resume=False, initial_position_radius_m=.05, target_position_radius_m=.05)
    config['training'].update(collection_batch=1024, requested_episodes=500000, device='cuda')
    config['validation'].update(enabled=True, episodes=256, every_episodes=1024)
    config['early_stopping'] = dict(enabled=True, window_evaluations=20,
        patience_episodes=20000, minimum_additional_episodes=20000,
        minimum_reward_improvement=2., minimum_relative_improvement=.01)
    config['live_scene'] = dict(enabled=True, source='actual_training_collection', retention='last_two_completed_batches')
    return model, task, config
