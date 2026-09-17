"""A fresh PPO outward-whip setup with independent exploration shaping."""
from copy import deepcopy


def settings_from_mppi(mppi,ppo):
    cfg=deepcopy(ppo)
    cfg['model_path']=mppi['model_path']
    cfg['observation_contract']='pva_whip_phase_v2'
    for key in ('action','limits','task'):cfg[key]=deepcopy(mppi[key])
    cfg['launch']['origin_m']=list(mppi['launch']['origin_m'])
    cfg['launch']['target_m']=list(mppi['launch']['target_m'])
    # Keep PPO's start/target randomization. Sampling a deterministic MPPI
    # proposal and learning a policy from exploration are different experiments.
    cfg['reward'].update(progress=60.,strike_quality=60.,success=400.,failure=100.,
        invalid_contact=50.,time_per_s=1.,displacement=.5,jerk=.02,
        pull_phase=80.,reverse_phase=120.,wave_progress=180.,reach_progress=100.,
        vertical_excursion=10.,vertical_tip_velocity=10.,vertical_tip_alignment=10.,
        horizontal_contact=120.,drone_approach=10.,lateral_excursion=10.,
        near_target_reach=10.,outward_contact=400.)
    cfg['training'].update(batch_size=2048,gae_lambda=.99,entropy_coefficient=.02,
        minimum_attempts=65536,plateau_attempts=32768,evaluate_every_updates=4)
    return cfg


def release_learning_settings(ppo):
    """Independent PPO trial addressing reward for accelerating into failure."""
    cfg=deepcopy(ppo);cfg['observation_contract']='pva_whip_phase_v3'
    cfg['reward'].update(pull_phase=20.,wave_progress=30.,brake_progress=60.,
        reverse_phase=200.,reach_progress=200.,progress=30.,failure=250.,
        success=800.,command_speed=30.)
    # Raw scores retain task units. Scaling only critic/GAE targets keeps the
    # value clipping window useful at the hundreds-of-reward task scale.
    cfg['training'].update(reward_scale=.01,initial_log_std=-.8,entropy_coefficient=.01,
        minimum_attempts=131072,plateau_attempts=65536,evaluate_every_updates=4)
    return cfg


def joint_strike_settings(ppo):
    cfg=deepcopy(ppo)
    cfg['reward'].update(pull_phase=20.,brake_progress=30.,reverse_phase=60.,
        wave_progress=60.,reach_progress=40.,progress=60.,strike_quality=0.,
        joint_strike=1500.,success=1500.,failure=600.,command_acceleration=120.)
    cfg['training'].update(reset_value_on_resume=True,entropy_coefficient=.005,
        minimum_attempts=98304,plateau_attempts=49152)
    return cfg


def first_contact_settings(ppo):
    cfg=deepcopy(ppo)
    cfg['training']['terminate_invalid_contact']=True
    cfg['training']['success_priority']=True
    cfg['reward']['invalid_contact']=600.
    return cfg
