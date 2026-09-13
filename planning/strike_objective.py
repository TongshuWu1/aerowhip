"""Targeted strike objective and a separate travelling-fold acceptance test.

Development implementation. This measures pre-contact kinematics, not impact
energy transferred to an object. No archived shape or vehicle-reversal reward
enters this objective. The geometric fold test is an operational definition,
not a measurement of wave energy or a proof of a travelling-wave solution.
"""
from copy import deepcopy
import math

import torch
from planning.mppi_live import RolloutCapture

SCHEMA = 'targeted_fold_strike_v1'
OBJECTIVE = dict(schema=SCHEMA, distance_scale_m=0.10, speed_scale_m_s=4.0,
                 intensity_weight=4.0, jerk_weight=0.01)
FOLD = dict(material_samples=21, half_window_fraction=0.10,
            minimum_fold_angle_deg=90.0, minimum_local_turn_deg=15.0,
            start_material_max=0.40, end_material_min=0.75,
            maximum_backward_step=0.05, maximum_forward_step=0.20,
            minimum_observations=4)
FIELDS = ('fold_tracking', 'fold_last_s', 'fold_count', 'fold_completed',
          'fold_completed_time_s', 'encounter_fold_valid', 'strike_value',
          'strike_distance', 'strike_velocity', 'strike_time', 'strike_valid')


def enabled(settings):
    return settings.get('trajectory_objective', {}).get('schema') == SCHEMA


def default_objective():
    return deepcopy(OBJECTIVE)


def validate_settings(cfg):
    from simulator.pva_commands import SCHEMA as COMMAND_SCHEMA
    if cfg.get('method') != 'mppi' or cfg.get('command_contract') != COMMAND_SCHEMA:
        raise ValueError('Targeted fold strikes require offline PVA MPPI')
    if 'reward' in cfg or cfg.get('ppo_objective'):
        raise ValueError('Legacy reward dictionaries cannot enter the new strike objective')
    objective = cfg['trajectory_objective']; fold = cfg['fold_constraint']
    if set(objective) != set(OBJECTIVE) or set(fold) != set(FOLD):
        raise ValueError('Use the explicit targeted-strike objective and fold settings')
    def finite(x):
        if isinstance(x, dict): return all(finite(v) for v in x.values())
        if isinstance(x, list): return all(finite(v) for v in x)
        return math.isfinite(x) if isinstance(x, (int, float)) else True
    if not finite(cfg): raise ValueError('Settings must be finite')
    for key in ('distance_scale_m', 'speed_scale_m_s'):
        if objective[key] <= 0: raise ValueError('Strike scales must be positive')
    if objective['intensity_weight'] <= 0 or objective['jerk_weight'] < 0:
        raise ValueError('Positive strike intensity and nonnegative smoothness weights required')
    task = cfg['task']; s = cfg['mppi']; limits = cfg['limits']
    if set(task) != {'duration_s', 'strike_direction', 'success_criterion'} or task['success_criterion'] != SCHEMA:
        raise ValueError('The new task has no legacy contact, speed, reversal, or wave gates')
    steps = round(task['duration_s']*30)
    if steps < 1 or not math.isclose(task['duration_s']*30, steps, abs_tol=1e-8):
        raise ValueError('Positive duration in whole 30 Hz intervals required')
    for value in (cfg['launch']['origin_m'], cfg['launch']['target_m'], task['strike_direction'], cfg['action']['jerk_limit_m_s3']):
        if len(value) != 3: raise ValueError('XYZ vectors required')
    if sum(x*x for x in task['strike_direction']) <= 0 or min(cfg['action']['jerk_limit_m_s3']) <= 0:
        raise ValueError('Nonzero direction and positive jerk bounds required')
    if limits['minimum_origin_z_m'] >= limits['maximum_origin_z_m']:
        raise ValueError('Ordered altitude limits required')
    for key in ('maximum_speed_m_s', 'maximum_specific_force_m_s2', 'minimum_specific_vertical_m_s2'):
        if limits[key] <= 0: raise ValueError('Positive vehicle limits required')
    if not 0 < limits['maximum_tilt_deg'] < 90: raise ValueError('Tilt bound must lie between 0 and 90 degrees')
    if s.get('mode') != 'open_loop' or s.get('parameterization') != 'targeted_fold_strike_v1':
        raise ValueError('The new objective requires its dedicated offline planner')
    for key in ('samples', 'proposal_count', 'support_points', 'iterations'):
        if isinstance(s[key], bool) or not isinstance(s[key], int) or s[key] < 1:
            raise ValueError('Positive integer MPPI budgets required')
    if s['proposal_count'] < 2 or s['samples'] % s['proposal_count'] or not 2 <= s['support_points'] <= steps:
        raise ValueError('Two template families, divisible sample budgets, and valid support points required')
    if not 0 < s['target_ess_fraction'] < 1: raise ValueError('ESS fraction must lie in (0,1)')
    knots = s['timing_source_knots']
    if len(knots) != 4 or knots[0] != 0 or knots[-1] != 1 or any(a >= b for a,b in zip(knots,knots[1:])):
        raise ValueError('Three ordered template timing intervals required')
    noise = [s[k] for k in ('control_point_noise_scales','timing_noise_scales','strength_noise_scales')]
    if not noise[0] or len({len(x) for x in noise}) != 1 or any(min(x) <= 0 for x in noise):
        raise ValueError('Positive matching noise mixtures required')
    n = fold['material_samples']; half = fold['half_window_fraction']*(n-1)
    if not isinstance(n, int) or n < 7 or not math.isclose(half, round(half)) or not 1 <= round(half) < (n-1)/2:
        raise ValueError('Fold window must cover whole material-grid intervals')
    if not 0 < fold['minimum_local_turn_deg'] < fold['minimum_fold_angle_deg'] < 180:
        raise ValueError('Ordered fold turning angles required')
    if not 0 < fold['start_material_max'] < fold['end_material_min'] < 1:
        raise ValueError('Ordered proximal and distal fold locations required')
    if not 0 <= fold['maximum_backward_step'] < fold['maximum_forward_step'] < fold['end_material_min']-fold['start_material_max']:
        raise ValueError('Fold tracking must reject disconnected proximal-to-distal jumps')
    if not isinstance(fold['minimum_observations'], int) or fold['minimum_observations'] < 3:
        raise ValueError('A fold requires at least three consecutive observations')


def event_terms(distance, tip_velocity, direction, settings):
    error = (distance / settings['distance_scale_m']).square()
    forward = (tip_velocity * direction).sum(-1).clamp_min(0)
    speed_squared = (forward / settings['speed_scale_m_s']).square()
    return dict(target=-error, strike=settings['intensity_weight'] * torch.exp(-error)
                * speed_squared / (1 + speed_squared))


def strike_terms(distance, tip_velocity, direction, normalized_jerk, settings):
    """One shared encounter supplies both target error and directed speed.

    The negative squared distance provides guidance even far from the target.
    The bounded, strictly increasing speed-squared term cannot overwhelm an
    arbitrarily large miss and has no hard speed cap. Jerk is a dimensionless
    mean over the complete fixed-duration command, not a second motion goal.
    """
    return dict(**event_terms(distance, tip_velocity, direction, settings),
                smoothness=-settings['jerk_weight'] * normalized_jerk.square().mean((-1, -2)))


def fold_features(q, material, settings):
    """World-rotation-invariant turning on a fixed material-coordinate grid.

    Opposing segment directions establish a folded-back shape. A wider local
    secant window locates the dominant bend without favoring the original mesh.
    """
    count = settings['material_samples']
    sample_s = torch.linspace(0, 1, count, device=q.device, dtype=q.dtype)
    right = torch.searchsorted(material.contiguous(), sample_s).clamp(1, len(material)-1)
    left = right-1
    blend = (sample_s-material[left])/(material[right]-material[left])
    points = q[..., left, :] + blend[:, None]*(q[..., right, :]-q[..., left, :])
    edge = points[..., 1:, :]-points[..., :-1, :]
    length = edge.norm(dim=-1, keepdim=True)
    tangent = edge/length.clamp_min(1e-12)
    pair_cosine = torch.einsum('...ic,...jc->...ij', tangent, tangent)
    opposition = torch.acos(pair_cosine.amin((-1, -2)).clamp(-1, 1))
    half = round(settings['half_window_fraction']*(count-1))
    incoming = points[..., half:-half, :]-points[..., :-2*half, :]
    outgoing = points[..., 2*half:, :]-points[..., half:-half, :]
    a_length = incoming.norm(dim=-1); b_length = outgoing.norm(dim=-1)
    cosine = (incoming*outgoing).sum(-1)/(a_length*b_length).clamp_min(1e-12)
    turning = torch.acos(cosine.clamp(-1, 1))
    turning = turning.masked_fill((a_length <= 1e-12) | (b_length <= 1e-12), 0)
    peak, index = turning.max(-1)
    location = sample_s[index+half]
    valid = torch.isfinite(q).all((-1, -2)) & (length[..., 0] > 1e-12).all(-1)
    return opposition, peak, location, valid


def advance_fold(tracking, previous_s, count, completed, features, eligible, settings):
    """Track a coherent dominant bend from proximal to distal material points.

    Static bends cannot advance. A peak jumping between disconnected portions
    resets the track. Completion needs several observations and is latched.
    """
    opposition, turning, location, finite = features
    visible = finite & eligible & (turning >= math.radians(settings['minimum_local_turn_deg']))
    seed = visible & (opposition >= math.radians(settings['minimum_fold_angle_deg']))
    seed &= location <= settings['start_material_max']
    step = location-previous_s
    connected = visible & tracking
    connected &= (step >= -settings['maximum_backward_step']-1e-12)
    connected &= (step <= settings['maximum_forward_step']+1e-12)
    next_tracking = connected | seed
    next_count = torch.where(connected, count+1, seed.long())
    next_s = torch.where(next_tracking, location, previous_s)
    reached = connected & (location >= settings['end_material_min'])
    reached &= next_count >= settings['minimum_observations']
    return next_tracking, next_s, next_count, completed | reached


def initialize(env):
    env.fold_tracking = torch.zeros_like(env.active)
    env.fold_last_s = torch.zeros_like(env.total)
    env.fold_count = torch.zeros_like(env.cutoff)
    env.fold_completed = torch.zeros_like(env.active)
    env.fold_completed_time_s = torch.full_like(env.total, float('inf'))
    env.encounter_fold_valid = torch.zeros_like(env.active)
    env.strike_value = torch.full_like(env.total, -torch.inf)
    env.strike_distance = torch.full_like(env.total, torch.inf)
    env.strike_velocity = env.encounter_tip_velocity.clone().zero_()
    env.strike_time = torch.full_like(env.total, torch.inf)
    env.strike_valid = torch.zeros_like(env.active)


def observe(env, previous, current, eligible, clock_left):
    """Record closest approach at physics rate, without contact-triggered stops.

    Position and velocity use the same segment fraction. Equal-distance ties
    retain the earliest encounter. The fold must already be confirmed at the
    start of this interval; later geometry cannot qualify an earlier encounter.
    """
    before = previous.positions_m[:, -1]
    after = current.positions_m[:, -1]
    displacement = after-before
    fraction = ((env.target-before)*displacement).sum(-1)
    fraction = (fraction/displacement.square().sum(-1).clamp_min(1e-20)).clamp(0, 1)
    closest = before+fraction[:, None]*displacement
    distance = (closest-env.target).norm(dim=-1)
    replace = eligible & (distance < env.encounter_distance)
    velocity = previous.velocities_m_s[:, -1]+fraction[:, None]*(
        current.velocities_m_s[:, -1]-previous.velocities_m_s[:, -1])
    env.encounter_distance = torch.where(replace, distance, env.encounter_distance)
    env.minimum_distance = env.encounter_distance.clone()
    env.encounter_tip_velocity = torch.where(replace[:, None], velocity, env.encounter_tip_velocity)
    env.encounter_time = torch.where(replace, clock_left+fraction*env.dt, env.encounter_time)
    env.encounter_fold_valid = torch.where(replace, env.fold_completed, env.encounter_fold_valid)
    # Task event and reporting event are distinct. Evaluate both endpoints and
    # the closest point of each physics segment, with co-timed velocities.
    # Only an already completed travelling fold can qualify an event.
    for event_fraction in (torch.zeros_like(fraction), fraction, torch.ones_like(fraction)):
        event_position = before + event_fraction[:, None]*displacement
        event_velocity = previous.velocities_m_s[:, -1] + event_fraction[:, None]*(
            current.velocities_m_s[:, -1]-previous.velocities_m_s[:, -1])
        event_distance = (event_position-env.target).norm(dim=-1)
        value = sum(event_terms(event_distance, event_velocity, env.direction,
                               env.settings['trajectory_objective']).values())
        forward = (event_velocity*env.direction).sum(-1) > 0
        select = eligible & env.fold_completed & forward & torch.isfinite(value) & (value > env.strike_value)
        env.strike_value = torch.where(select, value, env.strike_value)
        env.strike_distance = torch.where(select, event_distance, env.strike_distance)
        env.strike_velocity = torch.where(select[:, None], event_velocity, env.strike_velocity)
        env.strike_time = torch.where(select, clock_left+event_fraction*env.dt, env.strike_time)
        env.strike_valid |= select
    q = previous.positions_m+fraction[:, None, None]*(current.positions_m-previous.positions_m)
    env.encounter_q = torch.where(replace[:, None, None], q, env.encounter_q)
    material = torch.cat((env.wave_material.new_zeros(1), env.wave_material, env.wave_material.new_ones(1)))
    features = fold_features(current.positions_m, material, env.settings['fold_constraint'])
    was_complete = env.fold_completed
    env.fold_tracking, env.fold_last_s, env.fold_count, env.fold_completed = advance_fold(
        env.fold_tracking, env.fold_last_s, env.fold_count, env.fold_completed,
        features, eligible, env.settings['fold_constraint'])
    env.fold_completed_time_s = torch.where(~was_complete & env.fold_completed,
        clock_left+env.dt, env.fold_completed_time_s)
    env.success = env.strike_valid & ~env.failed


class StrikeCapture(RolloutCapture):
    def score(self, env, result, actions, settings):
        terms = strike_terms(env.strike_distance, env.strike_velocity,
                             env.direction, actions, settings)
        total = sum(terms.values())
        finite = torch.isfinite(total)
        admissible = finite & ~result['failed'] & env.strike_valid
        return total.masked_fill(~admissible, -torch.inf), terms
