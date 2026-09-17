"""Targeted strike objective with separately reported travelling-fold diagnostics.

Development implementation. This measures pre-contact kinematics, not impact
energy transferred to an object. No archived shape or vehicle-reversal reward
enters this objective. The geometric fold test is an operational definition,
not a measurement of wave energy or a proof of a travelling-wave solution.
"""
from copy import deepcopy
import math

import torch
from planning.mppi_live import RolloutCapture
from planning import free_whip

SCHEMA = 'targeted_fold_strike_v1'
OBJECTIVE = dict(schema=SCHEMA, distance_scale_m=0.10, speed_scale_m_s=4.0,
                 intensity_weight=4.0, jerk_weight=0.01)
FOLD = dict(material_samples=21, half_window_fraction=0.10,
            minimum_fold_angle_deg=90.0, minimum_local_turn_deg=15.0,
            start_material_max=0.40, end_material_min=0.75,
            maximum_backward_step=0.05, maximum_forward_step=0.20,
            minimum_observations=4)
FIELDS = ('strike_horizontal_error_m', 'fold_tracking', 'fold_last_s', 'fold_count', 'fold_completed',
          'fold_completed_time_s', 'encounter_fold_valid', 'strike_value',
          'strike_distance', 'strike_velocity', 'strike_root_velocity', 'maximum_directional_speed_gain', 'strike_time', 'strike_valid', 'strike_fold_valid') + free_whip.FIELDS


def enabled(settings):
    return settings.get('trajectory_objective', {}).get('schema') == SCHEMA


def uses_templates(settings):
    return settings.get('mppi',{}).get('initialization','templates')!='from_scratch'


def requires_fold(settings):
    return settings.get('fold_requirement','required')=='required'


def default_objective():
    return deepcopy(OBJECTIVE)


def validate_settings(cfg):
    from simulator.pva_commands import SCHEMA as COMMAND_SCHEMA
    from planning.position_spline import SCHEMA as SPLINE_SCHEMA
    spline=cfg.get('command_contract')==SPLINE_SCHEMA
    if cfg.get('method') != 'mppi' or cfg.get('command_contract') not in (COMMAND_SCHEMA,SPLINE_SCHEMA):
        raise ValueError('Targeted fold strikes require offline PVA MPPI')
    if 'reward' in cfg or cfg.get('ppo_objective'):
        raise ValueError('Legacy reward dictionaries cannot enter the new strike objective')
    if cfg.get('fold_requirement','required') not in ('required','diagnostic_only'):
        raise ValueError('Fold requirement must be required or diagnostic_only')
    objective = cfg['trajectory_objective']; fold = cfg['fold_constraint']
    optional = {'lateral_weight', 'lateral_scale_m', 'speed_proximity_scale_m', 'maximum_strike_angle_deg', 'prefer_aligned_strike', 'speed_metric', 'minimum_tip_speed_gain_m_s', 'horizontal_cable_weight', 'horizontal_cable_scale_m', 'free_target', 'pullback', 'minimum_release_time_s'}
    if not set(OBJECTIVE) <= set(objective) or set(objective)-set(OBJECTIVE)-optional or set(fold) != set(FOLD):
        raise ValueError('Use the explicit targeted-strike objective and fold settings')
    def finite(x):
        if isinstance(x, dict): return all(finite(v) for v in x.values())
        if isinstance(x, list): return all(finite(v) for v in x)
        return math.isfinite(x) if isinstance(x, (int, float)) else True
    if not finite(cfg): raise ValueError('Settings must be finite')
    if not isinstance(objective.get('free_target',False),bool):raise ValueError('Free-target setting must be boolean')
    if objective.get('free_target',False):
        if 'minimum_tip_speed_gain_m_s' not in objective:raise ValueError('Free-target whipping requires a minimum tip speed gain')
        pullback=objective.get('pullback',{})
        if set(pullback)!=set(free_whip.DEFAULT_PULLBACK) or any(v<=0 for v in pullback.values()):
            raise ValueError('Free-target whipping requires explicit positive pullback criteria')
    elif 'pullback' in objective:raise ValueError('Pullback criteria belong to the free-target task')
    if 'minimum_release_time_s' in objective and (not objective.get('free_target') or not 0<=objective['minimum_release_time_s']<cfg['task']['duration_s']):
        raise ValueError('Minimum release time must be inside a free-target maneuver')
    for key in ('distance_scale_m', 'speed_scale_m_s'):
        if objective[key] <= 0: raise ValueError('Strike scales must be positive')
    if objective.get('speed_proximity_scale_m', objective['distance_scale_m']) <= 0:
        raise ValueError('Positive speed proximity scale required')
    if 'maximum_strike_angle_deg' in objective and not 0 < objective['maximum_strike_angle_deg'] <= 90:
        raise ValueError('Maximum strike angle must be in (0, 90] degrees')
    if objective.get('speed_metric','absolute_tip_speed') not in ('absolute_tip_speed','tip_gain_over_root'):
        raise ValueError('Unknown strike speed metric')
    if 'minimum_tip_speed_gain_m_s' in objective:
        if objective.get('speed_metric')!='tip_gain_over_root' or objective['minimum_tip_speed_gain_m_s']<=0:
            raise ValueError('Positive minimum tip speed gain requires the root-relative speed metric')
    if not isinstance(objective.get('prefer_aligned_strike',False),bool):
        raise ValueError('Aligned-strike preference must be boolean')
    if objective.get('prefer_aligned_strike',False) and 'maximum_strike_angle_deg' not in objective:
        raise ValueError('Aligned-strike preference requires a maximum strike angle')
    if objective['intensity_weight'] <= 0 or objective['jerk_weight'] < 0:
        raise ValueError('Positive strike intensity and nonnegative smoothness weights required')
    if objective.get('lateral_weight', 0) < 0 or objective.get('lateral_scale_m', 1) <= 0:
        raise ValueError('Lateral weight must be nonnegative and its distance scale positive')
    if objective.get('horizontal_cable_weight', 0) < 0 or objective.get('horizontal_cable_scale_m', .1) <= 0:
        raise ValueError('Horizontal cable weight must be nonnegative and its height scale positive')
    if objective.get('lateral_weight', 0) > 0 and sum(x*x for x in cfg['task']['strike_direction'][:2]) <= 0:
        raise ValueError('Lateral preference requires a horizontal strike direction')
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
    if s.get('mode') != 'open_loop' or s.get('parameterization') != (SPLINE_SCHEMA if spline else 'targeted_fold_strike_v1'):
        raise ValueError('The new objective requires its dedicated offline planner')
    for key in ('samples', 'proposal_count', 'support_points', 'iterations'):
        if isinstance(s[key], bool) or not isinstance(s[key], int) or s[key] < 1:
            raise ValueError('Positive integer MPPI budgets required')
    if s['proposal_count'] < 2 or s['samples'] % s['proposal_count'] or not 2 <= s['support_points'] <= steps:
        raise ValueError('Two template families, divisible sample budgets, and valid support points required')
    if not 0 < s['target_ess_fraction'] < 1: raise ValueError('ESS fraction must lie in (0,1)')
    if spline:
        from deployment.braking_recovery import validate as validate_recovery
        validate_recovery(cfg['recovery'])
        if s.get('initialization','templates') not in ('templates','from_scratch'):
            raise ValueError('Unknown spline initialization')
        if not uses_templates(cfg):
            if not 0 <= s.get('fresh_sample_fraction',-1) <= 1:
                raise ValueError('Fresh sample fraction must lie in [0,1]')
            if 'proposal_templates_path' in cfg:
                raise ValueError('From-scratch planning must not specify previous motion templates')
        elif s.get('position_correlation_length',0)<=0 or not s.get('initial_strengths') or not all(0<x<=1 for x in s['initial_strengths']):
            raise ValueError('Positive spline correlation length and seed strengths in (0,1] required')
        if s.get('position_noise_basis','correlated') not in ('correlated','jerk'):
            raise ValueError('Unknown position-noise basis')
        if s['support_points']!=9 or not s.get('position_noise_scales_m') or min(s['position_noise_scales_m'])<=0:
            raise ValueError('Spline requires nine free control points and positive position noise scales')
    else:
        _validate_timing(s)
    _validate_fold(fold)


def _validate_timing(s):
    knots = s['timing_source_knots']
    if len(knots) != 4 or knots[0] != 0 or knots[-1] != 1 or any(a >= b for a,b in zip(knots,knots[1:])):
        raise ValueError('Three ordered template timing intervals required')
    noise = [s[k] for k in ('control_point_noise_scales','timing_noise_scales','strength_noise_scales')]
    if not noise[0] or len({len(x) for x in noise}) != 1 or any(min(x) <= 0 for x in noise):
        raise ValueError('Positive matching noise mixtures required')


def _validate_fold(fold):
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


def strike_angle_deg(velocity, direction):
    """Angle of nonzero tip velocity to the requested world-frame direction."""
    cosine=(velocity*direction).sum(-1)/(velocity.norm(dim=-1)*direction.norm()).clamp_min(1e-12)
    return torch.rad2deg(torch.acos(cosine.clamp(-1,1)))


def strike_direction_allowed(velocity, direction, settings):
    """Apply the optional cone at the same instant as distance and speed."""
    forward=(velocity*direction).sum(-1)
    allowed=forward>0
    if 'maximum_strike_angle_deg' in settings:
        threshold=math.cos(math.radians(settings['maximum_strike_angle_deg']))
        # Relative tolerance only covers floating-point roundoff at the boundary.
        norm=velocity.norm(dim=-1)*direction.norm()
        allowed &= forward >= (threshold-1e-12)*norm
    return allowed


def rewarded_speed(tip_velocity, direction, settings, root_velocity=None):
    forward=(tip_velocity*direction).sum(-1).clamp_min(0)
    if settings.get('speed_metric','absolute_tip_speed')=='tip_gain_over_root':
        if root_velocity is None:raise ValueError('Root velocity is required for the speed-gain objective')
        root_forward=(root_velocity*direction).sum(-1).clamp_min(0)
        forward=(forward-root_forward).clamp_min(0)
    return forward


def strike_speed_allowed(tip_velocity, direction, settings, root_velocity=None):
    gain=rewarded_speed(tip_velocity,direction,settings,root_velocity)
    if 'minimum_tip_speed_gain_m_s' not in settings:return torch.ones_like(gain,dtype=torch.bool)
    return gain >= settings['minimum_tip_speed_gain_m_s']


def horizontal_cable_error(q):
    """Arclength RMS height relative to the attachment plane, in meters.

    Integrate the squared height of the piecewise-linear centerline exactly.
    A sideways-moving hanging cable remains costly; horizontal translation or
    yaw rotation does not change the error. Subdividing segments is invariant.
    """
    height=q[...,2]-q[...,:1,2]
    a,b=height[...,:-1],height[...,1:]
    length=(q[...,1:,:]-q[...,:-1,:]).norm(dim=-1)
    return ((length*(a.square()+a*b+b.square())/3).sum(-1)
            /length.sum(-1).clamp_min(1e-12)).clamp_min(0).sqrt()


def event_terms(distance, tip_velocity, direction, settings, *, root_velocity=None, horizontal_error_m=None):
    error = (distance / settings['distance_scale_m']).square()
    forward = rewarded_speed(tip_velocity,direction,settings,root_velocity)
    speed_squared = (forward / settings['speed_scale_m_s']).square()
    # Missing scale preserves every historical objective exactly.
    proximity = (distance / settings.get('speed_proximity_scale_m', settings['distance_scale_m'])).square()
    if settings.get('free_target',False):
        error=torch.zeros_like(distance);proximity=torch.zeros_like(distance)
    alignment=1.
    if settings.get('prefer_aligned_strike',False):
        cosine=(tip_velocity*direction).sum(-1)/(tip_velocity.norm(dim=-1)*direction.norm()).clamp_min(1e-12)
        edge=math.cos(math.radians(settings['maximum_strike_angle_deg']))
        # Full speed credit on-axis, zero at the cone edge; no extra angle weight.
        alignment=((cosine-edge)/(1-edge)).clamp(0,1)
    terms=dict(target=-error, strike=settings['intensity_weight'] * torch.exp(-proximity)
                * speed_squared / (1 + speed_squared) * alignment)
    if settings.get('horizontal_cable_weight',0)>0:
        if horizontal_error_m is None:raise ValueError('Horizontal strike requires co-timed cable geometry')
        terms['horizontal_cable']=-settings['horizontal_cable_weight']*(horizontal_error_m/settings.get('horizontal_cable_scale_m',.1)).square()
    return terms


def strike_terms(distance, tip_velocity, direction, normalized_jerk, settings, *, root_velocity=None, horizontal_error_m=None):
    """One shared encounter supplies both target error and directed speed.

    The negative squared distance provides guidance even far from the target.
    The bounded, strictly increasing speed-squared term cannot overwhelm an
    arbitrarily large miss and has no hard speed cap. Jerk is a dimensionless
    mean over the complete fixed-duration command, not a second motion goal.
    """
    return dict(**event_terms(distance, tip_velocity, direction, settings,root_velocity=root_velocity,horizontal_error_m=horizontal_error_m),
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
    free_whip.initialize(env)
    env.strike_horizontal_error_m=torch.zeros_like(env.total)
    env.fold_tracking = torch.zeros_like(env.active)
    env.fold_last_s = torch.zeros_like(env.total)
    env.fold_count = torch.zeros_like(env.cutoff)
    env.fold_completed = torch.zeros_like(env.active)
    env.fold_completed_time_s = torch.full_like(env.total, float('inf'))
    env.encounter_fold_valid = torch.zeros_like(env.active)
    env.strike_value = torch.full_like(env.total, -torch.inf)
    env.strike_distance = torch.full_like(env.total, torch.inf)
    env.strike_velocity = env.encounter_tip_velocity.clone().zero_()
    env.strike_root_velocity = env.encounter_tip_velocity.clone().zero_()
    env.maximum_directional_speed_gain = torch.full_like(env.total,-torch.inf)
    env.strike_time = torch.full_like(env.total, torch.inf)
    env.strike_valid = torch.zeros_like(env.active)
    env.strike_fold_valid = torch.zeros_like(env.active)


def observe(env, previous, current, eligible, clock_left):
    """Record closest approach at physics rate, without contact-triggered stops.

    Position and velocity use the same segment fraction. Equal-distance ties
    retain the earliest encounter. The fold must already be confirmed at the
    start of this interval when required; later geometry cannot qualify an earlier encounter.
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
    # Historical runs require fold completion; new runs record it diagnostically.
    free_target=env.settings['trajectory_objective'].get('free_target',False)
    fractions=(torch.ones_like(fraction),) if free_target else (torch.zeros_like(fraction), fraction, torch.ones_like(fraction))
    for event_fraction in fractions:
        event_position = before + event_fraction[:, None]*displacement
        event_velocity = previous.velocities_m_s[:, -1] + event_fraction[:, None]*(
            current.velocities_m_s[:, -1]-previous.velocities_m_s[:, -1])
        event_distance = (event_position-env.target).norm(dim=-1)
        root_velocity=previous.velocities_m_s[:,0]+event_fraction[:,None]*(current.velocities_m_s[:,0]-previous.velocities_m_s[:,0])
        horizontal_error=torch.zeros_like(event_distance)
        if env.settings['trajectory_objective'].get('horizontal_cable_weight',0)>0:
            event_q=previous.positions_m+event_fraction[:,None,None]*(current.positions_m-previous.positions_m)
            horizontal_error=horizontal_cable_error(event_q)
        value = sum(event_terms(event_distance, event_velocity, env.direction,
                               env.settings['trajectory_objective'],root_velocity=root_velocity,
                               horizontal_error_m=horizontal_error).values())
        forward = strike_direction_allowed(event_velocity, env.direction, env.settings['trajectory_objective'])
        fold_allowed=env.fold_completed if requires_fold(env.settings) else torch.ones_like(eligible)
        gain=rewarded_speed(event_velocity,env.direction,env.settings['trajectory_objective'],root_velocity)
        env.maximum_directional_speed_gain=torch.maximum(env.maximum_directional_speed_gain,
            torch.where(eligible & fold_allowed & forward,gain,-torch.inf))
        speed_allowed=strike_speed_allowed(event_velocity,env.direction,env.settings['trajectory_objective'],root_velocity)
        select = eligible & fold_allowed & forward & speed_allowed & torch.isfinite(value) & (value > env.strike_value)
        backward=torch.zeros_like(event_distance)
        if free_target:
            root_position=previous.positions_m[:,0]+event_fraction[:,None]*(current.positions_m[:,0]-previous.positions_m[:,0])
            position=((root_position-env.initial_state.positions_m[:,0])*env.direction).sum(-1)
            root_speed=(root_velocity*env.direction).sum(-1)
            env.whip_peak_forward,env.whip_loaded,backward,pullback_allowed,deficit=free_whip.advance(
                env.whip_peak_forward,env.whip_loaded,position,root_speed,eligible,env.settings['trajectory_objective']['pullback'])
            tip_forward=(event_velocity*env.direction).sum(-1)
            deficit+=((env.settings['trajectory_objective']['minimum_tip_speed_gain_m_s']-tip_forward).clamp_min(0)).square()
            guide=value-4*deficit
            release_time=clock_left+event_fraction*env.dt
            after_preparation=release_time>=env.settings['trajectory_objective'].get('minimum_release_time_s',0.)
            env.whip_guidance=torch.maximum(env.whip_guidance,torch.where(eligible & forward & after_preparation,guide,-torch.inf))
            select &= pullback_allowed & after_preparation
        env.strike_position=torch.where(select[:,None],event_position,env.strike_position)
        env.strike_backward_travel=torch.where(select,backward,env.strike_backward_travel)
        env.strike_fold_valid=torch.where(select,env.fold_completed,env.strike_fold_valid)
        env.strike_value = torch.where(select, value, env.strike_value)
        env.strike_horizontal_error_m=torch.where(select,horizontal_error,env.strike_horizontal_error_m)
        env.strike_distance = torch.where(select, event_distance, env.strike_distance)
        env.strike_velocity = torch.where(select[:, None], event_velocity, env.strike_velocity)
        env.strike_root_velocity = torch.where(select[:,None],root_velocity,env.strike_root_velocity)
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


def lateral_cost(packets, origin, direction, settings):
    """Soft drift cost about the vertical plane through the launch/strike axis.

    Uses all equally spaced 30 Hz reference positions, including the launch.
    No axis is disabled and no lateral feasibility threshold is introduced.
    """
    normal = torch.stack((-direction[1], direction[0], torch.zeros_like(direction[0])))
    normal = normal / normal.norm().clamp_min(1e-12)
    displacement = ((packets[..., :3] - origin[:, None, :]) * normal).sum(-1)
    return -settings.get('lateral_weight', 0.) * (displacement / settings.get('lateral_scale_m', 1.)).square().mean(-1)


class StrikeCapture(RolloutCapture):
    def score(self, env, result, actions, settings):
        terms = strike_terms(env.strike_distance, env.strike_velocity,
                             env.direction, actions, settings,root_velocity=env.strike_root_velocity,
                             horizontal_error_m=env.strike_horizontal_error_m)
        if settings.get('lateral_weight', 0.) > 0:
            terms['lateral'] = lateral_cost(result['packets'], env.origin0, env.direction, settings)
        total = sum(terms.values())
        finite = torch.isfinite(total)
        admissible = finite & ~result['failed'] & env.strike_valid
        admissible &= strike_direction_allowed(env.strike_velocity, env.direction, settings)
        admissible &= strike_speed_allowed(env.strike_velocity,env.direction,settings,env.strike_root_velocity)
        return total.masked_fill(~admissible, -torch.inf), terms
