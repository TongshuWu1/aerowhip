"""Optional target-free pullback task; kinematic criteria, not an energy proof."""
import torch

FIELDS=('whip_peak_forward','whip_loaded','whip_guidance','strike_position',
        'strike_backward_travel','whip_velocity_peaks','whip_velocity_peak_times',
        'strike_velocity_peaks','strike_velocity_peak_times','whip_loading_velocity',
        'whip_aligned_duration','strike_loading_velocity','strike_aligned_duration')
DEFAULT_PULLBACK=dict(minimum_pull_distance_m=.25,minimum_pull_speed_m_s=1.,
                     minimum_backward_distance_m=.1,minimum_backward_speed_m_s=.5)
DEFAULT_PROPAGATION=dict(minimum_peak_delay_s=.08,maximum_peak_span_s=.8,
                        maximum_distal_peak_age_s=.2,minimum_distal_amplification=1.25)
DEFAULT_CURVED_RELEASE=dict(minimum_turn_deg=45.,maximum_turn_deg=135.,minimum_aligned_duration_s=.08)

def initialize(env):
    spec=env.settings['trajectory_objective'].get('loading_direction')
    env.whip_loading_axis=env.direction if spec is None else env.total.new_tensor(spec)
    env.whip_loading_axis=env.whip_loading_axis/env.whip_loading_axis.norm()
    env.whip_peak_forward=torch.zeros_like(env.total)
    env.whip_loaded=torch.zeros_like(env.active)
    env.whip_guidance=torch.full_like(env.total,-torch.inf)
    env.strike_position=torch.zeros_like(env.encounter_tip_velocity)
    env.strike_backward_travel=torch.zeros_like(env.total)
    for name in ('whip_velocity_peaks','whip_velocity_peak_times','strike_velocity_peaks','strike_velocity_peak_times'):
        setattr(env,name,torch.zeros_like(env.encounter_tip_velocity))
    env.whip_loading_velocity=torch.zeros_like(env.encounter_tip_velocity)
    env.strike_loading_velocity=torch.zeros_like(env.encounter_tip_velocity)
    env.whip_aligned_duration=torch.zeros_like(env.total)
    env.strike_aligned_duration=torch.zeros_like(env.total)


def loading_turn_deg(loading_velocity,tip_velocity):
    """Horizontal change from the recorded loading direction to tip release."""
    a,b=loading_velocity[...,:2],tip_velocity[...,:2]
    cosine=(a*b).sum(-1)/(a.norm(dim=-1)*b.norm(dim=-1)).clamp_min(1e-12)
    return torch.rad2deg(torch.acos(cosine.clamp(-1,1)))


def curved_release(loading_velocity,tip_velocity,aligned_duration,settings):
    angle=loading_turn_deg(loading_velocity,tip_velocity)
    allowed=(loading_velocity[...,:2].norm(dim=-1)>1e-8)
    allowed &= (angle>=settings['minimum_turn_deg'])&(angle<=settings['maximum_turn_deg'])
    allowed &= aligned_duration+1e-12>=settings['minimum_aligned_duration_s']
    deficit=((settings['minimum_turn_deg']-angle).clamp_min(0)/settings['minimum_turn_deg']).square()
    deficit+=((angle-settings['maximum_turn_deg']).clamp_min(0)/(180-settings['maximum_turn_deg'])).square()
    deficit+=((settings['minimum_aligned_duration_s']-aligned_duration).clamp_min(0)/settings['minimum_aligned_duration_s']).square()
    return allowed,deficit


def aligned_interval_duration(duration,before_allowed,after_allowed,dt):
    """Count complete valid physics intervals, not the first isolated sample."""
    return torch.where(before_allowed & after_allowed,duration+dt,torch.zeros_like(duration))

def group_speeds(velocity,material,direction):
    """Three material bands, equally sampled in rest arclength; root excluded."""
    s=torch.linspace(0,1,21,device=velocity.device,dtype=velocity.dtype)
    right=torch.searchsorted(material.contiguous(),s).clamp(1,len(material)-1);left=right-1
    f=(s-material[left])/(material[right]-material[left])
    sampled=(velocity[:,left]*(1-f[:,None])+velocity[:,right]*f[:,None])@direction
    return torch.stack((sampled[:,1:8].mean(-1),sampled[:,8:15].mean(-1),sampled[:,15:].mean(-1)),-1)

def propagation(peaks,times,clock,settings):
    """Kinematic pulse proxy: delayed, amplified velocity peaks toward the tip.

    This rejects a synchronized rigid rotation; it does not measure energy flux.
    """
    delays=times[:,1:]-times[:,:-1];span=times[:,-1]-times[:,0];age=clock-times[:,-1]
    allowed=(delays>=settings['minimum_peak_delay_s']).all(-1)
    allowed &= (span<=settings['maximum_peak_span_s'])&(age>=0)&(age<=settings['maximum_distal_peak_age_s'])
    allowed &= peaks[:,-1]>=settings['minimum_distal_amplification']*peaks[:,0]
    deficit=((settings['minimum_peak_delay_s']-delays).clamp_min(0)/settings['minimum_peak_delay_s']).square().sum(-1)
    deficit+=((span-settings['maximum_peak_span_s']).clamp_min(0)/settings['maximum_peak_span_s']).square()
    deficit+=((age-settings['maximum_distal_peak_age_s']).clamp_min(0)/settings['maximum_distal_peak_age_s']).square()
    deficit+=(settings['minimum_distal_amplification']*peaks[:,0]-peaks[:,-1]).clamp_min(0).square()
    return allowed,deficit

def advance(peak,loaded,position,speed,eligible,settings,*,loading_position=None,loading_speed=None,loading_allowed=None):
    separate_loading=loading_position is not None
    load_p=loading_position if separate_loading else position
    load_v=loading_speed if separate_loading else speed
    load_ok=loading_allowed if separate_loading else torch.ones_like(eligible)
    peak=torch.where(eligible,torch.maximum(peak,position),peak)
    loaded=loaded | (eligible & load_ok & (load_p>=settings['minimum_pull_distance_m'])
                    & (load_v>=settings['minimum_pull_speed_m_s']))
    backward=peak-position
    allowed=loaded & (backward>=settings['minimum_backward_distance_m'])
    allowed &= speed<=-settings['minimum_backward_speed_m_s']
    # Smooth proposal guidance is never used to qualify or export a plan.
    deficit=((settings['minimum_pull_distance_m']-peak).clamp_min(0)/settings['minimum_pull_distance_m']).square()
    if separate_loading:
        deficit=torch.where(loaded,torch.zeros_like(deficit),
            ((settings['minimum_pull_distance_m']-load_p).clamp_min(0)/settings['minimum_pull_distance_m']).square())
    deficit+=((settings['minimum_backward_distance_m']-backward).clamp_min(0)/settings['minimum_backward_distance_m']).square()
    deficit+=((speed+settings['minimum_backward_speed_m_s']).clamp_min(0)/settings['minimum_backward_speed_m_s']).square()
    deficit+=(~loaded).to(position.dtype)
    return peak,loaded,backward,allowed,deficit
