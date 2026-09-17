"""Optional target-free pullback task; kinematic criteria, not an energy proof."""
import torch

FIELDS=('whip_peak_forward','whip_loaded','whip_guidance','strike_position',
        'strike_backward_travel')
DEFAULT_PULLBACK=dict(minimum_pull_distance_m=.25,minimum_pull_speed_m_s=1.,
                     minimum_backward_distance_m=.1,minimum_backward_speed_m_s=.5)

def initialize(env):
    env.whip_peak_forward=torch.zeros_like(env.total)
    env.whip_loaded=torch.zeros_like(env.active)
    env.whip_guidance=torch.full_like(env.total,-torch.inf)
    env.strike_position=torch.zeros_like(env.encounter_tip_velocity)
    env.strike_backward_travel=torch.zeros_like(env.total)

def advance(peak,loaded,position,speed,eligible,settings):
    peak=torch.where(eligible,torch.maximum(peak,position),peak)
    loaded=loaded | (eligible & (position>=settings['minimum_pull_distance_m'])
                    & (speed>=settings['minimum_pull_speed_m_s']))
    backward=peak-position
    allowed=loaded & (backward>=settings['minimum_backward_distance_m'])
    allowed &= speed<=-settings['minimum_backward_speed_m_s']
    # Smooth proposal guidance is never used to qualify or export a plan.
    deficit=((settings['minimum_pull_distance_m']-peak).clamp_min(0)/settings['minimum_pull_distance_m']).square()
    deficit+=((settings['minimum_backward_distance_m']-backward).clamp_min(0)/settings['minimum_backward_distance_m']).square()
    deficit+=((speed+settings['minimum_backward_speed_m_s']).clamp_min(0)/settings['minimum_backward_speed_m_s']).square()
    deficit+=(~loaded).to(position.dtype)
    return peak,loaded,backward,allowed,deficit
