"""Versioned task outcomes; missing versions preserve historical semantics."""
import torch
import math

LEGACY='legacy_strike_v1'
TIP_CONTACT='tip_contact_v1'
TWO_TARGET='ordered_two_target_v1'


def hit_time_bonus(weights,success,failed,time_s):
    """One bounded bonus at actual successful entry; missing weight is historical zero.

    The decay time is independent of planning lookahead and episode duration.
    Misses/invalid times never earn credit, including NaN/inf placeholders.
    """
    weight=float(weights.get('early_hit',0.));scale=float(weights.get('early_hit_scale_s',1.))
    if not math.isfinite(weight) or weight<0 or not math.isfinite(scale) or scale<=0:
        raise ValueError('Early-hit bonus needs a nonnegative weight and positive time scale')
    eligible=success&~failed&torch.isfinite(time_s)&(time_s>=0)
    safe_time=torch.where(eligible,time_s,torch.zeros_like(time_s))
    return torch.where(eligible,weight*torch.exp(-safe_time/scale),torch.zeros_like(time_s))


def record_hit_velocity(previous,current,fraction,hit,saved):
    """Latch velocity at successful tip entry, independently of other-node contact."""
    velocity=previous+fraction.clamp(0,1)[:,None]*(current-previous)
    return torch.where(hit[:,None],velocity,saved)


def impact_bonus(weights,success,failed,tip_velocity,direction):
    """Soft directed tip-energy proxy; no measured impact-force claim or speed cap."""
    weight=float(weights.get('impact',0.));scale=float(weights.get('impact_scale_m_s',4.))
    if not math.isfinite(weight) or weight<0 or not math.isfinite(scale) or scale<=0:
        raise ValueError('Impact bonus needs a nonnegative weight and positive speed scale')
    valid=success&~failed&torch.isfinite(tip_velocity).all(-1)
    velocity=torch.where(valid[:,None],tip_velocity,torch.zeros_like(tip_velocity))
    speed=(velocity*direction).sum(-1).clamp_min(0)
    energy=(speed/scale).square()
    return weight*(1-1/(1+energy))


def criterion(task):
    value=task.get('success_criterion',LEGACY)
    if value not in (LEGACY,TIP_CONTACT,TWO_TARGET):raise ValueError('Unknown PVA success criterion: '+str(value))
    return value


def label(task):
    if criterion(task)==TWO_TARGET:return 'Tip reaches T1 then T2'
    return 'Tip reaches target' if criterion(task)==TIP_CONTACT else 'Legacy speed / strategy / wave gates'


def contact_success(task,running,tip_entry,legacy_hit):
    # Feasibility is already included in running; end-of-command validation
    # can still revoke a hit. Other cable nodes do not veto a tip-contact task.
    return running&torch.isfinite(tip_entry) if criterion(task) in (TIP_CONTACT,TWO_TARGET) else legacy_hit
