"""Non-trainable time-indexed initialization for bounded force policies."""
import math
import torch


def validate_action_prior(value, action_dim):
    if value is None or not value.get('enabled', True):
        return None
    durations=[int(x) for x in value['phase_control_steps']]
    actions=[[float(x) for x in row] for row in value['actions']]
    total=int(value['total_control_steps'])
    if not durations or len(durations)!=len(actions) or min(durations)<1 or sum(durations)>total:
        raise ValueError('Invalid action-prior phase durations.')
    if any(len(row)!=action_dim or any(not math.isfinite(x) or abs(x)>=1 for x in row) for row in actions):
        raise ValueError('Action-prior vectors must match the action dimension and lie inside (-1, 1).')
    return dict(enabled=True,source=str(value.get('source','configured_action_prior')),
                total_control_steps=total,phase_control_steps=durations,actions=actions)


def prior_latent(observation, prior, action_dim):
    action=observation.new_zeros((*observation.shape[:-1],action_dim))
    if prior is None:return action
    elapsed=torch.round((1-observation[...,-1]).clamp(0,1)*prior['total_control_steps'])
    start=0
    for duration,values in zip(prior['phase_control_steps'],prior['actions'],strict=True):
        end=start+duration
        action=torch.where(((elapsed>=start)&(elapsed<end))[...,None],action.new_tensor(values),action)
        start=end
    return torch.atanh(action.clamp(-1+1e-6,1-1e-6))
