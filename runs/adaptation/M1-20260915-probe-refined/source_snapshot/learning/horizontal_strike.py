"""Optional soft MPPI preferences. No additional feasibility or hit constraints."""
import torch


def features(q,v):
    edge=q[:,-3:]-q[:,-4:-1]
    unit=edge/edge.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    vertical_alignment=unit[:,:,2].square().mean(-1)
    tip=v[:,-1]
    vertical_velocity=tip[:,2].square()/tip.square().sum(-1).clamp_min(.25)
    return vertical_alignment,vertical_velocity


def cost_rate(q,v,origin,origin0,target,prepared,weights):
    alignment,velocity=features(q,v)
    distance=(q[:,-1]-target).norm(dim=-1)
    near=torch.exp(-(distance/weights['proximity_scale_m']).square())*prepared
    return (weights.get('vertical_excursion',0.)*(origin[:,2]-origin0[:,2]).square()
        +near*(weights.get('vertical_tip_velocity',0.)*velocity+weights.get('vertical_tip_alignment',0.)*alignment))


def contact_bonus(q,v,weights):
    alignment,velocity=features(q,v)
    return weights.get('horizontal_contact',0.)*(1-alignment)*(1-velocity)


def outward_reach(q,direction,length):
    """Forward projection from the actual attachment, normalized by rest length.

    Translation cannot improve this feature. No straightness is demanded during
    preparation: a fold is free to form and travel before the outward extension.
    """
    return (((q[:,-1]-q[:,0])*direction).sum(-1)/length).clamp(0,1)


def reach_cost_rate(q,origin,origin0,target,prepared,direction,length,weights):
    displacement=origin-origin0
    forward=(displacement*direction).sum(-1)
    # Sideways drift in the horizontal plane also allows an oblique swing to
    # masquerade as an outward cast in the side view.
    lateral=displacement-forward[:,None]*direction
    near=torch.exp(-((q[:,-1]-target).norm(dim=-1)/weights['proximity_scale_m']).square())*prepared
    return (weights.get('drone_approach',0.)*forward.clamp_min(0).square()
        +weights.get('lateral_excursion',0.)*lateral[:,:2].square().sum(-1)
        +weights.get('near_target_reach',0.)*near*(1-outward_reach(q,direction,length)).square())


def reach_contact_bonus(q,direction,length,weights):
    return weights.get('outward_contact',0.)*outward_reach(q,direction,length).square()
