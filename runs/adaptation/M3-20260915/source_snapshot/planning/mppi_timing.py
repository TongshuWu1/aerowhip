"""Batched monotone retiming of exact 30 Hz jerk proposals.

Three source phases cover preparation, reversal and follow-through. Phase
durations and axis strengths are editable; no predicted state is imported.
"""
import torch


def phase_durations(parameters,source_knots):
    logits=torch.cat((parameters[...,-3:-1,:],torch.zeros_like(parameters[...,-1:,:])),-2)
    # Bounded logits avoid degenerate numerical intervals, not a vehicle limit.
    return torch.softmax(logits.clamp(-8,8)+(source_knots[1:]-source_knots[:-1]).log()[:,None],dim=-2)


def retime_jerk(baselines,durations,source_knots):
    """Exact interval averages of a retimed zero-order-held source sequence.

    Integrate each phase's overlapping destination interval separately. Point
    sampling/interpolating at knots would alias short, shifted jerk pulses.
    Input shapes (..., steps, 3), (..., 3 phases, 3 axes).
    """
    steps=baselines.shape[-2]
    prefix=torch.cat((torch.zeros_like(baselines[...,:1,:]),baselines.cumsum(-2)),-2)
    t=torch.arange(steps+1,device=baselines.device,dtype=baselines.dtype)[:,None]/steps
    destination=torch.cat((torch.zeros_like(durations[...,:1,:]),durations.cumsum(-2)),-2)
    def integral(u):
        index=(u*steps).floor().long().clamp(0,steps-1)
        return prefix.gather(-2,index)+baselines.gather(-2,index)*(u*steps-index)
    output=torch.zeros_like(baselines)
    for phase in range(3):
        left=destination[...,phase:phase+1,:];right=destination[...,phase+1:phase+2,:]
        a=torch.maximum(t[:-1],left).minimum(right)
        b=torch.maximum(t[1:],left).minimum(right)
        slope=(source_knots[phase+1]-source_knots[phase])/durations[...,phase:phase+1,:]
        ua=(source_knots[phase]+(a-left)*slope).clamp(0,1)
        ub=(source_knots[phase]+(b-left)*slope).clamp(0,1)
        # Prefix integrates in source command-index units; /slope gives
        # destination index units, so the interval average needs no extra /dt.
        output+=(integral(ub)-integral(ua))/slope
    return output


class TimedProposals:
    def __init__(self,baselines,basis,settings):
        self.baselines=baselines;self.basis=basis;self.settings=settings
        self.groups=settings['proposal_count'];self.per=settings['samples']//self.groups
        self.rows=basis.shape[1]+3
        self.family=torch.arange(self.groups,device=baselines.device)%len(baselines)
        self.source_knots=baselines.new_tensor(settings['timing_source_knots'])

    def sample(self,means,rng):
        noise=torch.randn(self.groups,self.per,self.rows,3,device=means.device,dtype=means.dtype,generator=rng)
        choice=torch.arange(self.per,device=means.device)%len(self.settings['control_point_noise_scales'])
        for sl,key in ((slice(None,-3),'control_point_noise_scales'),(slice(-3,-1),'timing_noise_scales'),(slice(-1,None),'strength_noise_scales')):
            noise[:,:,sl]*=means.new_tensor(self.settings[key])[choice][None,:,None,None]
        return means[:,None]+noise

    def decode(self,parameters):
        # Always preserve the group axis, including one mean per group.
        extra=parameters.shape[1:-2]
        base=self.baselines[self.family].reshape((self.groups,)+(1,)*len(extra)+self.baselines.shape[1:])
        base=base.expand(parameters.shape[:-2]+self.baselines.shape[1:])
        duration=phase_durations(parameters,self.source_knots)
        warped=retime_jerk(base,duration,self.source_knots)
        gain=parameters[...,-1:,:].clamp(-4,4).exp()
        residual=torch.einsum('hp,...pc->...hc',self.basis,parameters[...,:-3,:])
        return torch.tanh(torch.atanh(warped.clamp(-1+1e-12,1-1e-12))*gain+residual)
