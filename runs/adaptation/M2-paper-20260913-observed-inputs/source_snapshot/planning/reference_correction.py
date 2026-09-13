"""Offline B-spline command correction toward a frozen physical-motion reference.

Uses the existing adaptive-temperature sampling update. No target event ends
the rollout and no original whip reward enters the tracking objective.
"""
import numpy as np
import torch
from simulator.cable import DderState
from simulator.research_pose import settled_initial
from simulator.research_physics import ResearchPhysics
from simulator.research_reference import reference_packet_validity


def tracking_cost(tip, quadrotor, commands, reference, weights):
    """Mean squared 3D distances at fixed times (m^2), not path-nearest errors."""
    terms = dict(
        tip=(tip-reference['tip']).square().sum(-1).mean(-1),
        quadrotor=(quadrotor-reference['quadrotor']).square().sum(-1).mean(-1),
        command=(commands[...,:3]-reference['command']).square().sum(-1).mean(-1))
    return sum(weights[k]*v for k,v in terms.items()), terms


class CoupledRollout:
    """Reusable inference batch using production pose and DDER propagation."""
    def __init__(self, engine, count, origin, initial_cable, limits):
        self.engine=engine;self.limits=limits
        self.origin=torch.as_tensor(origin,device='cuda',dtype=torch.float64)
        root=(self.origin+self.origin.new_tensor(engine.offset))[None].expand(count,-1)
        self.pose=settled_initial(root,torch.zeros_like(root),engine.offset)
        q=torch.as_tensor(initial_cable,device='cuda',dtype=torch.float64)[None].expand(count,-1,-1).clone()
        if not torch.allclose(q[:,0],root,atol=1e-10,rtol=0):
            raise ValueError('Frozen initial cable root differs from the model attachment')
        self.state=DderState(q,torch.zeros_like(q))
        self.hover=torch.cat((self.pose.position,root.new_zeros(count,8)),-1)
        self.step=ResearchPhysics(engine.physics,self.state,engine.dt_s,
            graph=True,fast_solve=True,fast_geometry=True)

    @torch.no_grad()
    def __call__(self,packets,times,grid):
        pose=self.engine.drone.predict(self.pose,packets,times,grid,self.engine.offset,
            graph=True,hover_command=self.hover,maximum_tilt_deg=self.limits['maximum_tilt_deg'])
        valid=pose['valid'].all(-1)
        p=pose['position_origin_m'];roots=pose['position_attachment_m']
        valid &= torch.isfinite(p).all((-1,-2))
        valid &= ((p[...,2]>=self.limits['minimum_origin_z_m']) &
                  (p[...,2]<=self.limits['maximum_origin_z_m'])).all(-1)
        state=self.state;qs=[state.positions_m];vs=[state.velocities_m_s]
        for k in range(1,len(grid)):
            boundary=torch.where(valid[:,None],roots[:,k],state.positions_m[:,0])
            proposed=self.step(state,boundary)
            q,v=proposed.positions_m,proposed.velocities_m_s
            good=(torch.isfinite(q).all((-1,-2)) & torch.isfinite(v).all((-1,-2)) &
                  (q.abs()<20).all((-1,-2)) & (v.norm(dim=-1)<100).all(-1) &
                  (q[...,2]>=self.limits['minimum_cable_z_m']).all(-1))
            valid &= good
            state=DderState(torch.where(valid[:,None,None],q,state.positions_m),
                           torch.where(valid[:,None,None],v,state.velocities_m_s))
            qs.append(state.positions_m);vs.append(state.velocities_m_s)
        return dict(pose,cable_positions_m=torch.stack(qs,1),
                    cable_velocities_m_s=torch.stack(vs,1),complete_valid=valid)


def command_valid(packets, limits):
    valid,_=reference_packet_validity(packets,limits)
    return (valid.all(-1) & (packets[...,2]>=limits['minimum_origin_z_m']).all(-1) &
            (packets[...,2]<=limits['maximum_origin_z_m']).all(-1))
