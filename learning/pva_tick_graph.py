"""Capture a complete PVA physics/reward tick without changing its equations.

The same PVAEnvironment._tick implements both eager and captured execution.
Packet events and substep durations are resolved on the CPU and copied as
inputs; graphs specialize only on the number of pose substeps, not time/index.
"""
from copy import copy
from dataclasses import replace
import math
import numpy as np
import torch
from simulator.cable import DderState
from simulator.research_physics import ResearchPhysics
from simulator.research_pose import tensor_midpoint
from planning.whip_objective import ENCOUNTER_FIELDS
from planning.strike_objective import FIELDS as STRIKE_FIELDS

POSE=('position','velocity','rotation','omega_tracking','compensation','rotation_command_from_tracking')
FIELDS=('active','evolving','success','failed','contact','tip_contact','minimum_distance','initial_distance',
        'best_quality','termination_time','cutoff','origin0','target','total',
        'pull_ready','reverse_ready','pull_peak','pull_credit','reverse_credit',
        'wave_stage','wave_dwell','wave_credit','wave_completion_time','reach_credit','brake_credit')+ENCOUNTER_FIELDS+STRIKE_FIELDS


def state_values(env):
    return (*(getattr(env.pose,n) for n in POSE),env.state.positions_m,env.state.velocities_m_s,
            *(getattr(env,n) for n in FIELDS+getattr(env,'extra_tick_fields',())))


def set_state(env,values):
    env.pose=replace(env.pose,**dict(zip(POSE,values[:6])))
    env.state=DderState(*values[6:8])
    for name,value in zip(FIELDS+getattr(env,'extra_tick_fields',()),values[8:]):setattr(env,name,value)


def schedule(env,left,right):
    events=np.arange(len(env.packets))/30+env.delay
    bounds=np.r_[left,events[(events>left+1e-12)&(events<right-1e-12)],right]
    result=[]
    for a,b in zip(bounds[:-1],bounds[1:]):
        i=np.searchsorted(events,(a+b)/2+1e-12,side='right')-1
        command=env.hover if i<0 else env.packets[i]
        n=max(1,math.ceil((b-a)/env.pose_maximum-1e-10))
        result.extend([(command,(b-a)/n)]*n)
    return result


class TickGraph:
    def __init__(self,env,reward,steps,left):
        self.env=copy(env)
        self.inputs=tuple(x.clone() for x in state_values(env))
        self.reward=reward.clone();self.left=env.total.clone();self.cutoff=env.cutoff.clone()
        self.commands=tuple(c.clone() for c,h in steps)
        self.durations=tuple(env.total.new_tensor(h) for c,h in steps)
        self.env.cable_stepper=ResearchPhysics(env.engine.physics,env.state,env.dt,graph=False,
            fast_solve=env.cable_stepper.fast_solve,fast_geometry=env.cable_stepper.fast_geometry)
        self.env._pose_advance=self.pose_advance
        self.load(env,reward,steps,left)
        stream=torch.cuda.Stream(device=env.device);stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.load(env,reward,steps,left);self.execute()
        torch.cuda.current_stream().wait_stream(stream)
        self.load(env,reward,steps,left)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.outputs=self.execute()

    def load(self,env,reward,steps,left):
        for dst,src in zip(self.inputs,state_values(env)):dst.copy_(src)
        self.reward.copy_(reward);self.left.fill_(left);self.cutoff.fill_(env.index+1)
        for dst,(src,h),duration in zip(self.commands,steps,self.durations):
            dst.copy_(src);duration.fill_(h)

    def pose_advance(self,left,right):
        env=self.env;valid=torch.ones_like(env.active)
        for command,h in zip(self.commands,self.durations):
            proposal,ok=tensor_midpoint(env.pose,command,h,env.pose_stepper.parameters,env.engine.drone.residual)
            if env.targeted_strike:
                from simulator.predicted_envelope import attitude_valid
                ok = ok & attitude_valid(proposal.rotation, proposal.rotation_command_from_tracking,
                                         env.settings['limits']['maximum_tilt_deg'])
            valid&=ok;accepted=valid&env.evolving
            env.pose=replace(env.pose,**{name:torch.where(accepted.reshape((-1,)+(1,)*(getattr(env.pose,name).ndim-1)),
                getattr(proposal,name),getattr(env.pose,name)) for name in POSE[:4]})
        return valid

    def execute(self):
        set_state(self.env,self.inputs)
        reward=self.env._tick(self.reward,0.,0.,self.left,self.cutoff)
        return (reward,*state_values(self.env))

    def __call__(self,env,reward,steps,left):
        self.load(env,reward,steps,left);self.graph.replay()
        outputs=tuple(x.clone() for x in self.outputs)
        set_state(env,outputs[1:]);return outputs[0]


def run_tick(env,reward,left,right):
    steps=schedule(env,left,right);key=len(steps)
    if key not in env.tick_graphs:env.tick_graphs[key]=TickGraph(env,reward,steps,left)
    return env.tick_graphs[key](env,reward,steps,left)
