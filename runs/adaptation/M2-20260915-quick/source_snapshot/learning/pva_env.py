"""Direct PVA planning in the fitted loaded-drone and DDER cable model.

No virtual force rollout. Training and MPPI call this same environment. The
open-loop sequence is frozen before export; observations here are simulated.
"""
from copy import deepcopy
from dataclasses import replace
import math
import numpy as np
import torch
from simulator.cable import DderState
from simulator.research_execution import ResearchExecutionModel
from simulator.research_pose import settled_initial,PoseStepper
from simulator.research_physics import ResearchPhysics
from simulator.research_reference import reference_packet_validity
from simulator.pva_commands import SCHEMA,integrate_jerk,sphere_entry
from learning.pva_success import criterion,contact_success,TIP_CONTACT,TWO_TARGET


def defaults(method='ppo'):
    if method not in ('ppo','mppi'):raise ValueError('Unknown PVA planner')
    cfg=dict(schema='pva_planner_settings_v1',method=method,command_contract=SCHEMA,
        model_path='runs/adaptation/20260909-pva-M0-bootstrap/candidate/model.json',
        performance=dict(fused_ticks=True,fast_solve=True,fast_geometry=True),
        launch=dict(origin_m=[-2.,0.,1.255],target_m=[-1.,0.,1.1],start_radius_m=.05,target_radius_m=.05),
        task=dict(duration_s=1.,target_radius_m=.05,minimum_directed_speed_m_s=4.,maximum_angle_deg=45.,
            strike_direction=[1.,0.,0.],first_contact_only=True),
        action=dict(jerk_limit_m_s3=[60.,60.,60.]),
        limits=dict(minimum_origin_z_m=.96,maximum_origin_z_m=2.8,minimum_cable_z_m=.02,
            maximum_specific_force_m_s2=18.28571428571429,minimum_specific_vertical_m_s2=2.,
            maximum_speed_m_s=5.,maximum_tilt_deg=60.),
        reward=dict(progress=60.,strike_quality=60.,success=200.,time_per_s=10.,failure=100.,
            invalid_contact=25.,displacement=5.,jerk=0.02,proximity_scale_m=.35),
        training=dict(batch_size=1024,hidden_dim=256,learning_rate=.0003,epochs=4,minibatch_size=8192,
            entropy_coefficient=.01,seed=655,minimum_attempts=20480,plateau_attempts=20480,
            maximum_attempts=500000,relative_improvement=.005,evaluate_every_updates=5),
        mppi=dict(samples=1024,iterations=0 if method=='mppi' else 100,temperature=10.,noise_std=.5,noise_correlation=.7,
            minimum_iterations=20,patience=15,seed=656),device='cuda')
    if method=='mppi':
        cfg['task']['success_criterion']=TIP_CONTACT
        cfg['task'].update(duration_s=5.,require_pullback=True,minimum_pull_distance_m=.25,minimum_pull_speed_m_s=1.,
            minimum_backward_distance_m=.1,minimum_backward_speed_m_s=.5)
        cfg['reward'].update(pull_phase=20.,reverse_phase=40.,impact=200.,impact_scale_m_s=4.)
        cfg['mppi'].update(mode='receding',initialization='pullback',horizon_s=2.,noise_std=.05,temperature=1.,minimum_iterations=3,patience=3,
            initial_minimum_iterations=20,initial_patience=15,control_prior=0.,
            terminal_distance=0.,terminal_velocity=0.,terminal_anchor=0.,terminal_command_speed=0.,terminal_command_acceleration=0.,
            strike_exit_speed=.5,strike_exit_climb=5.,strike_exit_acceleration=1.)
    return cfg


def update_pullback(env,running):
    """Ordered physical carrier motion, not a reversal of the desired packet."""
    task=env.settings['task'];weights=env.settings['reward']
    displacement=((env.pose.position-env.origin0)*env.direction).sum(-1)
    speed=(env.pose.velocity*env.direction).sum(-1)
    env.pull_peak=torch.where(running,torch.maximum(env.pull_peak,displacement),env.pull_peak)
    forward=torch.minimum((displacement/task['minimum_pull_distance_m']).clamp(0,1),
        (speed/task['minimum_pull_speed_m_s']).clamp(0,1))
    env.pull_ready|=running&(forward>=1)
    backward_distance=env.pull_peak-displacement
    backward=torch.minimum((backward_distance/task['minimum_backward_distance_m']).clamp(0,1),
        (-speed/task['minimum_backward_speed_m_s']).clamp(0,1))*env.pull_ready
    new_forward=torch.where(running,torch.maximum(env.pull_credit,forward),env.pull_credit)
    new_backward=torch.where(running,torch.maximum(env.reverse_credit,backward),env.reverse_credit)
    reward=weights.get('pull_phase',0.)*(new_forward-env.pull_credit)+weights.get('reverse_phase',0.)*(new_backward-env.reverse_credit)
    env.pull_credit,env.reverse_credit=new_forward,new_backward
    if weights.get('brake_progress',0.):
        # Credit actual deceleration continuously, before reverse speed/distance
        # crosses the release threshold. A monotone account prevents farming.
        brake=((task['minimum_pull_speed_m_s']-speed)/(
            task['minimum_pull_speed_m_s']+task['minimum_backward_speed_m_s'])).clamp(0,1)
        credit=torch.where(running&env.pull_ready,torch.maximum(env.brake_credit,brake),env.brake_credit)
        reward+=weights['brake_progress']*(credit-env.brake_credit)
        env.brake_credit=credit
    env.reverse_ready|=running&(backward>=1)
    # Returning forward after a token reversal cannot satisfy the new task.
    allowed=env.pull_ready&(backward_distance>=task['minimum_backward_distance_m'])&(speed<=-task['minimum_backward_speed_m_s'])
    return reward,allowed


class PVAEnvironment:
    def __init__(self,model,settings,*,root=None,batch_size=1,device='cuda',graph=True,fused_ticks=None,fast_solve=None,fast_geometry=None):
        performance=settings.get('performance',{})
        fused_ticks=performance.get('fused_ticks',True) if fused_ticks is None else fused_ticks
        fast_solve=performance.get('fast_solve',True) if fast_solve is None else fast_solve
        fast_geometry=performance.get('fast_geometry',True) if fast_geometry is None else fast_geometry
        # CPU remains the checked tensor reference; specialized solvers require CUDA.
        fast_solve=fast_solve and torch.device(device).type=='cuda'
        self.fused_ticks=fused_ticks and graph and torch.device(device).type=='cuda';self.tick_graphs={}
        self.model_config=deepcopy(model);self.settings=deepcopy(settings);self.device=torch.device(device);self.root=root
        from planning.strike_objective import enabled
        self.targeted_strike=enabled(self.settings)
        if self.targeted_strike:
            # The new objective bypasses legacy shaping and contact termination.
            # These three zero values are internal bookkeeping, not saved rewards.
            self.settings['reward']=dict(success=0.,failure=0.,jerk=0.)
        criterion(self.settings['task'])
        from planning.position_spline import SCHEMA as SPLINE_SCHEMA
        if settings.get('command_contract') not in (SCHEMA, SPLINE_SCHEMA):raise ValueError('PVA action contract required; a force checkpoint cannot be reinterpreted')
        self.batch_size=int(batch_size);self.engine=ResearchExecutionModel.from_mapping(model,root=root,device=device)
        self.dt=self.engine.dt_s;self.control_dt=1/30;self.stride=round(self.control_dt/self.dt)
        if self.stride<1 or not math.isclose(self.dt*self.stride,self.control_dt,abs_tol=1e-10):raise ValueError('PVA/physics clocks do not align')
        self.steps=round(settings['task']['duration_s']*30)
        if self.steps<1 or not math.isclose(self.steps/30,settings['task']['duration_s'],abs_tol=1e-9):raise ValueError('Duration must be a positive multiple of 1/30 s')
        if self.batch_size<1:raise ValueError('Positive batch required')
        self.limit=self.tensor(settings['action']['jerk_limit_m_s3'])
        if self.limit.shape!=(3,) or not bool(torch.isfinite(self.limit).all()&(self.limit>0).all()):raise ValueError('Positive finite XYZ jerk bounds required')
        self.direction=self.tensor(settings['task']['strike_direction'])
        if self.direction.shape!=(3,) or not bool(torch.isfinite(self.direction).all()) or self.direction.norm()<1e-8:raise ValueError('A nonzero strike direction is required')
        self.direction=self.direction/self.direction.norm()
        self.offset_tensor=self.tensor(self.engine.offset)
        self.delay=float(self.engine.drone.parameters.delay_s);self.queue_size=8
        if math.ceil(self.delay*30)+2>self.queue_size:raise ValueError('Fitted delay exceeds the saved PVA observation history')
        self.reset()
        self.pose_stepper=PoseStepper(self.pose,self.engine.drone.parameters,self.engine.drone.residual,graph=graph)
        self.cable_stepper=ResearchPhysics(self.engine.physics,self.state,self.dt,graph=graph,fast_solve=fast_solve,fast_geometry=fast_geometry)
        p=self.engine.drone.parameters
        self.pose_maximum=min(.005,float(p.attitude_time_constant_s)/4,
            .25/max(float(p.kd_xy)+math.sqrt(float(p.kp_xy)),float(p.kd_z)+math.sqrt(float(p.kp_z))))
        self.observation_dim=self.observation().shape[1]

    def tensor(self,value):return torch.as_tensor(value,device=self.device,dtype=torch.float64)

    @torch.no_grad()
    def reset(self,*,origin=None,target=None,randomize=False,generator=None):
        b=self.batch_size;launch=self.settings['launch']
        def positions(value):
            t=self.tensor(value)
            return t[None].expand(b,-1).clone() if t.ndim==1 else t.clone()
        self.origin0=positions(launch['origin_m'] if origin is None else origin)
        self.target=positions(launch['target_m'] if target is None else target)
        if self.origin0.shape!=(b,3) or self.target.shape!=(b,3):raise ValueError('Launch dimensions differ')
        if randomize:
            for values,radius in ((self.origin0,launch['start_radius_m']),(self.target,launch['target_radius_m'])):
                direction=torch.randn((b,3),device=self.device,dtype=torch.float64,generator=generator)
                r=torch.rand((b,1),device=self.device,dtype=torch.float64,generator=generator).pow(1/3)*radius
                values.add_(direction/direction.norm(dim=-1,keepdim=True).clamp_min(1e-12)*r)
        root=self.origin0+self.tensor(self.engine.offset)
        self.pose=settled_initial(root,torch.zeros_like(root),self.engine.offset)
        # Structural rest lengths, free pivot and no privileged measured cable state.
        lengths=self.tensor(self.engine.cable.rest_lengths_m)
        self.cable_length=lengths.sum()
        distance=torch.cat((lengths.new_zeros(1),lengths.cumsum(0)))
        self.wave_material=distance[1:-1]/distance[-1]
        q=root[:,None]+torch.stack((torch.zeros_like(distance),torch.zeros_like(distance),-distance),-1)[None]
        self.state=DderState(q,torch.zeros_like(q));self.initial_state=self.state
        self.initial_pose=self.pose
        self.command=torch.cat((self.origin0,self.origin0.new_zeros(b,8)),-1)
        self.hover=self.command.clone();self.packets=[self.command.clone()];self.actions=[]
        self.index=0;self.active=torch.ones(b,device=self.device,dtype=torch.bool)
        self.success=torch.zeros_like(self.active);self.failed=torch.zeros_like(self.active);self.contact=torch.zeros_like(self.active)
        self.tip_contact=torch.zeros_like(self.active)
        self.pull_ready=torch.zeros_like(self.active);self.reverse_ready=torch.zeros_like(self.active)
        self.pull_peak=self.origin0.new_zeros(b);self.pull_credit=self.pull_peak.clone();self.reverse_credit=self.pull_peak.clone()
        self.wave_stage=torch.zeros(b,device=self.device,dtype=torch.long)
        self.wave_dwell=self.pull_peak.clone();self.wave_credit=self.pull_peak.clone()
        self.reach_credit=self.pull_peak.clone()
        self.brake_credit=self.pull_peak.clone()
        self.wave_completion_time=self.pull_peak.new_full((b,),float('inf'))
        self.total=self.origin0.new_zeros(b);self.minimum_distance=(q[:,-1]-self.target).norm(dim=-1)
        self.initial_distance=self.minimum_distance.clone();self.best_quality=self.total.clone()
        self.encounter_time=self.total.clone();self.encounter_distance=self.minimum_distance.clone()
        self.encounter_q=q.clone();self.encounter_tip_velocity=q[:,-1].new_zeros(b,3)
        self.hit_tip_velocity=self.encounter_tip_velocity.clone()
        self.encounter_drone_velocity=self.encounter_tip_velocity.clone();self.encounter_origin=self.origin0.clone()
        self.encounter_backward=self.total.clone();self.encounter_tip_first=self.active.clone().zero_()
        self.termination_time=self.total.new_full((b,),self.steps*self.control_dt)
        self.cutoff=torch.full((b,),self.steps,device=self.device,dtype=torch.long)
        from planning.strike_objective import initialize as initialize_strike
        initialize_strike(self)
        self.frames=[]
        from learning.two_target_whip import initialize
        initialize(self)
        return self.observation()

    def observation(self):
        q,v=self.state.positions_m,self.state.velocities_m_s;p=self.pose.position
        history=self.packets[-self.queue_size:]
        history=[self.hover]*(self.queue_size-len(history))+history
        commands=torch.stack(history,1)
        commands=torch.cat((commands[:,:,:3]-p[:,None],commands[:,:,3:6]/5,commands[:,:,6:9]/10),-1).flatten(1)
        extra=torch.stack((self.minimum_distance,self.best_quality,self.contact.double(),
            self.total.new_full(self.total.shape,self.index/self.steps)),1)
        obs=torch.cat(((q-q[:,:1]).flatten(1),v.flatten(1)/5,
            p-self.origin0,self.target-p,self.pose.velocity/5,self.pose.rotation.flatten(1),
            self.pose.omega_tracking/10,commands,extra),-1).float()
        if self.settings['task'].get('require_pullback',False):
            obs=torch.cat((obs,torch.stack((self.pull_ready.double(),self.reverse_ready.double(),self.pull_peak,
                self.pull_credit,self.reverse_credit),1).float()),1)
        if self.settings.get('observation_contract') in ('pva_whip_phase_v2','pva_whip_phase_v3'):
            # Ordered reward/hit history is part of the state seen by PPO.
            phase=torch.stack((self.wave_stage/3,
                (self.wave_dwell/self.settings['task']['wave_dwell_s']).clamp(0,1),
                self.wave_credit,self.reach_credit),1).float()
            obs=torch.cat((obs,phase),1)
        if self.settings.get('observation_contract')=='pva_whip_phase_v3':
            obs=torch.cat((obs,self.brake_credit[:,None].float()),1)
        return obs

    @torch.no_grad()
    def branch_from(self,source):
        """Independent candidate rows at the SAME absolute time and command queue."""
        if source.batch_size!=1 or self.steps!=source.steps or self.dt!=source.dt:
            raise ValueError('Branches require one source state and matching episode clocks')
        def expand(value):return value.expand((self.batch_size,)+value.shape[1:]).clone()
        names=('position','velocity','rotation','omega_tracking','compensation','rotation_command_from_tracking')
        self.pose=replace(source.pose,**{name:expand(getattr(source.pose,name)) for name in names})
        self.state=DderState(expand(source.state.positions_m),expand(source.state.velocities_m_s))
        for name in ('active','success','failed','contact','tip_contact','minimum_distance','initial_distance','best_quality',
                     'termination_time','cutoff','origin0','target','total','command','hover',
                     'pull_ready','reverse_ready','pull_peak','pull_credit','reverse_credit',
                     'wave_stage','wave_dwell','wave_credit','wave_completion_time','reach_credit','brake_credit'):
            setattr(self,name,expand(getattr(source,name)))
        self.evolving=expand(getattr(source,'evolving',source.active))
        from planning.whip_objective import ENCOUNTER_FIELDS
        for name in ENCOUNTER_FIELDS:setattr(self,name,expand(getattr(source,name)))
        from planning.strike_objective import FIELDS as STRIKE_FIELDS
        for name in STRIKE_FIELDS:setattr(self,name,expand(getattr(source,name)))
        for name in self.extra_tick_fields:setattr(self,name,expand(getattr(source,name)))
        self.packets=[expand(p) for p in source.packets]
        self.actions=[expand(a) for a in source.actions]
        self.index=source.index;self.frames=[]

    def _pose_advance(self,left,right):
        events=np.arange(len(self.packets))/30+self.delay
        bounds=np.r_[left,events[(events>left+1e-12)&(events<right-1e-12)],right]
        valid=torch.ones_like(self.active)
        for a,b in zip(bounds[:-1],bounds[1:]):
            i=np.searchsorted(events,(a+b)/2+1e-12,side='right')-1
            command=self.hover if i<0 else self.packets[i]
            n=max(1,math.ceil((b-a)/self.pose_maximum-1e-10))
            for _ in range(n):
                proposal,ok=self.pose_stepper(self.pose,command,(b-a)/n)
                if self.targeted_strike:
                    from simulator.predicted_envelope import attitude_valid
                    ok = ok & attitude_valid(proposal.rotation, proposal.rotation_command_from_tracking,
                                             self.settings['limits']['maximum_tilt_deg'])
                valid&=ok
                accepted=valid&self.evolving
                self.pose=replace(self.pose,**{name:torch.where(accepted.reshape((-1,)+(1,)*(getattr(self.pose,name).ndim-1)),
                    getattr(proposal,name),getattr(self.pose,name)) for name in ('position','velocity','rotation','omega_tracking')})
        return valid

    def _tick(self,reward,left,right,clock_left,clock_cutoff,*,trace=False):
        weights=self.settings['reward'];limits=self.settings['limits'];task=self.settings['task']
        running=self.active.clone();previous=self.state
        previous_origin=self.pose.position;previous_velocity=self.pose.velocity;previous_pull_ready=self.pull_ready.clone()
        previous_pull_peak=self.pull_peak.clone()
        previous_wave_ready=self.wave_stage>=3
        pose_ok=self._pose_advance(left,right)
        root=self.pose.position+torch.einsum('bij,j->bi',self.pose.rotation,self.offset_tensor)
        candidate=self.cable_stepper(self.state,root)
        q,v=candidate.positions_m,candidate.velocities_m_s
        finite=torch.isfinite(q).flatten(1).all(-1)&torch.isfinite(v).flatten(1).all(-1)
        domain=finite&(q.abs().flatten(1).amax(-1)<20)&(v.norm(dim=-1).amax(-1)<100)&pose_ok
        workspace=(self.pose.position[:,2]>=limits['minimum_origin_z_m'])&(self.pose.position[:,2]<=limits['maximum_origin_z_m'])
        workspace&=q[:,:,2].amin(-1)>=limits['minimum_cable_z_m']
        invalid=self.evolving&(~domain|~workspace)
        # A failure before the export boundary also invalidates a sub-tick hit.
        revoke=invalid&self.success
        reward-=revoke*weights['success'];self.success&=~invalid
        reward-=invalid*weights['failure'];self.failed|=invalid;self.active&=~invalid;self.evolving&=~invalid
        running&=~invalid
        self.state=DderState(torch.where(self.evolving[:,None,None],q,previous.positions_m),
            torch.where(self.evolving[:,None,None],v,previous.velocities_m_s))
        q,v=self.state.positions_m,self.state.velocities_m_s
        if self.targeted_strike:
            from planning.strike_objective import observe
            observe(self,previous,self.state,running,clock_left)
            self.termination_time=torch.where(invalid,clock_left,self.termination_time)
            self.cutoff=torch.where(invalid,clock_cutoff,self.cutoff)
            if trace:self.frames.append(dict(time_s=right,origin=self.pose.position.clone(),rotation=self.pose.rotation.clone(),cable=q.clone(),
                origin_velocity=self.pose.velocity.clone(),cable_velocity=v.clone()))
            return reward
        delta=q[:,-1]-previous.positions_m[:,-1]
        fraction=((self.target-previous.positions_m[:,-1])*delta).sum(-1)/delta.square().sum(-1).clamp_min(1e-20)
        nearest=previous.positions_m[:,-1]+fraction.clamp(0,1)[:,None]*delta
        dist=(nearest-self.target).norm(dim=-1)
        progress=(self.minimum_distance-dist).clamp_min(0)/self.initial_distance.clamp_min(.05)
        directed=(v[:,-1]*self.direction).sum(-1)
        quality=torch.exp(-(dist/weights['proximity_scale_m']).square())*(directed/task['minimum_directed_speed_m_s']).clamp(0,1)
        pullback_allowed=torch.ones_like(running)
        if task.get('require_pullback',False):
            phase_reward,pullback_allowed=update_pullback(self,running);reward+=phase_reward
            quality*=pullback_allowed
        if task.get('require_wave',False):
            from learning.whip_wave import update
            reward+=update(self,running,clock_left+self.dt)
            quality*=previous_wave_ready
        quality_weight=weights['strike_quality']
        if weights.get('joint_strike',0.):
            from learning.ppo_strike import joint_quality
            backward=self.pull_peak-((self.pose.position-self.origin0)*self.direction).sum(-1)
            quality=joint_quality(dist,v[:,-1],self.pose.velocity,backward,self.pull_ready,
                self.wave_credit,self.direction,task,weights['proximity_scale_m'])
            quality_weight=weights['joint_strike']
        reward+=running*(weights['progress']*progress+quality_weight*(quality-self.best_quality).clamp_min(0))
        self.minimum_distance=torch.where(running,torch.minimum(self.minimum_distance,dist),self.minimum_distance)
        self.best_quality=torch.where(running,torch.maximum(self.best_quality,quality),self.best_quality)
        entry=sphere_entry(previous.positions_m,q,self.target,task['target_radius_m'])
        if self.settings.get('trajectory_objective',{}).get('schema')=='preferred_fold_v1':
            from planning.whip_objective import record_encounter
            record_encounter(self,previous,previous_origin,previous_velocity,q,v,entry,fraction,running,clock_left,previous_pull_peak)
        tip=entry[:,-1];other=entry[:,:-1].amin(-1)
        entered=running&torch.isfinite(tip);first_allowed=~self.contact if task['first_contact_only'] else torch.ones_like(running)
        self.tip_contact|=entered
        if task.get('require_pullback',False):
            fraction=tip.clamp(0,1)[:,None]
            contact_origin=previous_origin+fraction*(self.pose.position-previous_origin)
            contact_drone_velocity=previous_velocity+fraction*(self.pose.velocity-previous_velocity)
            contact_tip_velocity=previous.velocities_m_s[:,-1]+fraction*(v[:,-1]-previous.velocities_m_s[:,-1])
            backward=self.pull_peak-((contact_origin-self.origin0)*self.direction).sum(-1)
            pullback_allowed=previous_pull_ready&(backward>=task['minimum_backward_distance_m'])&(
                (contact_drone_velocity*self.direction).sum(-1)<=-task['minimum_backward_speed_m_s'])
            directed=(contact_tip_velocity*self.direction).sum(-1)
            contact_speed=contact_tip_velocity.norm(dim=-1)
        else:contact_speed=v[:,-1].norm(dim=-1)
        angle_ok=torch.ones_like(entered) if task.get('angle_mode')=='soft_reward' else directed/contact_speed.clamp_min(1e-12)>=math.cos(math.radians(task['maximum_angle_deg']))
        hit=entered&(tip<other)&first_allowed&(directed>=task['minimum_directed_speed_m_s'])&angle_ok&pullback_allowed
        if task.get('require_wave',False):
            # Conservative causal check: all stages must finish BEFORE the
            # contact interval; interval-end evidence cannot qualify an early hit.
            hit&=previous_wave_ready
        hit=contact_success(task,running,tip,hit)
        if criterion(task)==TWO_TARGET:
            from learning.two_target_whip import advance
            hit,tip=advance(self,previous,q,v,running,clock_left)
        from learning.pva_success import record_hit_velocity
        self.hit_tip_velocity=record_hit_velocity(previous.velocities_m_s[:,-1],v[:,-1],tip,hit,self.hit_tip_velocity)
        any_contact=running&torch.isfinite(entry).any(-1)
        invalid_contact=any_contact&~hit&~self.contact
        if criterion(task) in (TIP_CONTACT,TWO_TARGET):invalid_contact=torch.zeros_like(hit)
        self.contact|=any_contact;self.success|=hit;self.active&=~hit
        terminal_contact=invalid_contact&task['first_contact_only'] if (
            self.settings['method']=='ppo' and self.settings['training'].get('terminate_invalid_contact',False)) else torch.zeros_like(hit)
        # A rejected first contact makes success impossible under this task.
        # Stop PPO credit collection here instead of rewarding a later release.
        self.active&=~terminal_contact;self.evolving&=~terminal_contact
        reward+=hit*weights['success']-invalid_contact*weights['invalid_contact']
        duration=torch.where(hit,tip.clamp(0,1)*self.dt,self.total.new_full(self.total.shape,self.dt))
        duration=torch.where(terminal_contact,entry.amin(-1).clamp(0,1)*self.dt,duration)
        if any(weights.get(k,0.) for k in ('vertical_excursion','vertical_tip_velocity','vertical_tip_alignment','horizontal_contact')):
            from learning.horizontal_strike import cost_rate,contact_bonus
            reward-=running*duration*cost_rate(q,v,self.pose.position,self.origin0,self.target,self.pull_ready,weights)
            # Soft contact quality is evaluated at the same interpolated entry
            # as hit detection. Infinite no-contact fractions are safely clamped.
            blend=tip.clamp(0,1)[:,None,None]
            contact_q=previous.positions_m+blend*(q-previous.positions_m)
            contact_v=previous.velocities_m_s+blend*(v-previous.velocities_m_s)
            reward+=hit*contact_bonus(contact_q,contact_v,weights)
        if any(weights.get(k,0.) for k in ('drone_approach','lateral_excursion','near_target_reach','outward_contact')):
            from learning.horizontal_strike import reach_cost_rate,reach_contact_bonus
            reward-=running*duration*reach_cost_rate(q,self.pose.position,self.origin0,self.target,
                self.pull_ready,self.direction,self.cable_length,weights)
            blend=tip.clamp(0,1)[:,None,None]
            contact_q=previous.positions_m+blend*(q-previous.positions_m)
            reward+=hit*reach_contact_bonus(contact_q,self.direction,self.cable_length,weights)
        if weights.get('reach_progress',0.):
            from learning.horizontal_strike import outward_reach
            # One bounded credit account: oscillating the cable cannot collect
            # the same reach reward repeatedly. Only forward-tip release counts.
            releasing=running&self.pull_ready&((self.pose.velocity*self.direction).sum(-1)<0)&((v[:,-1]*self.direction).sum(-1)>0)
            reach=outward_reach(q,self.direction,self.cable_length)
            credit=torch.where(releasing,torch.maximum(self.reach_credit,reach),self.reach_credit)
            reward+=weights['reach_progress']*(credit-self.reach_credit)
            self.reach_credit=credit
        reward-=running*(weights['time_per_s']*duration+weights['displacement']*(self.pose.position-self.origin0).square().sum(-1)*duration)
        ended=hit|invalid|terminal_contact
        terminal_fraction=torch.where(terminal_contact,entry.amin(-1),tip).clamp(0,1)
        self.termination_time=torch.where(ended,clock_left+torch.where(hit|terminal_contact,terminal_fraction*self.dt,torch.zeros_like(tip)),self.termination_time)
        self.cutoff=torch.where(ended,clock_cutoff,self.cutoff)
        if trace:self.frames.append(dict(time_s=right,origin=self.pose.position.clone(),rotation=self.pose.rotation.clone(),cable=q.clone(),
            origin_velocity=self.pose.velocity.clone(),cable_velocity=v.clone()))
        return reward

    @torch.no_grad()
    def step(self,action,*,trace=False,next_packet=None):
        if self.index>=self.steps:raise ValueError('Reset a completed PVA rollout before stepping')
        action=action.to(device=self.device,dtype=torch.float64)
        if action.shape!=(self.batch_size,3) or not bool(torch.isfinite(action).all()) or (next_packet is None and bool((action.abs()>1+1e-6).any())):raise ValueError('Finite normalized bounded XYZ jerk action required')
        mask=self.active.clone();self.evolving=mask.clone();weights=self.settings['reward'];limits=self.settings['limits'];task=self.settings['task']
        if weights.get('early_hit',0.):
            from learning.pva_success import hit_time_bonus
            previous_hit_bonus=hit_time_bonus(weights,self.success,self.failed,self.termination_time)
        if weights.get('impact',0.):
            from learning.pva_success import impact_bonus
            previous_impact_bonus=impact_bonus(weights,self.success,self.failed,self.hit_tip_velocity,self.direction)
        reward=self.total.new_zeros(self.batch_size)
        from planning.position_spline import SCHEMA as SPLINE_SCHEMA
        spline=self.settings['command_contract']==SPLINE_SCHEMA
        if spline != (next_packet is not None):raise ValueError('Reference packets must match the saved spline/jerk contract')
        next_command=next_packet if spline else integrate_jerk(self.command,action*self.limit,self.control_dt)
        if weights.get('command_speed',0.):
            # Soft cost with no added speed cutoff or action clipping.
            speed_ratio=next_command[:,3:6].norm(dim=-1)/limits['maximum_speed_m_s']
            reward-=mask*weights['command_speed']*speed_ratio.pow(4)*self.control_dt
        if weights.get('command_acceleration',0.):
            specific=next_command[:,6:9]+next_command.new_tensor([0.,0.,9.80665])
            utilization=specific.norm(dim=-1)/limits['maximum_specific_force_m_s2']
            reward-=mask*weights['command_acceleration']*utilization.pow(8)*self.control_dt
        packet_ok,_=reference_packet_validity(next_command[:,None],limits);packet_ok=packet_ok[:,0]
        packet_ok&=(next_command[:,2]>=limits['minimum_origin_z_m'])&(next_command[:,2]<=limits['maximum_origin_z_m'])
        # Reject infeasible next knots explicitly; no clipping to inconsistent P/V/A.
        bad=mask&~packet_ok;self.failed|=bad;self.active&=~bad;self.evolving&=~bad
        reward-=bad*weights['failure'];self.termination_time=torch.where(bad,self.total.new_full(self.total.shape,self.index/30),self.termination_time)
        self.cutoff=torch.where(bad,torch.full_like(self.cutoff,self.index),self.cutoff)
        for tick in range(self.stride):
            left=(self.index*self.stride+tick)*self.dt;right=left+self.dt
            if self.fused_ticks and not trace:
                from learning.pva_tick_graph import run_tick
                reward=run_tick(self,reward,left,right)
            else:
                reward=self._tick(reward,left,right,self.total.new_full(self.total.shape,left),
                    torch.full_like(self.cutoff,self.index+1),trace=trace)
        reward-=mask*weights['jerk']*action.square().mean(-1)*self.control_dt
        self.actions.append(action.clone());self.command=torch.where(self.evolving[:,None],next_command,self.command)
        self.packets.append(self.command.clone());self.index+=1
        if self.index>=self.steps:self.active.zero_()
        if weights.get('early_hit',0.):
            # Difference of terminal credit: count once and revoke if invalidated.
            reward+=hit_time_bonus(weights,self.success,self.failed,self.termination_time)-previous_hit_bonus
        if weights.get('impact',0.):
            reward+=impact_bonus(weights,self.success,self.failed,self.hit_tip_velocity,self.direction)-previous_impact_bonus
        self.total+=reward
        return self.observation(),reward.float()[:,None],(~self.active).float()[:,None],mask.float()[:,None]

    @torch.no_grad()
    def rollout(self,policy=None,actions=None,*,trace=False,max_steps=None,observer=None,packets=None):
        if (policy is None)==(actions is None):raise ValueError('Supply one policy or action sequence')
        count=self.steps-self.index if max_steps is None else min(int(max_steps),self.steps-self.index)
        if count<1:raise ValueError('Rollout needs at least one remaining interval')
        if packets is not None:
            if policy is not None or packets.shape!=(self.batch_size,count+1,11) or not bool(torch.isfinite(packets).all()):
                raise ValueError('Finite complete spline PVA packets required')
            if not torch.allclose(packets[:,0],self.command,atol=1e-9,rtol=0):
                raise ValueError('Spline initial PVA must equal the rollout initial command')
        from learning.ppo_trajectory_reward import enabled,TrajectoryReward
        objective=TrajectoryReward(self) if enabled(self.settings) else None
        from learning.pva_ppo_rollout import decision_steps
        repeat=decision_steps(self.settings) if policy is not None else 1
        if policy is not None and self.index%repeat:
            raise ValueError('Policy rollout must start at a decision boundary')
        if observer is not None:observer(self)
        for i in range(count):
            if policy is not None and i%repeat==0:held_action=policy(self.observation())
            self.step(held_action if policy is not None else actions[:,i],trace=trace,
                      next_packet=None if packets is None else packets[:,i+1])
            if objective is not None:objective.observe(self)
            if observer is not None:observer(self)
            if not bool(self.active.any()):break
        result=self.result()
        return objective.finish(self,result) if objective is not None else result

    def result(self):
        extra={}
        if self.targeted_strike:
            extra.update(closest_approach_time_s=self.encounter_time.clone(),
                closest_tip_velocity_m_s=self.encounter_tip_velocity.clone(),
                fold_valid=(self.strike_valid & self.strike_fold_valid & ~self.failed).clone(),
                strike_time_s=self.strike_time.clone(), strike_distance_m=self.strike_distance.clone(),
                strike_tip_velocity_m_s=self.strike_velocity.clone(),
                strike_root_velocity_m_s=self.strike_root_velocity.clone(),
                fold_completed_time_s=self.fold_completed_time_s.clone())
        if self.extra_tick_fields:
            extra=dict(target_hits=self.target_hits.clone()&~self.failed[:,None],target_hit_times_s=self.target_hit_times.clone(),
                target_minimum_distances_m=self.target_minimum_distances.clone(),target_hit_velocities_m_s=self.target_hit_velocities.clone())
        return dict(**extra,reward=self.total.clone(),success=self.success.clone(),failed=self.failed.clone(),
            minimum_tip_distance_m=self.minimum_distance.clone(),duration_s=self.termination_time.clone(),
            cutoffs=self.cutoff.clone(),packets=torch.stack(self.packets,1),actions=torch.stack(self.actions,1))
