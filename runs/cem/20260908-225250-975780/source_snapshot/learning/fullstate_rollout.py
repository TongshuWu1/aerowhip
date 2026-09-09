"""Frozen force plan -> 30 Hz reference -> empirical execution -> cable reward.

The policy never observes predicted execution feedback. Recovery is excluded
from this mode's objective; export still supplies the gentle return separately.
"""
from dataclasses import replace
from pathlib import Path
import torch
from simulator.cable import CableConfiguration,DderModel,DderState,START_PINNED_FREE_END
from simulator.cable.residual import FrozenMotionResidual
from simulator.fullstate_execution import FullStateAttachmentModel,sample_kinematic_reference,CudaGraphCableBoundary
from .point_force_env import PointForceWhipEnvironment,_replace_state_rows


def checkpoint_path(value):
    path=Path(value)
    return path if path.is_absolute() else Path(__file__).resolve().parents[1]/path


@torch.no_grad()
def frozen_reference(nominal,batch,forces,cutoffs):
    """Replay virtual forces from the estimated initial state, before execution."""
    state=batch.estimate
    positions=[state.positions_m[:,0].clone()];velocities=[state.velocities_m_s[:,0].clone()]
    steps=int(cutoffs.max())
    if nominal.ppo_config['deployment'].get('termination')=='execution_success_or_timeout':
        # These are the exact virtual states already used to generate actions.
        # Failed rows were frozen by the planner; their failure time is carried
        # separately so an unexecuted future failure cannot erase an earlier hit.
        p=torch.stack(nominal._virtual_reference_positions,1)
        v=torch.stack(nominal._virtual_reference_velocities,1)
        if p.shape[1]<steps+1:
            extra=steps+1-p.shape[1]
            p=torch.cat((p,p[:,-1:].expand(-1,extra,-1)),1)
            v=torch.cat((v,v[:,-1:].expand(-1,extra,-1)),1)
        return p[:,:steps+1],v[:,:steps+1]
    for i in range(steps):
        from .training_control import check_training_stop
        check_training_stop()
        candidate=nominal.model.step_runtime(state,forces[i],nominal.physics_dt_s).state
        state=_replace_state_rows(candidate,state,i<cutoffs)
        positions.append(state.positions_m[:,0].clone());velocities.append(state.velocities_m_s[:,0].clone())
    return torch.stack(positions,1),torch.stack(velocities,1)


@torch.no_grad()
def execute_fullstate_batch(nominal,batch,forces,cutoffs,settings,*,trace=None,progress=None):
    execution=nominal.model_config['fullstate_execution']
    if execution.get('schema')=='tracked_pose_execution_v1':
        from .research_rollout import execute_research_batch
        return execute_research_batch(nominal,batch,forces,cutoffs,settings,trace=trace,progress=progress)
    if execution.get('schema')!='effective_attachment_execution_v1':raise ValueError('Unknown full-state execution model')
    if execution.get('recovery_objective')!='excluded':raise ValueError('Recovery is not identified by this strike model')
    score=PointForceWhipEnvironment(nominal.model_config,nominal.task_config,nominal.ppo_config,
        batch_size=nominal.batch_size,device=nominal.device)
    score.target=(batch.target_position_m if batch.target_position_m is not None else nominal.target).clone()
    score.reset(batch.truth)
    score.set_success_condition(target_radius_m=nominal.target_radius_m,
        minimum_directed_speed_m_s=nominal.minimum_directed_speed_m_s,
        maximum_tip_velocity_direction_error_deg=nominal.maximum_tip_velocity_direction_error_deg)
    score.physics_steps_per_control=1
    count=int(cutoffs.max());score.control_step_count=max(count+1,1)
    dt=nominal.physics_dt_s;deployed=cutoffs>0
    state=batch.truth;failed=nominal.failed.clone()
    maximum_displacement=state.positions_m.new_zeros(nominal.batch_size)
    if count:
        p,v=frozen_reference(nominal,batch,forces,cutoffs)
        tracker=FullStateAttachmentModel(checkpoint_path(execution['checkpoint']),execution['sha256'],device=nominal.device)
        calibrated=nominal.model_config['recorded_data']['optitrack_to_attachment_offset_body_m']
        if any(abs(a-b)>1e-9 for a,b in zip(tracker.offset_body_m,calibrated)):
            raise ValueError('Drone response and cable calibration use different attachment geometry')
        times=torch.arange(count,device=p.device,dtype=p.dtype)*dt
        # Exact 30 Hz zero-order-held packets after the effective fitted delay.
        def commands(extra_delay):
            query=times[None].expand(len(p),-1)-tracker.delay_s-extra_delay
            sample=(query*30.+1e-10).floor()/30.
            sample=sample.clamp_min(0)
            sample=torch.minimum(sample,cutoffs[:,None].to(p.dtype)*dt)
            result=sample_kinematic_reference(p,v,dt,sample)
            hover=torch.cat((p[:,0],torch.zeros_like(p[:,0]),torch.zeros_like(p[:,0])),dim=-1)
            return torch.where((query<0)[...,None],hover[:,None],result)
        command,past=commands(0.),commands(tracker.history_s)
        predicted_root=tracker.predict(batch.truth.positions_m[:,0],batch.truth.velocities_m_s[:,0],
            command,past,p.new_full((len(p),count),dt))[0]
        cable=CableConfiguration.from_mapping(nominal.model_config['cable'])
        physics=DderModel(cable.dder_parameters(EI=nominal.model_config['cable']['EI_n_m2'],Cb=nominal.model_config['cable']['Cb_n_m2_s']))
        residual=nominal.model_config.get('motion_residual',{})
        if residual.get('enabled'):
            physics.motion_residual=FrozenMotionResidual(checkpoint_path(residual['checkpoint']),residual['sha256'])
        constants=physics.runtime_constants(state.positions_m)
        constants=replace(constants,bending_stiffness_n_m2=constants.bending_stiffness_n_m2*batch.stiffness_scale,
            bending_damping_n_m2_s=constants.bending_damping_n_m2_s*batch.damping_scale)
        graph=CudaGraphCableBoundary(physics,state,dt,constants) if state.positions_m.is_cuda else None
        dummy=state.positions_m.new_zeros(len(p),3)
        for i in range(count):
            from .training_control import check_training_stop
            check_training_stop()
            striking=deployed&(i<cutoffs);previous=state
            candidate=graph(previous,predicted_root[:,i+1]) if graph else physics.step_runtime(previous,
                predicted_root[:,i+1,None],p.new_full((len(p),),dt),constants,pinned_endpoints=START_PINNED_FREE_END,
                iterative_damping=False,dense_constraint_solve=True,analytic_bending=True,create_graph=False)
            finite=torch.isfinite(candidate.positions_m).flatten(1).all(-1)&torch.isfinite(candidate.velocities_m_s).flatten(1).all(-1)
            position_limit=candidate.positions_m.abs().flatten(1).amax(-1)>nominal.numerical_position_limit_m
            speed_limit=candidate.velocities_m_s.norm(dim=-1).amax(-1)>nominal.numerical_speed_limit_m_s
            failed|=striking&(~finite|position_limit|speed_limit)
            score.episode_nonfinite|=striking&~finite
            score.episode_position_limit|=striking&finite&position_limit
            score.episode_speed_limit|=striking&finite&speed_limit
            state=_replace_state_rows(candidate,previous,striking&~failed)
            score.active&=striking
            transition=score.model._result(previous,candidate,forces[i],dt)
            score.model.step_runtime=lambda _state,_force,_dt:transition
            score.step(dummy)
            maximum_displacement=torch.maximum(maximum_displacement,(state.positions_m[:,0]-batch.truth.positions_m[:,0]).norm(dim=-1)*deployed)
            if trace is not None:trace(i,forces[i].clone(),state,striking.clone(),score.episode_success.clone())
            if progress is not None:progress('Predicting full-state execution and cable strike',i+1,count)
    scored_seconds=torch.where(score.episode_success,score.episode_hit_time_s,cutoffs*dt)
    remaining=score.reward_weights.time_per_s*(cutoffs*dt-scored_seconds)
    score.episode_reward-=remaining;score.episode_component_sums['time']-=remaining
    score.failed|=failed
    score.episode_timed_out=deployed&~score.episode_success&~score.failed
    score.episode_reward=torch.where(deployed,score.episode_reward,nominal.episode_reward)
    for name in score.episode_component_sums:
        score.episode_component_sums[name]=torch.where(deployed,score.episode_component_sums[name],nominal.episode_component_sums[name])
    score.episode_component_sums['recovery']=torch.zeros_like(score.episode_reward)
    score.deployment=dict(planned=deployed,predicted_valid_hit=nominal.episode_success.clone(),
        recovered=torch.zeros_like(deployed),joint_success=torch.zeros_like(deployed),recovery_evaluated=torch.zeros_like(deployed),
        duration_s=cutoffs*dt,recovery_position_error_m=torch.zeros_like(score.episode_reward),
        maximum_execution_drone_displacement_m=maximum_displacement,nominal=batch.nominal,
        planning_minimum_tip_distance_m=nominal.episode_minimum_tip_distance.clone(),
        planning_maximum_drone_displacement_m=nominal.episode_maximum_point_displacement.clone(),
        planning_failure=nominal.failed.clone(),planning_non_tip_first=nominal.episode_non_tip_first.clone(),
        planning_invalid_tip_entry=nominal.episode_invalid_tip_entry.clone(),
        planning_tip_entry_speed=nominal.episode_first_tip_directed_speed.clone(),
        planning_tip_entry_angle=nominal.episode_first_tip_angle.clone(),
        **{f'target_{axis}_m':score.target[:,j].clone() for j,axis in enumerate('xyz')},
        **{f'initial_attachment_{axis}_m':batch.truth.positions_m[:,0,j].clone() for j,axis in enumerate('xyz')})
    score.execution_state=state
    return score
