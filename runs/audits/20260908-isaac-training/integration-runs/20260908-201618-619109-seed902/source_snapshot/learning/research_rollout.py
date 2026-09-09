"""Accepted pose + cable model, shared frozen-reference execution and scoring."""
from pathlib import Path
import numpy as np
import torch
from simulator.cable import CableConfiguration,DderModel
from simulator.cable.residual import FrozenMotionResidual
from simulator.research_pose import ResearchPoseModel,settled_initial
from simulator.research_physics import ResearchPhysics
from simulator.research_reference import reference_packets,reference_feasibility
from .point_force_env import PointForceWhipEnvironment,_replace_state_rows
from .fullstate_rollout import frozen_reference,checkpoint_path


def invalidate_attempts(score,invalid):
    """A later failure invalidates an earlier hit and its success rewards."""
    for key in ['success','forward_return','release','terminal_displacement']:
        old=score.episode_component_sums[key]
        removed=torch.where(invalid,old,torch.zeros_like(old))
        score.episode_reward-=removed;score.episode_component_sums[key]=old-removed
    old=score.episode_component_sums['numerical_failure']
    new=torch.where(invalid,torch.minimum(old,old.new_full(old.shape,-score.reward_weights.numerical_failure)),old)
    score.episode_reward+=new-old;score.episode_component_sums['numerical_failure']=new
    score.episode_success&=~invalid;score.failed|=invalid
    score.episode_hit_time_s=torch.where(invalid,torch.nan,score.episode_hit_time_s)


def charge_planned_duration(score, duration_s):
    """Charge the frozen command duration, even after early contact/failure.

    Execution is open loop: contact does not shorten its preplanned commands.
    Replace the accumulated component so there is no double charge, including
    on invalid contacts where the first-contact accumulator stopped early.
    Non-deployed rows are subsequently replaced by nominal planning scores.
    """
    old = score.episode_component_sums['time']
    new = -score.reward_weights.time_per_s * duration_s.to(old)
    score.episode_reward += new - old
    score.episode_component_sums['time'] = new


@torch.no_grad()
def execute_research_batch(nominal,batch,forces,cutoffs,settings,*,trace=None,progress=None):
    if settings.get('termination')=='execution_success_or_timeout':
        from .research_terminal import execute_terminal_batch
        return execute_terminal_batch(nominal,batch,forces,cutoffs,settings,trace=trace,progress=progress)
    model=nominal.model_config;execution=model['fullstate_execution'];dt=nominal.physics_dt_s
    score=PointForceWhipEnvironment(model,nominal.task_config,nominal.ppo_config,batch_size=nominal.batch_size,device=nominal.device)
    score.target=(batch.target_position_m if batch.target_position_m is not None else nominal.target).clone();score.reset(batch.truth)
    score.set_success_condition(target_radius_m=nominal.target_radius_m,minimum_directed_speed_m_s=nominal.minimum_directed_speed_m_s,
        maximum_tip_velocity_direction_error_deg=nominal.maximum_tip_velocity_direction_error_deg)
    score.physics_steps_per_control=1;count=int(cutoffs.max());score.control_step_count=max(1,count+1)
    deployed=cutoffs>0;failed=nominal.failed.clone();state=batch.truth;maximum=score.episode_reward.new_zeros(len(cutoffs))
    feasible=torch.ones_like(deployed);pose_failed=torch.zeros_like(deployed)
    reference_info={key:score.episode_reward.new_zeros(len(cutoffs)) for key in ['maximum_reference_tilt_deg',
        'maximum_reference_specific_force_m_s2','minimum_reference_specific_vertical_m_s2','maximum_reference_speed_m_s','reference_position_correction_m']}
    reference=None;pose=None
    if count:
        positions,velocities=frozen_reference(nominal,batch,forces,cutoffs)
        offset=model['recorded_data']['optitrack_to_attachment_offset_body_m']
        packets,reference=reference_packets(positions,velocities,dt,cutoffs,offset)
        feasible,reference_info=reference_feasibility(packets,cutoffs,dt,execution['feasibility'])
        reference_info['reference_position_correction_m']=reference['integration_position_correction_m']
        if not hasattr(nominal,'_research_tracker'):
            nominal._research_tracker=ResearchPoseModel(checkpoint_path(execution['checkpoint']),execution['sha256'],nominal.device)
        initial=settled_initial(batch.truth.positions_m[:,0],batch.truth.velocities_m_s[:,0],offset)
        # Failed references are not sent to the response predictor. Their rows
        # remain explicitly invalid for scoring, rather than being accepted hover.
        hover=packets[:,0].clone();hover[:,3:9]=0.
        safe_packets=torch.where(feasible[:,None,None],packets,hover[:,None])
        pose=nominal._research_tracker.predict(initial,safe_packets,np.arange(packets.shape[1])/30.,
            np.arange(count+1)*dt,offset,graph=bool(nominal.ppo_config.get('cuda_graph_physics',False)),hover_command=hover)
        within=torch.arange(count+1,device=nominal.device)[None]<=cutoffs[:,None]
        pose_failed=deployed&((~pose['valid'])&within).any(1);failed|=deployed&(~feasible|pose_failed)
        if not hasattr(nominal,'_research_cable'):
            cable=CableConfiguration.from_mapping(model['cable']);physics=DderModel(cable.dder_parameters(EI=model['cable']['EI_n_m2'],Cb=model['cable']['Cb_n_m2_s']))
            residual=model['motion_residual'];physics.motion_residual=FrozenMotionResidual(checkpoint_path(residual['checkpoint']),residual['sha256'])
            nominal._research_cable=ResearchPhysics(physics,state,dt,graph=bool(nominal.ppo_config.get('cuda_graph_physics',False)))
        dummy=state.positions_m.new_zeros(len(cutoffs),3)
        for i in range(count):
            from .training_control import check_training_stop
            check_training_stop();striking=deployed&(i<cutoffs);previous=state
            candidate=nominal._research_cable(previous,pose['position_attachment_m'][:,i+1])
            finite=torch.isfinite(candidate.positions_m).flatten(1).all(-1)&torch.isfinite(candidate.velocities_m_s).flatten(1).all(-1)
            pos_limit=candidate.positions_m.abs().flatten(1).amax(-1)>nominal.numerical_position_limit_m
            speed_limit=candidate.velocities_m_s.norm(dim=-1).amax(-1)>nominal.numerical_speed_limit_m_s
            failed|=striking&(~finite|pos_limit|speed_limit)
            score.episode_nonfinite|=striking&~finite;score.episode_position_limit|=striking&finite&pos_limit;score.episode_speed_limit|=striking&finite&speed_limit
            state=_replace_state_rows(candidate,previous,striking&~failed)
            score.active&=striking&feasible&~pose_failed
            transition=score.model._result(previous,candidate,forces[i],dt)
            score.model.step_runtime=lambda _state,_force,_dt:transition
            score.step(dummy)
            session=getattr(nominal,'_isaac_session',None)
            if session is not None:
                session.execution_step(score,state,pose,i+1,cutoffs,failed)
            maximum=torch.maximum(maximum,(state.positions_m[:,0]-batch.truth.positions_m[:,0]).norm(dim=-1)*deployed)
            if trace is not None:trace(i,forces[i].clone(),state,striking.clone(),score.episode_success.clone()&~failed)
            if progress is not None:progress('Predicting tracked drone pose and cable',i+1,count)
        score.reference_packets=packets;score.reference_trajectory=reference;score.predicted_pose=pose
    charge_planned_duration(score, cutoffs*dt)
    invalidate_attempts(score,failed)
    score.episode_timed_out=deployed&~score.episode_success&~score.failed
    score.episode_reward=torch.where(deployed,score.episode_reward,nominal.episode_reward)
    for key in score.episode_component_sums:
        score.episode_component_sums[key]=torch.where(deployed,score.episode_component_sums[key],nominal.episode_component_sums[key])
    score.episode_component_sums['recovery']=torch.zeros_like(score.episode_reward)
    score.deployment=dict(planned=deployed,predicted_valid_hit=nominal.episode_success.clone(),
        recovered=torch.zeros_like(deployed),joint_success=torch.zeros_like(deployed),recovery_evaluated=torch.zeros_like(deployed),
        duration_s=cutoffs*dt,recovery_position_error_m=torch.zeros_like(score.episode_reward),
        maximum_execution_drone_displacement_m=maximum,nominal=batch.nominal,
        planning_minimum_tip_distance_m=nominal.episode_minimum_tip_distance.clone(),
        planning_maximum_drone_displacement_m=nominal.episode_maximum_point_displacement.clone(),
        planning_failure=nominal.failed.clone(),planning_non_tip_first=nominal.episode_non_tip_first.clone(),
        planning_invalid_tip_entry=nominal.episode_invalid_tip_entry.clone(),
        planning_tip_entry_speed=nominal.episode_first_tip_directed_speed.clone(),planning_tip_entry_angle=nominal.episode_first_tip_angle.clone(),
        reference_infeasible=deployed&~feasible,pose_domain_failure=pose_failed,
        **{f'target_{axis}_m':score.target[:,j].clone() for j,axis in enumerate('xyz')},
        **{f'initial_attachment_{axis}_m':batch.truth.positions_m[:,0,j].clone() for j,axis in enumerate('xyz')},**reference_info)
    score.execution_state=state
    return score
