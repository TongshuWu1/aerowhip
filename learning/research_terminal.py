"""Native execution episodes ending at first valid hit or the time limit.

Actions are generated offline without measured execution feedback. The fitted
execution, rather than the virtual cable, decides the episode's terminal time.
Unused tails can be computed in a batch but never affect its return or outcome.
"""
import numpy as np
import torch
from simulator.cable import CableConfiguration,DderModel
from simulator.cable.residual import FrozenMotionResidual
from simulator.research_pose import ResearchPoseModel,settled_initial
from simulator.research_physics import ResearchPhysics
from simulator.research_reference import reference_packets,reference_feasibility,reference_packet_validity
from .point_force_env import PointForceWhipEnvironment,_replace_state_rows
from .fullstate_rollout import frozen_reference,checkpoint_path
from .research_rollout import invalidate_attempts


@torch.no_grad()
def execute_terminal_batch(nominal,batch,forces,cutoffs,settings,*,trace=None,progress=None):
    model=nominal.model_config;execution=model['fullstate_execution'];dt=nominal.physics_dt_s
    score=PointForceWhipEnvironment(model,nominal.task_config,nominal.ppo_config,
        batch_size=nominal.batch_size,device=nominal.device)
    score.target=(batch.target_position_m if batch.target_position_m is not None else nominal.target).clone()
    score.reset(batch.truth)
    score.set_success_condition(target_radius_m=nominal.target_radius_m,
        minimum_directed_speed_m_s=nominal.minimum_directed_speed_m_s,
        maximum_tip_velocity_direction_error_deg=nominal.maximum_tip_velocity_direction_error_deg)
    score.terminate_on_invalid_contact=False
    score.physics_steps_per_control=1;count=int(cutoffs.max());score.control_step_count=max(1,count+1)
    state=batch.truth;deployed=cutoffs>0;score.active&=deployed
    failed=~deployed;terminal=cutoffs.clone();maximum=score.episode_reward.new_zeros(len(cutoffs))
    reference_failed=torch.zeros_like(failed);pose_failed=torch.zeros_like(failed);planning_failed=torch.zeros_like(failed)
    positions,velocities=frozen_reference(nominal,batch,forces,cutoffs)
    offset=model['recorded_data']['optitrack_to_attachment_offset_body_m']
    packets,reference=reference_packets(positions,velocities,dt,cutoffs,offset)
    packet_valid,_=reference_packet_validity(packets,execution['feasibility'])
    if not hasattr(nominal,'_research_tracker'):
        nominal._research_tracker=ResearchPoseModel(checkpoint_path(execution['checkpoint']),execution['sha256'],nominal.device)
    initial=settled_initial(batch.truth.positions_m[:,0],batch.truth.velocities_m_s[:,0],offset)
    hover=packets[:,0].clone();hover[:,:3]=initial.position;hover[:,3:]=0.
    # Sanitize only invalid packets to keep speculative predictor math finite.
    # Each invalid command still causes failure if its time is reached before a hit.
    safe_packets=torch.where(packet_valid[:,:,None],packets,hover[:,None])
    if progress is not None:progress('Computing fitted drone response (scene held)',0,count)
    pose=nominal._research_tracker.predict(initial,safe_packets,np.arange(packets.shape[1])/30.,
        np.arange(count+1)*dt,offset,graph=bool(nominal.ppo_config.get('cuda_graph_physics',False)),hover_command=hover)
    if not hasattr(nominal,'_research_cable'):
        cable=CableConfiguration.from_mapping(model['cable'])
        physics=DderModel(cable.dder_parameters(EI=model['cable']['EI_n_m2'],Cb=model['cable']['Cb_n_m2_s']))
        residual=model.get('motion_residual',{})
        if residual.get('enabled'):
            physics.motion_residual=FrozenMotionResidual(checkpoint_path(residual['checkpoint']),residual['sha256'])
        nominal._research_cable=ResearchPhysics(physics,state,dt,graph=bool(nominal.ppo_config.get('cuda_graph_physics',False)))
    planning_failure_steps=getattr(nominal,'_planning_failure_steps',torch.full_like(cutoffs,count+1))
    dummy=state.positions_m.new_zeros(len(cutoffs),3)
    for i in range(count):
        from .training_control import check_training_stop
        check_training_stop()
        running=score.active & (i<cutoffs);previous=state
        reference_bad=running & ~packet_valid[:,i//nominal.physics_steps_per_control]
        pose_bad=running & ~pose['valid'][:,i+1]
        planning_bad=running & ((i+1)>=planning_failure_steps)
        reference_failed|=reference_bad;pose_failed|=pose_bad;planning_failed|=planning_bad
        candidate=nominal._research_cable(previous,pose['position_attachment_m'][:,i+1])
        finite=torch.isfinite(candidate.positions_m).flatten(1).all(-1)&torch.isfinite(candidate.velocities_m_s).flatten(1).all(-1)
        pos_limit=candidate.positions_m.abs().flatten(1).amax(-1)>nominal.numerical_position_limit_m
        speed_limit=candidate.velocities_m_s.norm(dim=-1).amax(-1)>nominal.numerical_speed_limit_m_s
        bad=running & (~finite|pos_limit|speed_limit|reference_bad|pose_bad|planning_bad)
        failed|=bad
        score.episode_nonfinite|=running & ~finite
        score.episode_position_limit|=running & finite & pos_limit
        score.episode_speed_limit|=running & finite & speed_limit
        state=_replace_state_rows(candidate,previous,running & ~bad)
        score.active&=running & ~bad
        transition=score.model._result(previous,state,forces[i],dt)
        score.model.step_runtime=lambda *_:transition
        score.step(dummy)
        maximum=torch.maximum(maximum,(state.positions_m[:,0]-batch.truth.positions_m[:,0]).norm(dim=-1)*deployed)
        done=running & (score.episode_success|bad|((i+1)>=cutoffs))
        terminal=torch.where(done,torch.full_like(terminal,i+1),terminal)
        session=getattr(nominal,'_isaac_session',None)
        if session is not None:
            session.execution_step(score,state,pose,i+1,terminal,failed)
        score.active&=~done
        if trace is not None:trace(i,forces[i].clone(),state,running.clone(),score.episode_success.clone())
        if progress is not None:progress('Predicting until hit or time limit',i+1,count)
        if not bool(score.active.any()):break
    invalidate_attempts(score,failed)
    duration=terminal.to(score.dtype)*dt
    old_time=score.episode_component_sums['time'];new_time=-score.reward_weights.time_per_s*duration
    score.episode_reward+=new_time-old_time;score.episode_component_sums['time']=new_time
    score.episode_timed_out=deployed & ~score.episode_success & ~failed
    timeout_cost=-score.reward_weights.timeout*score.episode_timed_out.to(score.dtype)
    score.episode_reward+=timeout_cost;score.episode_component_sums['timeout']+=timeout_cost
    score.episode_component_sums['recovery']=torch.zeros_like(duration)
    stride=nominal.physics_steps_per_control
    command_cutoffs=((terminal+stride-1)//stride*stride).clamp_max(count)
    # Diagnostics describe the executed prefix, not discarded future commands.
    _,reference_info=reference_feasibility(packets,terminal,dt,execution['feasibility'])
    within=torch.arange(positions.shape[1],device=nominal.device)[None]<=terminal[:,None]
    correction=torch.where(within,(reference['position_origin_m']+positions.new_tensor(offset)-positions).norm(dim=-1),0.).amax(-1)
    score.terminal_steps=terminal;score.command_cutoffs=command_cutoffs
    score.reference_packets=packets;score.reference_trajectory=reference;score.predicted_pose=pose
    score.execution_state=state
    score.deployment=dict(planned=deployed,predicted_valid_hit=nominal.episode_success.clone(),
        recovered=torch.zeros_like(deployed),joint_success=torch.zeros_like(deployed),recovery_evaluated=torch.zeros_like(deployed),
        duration_s=command_cutoffs.to(score.dtype)*dt,termination_time_s=duration,
        recovery_position_error_m=torch.zeros_like(duration),maximum_execution_drone_displacement_m=maximum,nominal=batch.nominal,
        planning_minimum_tip_distance_m=nominal.episode_minimum_tip_distance.clone(),
        planning_maximum_drone_displacement_m=nominal.episode_maximum_point_displacement.clone(),
        planning_failure=planning_failed,planning_non_tip_first=nominal.episode_non_tip_first.clone(),
        planning_invalid_tip_entry=nominal.episode_invalid_tip_entry.clone(),
        planning_tip_entry_speed=nominal.episode_first_tip_directed_speed.clone(),planning_tip_entry_angle=nominal.episode_first_tip_angle.clone(),
        reference_infeasible=reference_failed,pose_domain_failure=pose_failed,reference_position_correction_m=correction,
        **{f'target_{axis}_m':score.target[:,j].clone() for j,axis in enumerate('xyz')},
        **{f'initial_attachment_{axis}_m':batch.truth.positions_m[:,0,j].clone() for j,axis in enumerate('xyz')},**reference_info)
    return score
