"""Saved M0 forecast, component diagnostics and unseen-flight cascade checks."""
import platform
from .current_adaptation_fit import *
from simulator.research_config import snapshot_assets
from simulator.research_pose import ResearchPoseModel


def freeze_fold(job,label):
    folder=Path(job)/'models'/label
    if (folder/'model.json').exists():return read(folder/'model.json')
    cable_dir=Path(job)/'cable_batched'/label;drone_dir=Path(job)/'drone_attitude_refined'/label
    cr=read(cable_dir/'result.json');dr=read(drone_dir/'result.json')
    if cr['training']!=dr['training'] or cr['heldout']!=dr['heldout']:raise ValueError('Component fold mismatch')
    model=read(Path(job)/'source_candidate/model.json')
    model['cable'].update(EI_n_m2=cr['selected_physics'][0],Cb_n_m2_s=cr['selected_physics'][1])
    model['motion_residual'].update(checkpoint=str((cable_dir/'cable_residual.pt').resolve()),
        sha256=sha256_file(cable_dir/'cable_residual.pt'),specification=cr['specification'])
    model['fullstate_execution'].update(checkpoint=str((drone_dir/'drone_model.json').resolve()),
        sha256=sha256_file(drone_dir/'drone_model.json'),source_job=str(Path(job).resolve()))
    model['adaptation']=dict(round='current_adp0',training=cr['training'],heldout=cr['heldout'],
        source_model='M0',prospective_validation=False,model_selected=False)
    folder.mkdir(parents=True);model=snapshot_assets(model,folder);save(folder/'model.json',model)
    return model


def summarize_prediction(trial,times,pose,cable):
    p,r,sites=trial.measured(times);marker_ids=list(CableConfiguration.from_mapping(trial.model['cable']).marker_node_indices[1:])
    predicted_p=pose['position_origin_m'][0].detach().cpu().numpy()
    predicted_r=pose['rotation_tracking_to_world'][0].detach().cpu().numpy()
    predicted_q=cable[0].detach().cpu().numpy()
    actual_valid=np.isfinite(r).all((1,2));angles=np.full(len(times),np.nan)
    angles[actual_valid]=np.rad2deg(np.linalg.norm(Rotation.from_matrix(predicted_r[actual_valid].transpose(0,2,1)@r[actual_valid]).as_rotvec(),axis=1))
    phase=(times>=0)&(times<=1)
    strike=.94;pred_at=interpolate_positions(times,predicted_q[:,-1],np.array([strike]))[0]
    measured_at=trial.measured(np.array([strike]))[2][0,-1]
    result=dict(observed_grid_span_s=[times[0],times[-1]],origin=metric(predicted_p,p,times),
        all_cable_markers=metric(predicted_q[:,marker_ids],sites[:,1:],times),
        tip=metric(predicted_q[:,-1],sites[:,-1],times),
        whip_attitude_rmse_deg=float(np.sqrt(np.nanmean(angles[phase]**2))),
        tip_prediction_error_at_saved_strike_m=float(np.linalg.norm(pred_at-measured_at)),
        saved_strike_time_s=strike,reference_to_actual_target='virtual target [-1,0,1.1]; old flight unchanged',
        measured_tip_target_distance_at_saved_strike_m=float(np.linalg.norm(measured_at-np.array([-1,0,1.1]))),
        predicted_tip_at_saved_strike_m=pred_at,measured_tip_at_saved_strike_m=measured_at)
    return result,dict(time=times,position=predicted_p,rotation=predicted_r,cable=predicted_q,measured_position=p,measured_sites=sites)


def evaluate_model(job,trial,model,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metrics.json').exists():return read(folder/'metrics.json')
    engine=ResearchExecutionModel.from_mapping(model,root=ROOT,device='cuda')
    # Stay within observed CSV coverage. Do not extrapolate trimmed tracking or include landing.
    end=min(11.19,float(trial.data['time'][-1])-.02)
    times=trial.grid(end);state,start,projection=trial.cable_state(engine.physics)
    pose=trial.initial_pose(engine.drone.parameters);d=trial.data
    if abs(start-pose.time_s)>1e-12:raise ValueError('Initial timestamps disagree')
    cache=Path(job)/'pose_cache'/model['fullstate_execution']['sha256']/trial.name
    cache.mkdir(parents=True,exist_ok=True)
    if (cache/'prediction.npz').exists():
        with np.load(cache/'prediction.npz') as z:
            np.testing.assert_allclose(z['time'],times,rtol=0,atol=1e-12)
            result={k:torch.tensor(z[k],device='cuda') for k in z.files if k!='time'}
    else:
        result=pose_prediction(trial,engine.drone,times)
        np.savez_compressed(cache/'prediction.npz',time=times,**{k:v.cpu().numpy() for k,v in result.items()})
    if not bool(result['valid'].all()):raise ValueError('Invalid complete drone prediction')
    if not torch.allclose(state.positions_m[:,0],result['position_attachment_m'][:,0],atol=1e-9,rtol=0):
        raise ValueError('Initial attachment mismatch')
    # Same checked cascade, with a reusable predicted-pose cache for ablations.
    q,_=cable_forward(engine,dict(q=state.positions_m,v=state.velocities_m_s,roots=result['position_attachment_m']),
        [model['cable']['EI_n_m2'],model['cable']['Cb_n_m2_s']])
    metrics,arrays=summarize_prediction(trial,times,result,q)
    metrics['initial_projection_max_coordinate_m']=projection
    metrics['all_pose_domains_valid']=bool(result['valid'].all())
    np.savez_compressed(folder/'prediction.npz',**arrays);save(folder/'metrics.json',metrics)
    return metrics


def baseline_run(job=JOB):
    job=Path(job);model=read(job/'source_candidate/model.json')
    m0=read(ROOT/'runs/rehearsals/20260908-203914-039721/model.json')
    for p in sorted((job/'inputs').iterdir()):
        t=Trial(job,p.name,model)
        # Original saved deployment forecast is assessed separately from causal
        # initialization using measured pre-flight states.
        d=t.data;sp,sr,ss=t.measured(d['saved_time'])
        saved=dict(origin=metric(d['saved_origin'],sp,d['saved_time']),tip=metric(d['saved_cable'][:,-1],ss[:,-1],d['saved_time']))
        save(p/'original_saved_forecast_metrics.json',saved)
        for label,m in [('M0_initialized',m0),('smooth_M0_initialized',model)]:
            note(job,'baseline '+label,take=t.name)
            evaluate_model(job,t,m,job/'validation'/label/t.name)


def validation_run(job=JOB):
    job=Path(job);model=read(job/'source_candidate/model.json')
    m0=read(ROOT/'runs/rehearsals/20260908-203914-039721/model.json')
    names=sorted(p.name for p in (job/'inputs').iterdir());review={}
    baseline_run(job)
    for label,training,heldout in folds(names):
        full=freeze_fold(job,label);targets=heldout or names
        review[label]={}
        for name in targets:
            trial=Trial(job,name,model)
            drone_only=deepcopy(m0);drone_only['fullstate_execution']=deepcopy(full['fullstate_execution'])
            cable_only=deepcopy(full);cable_only['fullstate_execution']=deepcopy(m0['fullstate_execution'])
            variants=[('both',full)] if not heldout else [('drone_only',drone_only),('cable_only',cable_only),('both',full)]
            for variant,m in variants:
                note(job,'whole-flight validation',fold=label,take=name,variant=variant)
                metrics=evaluate_model(job,trial,m,job/'validation'/label/name/variant)
                review[label].setdefault(name,{})[variant]=metrics
        save(job/'validation/results_in_progress.json',review)
    rows=[]
    for n in names:
        base=read(job/'validation/M0_initialized'/n/'metrics.json')
        new=review['leave_out_'+n][n]['both']
        rows.append(dict(take=n,
            baseline_drone_whip_rmse_m=base['origin']['whip']['rmse_m'],adapted_drone_whip_rmse_m=new['origin']['whip']['rmse_m'],
            baseline_tip_whip_rmse_m=base['tip']['whip']['rmse_m'],adapted_tip_whip_rmse_m=new['tip']['whip']['rmse_m'],
            baseline_marker_whip_rmse_m=base['all_cable_markers']['whip']['rmse_m'],adapted_marker_whip_rmse_m=new['all_cable_markers']['whip']['rmse_m'],
            baseline_tip_strike_error_m=base['tip_prediction_error_at_saved_strike_m'],adapted_tip_strike_error_m=new['tip_prediction_error_at_saved_strike_m'],
            baseline_drone_early_recovery_rmse_m=base['origin']['early_recovery']['rmse_m'],adapted_drone_early_recovery_rmse_m=new['origin']['early_recovery']['rmse_m'],
            baseline_tip_complete_rmse_m=base['tip']['complete']['rmse_m'],adapted_tip_complete_rmse_m=new['tip']['complete']['rmse_m']))
    summary={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='take'}
    improved=sum(r['adapted_tip_whip_rmse_m']<r['baseline_tip_whip_rmse_m'] for r in rows)
    recommendation=(summary['adapted_tip_whip_rmse_m']<summary['baseline_tip_whip_rmse_m'] and
        summary['adapted_drone_whip_rmse_m']<summary['baseline_drone_whip_rmse_m'] and improved>=4)
    save(job/'validation/results.json',dict(per_fold=review,heldout_rows=rows,equal_flight_mean=summary,
        improved_tip_flights=improved,recommend_for_new_planning=bool(recommendation),
        selection_rule='Mean held-out whip tip and drone RMS improve, tip improves in at least four of five flights; recovery reviewed separately',
        independent_future_flight_test=False,flight_authorized=False))
    note(job,'validation complete',recommend_for_new_planning=bool(recommendation),means=summary)
