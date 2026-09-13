"""Plot the saved physical drone/cable motion without regenerating its ghost."""
import numpy as np


def draw(figure,arrays,metadata,settings):
    figure.clear();axes=figure.subplots(4,1,sharex=True)
    direction=np.asarray(settings['task']['strike_direction'],float);direction/=np.linalg.norm(direction)
    mask=arrays['prediction_time_s']<=metadata['whip_end_s']+1e-10
    t=arrays['prediction_time_s'][mask]
    origin=np.asarray(metadata['initial_tracking_origin_m']);target=np.asarray(metadata['target_position_m'])
    p=(arrays['origin_positions_m'][mask]-origin)@direction
    tip=(arrays['cable_positions_m'][mask,-1]-origin)@direction
    velocity=arrays['origin_velocities_m_s'][mask]@direction
    cable=arrays['cable_velocities_m_s'][mask]@direction
    axes[0].plot(t,p,label='Modeled drone',color='#2563eb');axes[0].plot(t,tip,label='Cable tip',color='#ea580c')
    axes[0].axhline((target-origin)@direction,color='#059669',ls=':',label='Target')
    axes[0].set(ylabel='Forward displacement [m]',title='Forward pull and backward release')
    axes[1].plot(t,velocity,label='Modeled drone',color='#2563eb');axes[1].plot(t,cable[:,-1],label='Cable tip',color='#ea580c')
    axes[1].axhline(0,color='#64748b',lw=.8);axes[1].set(ylabel='Forward velocity [m/s]')
    limit=max(1.,float(np.abs(cable).max()))
    mesh=axes[2].pcolormesh(t,np.arange(cable.shape[1]),cable.T,cmap='coolwarm',vmin=-limit,vmax=limit,shading='nearest')
    axes[2].set(ylabel='Cable vertex (0 = drone)',xlabel='Whip time [s]')
    figure.colorbar(mesh,ax=axes[2],label='Forward velocity [m/s]')
    q=arrays['cable_positions_m'][mask]
    edge=np.diff(q,axis=1);unit=edge/np.maximum(np.linalg.norm(edge,axis=-1,keepdims=True),1e-12)
    bend=np.arctan2(np.linalg.norm(np.cross(unit[:,:-1],unit[:,1:]),axis=-1),
        np.clip(np.sum(unit[:,:-1]*unit[:,1:],axis=-1),-1,1))
    lengths=np.linalg.norm(edge[0],axis=-1);material=np.cumsum(lengths)[:-1]/lengths.sum()
    mesh=axes[3].pcolormesh(t,material,bend.T,cmap='magma',shading='nearest')
    peak=material[bend.argmax(1)]
    axes[3].plot(t,np.where(bend.max(1)>=.05,peak,np.nan),color='cyan',lw=1)
    axes[3].set(ylabel='Bend location (0 = attachment)',xlabel='Whip time [s]')
    if metadata.get('objective_schema')!='targeted_fold_strike_v1':
        figure.colorbar(mesh,ax=axes[3],label='Local turning angle [rad]')
    hit=metadata.get('strike_time_s',metadata.get('predicted_hit_time_s'))
    for ax in axes:
        if hit is not None:ax.axvline(hit,color='#111827',ls='--',lw=1)
        ax.spines[['top','right']].set_visible(False)
    for ax in axes[:2]:ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.2)
    if metadata.get('objective_schema')=='targeted_fold_strike_v1':
        import torch
        from planning.strike_objective import fold_features
        material_grid=np.r_[0.,np.cumsum(lengths)]/lengths.sum()
        opposition,turn,location,_=fold_features(torch.as_tensor(q),torch.as_tensor(material_grid),settings['fold_constraint'])
        axes[0].set_title('Quadrotor motion and tip strike')
        axes[3].clear();axes[3].plot(t,location.numpy(),color='#ea580c',label='Tracked dominant bend')
        axes[3].set(ylabel='Material coordinate',xlabel='Whip time [s]',ylim=(0,1))
        axes[3].axhline(settings['fold_constraint']['start_material_max'],color='#64748b',ls=':')
        axes[3].axhline(settings['fold_constraint']['end_material_min'],color='#64748b',ls=':')
        axes[3].axvline(metadata['fold_completed_time_s'],color='#2563eb',ls='--',label='Fold completed')
        if hit is not None:axes[3].axvline(hit,color='#111827',ls='--',label='Scored strike')
        axes[3].legend(frameon=False,fontsize=8)
        return dict(fold_valid=metadata['predicted_fold_valid'],strike_time_s=hit,
            directed_tip_speed_m_s=metadata['directed_tip_speed_m_s'],strike_distance_m=metadata['strike_distance_m'],
            source='Saved simulated geometry and velocity; no physical impact measurement')
    cfg=settings['task'];pull=(p>=cfg.get('minimum_pull_distance_m',.25))&(velocity>=cfg.get('minimum_pull_speed_m_s',1.))
    loaded=np.maximum.accumulate(pull)
    back=np.maximum.accumulate(p)-p
    release=loaded&(back>=cfg.get('minimum_backward_distance_m',.1))&(velocity<=-cfg.get('minimum_backward_speed_m_s',.5))
    first=lambda flag:float(t[np.flatnonzero(flag)[0]]) if flag.any() else None
    result=dict(first_pull_qualified_frame_s=first(pull),first_backward_release_qualified_frame_s=first(release),
        peak_forward_drone_speed_m_s=float(velocity.max()),minimum_forward_drone_speed_m_s=float(velocity.min()),
        source='Exact saved model velocities; no new ghost or finite-difference estimate',required=cfg.get('require_pullback',False))
    if hit is not None:
        index=min(int(np.searchsorted(t,hit,side='left')),len(t)-1)
        contact_velocity=float(np.interp(hit,t,velocity));backward=float(p[t<=hit].max()-np.interp(hit,t,p))
        result.update(contact_time_s=hit,contact_interval_end_s=float(t[index]),drone_forward_speed_at_contact_m_s=contact_velocity,
            tip_forward_speed_at_contact_m_s=float(np.interp(hit,t,cable[:,-1])),backward_travel_at_contact_m=backward,
            ordered_pullback_at_contact=bool(pull[t<hit].any() and backward>=cfg.get('minimum_backward_distance_m',.1)-1e-9 and contact_velocity<=-cfg.get('minimum_backward_speed_m_s',.5)+1e-9))
    return result
