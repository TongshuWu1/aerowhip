"""Local command inversion toward a fixed physical-motion reference."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint


def active_spline_basis(spline, cutoff):
    """Scale active controls by command P/V/A and full-domain jerk effects.

    The last control has no effect on this experiment's executed prefix. It
    remains fixed, including in the unchanged full-domain jerk constraints.
    Scales are conditioning units, not relaxed command or success limits.
    """
    matrices=[b.detach().cpu().numpy()[:,3:] for b in spline.matrices]
    executed=np.concatenate([b[:cutoff] for b in matrices])
    active=np.flatnonzero(np.linalg.norm(executed,axis=0)>1e-10)
    jerk=spline.jerk_control_basis.detach().cpu().numpy()[:,3:]
    operators=[b[:cutoff,active]/(scale*np.sqrt(cutoff))
               for b,scale in zip(matrices,(.05,.3,2.))]
    operators.append(jerk[:,active]/(20*np.sqrt(len(jerk))))
    operator=np.concatenate(operators)
    _,singular,right=np.linalg.svd(operator,full_matrices=False)
    if singular[-1]<1e-10:raise ValueError('Unobservable correction coordinates')
    basis=np.zeros((9,len(active)))
    basis[active]=right.T/singular
    return np.kron(basis,np.eye(3)),active


def residual_vector(tip,quad,commands,reference,weights):
    """Squared norm equals the original mean-squared-distance objective."""
    return np.concatenate([
        (np.asarray(tip)-reference['tip']).reshape(-1)*np.sqrt(weights['tip']/len(tip)),
        (np.asarray(quad)-reference['quadrotor']).reshape(-1)*np.sqrt(weights['quadrotor']/len(quad)),
        (np.asarray(commands)[:,:3]-reference['command']).reshape(-1)*np.sqrt(weights['command']/len(commands))])


def strike_metrics(tip,grid,reference_tip,strike_time,target=None):
    """Fixed-time interpolation, distinct from nearest approach."""
    grid=np.asarray(grid);tip=np.asarray(tip);reference_tip=np.asarray(reference_tip)
    if (len(grid)<2 or not np.isfinite(grid).all() or np.any(np.diff(grid)<=0)
            or not grid[0]<=strike_time<=grid[-1]):
        raise ValueError('Strike time must lie inside a finite increasing rollout grid')
    at=lambda values:np.array([np.interp(strike_time,grid,values[:,j]) for j in range(3)])
    point=at(tip)
    result=dict(time_s=float(strike_time),reference_error_m=float(np.linalg.norm(point-at(reference_tip))))
    if target is not None:result['target_distance_m']=float(np.linalg.norm(point-np.asarray(target)))
    return result


def strike_allowed(candidate,baseline,mode='none',tolerance=0.):
    if mode not in ('none','reference','target') or tolerance<0:
        raise ValueError('Invalid strike guard')
    if mode=='none':return True
    key={'reference':'reference_error_m','target':'target_distance_m'}[mode]
    if key not in baseline or key not in candidate:raise ValueError('Strike guard needs strike metadata')
    return bool(np.isfinite(candidate[key]) and candidate[key]<=baseline[key]+tolerance)


def update_trust(radius,damping,ratio,step_norm,maximum_radius,minimum_radius):
    """Use actual/predicted reduction, including failed nonlinear trials."""
    if not np.isfinite(ratio) or ratio<.25:
        return max(minimum_radius,radius*.5),min(1e8,damping*4)
    if ratio>.75:
        return min(maximum_radius,radius*1.5 if step_norm>=.8*radius else radius),max(1e-12,damping*.5)
    return radius,damping


def local_step(residual,jacobian,radius,jerk_map,current_jerk,jerk_limits,damping=1e-4,
               packet_map=None,current_packets=None,limits=None):
    """Regularized Gauss-Newton subproblem, bounded step and exact linear jerk.

    Nonlinear vehicle/cable limits and complete recovery are checked on actual
    candidate rollouts outside this local approximation.
    """
    hessian=jacobian.T@jacobian+damping*np.eye(jacobian.shape[1])
    gradient=jacobian.T@residual
    def objective(delta):
        return .5*delta@hessian@delta+gradient@delta,hessian@delta+gradient
    constraints=[LinearConstraint(jerk_map,-jerk_limits-current_jerk,jerk_limits-current_jerk)]
    if packet_map is not None:
        # Spline commands are affine in the coefficients. These inequalities
        # constrain actual commands; model predictions are checked separately.
        def envelope(delta):
            p=current_packets+np.einsum('tcd,d->tc',packet_map,delta)
            v=p[:,3:6];f=p[:,6:9]+np.array([0,0,9.80665])
            tangent=np.tan(np.deg2rad(limits['maximum_tilt_deg']))
            return np.concatenate((p[:,2]-limits['minimum_origin_z_m'],
                limits['maximum_origin_z_m']-p[:,2],
                limits['maximum_speed_m_s']**2-(v*v).sum(-1),
                limits['maximum_specific_force_m_s2']**2-(f*f).sum(-1),
                f[:,2]-limits['minimum_specific_vertical_m_s2'],
                tangent*f[:,2]-np.linalg.norm(f[:,:2],axis=-1)))
        constraints.append(dict(type='ineq',fun=envelope))
    result=minimize(objective,np.zeros(len(gradient)),jac=True,method='SLSQP',
        bounds=[(-radius,radius)]*len(gradient),constraints=constraints,
        options={'ftol':1e-11,'maxiter':300})
    if not result.success:raise ValueError('Local command solve failed: '+result.message)
    if np.max(np.abs(current_jerk+jerk_map@result.x)-jerk_limits)>1e-7:
        raise ValueError('Local step violates unchanged continuous jerk bounds')
    return result.x


def optimize(spline,baseline,origin,cutoff,engine,initial_q,times,grid,reference,
             weights,limits,jerk_limits,finish,job,status,settings):
    """Checked finite-difference Gauss-Newton with nonlinear backtracking.

    No recordings are read here. The frozen model supplies updated dynamics;
    the desired M0 physical trajectories and objective remain fixed.
    """
    import time
    import torch
    from experimental_data.io import atomic_json
    from planning.reference_correction import CoupledRollout,command_valid,tracking_cost
    started=time.perf_counter()
    tensor=lambda x:torch.as_tensor(x,device=baseline.device,dtype=baseline.dtype)
    basis,active=active_spline_basis(spline,cutoff);dimension=basis.shape[1]
    operator=tensor(basis)
    batch=CoupledRollout(engine,2*dimension+1,origin,initial_q,limits)
    single=CoupledRollout(engine,1,origin,initial_q,limits)
    calls=0
    def decode(z):
        return baseline+(tensor(z)@operator.T).reshape(*np.shape(z)[:-1],9,3)
    def residual(pred,packets):
        return torch.cat(((pred['cable_positions_m'][:,:,-1]-reference['tip']).flatten(1)*np.sqrt(weights['tip']/len(grid)),
            (pred['position_origin_m']-reference['quadrotor']).flatten(1)*np.sqrt(weights['quadrotor']/len(grid)),
            (packets[:,:,:3]-reference['command']).flatten(1)*np.sqrt(weights['command']/cutoff)),dim=1)
    def evaluate(z,executor):
        nonlocal calls
        free=decode(z);packets,_=spline.decode(free,origin);packets=packets[:,:cutoff]
        pred=executor(packets,times,grid);calls+=len(z)
        return residual(pred,packets).cpu().numpy(),pred,packets,free
    z=np.zeros(dimension);history=[]
    radius=settings['initial_trust_radius'];h=settings['finite_difference_step']
    damping=settings['damping'];small_steps=0
    minimum_radius=settings.get('minimum_trust_radius',1e-4)
    guard=settings.get('strike_guard','none')
    strike_time=settings.get('planned_strike_time_s')
    def strike(prediction):
        if strike_time is None:return {}
        return strike_metrics(prediction['cable_positions_m'][0,:,-1].cpu().numpy(),grid,
            reference['tip'].cpu().numpy(),strike_time,settings.get('target_position_m'))
    r,pred,packets,free=evaluate(z[None],single)
    baseline_strike=strike(pred)
    strike_allowed(baseline_strike,baseline_strike,guard,settings.get('strike_tolerance_m',0.))
    atomic_json(job/'strike_baseline.json',baseline_strike)
    if not bool(pred['complete_valid'][0]):raise ValueError('Original command is outside the M1 model envelope')
    finish(baseline)
    cost=float(r[0]@r[0]);initial_cost=cost;stop='Iteration budget reached'
    jerk_basis=spline.jerk_control_basis.cpu().numpy()
    jerk_map=np.kron(jerk_basis[:,3:],np.eye(3))@basis
    jerk_bound=np.tile(jerk_limits,len(jerk_basis))
    packet_map=np.zeros((cutoff,11,dimension))
    for index,matrix in enumerate(spline.matrices):
        packet_map[:,index*3:index*3+3]=(np.kron(matrix[:cutoff,3:].cpu().numpy(),np.eye(3))@basis).reshape(cutoff,3,dimension)
    for iteration in range(1,settings['maximum_iterations']+1):
        if (job/'STOP').exists():raise InterruptedError('Correction stopped; no export')
        status('Local command correction',iteration=iteration,best_cost_m2=cost,rollouts=calls)
        # Compare complete Jacobians at h and h/2. A probe outside the model
        # domain cannot be treated as a zero sensitivity.
        refinements=[]
        for refinement in range(settings['maximum_derivative_refinements']+1):
            jacobians=[];checks=[]
            for step in (h,h/2):
                probes=np.concatenate((z[None],z[None]+step*np.eye(dimension),z[None]-step*np.eye(dimension)))
                values,prediction,_,_=evaluate(probes,batch)
                valid=prediction['complete_valid'].cpu().numpy() & np.isfinite(values).all(-1)
                if not valid[0]:raise ValueError('Incumbent model validity changed')
                columns=[];one_sided=0
                for j in range(dimension):
                    plus,minus=1+j,1+dimension+j
                    if valid[plus] and valid[minus]:columns.append((values[plus]-values[minus])/(2*step))
                    elif valid[plus]:columns.append((values[plus]-values[0])/step);one_sided+=1
                    elif valid[minus]:columns.append((values[0]-values[minus])/step);one_sided+=1
                    else:raise ValueError('Both sensitivity probes invalid; no local update exported')
                jacobians.append(np.array(columns).T);checks.append(one_sided)
            agreement=float(np.linalg.norm(jacobians[0]-jacobians[1])/max(np.linalg.norm(jacobians[1]),1e-12))
            refinements.append(dict(step=h,relative_difference=agreement,one_sided_columns=checks))
            atomic_json(job/'derivative_check.json',dict(iteration=iteration,attempts=refinements))
            if agreement<=settings['maximum_jacobian_relative_difference']:break
            h*=.25
        else:
            raise ValueError(f'Command Jacobian step-size check failed after refinement: {agreement:.6g}')
        jac=jacobians[1];current=decode(z).cpu().numpy()
        current_jerk=(spline.jerk_values(tensor(current),origin).cpu().numpy().reshape(-1)
            if hasattr(spline,'jerk_values') else
            (jerk_basis@np.concatenate((np.tile(origin,(3,1)),current))).reshape(-1))
        current_packets=spline.decode(tensor(current),origin)[0][:cutoff].cpu().numpy()
        # Unit-radius projected gradient mapping incorporates command and jerk
        # constraints. It is a local stationarity diagnostic, not a global proof.
        projected=local_step(jac.T@r[0],np.eye(dimension),1.,jerk_map,current_jerk,jerk_bound,
            damping=0.,packet_map=packet_map,current_packets=current_packets,limits=limits)
        optimality=float(np.linalg.norm(projected,np.inf))
        delta=local_step(r[0],jac,radius,jerk_map,current_jerk,jerk_bound,
            damping=damping,packet_map=packet_map,current_packets=current_packets,limits=limits)
        accepted=False;rejections=[];before=cost;attempts=[];ratio=float('-inf')
        for alpha in settings['backtracking_scales']:
            proposal=z+alpha*delta;candidate=decode(proposal)
            cp,_=spline.decode(candidate,origin);cp=cp[:cutoff]
            attempt=dict(step_scale=alpha,accepted=False,strike=None);attempts.append(attempt)
            if not bool(spline.jerk_valid(candidate,origin,tensor(jerk_limits))) or not bool(command_valid(cp[None],limits)[0]):
                attempt['reason']='command envelope; rollout not evaluated';rejections.append(attempt['reason']);continue
            rr,pp,cc,ff=evaluate(proposal[None],single);value=float(rr[0]@rr[0])
            attempt.update(cost_m2=value,strike=strike(pp))
            predicted=before-float(np.square(r[0]+jac@(alpha*delta)).sum())
            ratio=(before-value)/predicted if predicted>0 and np.isfinite(value) else float('-inf')
            attempt.update(predicted_reduction_m2=predicted,actual_reduction_m2=before-value,
                reduction_ratio=ratio if np.isfinite(ratio) else None)
            if not bool(pp['complete_valid'][0]) or not np.isfinite(value) or value>=cost-1e-10:
                attempt['reason']='nonlinear objective/domain';rejections.append(attempt['reason']);continue
            if not strike_allowed(attempt['strike'],baseline_strike,guard,settings.get('strike_tolerance_m',0.)):
                attempt['reason']='fixed-time strike guard';rejections.append(attempt['reason']);continue
            try:finish(candidate)
            except ValueError as exc:attempt['reason']=str(exc);rejections.append(str(exc));continue
            attempt['accepted']=True
            z=proposal;r=rr;cost=value;accepted=True;break
        row=dict(iteration=iteration,cost_m2=cost,accepted=accepted,
            jacobian_relative_difference=agreement,one_sided_columns=checks,derivative_refinements=refinements,
            step_scale=alpha if accepted else 0.,trust_radius=radius,damping=damping,
            command_feasible_optimality=optimality,attempts=attempts,
            rollout_count=calls,rejections=rejections,elapsed_s=time.perf_counter()-started)
        history.append(row);atomic_json(job/'history.json',history)
        saved_controls={getattr(spline,'control_artifact_key','position_control_points_m'):decode(z).cpu().numpy()}
        np.savez_compressed(job/'current_best.npz',**saved_controls,
            command_packets=spline.decode(decode(z),origin)[0][:cutoff].cpu().numpy(),command_time_s=times)
        radius,damping=update_trust(radius,damping,ratio if accepted else float('-inf'),
            float(np.linalg.norm(alpha*delta,np.inf)) if accepted else 0.,settings['maximum_trust_radius'],minimum_radius)
        small=(accepted and alpha>=.5 and ratio>=.25 and
            (before-cost)/max(before,1e-12)<settings['relative_improvement_stop'])
        small_steps=small_steps+1 if small else 0
        if small_steps>=settings.get('small_improvement_patience',3) and optimality<=settings.get('optimality_tolerance',1e-5):
            stop='Repeated small improvements with command-constrained stationarity';break
        if not accepted and radius<=minimum_radius:
            stop='Trust radius exhausted without an improving feasible step; convergence unverified';break
    best=decode(z)
    rr,pp,cc,_=evaluate(z[None],single)
    final,terms=tracking_cost(pp['cable_positions_m'][:,:,-1],pp['position_origin_m'],cc,reference,weights)
    if abs(float(final[0])-cost)>1e-8:raise ValueError('Local final objective changed')
    if cost>=initial_cost:raise ValueError('No improving correction found')
    info=dict(optimizer='Bounded finite-difference Gauss-Newton',active_coordinates=dimension,
        active_controls=active.tolist(),iterations=len(history),rollout_count=calls,
        stop_reason=stop,optimization_elapsed_s=time.perf_counter()-started,
        baseline_strike=baseline_strike,final_strike=strike(pp),strike_guard=guard,
        maximum_jacobian_relative_difference=max(x['jacobian_relative_difference'] for x in history))
    atomic_json(job/'optimization.json',info)
    return best,cost,{k:float(v[0]) for k,v in terms.items()},info
