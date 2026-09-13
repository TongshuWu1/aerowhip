"""Local M1 command inversion toward a fixed M0 physical-motion reference."""
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

    No recordings are read here. Finalized M1 supplies the updated dynamics;
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
    r,pred,packets,free=evaluate(z[None],single)
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
        status('Local M1 command correction',iteration=iteration,best_cost_m2=cost,rollouts=calls)
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
        current_jerk=(jerk_basis@np.concatenate((np.tile(origin,(3,1)),current))).reshape(-1)
        current_packets=spline.decode(tensor(current),origin)[0][:cutoff].cpu().numpy()
        delta=local_step(r[0],jac,radius,jerk_map,current_jerk,jerk_bound,
            damping=settings['damping'],packet_map=packet_map,current_packets=current_packets,limits=limits)
        accepted=False;rejections=[];before=cost
        for alpha in settings['backtracking_scales']:
            proposal=z+alpha*delta;candidate=decode(proposal)
            cp,_=spline.decode(candidate,origin);cp=cp[:cutoff]
            if not bool(spline.jerk_valid(candidate,origin,tensor(jerk_limits))) or not bool(command_valid(cp[None],limits)[0]):
                rejections.append('command envelope');continue
            rr,pp,cc,ff=evaluate(proposal[None],single);value=float(rr[0]@rr[0])
            if not bool(pp['complete_valid'][0]) or not np.isfinite(value) or value>=cost-1e-10:
                rejections.append('nonlinear objective/domain');continue
            try:finish(candidate)
            except ValueError as exc:rejections.append(str(exc));continue
            z=proposal;r=rr;cost=value;accepted=True;break
        row=dict(iteration=iteration,cost_m2=cost,accepted=accepted,
            jacobian_relative_difference=agreement,one_sided_columns=checks,derivative_refinements=refinements,
            step_scale=alpha if accepted else 0.,trust_radius=radius,
            rollout_count=calls,rejections=rejections,elapsed_s=time.perf_counter()-started)
        history.append(row);atomic_json(job/'history.json',history)
        np.savez_compressed(job/'current_best.npz',position_control_points_m=decode(z).cpu().numpy())
        if accepted:
            radius=min(settings['maximum_trust_radius'],radius*1.3) if alpha==1 else max(.025,radius*.7)
            if (before-cost)/max(before,1e-12)<settings['relative_improvement_stop']:
                stop='Small verified objective improvement';break
        else:
            radius*=.25
            if radius<.025:stop='No improving feasible local step';break
    best=decode(z)
    rr,pp,cc,_=evaluate(z[None],single)
    final,terms=tracking_cost(pp['cable_positions_m'][:,:,-1],pp['position_origin_m'],cc,reference,weights)
    if abs(float(final[0])-cost)>1e-8:raise ValueError('Local final objective changed')
    if cost>=initial_cost:raise ValueError('No improving correction found')
    info=dict(optimizer='Bounded finite-difference Gauss-Newton',active_coordinates=dimension,
        active_controls=active.tolist(),iterations=len(history),rollout_count=calls,
        stop_reason=stop,optimization_elapsed_s=time.perf_counter()-started,
        maximum_jacobian_relative_difference=max(x['jacobian_relative_difference'] for x in history))
    atomic_json(job/'optimization.json',info)
    return best,cost,{k:float(v[0]) for k,v in terms.items()},info
