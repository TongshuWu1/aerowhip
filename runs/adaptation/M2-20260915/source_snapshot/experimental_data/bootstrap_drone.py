"""Whip-only full-maneuver residual fit with corrected nominal pose frozen."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from .bootstrap_bundle import read,VERSION,snapshot_stage,verify
from .nominal_pose_fit import PreparedTrial,integration_steps,score_pose
from .io import atomic_json,sha256_file
from .historical_fit import progress,check_stop
from simulator.drone_pose_response import PoseResponseParameters,nominal_acceleration,predict_pose,CommandSchedule
from simulator.drone_pose_residual import DronePoseResidual,save_residual


def trial_args(trial,params,*,device='cuda',maneuver_only=False):
    g=np.array([getattr(params,k) for k in ['kp_xy','kp_z','kd_xy','kd_z','feedforward_xy','feedforward_z']])
    scales=(params.attitude_acceleration_scale_xy,params.attitude_acceleration_scale_z)
    state=trial.state(g,device,params.attitude_time_constant_s,params.delay_s,scales)
    s=trial.schedule;stream=CommandSchedule(s.time,s.values.to(device),coverage_end_s=s.coverage_end,valid=s.valid,valid_until_s=s.valid_until)
    times=trial.time[trial.time<=trial.context['csv_end_s']] if maneuver_only else trial.time
    return dict(initial=state,schedule=stream,output_time_s=times,parameters=params,offset_tracking_m=trial.offset)


class TranslationBatch:
    def __init__(self,trials,params,device='cuda'):
        self.params=params;self.trials=trials;self.plans=[];self.states=[];self.times=[]
        for t in trials:
            times=t.time[t.time<=t.context['csv_end_s']];self.times.append(times)
            self.plans.append(integration_steps(t.schedule,times,params.delay_s));self.states.append(trial_args(t,params,device=device)['initial'])
        n=max(len(x[0]) for x in self.plans);b=len(trials)
        dt=np.zeros((n,b,1));cmd=np.zeros((n,b,11))
        for i,(h,c,_) in enumerate(self.plans):dt[:len(h),i,0]=h;cmd[:len(h),i]=c;cmd[len(h):,i]=c[-1]
        self.dt=torch.as_tensor(dt,dtype=torch.float64,device=device);self.cmd=torch.as_tensor(cmd,dtype=torch.float64,device=device)
        self.p=torch.cat([s.position for s in self.states]);self.v=torch.cat([s.velocity for s in self.states]);self.bias=torch.cat([s.compensation for s in self.states])
        self.masks=[torch.as_tensor(t.masks['fit_position'][:len(times)],device=device) for t,times in zip(trials,self.times)]
        self.truth=[torch.as_tensor(t.truth['position_origin_m'][:len(times)],device=device) for t,times in zip(trials,self.times)]

    def predict(self,network):
        p=self.p;v=self.v;ps=[p];penalty=p.new_zeros(len(p))
        for h,c in zip(self.dt,self.cmd):
            extra=network(p,v,self.bias,c) if network is not None else torch.zeros_like(p)
            a=nominal_acceleration(p,v,self.bias,c,self.params)+extra
            pm=p+.5*h*v;vm=v+.5*h*a
            delta=network(pm,vm,self.bias,c) if network is not None else torch.zeros_like(p)
            am=nominal_acceleration(pm,vm,self.bias,c,self.params)+delta
            p=p+h*vm;v=v+h*am;ps.append(p)
            penalty=penalty+h[:,0]*delta.square().mean(-1)
        allp=torch.stack(ps)
        pred=[allp[torch.as_tensor(plan[2],device=p.device),i] for i,plan in enumerate(self.plans)]
        return pred,(penalty/self.dt[:,:,0].sum(0)).mean()

    def loss(self,network,regularization=.01):
        predictions,penalty=self.predict(network)
        loss=torch.stack([((p[m]-y[m])/.05).square().mean() for p,y,m in zip(predictions,self.truth,self.masks)]).mean()
        return loss+regularization*penalty/.5**2


def run(job):
    job=Path(job);settings=read(job/'protocol.json');output=job/'drone';output.mkdir()
    snapshot_stage(job,'drone',['experimental_data/bootstrap_drone.py','experimental_data/bootstrap_bundle.py',
        'experimental_data/nominal_pose_fit.py','experimental_data/drone_pose_response_data.py','experimental_data/drone_pose_initialization.py',
        'simulator/drone_pose_response.py','simulator/drone_pose_residual.py'])
    reviews=read(job/'nominal_source_protocol.json')['reviews']
    trials=[PreparedTrial(job/'pose_inputs'/VERSION/n,review=reviews[n]) for n in settings['drone_response_takes']]
    summary={}
    for label in ['final_all_three']+['leave_out_'+t.name for t in trials]:
        folder=output/label;folder.mkdir();nominal=read(job/'nominal'/(label+'.json'));params=PoseResponseParameters(**nominal['parameters'])
        train=trials if label=='final_all_three' else [t for t in trials if t.name!=label.removeprefix('leave_out_')]
        torch.manual_seed(settings['seed']);network=DronePoseResidual(settings['drone_hidden'],settings['drone_acceleration_limit_m_s2']).to(device='cuda',dtype=torch.float64)
        batch=TranslationBatch(train,params);optimizer=torch.optim.Adam(network.parameters(),lr=settings['drone_learning_rate'],weight_decay=1e-4)
        best=deepcopy(network.state_dict());best_update=0;best_loss=float(batch.loss(network).detach());history=[]
        for update in range(1,settings['drone_updates']+1):
            check_stop(job);optimizer.zero_grad();loss=batch.loss(network,settings['drone_regularization'])
            if not torch.isfinite(loss):raise ValueError('Nonfinite drone residual loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(network.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            # Compare a loss with the exact weights that produced it.
            if update%10==0 or update==settings['drone_updates']:
                with torch.no_grad():score=float(batch.loss(network,settings['drone_regularization']))
                if score<best_loss:best=deepcopy(network.state_dict());best_loss=score;best_update=update
                row=dict(update=update,loss=float(loss.detach()),selection_loss=score,gradient_norm=float(norm));history.append(row)
                atomic_json(folder/'history.json',history);progress(output,label+' residual',**row)
        network.load_state_dict(best);network.eval();network.requires_grad_(False);save_residual(folder/'drone_residual.pt',network)
        results={}
        with torch.no_grad():
            for trial in trials:
                args=trial_args(trial,params);baseline=predict_pose(**args);corrected=predict_pose(**args,residual=network)
                basearrays,base=score_pose(trial,baseline);arrays,score=score_pose(trial,corrected)
                cache=TranslationBatch([trial],params);pred,_=cache.predict(network)
                count=len(pred[0]);error=float((pred[0]-corrected['position_origin_m'][0,:count]).abs().max())
                if error>1e-10:raise AssertionError('Training/evaluation residual recurrence mismatch')
                np.savez_compressed(folder/(trial.name+'.npz'),time_s=trial.time,**arrays,measured_markers=trial.truth.get('cable_position_m',np.empty(0)))
                results[trial.name]=dict(nominal=base,residual=score,training_evaluator_max_difference_m=error)
        candidate=dict(nominal=nominal,residual=dict(schema='drone_pose_residual_v1',checkpoint=str((folder/'drone_residual.pt').resolve()),
            sha256=sha256_file(folder/'drone_residual.pt'),specification=network.specification(),target='Additional realized origin acceleration; no direct injection into attitude drive'))
        atomic_json(folder/'candidate.json',candidate)
        summary[label]=dict(training=[t.name for t in train],selected_update=best_update,selected_regularized_loss=best_loss,results=results)
        atomic_json(folder/'review.json',summary[label])
    atomic_json(output/'results.json',summary);atomic_json(job/'drone_verification.json',verify(job))
