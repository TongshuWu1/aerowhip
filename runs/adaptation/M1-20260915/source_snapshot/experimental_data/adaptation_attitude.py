"""Fit the attitude mapping after freezing the adapted translation and NN.

The translational residual indirectly changes the nominal attitude drive by
changing the simulated position/velocity errors. Cache that exact recurrence.
"""
from .current_adaptation_fit import *
from simulator.drone_pose_residual import load_residual
from simulator.drone_pose_response import nominal_acceleration
import shutil


class ResidualAttitudeTrial(AttitudeTrial):
    def __init__(self,trial,params,network):
        gains=[getattr(params,k) for k in GAIN_NAMES]
        super().__init__(trial,gains,params.delay_s)
        # Match the native executor's ceiling tolerance exactly. The legacy
        # nominal fitting helper can add a redundant substep at roundoff.
        events=trial.data['packet_time']+params.delay_s;dt=[];commands=[];indices=[0]
        maximum=min(.005,params.attitude_time_constant_s/4,.25/max(params.kd_xy+np.sqrt(params.kp_xy),params.kd_z+np.sqrt(params.kp_z)))
        for left,right in zip(self.time[:-1],self.time[1:]):
            local=np.r_[left,events[(events>left+1e-12)&(events<right-1e-12)],right]
            for a,b in zip(local[:-1],local[1:]):
                index=np.searchsorted(trial.data['packet_time'],(a+b)*.5-params.delay_s+1e-12,side='right')-1
                cmd=trial.data['hover_commands'][-1] if index<0 else trial.data['packets'][index]
                count=max(1,int(np.ceil((b-a)/maximum-1e-10)))
                dt.extend([(b-a)/count]*count);commands.extend([cmd]*count)
            indices.append(len(dt))
        dt=np.array(dt);commands=np.array(commands)
        self.dt=dt;self.indices=np.array(indices);self.yaw=commands[:,9]
        state=trial.state(gains,'cpu',params.attitude_time_constant_s,params.delay_s,
            (params.attitude_acceleration_scale_xy,params.attitude_acceleration_scale_z))
        p=state.position;v=state.velocity;b=state.compensation;aa=[];am=[]
        with torch.no_grad():
            for h,cmd in zip(dt,commands):
                c=torch.tensor(cmd[None],dtype=torch.float64)
                a=nominal_acceleration(p,v,b,c,params)
                full=a+network(p,v,b,c)
                pm=p+.5*h*v;vm=v+.5*h*full
                mid=nominal_acceleration(pm,vm,b,c,params)
                p=p+h*vm;v=v+h*(mid+network(pm,vm,b,c))
                aa.append(a[0].numpy());am.append(mid[0].numpy())
        self.a=np.array(aa);self.am=np.array(am)


def run(job=JOB):
    job=Path(job);model=read(job/'source_candidate/model.json');names=sorted(p.name for p in (job/'inputs').iterdir())
    old=load_engine(job).drone.parameters
    prior=np.array([old.attitude_acceleration_scale_xy,old.attitude_acceleration_scale_z,old.attitude_time_constant_s])
    lower=np.maximum(prior*.5,[.1,.05,.02]);upper=np.minimum(prior*2,[3.,1.5,.30])
    for label,training,heldout in folds(names):
        src=job/'drone'/label;out=job/'drone_attitude_refined'/label
        if (out/'result.json').exists():continue
        payload=read(src/'drone_model.json');params=PoseResponseParameters(**payload['nominal']['parameters'])
        net=load_residual(src/'drone_residual.pt',payload['residual']['sha256'],'cpu')
        trials=[Trial(job,n,model) for n in training];cached=[ResidualAttitudeTrial(t,params,net) for t in trials]
        initial=np.array([params.attitude_acceleration_scale_xy,params.attitude_acceleration_scale_z,params.attitude_time_constant_s])
        history=[]
        def residual(x):
            rows=[]
            for t,c in zip(trials,cached):
                predicted=c.predict(np.exp(x))
                valid=t.weights>0
                error=Rotation.from_matrix(predicted[valid].transpose(0,2,1)@c.truth[valid]).as_rotvec()
                rows.append((error*np.sqrt(t.weights[valid,None]/len(trials))/.15).ravel())
            rows.append(np.sqrt(.03)*(x-np.log(prior)))
            value=np.concatenate(rows);history.append(dict(values=np.exp(x),objective=float(value@value)))
            return value
        before=residual(np.log(initial));fit=least_squares(residual,np.log(initial),bounds=(np.log(lower),np.log(upper)),
            max_nfev=30,diff_step=1e-4,ftol=1e-6,xtol=1e-6,gtol=1e-6)
        after=residual(fit.x)
        values=np.exp(fit.x) if after@after<before@before else initial
        payload['nominal']['parameters'].update(attitude_acceleration_scale_xy=values[0],attitude_acceleration_scale_z=values[1],attitude_time_constant_s=values[2])
        payload['nominal']['attitude_stage']='training-only refit after frozen adapted translational residual'
        out.mkdir(parents=True);shutil.copy2(src/'drone_residual.pt',out/'drone_residual.pt')
        save(out/'drone_model.json',payload)
        save(out/'result.json',dict(training=training,heldout=heldout,initial=initial,selected=values,
            before_objective=float(before@before),after_objective=float(after@after),history=history,
            residual_sha256=sha256_file(out/'drone_residual.pt')))
        note(job,'attitude after residual',fold=label,before=float(before@before),after=float(after@after),values=values)
