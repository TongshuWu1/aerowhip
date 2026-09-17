"""Actual-command full-whip derivative and inference agreement after fitting."""
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experimental_data.current_adaptation_validation import *

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True)
    JOB=parser.parse_args().job.resolve()
    torch.set_num_threads(1);model=freeze_fold(JOB,'all_five');trial=Trial(JOB,'whip_adp_0_001',model)
    engine=ResearchExecutionModel.from_mapping(model,device='cuda');initial=trial.initial_pose(engine.drone.parameters)
    cable,_,_=trial.cable_state(engine.physics);times=trial.grid(1.)
    packets=torch.tensor(trial.data['packets'][None],device='cuda',dtype=torch.float64)
    pt=trial.data['packet_time'];s=np.clip(pt,0,1)
    g=64*s**3*(1-s)**3
    gv=192*s**2-768*s**3+960*s**4-384*s**5
    ga=384*s-2304*s**2+3840*s**3-1920*s**4
    mask=(pt>=0)&(pt<=1);direction=torch.zeros_like(packets)
    for column,values in [(0,g),(3,gv),(6,ga)]:direction[0,:,column]=packets.new_tensor(values*mask)
    hover=torch.tensor(trial.data['hover_commands'][-1:],device='cuda',dtype=torch.float64)
    def predict(amplitude,gradients):
        return engine.predict(initial,cable,packets+amplitude*direction,pt,times,
            gradients=gradients,graph=not gradients,hover_command=hover)
    target=packets.new_tensor(trial.measured(np.array([times[-1]]))[2][0,-1])
    amplitude=torch.tensor(0.,device='cuda',dtype=torch.float64,requires_grad=True)
    start=time.perf_counter();out=predict(amplitude,True)
    loss=(out['cable_positions_m'][0,-1,-1]-target).square().sum();loss.backward()
    ad=float(amplitude.grad);forward_backward_s=time.perf_counter()-start
    inference=predict(0.,False)
    parity=max(float((out[k].detach()-inference[k]).abs().max()) for k in ['position_origin_m','rotation_tracking_to_world','cable_positions_m'])
    del out,loss
    derivatives=[]
    for eps in [1e-5,1e-6]:
        a=predict(eps,False);b=predict(-eps,False)
        fd=float(((a['cable_positions_m'][0,-1,-1]-target).square().sum()-(b['cable_positions_m'][0,-1,-1]-target).square().sum())/(2*eps))
        derivatives.append(dict(epsilon_m=eps,finite_difference=fd,absolute_difference=abs(fd-ad)))
    passed=bool(np.isfinite(ad) and parity<1e-10 and abs(derivatives[-1]['finite_difference']-ad)<1e-3+.01*abs(ad))
    save(JOB/'verification/full_whip_gradient.json',dict(passed=passed,automatic_derivative=ad,finite_differences=derivatives,
        inference_gradient_max_abs_difference=parity,forward_backward_s=forward_backward_s,gpu=torch.cuda.get_device_name(0),
        model_sha256=sha256_file(JOB/'models/all_five/model.json'),new_commands_exported=False,parameters_fitted_in_this_check=False))
    print(read(JOB/'verification/full_whip_gradient.json'),flush=True)
    if not passed:raise AssertionError('Fitted full-whip derivative check failed')
