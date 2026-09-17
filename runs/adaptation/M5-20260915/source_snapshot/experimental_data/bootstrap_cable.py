"""New cable physics/residual fits, with fixed verified geometry and zero drag."""
from copy import deepcopy
from pathlib import Path
import itertools
import numpy as np
import torch
from .bootstrap_bundle import read,snapshot_stage,verify
from .historical_dataset import read_inputs
from .historical_fit import (cable_takes,cable_metrics,mean_metric,local_gradient_check,BiasPopulation,progress,check_stop)
from .constrained_identification import population,combine
from .differentiable_fit import rollout,prediction_loss,subset,save_weights
from .io import atomic_json,sha256_file
from simulator.cable.residual import MotionResidual

def run(job):
    job=Path(job);settings=read(job/'protocol.json');data=read_inputs(job);original=read(job/'original_model.json')
    snapshot_stage(job,'cable',['experimental_data/bootstrap_bundle.py','experimental_data/bootstrap_cable.py',
        'experimental_data/historical_fit.py','experimental_data/historical_dataset.py','experimental_data/differentiable_fit.py',
        'experimental_data/constrained_identification.py','experimental_data/cable_residual_fit.py','experimental_data/state_initialization.py',
        'experimental_data/force_dataset.py','simulator/cable/dder.py','simulator/cable/config.py','simulator/cable/residual.py'])
    output=job/'cable';output.mkdir();summaries={}
    for excluded in [[],*settings['folds']]:
        label='final' if not excluded else 'leave_out_'+next(n for n in excluded if n.startswith('whip'))
        folder=output/label;folder.mkdir();train=[n for n in data if n not in excluded]
        candidate=deepcopy(original);candidate.pop('motion_residual',None);candidate['cable']['external_drag_s_inv']=0.
        torch.manual_seed(settings['seed']);rng=np.random.default_rng(settings['seed'])
        progress(output,label+' physical fit',training=train,excluded=excluded)
        cable,model,takes=cable_takes(data,candidate,.65,settings['cable_windows_per_take'],'cuda')
        training=[t for t in takes if t.take_id in train]
        if len(takes)!=len(data):raise ValueError('A take has no usable cable windows; inspect masks')
        atomic_json(folder/'windows.json',{t.take_id:dict(starts=list(t.starts),horizon_s=.65) for t in takes})
        ei=sorted(set(settings['physical_EI_grid']+[candidate['cable']['EI_n_m2']]))
        cb=sorted(set(settings['physical_Cb_grid']+[candidate['cable']['Cb_n_m2_s']]))
        candidates=[[e,c,0.] for e,c in itertools.product(ei,cb)];labels={t.take_id:['balanced']*t.window_count for t in training}
        scores,_=population(training,labels,model,cable,candidates);chosen=np.array(candidates[int(scores.argmin())])
        atomic_json(folder/'physical_grid.json',dict(candidates=candidates,objectives=scores.tolist()))
        refined=np.unique([[*np.clip(chosen[:2]*[e,c],[1e-9,1e-8],[2.8e-4,.02]),0.] for e,c in itertools.product([.25,1,4],repeat=2)],axis=0)
        scores,_=population(training,labels,model,cable,refined);chosen=refined[int(scores.argmin())]
        candidate['cable'].update(EI_n_m2=float(chosen[0]),Cb_n_m2_s=float(chosen[1]))
        atomic_json(folder/'physical_refinement.json',dict(candidates=refined.tolist(),objectives=scores.tolist(),selected=chosen.tolist()))
        atomic_json(folder/'physical_model.json',candidate)
        cable,model,takes=cable_takes(data,candidate,.65,settings['cable_windows_per_take'],'cuda');training=[t for t in takes if t.take_id in train]
        _,_,short=cable_takes(data,candidate,.05,settings['cable_training_windows_per_take'],'cuda');short=[t for t in short if t.take_id in train]
        params=torch.tensor(chosen,device='cuda',dtype=torch.float64)
        baseline=cable_metrics(takes,model,cable,params,folder/'physics_predictions.npz');atomic_json(folder/'physics_metrics.json',baseline)
        network=MotionResidual(cable.node_count,hidden=settings['cable_hidden'],mode='dissipative',damping_limit_s_inv=2.).to(device='cuda',dtype=torch.float64)
        model.motion_residual=network
        check=local_gradient_check(combine([subset(t,[0]) for t in short]),model,cable,params,network)
        atomic_json(folder/'gradient_check.json',check)
        if not check['passed']:raise ValueError('Cable residual gradient check failed')
        best=deepcopy(network.state_dict());best_score=mean_metric(baseline,train,'objective');best_update=0;history=[]
        optimizer=torch.optim.Adam(network.parameters(),lr=settings['cable_learning_rate'])
        for update in range(1,settings['cable_updates']+1):
            check_stop(job);batch=combine([subset(t,rng.integers(0,t.window_count,2)) for t in short]);optimizer.zero_grad()
            prediction=rollout(batch,model,cable,params,gradients=True)
            correction=network(batch.initial_positions_m,batch.initial_velocities_m_s)
            loss=prediction_loss(prediction,batch.measured_marker_positions_m)+settings['cable_regularization']*correction.square().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite cable loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(network.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            row=dict(update=update,loss=float(loss.detach()),gradient_norm=float(norm))
            if update%12==0:
                score=mean_metric(cable_metrics(training,model,cable,params),train,'objective');row['selection_score']=score
                if score<best_score:best=deepcopy(network.state_dict());best_score=score;best_update=update
            history.append(row);atomic_json(folder/'history.json',history)
            if update%6==0:progress(output,label+' residual fit',**row)
        network.load_state_dict(best)
        with torch.no_grad():
            shifts=settings['cable_bias_grid'];model.motion_residual=BiasPopulation(network,shifts,sum(t.window_count for t in training))
            scores,_=population(training,labels,model,cable,[chosen.tolist()]*len(shifts));shift=float(shifts[int(scores.argmin())])
            network.net[-1].bias.add_(shift);model.motion_residual=network
        atomic_json(folder/'bias_selection.json',dict(dimensionless_nn_bias_candidates=shifts,objectives=scores.tolist(),chosen_shift=shift,horizon_s=.65,training_only=True))
        save_weights(folder/'cable_residual.pt',network)
        metrics=cable_metrics(takes,model,cable,params,folder/'residual_predictions.npz');atomic_json(folder/'residual_metrics.json',metrics)
        candidate['motion_residual']=dict(enabled=True,checkpoint=str((folder/'cable_residual.pt').resolve()),sha256=sha256_file(folder/'cable_residual.pt'),specification=network.specification(),drag_mode='nn_only')
        candidate['force_accounting']['aerodynamic_drag']='NN dissipation on free cable nodes; fixed external drag exactly zero'
        atomic_json(folder/'candidate_model.json',candidate)
        summaries[label]=dict(training=train,excluded=excluded,physics=baseline,residual=metrics,selected_update=best_update,geometry_frozen=True)
        atomic_json(folder/'review.json',summaries[label]);progress(output,label+' complete')
    atomic_json(output/'results.json',summaries);atomic_json(job/'cable_verification.json',verify(job))
