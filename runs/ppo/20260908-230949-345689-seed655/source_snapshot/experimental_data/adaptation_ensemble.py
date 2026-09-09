"""Independent cable fits in one GPU batch; no shared learnable weights."""
from .current_adaptation_fit import *


class CableEnsemble(torch.nn.Module):
    def __init__(self,networks):
        super().__init__();self.networks=torch.nn.ModuleList(networks)

    def forward(self,q,v):
        if len(q)%len(self.networks):raise ValueError('Ensemble batch is not divisible by fold count')
        return torch.cat([net(a,b) for net,a,b in zip(self.networks,q.chunk(len(self.networks)),v.chunk(len(self.networks)))])

    def components(self,q,v):
        parts=[net.components(a,b) for net,a,b in zip(self.networks,q.chunk(len(self.networks)),v.chunk(len(self.networks)))]
        return tuple(torch.cat([p[i] for p in parts]) for i in range(2))

    def drag_rates(self,q):
        return q.new_zeros(q.shape[-2])


def repeat_data(data,count):
    return {k:v.repeat((count,)+(1,)*(v.ndim-1)) for k,v in data.items()}


def run(job=JOB):
    job=Path(job);settings=read(job/'protocol.json');model=read(job/'source_candidate/model.json')
    names=sorted(p.name for p in (job/'inputs').iterdir());trials=[Trial(job,n,model) for n in names]
    engine=load_engine(job,trainable=True);splits=folds(names);count=len(splits)
    long,rejected=cable_windows(trials,engine.physics,[0.,1.2,2.0],1.02)
    short,rejected_short=cable_windows(trials,engine.physics,settings['cable_window_starts_s'],settings['cable_window_s'])
    ld=repeat_data(join_windows(long),count);sd=repeat_data(join_windows(short),count)
    out=job/'cable_batched';out.mkdir(exist_ok=True)
    if (out/'completed.json').exists():return
    save(out/'windows.json',dict(long=[{k:r[k] for k in ['name','cutoff','time','projection_m']} for r in long],
        short=[{k:r[k] for k in ['name','cutoff','time','projection_m']} for r in short],rejected=rejected+rejected_short))
    with np.load(job/'cable/physical_grid.npz') as z:losses=z['losses'];candidates=z['candidates']
    prior=np.array([model['cable']['EI_n_m2'],model['cable']['Cb_n_m2_s']]);selected=[]
    for label,training,heldout in splits:
        weights=record_weights(long,training);scores=weights@losses+.01*(np.log(candidates/prior)**2).mean(1)
        selected.append(candidates[scores.argmin()]);folder=out/label;folder.mkdir(exist_ok=True)
        save(folder/'physics.json',dict(training=training,heldout=heldout,candidates=candidates,scores=scores,selected=selected[-1]))
    networks=[deepcopy(engine.physics.motion_residual) for _ in splits]
    engine.physics.motion_residual=CableEnsemble(networks)
    sw=torch.tensor(np.stack([record_weights(short,tr) for _,tr,_ in splits]),dtype=torch.float64,device='cuda')
    lw=torch.tensor(np.stack([record_weights(long,tr) for _,tr,_ in splits]),dtype=torch.float64,device='cuda')
    sp=torch.tensor(np.repeat(selected,len(short),axis=0).T,dtype=torch.float64,device='cuda')
    lp=torch.tensor(np.repeat(selected,len(long),axis=0).T,dtype=torch.float64,device='cuda')
    marker_indices=list(engine.cable.marker_node_indices[1:]);original=[deepcopy(n.state_dict()) for n in networks]
    optimizers=[torch.optim.Adam([{'params':list(n.net.parameters()),'lr':settings['cable_learning_rate']*.25},
        {'params':list(n.correction_head.parameters()),'lr':settings['cable_learning_rate']}]) for n in networks]
    def selection():
        with torch.no_grad():
            q,v=cable_forward(engine,ld,lp)
            scores=(cable_objectives(q,ld['truth'],marker_indices).reshape(count,-1)*lw).sum(1)
            _,extra=engine.physics.motion_residual.components(q.reshape(-1,q.shape[2],3),v.reshape(-1,v.shape[2],3))
            # Magnitude regularization also excludes held-out windows.
            pen=extra.square().reshape(count,len(long),-1).mean(-1)/.5**2
            return (scores+.01*(pen*lw).sum(1)).cpu().numpy()
    best_scores=selection();best=[deepcopy(n.state_dict()) for n in networks];updates=[0]*count;history=[]
    for update in range(1,settings['cable_updates']+1):
        start=time.perf_counter()
        for opt in optimizers:opt.zero_grad()
        q,v=cable_forward(engine,sd,sp,gradients=True)
        scores=(cable_objectives(q,sd['truth'],marker_indices).reshape(count,-1)*sw).sum(1)
        _,extra=engine.physics.motion_residual.components(q.reshape(-1,q.shape[2],3),v.reshape(-1,v.shape[2],3))
        pen=extra.square().reshape(count,len(short),-1).mean(-1)/.5**2
        loss=scores+.01*(pen*sw).sum(1)
        for j,n in enumerate(networks):
            loss=loss+torch.nn.functional.one_hot(torch.tensor(j,device='cuda'),count)*(.01*torch.stack([(p-original[j][k]).square().mean() for k,p in n.named_parameters()]).mean())
        loss.sum().backward()
        norms=[float(torch.nn.utils.clip_grad_norm_(n.parameters(),1.,error_if_nonfinite=True)) for n in networks]
        for opt in optimizers:opt.step()
        row=dict(update=update,losses=loss.detach().cpu().numpy(),gradient_norms=norms,elapsed_s=time.perf_counter()-start)
        if update%6==0:
            values=selection();row['complete_window_selection']=values
            for j,value in enumerate(values):
                if value<best_scores[j]:best_scores[j]=value;best[j]=deepcopy(networks[j].state_dict());updates[j]=update
        history.append(row);save(out/'history.json',history)
        # Durable independent optimizer checkpoints permit safe continuation.
        torch.save(dict(update=update,networks=[n.state_dict() for n in networks],optimizers=[o.state_dict() for o in optimizers],
            best=best,best_scores=best_scores,best_updates=updates),out/'progress_checkpoint.pt')
        note(job,'six independent cable fits',**row)
    for net,state in zip(networks,best):net.load_state_dict(state);net.requires_grad_(False)
    with torch.no_grad():q,v=cable_forward(engine,ld,lp)
    objectives=cable_objectives(q,ld['truth'],marker_indices).reshape(count,-1).cpu().numpy()
    for j,(label,training,heldout) in enumerate(splits):
        folder=out/label;save_weights(folder/'cable_residual.pt',networks[j])
        np.savez_compressed(folder/'predictions.npz',positions=q[j*len(long):(j+1)*len(long)].cpu().numpy(),
            velocities=v[j*len(long):(j+1)*len(long)].cpu().numpy())
        save(folder/'result.json',dict(training=training,heldout=heldout,selected_physics=selected[j],selected_update=updates[j],
            selection_score=best_scores[j],per_window_objectives=objectives[j],specification=networks[j].specification()))
    save(out/'completed.json',dict(folds=count,updates=settings['cable_updates'],selected_updates=updates,shared_weights=False))
