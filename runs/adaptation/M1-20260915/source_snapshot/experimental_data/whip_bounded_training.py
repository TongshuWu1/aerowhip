"""Explicit per-stage budgets with unchanged residual objective and verification."""
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
import time
import torch
from .plateau import Plateau
from .io import atomic_json
from .whip_full_optim import progress
from .whip_full_continuation import verify_selected_residual


def train_bounded(net,objective,contract,folder,job,label):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    settings=contract['residual_stopping'];budget=contract['stage_budgets'][label]
    started=time.perf_counter();optimizer=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=1e-4)
    score=float(objective().detach());baseline=score;best=deepcopy(net.state_dict());best_update=0
    stop=Plateau(settings['minimum'],settings['patience'],settings['relative']);stop.observe(0,score)
    history=[];reason='update_budget'
    torch.save(dict(update=0,loss=score,state_dict=best),folder/'best-000000.pt')
    for update in range(1,budget['maximum_updates']+1):
        progress(job,label,update=update,best_loss=stop.best,budget=budget)
        optimizer.zero_grad();loss=objective()
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite '+label)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        elapsed=time.perf_counter()-started;timed_out=elapsed>=budget['maximum_seconds']
        if update%settings['check_every']==0 or update==budget['maximum_updates'] or timed_out:
            with torch.no_grad():score=float(objective())
            improved,done=stop.observe(update,score)
            if improved:best=deepcopy(net.state_dict());best_update=update
            history.append(dict(update=update,loss=float(loss.detach()),selection_loss=score,best_loss=stop.best,
                baseline_loss=baseline,gradient_norm=float(norm),elapsed_s=time.perf_counter()-started))
            atomic_json(folder/'history.json',history)
            torch.save(dict(update=update,current=net.state_dict(),best=best,optimizer=optimizer.state_dict(),
                best_update=best_update,plateau=asdict(stop)),folder/'state.tmp')
            (folder/'state.tmp').replace(folder/'state.pt')
            if improved:torch.save(dict(update=update,loss=score,state_dict=best),folder/f'best-{update:06d}.pt')
            if done:reason='practical_plateau';break
            if timed_out:reason='wall_time_budget';break
    net.load_state_dict(best)
    result=dict(updates=update,selected_update=best_update,best_loss=stop.best,baseline_loss=baseline,
        stop_reason=reason,elapsed_s=time.perf_counter()-started,budget=budget,
        time_budget_scope='Training loop; final checkpoint numerical checks are additional',validation_used_for_selection=False)
    atomic_json(folder/'result.json',result)
    result=verify_selected_residual(net,objective,folder,result)
    net.requires_grad_(False)
    return result
