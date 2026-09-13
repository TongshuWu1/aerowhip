"""Freeze an unselected preliminary-only M0 variant and compare coupled rollouts."""
from pathlib import Path
import sys,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experimental_data.current_adaptation import read,save
from experimental_data.io import sha256_file
from experimental_data.preliminary_prepare import PreliminaryTrial
from experimental_data.preliminary_fit import rms
from simulator.research_execution import ResearchExecutionModel


def main():
    torch.set_num_threads(4)
    source=ROOT/'runs/adaptation/20260909-preliminary1-M0-v2'
    out=ROOT/'runs/audits/preliminary1-gradient-resolution'
    candidate=out/'candidate';candidate.mkdir(exist_ok=False)
    shutil.copy2(__file__,out/'coupled_source.py')
    original=read(source/'candidate/model.json');model=read(source/'candidate/model.json')
    selection=read(out/'dynamic-mismatch/selection.json')
    for name in ('drone_model.json','drone_residual.pt'):
        shutil.copy2(source/'candidate'/name,candidate/name)
    model['cable'].update(curvature_frame_regularization=2e-5,external_drag_s_inv=selection['selected'])
    model['motion_residual']=dict(enabled=False,reason='Small physical baseline; no cable NN in this variant')
    model['provenance'].update(label='M0 development - preliminary1 - smooth damping',
        fit_complete=False,flight_ready=False,selected_model=False,
        source_model=str(source/'candidate/model.json'),source_sha256=sha256_file(source/'candidate/model.json'),
        fit_job=str(out),development_stage='Cable-only diagnostic; inherited drone fit, not a complete new fit',
        quality_note='Unselected preliminary hypothesis. Prospective whip data still required.',
        causal_cable_initialization=dict(history_s=1.,velocity_weight_tau_s=.02),
        numerical_change='Curvature-frame regularization 2e-7 -> 2e-5; explicitly changes regularized damping',
        scalar_damping='0.4 per second effective velocity damping selected on four training takes; not measured aerodynamics',
        prospective_flight_evidence=False,adaptation_generation='M0 development, not M1; no new whip data')
    save(candidate/'model.json',model)
    engines={'original_M0_weighted':ResearchExecutionModel.from_mapping(original,root=source/'candidate',device='cuda'),
        'development_M0_weighted':ResearchExecutionModel.from_mapping(model,root=candidate,device='cuda')}
    windows=read(source/'windows.json');base=read(source/'source_candidate/model.json')
    chosen=read(source/'candidate/coupled_diagnostics.json')['takes']
    report={}
    for take,row in chosen.items():
        w=next(w for w in windows if w['name']==row['fitted']['window'])
        t=PreliminaryTrial(source,w,base);t.cable_history_s=1.;t.cable_velocity_weight_tau_s=.02
        report[take]=dict(role=t.role,window=t.name,models={})
        previous_drone=None
        for label,e in engines.items():
            state,start,projection=t.cable_state(e.physics)
            times=start+np.arange(301)*e.dt_s
            measured,_,truth=t.measured(times)
            result=e.predict(t.initial_pose(e.drone.parameters),state,
                torch.tensor(t.data['packets'][None],device='cuda',dtype=torch.float64),
                t.data['packet_time'],times,graph=True,
                hover_command=torch.tensor(t.data['hover_commands'][-1:],device='cuda',dtype=torch.float64))
            q=result['cable_positions_m'][0].cpu().numpy();p=result['position_origin_m'][0].cpu().numpy()
            assert np.isfinite(q).all() and np.isfinite(p).all()
            if previous_drone is not None:np.testing.assert_array_equal(p,previous_drone)
            previous_drone=p
            report[take]['models'][label]=dict(drone=rms(p[1:],measured[1:]),tip=rms(q[1:,-1],truth[1:,-1]),projection_m=projection)
            np.savez_compressed(out/f'{take}-{label}-coupled.npz',time_s=times,
                predicted_origin=p,predicted_cable=q,measured_origin=measured,measured_sites=truth)
            print(take,label,report[take]['models'][label],flush=True)
        save(out/'coupled_diagnostics.json',dict(takes=report,
            evidence='Two-second command-driven pilot, same weighted initial state; no measured future attachment',
            unchanged_drone_prediction=True,selected_model=False))
    before=read(ROOT/'runs/audits/preliminary1-cable-only-pilot/protected_before.json')
    changed=[p for p,h in before.items() if sha256_file(p)!=h]
    save(out/'integrity.json',dict(protected_files=len(before),changed=changed));assert not changed
    save(out/'status.json',dict(status='completed',candidate=str(candidate/'model.json'),selected_model=False,
        residual_training=False,mppi_restarted=False,evidence='Preliminary development; M1 requires new reviewed real data'))


if __name__=='__main__':main()
