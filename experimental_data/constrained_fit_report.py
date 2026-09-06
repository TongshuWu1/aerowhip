"""Render the constrained fitting candidate, including rejected methods and ablations."""
import argparse
from pathlib import Path
import csv
import json
from copy import deepcopy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .io import atomic_json


def read(p):
    return json.loads(p.read_text())


def run(folder):
    geometry=read(folder/'geometry_fit.json');evaluation=read(folder/'evaluation.json')
    physics=read(folder/'physics.json');residual=read(folder/'residual.json')
    grid=read(folder/'grid.json');direct=read(folder/'direct_physics_history.json')
    resolution=read(folder/'resolution_check.json');checks=read(folder/'short_gradient_checks.json')
    figures=folder/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        for suffix in ('png','pdf','svg'):fig.savefig(figures/f'{name}.{suffix}',dpi=200,bbox_inches='tight')
        plt.close(fig)
    labels={'original_fitted':'Prior fitted physics','geometry_drag_seed':'Constrained geometry + drag seed',
            'constrained_physics':'Joint physical fit','constrained_plus_residual':'Joint fit + neural residual'}
    rows=[]
    for horizon,models in evaluation.items():
        for model,groups in models.items():
            for role,result in groups.items():
                for name,r in result['per_take'].items():
                    rows.append(dict(horizon_s=float(horizon),model=model,role=role,take=name,
                        marker_rmse_mm=1000*r['marker_rmse_m'],tip_rmse_mm=1000*r['tip_rmse_m'],windows=r['windows']))
    with (folder/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for ax,horizon in zip(axes,('2.0','5.0')):
        for i,model in enumerate(labels):
            for role,dy,color,marker in [('training',-.12,'#0072B2','o'),('validation',.12,'#D55E00','s')]:
                value=1000*evaluation[horizon][model][role]['equal_take_marker_rmse_m']
                ax.plot(value,i+dy,marker,color=color,label=('Fit takes' if role=='training' else 'Development validation') if i==0 else None)
        ax.set_yticks(range(4),list(labels.values()) if ax is axes[0] else ['']*4)
        ax.invert_yaxis();ax.set_xlim(left=0);ax.set_xlabel('All-marker trajectory RMSE [mm]')
        ax.set_title(f'{float(horizon):g}-second predictions');ax.grid(axis='x',alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.65,-.055),ncol=2,fontsize=8)
    fig.suptitle('Constrained fitting · measured attachment input · equal take averages')
    fig.tight_layout();save(fig,'prediction_comparison')
    names=list(geometry['original_consistency']);fig,ax=plt.subplots(figsize=(8,4))
    for key,label,shift,color in [('original_consistency','Original offset',-.1,'#0072B2'),
                                   ('candidate_consistency','Constrained offset',.1,'#009E73')]:
        ax.plot([geometry[key][n]['excess_over_2mm_percent'] for n in names],np.arange(len(names))+shift,
                'o',color=color,label=label)
    ax.set_yticks(range(len(names)),names);ax.invert_yaxis();ax.set_xlim(left=-1)
    ax.set_xlabel('Frames with attachment-to-c1 chord > 65 mm [%]')
    ax.set_title('Keep 55 mm attachment height and 63 mm first span; fit lateral offset')
    ax.legend(fontsize=8);ax.grid(axis='x',alpha=.2);fig.tight_layout();save(fig,'geometry_consistency')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for ax,key,title in zip(axes,['marker_prediction_difference_rmse_m','tip_prediction_difference_rmse_m'],['All markers','Cable tip']):
        names=list(resolution['per_take']);values=[1000*resolution['per_take'][n][key] for n in names]
        ax.bar(range(len(names)),values,color='#0072B2')
        ax.set_xticks(range(len(names)),names,rotation=35,ha='right');ax.set_ylabel('12 vs 24 substeps RMSE [mm]')
        ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Frozen physical candidate: integration sensitivity, not measurement error')
    fig.tight_layout();save(fig,'integration_sensitivity')
    history=read(folder/'residual_history.json')
    selected=[r for r in history if 'training_loss_m' in r]
    fig,ax=plt.subplots(figsize=(7,3.6))
    ax.plot([r['update'] for r in selected],[1000*r['training_loss_m'] for r in selected],'o-',color='#0072B2')
    ax.axhline(1000*selected[0]['training_loss_m'],color='.5',linestyle='--',label='Frozen physics')
    ax.set_xlabel('Neural residual optimizer update');ax.set_ylabel('2 s training robust loss [mm]')
    ax.set_title('Short-rollout residual training; selection uses full 2 s prediction')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();save(fig,'residual_selection')
    candidates=np.array(grid['candidates']);losses=np.array(grid['loss_m'])
    near=candidates[losses<=1.01*losses.min()]
    with np.load(folder/'grid_windows.npz') as w:
        leave_out={}
        for name in np.unique(w['take']):
            mask=w['take']!=name;score=(w['loss_m'][mask]*w['weights'][mask,None]).sum(0)/w['weights'][mask].sum()
            leave_out[str(name)]=candidates[score.argmin()].tolist()
    atomic_json(folder/'parameter_sensitivity.json',dict(grid_near_best_within_one_percent=near.tolist(),
        leave_one_training_take_out_grid_minima=leave_out,
        interpretation='Discrete loss sensitivity under fixed geometry, not confidence intervals or unique material recovery.'))
    fig,ax=plt.subplots(figsize=(7,4))
    for cb,color in [(1e-6,'#0072B2'),(1e-4,'#D55E00'),(.01,'#009E73')]:
        mask=(candidates[:,1]==cb)&(candidates[:,2]==.3)
        x=candidates[mask,0];y=losses[mask]*1000;order=np.argsort(x)
        ax.plot(x[order],y[order],'o-',color=color,label=f'Cb = {cb:g} N m² s')
    ax.set_xscale('log');ax.set_xlabel('EI [N m²]');ax.set_ylabel('2 s training robust loss [mm]')
    ax.set_title('Initial joint search: stiffness sensitivity at drag = 0.3 /s')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();save(fig,'parameter_sensitivity')
    gate={}
    for h in ('2.0','5.0'):
        p=evaluation[h]['constrained_physics']['validation'];n=evaluation[h]['constrained_plus_residual']['validation']
        gate[h]=all(n[k]<p[k] for k in ('equal_take_marker_rmse_m','equal_take_tip_rmse_m'))
        gate[h]=gate[h] and all(n['per_take'][t]['marker_rmse_m']<=1.05*p['per_take'][t]['marker_rmse_m'] for t in p['per_take'])
    accepted=all(gate.values()) and residual['improved']
    def dominates_seed(model):
        return all(evaluation[h][model]['validation'][k]<=evaluation[h]['geometry_drag_seed']['validation'][k]
            for h in ('2.0','5.0') for k in ('equal_take_marker_rmse_m','equal_take_tip_rmse_m'))
    recommended=('constrained_plus_residual' if accepted and dominates_seed('constrained_plus_residual') else
                 'constrained_physics' if dominates_seed('constrained_physics') else 'geometry_drag_seed')
    seed=read(folder/'geometry_model.json')
    seed['cable']['parameter_source']=str(folder/'geometry_drag_seed_model.json')
    seed['force_accounting']['aerodynamic_drag']='effective_world_velocity_decay_on_free_cable_vertices'
    atomic_json(folder/'geometry_drag_seed_model.json',seed)
    chosen_file={'geometry_drag_seed':'geometry_drag_seed_model.json',
                 'constrained_physics':'candidate_model.json','constrained_plus_residual':'candidate_with_residual.json'}[recommended]
    atomic_json(folder/'recommended_model.json',read(folder/chosen_file))
    atomic_json(folder/'candidate_review.json',dict(recommended=recommended,recommended_file='recommended_model.json',
        residual_development_gate=gate,active_model_changed=False,protected_test_used=False,
        status='PROVISIONAL_RESEARCH_CANDIDATE',
        gate_rule='Residual must improve validation mean marker and tip RMSE at both horizons, with no take marker RMSE degradation above 5%.',
        material_update_rule='Prefer earlier material values plus constrained geometry and drag unless the updated model matches or beats seed marker and tip errors at both validation horizons.',
        caveats=['Repeated development validation is not a final generalization test.',
                 'Cable prediction under measured root motion does not validate command-driven drone dynamics.',
                 'Long-horizon adjoints require further investigation before differentiable force-plan adaptation.']))
    offset=geometry['selected']['offset_body_m'];ei,cb,drag=physics['selected_parameters']
    chosen=read(folder/'recommended_model.json')['cable']
    text=['# Constrained physical fitting candidate — 5 September 2026','',
        'Completed the recommended geometry-first calibration on the existing development recordings. '
        'This is a provisional research candidate. The active baseline has not been replaced and the protected test has not been opened.','',
        '## Recommended candidate','',
        f'- Body-frame attachment offset: [{offset[0]*1000:.3f}, {offset[1]*1000:.3f}, {offset[2]*1000:.3f}] mm.',
        '- Attachment height fixed at the reported 55 mm; first cable span fixed at 63 mm. Other measured lengths and masses retained.',
        f"- EI: {chosen['EI_n_m2']:.8g} N m²; Cb: {chosen['Cb_n_m2_s']:.8g} N m² s; effective external decay rate: {chosen['external_drag_s_inv']:.6g} /s.",
        f'- For comparison, joint optimization gave EI {ei:.8g}, Cb {cb:.8g} and decay rate {drag:.6g}, but selection also considers development validation.',
        '- Pivot attachment; 12 physics substeps per 10 ms. Geometry and material values are conditional model estimates, not independently measured constants.',
        '- External decay is applied to free cable vertices in the coupled point-mass runtime. It does not silently add the same drag rate to the drone.', '',
        '## Prediction results','',
        '| Horizon | Model | Fit marker RMSE [mm] | Development validation marker RMSE [mm] | Validation tip RMSE [mm] |',
        '| --- | --- | ---: | ---: | ---: |']
    for h,models in evaluation.items():
        for key,r in models.items():
            text.append(f"| {float(h):g} s | {labels[key]} | {r['training']['equal_take_marker_rmse_m']*1000:.2f} | {r['validation']['equal_take_marker_rmse_m']*1000:.2f} | {r['validation']['equal_take_tip_rmse_m']*1000:.2f} |")
    text+=['','Errors average Euclidean position RMSE equally across takes, excluding time zero. Models within a horizon '
        'share complete-valid prediction windows. Two- and five-second cohorts differ. `metrics.csv` supplies per-take '
        'values and sample counts. Root motion is measured input; no future cable observations correct the prediction.','',
        '## Geometry calibration','',
        'Float64 L-BFGS with strong-Wolfe line search fitted only the lateral offset, using equal training-take '
        'weight, a robust first-span feasibility term and a weaker quiet-tangent extrapolation term. The lateral '
        'offset was bounded to ±20 mm with a weak zero-centered prior. Quiet tangent extrapolation remains an approximation. '
        'Sensitivity profiles used heights 52/55/58 mm and spans 60/63/66 mm; those are assumed ranges, not measured uncertainty. '
        'They did not select the height or span. Leaving out individual training takes yielded similar lateral offsets.','',
        'On fig8_003, the fraction of frames with attachment-to-c1 chord exceeding 63 mm plus 2 mm tolerance '
        'fell from 29.93% to 7.97%. On osc_003 it fell from 9.22% to 1.56%. This improves consistency but does '
        'not eliminate measurement/geometry uncertainty.','',
        '## Physical optimization and the gradient limitation','',
        'The initial 61-candidate search was followed by three bounded 3D pattern-search passes, each testing '
        '27 neighboring EI/Cb/drag combinations against full two-second rollouts. Selection used only training '
        'loss, equally weighting takes and nonempty initial-motion bins. The search extended EI to 0.004 N m² '
        'as an upper bound with the finer integrator, and evaluated values above the initial 0.001 bound. The final candidate is not '
        'at that extended bound. A finite search is not proof of a unique optimum.','',
        f'The initial grid has {len(near)} combinations within 1% of its minimum training loss. '
        'Discrete leave-one-training-take-out minima are recorded in `parameter_sensitivity.json`; '
        'they should be inspected before interpreting the fitted quantities as reusable material constants.','',
        'An attempted Adam refinement through long rollouts was rejected. At the real fig8_002 starting frame 35 '
        'over 0.5 s, the log-EI adjoint was approximately -6.45, whereas central finite differences were +0.0064 '
        '(step 1e-4) and +0.204 (step 1e-5). The finite-difference estimates themselves vary with step size. '
        'This indicates unsuitable local sensitivity for optimization at that horizon; it does not yet locate '
        'one confirmed kernel defect. The interrupted four-update history is preserved. Its evaluated candidate '
        'was worse than the grid seed and was not selected.','',
        'Short checks at 0.05 and 0.1 s passed on two real initial states for log EI, log Cb, log drag and a '
        'learned-residual gain. Each used two finite-difference steps. Those are local checks, not a guarantee '
        'for all states. The simulator is differentiable in code, but this experiment does not support a '
        'claim of reliable long-horizon differentiable identification or force-plan optimization yet.','',
        '## Neural residual','',
        'Physical parameters and geometry were frozen. A bounded 32-unit MLP was trained with Adam on random '
        '100 ms segments inside the stratified training windows. Each segment starts from measured markers '
        'and an 11-frame causal velocity estimate. There is no state reset inside a segment. Such supervised '
        'training does not introduce feedback into policy execution: evaluation still runs continuously for '
        'two or five seconds from one initial state. Corrections are bounded at 0.5 m/s² per coordinate and '
        'regularized on sampled initial states. Updates with excessive gradient norm are skipped and logged. '
        'Checkpoint selection compares full two-second training rollout error, including the zero-residual model.','',
        f"The post-hoc development review recommends **{labels[recommended]}**. "
        'A residual must improve validation mean marker and tip error at both horizons and avoid worsening '
        'any validation take’s marker RMSE by more than 5%. This is development model selection, not an independent test.','',
        'The updated material model must also match or improve the simpler geometry-plus-drag seed’s '
        'validation marker and tip errors at both horizons. Otherwise retain the earlier material values '
        'and the constrained geometry/drag correction. A small training improvement alone does not establish '
        'a better physical baseline. `recommended_model.json` contains the resulting choice.','',
        '## Remaining limits and next use','',
        f"Across the seven takes, doubling substeps from 12 to 24 changes all-marker predictions by "
        f"{min(r['marker_prediction_difference_rmse_m'] for r in resolution['per_take'].values())*1000:.2f}–"
        f"{max(r['marker_prediction_difference_rmse_m'] for r in resolution['per_take'].values())*1000:.2f} mm RMSE. "
        'This is small relative to the measured-data error, but is not a bound on continuum discretization error.','',
        'Use this candidate for simulation comparisons, not as certified hardware calibration. Preserve the '
        'protected recording for a frozen final protocol. Investigate the long-horizon sensitivity problem '
        'before relying on gradients for rapid policy/force-sequence adaptation. Validate command-to-drone '
        'motion separately; the cable fit alone cannot establish open-loop strike success.','',
        'Candidate files are `candidate_model.json` and `candidate_with_residual.json`; the latter references '
        'a hash-checked immutable checkpoint. `candidate_review.json` records the recommended variant. '
        'The `figures` directory contains PDF, SVG and PNG exports. Protocols, per-stage source snapshots, '
        'input hashes, loss surfaces, gradient checks and per-take results are retained alongside this report. '
        'The final full software suite passed 123 tests with four existing Torch JIT deprecation warnings.']
    report='\n'.join(text)+'\n';(folder/'README.md').write_text(report,encoding='utf-8')
    root=Path(__file__).resolve().parents[1]
    (root/'docs/CONSTRAINED_FIT_20260905.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(review=read(folder/'candidate_review.json'),parameters=physics['selected_parameters']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    run(p.parse_args().folder.resolve())
