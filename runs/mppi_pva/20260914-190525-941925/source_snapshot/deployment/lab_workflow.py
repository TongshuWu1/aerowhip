"""Portable, explicit lab-day workflow around the reviewed research implementation.

Importing or reading this module never imports Torch, fits, plans, or flies. Heavy
operations are explicit methods run by tools/lab.py in a separate process.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import re
import shutil

SCHEMA = 'deployment_lab_study_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')


def safe_name(name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
        raise ValueError('Use a short name containing letters, digits, hyphens or underscores.')
    if name.upper() in {'CON', 'PRN', 'AUX', 'NUL', *('COM'+str(i) for i in range(1,10)), *('LPT'+str(i) for i in range(1,10))}:
        raise ValueError('This name is reserved on Windows.')
    if 'fig8vertical_002' in name.lower(): raise ValueError('Protected recording is excluded.')
    return name


def planned_slots():
    slots = []
    for generation in ('M0', 'M1'):
        for i in range(1, 6):
            slots.append(dict(stage=generation, take=f'whip_{i:03}', generation=generation,
                              role='adaptation' if i in (1,2,4) else 'validation', status='pending', review={}))
    pair_order=(('M2','M0'),('M0','M2'),('M0','M2'),('M2','M0'),('M0','M2'))
    for i,order in enumerate(pair_order,1):
        for generation in order:
            slots.append(dict(stage='final', take=f'pair_{i:02}_{generation}', generation=generation,
                              pair=i, role='final', status='pending', review={}))
    return slots


def portable_outputs(root, paths):
    """Publish newly generated metadata with relative paths and dependent hashes.

    Only the supplied new output trees are rewritten. Raw CSV/NPZ/checkpoint
    bytes and source snapshots are untouched. Hash references are rebuilt in
    dependency order before these outputs become frozen study inputs.
    """
    root = Path(root).resolve(); files = set()
    for path in paths:
        path = Path(path).resolve()
        if not path.is_relative_to(root): raise ValueError('Output must stay inside the repository.')
        candidates = [path] if path.is_file() else path.rglob('*.json')
        files.update(p for p in candidates if p.suffix == '.json' and 'source_snapshot' not in p.parts and 'history' not in p.parts)
    values = {}
    prefixes = (str(root) + os.sep, root.as_posix() + '/')
    def convert(value, file, key=None):
        if isinstance(value, dict):
            return {convert(k,file):convert(v,file,k) for k,v in value.items()}
        if isinstance(value, list): return [convert(v,file) for v in value]
        if isinstance(value, str):
            for prefix in prefixes:
                if value.startswith(prefix):
                    local = Path(value)
                    if key == 'checkpoint': return Path(os.path.relpath(local, file.parent)).as_posix()
                    return local.relative_to(root).as_posix()
        return value
    for file in files: values[file] = convert(read(file), file)
    done = set(); visiting = set()
    def refresh(file):
        if file in done: return
        if file in visiting: raise ValueError('Cyclic generated evidence hashes: ' + str(file))
        visiting.add(file)
        def hashed(path):
            path = path.resolve()
            if path in values: refresh(path)
            return digest(path)
        def walk(value):
            if isinstance(value,list): return [walk(v) for v in value]
            if not isinstance(value,dict): return value
            out = {k:walk(v) for k,v in value.items()}
            for k,v in list(out.items()):
                if isinstance(v,str) and re.fullmatch(r'[0-9a-f]{64}',v):
                    candidate = root / k
                    if candidate.is_file() and candidate.resolve()!=file: out[k] = hashed(candidate)
            if isinstance(out.get('checkpoint'),str) and out.get('sha256'):
                asset = file.parent/out['checkpoint']
                if asset.is_file(): out['sha256'] = hashed(asset)
            for hash_key, path_key in [('report_sha256','comparison'),('model_source_sha256','model_source'),
                                       ('source_checkpoint_sha256','source_checkpoint')]:
                if out.get(hash_key) and isinstance(out.get(path_key),str):
                    target=root/out[path_key]
                    if hash_key=='report_sha256': target=target/'report.json'
                    if target.is_file(): out[hash_key]=hashed(target)
            if out.get('candidate_sha256') and file.name=='selection_frozen.json':
                out['candidate_sha256']=hashed(file.parent.parent/'candidate/model.json')
            return out
        write(file,walk(values[file])); visiting.remove(file); done.add(file)
    for file in sorted(files): refresh(file)


class LabWorkspace:
    def __init__(self, root, study=None):
        self.root = Path(root).resolve()
        self.name = safe_name(study) if study else None

    def path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root): raise ValueError('Path leaves the repository.')
        return path

    def rel(self, path): return Path(path).resolve().relative_to(self.root).as_posix()

    @contextmanager
    def runtime(self):
        previous = Path.cwd(); os.chdir(self.root)
        try: yield
        finally: os.chdir(previous)

    @property
    def directory(self):
        if not self.name: raise ValueError('Choose or create a study first.')
        return self.root/'experiments'/self.name

    def _load(self):
        file=self.directory/'study.json'
        if not file.is_file(): raise ValueError('Create the study first.')
        value=read(file)
        if value.get('schema')!=SCHEMA: raise ValueError('Unsupported study format.')
        if [(s['stage'],s['take'],s['generation'],s['role']) for s in value['slots']] != [(s['stage'],s['take'],s['generation'],s['role']) for s in planned_slots()]:
            raise ValueError('The frozen whole-take allocation changed.')
        return value

    def _save(self, state):
        file=self.directory/'study.json'
        if file.exists():
            history=self.directory/'history'/f'{stamp()}-{digest(file)[:10]}.json'
            history.parent.mkdir(exist_ok=True); shutil.copy2(file,history)
        write(file,state)

    def _slot(self,state,stage,take):
        slot=next((s for s in state['slots'] if s['stage']==stage and s['take']==take),None)
        if slot is None: raise ValueError('Choose one of the predeclared take slots.')
        return slot

    def _verify(self, hashes):
        for path,expected in hashes.items():
            if digest(self.path(path))!=expected: raise ValueError('Frozen input changed: '+path)

    def overview(self):
        studies=sorted(p.parent.name for p in (self.root/'experiments').glob('*/study.json'))
        selected=self.name or (studies[-1] if studies else None)
        state=LabWorkspace(self.root,selected)._load() if selected else None
        manifest=self.root/'workspace/baseline/manifest.json'
        baseline=read(manifest) if manifest.is_file() else None
        if baseline: baseline={k:v for k,v in baseline.items() if k!='files'}
        result=dict(studies=studies,study=state,baseline=baseline,
                    slots=state['slots'] if state else [],models=state['models'] if state else {})
        if state:
            for generation,update in state.get('updates',{}).items():
                status=self.path(update['job'])/'status.json'
                if status.is_file(): update.update(read(status))
        return result

    def create(self,name=None,*,planner_profile='config/pva/systematic_strike.json'):
        from deployment.lab_seed import load_baseline,model_identity
        if name: self.name=safe_name(name)
        if self.directory.exists(): raise ValueError('Study already exists; choose it to continue.')
        with self.runtime():
            baseline=load_baseline(self.root)
            model=self.path(baseline['m0_model'])
            signature=model_identity(model)
            state=dict(schema=SCHEMA,name=self.name,created_at=datetime.now(timezone.utc).isoformat(),
                       measurement=dict(metric='minimum observed 3D tip-to-target distance',window_s=[0.,1.5],
                           aggregation='Equal take weight; five paired final comparisons; retain exclusions and missing coverage'),
                       slots=planned_slots(),models={'M0':dict(model=self.rel(model),signature=signature,
                           rehearsal=baseline['m0_rehearsal'],status='Retained frozen M0')},updates={},results=None)
            if planner_profile is not None:
                from planning.pva_job import validate_settings
                from planning.strike_objective import uses_templates
                settings=self.path(planner_profile); cfg=read(settings);validate_settings(cfg)
                templates=(self.path(cfg['spline_seed_directory']) if cfg.get('spline_seed_directory') else self.path(baseline['mppi_settings']).parent)/'proposal_baselines.npz'
                if uses_templates(cfg) and not templates.is_file():raise FileNotFoundError('Frozen command templates are missing')
                frozen=self.directory/'planner';frozen.mkdir(parents=True)
                shutil.copy2(settings,frozen/'settings.json')
                if uses_templates(cfg):shutil.copy2(templates,frozen/'proposal_baselines.npz')
                if cfg.get('spline_seed_directory'):
                    seed_root=self.path(cfg['spline_seed_directory'])
                    for filename in ('initial_proposal.npz',cfg['trajectory_objective']['reference_file']):shutil.copy2(seed_root/filename,frozen/filename)
                    cfg['spline_seed_directory']=self.rel(frozen);write(frozen/'settings.json',cfg)
                state['planner_profile']=self.rel(frozen/'settings.json')
                state['planner_hashes']={self.rel(p):digest(p) for p in frozen.iterdir()}
                state['models']['M0'].pop('rehearsal')
                state['models']['M0']['status']='Retained M0 model; new targeted-strike plan required'
            self._save(state)
            from experimental_data.model_evaluation import load_catalog,save_catalog,model_identity as identity
            catalog=load_catalog(self.root)
            signature,hashes=identity(model)
            training=[]
            preliminary=self.path(baseline['preliminary'])
            for take in baseline.get('preliminary_training_takes',[]):
                provenance=read(preliminary/'inputs'/take/'provenance.json')
                training.extend(provenance['source_hashes'].values())
            catalog['models'].append(dict(id=self.name+'-M0',parent=None,generation_index=0,model=self.rel(model),
                signature=signature,hashes={self.rel(p):h for p,h in hashes.items()},training_sources=sorted(set(training)),job=None,status='Retained M0'))
            save_catalog(self.root,catalog)
            return state

    def _assert_model(self,state,generation):
        from deployment.lab_seed import model_identity
        if generation not in state['models']: raise ValueError('Complete the previous model update first.')
        item=state['models'][generation]
        if model_identity(self.path(item['model']))!=item['signature']: raise ValueError('Frozen model changed.')
        return item

    def plan(self,generation,device='cuda'):
        with self.runtime():
            state=self._load(); item=self._assert_model(state,generation)
            if item.get('export'): raise ValueError('This generation is frozen for collection; its command cannot be replanned.')
            from deployment.lab_seed import load_baseline
            from planning.pva_job import prepare,run
            baseline=load_baseline(self.root)
            self._verify(state.get('planner_hashes',{}))
            settings=self.path(state.get('planner_profile',baseline['mppi_settings'])); cfg=read(settings)
            cfg.update(model_path=item['model'],device=device)
            job,_=prepare(self.root,cfg,self.name+'-'+generation,development_review='Frozen lab study offline planning; physical execution remains external.')
            for source in (settings.parent.glob('*.npz') if cfg.get('mppi',{}).get('initialization')!='from_scratch' else []):
                if not (job/source.name).exists(): shutil.copy2(source,job/source.name)
            item.update(plan=self.rel(job),status='Planning'); item.pop('rehearsal',None); self._save(state)
            try: run(job)
            finally:
                portable_outputs(self.root,[job])
                state=self._load();state['models'][generation]['status']=read(job/'status.json').get('status','failed');self._save(state)
            return state['models'][generation]

    def export(self,generation,device='cuda'):
        with self.runtime():
            state=self._load();item=self._assert_model(state,generation)
            self._verify(state.get('planner_hashes',{}))
            if item.get('export'):
                self._verify(item['export_hashes']);return item
            if not item.get('rehearsal'):
                if not item.get('plan'): raise ValueError('Plan this generation first.')
                from deployment.pva_rehearsal import generate
                job=self.path(item['plan'])
                if read(job/'status.json').get('status')!='completed': raise ValueError('Only a completed plan can be exported.')
                if state.get('planner_profile'):
                    frozen=read(self.path(state['planner_profile'])); planned=read(job/'settings.json')
                    for key in ('trajectory_objective','fold_constraint','mppi','task','launch','action','limits'):
                        if frozen.get(key)!=planned.get(key):raise ValueError('Plan differs from the frozen study settings: '+key)
                    for key in ('command_contract','fold_requirement','recovery'):
                        if frozen.get(key)!=planned.get(key):raise ValueError('Plan differs from the frozen study settings: '+key)
                rehearsal=self.directory/'rehearsals'/generation
                generate(job,rehearsal,device=device)
                portable_outputs(self.root,[rehearsal]);item['rehearsal']=self.rel(rehearsal)
            rehearsal=self.path(item['rehearsal'])
            from deployment.lab_seed import model_identity
            if model_identity(rehearsal/'model.json')!=item['signature']: raise ValueError('Rehearsal belongs to another model.')
            metadata=read(rehearsal/'rehearsal.json')
            if state.get('planner_profile'):
                frozen=read(self.path(state['planner_profile']))
                requirement=frozen.get('fold_requirement','required' if frozen.get('trajectory_objective',{}).get('schema')=='targeted_fold_strike_v1' else None)
                if requirement is not None and metadata.get('fold_requirement','required')!=requirement:
                    raise ValueError('Rehearsal fold requirement differs from the frozen study')
                if requirement=='required' and metadata.get('predicted_fold_valid') is not True:
                    raise ValueError('Study requires a travelling-fold rehearsal')
            if digest(rehearsal/'fullstate_30hz.csv')!=metadata['csv_sha256']: raise ValueError('Rehearsal CSV changed.')
            output=self.root/'exports'/self.name/generation
            output.mkdir(parents=True,exist_ok=False)
            csv_path=output/'fullstate_30hz.csv';shutil.copy2(rehearsal/'fullstate_30hz.csv',csv_path)
            write(output/'manifest.json',dict(schema='deployment_csv_export_v1',study=self.name,generation=generation,
                model_signature=item['signature'],csv_sha256=digest(csv_path),rehearsal=item['rehearsal'],
                forecast_sha256=digest(rehearsal/'rehearsal.npz'),command_rate_hz=30,flight_sender=False,
                meaning='Complete open-loop command with recovery; external flight program consumes this CSV.'))
            item.update(export=self.rel(csv_path),export_hashes={self.rel(p):digest(p) for p in [csv_path,output/'manifest.json',rehearsal/'rehearsal.npz',rehearsal/'model.json']},status='CSV frozen')
            self._save(state);return item

    def _batch(self,state,stage,generation):
        return self.directory/'batches'/(stage if stage!='final' else 'final-'+generation)

    def _ensure_batch(self,state,stage,generation):
        from experimental_data.whip_adaptation import setup
        item=self._assert_model(state,generation)
        if not item.get('export'): raise ValueError('Export the frozen command before importing flights.')
        self._verify(item['export_hashes']);batch=self._batch(state,stage,generation)
        if batch.exists(): return batch
        package=self.path(item['export']).parent
        # Production setup verifies package/files and the exact original forecast.
        package_manifest=read(package/'manifest.json')
        package_manifest['files']={'fullstate_30hz.csv':digest(self.path(item['export']))}
        # This manifest is metadata owned by the export; do not change its published bytes.
        binding=self.directory/'bindings'/(stage+'-'+generation);binding.mkdir(parents=True,exist_ok=False)
        shutil.copy2(self.path(item['export']),binding/'fullstate_30hz.csv')
        write(binding/'manifest.json',package_manifest)
        selection=binding/'selection.json'
        write(selection,dict(rehearsal=item['rehearsal'],package=self.rel(binding),command_sha256=digest(binding/'fullstate_30hz.csv'),forecast_sha256=digest(self.path(item['rehearsal'])/'rehearsal.npz')))
        setup(self.root,batch,selection)
        protocol=read(batch/'protocol.json')
        protocol.update(model_id=self.name+'-'+generation,planned_roles={s['take']:('validation' if stage=='final' else s['role']) for s in state['slots'] if s['stage']==stage and s['generation']==generation},
                        model_update='Staged full-model update' if stage!='final' else 'Final diagnostics only; never fitting',neural_training=stage!='final')
        write(batch/'protocol.json',protocol);portable_outputs(self.root,[batch,binding]);return batch

    def import_take(self,stage,take,tracking,controller,*,offset_s,clock_source,clock_verified=False,drone_label=None):
        if not math.isfinite(offset_s) or not clock_source.strip(): raise ValueError('Provide a finite clock offset and its timestamp/shared-event source.')
        for source in (tracking,controller):
            if 'fig8vertical_002' in str(source).lower(): raise ValueError('Protected recording is excluded.')
            if not Path(source).is_file() or Path(source).suffix.lower()!='.csv': raise ValueError('Choose an existing CSV file for each recording.')
        with self.runtime():
            state=self._load();slot=self._slot(state,stage,take)
            if slot['status']!='pending': raise ValueError('This slot already owns immutable recordings. No files were replaced.')
            if stage=='M1' and 'M1' not in state['models']: raise ValueError('Fit M1 before collecting its flights.')
            if stage=='final' and 'M2' not in state['models']: raise ValueError('Freeze M2 before final collection.')
            tracking_hash=digest(tracking);controller_hash=digest(controller)
            if any(tracking_hash in s.get('raw_hashes',{}).values() or controller_hash in s.get('raw_hashes',{}).values() for s in state['slots']):
                raise ValueError('These raw bytes already belong to another slot; repeated analyses are not new flights.')
            batch=self._ensure_batch(state,stage,slot['generation']);folder=batch/'flight_take'
            destinations=[folder/(take+'.csv'),folder/('experiment_'+take+'.csv')]
            if any(p.exists() for p in destinations): raise ValueError('An earlier incomplete import owns these files; they were preserved. Use a new study or inspect the interrupted import.')
            for source,destination in zip((tracking,controller),destinations): shutil.copy2(source,destination)
            if digest(destinations[0])!=tracking_hash or digest(destinations[1])!=controller_hash: raise ValueError('Source changed while copying; keep these files for inspection.')
            if drone_label: write(destinations[0].with_suffix('.tracking.json'),dict(drone=drone_label,optitrack_sha256=tracking_hash))
            align=read(batch/'time_alignment.json');align[take]=dict(offset_s=float(offset_s),source=clock_source.strip(),clock_verified=bool(clock_verified),optitrack_sha256=tracking_hash,controller_sha256=controller_hash);write(batch/'time_alignment.json',align)
            slot.update(status='imported',tracking=self.rel(destinations[0]),controller=self.rel(destinations[1]),batch=self.rel(batch),offset_s=float(offset_s),clock_source=clock_source,clock_verified=bool(clock_verified),raw_hashes={self.rel(destinations[0]):tracking_hash,self.rel(destinations[1]):controller_hash})
            self._save(state)
            # Original forecast diagnostics are computed without new simulation.
            from experimental_data.whip_adaptation import compare
            output=self.directory/'comparisons'/(stage+'-'+stamp())
            try:
                compare(self.root,batch,output);portable_outputs(self.root,[output]);slot['comparison']=self.rel(output)
                self._save(state)
            except Exception as exc:
                slot['import_error']=str(exc);self._save(state)
                raise ValueError('Raw files were preserved, but comparison needs attention: '+str(exc)) from exc
            return slot

    def estimate_timing(self,tracking,controller,drone_label=None):
        """Read-only existing measured-stream estimate, requiring operator review."""
        import numpy as np
        from experimental_data.adaptation_rounds import read_optitrack,read_controller
        from experimental_data.preliminary_prepare import measured_clock_alignment
        paths=[Path(tracking).resolve(),Path(controller).resolve()]
        for path in paths:
            if 'fig8vertical_002' in str(path).lower(): raise ValueError('Protected recording is excluded.')
            if not path.is_file() or path.suffix.lower()!='.csv': raise ValueError('Choose two existing raw CSV recordings.')
        hashes=[digest(p) for p in paths]
        measured=read_optitrack(paths[0],drone_label=drone_label,allow_external_filename=True)
        logged=read_controller(paths[1],allow_external_filename=True)
        valid=np.isfinite(measured['time']) & np.isfinite(measured['drone']).all(1)
        if valid.sum()<30: raise ValueError('Insufficient finite measured positions for timing estimation.')
        # Filtering is local to this timing diagnostic. The imported recording,
        # full trajectory and its native missingness masks remain untouched.
        timing_input={'time':np.asarray(measured['time'])[valid].copy(),'drone':np.asarray(measured['drone'])[valid].copy()}
        result=measured_clock_alignment(timing_input,logged)
        if [digest(p) for p in paths]!=hashes: raise ValueError('A raw file changed while timing was estimated.')
        chunks=[c['offset_s'] for c in result.get('chunks',[])]
        result.update(clock_verified=False,requires_operator_review=True,
            chunk_spread_s=float(max(chunks)-min(chunks)) if chunks else None,
            discarded_tracking_samples=int((~valid).sum()),
            maximum_retained_tracking_gap_s=float(np.diff(timing_input['time']).max()),
            optitrack_sha256=hashes[0],controller_sha256=hashes[1])
        if not valid.all(): result['limitation']+=' Nonfinite positions were excluded only from this timing estimate; review gaps and chunk spread.'
        return result

    def review_take(self,stage,take,*,reviewer,free_motion_end_s=None,physical_contact='none',clock_reviewed=False,same_hardware=False,no_intervention=False,notes='',exclude=False):
        state=self._load();slot=self._slot(state,stage,take)
        if slot['status']=='pending': raise ValueError('Import the recording pair first.')
        self._verify(slot['raw_hashes'])
        if any(u.get('source_stage')==stage for u in state['updates'].values()) or (state.get('results') or {}).get('predictions_complete'):
            raise ValueError('This review is frozen in prepared evidence and cannot be changed.')
        if not reviewer.strip(): raise ValueError('Name the person reviewing this take.')
        if exclude:
            if not notes.strip(): raise ValueError('Record the reason for exclusion; raw data will remain.')
        elif not all((clock_reviewed,same_hardware,no_intervention)):
            raise ValueError('Review timing, unchanged hardware/controller and absence of intervention.')
        elif physical_contact not in ('none','at_or_after_end') or free_motion_end_s is None or not math.isfinite(free_motion_end_s) or free_motion_end_s<=0:
            raise ValueError('Enter the observed free-motion interval and contact status.')
        slot['review']=dict(role='excluded' if exclude else ('validation' if stage=='final' else slot['role']),accepted=not exclude,
            reviewed_by=reviewer.strip(),clock_reviewed=bool(clock_reviewed),same_controller_and_hardware=bool(same_hardware),no_intervention=bool(no_intervention),
            physical_contact=physical_contact,free_motion_end_s=free_motion_end_s,notes=notes,reviewed_at=datetime.now(timezone.utc).isoformat())
        slot.update(status='reviewed',excluded=bool(exclude));self._save(state);return slot

    def align_take(self,stage,take,*,offset_s,clock_source,clock_verified=False):
        """Correct timing metadata before freezing; retain each previous revision."""
        if not math.isfinite(offset_s) or not clock_source.strip(): raise ValueError('Enter a finite offset and its timestamp/shared-event source.')
        with self.runtime():
            state=self._load();slot=self._slot(state,stage,take)
            if slot['status']=='pending': raise ValueError('Import the pair first.')
            if any(u.get('source_stage')==stage for u in state['updates'].values()) or (state.get('results') or {}).get('predictions_complete'):
                raise ValueError('Timing is frozen in prepared evidence.')
            self._verify(slot['raw_hashes']);batch=self.path(slot['batch']);file=batch/'time_alignment.json'
            history=batch/'alignment_history'/stamp();history.mkdir(parents=True);shutil.copy2(file,history/'time_alignment.json')
            align=read(file);align[take].update(offset_s=float(offset_s),source=clock_source.strip(),clock_verified=bool(clock_verified));write(file,align)
            slot.update(offset_s=float(offset_s),clock_source=clock_source.strip(),clock_verified=bool(clock_verified),status='imported',review={});slot.pop('excluded',None)
            slot.pop('import_error',None);self._save(state)
            from experimental_data.whip_adaptation import compare
            output=self.directory/'comparisons'/(stage+'-'+stamp())
            try:
                compare(self.root,batch,output);portable_outputs(self.root,[output]);slot['comparison']=self.rel(output);self._save(state)
            except Exception as exc:
                slot['import_error']=str(exc);self._save(state);raise
            return slot

    def _prepare_records(self,state,stage,generation,output,*,diagnostics_only=False):
        from experimental_data import whip_adaptation as data
        slots=[s for s in state['slots'] if s['stage']==stage and s['generation']==generation]
        if not slots or any(s['status']!='reviewed' for s in slots): raise ValueError('Import and review every planned take, retaining explicit exclusions.')
        for slot in slots: self._verify(slot['raw_hashes'])
        batch=self._batch(state,stage,generation)
        comparison=self.directory/'comparisons'/(stage+'-'+generation+'-'+stamp())
        data.compare(self.root,batch,comparison)
        review=read(comparison/'review.template.json');review['takes']={s['take']:deepcopy(s['review']) for s in slots}
        if diagnostics_only:
            for name,item in review['takes'].items():
                if item['role']=='excluded': continue
                if item['free_motion_end_s']<state['measurement']['window_s'][1]:
                    raise ValueError(name+': final diagnostics require reviewed free motion through 1.5 s, or an explicit exclusion.')
                item['free_motion_end_s']=state['measurement']['window_s'][1]
        write(comparison/'review.json',review)
        data.prepare(self.root,batch,comparison,comparison/'review.json',output,full_model=True,diagnostics_only=diagnostics_only)
        portable_outputs(self.root,[comparison,output]);return comparison

    def prepare_update(self,generation,*,retry=False):
        if generation not in ('M1','M2'): raise ValueError('Only M1 and M2 are fitted; retained M0 is never refitted.')
        with self.runtime():
            state=self._load();parent='M0' if generation=='M1' else 'M1';self._assert_model(state,parent)
            previous=state['updates'].get(generation)
            if previous and not retry: raise ValueError('Update inputs are already frozen. Run the prepared update explicitly, or explicitly prepare a new attempt after a failed/stopped fit.')
            if retry and not previous: raise ValueError('There is no earlier update to retry.')
            if retry:
                old_job=self.path(previous['job'])
                if read(old_job/'status.json').get('status') not in ('failed','stopped'): raise ValueError('Only a failed or stopped fit can have a new attempt prepared.')
                if generation in state['models']: raise ValueError('This generation already has a frozen model.')
                from experimental_data.model_evaluation import load_catalog
                if any(m['id']==self.name+'-'+generation for m in load_catalog(self.root)['models']):
                    raise ValueError('A completed candidate is already registered; its original result must be retained.')
                for name in ('prepared_hashes.json','source_hashes.json'): self._verify(read(old_job/name))
            if any(s['status']!='pending' for s in state['slots'] if s['stage']=='final'): raise ValueError('Final collection has started; model updates are frozen.')
            source=self.path(previous['source']) if retry else self.directory/'prepared'/(parent+'-whips-'+stamp())
            attempt=dict(generation=generation,source=self.rel(source),status='preparing',started_at=datetime.now(timezone.utc).isoformat())
            if retry: attempt['retry_of']=previous['job']
            state.setdefault('preparation_attempts',[]).append(attempt);self._save(state)
            try:
                from experimental_data.whip_full_data import default_contract,prepare
                from deployment.lab_seed import load_baseline
                if retry:
                    old_protocol=read(old_job/'protocol.json');contract=deepcopy(old_protocol['full_update'])
                    comparison=self.path(previous['comparison']);preliminary=self.path(old_protocol['preliminary_source'])
                else:
                    comparison=self._prepare_records(state,parent,parent,source)
                    contract=default_contract();contract.update(candidate_id=self.name+'-'+generation,parent_id=self.name+'-'+parent,
                        prior_whip_sources=[state['updates']['M1']['source']] if generation=='M2' else [])
                    preliminary=self.path(load_baseline(self.root)['preliminary'])
                job=self.root/'runs/adaptation'/(self.name+'-'+generation+'-'+stamp())
                attempt['job']=self.rel(job)
                prepare(job,source,preliminary,contract)
                portable_outputs(self.root,[job])
            except Exception as exc:
                attempt.update(status='failed',error=str(exc));self._save(state);raise
            attempt['status']='prepared'
            if retry: state.setdefault('update_history',{}).setdefault(generation,[]).append(deepcopy(previous))
            state['updates'][generation]=dict(job=self.rel(job),source=self.rel(source),source_stage=parent,comparison=self.rel(comparison),status='prepared')
            self._save(state);return state['updates'][generation]

    def fit_update(self,generation,device='cuda'):
        with self.runtime():
            from experimental_data import whip_full_data
            if Path(whip_full_data.ROOT).resolve()!=self.root:
                raise ValueError('Launch tools/lab.py from this repository checkout so fit registration stays in the same repository.')
            state=self._load()
            if generation not in state['updates']: raise ValueError('Prepare and freeze the update inputs first.')
            if generation in state['models']: raise ValueError('This generation already has a frozen model.')
            if any(s['status']!='pending' for s in state['slots'] if s['stage']=='final'): raise ValueError('Final recordings cannot influence an update.')
            job=self.path(state['updates'][generation]['job'])
            if (job/'fit').exists(): raise ValueError('This fit already started. Its artifacts are retained; it will not restart automatically.')
            from experimental_data.whip_full_fit import fit
            try: result=fit(job,device)
            finally:
                paths=[job,self.root/'config/evaluation/campaign.json']
                evaluation=self.root/'runs/evaluation'/job.name
                if evaluation.exists(): paths.append(evaluation)
                portable_outputs(self.root,paths)
            if result.get('status')!='completed': raise ValueError('Fit did not publish a completed candidate.')
            from experimental_data.model_evaluation import model_identity,load_catalog,save_catalog
            model=job/'candidate/model.json';signature,hashes=model_identity(model)
            catalog=load_catalog(self.root)
            for item in catalog['models']:
                if item['id']==self.name+'-'+generation:
                    item.update(signature=signature,model=self.rel(model))
                    item['hashes'].update({self.rel(p):h for p,h in hashes.items()})
            save_catalog(self.root,catalog)
            state=self._load();state['models'][generation]=dict(model=self.rel(model),signature=signature,status='Fit completed; ready for explicit planning')
            state['updates'][generation]['status']='completed';self._save(state);return result

    def stop(self):
        state=self._load();requested=[]
        for item in [*state['models'].values(),*state['updates'].values()]:
            relative=item.get('plan') or item.get('job')
            if relative:
                job=self.path(relative);status=job/'status.json'
                if status.is_file() and read(status).get('status')=='running':
                    (job/'STOP').write_text('Operator requested a cooperative stop.\n',encoding='utf-8');requested.append(relative)
        return dict(requested=requested)

    def evaluate(self,device='cuda',predictions=True):
        if predictions and device!='cuda': raise ValueError('Matched full-model diagnostics use the reviewed CUDA path. Use physical-only results on CPU.')
        with self.runtime():
            state=self._load();self._verify(state.get('planner_hashes',{}))
            output=self.directory/'results'/stamp();output.mkdir(parents=True)
            write(output/'study_snapshot.json',state)
            from experimental_data.adaptation_check import load_comparison
            from experimental_data.flight_performance import encounter,local_velocity
            import numpy as np
            rows=[]
            for slot in state['slots']:
                if slot['status']=='pending': continue
                self._verify(slot['raw_hashes'])
                row={k:slot[k] for k in ('stage','take','generation','role')};row.update(pair=slot.get('pair'),excluded=slot.get('excluded',False),reviewed=slot['status']=='reviewed',notes=slot.get('review',{}).get('notes',''))
                try:
                    item=state['models'][slot['generation']];self._verify(item['export_hashes'])
                    data=load_comparison(self.root,self.path(slot['batch']),slot['take'],self.path(item['rehearsal']))
                    mask=(data['time']>=0)&(data['time']<=state['measurement']['window_s'][1])
                    result=encounter(data['time'][mask],data['measured_cable'][mask,-1],data['target'],.05)
                    if state.get('planner_profile'):
                        velocity=None
                        end=slot.get('review',{}).get('free_motion_end_s')
                        if end is not None and slot['status']=='reviewed':
                            tip=data['measured_cable'][mask,-1].copy()
                            tip[data['time'][mask]>end]=np.nan
                            velocity=local_velocity(data['time'][mask],tip)
                        profile=read(self.path(state['planner_profile']))
                        measured=encounter(data['time'][mask],data['measured_cable'][mask,-1],data['target'],.05,
                            velocity=velocity,direction=profile['task']['strike_direction'])
                        row['directed_tip_speed_at_closest_m_s']=(measured['nearest'] or {}).get('outward_speed_m_s')
                        row['speed_estimator']='Five-sample centered quadratic derivative; contiguous reviewed free-motion observations only'
                        row['physical_fold_status']='Requires review of measured cable/video; simulated acceptance is not physical confirmation'
                    t=data['time'][mask];native_step=float(np.median(np.diff(t)))
                    complete=t[0]<=1.5*native_step and t[-1]>=state['measurement']['window_s'][1]-1.5*native_step
                    row.update(minimum_tip_target_m=result['nearest']['distance_m'] if result['nearest'] else None,
                        sample_coverage=result['sample_coverage'],segment_coverage=result['contiguous_segment_coverage'],
                        observation_end_s=float(t[-1]),window_complete=bool(complete),
                        fully_observed=bool(complete and result['sample_coverage']==1. and result['contiguous_segment_coverage']==1.),
                        source='Adjacent observed segments; no interpolation across missing frames')
                except (ValueError,IndexError) as exc: row.update(minimum_tip_target_m=None,error=str(exc))
                rows.append(row)
            paired=[]
            for i in range(1,6):
                selected={r['generation']:r for r in rows if r['stage']=='final' and r['pair']==i and not r['excluded'] and r['reviewed'] and r.get('minimum_tip_target_m') is not None}
                if set(selected)=={'M0','M2'}:
                    a,b=selected['M0']['minimum_tip_target_m'],selected['M2']['minimum_tip_target_m']
                    paired.append(dict(pair=i,M0_m=a,M2_m=b,improvement_m=a-b,
                        fully_observed=all(r['fully_observed'] for r in selected.values()),
                        coverage={g:{k:r[k] for k in ('sample_coverage','segment_coverage','window_complete','observation_end_s')} for g,r in selected.items()}))
            mean=lambda key:float(np.mean([p[key] for p in paired])) if paired else None
            report=dict(schema='deployment_lab_results_v1',study=self.name,measurement=state['measurement'],physical=rows,paired=paired,
                summary=dict(paired_count=len(paired),M0_mean_m=mean('M0_m'),M2_mean_m=mean('M2_m'),paired_mean_improvement_m=mean('improvement_m'),
                    fully_observed_pair_count=sum(p['fully_observed'] for p in paired),
                    fully_observed_paired_mean_improvement_m=float(np.mean([p['improvement_m'] for p in paired if p['fully_observed']])) if any(p['fully_observed'] for p in paired) else None),
                qualification='All distances and paired means use observed adjacent segments only. Truncated windows and gaps may hide a closer encounter; they remain explicitly flagged in each pair. The fully observed subset requires the complete 1.5 s window and every native sample/segment observed. Positive paired improvement means smaller observed M2 error. Development takes are not final evidence.',predictions=[],prediction_summary=[])
            write(output/'report.json',report)
            columns=['stage','take','generation','role','pair','excluded','reviewed','minimum_tip_target_m','sample_coverage','segment_coverage','window_complete','fully_observed','observation_end_s','directed_tip_speed_at_closest_m_s','speed_estimator','physical_fold_status','notes','error']
            with (output/'physical_results.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
            evidence={p:h for slot in state['slots'] for p,h in slot.get('raw_hashes',{}).items()}
            for model in state['models'].values(): evidence.update(model.get('export_hashes',{}))
            def publish_evidence():
                for file in output.rglob('*'):
                    if file.is_file() and file.name!='evidence_hashes.json' and 'source_snapshot' not in file.parts:
                        evidence[self.rel(file)]=digest(file)
                write(output/'evidence_hashes.json',evidence)
            publish_evidence()
            state['results']=dict(report=self.rel(output/'report.json'),path=self.rel(output),predictions_complete=False);self._save(state)
            if predictions:
                if any(s['status']!='reviewed' for s in state['slots'] if s['stage']=='final'): raise ValueError('Physical summary saved. Review all final slots before the matched model comparison, or use --physical-only.')
                from experimental_data.whip_full_data import default_contract
                from experimental_data.whip_full_fit import evaluate_pair
                models={g:self.path(self._assert_model(state,g)['model']) for g in ('M0','M1','M2')}
                training_hashes={h for s in state['slots'] if s['role']=='adaptation' for h in s.get('raw_hashes',{}).values()}
                for s in state['slots']:
                    if s['stage']=='final' and training_hashes.intersection(s['raw_hashes'].values()): raise ValueError('Final data appear in adaptation ancestry.')
                for generation in ('M0','M2'):
                    job=output/('prepared-final-'+generation)
                    self._prepare_records(state,'final',generation,job,diagnostics_only=True)
                    protocol=read(job/'protocol.json');protocol['full_update']=default_contract();write(job/'protocol.json',protocol)
                    portable_outputs(self.root,[job])
                    folder=output/('matched-'+generation);evaluate_pair(job,folder,models,device)
                    portable_outputs(self.root,[folder]);report['predictions'].append(self.rel(folder/'report.json'))
                for generation in models:
                    metrics=[take['metrics'] for file in report['predictions'] for take in read(self.path(file))['models'][generation]['takes'].values()]
                    report['prediction_summary'].append(dict(model=generation,takes=len(metrics),
                        tip_rmse_m=float(np.mean([m['command_driven_tip']['rmse_m'] for m in metrics])),
                        drone_rmse_m=float(np.mean([m['drone']['rmse_m'] for m in metrics]))))
                write(output/'report.json',report)
                publish_evidence()
            state['results']=dict(report=self.rel(output/'report.json'),path=self.rel(output),predictions_complete=bool(predictions));self._save(state)
            return report
