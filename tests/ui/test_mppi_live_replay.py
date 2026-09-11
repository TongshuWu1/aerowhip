import csv,os,time
import numpy as np
from experimental_data.io import atomic_json
from simulator.gui.mppi_dashboard import snapshot,readable_log


def test_lookahead_progress_resets_patience_without_claiming_committed_hits():
    cfg={'mppi':dict(mode='receding',initial_minimum_iterations=20,minimum_iterations=5,initial_patience=12,patience=3),'task':{'duration_s':5}}
    rows=[dict(command_step=0,window_iteration=i,iteration=i,lookahead_score=v,elapsed_s=i*5,success=.7) for i,v in enumerate([100,100.2,101,101.1],1)]
    s=snapshot(dict(status='running',command_step=0,window_iteration=5),cfg,rows,[])
    assert s['stale']==1 and s['minimum']==20 and s['step']==0 and s['maneuver']==0
    assert s['committed_distance'] is None and s['hit_fraction']==.7
    pending=snapshot(dict(status='running',command_step=1),cfg,rows,[])
    assert pending['hit_fraction'] is None and pending['stale']==0
    rows.append(dict(command_step=1,window_iteration=1,iteration=5,lookahead_score=1))
    s=snapshot(dict(status='running',command_step=1),cfg,rows,[dict(command_step=1,time_s=1/30,actual_minimum_tip_distance_m=.8)])
    assert s['stale']==0 and s['minimum']==5 and s['patience']==3 and s['committed_distance']==.8


def test_log_keeps_errors_and_partial_lines():
    text='Traceback: test\n{"stage":"Optimizing","iteration":3,"window_iteration":2}\n{"partial":'
    result=readable_log(text)
    assert 'total iteration 3' in result and 'lookahead iteration 2' in result
    assert 'Traceback: test' in result and '{"partial":' in result


def app():
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_live_mppi_dashboard_reads_first_window_and_stops_animation(tmp_path):
    qapp=app()
    from simulator.gui.pva_workspace import PVAPlannerPage
    from planning.pva_job import load_settings
    job=tmp_path/'runs/mppi_pva/live';cfg=load_settings(tmp_path,'mppi')
    atomic_json(job/'settings.json',cfg);atomic_json(job/'identity.json',dict(name='Live MPPI'))
    atomic_json(job/'status.json',dict(status='running',command_step=0,iteration=4,window_iteration=4))
    atomic_json(job/'history.json',[dict(iteration=i,window_iteration=i,command_step=0,lookahead_score=i,elapsed_s=i*4,predicted_distance_m=.5) for i in (1,2,3)])
    (job/'console.log').write_text('{"stage":"MPPI lookahead optimization","iteration":4}\n')
    before=(job/'status.json').read_bytes();page=PVAPlannerPage(tmp_path,'mppi');page.poll()
    assert page.mppi_dashboard.values['iteration'].text()=='4' and page.mppi_dashboard.work.maximum()==0
    assert page.log_toggle.isChecked() and 'total iteration 4' in page.saved_log.toPlainText()
    assert len(page.figure.axes)==2 and len(page.figure.axes[0].lines[0].get_xdata())==3
    assert not (job/'STOP').exists() and (job/'status.json').read_bytes()==before
    atomic_json(job/'status.json',dict(status='stopped',command_step=0));page.poll()
    assert page.mppi_dashboard.work.maximum()==1 and not page.stop.isEnabled()
    page.shutdown();page.close();qapp.processEvents()


def test_live_view_follows_selected_run_loads_and_animates_without_archiving(tmp_path):
    qapp=app()
    from simulator.gui.pva_workspace import PVAPlannerPage
    from planning.pva_job import load_settings
    jobs=[]
    for name,iteration in [('a',3),('b',7)]:
        job=tmp_path/'runs/mppi_pva'/name;jobs.append(job)
        atomic_json(job/'settings.json',load_settings(tmp_path,'mppi'))
        atomic_json(job/'identity.json',dict(name=name))
        atomic_json(job/'status.json',dict(status='completed',iterations=iteration))
        q=np.zeros((1,3,3,3));q[:,:,:,2]=[1.2,.9,.6];q[0,:,2,0]=[0,.1,.2]
        np.savez(job/'live.npz',schema='mppi_live_v1',cable_positions_m=q,
            origin_positions_m=q[:,:,0],origin_rotations=np.tile(np.eye(3),(1,3,1,1)),
            time_s=np.array([0.,.1,.2]),target_position_m=[1.,0.,1.],
            labels=['Best proposal'],failed=[False],success=[True],scores=[5.],
            frame_counts=[3],committed_origin_m=q[0,:1,0],committed_tip_m=q[0,:1,-1],
            iteration=iteration,command_step=0,series_iterations=[iteration],
            created_unix_s=time.time(),candidate_ids=[0],termination_time_s=[.2])
    page=PVAPlannerPage(tmp_path,'mppi');page.current_run=jobs[0];page.poll()
    assert page.live_view.job==jobs[0]
    page.tabs.setCurrentWidget(page.live_view);page.set_page_active(True)
    live=page.live_view
    assert live.data is not None and live.viewer is not None and live.candidate.count()==1
    live.slider.setValue(1);assert '0.100 s' in live.frame_note.text()
    live.last_tick=time.perf_counter()-.04;live.tick();assert live.clock>.1
    page.progress_runs.setCurrentIndex(page.progress_runs.findData(str(jobs[1])))
    assert live.job==jobs[1] and int(live.data['iteration'])==7
    page.set_page_active(False);assert not live.timer.isActive()
    assert not any((job/'ARCHIVED').exists() for job in jobs)
    page.shutdown();page.close();qapp.processEvents()


def test_archived_rehearsals_hidden_without_deleting_files(tmp_path):
    from types import SimpleNamespace
    from PySide6.QtWidgets import QComboBox
    from simulator.gui.pva_main_window import PVAResearchWindow
    qapp=app();combo=QComboBox()
    for name in ('retired','selected'):
        directory=tmp_path/'runs/rehearsals_pva'/name
        atomic_json(directory/'rehearsal.json',dict(planner='MPPI'))
        if name=='retired':(directory/'ARCHIVED').touch()
    PVAResearchWindow.refresh_rehearsals(SimpleNamespace(root=tmp_path,rehearsals=combo))
    assert combo.count()==1 and combo.currentData().endswith('selected')
    assert (tmp_path/'runs/rehearsals_pva/retired/rehearsal.json').exists()
    combo.close();qapp.processEvents()


def native_take(folder,name):
    folder.mkdir(parents=True,exist_ok=True);path=folder/(name+'.csv')
    columns=[('cf_3','Position',a) for a in 'XYZ']+[('cf_3','Rotation',a) for a in 'XYZW']+[(f'cable1:c{k}','Position',a) for k in range(1,11) for a in 'XYZ']
    rows=[['Length Units','Meters','Coordinate Space','Global','Rotation Type','Quaternion'],[],[],['','']+[c[0] for c in columns],[],['','']+[c[1] for c in columns],['Frame','Time (Seconds)']+[c[2] for c in columns]]
    for i in range(20):rows.append([i,10+i*.01,i*.01,0,1.7,0,0,0,1]+([v for k in range(10) for v in [0,0,1.6-.1*k]] if i<15 else ['']*30))
    with path.open('w',newline='') as stream:csv.writer(stream).writerows(rows)
    atomic_json(folder/'experiment.json',dict(files={'optitrack':path.name},fit_role='validation'))
    return path


def test_raw_replay_preserves_time_height_and_marker_gaps(tmp_path):
    from simulator.gui.recorded_takes_replay import discover_takes,load_take
    p=native_take(tmp_path/'data/raw_takes/take_001','take_001');before=p.read_bytes();data=load_take(p)
    assert data['time'][0]==10 and data['elapsed'][0]==0 and np.all(data['drone'][:,2]==1.7)
    assert data['marker_valid'][:15].all() and not data['marker_valid'][15:].any()
    assert np.isnan(data['cable'][15:]).all() and p.read_bytes()==before
    assert list(discover_takes(tmp_path).values())[0]['role']=='validation'


def test_recorded_replay_selection_scrub_playback_and_hidden_pause(tmp_path):
    from simulator.gui.recorded_takes_replay import RecordedTakesReplay
    qapp=app();a=native_take(tmp_path/'data/raw_takes/a','a');b=native_take(tmp_path/'data/raw_takes/b','b')
    page=RecordedTakesReplay(tmp_path);assert page.takes.count()==2 and page.worker is None
    page.open_take('b');page.set_page_active(True)
    deadline=time.perf_counter()+5
    while page.worker is not None and time.perf_counter()<deadline:qapp.processEvents();time.sleep(.01)
    assert page.worker is None and page.takes.currentData()==str(b.resolve())
    page.timeline.setValue(19);assert '0/10' in page.time_note.text()
    page.toggle_play();assert page.playing and page.timeline.value()==0
    page.set_page_active(False);assert not page.playing and not page.timer.isActive()
    assert page.shutdown();page.close();qapp.processEvents()
