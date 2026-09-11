from PySide6.QtWidgets import QApplication
from simulator.gui.system_comparison_page import SystemComparisonPage
from experimental_data.io import atomic_json,sha256_file


def test_empty_and_changed_study_clear_scores_without_starting_jobs(tmp_path):
    app=QApplication.instance() or QApplication([])
    page=SystemComparisonPage(tmp_path);page.resize(950,700);page.show();app.processEvents()
    assert page.report is None and page.take_table.rowCount()==0 and not page.replay.isEnabled()
    assert 'No reviewed' in page.status.text()
    atomic_json(tmp_path/'config/evaluation/system_review.json',dict(report='missing.json',sha256='bad'))
    page.refresh()
    assert 'Evidence unavailable' in page.status.text() and page.report is None
    assert not (tmp_path/'runs').exists()
    page.close()


def test_modes_keep_original_ghost_and_filter_ancestry(tmp_path):
    app=QApplication.instance() or QApplication([])
    models=[];flights={};ds=dict(roles={'take_001':'adaptation','take_003':'validation'},models={},excluded=[])
    score=dict(rmse_m=.1,coverage=1.)
    for i in range(3):
        mid=f'M{i}';models.append(dict(id=mid,label=mid,signature=mid,interpretation='Test fixture'))
        flight=dict(mean_tip_rms_m=.1,mean_drone_rms_m=.1,observed_entries=0,batch=f'batch{i}',takes={})
        name=f'own{i}_001'
        flight['takes'][name]=dict(role='adaptation',drone=score,tip=score,encounter=dict(nearest=dict(distance_m=.1,outward_speed_m_s=4.)))
        flights[mid]=flight
        ds['models'][mid]={t:dict(data_use='Training or training replay' if t=='take_001' and i==2 else 'Excluded from this model fitting',
            metrics={k:score for k in ('drone','conditional_cable_tip','command_driven_tip')}) for t in ds['roles']}
    report=tmp_path/'review.json';atomic_json(report,dict(schema='system_comparison_v1',models=models,flights=flights,datasets={'New flights':ds},source_hashes={}))
    atomic_json(tmp_path/'config/evaluation/system_review.json',dict(report='review.json',sha256=sha256_file(report)))
    page=SystemComparisonPage(tmp_path);page.show();app.processEvents()
    assert page.take_table.rowCount()==3 and len(page.cards)==3
    observed=[];page.flight_requested.connect(lambda b,t:observed.append((b,t)))
    page.take_table.selectRow(1);page.open_replay()
    assert observed==[('batch1','own1_001')]
    page.mode.setCurrentIndex(1);app.processEvents()
    assert page.take_table.rowCount()==3 # Only 003, compared under three models.
    assert all(page.take_table.item(i,0).text().endswith('003') for i in range(3))
    page.selection.setCurrentIndex(2)
    assert page.take_table.rowCount()==6
    assert 'Training' in page.take_table.item(2,1).text()
    report.write_text('{}');page.refresh()
    assert page.take_table.rowCount()==0 and not page.figure.axes and page.report is None
    page.close()
