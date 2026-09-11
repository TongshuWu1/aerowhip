from PySide6.QtWidgets import QApplication
from simulator.gui.model_evolution_page import ModelEvolutionPage
from experimental_data.model_evaluation import save_catalog,CATALOG


def test_empty_page_has_no_fabricated_models_results_or_jobs(tmp_path):
    app=QApplication.instance() or QApplication([])
    page=ModelEvolutionPage(tmp_path)
    assert page.models.rowCount()==0
    assert page.scores.rowCount()==0 and page.flights.rowCount()==0
    assert not page.run.isEnabled() and not page.job.running
    assert page.tabs.count()==4
    page.resize(1200,900);page.show();app.processEvents()
    assert not page.grab().isNull()  # Exercise native QWidget metric/painting callbacks.
    assert not (tmp_path/'runs').exists()
    assert page.shutdown();page.close()


def test_changed_model_is_visible_but_not_reported_as_verified(tmp_path):
    app=QApplication.instance() or QApplication([])
    save_catalog(tmp_path,dict(schema=CATALOG,models=[dict(id='M0',parent=None,model=str(tmp_path/'missing.json'),
        signature='abc',hashes={str(tmp_path/'missing.json'):'abc'},status='Development baseline')],flights=[]))
    page=ModelEvolutionPage(tmp_path)
    assert page.models.item(0,2).text()=='Evidence changed / unavailable'
    assert not page.job.running
    page.shutdown();page.close()


def test_full_fit_progress_shows_stages_and_current_loss(tmp_path):
    from experimental_data.io import atomic_json
    from experimental_data.whip_adaptation import SCHEMA
    app=QApplication.instance() or QApplication([])
    job=tmp_path/'runs/adaptation/full';job.mkdir(parents=True)
    atomic_json(job/'protocol.json',dict(schema=SCHEMA,full_update=dict(schema='whip_full_model_v1')))
    atomic_json(job/'status.json',dict(status='running',stage='cable_residual'))
    atomic_json(job/'progress.json',dict(stage='cable_residual',update=10,best_loss=1.25))
    for name in ('drone_nominal','drone_residual','attitude_refinement','cable_physics'):
        (job/name).mkdir();atomic_json(job/name/'result.json',dict(status='completed'))
    (job/'cable_residual').mkdir();atomic_json(job/'cable_residual/history.json',[dict(update=5,best_loss=1.4),dict(update=10,best_loss=1.25)])
    page=ModelEvolutionPage(tmp_path);page.draw_fit()
    assert page.fit_progress.maximum()==6 and page.fit_progress.value()==4
    assert 'Update 10' in page.fit_log.text() and '1.25' in page.fit_log.text()
    assert len(page.fit_figure.axes[0].lines[0].get_xdata())==2
    assert not page.job.running
    page.shutdown();page.close()


def test_sibling_candidates_do_not_appear_as_successive_generations(tmp_path):
    app=QApplication.instance() or QApplication([])
    models=[dict(id=name,parent=parent,model=str(tmp_path/(name+'.json')),signature=name,
        hashes={str(tmp_path/(name+'.json')):'missing'},status='Candidate')
        for name,parent in [('M0',None),('M1','M0'),('M1-full','M0')]]
    save_catalog(tmp_path,dict(schema=CATALOG,models=models,flights=[]))
    page=ModelEvolutionPage(tmp_path)
    assert page.cards[0].text()=='M0 → M1 · M0 → M1-full'
    assert page.canvas.minimumHeight()>=180
    assert not page.job.running
    page.shutdown();page.close()
