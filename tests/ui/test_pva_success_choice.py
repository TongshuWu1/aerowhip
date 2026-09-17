import os
from experimental_data.io import atomic_json
from learning.pva_env import defaults


def test_success_selector_hides_unrequired_gates_and_preserves_legacy_profiles(tmp_path):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from simulator.gui.pva_workspace import PVAPlannerPage
    app=QApplication.instance() or QApplication([])
    cfg=defaults('mppi');cfg['task'].pop('success_criterion')
    atomic_json(tmp_path/'config/pva/mppi.json',cfg)
    page=PVAPlannerPage(tmp_path,'mppi')
    assert page.success_rule.currentData()=='legacy_strike_v1'
    assert not page.first_contact.isHidden()
    page.success_rule.setCurrentIndex(page.success_rule.findData('tip_contact_v1'))
    assert page.first_contact.isHidden()
    assert page.fields[('task','minimum_directed_speed_m_s')].isHidden()
    assert not page.fields[('task','target_radius_m')].isHidden()
    # No run/model selection is needed to stage this choice.
    assert 'do not gate a hit' in page.success_note.text()
    page.success_rule.setCurrentIndex(page.success_rule.findData('legacy_strike_v1'))
    assert not page.first_contact.isHidden()
    page.shutdown();page.close();app.processEvents()
