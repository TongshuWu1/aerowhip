import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

def test_free_target_jerk_dashboard_shows_release_geometry_instead_of_hit_distance():
    from PySide6.QtWidgets import QApplication
    from simulator.gui.mppi_dashboard import MPPIDashboard
    app=QApplication.instance() or QApplication([])
    page=MPPIDashboard()
    cfg=dict(command_contract='bounded_jerk_pva_30hz_v1',task=dict(duration_s=5.3),
        trajectory_objective=dict(free_target=True,speed_metric='tip_gain_over_root'),
        mppi=dict(mode='open_loop',samples=512,iterations=40,support_points=16,proposal_count=4))
    row=dict(iteration=1,elapsed_s=5.,strike_distance_m=None,horizontal_cable_rms_m=.12,
        directed_tip_speed_m_s=6.,rewarded_tip_speed_m_s=6.,accepted_fraction=.2,physics_valid_fraction=.8)
    page.update_run(dict(status='running',iteration=1),cfg,[row],[])
    assert page.labels['distance'].text()=='Cable height RMS at release'
    assert page.values['distance'].text()=='12.00 cm'
    assert page.values['step'].text()=='6.00 m/s'
    assert 'hits' not in page.labels['hit_fraction'].text().lower()
    page.close()
