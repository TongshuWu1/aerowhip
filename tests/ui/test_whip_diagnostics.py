"""Diagnostic-only folds must remain viewable without a completion timestamp."""
import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from deployment.whip_diagnostics import draw
from planning.strike_objective import FOLD


@pytest.mark.parametrize('completion', [None, 'missing', float('nan'), float('inf'), 0., .2])
def test_optional_fold_completion_can_be_rendered(completion):
    t=np.array([0., .1, .2, .3])
    q=np.zeros((len(t),7,3));q[:,:,2]=np.linspace(1.2,.2,7)
    arrays=dict(prediction_time_s=t,origin_positions_m=q[:,0],
        origin_velocities_m_s=np.zeros((len(t),3)),cable_positions_m=q,
        cable_velocities_m_s=np.zeros_like(q))
    finite=isinstance(completion,(int,float)) and np.isfinite(completion)
    metadata=dict(whip_end_s=.3,initial_tracking_origin_m=[0,0,1.2],target_position_m=[1,0,1],
        objective_schema='targeted_fold_strike_v1',predicted_fold_valid=bool(finite),
        strike_time_s=.25,directed_tip_speed_m_s=4.,strike_distance_m=.01)
    if completion!='missing':metadata['fold_completed_time_s']=completion
    settings=dict(task={'strike_direction':[1,0,0]},fold_constraint=copy.deepcopy(FOLD))
    figure=Figure();canvas=FigureCanvasAgg(figure)
    result=draw(figure,arrays,metadata,settings);canvas.draw()
    axis=figure.axes[3]
    markers=[line for line in axis.lines if line.get_label()=='Fold completed']
    assert len(markers)==int(finite)
    if finite:np.testing.assert_array_equal(markers[0].get_xdata(),[completion,completion])
    else:assert any(text.get_text()=='No fold completion recorded' for text in axis.texts)
    assert result['fold_valid']==bool(finite)
    assert any(line.get_label()=='Scored strike' for line in axis.lines)


def test_saved_recording_candidate_loads_in_rehearsal_workspace():
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from simulator.gui.rehearsal_workspace import RehearsalWorkspace
    root=Path(__file__).resolve().parents[2]
    folder=root/'runs/rehearsals_pva/20260912-223119-612102-M0-recording-candidate'
    if not (folder/'rehearsal.json').exists():pytest.skip('Private recording candidate is unavailable')
    metadata=json.loads((folder/'rehearsal.json').read_text())
    assert metadata['fold_completed_time_s'] is None
    app=QApplication.instance() or QApplication([])
    page=RehearsalWorkspace(root,inspection_only=True)
    try:
        page.load_result(folder)
        page.whip_canvas.draw()
        assert page.arrays is not None and page.play.isEnabled()
        assert 'no verified fold' in page.status.text()
        assert any(text.get_text()=='No fold completion recorded' for text in page.whip_figure.axes[3].texts)
    finally:
        page.close();app.processEvents()
