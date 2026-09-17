import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import json
import numpy as np
from PySide6.QtWidgets import QApplication


def test_shared_launch_defaults_override_checkpoint_fallback(tmp_path):
    from simulator.launch_setup import launch_positions
    folder=tmp_path/'config';folder.mkdir()
    (folder/'launch_setup.json').write_text(json.dumps(dict(initial_tracking_origin_m=[-2,0,1.255],target_position_m=[-1,0,1.1])))
    origin,target=launch_positions(tmp_path,[-2,.012874,1.255],[1,0,1.4])
    np.testing.assert_array_equal(origin,[-2,0,1.255]);np.testing.assert_array_equal(target,[-1,0,1.1])
    # The UI only applies defaults for new plans, not while opening saved artifacts.
    from simulator.gui.rehearsal_workspace import RehearsalWorkspace
    from pathlib import Path
    import shutil
    root=Path(__file__).resolve().parents[2]
    for file in ('model.json','task.json','ppo.json'):
        shutil.copy2(root/'config/research_30hz'/file,folder/file)
    app=QApplication.instance() or QApplication([])
    page=RehearsalWorkspace(tmp_path)
    # The inspector no longer creates policy plans or applies new-launch defaults.
    assert all(not spin.isEnabled() for spin in page.start_spins+page.target_spins)
    assert not hasattr(page,'checkpoints') and not hasattr(page,'generate')
    page.shutdown();page.close();app.processEvents()
