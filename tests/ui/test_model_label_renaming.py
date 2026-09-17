import json
from simulator.gui.pva_workspace import model_label

def test_registered_rename_preserves_frozen_model_provenance(tmp_path):
    model=tmp_path/'runs/adaptation/renamed/candidate/model.json'
    model.parent.mkdir(parents=True)
    model.write_text(json.dumps({'provenance':{'label':'M7 full adaptation candidate'}}))
    original=model.read_bytes()
    catalog=tmp_path/'config/evaluation/campaign.json';catalog.parent.mkdir(parents=True)
    catalog.write_text(json.dumps({'models':[{'id':'M6','model':str(model.relative_to(tmp_path)),
        'renaming':{'previous_id':'M7'}}]}))
    assert model_label(model)=='M6 full adaptation candidate'
    assert model.read_bytes()==original
    other=tmp_path/'unregistered/model.json';other.parent.mkdir();other.write_bytes(original)
    assert model_label(other)=='M7 full adaptation candidate'
