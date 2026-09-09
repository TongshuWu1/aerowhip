import pytest
from experimental_data.io import atomic_json,canonical_json_hash,sha256_file
from experimental_data.historical_report import verify_combined_sources


def test_review_rejects_old_combined_evidence_after_model_change(tmp_path):
    atomic_json(tmp_path/'protocol.json',{'takes':['example']})
    cable=tmp_path/'cable'
    for label in ('fold_1','fold_2','fold_3','final'):
        model={'cable':{'EI_n_m2':1e-8}}
        atomic_json(cable/label/'candidate_model.json',model)
        drone=tmp_path/'drone_attachment'/label/'drone_residual.pt'
        drone.parent.mkdir(parents=True);drone.write_bytes(b'immutable test checkpoint')
        atomic_json(tmp_path/'combined'/label/'source_models.json',dict(
            cable_model_sha256=canonical_json_hash(model),drone_sha256=sha256_file(drone),
            input_protocol_sha256=sha256_file(tmp_path/'protocol.json')))
    verify_combined_sources(tmp_path,cable,'combined')
    atomic_json(cable/'final/candidate_model.json',{'cable':{'EI_n_m2':2e-8}})
    with pytest.raises(ValueError,match='final: combined review does not match'):
        verify_combined_sources(tmp_path,cable,'combined')
