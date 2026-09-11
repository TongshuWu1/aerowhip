import pytest
from experimental_data.io import atomic_json,sha256_file
from experimental_data.system_comparison import load_review,eligible_takes,equal_take_mean


def test_paired_filter_uses_training_ancestry_for_every_model():
    ds=dict(roles={'001':'adaptation','003':'validation','005':'validation'},models={
        'M0':{'001':dict(data_use='Excluded from this model fitting'),'003':dict(data_use='Excluded from this model fitting'),'005':dict(data_use='Excluded from this model fitting')},
        'M2':{'001':dict(data_use='Training or training replay'),'003':dict(data_use='Excluded from this model fitting')}})
    assert eligible_takes(ds,['M0','M2'])==['003']
    assert eligible_takes(ds,['M0','M2'],'all')==['001','003']
    # A future adaptation designation does not contaminate already frozen models.
    ds['models']['M2']['001']['data_use']='Excluded from this model fitting'
    assert eligible_takes(ds,['M0','M2'])==['001','003']


def test_equal_take_summary_does_not_replace_missing_values_with_zero():
    assert equal_take_mean([None,float('nan')]) is None
    assert equal_take_mean([.1,.3,None])==pytest.approx(.2)


def test_evidence_changes_invalidate_previously_readable_review(tmp_path):
    source=tmp_path/'model.json';source.write_text('original')
    report=tmp_path/'report.json'
    atomic_json(report,dict(schema='system_comparison_v1',models=[dict(id='M0')],source_hashes={str(source):sha256_file(source)}))
    pointer=tmp_path/'config/evaluation/system_review.json'
    atomic_json(pointer,dict(report='report.json',sha256=sha256_file(report)))
    assert load_review(tmp_path)['models'][0]['id']=='M0'
    source.write_text('changed')
    with pytest.raises(ValueError,match='evidence changed'):load_review(tmp_path)
    report.write_text('{}')
    with pytest.raises(ValueError,match='report changed'):load_review(tmp_path)


def test_missing_review_does_not_create_files(tmp_path):
    assert load_review(tmp_path) is None
    assert not list(tmp_path.iterdir())
