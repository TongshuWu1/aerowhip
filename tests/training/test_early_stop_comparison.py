import hashlib
from tools.early_stop_comparison import assess, check_run, DEFAULTS
from simulator.workflow import atomic_json, read_json


def test_manual_completion_evaluates_explicit_snapshot_and_rejects_corruption(tmp_path):
    import pytest
    from tools.early_stop_comparison import evaluation_checkpoint
    path=tmp_path/'checkpoints/manual.pt';path.parent.mkdir();path.write_bytes(b'user selected weights')
    atomic_json(tmp_path/'manual_completion.json',dict(final_evaluation_authorized=True,
        checkpoint='checkpoints/manual.pt',checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    decision={'stop_requested':False}
    assert evaluation_checkpoint(tmp_path,decision,{'status':'STOPPED'})[0]=='checkpoints/manual.pt'
    assert evaluation_checkpoint(tmp_path,decision,{'status':'FAILED'}) is None
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):evaluation_checkpoint(tmp_path,decision,{'status':'STOPPED'})


def records(rewards):
    return [dict(training_episodes=i*32768, mean_episode_reward=r,
                 record_id=str(i), evaluation_id=str(i)) for i, r in enumerate(rewards)]


def test_improving_and_plateau_without_minimum_budget():
    assert not assess(records([100, 110, 120, 130, 140, 150, 160, 170]), DEFAULTS)['should_stop']
    assert not assess(records([100]*5), DEFAULTS)['should_stop']
    result = assess(records([100]*6), DEFAULTS)  # five flat checks: 163840 episodes
    assert result['should_stop'] and result['best']['training_episodes'] == 0


def test_small_improvements_accumulate_and_actual_best_is_preserved():
    result = assess(records([100, 101, 102, 101, 102.5, 103, 103.5, 103.9]), DEFAULTS)
    assert result['should_stop']
    assert result['best']['mean_episode_reward'] == 103.9
    assert not assess(records([100, 101, 102, 101, 102.5, 103, 103.5, 104]), DEFAULTS)['should_stop']


def test_reused_or_nonfinite_evaluations_do_not_exhaust_patience():
    rows = records([100]*8)
    for row in rows[1:]:
        row['evaluation_reused'] = True
    assert assess(rows, DEFAULTS)['stale_checks'] == 0
    rows[-1].update(evaluation_reused=False, mean_episode_reward=float('nan'))
    assert not assess(rows, DEFAULTS)['should_stop']


def test_cooperative_stop_and_exact_reward_checkpoint(tmp_path):
    run = tmp_path/'run'
    (run/'validation').mkdir(parents=True)
    (run/'checkpoints').mkdir()
    for row in records([100]*8):
        path = run/'validation'/f"{int(row['record_id']):010d}.pt"
        path.write_bytes(row['record_id'].encode())
        row.update(checkpoint=str(path.relative_to(run)),
                   checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        atomic_json(path.with_suffix('.json'), row)
    atomic_json(run/'status.json', {'status': 'RUNNING'})
    result = check_run(run, DEFAULTS)
    assert result['stop_requested']
    assert (run/'STOP_REQUESTED').exists()
    assert not (tmp_path/'STOP_REQUESTED').exists()
    assert (run/'checkpoints/best_reward.pt').read_bytes() == b'0'
    assert check_run(run, DEFAULTS)['stop_requested']
    assert read_json(run/'early_stopping.json')['best_reward'] == 100
