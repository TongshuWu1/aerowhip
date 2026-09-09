"""Record independent deterministic PPO executions using the run's frozen model.

Run with the project's Python, not Isaac's Python. No optimizer or flight calls.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--num-envs', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=20260908)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if not 1 <= args.num_envs <= 1024 or not 1 <= args.batch_size <= 1024:
        p.error('Environment and batch counts must be between 1 and 1024.')
    checkpoint = Path(args.checkpoint).resolve()
    source = checkpoint.parent.parent
    snapshot = source / 'source_snapshot'
    if not (snapshot / 'learning/research_rollout.py').is_file():
        raise ValueError('Select a native 30 Hz run with a frozen research source snapshot.')
    manifest = json.loads((source / 'source_snapshot_manifest.json').read_text())['files']
    for name, digest in manifest.items():
        if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen source hash mismatch: {name}')
    # Snapshot packages win; this script never changes or imports the live trainer.
    sys.path.insert(0, str(snapshot))
    import numpy as np
    import torch
    from experimental_data.io import atomic_json, sha256_file
    from run_ppo import build_agent, _load_checkpoint, configure_accelerator, validate_contract
    from learning.point_force_env import PointForceWhipEnvironment
    from learning.deployment_rollout import sample_batch, plan_batch
    from learning.research_rollout import execute_research_batch
    from simulator.research_pose import settled_initial

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'replay.json').exists() or (output / 'policy.pt').exists():
        raise ValueError('Use a new output folder; saved presentations are immutable.')
    model, task, config = [json.loads((source / f'{n}.json').read_text()) for n in ('model', 'task', 'ppo')]
    if model.get('fullstate_execution', {}).get('schema') != 'tracked_pose_execution_v1':
        raise ValueError('Only native 30 Hz tracked-pose policies are supported.')
    for spec in (model['motion_residual'], model['fullstate_execution']):
        if not Path(spec['checkpoint']).is_absolute():
            spec['checkpoint'] = str((source / spec['checkpoint']).resolve())
    validate_contract(model, task, config)
    if abs(float(task['control_dt_s'])-1/30) > 1e-12:
        raise ValueError('This presentation supports native 30 Hz policies only.')
    shutil.copyfile(checkpoint, output / 'policy.pt')
    for name in ('model.json', 'task.json', 'ppo.json', 'source_snapshot_manifest.json'):
        shutil.copyfile(source / name, output / name)
    device = torch.device(args.device)
    configure_accelerator(device)
    agent = build_agent(config, device)
    saved = _load_checkpoint(agent, output / 'policy.pt', load_optimizer=False)
    agent.policy.eval()
    rng = torch.Generator(device=device).manual_seed(args.seed)
    chunks = []
    offset = np.asarray(model['recorded_data']['optitrack_to_attachment_offset_body_m'])
    max_attachment_error = 0.
    env = None

    with torch.no_grad():
        for start in range(0, args.num_envs, args.batch_size):
            size = min(args.batch_size, args.num_envs - start)
            def progress(label, step=0, total=1):
                atomic_json(output / 'progress.json', dict(label=f'{start}/{args.num_envs} recorded · {label}', step=step, total=total))
            if env is None or env.batch_size != size:
                env = PointForceWhipEnvironment(model, task, config, batch_size=size, device=device)
            physics_dt = env.physics_dt_s
            batch = sample_batch(env, config['deployment'], rng)
            positions = [batch.truth.positions_m.cpu().numpy().copy()]
            hits = [np.zeros(size, dtype=bool)]
            def trace(i, force, state, striking, hit):
                positions.append(state.positions_m.cpu().numpy().copy())
                hits.append(hit.cpu().numpy().copy())
            forces, cutoffs = plan_batch(env, agent, batch, progress=progress)
            score = execute_research_batch(env, batch, forces, cutoffs, config['deployment'], trace=trace, progress=progress)
            q = np.stack(positions)
            cutoff = getattr(score,'terminal_steps',cutoffs).cpu().numpy()
            initial = settled_initial(batch.truth.positions_m[:, 0], batch.truth.velocities_m_s[:, 0], offset)
            origin = np.repeat(initial.position.cpu().numpy()[None], len(q), axis=0)
            rotation = np.repeat(initial.rotation.cpu().numpy()[None], len(q), axis=0)
            valid = np.ones(q.shape[:2], dtype=bool)
            if hasattr(score, 'predicted_pose'):
                pose = score.predicted_pose
                origin = pose['position_origin_m'].cpu().numpy().transpose(1, 0, 2).copy()
                rotation = pose['rotation_tracking_to_world'].cpu().numpy().transpose(1, 0, 2, 3).copy()
                valid = pose['valid'].cpu().numpy().T.copy()
            # Cable execution freezes at each plan's cutoff. Freeze O/R at that
            # same index. Invalid executions are retained, marked and frozen at
            # the last consistent frame; never draw a detached drone/cable.
            indices = np.minimum(np.arange(len(q))[:, None], cutoff[None])
            columns = np.arange(size)[None]
            origin, rotation, valid = origin[indices, columns], rotation[indices, columns], valid[indices, columns]
            error = np.linalg.norm(origin + np.einsum('tbij,j->tbi', rotation, offset) - q[:, :, 0], axis=-1)
            valid &= np.isfinite(q).all(axis=(2, 3)) & np.isfinite(error) & (error < 1e-7)
            valid[0] = True
            valid = np.logical_and.accumulate(valid, axis=0)
            last = np.maximum.accumulate(np.where(valid, np.arange(len(q))[:, None], 0), axis=0)
            q, origin, rotation = q[last, columns], origin[last, columns], rotation[last, columns]
            max_attachment_error = max(max_attachment_error, float(np.max(np.linalg.norm(origin + np.einsum('tbij,j->tbi', rotation, offset) - q[:, :, 0], axis=-1))))
            failed = score.failed.cpu().numpy()
            success = score.episode_success.cpu().numpy()
            chunks.append(dict(cable=q, origin=origin, rotation=rotation,
                hit=np.stack(hits) & success[None], valid=valid,
                target=batch.target_position_m.cpu().numpy(), cutoff=cutoff,
                success=success, failed=failed, reward=score.episode_reward.cpu().numpy()))
            progress('Batch complete', start + size, args.num_envs)
            del score, batch, forces
            import gc
            gc.collect()
            if device.type == 'cuda': torch.cuda.empty_cache()

    frames = max(len(c['cable']) for c in chunks)
    arrays = {}
    for key in ('cable', 'origin', 'rotation', 'hit', 'valid'):
        arrays[key] = np.concatenate([np.concatenate([c[key], np.repeat(c[key][-1:], frames - len(c[key]), axis=0)]) for c in chunks], axis=1)
    for key in ('target', 'cutoff', 'success', 'failed', 'reward'):
        arrays[key] = np.concatenate([c[key] for c in chunks])
    arrays['time_s'] = np.arange(frames) * physics_dt
    arrays['attachment_offset_m'] = offset
    np.savez_compressed(output / 'replay.npz', **arrays)
    metadata = dict(schema='multidrone_replay_v1', num_envs=args.num_envs, seed=args.seed,
        generation_batch_size=args.batch_size, checkpoint=str(checkpoint), source_snapshot=str(snapshot),
        checkpoint_sha256=sha256_file(output / 'policy.pt'), training_attempts=int(saved.get('episodes', 0)),
        recording='Deterministic checkpoint replay; independent sampled scenarios, not live training footage',
        physics='Frozen calibrated drone + cable model and both residuals; Isaac is visualization only',
        frame='World XYZ metres; unshifted tracked origin O, attachment A=O+R*offset',
        force_rate_hz=30, fullstate_rate_hz=30, physics_rate_hz=1/float(arrays['time_s'][1]) if frames>1 else 150,
        success_count=int(arrays['success'].sum()), failure_count=int(arrays['failed'].sum()),
        max_attachment_error_m=max_attachment_error, replay_sha256=sha256_file(output/'replay.npz'),
        display='Grid translations only; thicker cable and illustrative drone mesh for visibility; no physical interactions between cells',
        invalid='Failed attempts retained and freeze at last consistent pose; no synthesized recovery or successful-only filtering')
    metadata['verified_source_files'] = len(manifest)
    metadata['exporter_sha256'] = sha256_file(Path(__file__))
    shutil.copyfile(__file__, output/'export_multidrone.py')
    atomic_json(output / 'replay.json', metadata)
    atomic_json(output / 'progress.json', dict(label=f'Complete · {args.num_envs} independent rollouts · {metadata["success_count"]} valid hits', step=1, total=1))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
