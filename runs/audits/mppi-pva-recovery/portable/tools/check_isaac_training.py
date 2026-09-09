"""Bounded numerical/scene audit of the Isaac-hosted external-model PPO collector."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--seconds', type=float, default=.2)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--project-only', action='store_true')
    # Keep the baseline executable in the project's Python without Isaac imports.
    if '--project-only' not in sys.argv:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = None
    if not args.project_only:
        app = AppLauncher(args, fast_shutdown=True).app
    import torch
    from copy import deepcopy
    from dataclasses import fields
    from simulator.research_config import workspace_configs
    from learning.point_force_env import PointForceWhipEnvironment, POINT_FORCE_OBSERVATION_DIM
    from learning.deployment_rollout import collect_deployment_rollout
    from learning.simple_ppo import PPORollout
    from learning.training_control import set_runtime_pump, reset_runtime_pump
    from run_ppo import build_agent, _load_checkpoint, configure_accelerator
    from experimental_data.io import atomic_json
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    model, task, config = deepcopy(workspace_configs(root))
    task['episode_duration_s'] = args.seconds
    config['live_scene'] = {'enabled': False}
    device = torch.device('cuda'); configure_accelerator(device)
    session = None; token = None
    try:
        if app:
            from simulator.isaac_training import IsaacLabTrainingSession
            session = IsaacLabTrainingSession(app, args.output, args.batch, headless=args.headless)
            token = set_runtime_pump(session.pump)
        results = []
        for hosted in ([False] if args.project_only else [False, True]):
            torch.manual_seed(761); torch.cuda.manual_seed_all(761)
            factory = session.environment if hosted else PointForceWhipEnvironment
            env = factory(model, task, config, batch_size=args.batch, device=device)
            agent = build_agent(config, device)
            if args.checkpoint:
                _load_checkpoint(agent, args.checkpoint, load_optimizer=False)
            rollout = PPORollout.allocate(env.control_step_count, args.batch,
                                         POINT_FORCE_OBSERVATION_DIM, 3, device=device)
            if hosted:
                env.collect_training_rollout(agent, rollout)
            else:
                collect_deployment_rollout(env, agent, rollout)
            data = {f.name: getattr(rollout, f.name).detach().cpu().clone() for f in fields(rollout)}
            data['rng'] = torch.cuda.get_rng_state().clone()
            agent.update(rollout, minibatch_size=256, epochs=1,
                         generator=torch.Generator(device=device).manual_seed(909))
            data['weights'] = {k: v.detach().cpu().clone() for k, v in agent.policy.state_dict().items()}
            data['values_weights'] = {k: v.detach().cpu().clone() for k, v in agent.value.state_dict().items()}
            results.append(data)
            del env, agent, rollout
        torch.save(results[0], args.output/'baseline.pt')
        if session:
            for key in results[0]:
                if isinstance(results[0][key], dict):
                    for name in results[0][key]:
                        torch.testing.assert_close(results[0][key][name], results[1][key][name], rtol=0, atol=0)
                else:
                    torch.testing.assert_close(results[0][key], results[1][key], rtol=0, atol=0)
            assert session.execution_frames > 0 and session.resets == 1
            assert len(session.drones.GetPositionsAttr().Get()) == args.batch
            assert len(session.cables.GetCurveVertexCountsAttr().Get()) == args.batch
            if not args.headless:
                import asyncio, time
                from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
                async def capture(path):
                    await capture_viewport_to_file(get_active_viewport(), file_path=str(path)).wait_for_result()
                for name, closeup in [('overview', False), ('closeup', True)]:
                    if closeup: session.close_up()
                    for _ in range(10): session.sim.render(); app.update()
                    future = asyncio.ensure_future(capture(args.output/f'{name}.png'))
                    deadline = time.monotonic()+30
                    while not future.done():
                        app.update()
                        if time.monotonic() > deadline: raise TimeoutError('Capture did not finish')
                    future.result()
        atomic_json(args.output/'verification.json', dict(
            passed=True, batch=args.batch, seconds=args.seconds, torch=torch.__version__,
            device=torch.cuda.get_device_name(), exact_hosted_parity=bool(session),
            compared='actions, observations, PPO tensors, rewards, CUDA RNG, policy and value weights after update',
            visible=bool(session and not args.headless)))
    except Exception as error:
        atomic_json(args.output/'verification.json', dict(passed=False, error=str(error)))
        import traceback
        traceback.print_exc();sys.stderr.flush();sys.stdout.flush()
        if app is not None:
            import omni.kit.app
            omni.kit.app.get_app().post_quit(1)
        raise
    finally:
        if token is not None: reset_runtime_pump(token)
        if session is not None: session.close()
        if app is not None: app.close()


if __name__ == '__main__':
    main()
