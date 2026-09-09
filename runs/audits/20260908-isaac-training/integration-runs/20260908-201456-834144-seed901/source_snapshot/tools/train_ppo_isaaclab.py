"""Train PPO in an Isaac Lab session with our calibrated external-model environment."""
import argparse
from pathlib import Path
import sys
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--config-directory',type=Path,required=True)
    parser.add_argument('--artifact-directory',type=Path,required=True)
    parser.add_argument('--episodes',type=int);parser.add_argument('--batch-size',type=int)
    parser.add_argument('--resume-checkpoint',type=Path)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    app_launcher=AppLauncher(args);app=app_launcher.app
    from simulator.isaac_training import IsaacLabTrainingSession
    from learning.training_control import set_runtime_pump,reset_runtime_pump
    from run_ppo import train,load_configs
    from experimental_data.io import atomic_json
    import torch
    config=load_configs(args.config_directory)[2]
    batch=args.batch_size or config['training']['collection_batch']
    session=None;token=None
    try:
        session=IsaacLabTrainingSession(app,args.artifact_directory,batch,headless=args.headless,
            render_stride=config.get('training_backend',{}).get('render_stride',5))
        token=set_runtime_pump(session.pump)
        atomic_json(args.artifact_directory/'isaac_runtime.json',dict(backend='isaaclab_external_model',
            python=sys.executable,torch=torch.__version__,device=torch.cuda.get_device_name(),
            physics='existing calibrated CUDA model; no PhysX drone/cable replacement'))
        train(device_name=args.device,requested_episodes=args.episodes,batch_size=args.batch_size,
            artifact=args.artifact_directory,resume_checkpoint=args.resume_checkpoint,
            config_directory=args.config_directory,environment_factory=session.environment)
        return 0
    except Exception as error:
        atomic_json(args.artifact_directory/'isaac_error.json',dict(error=str(error),type=type(error).__name__))
        raise
    finally:
        if token is not None:reset_runtime_pump(token)
        if session is not None:session.close()
        app.close()


if __name__=='__main__':
    raise SystemExit(main())
