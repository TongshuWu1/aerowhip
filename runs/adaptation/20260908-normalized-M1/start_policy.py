from pathlib import Path
import sys,json,torch,numpy as np
root=Path.cwd();sys.path.insert(0,str(root))
from simulator.workflow import prepare_training,read_json
from experimental_data.io import atomic_json,sha256_file
from tools.run_m1_policy_study import launch
parent=root/'runs/ppo/20260908-195207-486249-seed655/checkpoints/best_validation.pt'
assert sha256_file(parent)=='d10657f471deb8b22cafbee9008792e8378c8c764ca97039d6dbe7afcc260eea'
model_path=root/'data/model_candidates/20260908-normalized-M1/model.json';model=read_json(model_path)
start=int(torch.load(parent,map_location='cpu',weights_only=False)['episodes'])
offset=np.array(model['recorded_data']['optitrack_to_attachment_offset_body_m'])
early=read_json(root/'config/research_30hz/ppo.json')['early_stopping'];assert early['enabled'] and early['patience_episodes']==20000
backend=dict(type='isaaclab_model',python=str(Path.home()/'env_isaaclab/Scripts/python.exe'),headless=False,render_stride=5)
assert Path(backend['python']).is_file()
directory,command=prepare_training(root,'ppo',seed=655,episodes=start+204800,batch=2048,device='cuda',resume=parent,model_path=model_path,keep_optimizer_state=False,keep_stopping_history=False,task_overrides=dict(initial_root_position_m=(np.array([-2.,0.,1.255])+offset).tolist(),target_position_m=[-1.,0.,1.1]),early_stopping=early,live_scene=False,training_backend=backend,run_name='M1 normalized - flown PPO adaptation - reward plateau')
config=read_json(directory/'launch_config/ppo.json');task=read_json(directory/'launch_config/task.json')
assert config['reward']==read_json(parent.parent.parent/'ppo.json')['reward']
assert task['control_dt_s']==1/30 and task['episode_duration_s']==1
assert config['deployment']['reset_optimizer_on_resume']
atomic_json(directory/'adaptation_protocol.json',dict(model=str(model_path),model_sha256=sha256_file(model_path),parent_policy=str(parent),parent_sha256=sha256_file(parent),origin=[-2,0,1.255],target=[-1,0,1.1],reward_unchanged=True,optimizer_reset=True,stopping=early,maximum_additional_attempts=204800,ceiling_is_not_convergence=True,no_M0_control_arm=True,real_flight_performance='Unknown until new flights',original_policy_unchanged=True))
process=launch(command,root,directory/'console.log')
atomic_json(root/'runs/adaptation/20260908-normalized-M1/policy_run.json',dict(directory=str(directory),pid=process.pid,command=command))
print(json.dumps(dict(run=str(directory),pid=process.pid,start=start,command=command)))

