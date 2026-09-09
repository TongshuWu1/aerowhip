"""Isaac Lab environment using our external CUDA dynamics and existing PPO collector.

Import only after AppLauncher. No replay files or second simulator process feed
this environment. Isaac Lab owns the scene/context; model.step_runtime owns the
drone/cable dynamics. The two-stage open-loop collection contract is preserved.
"""
from pathlib import Path
import time
import numpy as np
import torch
from learning.point_force_env import PointForceWhipEnvironment
from learning.training_control import TrainingStopped
from experimental_data.io import atomic_json


class IsaacLabWhipEnvironment(PointForceWhipEnvironment):
    def __init__(self,*args,session,**kwargs):
        self._isaac_session=session
        super().__init__(*args,**kwargs)

    def reset(self,state=None):
        observation=super().reset(state)
        self._isaac_session.planning_step(self,reset=True)
        return observation

    def step(self,*args,**kwargs):
        transition=super().step(*args,**kwargs)
        self._isaac_session.planning_step(self)
        return transition

    def collect_training_rollout(self,agent,rollout,progress=None):
        from learning.deployment_rollout import collect_deployment_rollout
        session=self._isaac_session
        session.collection+=1;session.in_collection=True
        def report(stage,step=0,total=0):
            session.stage=stage
            if progress:progress(stage,step,total)
        try:
            score=collect_deployment_rollout(self,agent,rollout,progress=report)
            session.end_collection(score,rollout)
        finally:
            session.in_collection=False


class IsaacLabTrainingSession:
    def __init__(self,app,artifact,batch,*,headless=False,render_stride=5,spacing=4.):
        from isaaclab.sim import SimulationContext,SimulationCfg
        from isaaclab.scene import InteractiveScene,InteractiveSceneCfg
        import omni.usd
        from pxr import UsdGeom,UsdLux,Gf,Vt
        self.app=app;self.artifact=Path(artifact);self.headless=headless
        self.render_stride=max(1,int(render_stride));self.collection=0;self.in_collection=False
        self.frames=0;self.planning_frames=0;self.execution_frames=0;self.resets=0
        self.stage='Initializing';self.paused=False;self.stopped=False;self.next_pump=0.;self.next_status=0.
        self.sim=SimulationContext(SimulationCfg(dt=1/150,render_interval=self.render_stride,device='cuda:0',use_fabric=False))
        self.scene=InteractiveScene(InteractiveSceneCfg(num_envs=batch,env_spacing=spacing,replicate_physics=False))
        self.offsets=self.scene.env_origins.detach().cpu().numpy().copy()
        self.sim.reset()
        self.usd=omni.usd.get_context().get_stage();self.UsdGeom=UsdGeom;self.Gf=Gf;self.Vt=Vt
        UsdGeom.SetStageUpAxis(self.usd,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(self.usd,1.)
        UsdLux.DomeLight.Define(self.usd,'/World/TrainingLight').CreateIntensityAttr(1000)
        self.drones=UsdGeom.PointInstancer.Define(self.usd,'/World/TrainingDrones')
        proto='/World/TrainingDrones/Prototype';UsdGeom.Xform.Define(self.usd,proto)
        body=UsdGeom.Cube.Define(self.usd,proto+'/Body');body.CreateSizeAttr(1.)
        body.AddScaleOp().Set(Gf.Vec3d(.12,.08,.045));body.CreateDisplayColorAttr([(.7,.8,.95)])
        for i,(x,y) in enumerate([(-.08,-.08),(-.08,.08),(.08,-.08),(.08,.08)]):
            rotor=UsdGeom.Cylinder.Define(self.usd,proto+f'/Rotor{i}');rotor.CreateRadiusAttr(.048);rotor.CreateHeightAttr(.012)
            rotor.AddTranslateOp().Set(Gf.Vec3d(x,y,0));rotor.CreateDisplayColorAttr([(.1,.6,.9)])
        self.drones.CreatePrototypesRel().SetTargets([proto])
        self.targets=UsdGeom.PointInstancer.Define(self.usd,'/World/TrainingTargets')
        self.sphere=UsdGeom.Sphere.Define(self.usd,'/World/TrainingTargets/Prototype');self.sphere.CreateRadiusAttr(.05)
        self.sphere.CreateDisplayColorAttr([(.9,.3,.1)]);self.targets.CreatePrototypesRel().SetTargets([self.sphere.GetPath()])
        self.cables=UsdGeom.BasisCurves.Define(self.usd,'/World/TrainingCables')
        self.cables.CreateTypeAttr('linear');self.cables.CreateWrapAttr('nonperiodic');self.cables.CreateWidthsAttr([.012]);self.cables.SetWidthsInterpolation('constant')
        self.colors=self.cables.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform)
        self.sim.set_camera_view(eye=[max(8.,np.sqrt(batch)*3),-max(8.,np.sqrt(batch)*3),max(6.,np.sqrt(batch)*2)],target=[0,0,1.2])
        self.label=None;self.metrics=None;self.selected=0
        if not headless:
            import omni.ui as ui
            self.panel=ui.Window('PPO training | external drone + cable model',width=640,height=245)
            with self.panel.frame:
                with ui.VStack(spacing=6):
                    ui.Label('Isaac Lab training environment | same-process PPO + calibrated CUDA physics')
                    self.label=ui.Label('Initializing model');self.metrics=ui.Label('')
                    ui.Label('Virtual force planning → fitted FullState execution → PPO update')
                    with ui.HStack(height=30):
                        ui.Button('Pause training',clicked_fn=lambda:setattr(self,'paused',not self.paused))
                        ui.Button('Stop and save',clicked_fn=self.request_stop)
                        ui.Button('Overview',clicked_fn=lambda:self.sim.set_camera_view(eye=[max(8.,np.sqrt(batch)*3),-max(8.,np.sqrt(batch)*3),max(6.,np.sqrt(batch)*2)],target=[0,0,1.2]))
                    with ui.HStack(height=25):
                        ui.Label('Drone',width=60);selected=ui.IntDrag(min=1,max=batch);selected.model.set_value(1)
                        selected.model.add_value_changed_fn(lambda m:setattr(self,'selected',max(0,min(batch-1,m.get_value_as_int()-1))))
                        ui.Button('Close-up',clicked_fn=self.close_up)
                    ui.Label('Green = provisional hit; red = failure. Completed return is credited before PPO updates.')
        self.write_status()

    def environment(self,*args,**kwargs):
        return IsaacLabWhipEnvironment(*args,session=self,**kwargs)

    def request_stop(self):
        self.stopped=True;(self.artifact/'STOP_REQUESTED').touch()

    def close_up(self):
        center=self.offsets[self.selected]+np.array([-1.5,0,1.5])
        self.sim.set_camera_view(eye=center+[3,-5,3],target=center)

    def pump(self):
        now=time.monotonic()
        if self.stopped or not self.app.is_running():raise TrainingStopped()
        if now<self.next_pump and not self.paused:return
        self.next_pump=now+.1
        self.app.update()
        while self.paused and not self.stopped and self.app.is_running():
            if (self.artifact/'STOP_REQUESTED').exists():self.stopped=True;break
            self.app.update();time.sleep(.01)
        if self.stopped or not self.app.is_running():raise TrainingStopped()
        if now>=self.next_status:
            self.next_status=now+1
            from simulator.workflow import read_json
            status=read_json(self.artifact/'status.json',{})
            if not self.in_collection:self.stage=status.get('stage','PPO optimization / validation')+' (scene held)'
            if self.label is not None:self.label.text=f'Collection {self.collection} · {self.stage}'
            self.write_status()

    def write_status(self):
        atomic_json(self.artifact/'isaac_environment.json',dict(schema='isaaclab_external_model_training_v1',
            collection=self.collection,frames=self.frames,planning_frames=self.planning_frames,
            execution_frames=self.execution_frames,resets=self.resets,stage=self.stage,
            environment_count=len(self.offsets),physics='calibrated CUDA model + both residuals',
            render_stride=self.render_stride,replay_source=None,paused=self.paused))

    def planning_step(self,env,reset=False):
        if not self.in_collection:return
        if reset:self.resets+=1
        q=env.state.positions_m
        origin=q[:,0]-q.new_tensor(env.model_config['recorded_data']['optitrack_to_attachment_offset_body_m'])
        self.stage='Virtual force planning (not fitted drone execution)'
        self.present(q,origin,None,env.target,env.episode_success,env.failed,env.target_radius_m)
        self.planning_frames+=1

    def execution_step(self,score,state,pose,index,cutoffs,failed):
        if not self.in_collection:return
        if getattr(self,'_pose_score',None) is not score:
            self._pose_score=score;self._pose_index=torch.zeros_like(cutoffs)
        self._pose_index=torch.where(~failed,torch.minimum(cutoffs,torch.full_like(cutoffs,index)),self._pose_index)
        if index%self.render_stride and index!=int(cutoffs.max()):return
        rows=torch.arange(len(cutoffs),device=cutoffs.device)
        sample=self._pose_index
        self.stage=f'Fitted drone + cable execution · {index*score.physics_dt_s:.3f} s'
        self.present(state.positions_m,pose['position_origin_m'][rows,sample],
            pose['rotation_tracking_to_world'][rows,sample],score.target,score.episode_success,failed,score.target_radius_m)
        self.execution_frames+=1
        if self.metrics is not None:
            self.metrics.text=f'{len(cutoffs)} environments · provisional hits {int(score.episode_success.sum())} · running mean reward {float(score.episode_reward.mean()):.2f}'

    def present(self,q,origin,rotation,target,success,failed,radius):
        from scipy.spatial.transform import Rotation
        n=len(q);offset=self.offsets[:n]
        q=q.detach().cpu().numpy();origin=origin.detach().cpu().numpy();target=target.detach().cpu().numpy()
        p=origin+offset;c=q+offset[:,None];t=target+offset
        self.drones.CreateProtoIndicesAttr([0]*n);self.targets.CreateProtoIndicesAttr([0]*n)
        self.drones.CreatePositionsAttr().Set(self.Vt.Vec3fArray.FromNumpy(p.astype(np.float32)))
        r=np.tile(np.eye(3),(n,1,1)) if rotation is None else rotation.detach().cpu().numpy()
        quat=Rotation.from_matrix(r).as_quat()
        self.drones.CreateOrientationsAttr().Set(self.Vt.QuathArray([self.Gf.Quath(float(w),self.Gf.Vec3h(float(x),float(y),float(z))) for x,y,z,w in quat]))
        self.targets.CreatePositionsAttr().Set(self.Vt.Vec3fArray.FromNumpy(t.astype(np.float32)));self.sphere.GetRadiusAttr().Set(float(radius))
        self.cables.CreateCurveVertexCountsAttr([q.shape[1]]*n);self.cables.CreatePointsAttr().Set(self.Vt.Vec3fArray.FromNumpy(c.reshape(-1,3).astype(np.float32)))
        colors=np.tile([.15,.6,.95],(n,1));colors[success.detach().cpu().numpy()]=[.1,.8,.25];colors[failed.detach().cpu().numpy()]=[1,.1,.1]
        self.colors.Set(self.Vt.Vec3fArray.FromNumpy(colors.astype(np.float32)))
        self.frames+=1
        self.sim.render();self.pump()

    def end_collection(self,score,rollout):
        mean=float(score.episode_reward.mean());success=int(score.episode_success.sum())
        credited=rollout.rewards.sum(0)[:,0].detach().cpu().numpy()
        np.testing.assert_allclose(credited,score.episode_reward.detach().cpu().numpy(),rtol=1e-6,atol=1e-4)
        row=dict(collection=self.collection,mean_reward=mean,successes=success,environments=score.batch_size,
                 frames=self.frames,credit_max_error=float(np.max(np.abs(credited-score.episode_reward.detach().cpu().numpy()))))
        with (self.artifact/'isaac_collections.jsonl').open('a',encoding='utf-8') as stream:
            import json
            stream.write(json.dumps(row)+'\n')
        self.stage='Batch complete; crediting execution return and updating PPO'
        if self.metrics is not None:self.metrics.text=f'Completed: {score.batch_size} attempts · {success} hits · credited mean return {mean:.3f}'
        self.write_status();self.sim.render()

    def close(self):
        self.write_status();self.sim.clear_all_callbacks();self.sim.clear_instance()
