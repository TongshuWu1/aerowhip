"""Isaac Sim presentation renderer. All motion comes from saved model rollouts."""
import argparse
import asyncio
import json
from pathlib import Path
import time

parser = argparse.ArgumentParser(description=__doc__)
source=parser.add_mutually_exclusive_group(required=True)
source.add_argument('--replay')
source.add_argument('--run',help='Follow actual PPO collections from this run')
parser.add_argument('--close-flag',help='Owned UI requests graceful viewer shutdown through this file')
parser.add_argument('--live-smoke-batches',type=int,default=0)
parser.add_argument('--num-envs', type=int, default=1024)
parser.add_argument('--smoke-frames', type=int, default=0)
parser.add_argument('--screenshot')
parser.add_argument('--record-dir', help='Render one fixed-timestep loop to PNGs, then exit')
parser.add_argument('--fps', type=int, default=30)
parser.add_argument('--speed', type=float, default=.5)
parser.add_argument('--spacing', type=float, default=2., help='Display-only grid spacing in metres')
parser.add_argument('--headless', action='store_true')
args = parser.parse_args()
if args.fps <= 0 or args.speed <= 0 or args.num_envs < 1 or args.spacing <= 0:
    parser.error('Positive frame rate, speed and environment count required.')

# Rendering lives in Isaac's separate Python, with no imports of project physics.
from isaacsim import SimulationApp
app = SimulationApp({'headless':args.headless, 'width':1600, 'height':900,
    'window_width':1600, 'window_height':1000, 'renderer':'RaytracedLighting',
    'anti_aliasing':2, 'sync_loads':True, 'fast_shutdown':True})

import numpy as np
import carb
from scipy.spatial.transform import Rotation
import omni.usd
import omni.ui as ui
from pxr import UsdGeom, UsdLux, UsdPhysics, UsdShade, Sdf, Gf, Vt
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
from multidrone_data import load_replay, display_frame, grid_offsets, read_live_batch


def main():
    if args.run:
        waiting=ui.Window('Live PPO — waiting for training',width=620,height=100)
        with waiting.frame:waiting_label=ui.Label('Waiting for the first completed training batch…',word_wrap=True)
        while app.is_running():
            if args.close_flag and Path(args.close_flag).exists():return
            first=read_live_batch(args.run)
            if first:
                data,meta,directory=first;args.replay=str(directory);break
            try:
                status=json.loads(Path(args.run,'status.json').read_text())
                waiting_label.text=f'{Path(args.run).name}\n{status.get("status", "Starting")} · {status.get("stage", "Waiting for first training collection")}'
            except (OSError,ValueError):pass
            app.update();time.sleep(.05)
        else:return
        waiting.visible=False
    else:data, meta = load_replay(args.replay)
    stage = omni.usd.get_context().get_stage()
    carb.settings.get_settings().set_bool('/app/viewport/grid/enabled',False)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    count = min(args.num_envs, meta['num_envs'])
    times = data['time_s']
    end = float(times[-1])
    ground = UsdGeom.Cube.Define(stage, '/World/Floor')
    ground.CreateSizeAttr(1.)
    ground.AddTranslateOp().Set(Gf.Vec3d(0,0,-.05))
    span = max(40., np.ceil(np.sqrt(count))*20)
    ground.AddScaleOp().Set(Gf.Vec3d(span,span,.1))
    ground.CreateDisplayColorAttr([(0.045,0.065,0.105)])
    material=UsdShade.Material.Define(stage,'/World/FloorMaterial')
    shader=UsdShade.Shader.Define(stage,'/World/FloorMaterial/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.018,.028,.05))
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.95)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(material)
    light = UsdLux.DomeLight.Define(stage, '/World/Light')
    light.CreateIntensityAttr(600)
    sun = UsdLux.DistantLight.Define(stage, '/World/Sun')
    sun.CreateIntensityAttr(900)
    sun.AddRotateXYZOp().Set(Gf.Vec3f(20,-35,15))

    def instancer(path, prototype_builder):
        actor = UsdGeom.PointInstancer.Define(stage,path)
        proto = path+'/Prototype'
        prototype_builder(proto)
        actor.CreatePrototypesRel().SetTargets([proto])
        actor.CreateProtoIndicesAttr([0]*count)
        return actor

    def drone(proto):
        UsdGeom.Xform.Define(stage, proto)
        body = UsdGeom.Cube.Define(stage, proto+'/Body')
        body.CreateSizeAttr(1.)
        body.AddTranslateOp().Set(Gf.Vec3d(0,0,-.025))
        body.AddScaleOp().Set(Gf.Vec3d(.10,.07,.05))
        body.CreateDisplayColorAttr([(0.8,.88,.98)])
        for i,(x,y) in enumerate([(-.08,-.08),(-.08,.08),(.08,-.08),(.08,.08)]):
            arm = UsdGeom.Cube.Define(stage,proto+f'/Arm{i}')
            arm.CreateSizeAttr(1.)
            arm.AddTranslateOp().Set(Gf.Vec3d(x/2,y/2,-.025))
            arm.AddRotateZOp().Set(45 if x*y>0 else -45)
            arm.AddScaleOp().Set(Gf.Vec3d(.12,.015,.012))
            arm.CreateDisplayColorAttr([(.3,.4,.55)])
            rotor = UsdGeom.Cylinder.Define(stage,proto+f'/Rotor{i}')
            rotor.CreateRadiusAttr(.049); rotor.CreateHeightAttr(.007)
            rotor.AddTranslateOp().Set(Gf.Vec3d(x,y,-.018))
            rotor.CreateDisplayColorAttr([(.15,.65,.95) if x>0 else (.7,.8,.9)])

    drones = instancer('/World/Drones',drone)
    def target(proto):
        task=json.loads(Path(args.replay,'task.json').read_text())
        s=UsdGeom.Sphere.Define(stage,proto);s.CreateRadiusAttr(float(meta.get('target_radius_m',task['success']['tip_target_distance_m'])))
        s.CreateDisplayColorAttr([(.98,.36,.16)])
    targets = instancer('/World/Targets',target)
    cable = UsdGeom.BasisCurves.Define(stage,'/World/Cables')
    cable.CreateTypeAttr('linear'); cable.CreateWrapAttr('nonperiodic')
    cable.CreateCurveVertexCountsAttr([data['cable'].shape[2]]*count)
    cable.CreateWidthsAttr([.012]); cable.SetWidthsInterpolation('constant')
    color = cable.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform)
    camera = UsdGeom.Camera.Define(stage,'/World/Camera')
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01,2000))
    camera.CreateFocalLengthAttr(32.)
    camera.CreateHorizontalApertureAttr(36.)
    camera.CreateVerticalApertureAttr(20.25)
    camera_op = camera.AddTransformOp()
    viewport = get_active_viewport()
    if viewport is None: raise RuntimeError('Isaac did not create a viewport.')
    viewport.camera_path = camera.GetPath()
    viewport.set_texture_resolution((1600,900))
    state = dict(playing=True, seconds=0., speed=args.speed, close=False, selected=0, frame=-1)

    def aim(eye,focus):
        camera_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*focus),Gf.Vec3d(0,0,1)).GetInverse())

    def camera_view(close=False):
        state['close']=close
        if close:
            o=grid_offsets(count,args.spacing)[state['selected']]
            aim(o+np.array([3.8,-5.,3.2]),o+np.array([0.,.1,1.5]))
        else:
            offsets=grid_offsets(count,args.spacing)
            low=data['cable'][:,:count].min(axis=(0,1,2))+offsets.min(0)-[.5,.5,.1]
            high=data['cable'][:,:count].max(axis=(0,1,2))+offsets.max(0)+[.5,.5,.3]
            center=(low+high)/2
            corners=np.array([[x,y,z] for x in [low[0],high[0]] for y in [low[1],high[1]] for z in [low[2],high[2]]])-center
            direction=np.array([.55,-.75,.65]);direction/=np.linalg.norm(direction)
            right=np.cross([0,0,1],direction);right/=np.linalg.norm(right);up=np.cross(direction,right)
            distance=1.08*np.max(corners@direction+np.maximum(np.abs(corners@right)/(18/32),np.abs(corners@up)/(10.125/32)))
            aim(center+direction*max(6,distance),center)

    camera_view()
    panel = ui.Window('Cable whipping | live PPO' if args.run else 'Cable whipping | checkpoint replay',width=650,height=335)
    with panel.frame:
        with ui.VStack(spacing=6):
            heading=ui.Label(f'{count} independent model rollouts | checkpoint at {meta["training_attempts"]:,} attempts')
            ui.Label('Calibrated drone + cable + both residuals | Isaac rendering only')
            identity=ui.Label('Actual exploratory PPO collection | latest completed batch' if args.run else 'Deterministic checkpoint replay, not live training | whip only')
            training_status=ui.Label('')
            batch_reward=ui.Label('')
            label = ui.Label('Loading…')
            with ui.HStack(height=28):
                ui.Button('Play / Pause',clicked_fn=lambda:state.update(playing=not state['playing']))
                ui.Button('Restart',clicked_fn=lambda:state.update(seconds=0.,playing=True))
                ui.Button('Grid overview',clicked_fn=lambda:camera_view(False))
                ui.Button('Close-up',clicked_fn=lambda:camera_view(True))
            with ui.HStack(height=24):
                ui.Label('Speed',width=50)
                speed=ui.FloatSlider(min=.1,max=2.);speed.model.set_value(args.speed)
                speed.model.add_value_changed_fn(lambda m:state.update(speed=m.get_value_as_float()))
                ui.Label('Drone',width=55)
                selected=ui.IntDrag(min=1,max=count);selected.model.set_value(1)
                def select(m):
                    state['selected']=max(0,min(count-1,m.get_value_as_int()-1))
                    state['frame']=-1
                    if state['close']:camera_view(True)
                selected.model.add_value_changed_fn(select)
            timeline=ui.FloatSlider(min=0.,max=max(end,.001))
            syncing=[False]
            def scrub(m):
                if not syncing[0]:state.update(seconds=m.get_value_as_float(),playing=False)
            timeline.model.add_value_changed_fn(scrub)
            ui.Label('Cable: blue = executing | green = valid hit | orange = miss | red = invalid')
            ui.Label('Cells translated for display. Drone illustrative; cable thickness enlarged.')

    def draw(seconds):
        frame = min(len(times)-1,max(0,int(np.searchsorted(times,min(seconds,end),side='right')-1)))
        if frame == state['frame']:return
        state['frame']=frame
        pos,rot,q,t = display_frame(data,frame,count,args.spacing)
        drones.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(pos.astype(np.float32)))
        quat=Rotation.from_matrix(rot).as_quat()
        drones.CreateOrientationsAttr().Set(Vt.QuathArray([Gf.Quath(float(w),Gf.Vec3h(float(x),float(y),float(z))) for x,y,z,w in quat]))
        targets.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(t.astype(np.float32)))
        cable.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(q.reshape(-1,3).astype(np.float32)))
        colors=np.tile([.16,.65,.98],(count,1))
        finished=frame>=data['cutoff'][:count]
        colors[finished]=[1.,.46,.14]
        colors[data['hit'][frame,:count]]=[.15,.95,.48]
        invalid=~data['valid'][frame,:count] | (finished & data['failed'][:count])
        colors[invalid]=[1.,.1,.18]
        color.Set(Vt.Vec3fArray.FromNumpy(colors.astype(np.float32)))
        label.text=f'{min(seconds,end):.3f} / {end:.3f} s | {int(data["hit"][frame,:count].sum())}/{count} valid hits | {int(invalid.sum())} invalid'
        if args.run:
            heading.text=f'{Path(args.run).name} | {count}/{meta["num_envs"]} displayed'
            identity.text=f'Training attempts {meta["batch_start_attempt"]:,}–{meta["batch_end_attempt"]:,} | exploratory collection'
            chosen=min(state['selected'],count-1)
            batch_reward.text=f'Whole-episode returns: batch mean {meta["mean_episode_reward"]:.3f} | drone {chosen+1}: {data["reward"][chosen]:.3f} | batch hits {meta["success_count"]}/{meta["num_envs"]}'
        syncing[0]=True;timeline.model.set_value(min(seconds,end));syncing[0]=False

    draw(0.)
    for _ in range(40):app.update()
    async def capture(path):
        await capture_viewport_to_file(viewport,file_path=str(path)).wait_for_result()

    def capture_blocking(path):
        future=asyncio.ensure_future(capture(path))
        started=time.monotonic()
        while not future.done():
            app.update()
            if time.monotonic()-started>30:raise TimeoutError('Viewport capture timed out')
        future.result()

    if args.record_dir:
        folder=Path(args.record_dir);folder.mkdir(parents=True,exist_ok=True)
        if any(folder.glob('frame_*.png')) or (folder/'recording.json').exists():
            raise ValueError('Use a new recording folder; existing video frames are preserved.')
        camera_view(False)
        locked_camera=Gf.Matrix4d(camera_op.Get())
        for i in range(int(np.ceil(end/args.speed*args.fps))+1):
            camera_op.Set(locked_camera)
            viewport.camera_path=camera.GetPath()
            draw(i*args.speed/args.fps)
            for _ in range(3):app.update()
            capture_blocking(folder/f'frame_{i:05d}.png')
        (folder/'recording.json').write_text(json.dumps(dict(checkpoint_sha256=meta['checkpoint_sha256'],
            replay_sha256=meta['replay_sha256'],num_envs=count,fps=args.fps,playback_speed=args.speed,
            frame_count=i+1,label=meta['recording'],physics=meta['physics']),indent=2))
        print(f'RECORDING_COMPLETE {folder}',flush=True)
        return
    if args.smoke_frames:
        started=time.perf_counter();max_error=0.
        for i in range(args.smoke_frames):
            draw(end*i/max(1,args.smoke_frames-1));app.update()
            expected=display_frame(data,state['frame'],count,args.spacing)
            max_error=max(max_error,float(np.max(np.abs(np.asarray(drones.GetPositionsAttr().Get())-expected[0]))),
                float(np.max(np.abs(np.asarray(cable.GetPointsAttr().Get()).reshape(expected[2].shape)-expected[2]))))
        elapsed=time.perf_counter()-started
        rigid_bodies=sum(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in stage.Traverse())
        assert rigid_bodies==0, 'Presentation must not contain a second simulated drone/cable'
        assert max_error<1e-4, 'USD geometry does not match translated model frames'
        if args.screenshot:
            for _ in range(10):app.update()
            capture_blocking(args.screenshot)
            camera_view(True)
            for _ in range(10):app.update()
            p=Path(args.screenshot)
            capture_blocking(p.with_name(p.stem+'-closeup'+p.suffix))
        audit=dict(num_envs=count,frames=args.smoke_frames,elapsed_s=elapsed,update_fps=args.smoke_frames/elapsed,
            max_usd_position_error_m=max_error,rigid_bodies=rigid_bodies,checkpoint_sha256=meta['checkpoint_sha256'])
        Path(args.replay,f'isaac-smoke-{count}.json').write_text(json.dumps(audit,indent=2))
        print(f'MULTIDRONE_SMOKE_OK count={count} frames={args.smoke_frames}',flush=True)
        return
    print(f'MULTIDRONE_READY count={count}',flush=True)
    def acknowledge():
        if args.run:
            Path(args.run,'live_scene/viewer_status.json').write_text(json.dumps(dict(
                run=str(Path(args.run).resolve()),generation=meta['generation'],
                batch_end_attempt=meta['batch_end_attempt'],mean_episode_reward=meta['mean_episode_reward'],
                success_count=meta['success_count'],num_envs=meta['num_envs'])))
    acknowledge()
    previous=time.perf_counter()
    last_poll=0.
    live_seen=[dict(generation=meta.get('generation'),batch_end_attempt=meta.get('batch_end_attempt'),mean_episode_reward=meta.get('mean_episode_reward'))]
    while app.is_running():
        if args.close_flag and Path(args.close_flag).exists():return
        now=time.perf_counter();elapsed=now-previous;previous=now
        if args.run and now-last_poll>1.:
            last_poll=now
            incoming=read_live_batch(args.run,meta.get('generation'))
            if incoming:
                data,meta,directory=incoming;args.replay=str(directory)
                old_count=count;count=min(args.num_envs,meta['num_envs'])
                times=data['time_s'];end=float(times[-1]);state.update(seconds=0.,frame=-1)
                state['selected']=min(state['selected'],count-1)
                timeline.max=max(end,.001)
                drones.GetProtoIndicesAttr().Set([0]*count);targets.GetProtoIndicesAttr().Set([0]*count)
                cable.GetCurveVertexCountsAttr().Set([data['cable'].shape[2]]*count)
                UsdGeom.Sphere.Get(stage,'/World/Targets/Prototype').GetRadiusAttr().Set(float(meta['target_radius_m']))
                if old_count!=count:camera_view(state['close'])
                print(f'LIVE_BATCH_LOADED {meta["generation"]} attempts={meta["batch_end_attempt"]} reward={meta["mean_episode_reward"]}',flush=True)
                live_seen.append(dict(generation=meta['generation'],batch_end_attempt=meta['batch_end_attempt'],mean_episode_reward=meta['mean_episode_reward']))
                acknowledge()
            try:
                status=json.loads(Path(args.run,'status.json').read_text())
                training_status.text=f'{status.get("status", "Starting")} · {status.get("stage", "")} | displayed batch stays frozen during optimization/validation'
            except (OSError,ValueError):pass
        if state['playing']:state['seconds']=(state['seconds']+min(elapsed,.2)*state['speed'])%(end+.5)
        draw(state['seconds'])
        app.update()
        if args.live_smoke_batches and len(live_seen)>=args.live_smoke_batches:
            if args.screenshot:
                for _ in range(10):app.update()
                capture_blocking(args.screenshot)
            Path(args.run,'live-viewer-verification.json').write_text(json.dumps(dict(batches=live_seen,num_envs=count),indent=2))
            print('LIVE_TRAINING_VIEWER_OK',flush=True);return
        time.sleep(.001)


try:
    main()
finally:
    app.close()
