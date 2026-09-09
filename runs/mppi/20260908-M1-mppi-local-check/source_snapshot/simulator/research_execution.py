"""FullState -> loaded drone -> rotated attachment -> DDER prediction.

One fixed-horizon forward model with optional full temporal gradients. The
production pose equations, command events and cable step are shared. This is
not the virtual force planner, a reward/termination surrogate, or a flight API.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .cable import CableConfiguration, DderModel, DderState
from .cable.residual import FrozenMotionResidual
from .research_physics import ResearchPhysics, research_physics_step
from .research_pose import ResearchPoseModel


class ResearchExecutionModel:
    def __init__(self, drone, cable, physics, offset, dt_s):
        self.drone=drone
        self.cable=cable
        self.physics=physics
        self.offset=tuple(offset)
        self.dt_s=float(dt_s)
        if not np.isfinite(self.dt_s) or self.dt_s<=0:
            raise ValueError('Positive cable timestep required')

    @classmethod
    def from_mapping(cls, model, *, root=None, device='cpu', trainable_residuals=False):
        """Load private model instances; never mutate a saved model or asset.

        trainable_residuals only enables weight derivatives, not optimization.
        With frozen weights, derivatives through both networks' inputs remain.
        Nominal tensor parameters can be supplied by replacing drone.parameters
        and by passing cable_parameters to predict().
        """
        model=deepcopy(model)
        base=Path(root) if root is not None else Path(__file__).resolve().parents[1]
        def resolve(path):
            path=Path(path)
            return path if path.is_absolute() else base/path
        spec=model['fullstate_execution']
        if not spec.get('enabled') or spec.get('schema')!='tracked_pose_execution_v1':
            raise ValueError('A native tracked-pose FullState execution model is required')
        drone=ResearchPoseModel(resolve(spec['checkpoint']),spec['sha256'],device)
        cable=CableConfiguration.from_mapping(model['cable'])
        physics=DderModel(cable.dder_parameters(EI=model['cable']['EI_n_m2'],Cb=model['cable']['Cb_n_m2_s']))
        spec=model.get('motion_residual',{})
        if spec.get('enabled'):
            frozen=FrozenMotionResidual(resolve(spec['checkpoint']),spec['sha256'])
            if 'specification' in spec and spec['specification']!=frozen.payload['specification']:
                raise ValueError('Cable residual specification differs from its checkpoint')
            if frozen.payload['specification']['node_count']!=cable.node_count:
                raise ValueError('Residual node count does not match the cable')
            mode=frozen.payload['specification'].get('mode','acceleration')
            if (mode in ('dissipative','dissipative_plus_acceleration') or spec.get('drag_mode')=='nn_only'
                    or frozen.payload['specification'].get('learn_drag')) and cable.external_drag_s_inv!=0:
                raise ValueError('Learned cable damping requires zero separate fixed drag')
            if trainable_residuals:
                reference=torch.empty(0,device=device,dtype=torch.float64)
                physics.motion_residual=deepcopy(frozen._network(reference)).requires_grad_(True)
            else:
                physics.motion_residual=frozen
        drone.residual.requires_grad_(trainable_residuals)
        return cls(drone,cable,physics,model['recorded_data']['optitrack_to_attachment_offset_body_m'],
            model['simulation']['dt_s'])

    def predict(self, initial_pose, initial_cable, packets, packet_times, output_times, *,
                gradients=False, graph=False, checkpoint_steps=25, hover_command=None,
                cable_parameters=None):
        """Recursive prediction, without measurements after initialization.

        Outputs lie on the saved cable timestep; interpolate for measurement
        losses outside this function. Optional EI/Cb are two scalar tensors.
        Scheduling, gravity, geometry and discrete termination are not fitted
        here. Invalid inputs/rollouts raise instead of supplying masked losses.
        Float64 matches the calibrated native rehearsal solver.
        """
        if gradients and graph:
            raise ValueError('CUDA graph replay is inference-only; set graph=False for gradients')
        if not isinstance(checkpoint_steps,int) or checkpoint_steps<0:
            raise ValueError('checkpoint_steps must be a nonnegative integer')
        if isinstance(output_times,torch.Tensor) and output_times.requires_grad:
            raise ValueError('Output timestamps are fixed scheduling data')
        times=np.asarray(output_times,dtype=float)
        if (times.ndim!=1 or len(times)<2 or not np.isfinite(times).all()
                or not np.allclose(np.diff(times),self.dt_s,atol=1e-10,rtol=0)):
            raise ValueError('Output times must use the saved uniform cable timestep')
        q,v=initial_cable.positions_m,initial_cable.velocities_m_s
        if q.shape!=(len(initial_pose.position),self.cable.node_count,3) or v.shape!=q.shape:
            raise ValueError('Initial cable dimensions differ from the model')
        if q.dtype!=torch.float64 or q.dtype!=initial_pose.position.dtype or v.dtype!=q.dtype:
            raise ValueError('Calibrated execution requires float64 pose and cable states')
        if q.device!=initial_pose.position.device or v.device!=q.device:
            raise ValueError('Pose and cable devices differ')
        if not bool(torch.isfinite(q).all()&torch.isfinite(v).all()):
            raise ValueError('Initial cable state must be finite')
        if initial_cable.endpoint_orientations is not None or initial_cable.endpoint_twist_rad is not None:
            raise ValueError('The freely pivoting execution cable has no clamped material frames')
        with torch.enable_grad() if gradients else torch.no_grad():
            predictor=self.drone.predict_differentiable if gradients else self.drone.predict
            options={} if gradients else {'graph':graph}
            pose=predictor(initial_pose,packets,packet_times,times,self.offset,
                hover_command=hover_command,**options)
            if not bool(pose['valid'].all()):
                raise ValueError('Invalid drone prediction; no complete cable rollout is supplied')
            roots=pose['position_attachment_m']
            if not torch.allclose(q[:,0],roots[:,0],atol=1e-9,rtol=0):
                raise ValueError('Initial cable root does not match the rotated drone attachment')
            constants=self.physics.runtime_constants(q)
            if cable_parameters is not None:
                p=torch.as_tensor(cable_parameters,device=q.device,dtype=q.dtype)
                if p.shape!=(2,) or not bool(torch.isfinite(p).all()) or not bool((p>0).all()):
                    raise ValueError('Cable parameters must be positive finite [EI, Cb]')
                constants=replace(constants,bending_stiffness_n_m2=p[0].expand(len(q)),
                    bending_damping_n_m2_s=p[1].expand(len(q)))
            dt=q.new_full((len(q),),self.dt_s)
            fast=None if gradients else ResearchPhysics(self.physics,initial_cable,self.dt_s,
                constants=constants,graph=graph)

            def advance(q,v,boundaries):
                positions=[];velocities=[]
                for boundary in boundaries.unbind(1):
                    state=DderState(q,v)
                    state=(research_physics_step(self.physics,state,boundary,dt,constants,create_graph=True)
                        if gradients else fast(state,boundary))
                    q,v=state.positions_m,state.velocities_m_s
                    positions.append(q);velocities.append(v)
                return q,v,torch.stack(positions,1),torch.stack(velocities,1)

            positions=[q[:,None]];velocities=[v[:,None]]
            block=checkpoint_steps or (len(times)-1)
            for start in range(1,len(times),block):
                from learning.training_control import check_training_stop
                check_training_stop()
                boundaries=roots[:,start:start+block]
                if gradients and checkpoint_steps:
                    q,v,qp,vp=checkpoint(advance,q,v,boundaries,use_reentrant=False)
                else:
                    q,v,qp,vp=advance(q,v,boundaries)
                if not bool(torch.isfinite(qp).all()&torch.isfinite(vp).all()):
                    raise ValueError('Nonfinite cable rollout; no prediction loss may be used')
                positions.append(qp);velocities.append(vp)
            return dict(pose,cable_positions_m=torch.cat(positions,1),
                cable_velocities_m_s=torch.cat(velocities,1))
